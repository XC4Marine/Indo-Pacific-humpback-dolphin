#!/usr/bin/env python3
"""Apply a locked v7 model to truth-isolated candidate-feature artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from core_v7 import EPS, ROOT, atomic_json


def assert_truth_isolation() -> None:
    forbidden = ("event_" + "channels.npz", "original_" + "click.wav", "event_" + "manifest.csv", "original_" + "channel.npz")
    text = Path(__file__).read_text(encoding="utf-8")
    leaked = [name for name in forbidden if name in text]
    if leaked: raise RuntimeError(f"truth-isolation violation: {leaked}")


def matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(names)), dtype=float)
    return np.asarray([[np.nan if row.get(name) is None else float(row.get(name, 0)) for name in names] for row in rows], dtype=float)


def scaled(rows: list[dict], model: dict) -> np.ndarray:
    value = matrix(rows, model["features"]); score_model = model["score_model"]
    lo, hi = np.asarray(score_model["lo"]), np.asarray(score_model["hi"])
    return np.clip(np.where(np.isfinite((value-lo)/(hi-lo)), (value-lo)/(hi-lo), .5), 0, 1)


def raw_score(rows: list[dict], model: dict) -> np.ndarray:
    method, score_model = model["method"], model["score_model"]
    if method == "v6_joint": return np.zeros(len(rows))
    z = scaled(rows, model); pos = {name: i for i, name in enumerate(model["features"])}
    if method == "coverage": return z[:, pos["event_coverage"]]
    if method == "delay_guard": return z[:, pos["resolvability"]]
    if method == "source_competition": return np.sqrt(z[:,pos["delta_bic_path_base"]] * z[:,pos["delta_bic_source_path"]])
    if method == "source_marginal": return np.sqrt(z[:,pos["source_marginal_stability"]] * z[:,pos["channel_allocation_ratio"]])
    if method == "product_physical":
        names = ("event_coverage", "source_marginal_stability", "channel_allocation_ratio", "delta_bic_path_base", "delta_bic_source_path", "resolvability", "frequency_coherence")
        return np.exp(np.mean(np.log(np.clip(z[:, [pos[name] for name in names]], EPS, 1)), axis=1))
    if method == "monotonic_physical":
        monotonic = score_model["monotonic"]; zz = (z - np.asarray(monotonic["mean"])) / np.asarray(monotonic["std"])
        return 1 / (1 + np.exp(-np.clip(monotonic["intercept"] + zz @ np.asarray(monotonic["coefficients"]), -30, 30)))
    raise ValueError(f"unsupported deployment method: {method}")


def calibrated(raw: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration["kind"] == "platt": return 1 / (1 + np.exp(-np.clip(calibration["intercept"] + calibration["coefficient"] * raw, -30, 30)))
    return np.interp(raw, np.asarray(calibration["x"]), np.asarray(calibration["y"]))


def rejection_reason(row: dict, confidence: float, model: dict) -> str | None:
    thresholds = model.get("reason_thresholds", {})
    if row.get("source_ambiguous", False): return "source_ambiguous"
    if float(row.get("event_coverage", 0)) < .6: return "insufficient_event_support"
    if float(row.get("resolvability", 0)) < float(thresholds.get("resolvability", -np.inf)): return "poorly_resolved"
    if float(row.get("source_marginal_stability", 0)) < float(thresholds.get("source_marginal_stability", -np.inf)): return "unstable_across_source_refits"
    if confidence < float(model["threshold"]): return "low_physical_confidence"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, default=ROOT / "evaluation" / "physical_confidence_model.json")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "outputs"); parser.add_argument("--output-root", type=Path, default=ROOT / "pseudolabels")
    args = parser.parse_args(); assert_truth_isolation()
    if not args.model.exists(): raise FileNotFoundError("locked deployment model is absent; no v7 method passed the gate or evaluation has not run")
    model = json.loads(args.model.read_text(encoding="utf-8")); args.output_root.mkdir(parents=True, exist_ok=True)
    for artifact in sorted(args.feature_root.glob("*/candidate_features.json")):
        payload = json.loads(artifact.read_text(encoding="utf-8")); rows = payload["candidates"]
        raw = raw_score(rows, model); probability = calibrated(raw, model["calibrator"]); labels = []
        for row, confidence in zip(rows, probability):
            reason = rejection_reason(row, float(confidence), model)
            labels.append({"path_id": row["path_id"], "delay_median_ms": row["delay_median_ms"], "relative_amplitude_median": row["relative_amplitude_median"], "event_coverage": row["event_coverage"], "components": {name: row.get(name) for name in model["features"]}, "physical_confidence": float(confidence), "accepted": reason is None, "rejection_reason": reason})
        atomic_json(args.output_root / payload["train_id"] / "pseudolabels.json", {"protocol": model["protocol"], "train_id": payload["train_id"], "model_method": model["method"], "threshold": model["threshold"], "pseudolabels": labels})


if __name__ == "__main__": main()
