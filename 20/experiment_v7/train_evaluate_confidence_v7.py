#!/usr/bin/env python3
"""Nested grouped truth evaluation for v7 physical-confidence candidates.

This is intentionally the only v7 entry point that opens simulation truth.
"""
from __future__ import annotations

import argparse, csv, json, math, os, re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from core_v7 import (DATA_ROOT, EPS, EVENT_COUNT, PROTOCOL, ROOT, STABLE_EVENT_COUNT,
                     TOLERANCE_MS, atomic_csv, atomic_json, v6_confidence)

SEED = 20260910
FEATURES = ("event_coverage", "source_marginal_stability", "channel_allocation_ratio",
            "delta_bic_path_base", "delta_bic_source_path", "resolvability",
            "frequency_coherence", "phase_consistency", "rss_improvement")
METHODS = ("v6_joint", "coverage", "delay_guard", "source_competition", "source_marginal", "product_physical", "monotonic_physical", "product_no_competition", "product_no_source_uncertainty", "product_no_resolvability", "product_no_frequency")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None: atomic_csv(path, rows)


def truth_paths(arrivals: np.ndarray, first_arrival_s: float) -> list[dict[str, Any]]:
    result = []
    for top, bottom in sorted({(int(row["top_bounces"]), int(row["bottom_bounces"])) for row in arrivals}):
        group = arrivals[(arrivals["top_bounces"] == top) & (arrivals["bottom_bounces"] == bottom)]
        delay = float(np.median(group["delay_s"]) - first_arrival_s) * 1e3
        name = "direct" if top == 0 and bottom == 0 else f"{top}_{bottom}"
        result.append({"name": name, "delay_ms": delay})
    direct = next(row for row in result if row["name"] == "direct")
    for row in result: row["relative_delay_ms"] = row["delay_ms"] - direct["delay_ms"]
    return result


def load_stable_truth(train_id: str) -> list[dict[str, Any]]:
    # Truth access is isolated to this evaluator.
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as payload:
        arrivals, first = np.asarray(payload["event_synthesis_arrivals"]), float(payload["first_arrival_s"])
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in range(EVENT_COUNT):
        for path in truth_paths(arrivals[event], first): by_name.setdefault(path["name"], []).append(path)
    return [{"name": name, "delay_median_ms": float(np.median([r["relative_delay_ms"] for r in rows]))}
            for name, rows in by_name.items() if name != "direct" and len(rows) >= STABLE_EVENT_COUNT]


def group_of(train_id: str) -> str:
    match = re.search(r"train_(\d+)", train_id)
    if not match: raise ValueError(f"cannot parse train group: {train_id}")
    return match.group(1)


def load_rows(feature_root: Path) -> list[dict[str, Any]]:
    rows = []
    for file in sorted(feature_root.glob("*/candidate_features.json")):
        payload = json.loads(file.read_text(encoding="utf-8")); train_id = payload["train_id"]
        targets = load_stable_truth(train_id)
        for item in payload["candidates"]:
            row = dict(item); row["group"] = group_of(train_id)
            row["correct"] = int(any(abs(row["delay_median_ms"] - target["delay_median_ms"]) <= TOLERANCE_MS for target in targets))
            row["v6_joint"] = v6_confidence(train_id, row["delay_median_ms"])
            rows.append(row)
    if not rows: raise RuntimeError(f"no candidate features under {feature_root}")
    return rows


def extracted_group_count(feature_root: Path) -> int:
    """Validate the source dataset coverage independently of candidate existence."""
    train_ids = []
    for file in sorted(feature_root.glob("*/candidate_features.json")):
        train_ids.append(json.loads(file.read_text(encoding="utf-8"))["train_id"])
    return len({group_of(train_id) for train_id in train_ids})


def values(rows: list[dict[str, Any]], names: tuple[str, ...] = FEATURES) -> np.ndarray:
    data = []
    for row in rows:
        item = []
        for name in names:
            value = row.get(name)
            item.append(np.nan if value is None else float(value))
        data.append(item)
    return np.asarray(data, dtype=float)


def normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.nanquantile(x, .05, axis=0), np.nanquantile(x, .95, axis=0)
    lo = np.where(np.isfinite(lo), lo, 0); hi = np.where(np.isfinite(hi) & (hi > lo + EPS), hi, lo + 1)
    return lo, hi


def scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = (x - lo) / (hi - lo)
    result = np.where(np.isfinite(result), result, .5)
    return np.clip(result, 0, 1)


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean, std = np.mean(x, axis=0), np.std(x, axis=0) + EPS
    z = (x - mean) / std
    def loss(theta: np.ndarray) -> float:
        linear = theta[0] + z @ theta[1:]; p = 1 / (1 + np.exp(-np.clip(linear, -30, 30)))
        return float(-np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)) + 1e-3 * np.sum(theta[1:] ** 2))
    result = minimize(loss, np.zeros(z.shape[1] + 1), method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * z.shape[1])
    if not result.success: raise RuntimeError(f"monotonic logistic optimisation failed: {result.message}")
    return {"mean": mean.tolist(), "std": std.tolist(), "intercept": float(result.x[0]), "coefficients": result.x[1:].tolist()}


def _logistic_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
    return 1 / (1 + np.exp(-np.clip(float(model["intercept"]) + z @ np.asarray(model["coefficients"]), -30, 30)))


def fit_score(method: str, train: list[dict[str, Any]]) -> dict[str, Any]:
    x = values(train); lo, hi = normalizer(x); z = scale(x, lo, hi)
    positions = {name: index for index, name in enumerate(FEATURES)}
    if method == "monotonic_physical":
        return {"method": method, "lo": lo.tolist(), "hi": hi.tolist(), "monotonic": _logistic_fit(z, np.asarray([r["correct"] for r in train]))}
    return {"method": method, "lo": lo.tolist(), "hi": hi.tolist(), "positions": positions}


def score(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    method = model["method"]
    if method == "v6_joint": return np.asarray([np.nan_to_num(r["v6_joint"], nan=0.0) for r in rows])
    z = scale(values(rows), np.asarray(model["lo"]), np.asarray(model["hi"])); pos = model.get("positions", {name: i for i, name in enumerate(FEATURES)})
    if method == "coverage": return z[:, pos["event_coverage"]]
    if method == "delay_guard": return np.sqrt(z[:, pos["resolvability"]] * z[:, pos["relative_source_support"]] if "relative_source_support" in pos else z[:, pos["resolvability"]])
    if method == "source_competition": return np.sqrt(z[:, pos["delta_bic_path_base"]] * z[:, pos["delta_bic_source_path"]])
    if method == "source_marginal": return np.sqrt(z[:, pos["source_marginal_stability"]] * z[:, pos["channel_allocation_ratio"]])
    if method.startswith("product_"):
        chosen = ("event_coverage", "source_marginal_stability", "channel_allocation_ratio", "delta_bic_path_base", "delta_bic_source_path", "resolvability", "frequency_coherence")
        removed = {"product_no_competition": {"delta_bic_path_base", "delta_bic_source_path"}, "product_no_source_uncertainty": {"source_marginal_stability", "channel_allocation_ratio"}, "product_no_resolvability": {"resolvability"}, "product_no_frequency": {"frequency_coherence"}}.get(method, set())
        chosen = tuple(item for item in chosen if item not in removed)
        return np.exp(np.mean(np.log(np.clip(z[:, [pos[item] for item in chosen]], EPS, 1)), axis=1))
    if method == "monotonic_physical": return _logistic_predict(model["monotonic"], z)
    raise ValueError(method)


def fit_calibrator(raw: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any]:
    if kind == "platt":
        estimator = LogisticRegression(C=1e6, solver="lbfgs").fit(raw.reshape(-1, 1), y)
        return {"kind": kind, "intercept": float(estimator.intercept_[0]), "coefficient": float(estimator.coef_[0, 0])}
    estimator = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return {"kind": kind, "x": estimator.X_thresholds_.tolist(), "y": estimator.y_thresholds_.tolist()}


def calibrate(model: dict[str, Any], raw: np.ndarray) -> np.ndarray:
    if model["kind"] == "platt": return 1 / (1 + np.exp(-np.clip(model["intercept"] + model["coefficient"] * raw, -30, 30)))
    return np.interp(raw, np.asarray(model["x"]), np.asarray(model["y"]))


def group_bootstrap_lower(rows: list[dict[str, Any]], accepted: np.ndarray, repetitions: int = 2000) -> float:
    groups = sorted({row["group"] for row in rows}); by_group = {group: np.asarray([i for i, row in enumerate(rows) if row["group"] == group]) for group in groups}
    rng = np.random.default_rng(SEED); scores = []
    for _ in range(repetitions):
        selected = np.concatenate([by_group[groups[i]] for i in rng.integers(0, len(groups), len(groups))])
        mask = accepted[selected]
        scores.append(float(np.mean([rows[i]["correct"] for i in selected[mask]])) if np.any(mask) else 0.0)
    return float(np.quantile(scores, .025))


def metrics(rows: list[dict[str, Any]], probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray([row["correct"] for row in rows]); probability = np.clip(probability, EPS, 1 - EPS); accepted = probability >= threshold
    precision = float(np.mean(y[accepted])) if np.any(accepted) else 0.0
    recall = float(np.sum(y[accepted]) / max(1, np.sum(y)))
    bins = np.linspace(0, 1, 11); ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if np.any(mask): ece += np.mean(mask) * abs(float(np.mean(y[mask])) - float(np.mean(probability[mask])))
    logits = np.log(probability / (1 - probability))
    slope = float(LogisticRegression(C=1e6, solver="lbfgs").fit(logits.reshape(-1, 1), y).coef_[0, 0]) if len(np.unique(y)) > 1 else float("nan")
    return {"reflection_brier": float(brier_score_loss(y, probability)), "nll": float(log_loss(y, probability, labels=[0, 1])), "ece": float(ece),
            "calibration_slope": slope, "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else float("nan"), "threshold": float(threshold),
            "precision_at_threshold": precision, "precision_lower_95": group_bootstrap_lower(rows, accepted), "recall": recall,
            "accepted_path_count": int(np.sum(accepted)), "covered_train_groups": int(len({rows[i]["group"] for i in np.flatnonzero(accepted)}))}


def rejection_mask(method: str, rows: list[dict[str, Any]], thresholds: dict[str, float]) -> np.ndarray:
    """Mechanism-specific abstention, shared by OOF selection and deployment."""
    accepted = np.ones(len(rows), dtype=bool)
    uses_competition = method in {"source_competition", "product_physical", "monotonic_physical", "product_no_source_uncertainty", "product_no_resolvability", "product_no_frequency"}
    uses_resolvability = method in {"delay_guard", "product_physical", "monotonic_physical", "product_no_competition", "product_no_source_uncertainty", "product_no_frequency"}
    uses_source_uncertainty = method in {"source_marginal", "product_physical", "monotonic_physical", "product_no_competition", "product_no_resolvability", "product_no_frequency"}
    for index, row in enumerate(rows):
        if uses_competition and bool(row.get("source_ambiguous", False)): accepted[index] = False
        if uses_resolvability and float(row.get("resolvability", 0)) < float(thresholds["resolvability"]): accepted[index] = False
        if uses_source_uncertainty and float(row.get("source_marginal_stability", 0)) < float(thresholds["source_marginal_stability"]): accepted[index] = False
    return accepted


def choose_threshold(method: str, rows: list[dict[str, Any]], probability: np.ndarray, thresholds: dict[str, float]) -> float:
    candidates = np.unique(np.r_[0.0, np.quantile(probability, np.linspace(.05, .99, 60)), 1.0])
    y = np.asarray([row["correct"] for row in rows])
    eligible = []
    for threshold in candidates:
        accepted = (probability >= threshold) & rejection_mask(method, rows, thresholds)
        precision = float(np.mean(y[accepted])) if np.any(accepted) else 0.0
        recall = float(np.sum(y[accepted]) / max(1, np.sum(y)))
        if precision >= .95 and np.any(accepted): eligible.append((recall, -threshold, threshold))
    return float(max(eligible)[2]) if eligible else 1.0


def learn_reason_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Select human-readable component rejection cutoffs on training data only."""
    result = {}
    y = np.asarray([row["correct"] for row in rows])
    for name in ("resolvability", "source_marginal_stability", "delta_bic_source_path"):
        x = np.asarray([float(row.get(name) or 0) for row in rows]); choices = np.unique(np.quantile(x, np.linspace(.05, .95, 25)))
        valid = [cut for cut in choices if np.any(x >= cut) and np.mean(y[x >= cut]) >= .95]
        result[name] = float(min(valid)) if valid else float(np.quantile(x, .95))
    return result


def inner_fit(method: str, train: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, float]]:
    y = np.asarray([row["correct"] for row in train]); groups = np.asarray([row["group"] for row in train])
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    raw = np.zeros(len(train)); indices = np.arange(len(train))
    for fit_index, valid_index in splitter.split(indices, y, groups):
        fitted = fit_score(method, [train[i] for i in fit_index]); raw[valid_index] = score(fitted, [train[i] for i in valid_index])
    choices = []
    for kind in ("platt", "isotonic"):
        calibrated = calibrate(fit_calibrator(raw, y, kind), raw); choices.append((brier_score_loss(y, calibrated), kind, calibrated))
    _, kind, inner_probability = min(choices, key=lambda item: item[0])
    final = fit_score(method, train); calibration = fit_calibrator(raw, y, kind); thresholds = learn_reason_thresholds(train)
    return final, calibration, choose_threshold(method, train, inner_probability, thresholds), thresholds


def nested_oof(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y, groups = np.asarray([row["correct"] for row in rows]), np.asarray([row["group"] for row in rows]); indices = np.arange(len(rows))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED); output, settings = [], []
    for fold, (fit_index, test_index) in enumerate(splitter.split(indices, y, groups), 1):
        train, test = [rows[i] for i in fit_index], [rows[i] for i in test_index]
        for method in METHODS:
            fitted, calibration, threshold, reason_thresholds = inner_fit(method, train)
            probability = calibrate(calibration, score(fitted, test))
            settings.append({"fold": fold, "method": method, "threshold": threshold, "calibrator": calibration["kind"]})
            accepted = (probability >= threshold) & rejection_mask(method, test, reason_thresholds)
            for local, item in enumerate(test): output.append({"fold": fold, "method": method, **item, "physical_confidence": float(probability[local]), "threshold": threshold, "accepted": int(accepted[local])})
    return output, settings


def plot_summary(predictions: list[dict[str, Any]], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for method in METHODS:
        rows = [row for row in predictions if row["method"] == method]; p = np.asarray([r["physical_confidence"] for r in rows]); y = np.asarray([r["correct"] for r in rows])
        order = np.argsort(p)[::-1]; coverage = np.arange(1, len(rows) + 1) / len(rows); risk = 1 - np.cumsum(y[order]) / np.arange(1, len(rows) + 1)
        axes[0].plot(coverage, risk, label=method)
        bins = np.linspace(0, 1, 11); xs=[]; ys=[]
        for low, high in zip(bins[:-1], bins[1:]):
            mask=(p>=low)&(p<(high if high<1 else p<=high))
            if np.any(mask): xs.append(np.mean(p[mask])); ys.append(np.mean(y[mask]))
        axes[1].plot(xs, ys, marker="o", label=method)
    axes[0].set(xlabel="coverage", ylabel="risk", title="Risk–coverage"); axes[1].plot([0,1],[0,1],"k--",lw=1); axes[1].set(xlabel="confidence", ylabel="observed precision", title="Reliability")
    for axis in axes: axis.legend(fontsize=7); axis.grid(alpha=.2)
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=170); plt.close(fig)


def plot_p2_case(predictions: list[dict[str, Any]], destination: Path) -> None:
    """Render the pre-specified P2 false-path case without threshold adjustment."""
    train_id = "click_099_train_718_pulse_016"
    rows = [row for row in predictions if row["method"] == "monotonic_physical" and row["train_id"] == train_id]
    fig, axis = plt.subplots(figsize=(8, 3.8))
    if rows:
        colours = ["#0072B2" if row["correct"] else "#D55E00" for row in rows]
        markers = ["o" if row["accepted"] else "x" for row in rows]
        for row, colour, marker in zip(rows, colours, markers):
            axis.scatter(row["delay_median_ms"], row["physical_confidence"], color=colour, marker=marker, s=70)
            axis.annotate(f"P{row['path_id']}\n{row['physical_confidence']:.2f}", (row["delay_median_ms"], row["physical_confidence"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        axis.axhline(float(np.median([row["threshold"] for row in rows])), color="#444444", ls="--", lw=1, label="fold threshold")
        axis.scatter([], [], color="#0072B2", label="matched true reflection")
        axis.scatter([], [], color="#D55E00", label="unmatched candidate")
        axis.scatter([], [], color="#444444", marker="x", label="rejected")
        axis.legend(fontsize=8)
    else:
        axis.text(.5, .5, "No reflection candidate was extracted for the pre-specified P2 case.", ha="center", va="center", transform=axis.transAxes)
    axis.set(title="P2 pre-specified case: click_099_train_718_pulse_016", xlabel="relative delay (ms)", ylabel="physical confidence")
    axis.grid(alpha=.2); fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=170); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--feature-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "evaluation"); args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.feature_root); groups = {row["group"] for row in rows}; all_group_count = extracted_group_count(args.feature_root)
    if all_group_count != 69: raise RuntimeError(f"expected 69 source train groups, found {all_group_count}")
    if len(groups) < 2: raise RuntimeError("fewer than two candidate-bearing groups; grouped evaluation is impossible")
    predictions, settings = nested_oof(rows); write_csv(args.output_root / "oof_path_predictions.csv", predictions); write_csv(args.output_root / "fold_settings.csv", settings)
    summary = []
    for method in METHODS:
        method_rows = [row for row in predictions if row["method"] == method]
        # Fold-specific thresholds remain attached to their OOF predictions.
        item = metrics(method_rows, np.asarray([row["physical_confidence"] for row in method_rows]), 2.0)
        accepted = np.asarray([bool(row["accepted"]) for row in method_rows]); y = np.asarray([row["correct"] for row in method_rows])
        item.update({"method": method, "threshold": float(np.median([row["threshold"] for row in method_rows])), "precision_at_threshold": float(np.mean(y[accepted])) if np.any(accepted) else 0.0,
                     "precision_lower_95": group_bootstrap_lower(method_rows, accepted), "recall": float(np.sum(y[accepted])/max(1,np.sum(y))),
                     "accepted_path_count": int(np.sum(accepted)), "covered_train_groups": len({r["group"] for r in method_rows if r["accepted"]})})
        summary.append(item)
    coverage = next(row for row in summary if row["method"] == "coverage"); v6 = next(row for row in summary if row["method"] == "v6_joint")
    winners = [row for row in summary if row["method"] in {"delay_guard", "source_competition", "source_marginal", "product_physical", "monotonic_physical"} and row["precision_at_threshold"] >= .95 and row["precision_lower_95"] >= .90 and row["accepted_path_count"] >= 30 and row["covered_train_groups"] >= 20 and row["reflection_brier"] < coverage["reflection_brier"] and row["reflection_brier"] < v6["reflection_brier"] and row["nll"] < coverage["nll"] and row["nll"] < v6["nll"]]
    winner = min(winners, key=lambda row: row["reflection_brier"]) if winners else None
    atomic_csv(args.output_root / "comparison.csv", [row for row in summary if not row["method"].startswith("product_no_")]); atomic_csv(args.output_root / "ablation.csv", [row for row in summary if row["method"].startswith("product_")]); plot_summary(predictions, args.output_root / "summary_v7.png"); plot_p2_case(predictions, args.output_root / "p2_case.png")
    amplitude_edges = np.quantile([row["relative_amplitude_median"] for row in rows], [.25, .5, .75])
    strata = []
    for method in METHODS:
        method_rows = [row for row in predictions if row["method"] == method]
        labels = {"source_support": lambda r: "inside" if r["relative_source_support"] <= 1 else "outside",
                  "amplitude_quartile": lambda r: f"Q{1 + sum(r['relative_amplitude_median'] > edge for edge in amplitude_edges)}",
                  "coverage_bin": lambda r: f"{math.floor(r['event_coverage'] * 10) / 10:.1f}-{min(1.0, math.floor(r['event_coverage'] * 10) / 10 + .1):.1f}",
                  "delay_region": lambda r: "near" if r["delay_median_ms"] < .2 else "middle" if r["delay_median_ms"] < 1 else "far"}
        for name, labeller in labels.items():
            for label in sorted({labeller(r) for r in method_rows}):
                subset = [r for r in method_rows if labeller(r) == label]; accepted = [r for r in subset if r["accepted"]]
                strata.append({"method": method, "stratum": name, "level": label, "candidate_count": len(subset), "accepted_count": len(accepted), "precision": float(np.mean([r["correct"] for r in accepted])) if accepted else 0.0})
    atomic_csv(args.output_root / "stratified_results.csv", strata)
    report = ["# 实验 v7 评估报告", "", f"候选数：{len(rows)}；原始数据 train group：{all_group_count}；含反射候选的 OOF train group：{len(groups)}。", "", "## 门槛结论", ""]
    if winner is None: report.append("当前数据不足以支持可靠伪标签：没有方法同时满足预注册精度、置信下界、覆盖数量以及相对两项基线的 Brier/NLL 改善门槛。")
    else: report.append(f"通过全部门槛的方法：**{winner['method']}**。可训练全数据部署模型。")
    report.extend(["", "P2 预注册案例 `click_099_train_718_pulse_016` 仅用于诊断展示，未参与阈值人工调整。", "", "| 方法 | Brier | NLL | Precision | 下界 | 接受数 | 组覆盖 |", "|---|---:|---:|---:|---:|---:|---:|"])
    report.extend([f"| {row['method']} | {row['reflection_brier']:.4f} | {row['nll']:.4f} | {row['precision_at_threshold']:.3f} | {row['precision_lower_95']:.3f} | {row['accepted_path_count']} | {row['covered_train_groups']} |" for row in summary])
    (args.output_root / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    atomic_json(args.output_root / "selection.json", {"protocol": PROTOCOL, "source_train_group_count": all_group_count, "candidate_train_group_count": len(groups), "winner": winner["method"] if winner else None, "criteria_passed": winner is not None, "comparisons": summary})
    if winner:
        method = winner["method"]; fitted, calibration, threshold, reason_thresholds = inner_fit(method, rows)
        atomic_json(args.output_root / "physical_confidence_model.json", {"protocol": PROTOCOL, "method": method, "features": FEATURES, "score_model": fitted, "calibrator": calibration, "threshold": threshold, "reason_thresholds": reason_thresholds, "training_groups": sorted(groups), "scope": "calibrated correctness probability under current simulation distribution and stated identifiability assumptions"})
    else:
        stale_model = args.output_root / "physical_confidence_model.json"
        if stale_model.exists(): stale_model.unlink()


if __name__ == "__main__": main()
