#!/usr/bin/env python3
"""Derive 4 ms, extra-10 dB multipath clicks from complete Bellhop arrivals.

This entry point deliberately does not invoke Bellhop.  It keeps the complete
unmodified arrivals from ``python_results`` for provenance, re-gates them at
four milliseconds after the global first arrival, and synthesizes the retained
arrivals after applying a controlled 10 dB amplitude loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from run_experiment_v2 import FREQUENCIES_HZ, subband_synthesis


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "python_results"
DEFAULT_OUTPUT = ROOT / "python_results_delay4ms_extra10dB"


@dataclass(frozen=True)
class DerivationConfig:
    source_root: str
    output_root: str
    arrival_window_s: float = 0.004
    relative_amplitude_threshold: float = 10 ** (-80 / 20)
    additional_attenuation_db: float = 10.0
    high_frequency_extra_at_190khz_db: float = 0.0
    max_arrivals_per_frequency: int | None = None
    minimum_direct_over_reflection_db: float | None = None

    def total_attenuation_db(self, frequency_hz: float) -> float:
        progress = np.clip((frequency_hz - 20_000.0) / (190_000.0 - 20_000.0), 0.0, 1.0)
        return float(self.additional_attenuation_db + progress * self.high_frequency_extra_at_190khz_db)

    def amplitude_scale(self, frequency_hz: float) -> float:
        return float(10 ** (-self.total_attenuation_db(frequency_hz) / 20))


def config_hash(config: DerivationConfig, source_hash: str) -> str:
    payload = {"derivation": asdict(config), "source_config_hash": source_hash}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    temporary = path.with_name(path.stem + ".tmp.wav")
    sf.write(temporary, data, sample_rate, subtype="FLOAT")
    os.replace(temporary, path)


def load_source(source_root: Path) -> tuple[pd.DataFrame, str]:
    manifest_file = source_root / "click_manifest.csv"
    config_file = source_root / "config.json"
    if not manifest_file.is_file() or not config_file.is_file():
        raise FileNotFoundError("源结果目录必须含 click_manifest.csv 和 config.json。")
    manifest = pd.read_csv(manifest_file)
    successful = manifest[manifest["status"] == "success"].copy().reset_index(drop=True)
    if len(successful) != 110 or successful["click_id"].nunique() != 110:
        raise ValueError(f"源结果必须恰有 110 条成功 click，当前为 {len(successful)}。")
    source_hash = str(json.loads(config_file.read_text(encoding="utf-8"))["hash"])
    required = ["channel.npz", "rx_raw.npy"]
    for row in successful.itertuples():
        source_dir = Path(row.output_path)
        if any(not (source_dir / name).is_file() for name in required):
            raise FileNotFoundError(f"源 click 缺少必要文件：{source_dir}")
    return successful, source_hash


def new_manifest(source: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    manifest = source.copy()
    manifest["output_path"] = [str(output_root / "clicks" / click_id) for click_id in manifest["click_id"]]
    manifest["status"] = "pending"
    manifest["successful_frequency_count"] = 0
    manifest["error_message"] = pd.Series([""] * len(manifest), dtype="object")
    manifest["runtime_s"] = np.nan
    manifest["first_arrival_s"] = np.nan
    manifest["arrivals_4ms"] = np.nan
    manifest["arrivals_4ms_candidates"] = np.nan
    manifest["selected_arrivals"] = np.nan
    manifest["raw_rms_gain"] = np.nan
    manifest["preview_scale"] = np.nan
    manifest = manifest.drop(columns=["arrivals_10ms"], errors="ignore")
    return manifest


def gate_and_scale(all_arrivals: np.ndarray, first_arrival_s: float, config: DerivationConfig) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    frame = pd.DataFrame.from_records(all_arrivals)
    candidate_tables: list[pd.DataFrame] = []
    selected_tables: list[pd.DataFrame] = []
    scaled_tables: list[pd.DataFrame] = []
    statuses: list[dict] = []
    for frequency_hz in FREQUENCIES_HZ:
        table = frame[np.isclose(frame["frequency_hz"], frequency_hz)].copy()
        if table.empty:
            raise RuntimeError(f"缺失 {frequency_hz / 1000:g} kHz 原始到达。")
        threshold = float(table["effective_gain"].max()) * config.relative_amplitude_threshold
        candidates = table[(table["delay_s"] >= first_arrival_s) &
                           (table["delay_s"] <= first_arrival_s + config.arrival_window_s) &
                           (table["effective_gain"] >= threshold)].copy()
        if candidates.empty:
            raise RuntimeError(f"{frequency_hz / 1000:g} kHz 在 4 ms 门限内无有效到达。")
        ranked = candidates.sort_values(["effective_gain", "delay_s"], ascending=[False, True], kind="stable")
        selected = ranked
        if config.max_arrivals_per_frequency is not None:
            selected = ranked.head(config.max_arrivals_per_frequency).copy()
        direct = candidates[(candidates["top_bounces"] == 0) & (candidates["bottom_bounces"] == 0)]
        if config.minimum_direct_over_reflection_db is not None:
            if direct.empty:
                raise RuntimeError(f"{frequency_hz / 1000:g} kHz 缺少无反弹直达径，无法施加直达径优势约束。")
            direct = direct.sort_values(["effective_gain", "delay_s"], ascending=[False, True], kind="stable").head(1)
            if not selected.index.isin(direct.index).any():
                selected = pd.concat([direct, ranked[~ranked.index.isin(direct.index)].head(len(selected) - 1)])
                selected = selected.sort_values(["effective_gain", "delay_s"], ascending=[False, True], kind="stable")
        scale = config.amplitude_scale(float(frequency_hz))
        scaled = selected.copy()
        reflection_scale = np.ones(len(selected), dtype=float)
        reflection_penalty_db = np.zeros(len(selected), dtype=float)
        if config.minimum_direct_over_reflection_db is not None:
            direct_gain = float(direct["effective_gain"].iloc[0])
            required_ratio = 10 ** (config.minimum_direct_over_reflection_db / 20)
            reflected = (selected["top_bounces"] != 0) | (selected["bottom_bounces"] != 0)
            reflection_scale[reflected.to_numpy()] = np.minimum(
                1.0, direct_gain / (selected.loc[reflected, "effective_gain"].to_numpy() * required_ratio))
            reflection_penalty_db[reflected.to_numpy()] = -20 * np.log10(reflection_scale[reflected.to_numpy()])
        total_scale = scale * reflection_scale
        scaled["complex_gain"] *= total_scale
        scaled["effective_gain"] *= total_scale
        scaled["base_amplitude_scale"] = scale
        scaled["reflection_amplitude_scale"] = reflection_scale
        scaled["reflection_extra_attenuation_db"] = reflection_penalty_db
        scaled["total_amplitude_scale"] = total_scale
        candidate_tables.append(candidates)
        selected_tables.append(selected)
        scaled_tables.append(scaled)
        statuses.append({"frequency_hz": frequency_hz, "status": "success", "source_arrival_count": len(table),
                         "arrivals_4ms_candidates": len(candidates), "selected_arrivals": len(selected),
                         "relative_threshold": config.relative_amplitude_threshold,
                         "additional_attenuation_db": config.additional_attenuation_db,
                         "high_frequency_extra_at_190khz_db": config.high_frequency_extra_at_190khz_db,
                         "total_attenuation_db": config.total_attenuation_db(float(frequency_hz)),
                         "amplitude_scale": scale,
                         "minimum_direct_over_reflection_db": config.minimum_direct_over_reflection_db,
                         "max_reflection_extra_attenuation_db": float(reflection_penalty_db.max())})
    return candidate_tables, selected_tables, scaled_tables, statuses


def derive_click(row: pd.Series, source_row: pd.Series, config: DerivationConfig, source_hash: str) -> dict:
    started = time.monotonic()
    output_dir = Path(row.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_row.output_path)
    source_waveform, sample_rate = sf.read(row.input_wav, dtype="float64", always_2d=False)
    with np.load(source_dir / "channel.npz", allow_pickle=True) as source_channel:
        all_arrivals = source_channel["all_arrivals"].copy()
        first_arrival_s = float(source_channel["first_arrival_s"])
        candidate_tables, selected_tables, scaled_tables, statuses = gate_and_scale(all_arrivals, first_arrival_s, config)
        max_excess_samples = int(np.ceil(max(float(table["delay_s"].max()) - first_arrival_s for table in selected_tables) * sample_rate))
        output_samples = len(source_waveform) + max_excess_samples
        received = subband_synthesis(source_waveform, int(sample_rate), scaled_tables, first_arrival_s, output_samples)
        raw_gated = pd.concat(candidate_tables, ignore_index=True).to_records(index=False)
        selected_gated = pd.concat(selected_tables, ignore_index=True).to_records(index=False)
        synthesis_gated = pd.concat(scaled_tables, ignore_index=True).to_records(index=False)
        atomic_npz(output_dir / "channel.npz", frequencies_hz=source_channel["frequencies_hz"],
                   first_arrival_s=first_arrival_s, all_arrivals=all_arrivals, gated_arrivals=raw_gated,
                   selected_arrivals=selected_gated, synthesis_arrivals=synthesis_gated, ssp_depth_m=source_channel["ssp_depth_m"],
                   ssp_speed_mps=source_channel["ssp_speed_mps"], source_depth_m=source_channel["source_depth_m"],
                   receiver_depth_m=source_channel["receiver_depth_m"], range_m=source_channel["range_m"],
                   random_seed=source_channel["random_seed"], source_result_root=str(Path(config.source_root).resolve()),
                   source_result_config_hash=source_hash, arrival_window_s=config.arrival_window_s,
                   relative_amplitude_threshold=config.relative_amplitude_threshold,
                   additional_attenuation_db=config.additional_attenuation_db,
                   high_frequency_extra_at_190khz_db=config.high_frequency_extra_at_190khz_db,
                   max_arrivals_per_frequency=-1 if config.max_arrivals_per_frequency is None else config.max_arrivals_per_frequency,
                   minimum_direct_over_reflection_db=-1.0 if config.minimum_direct_over_reflection_db is None else config.minimum_direct_over_reflection_db,
                   config_hash=config_hash(config, source_hash))
    atomic_csv(pd.DataFrame(statuses), output_dir / "frequency_status.csv")
    preview_scale = 1.0 / max(float(np.max(np.abs(received))), np.finfo(float).eps)
    np.save(output_dir / "rx_raw.npy", received)
    atomic_wav(output_dir / "rx_preview.wav", received * preview_scale, int(sample_rate))
    return {"status": "success", "successful_frequency_count": len(FREQUENCIES_HZ),
            "runtime_s": time.monotonic() - started, "first_arrival_s": first_arrival_s,
            "arrivals_4ms": len(raw_gated), "arrivals_4ms_candidates": len(raw_gated),
            "selected_arrivals": len(selected_gated),
            "raw_rms_gain": float(np.sqrt(np.mean(received ** 2)) / max(np.sqrt(np.mean(source_waveform ** 2)), np.finfo(float).eps)),
            "preview_scale": preview_scale, "error_message": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="重建目标目录中已完成的 click。")
    parser.add_argument("--limit", type=int, default=None, help="仅派生前 N 条，用于验收。")
    parser.add_argument("--max-arrivals-per-frequency", type=int, default=None, help="每个频带最多保留的最强到达数。")
    parser.add_argument("--hf-extra-at-190khz-db", type=float, default=0.0, help="20 kHz 为 0、190 kHz 为该值的附加高频幅度衰减。")
    parser.add_argument("--minimum-direct-over-reflection-db", type=float, default=None, help="直达径相对每条反射径保证的最小幅度优势。")
    args = parser.parse_args()
    if args.max_arrivals_per_frequency is not None and args.max_arrivals_per_frequency < 1:
        raise ValueError("每频带最大到达数必须为正整数。")
    if args.hf_extra_at_190khz_db < 0:
        raise ValueError("高频附加衰减必须为非负 dB 值。")
    if args.minimum_direct_over_reflection_db is not None and args.minimum_direct_over_reflection_db < 0:
        raise ValueError("直达径优势必须为非负 dB 值。")
    config = DerivationConfig(str(args.source_root.resolve()), str(args.output_root.resolve()),
                              high_frequency_extra_at_190khz_db=args.hf_extra_at_190khz_db,
                              max_arrivals_per_frequency=args.max_arrivals_per_frequency,
                              minimum_direct_over_reflection_db=args.minimum_direct_over_reflection_db)
    source, source_hash = load_source(args.source_root)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "clicks").mkdir(exist_ok=True)
    digest = config_hash(config, source_hash)
    config_file = output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")).get("hash") != digest:
        raise RuntimeError("目标目录配置哈希不一致，拒绝续跑。")
    config_file.write_text(json.dumps({"hash": digest, "derivation": asdict(config), "source_config_hash": source_hash}, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = new_manifest(source, output)
    manifest_file = output / "click_manifest.csv"
    if args.resume and manifest_file.is_file() and not args.force:
        old = pd.read_csv(manifest_file)
        if list(old.get("click_id", [])) == list(manifest["click_id"]):
            for column in ["status", "successful_frequency_count", "error_message", "runtime_s", "first_arrival_s", "arrivals_4ms", "arrivals_4ms_candidates", "selected_arrivals", "raw_rms_gain", "preview_scale"]:
                if column in old:
                    manifest[column] = old[column]
    active = manifest if args.limit is None else manifest.head(args.limit)
    overall = time.monotonic()
    for completed_count, (index, row) in enumerate(active.iterrows(), start=1):
        click_dir = Path(row.output_path)
        completed = row.status == "success" and (click_dir / "rx_raw.npy").is_file() and (click_dir / "channel.npz").is_file()
        if not (completed and args.resume and not args.force):
            source_row = source[source["click_id"] == row.click_id].iloc[0]
            try:
                result = derive_click(row, source_row, config, source_hash)
            except Exception as exc:
                result = {"status": "failed", "successful_frequency_count": 0, "runtime_s": np.nan,
                          "error_message": repr(exc)}
                click_dir.mkdir(parents=True, exist_ok=True)
                (click_dir / "failure.log").write_text(repr(exc), encoding="utf-8")
            for key, value in result.items():
                manifest.loc[index, key] = value
        elapsed = time.monotonic() - overall
        remaining = len(active) - completed_count
        eta = elapsed / completed_count * remaining
        print(f"[{completed_count / len(active):6.2%}] click {completed_count:3d}/{len(active)} | success={sum(manifest.status == 'success')} failed={sum(manifest.status == 'failed')} | elapsed={elapsed:.1f} s ETA={eta:.1f} s", flush=True)
        atomic_csv(manifest, manifest_file)
    atomic_csv(manifest, output / "result_summary.csv")
    print(f"Done. Derived signals: {output / 'clicks'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
