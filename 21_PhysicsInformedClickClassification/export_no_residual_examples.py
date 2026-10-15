#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出无残差 Physics 模型的低、中、高 SNR 样本分解结果。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


EXPERIMENT_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification")
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_snr_m15_p15_no_residual_5seeds"
DESTINATION = OUTPUT_DIR / "sample_decomposition"
PIPELINE_PATH = EXPERIMENT_DIR / "pipeline.py"
CONDITION = "recon_physics_bce_no_residual"
SEED = 20_260_826
TARGETS = (("-5 dB", -5.0), ("0 dB", 0.0), ("5 dB", 5.0), ("10 dB", 10.0))


def load_pipeline():
    spec = importlib.util.spec_from_file_location("click_pipeline_example_export", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载流水线：{PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def choose_samples(predictions: pd.DataFrame) -> pd.DataFrame:
    candidates = predictions[
        (predictions["condition"] == CONDITION)
        & (predictions["seed"].astype(int) == SEED)
        & (predictions["label"].astype(int) == 1)
        & predictions["target_snr_db"].notna()
    ].copy()
    if len(candidates) != 3_500:
        raise RuntimeError(f"预期3500条正样本预测，实际为{len(candidates)}")

    chosen = []
    available = candidates.copy()
    for stratum, target in TARGETS:
        available["distance_to_target_db"] = (available["target_snr_db"].astype(float) - target).abs()
        row = available.sort_values(
            ["distance_to_target_db", "sample_id"], kind="stable"
        ).iloc[0].copy()
        row["snr_stratum"] = stratum
        row["selection_target_snr_db"] = target
        chosen.append(row)
        available = available[available["sample_id"] != row["sample_id"]].copy()
    return pd.DataFrame(chosen).reset_index(drop=True)


def active_window_ms(signal: np.ndarray, sample_rate: int) -> tuple[float, float]:
    peak_ms = float(np.argmax(np.abs(signal)) / sample_rate * 1_000.0)
    duration_ms = len(signal) / sample_rate * 1_000.0
    return max(0.0, peak_ms - 0.75), min(duration_ms, peak_ms + 1.25)


def main() -> None:
    pipeline = load_pipeline()
    DESTINATION.mkdir(parents=True, exist_ok=True)

    config_values = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8-sig"))
    sample_rate = int(config_values["sample_rate"])
    sample_count = int(config_values["sample_count"])
    response_count = int(config_values["response_count"])
    manifest = pd.read_csv(OUTPUT_DIR / "dataset_manifest.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "predictions" / "sample_predictions.csv")
    waveforms = np.load(OUTPUT_DIR / "cache" / "preprocessed_waveforms.npy", mmap_mode="r")
    selected = choose_samples(predictions)

    models: dict[int, torch.nn.Module] = {}
    value_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="样本分解导出", unit="sample"):
        fold = int(row["test_fold"])
        if fold not in models:
            checkpoint_path = OUTPUT_DIR / "models" / f"{CONDITION}_seed_{SEED}_fold_{fold}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not checkpoint.get("training_complete", False):
                raise RuntimeError(f"检查点未完成：{checkpoint_path}")
            model = pipeline.PhysicalClassifier(sample_count, response_count, include_residual=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            models[fold] = model

        sample_index = int(row["sample_index"])
        manifest_row = manifest.iloc[sample_index]
        if str(manifest_row["sample_id"]) != str(row["sample_id"]):
            raise RuntimeError(f"样本索引与ID不一致：{sample_index}")
        if int(manifest_row["test_fold"]) != fold:
            raise RuntimeError(f"样本折号不一致：{row['sample_id']}")

        original = np.asarray(waveforms[sample_index], dtype=np.float32).copy()
        with torch.no_grad():
            outputs = models[fold](torch.from_numpy(original).unsqueeze(0))
        source = outputs["source"][0].cpu().numpy()
        response = outputs["response"][0].cpu().numpy()
        reconstruction = outputs["reconstruction"][0].cpu().numpy()
        probability = float(torch.sigmoid(outputs["logit"])[0].cpu())
        if abs(probability - float(row["probability"])) > 1e-5:
            raise RuntimeError(f"重新推理概率与已保存预测不一致：{row['sample_id']}")

        sample_id = str(row["sample_id"])
        waveform_time_ms = np.arange(sample_count, dtype=float) / sample_rate * 1_000.0
        response_time_ms = np.arange(response_count, dtype=float) / sample_rate * 1_000.0
        for component, times, values in (
            ("original_y", waveform_time_ms, original),
            ("predicted_x_hat", waveform_time_ms, source),
            ("reconstruction_y_hat", waveform_time_ms, reconstruction),
            ("predicted_h_hat", response_time_ms, response),
        ):
            value_frames.append(pd.DataFrame({
                "snr_stratum": row["snr_stratum"],
                "sample_id": sample_id,
                "seed": SEED,
                "test_fold": fold,
                "target_snr_db": float(row["target_snr_db"]),
                "component": component,
                "sample_index_within_component": np.arange(len(values)),
                "time_ms": times,
                "value": values,
            }))

        error = original - reconstruction
        summary_rows.append({
            "snr_stratum": row["snr_stratum"],
            "selection_target_snr_db": float(row["selection_target_snr_db"]),
            "sample_id": sample_id,
            "ori_file_num": int(row["ori_file_num"]),
            "seed": SEED,
            "test_fold": fold,
            "target_snr_db": float(row["target_snr_db"]),
            "realized_snr_db": float(row["realized_snr_db"]),
            "classification_probability": probability,
            "classification_threshold": float(row["threshold"]),
            "ncc_x_hat_vs_y": normalized_correlation(source, original),
            "ncc_y_hat_vs_y": normalized_correlation(reconstruction, original),
            "reconstruction_rmse": float(np.sqrt(np.mean(error ** 2))),
            "reconstruction_nrmse": float(np.linalg.norm(error) / max(np.linalg.norm(original), 1e-12)),
            "h0": float(response[0]),
            "h_tail_mean_abs": float(np.mean(np.abs(response[1:]))),
        })
        plot_rows.append({
            "stratum": row["snr_stratum"], "sample_id": sample_id,
            "snr": float(row["target_snr_db"]), "probability": probability,
            "time": waveform_time_ms, "original": original, "source": source,
            "reconstruction": reconstruction, "response_time": response_time_ms,
            "response": response,
        })

    values = pd.concat(value_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    values.to_csv(DESTINATION / "sample_decomposition_values.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DESTINATION / "sample_decomposition_summary.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
    })
    colors = {"original": "#222222", "source": "#0072B2", "reconstruction": "#D55E00", "response": "#009E73"}
    figure, axes = plt.subplots(4, 4, figsize=(16, 12.5), constrained_layout=True)
    for row_index, item in enumerate(plot_rows):
        start_ms, end_ms = active_window_ms(item["original"], sample_rate)
        title_prefix = f"{item['stratum']}: {item['sample_id']}\nSNR={item['snr']:.3f} dB, p={item['probability']:.4f}"

        axis = axes[row_index, 0]
        axis.plot(item["time"], item["original"], color=colors["original"], lw=0.8, label="Original input y")
        axis.plot(item["time"], item["source"], color=colors["source"], lw=0.8, alpha=0.9, label="Predicted source x_hat")
        axis.set_title(title_prefix + "\nFull signal and source")
        axis.set_xlim(item["time"][0], item["time"][-1])

        axis = axes[row_index, 1]
        axis.plot(item["time"], item["source"], color=colors["source"], lw=0.9, alpha=0.9, label="Predicted source x_hat")
        axis.set_title(f"Predicted source x_hat (zoom {start_ms:.2f}-{end_ms:.2f} ms)")
        axis.set_xlim(start_ms, end_ms)

        axis = axes[row_index, 2]
        axis.plot(item["time"], item["original"], color=colors["original"], lw=0.9, label="Original input y")
        axis.plot(item["time"], item["reconstruction"], color=colors["reconstruction"], lw=0.9, alpha=0.9, label="Reconstruction y_hat")
        axis.set_title(f"Reconstruction comparison (zoom {start_ms:.2f}-{end_ms:.2f} ms)")
        axis.set_xlim(start_ms, end_ms)

        axis = axes[row_index, 3]
        axis.plot(item["response_time"][1:], item["response"][1:], color=colors["response"], lw=0.8)
        axis.axhline(0.0, color="#A6A6A6", lw=0.6)
        axis.set_title(f"Predicted response tail h_hat (h0={item['response'][0]:.1f})")
        axis.set_xlim(item["response_time"][1], item["response_time"][-1])

        for column in range(4):
            axes[row_index, column].grid(alpha=0.18, lw=0.5)
            axes[row_index, column].set_xlabel("Time (ms)")
            axes[row_index, column].set_ylabel("Amplitude")
        for column in (0, 1, 2):
            axes[row_index, column].legend(loc="upper right", frameon=False)

    figure.suptitle("No-residual Recon + Physics + BCE: four deterministic test-sample decompositions", fontsize=13)
    figure.savefig(DESTINATION / "sample_decomposition_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(DESTINATION / "sample_decomposition_comparison.svg", bbox_inches="tight")
    plt.close(figure)

    manifest_out = {
        "condition": CONDITION,
        "seed": SEED,
        "selection_rule": "Positive test samples closest to target SNR values -5.0, 0.0, 5.0, and 10.0 dB; no performance-based selection.",
        "sample_count": len(summary),
        "values_csv": str((DESTINATION / "sample_decomposition_values.csv").resolve()),
        "summary_csv": str((DESTINATION / "sample_decomposition_summary.csv").resolve()),
        "png": str((DESTINATION / "sample_decomposition_comparison.png").resolve()),
        "svg": str((DESTINATION / "sample_decomposition_comparison.svg").resolve()),
    }
    (DESTINATION / "export_manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(manifest_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
