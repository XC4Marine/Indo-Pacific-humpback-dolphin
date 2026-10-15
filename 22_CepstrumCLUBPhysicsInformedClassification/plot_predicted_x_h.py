# -*- coding: utf-8 -*-
"""Render paired fold-0 examples of v5 predicted source and channel responses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_deep_physics_mim as v5


CONDITIONS = ("deep_without_physics", "deep_with_physics")
COLORS = {"deep_without_physics": "#0072B2", "deep_with_physics": "#D55E00"}
TITLES = {"deep_without_physics": "Without Physics", "deep_with_physics": "With Physics"}


def choose_samples(predictions: pd.DataFrame, fold: int) -> pd.DataFrame:
    frame = predictions[predictions["fold"].eq(fold)].copy()
    common = frame.groupby("sample_id")["condition"].nunique()
    frame = frame[frame["sample_id"].isin(common[common.eq(2)].index)]
    selected = []
    for label in (1, 0):
        candidates = frame[frame["label"].eq(label)].drop_duplicates("sample_id")
        selected.append(candidates.nsmallest(2, "sample_index"))
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 4 or result["sample_id"].nunique() != 4:
        raise RuntimeError("Expected exactly two positive and two negative paired examples")
    return result.sort_values(["label", "sample_index"], ascending=[False, True]).reset_index(drop=True)


@torch.no_grad()
def infer_condition(condition: str, sample_indices: np.ndarray, config: v5.Config,
                    waveforms: np.ndarray, device: torch.device, fold: int) -> tuple[np.ndarray, np.ndarray, dict]:
    checkpoint = (config.output_dir / "runs" / condition / f"seed_{config.seed}" /
                  f"fold_{fold}" / "best_checkpoint.pt")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("config_hash") != config.config_hash:
        raise RuntimeError(f"Checkpoint config hash mismatch: {checkpoint}")
    model = v5.DeepPhysicsModel().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    received = torch.from_numpy(np.asarray(waveforms[sample_indices], dtype=np.float32)).to(device)
    output = model(received)
    return (output["x_hat"].cpu().numpy(), output["h_hat"].cpu().numpy(),
            {"checkpoint": str(checkpoint.resolve()), "best_epoch": int(saved["best_epoch"]),
             "validation_auroc": float(saved["validation_auroc"]), "config_hash": saved["config_hash"]})


def source_plot(x_values: dict[str, np.ndarray], selected: pd.DataFrame, output: Path) -> None:
    time_ms = (np.arange(5761) - 2880) / 576_000 * 1_000
    figure, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True)
    for column, condition in enumerate(CONDITIONS):
        axes[0, column].set_title(TITLES[condition], fontsize=13, fontweight="bold")
        for row, sample in selected.iterrows():
            axis = axes[row, column]
            axis.plot(time_ms, x_values[condition][row], color=COLORS[condition], linewidth=0.9)
            axis.axhline(0, color="#A6A6A6", linewidth=0.5)
            axis.set_xlim(-5, 5); axis.set_ylim(-1.05, 1.05)
            if column == 0:
                label_name = "positive click" if int(sample.label) == 1 else "noise"
                axis.set_ylabel(f"{label_name}\nID {sample.sample_id}", fontsize=8)
            if row == 3: axis.set_xlabel("Time relative to waveform centre (ms)")
    figure.suptitle("Predicted source waveform x_hat: paired fold-0 test examples", fontsize=15)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def channel_plot(h_values: dict[str, np.ndarray], selected: pd.DataFrame, output: Path) -> None:
    time_ms = np.arange(2881) / 576_000 * 1_000
    figure, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True)
    for column, condition in enumerate(CONDITIONS):
        axes[0, column].set_title(TITLES[condition], fontsize=13, fontweight="bold")
        for row, sample in selected.iterrows():
            axis = axes[row, column]
            axis.plot(time_ms[1:], h_values[condition][row, 1:], color=COLORS[condition], linewidth=0.9)
            axis.axhline(0, color="#A6A6A6", linewidth=0.5)
            axis.set_xlim(0, 5); axis.set_ylim(-0.26, 0.26)
            inset = inset_axes(axis, width="28%", height="52%", loc="upper right", borderpad=0.7)
            inset.plot(time_ms[:58], h_values[condition][row, :58], color=COLORS[condition], linewidth=0.7)
            inset.set_xlim(0, 0.10); inset.set_ylim(-0.30, 1.05)
            inset.tick_params(labelsize=5, length=1)
            if column == 0:
                label_name = "positive click" if int(sample.label) == 1 else "noise"
                axis.set_ylabel(f"{label_name}\nID {sample.sample_id}", fontsize=8)
            if row == 3: axis.set_xlabel("Channel delay (ms)")
    figure.suptitle("Predicted channel response h_hat tail; inset includes fixed h_hat[0] = 1", fontsize=15)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=v5.DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = v5.Config(output_dir=arguments.output_dir.resolve(), device=arguments.device)
    if arguments.fold not in range(5): raise ValueError("fold must be in 0..4")
    device = v5.base.resolve_device(arguments.device)
    visual_dir = config.output_dir / "visualizations" / "x_h_examples_fold0"
    visual_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(config.output_dir / "predictions" / "sample_predictions.csv")
    selected = choose_samples(predictions, arguments.fold)
    sample_indices = selected["sample_index"].to_numpy(int)
    manifest, waveforms, contract = v5.base.load_locked_cache(config)
    selection = selected[["sample_index", "sample_id", "label", "target_snr_db"]].copy()
    for condition in CONDITIONS:
        probability = predictions[(predictions["fold"].eq(arguments.fold)) & (predictions["condition"].eq(condition))]
        selection = selection.merge(probability[["sample_id", "probability"]].rename(columns={"probability": f"probability_{condition}"}), on="sample_id", how="left", validate="one_to_one")
    x_values, h_values, checkpoints = {}, {}, {}
    for condition in CONDITIONS:
        x_values[condition], h_values[condition], checkpoints[condition] = infer_condition(condition, sample_indices, config, waveforms, device, arguments.fold)
    if not all(np.isfinite(value).all() for value in [*x_values.values(), *h_values.values()]):
        raise RuntimeError("Non-finite prediction found")
    if not all(np.array_equal(value[:, 0], np.ones(len(value))) for value in h_values.values()):
        raise RuntimeError("The fixed channel direct path was not preserved")
    source_plot(x_values, selected, visual_dir / "predicted_x_hat_examples.png")
    channel_plot(h_values, selected, visual_dir / "predicted_h_hat_examples.png")
    selection.to_csv(visual_dir / "selected_samples.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(visual_dir / "predicted_x_h_examples.npz", sample_index=sample_indices,
                        sample_id=selected["sample_id"].to_numpy(), label=selected["label"].to_numpy(),
                        x_hat_without_physics=x_values["deep_without_physics"], h_hat_without_physics=h_values["deep_without_physics"],
                        x_hat_with_physics=x_values["deep_with_physics"], h_hat_with_physics=h_values["deep_with_physics"])
    contract_text = """# Visual contract\n\n- Artifact: paired predicted source and channel waveform examples.\n- Format: diagnostic PNG, 250 dpi.\n- Evidence layer: qualitative diagnostic only; no performance claim.\n- Source: v5 fold-0 best checkpoints and locked experiment-22 waveform cache.\n- Samples: first two positive and first two negative paired test samples by sample_index.\n- Panels: 4 samples x 2 paired conditions; source uses a -5 to 5 ms axis, channel tail uses a 0 to 5 ms axis.\n- Direct path: h_hat[0] is fixed to 1 and shown in each channel inset.\n- Palette: Okabe-Ito blue (#0072B2) without Physics; orange (#D55E00) with Physics.\n- Traceability: selected_samples.csv, predicted_x_h_examples.npz, and checkpoint metadata in render_metadata.json.\n"""
    (visual_dir / "visual_contract.md").write_text(contract_text, encoding="utf-8")
    (visual_dir / "render_metadata.json").write_text(json.dumps({"fold": arguments.fold, "device": str(device), "data_contract": contract, "checkpoints": checkpoints}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"visual_dir": str(visual_dir.resolve()), "selected_samples": len(selection), "finite": True}, ensure_ascii=False))


if __name__ == "__main__": main()
