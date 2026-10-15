# -*- coding: utf-8 -*-
"""Render five deterministic random v7 no-MIM test examples from saved checkpoints."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as base
spec = importlib.util.spec_from_file_location("v7_plot", HERE / "experiment_cross_convolution_consistency.py")
v7 = importlib.util.module_from_spec(spec); assert spec.loader; sys.modules[spec.name] = v7; spec.loader.exec_module(v7)

ROOT = HERE / "outputs_v7_cross_convolution_no_mim_seed20260826_2conditions"
OUT = ROOT / "figures"
SEED = 20260905

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_csv(ROOT / "predictions" / "sample_predictions.csv")
    pool = prediction[prediction.condition.eq("no_mim_reference")].copy()
    chosen = pool.iloc[np.random.default_rng(SEED).choice(len(pool), 5, replace=False)].sort_values("fold")
    config = v7.Config(output_dir=ROOT)
    manifest, waves, _ = base.load_locked_cache(config)
    device = base.resolve_device("cuda")
    cache: dict[int, v7.HeterogeneousXHModel] = {}
    rows, data = [], []
    for row in chosen.itertuples(index=False):
        fold = int(row.fold)
        if fold not in cache:
            model = v7.HeterogeneousXHModel(config).to(device)
            state = torch.load(ROOT / "runs" / "no_mim_reference" / "seed_20260826" / f"fold_{fold}" / "best_checkpoint.pt", map_location=device, weights_only=True)
            model.load_state_dict(state["model"]); model.eval(); cache[fold] = model
        received = torch.from_numpy(waves[int(row.sample_index)]).unsqueeze(0).to(device)
        with torch.no_grad(): output = cache[fold](received)
        r, x, h, recon = (value.squeeze(0).cpu().numpy() for value in (received, output["x_hat"], output["h_hat"], output["r_hat"]))
        rel_l2 = float(np.linalg.norm(r - recon) / max(np.linalg.norm(r), 1e-12))
        rows.append({"sample_id": row.sample_id, "sample_index": int(row.sample_index), "fold": fold, "label": int(row.label), "probability": float(row.probability), "relative_l2_reconstruction_error": rel_l2})
        data.append((r, x, h, recon, rows[-1]))
    fig, axes = plt.subplots(5, 4, figsize=(18, 14), constrained_layout=True)
    palette = {"received": "#222222", "x": "#0072B2", "h": "#009E73", "recon": "#D55E00", "error": "#CC79A7"}
    time_ms = np.arange(5761) / config.sample_rate * 1000
    delay_ms = np.arange(2881) / config.sample_rate * 1000
    for i, (r, x, h, recon, meta) in enumerate(data):
        title = f"Sample {i + 1}: fold {meta['fold']}, y={meta['label']}, p={meta['probability']:.3f}, rel-L2={meta['relative_l2_reconstruction_error']:.3f}"
        axes[i, 0].plot(time_ms, r, color=palette["received"], lw=.75); axes[i, 0].set_title(title, fontsize=9, loc="left"); axes[i, 0].set_ylabel("Amplitude")
        axes[i, 1].plot(time_ms, x, color=palette["x"], lw=.75); axes[i, 1].set_title(r"Predicted $\hat{x}$", fontsize=10)
        axes[i, 2].plot(delay_ms, h, color=palette["h"], lw=.75); axes[i, 2].set_title(r"Predicted $\hat{h}$", fontsize=10)
        axes[i, 3].plot(time_ms, r, color=palette["received"], lw=.65, label=r"$r$"); axes[i, 3].plot(time_ms, recon, color=palette["recon"], lw=.65, alpha=.9, label=r"$\hat r$"); axes[i, 3].plot(time_ms, r - recon, color=palette["error"], lw=.5, alpha=.7, label=r"$r-\hat r$"); axes[i, 3].set_title(r"Reconstruction and residual", fontsize=10)
        for j, ax in enumerate(axes[i]):
            ax.axhline(0, color="#A6A6A6", lw=.5); ax.tick_params(labelsize=8); ax.set_xlabel("Time (ms)" if j != 2 else "Delay (ms)")
        axes[i, 3].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0, 0].set_title("Received waveform r", fontsize=10)
    fig.suptitle("Five random held-out examples: no-MIM reference reconstruction", fontsize=15, fontweight="bold")
    png = OUT / "v7_no_mim_five_random_x_h_reconstructions.png"
    fig.savefig(png, dpi=220, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "v7_no_mim_five_random_x_h_reconstructions.csv", index=False, encoding="utf-8-sig")
    print(png)

if __name__ == "__main__": main()
