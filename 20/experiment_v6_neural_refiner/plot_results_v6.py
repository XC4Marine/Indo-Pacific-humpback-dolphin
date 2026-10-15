#!/usr/bin/env python3
"""Render traceable hold-out result figures for Experiment 20 v6.

This script runs after blind inference and truth-only evaluation.  It does not
participate in training or inference; the representative case panel uses truth
solely to make the already-evaluated response shape inspectable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "experiment_v5"
V4 = HERE.parent / "experiment_v4"
sys.path[:0] = [str(V5), str(V4)]
from protocol_v5 import DATA_ROOT, discover_inputs, group_split  # noqa: E402
from evaluate_bcsc_v4 import transfer_from_arrivals  # noqa: E402
from experiment_v6 import CONDITIONS, FREQ, OUT  # noqa: E402

FIGURES = OUT / "figures"
METHODS = ("v4_independent", "neural_ensemble")
LABEL = {"v4_independent": "BCSC (v4 independent)", "neural_ensemble": "Neural residual refiner (ensemble)"}
COLOR = {"v4_independent": "#0072B2", "neural_ensemble": "#D55E00", "truth": "#222222"}
CONDITION = {"clean": "Clean", "snr_m05": "−5 dB AWGN", "snr_p05": "5 dB AWGN"}


def configure() -> None:
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "figure.dpi": 160, "savefig.dpi": 300})


def load_clicktrain_medians(metrics: pd.DataFrame) -> pd.DataFrame:
    return (metrics.groupby(["condition", "train_id", "method"], as_index=False)["channel_amplitude_rmse_db"]
            .median().rename(columns={"channel_amplitude_rmse_db": "rmse_db"}))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_quantitative(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    med = load_clicktrain_medians(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7), gridspec_kw={"width_ratios": [1.02, 1.18, 1.0]})

    # (a) paired clean hold-out samples, sorted only to expose the paired pattern.
    ax = axes[0]
    clean = med[med.condition == "clean"].pivot(index="train_id", columns="method", values="rmse_db").sort_values("v4_independent")
    y = np.arange(len(clean))
    for i, (_, row) in enumerate(clean.iterrows()):
        ax.plot([row.v4_independent, row.neural_ensemble], [i, i], color="#A6A6A6", lw=1.1, zorder=1)
    ax.scatter(clean.v4_independent, y, color=COLOR["v4_independent"], marker="o", s=32, zorder=2, label=LABEL["v4_independent"])
    ax.scatter(clean.neural_ensemble, y, color=COLOR["neural_ensemble"], marker="D", s=35, zorder=3, label=LABEL["neural_ensemble"])
    ax.set(yticks=[], xlabel="Channel amplitude RMSE (dB)", title="(a) Clean: each held-out clicktrain")
    ax.grid(axis="x", color="#E6E6E6", lw=.8); ax.legend(loc="lower right", fontsize=8, frameon=False)

    # (b) distribution across all conditions; dots retain every hold-out clicktrain.
    ax = axes[1]
    positions, values, colors, labels = [], [], [], []
    rng = np.random.default_rng(20260908)
    for c_idx, condition in enumerate(CONDITIONS):
        for m_idx, method in enumerate(METHODS):
            positions.append(c_idx * 3 + m_idx); values.append(med[(med.condition == condition) & (med.method == method)].rmse_db.to_numpy())
            colors.append(COLOR[method]); labels.append(method)
    boxes = ax.boxplot(values, positions=positions, widths=.6, patch_artist=True, showfliers=False,
                       medianprops={"color": "#222222", "lw": 1.5}, whiskerprops={"color": "#666666"}, capprops={"color": "#666666"})
    for box, color in zip(boxes["boxes"], colors): box.set(facecolor=color, alpha=.24, edgecolor=color, lw=1.4)
    for position, item, color in zip(positions, values, colors):
        ax.scatter(position + rng.uniform(-.12, .12, len(item)), item, s=19, color=color, alpha=.72, edgecolors="white", linewidths=.35, zorder=3)
    ax.set(xticks=[.5, 3.5, 6.5], xticklabels=[CONDITION[c] for c in CONDITIONS], ylabel="Channel amplitude RMSE (dB)", title="(b) All held-out groups (n = 16 / condition)")
    ax.grid(axis="y", color="#E6E6E6", lw=.8); ax.set_axisbelow(True)
    ax.text(.5, ax.get_ylim()[1] * .96, "● BCSC    ◆ Neural", ha="center", va="top", fontsize=9, color="#333333")

    # (c) group-level paired bootstrap result.  The zero line is the neutral point.
    ax = axes[2]
    ordered = paired.set_index("condition").loc[list(CONDITIONS)]
    y = np.arange(len(ordered))
    central = ordered.median_delta_rmse_db.to_numpy(); low = ordered.ci_low.to_numpy(); high = ordered.ci_high.to_numpy()
    ax.axvline(0, color="#666666", ls="--", lw=1)
    ax.errorbar(central, y, xerr=np.vstack((central - low, high - central)), fmt="D", color=COLOR["neural_ensemble"], ecolor=COLOR["neural_ensemble"], elinewidth=2, capsize=4, ms=7)
    for x, yy in zip(central, y):
        offset = -.16 if yy == y.max() else .16
        ax.text(x + .12, yy + offset, f"{x:.2f} dB", ha="left", va="top" if offset < 0 else "bottom", fontsize=9, color=COLOR["neural_ensemble"])
    ax.set(yticks=y, yticklabels=[CONDITION[c] for c in CONDITIONS], xlabel="Paired change (neural − BCSC, dB)", title="(c) Group-level paired median [95% CI]")
    ax.grid(axis="x", color="#E6E6E6", lw=.8); ax.set_axisbelow(True)
    fig.suptitle("Experiment 20 v6 — Channel amplitude recovery on unseen source groups", fontsize=14, fontweight="bold", y=1.03)
    fig.text(.5, -.02, "Lower RMSE is better.  Each test source group is held out from neural training; all three intervals lie below zero.", ha="center", fontsize=9, color="#444444")
    save(fig, "v6_rmse_comparison")


def truth_shape(train_id: str) -> np.ndarray:
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as item:
        arrivals, first = item["event_synthesis_arrivals"], float(item["first_arrival_s"])
    shapes = []
    for event in arrivals:
        response = transfer_from_arrivals(event, first); amplitude = 20 * np.log10(np.abs(response) + 1e-15)
        shapes.append(amplitude - np.median(amplitude))
    return np.asarray(shapes, dtype=float)


def choose_representative(metrics: pd.DataFrame) -> tuple[str, int]:
    clean = metrics[(metrics.condition == "clean") & (metrics.method == "neural_ensemble")].copy()
    target = clean.channel_amplitude_rmse_db.median()
    nearest = clean.assign(distance=(clean.channel_amplitude_rmse_db - target).abs()).sort_values(["distance", "train_id", "event_index"]).iloc[0]
    return str(nearest.train_id), int(nearest.event_index) - 1


def plot_response_example(metrics: pd.DataFrame) -> None:
    train_id, event = choose_representative(metrics)
    prediction = np.load(OUT / "blind_predictions" / "clean" / f"{train_id}.npz")
    target = truth_shape(train_id)[event]; baseline = prediction["baseline_shape"][event]; neural = prediction["corrected_shape"][event]
    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    freq = FREQ / 1000
    ax.plot(freq, target, color=COLOR["truth"], lw=2.6, label="Truth", zorder=4)
    ax.plot(freq, baseline, color=COLOR["v4_independent"], marker="o", ms=3.3, lw=1.6, label="BCSC (v4 independent)")
    ax.plot(freq, neural, color=COLOR["neural_ensemble"], marker="D", ms=3.2, lw=1.8, label="Neural residual refiner (ensemble)")
    ax.axhline(0, color="#BDBDBD", lw=.8, ls=":")
    rmse = lambda x: np.sqrt(np.mean((x - target) ** 2))
    ax.text(.985, .04, f"BCSC RMSE = {rmse(baseline):.2f} dB\nNeural RMSE = {rmse(neural):.2f} dB", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#BBBBBB", "alpha": .95})
    ax.set(xlabel="Frequency (kHz)", ylabel="Relative amplitude (dB)", title="Representative clean event at the median neural error level")
    ax.legend(loc="upper left", frameon=False); ax.grid(color="#E6E6E6", lw=.8); ax.set_axisbelow(True)
    fig.text(.5, -.01, f"Held-out source group: {train_id}; event {event + 1}.  The panel is selected deterministically by proximity to the median neural event error.", ha="center", fontsize=8.5, color="#444444")
    save(fig, "v6_representative_response")


def main() -> None:
    configure()
    metrics = pd.read_csv(OUT / "event_metrics.csv"); paired = pd.read_csv(OUT / "paired_deltas.csv")
    plot_quantitative(metrics, paired); plot_response_example(metrics)
    print(f"Generated figures: {FIGURES}")


if __name__ == "__main__":
    main()
