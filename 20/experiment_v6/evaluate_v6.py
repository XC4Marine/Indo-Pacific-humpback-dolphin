#!/usr/bin/env python3
"""Truth-only evaluation and figures for Experiment v6 consensus outputs.

Do not import this module from the extraction entry point.  This is the only v6
module that opens the simulation event-arrival asset.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
OUTPUT_ROOT = ROOT / "outputs_independent_consensus"
EVALUATION_ROOT = ROOT / "evaluation_independent_consensus"
TOLERANCE_MS = .03
EVENT_COUNT = 10
STABLE_EVENT_COUNT = 6
EPSILON = 1e-18


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def truth_paths(arrivals: np.ndarray, first_arrival_s: float) -> list[dict[str, Any]]:
    """Reduce frequency-sampled arrivals to one physical path per bounce pair."""
    result = []
    for top, bottom in sorted({(int(row["top_bounces"]), int(row["bottom_bounces"])) for row in arrivals}):
        group = arrivals[(arrivals["top_bounces"] == top) & (arrivals["bottom_bounces"] == bottom)]
        delay = float(np.median(group["delay_s"]) - first_arrival_s) * 1e3
        gain = float(np.median(group["effective_gain"]))
        name = "direct" if top == 0 and bottom == 0 else f"{top}_{bottom}"
        result.append({"name": name, "delay_ms": delay, "gain": gain})
    direct = next(row for row in result if row["name"] == "direct")
    for row in result:
        row["relative_delay_ms"] = float(row["delay_ms"] - direct["delay_ms"])
        row["relative_amplitude"] = float(row["gain"] / (direct["gain"] + EPSILON))
    return result


def load_truth(train_id: str) -> list[list[dict[str, Any]]]:
    # This isolated evaluator is intentionally the sole v6 reader of truth data.
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as payload:
        arrivals = np.asarray(payload["event_synthesis_arrivals"])
        first_arrival = float(payload["first_arrival_s"])
    return [truth_paths(arrivals[event], first_arrival) for event in range(EVENT_COUNT)]


def stable_truth_paths(per_event: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for event, paths in enumerate(per_event):
        for path in paths:
            grouped.setdefault(str(path["name"]), []).append((event, path))
    stable = []
    for name, rows in grouped.items():
        if len(rows) >= STABLE_EVENT_COUNT or name == "direct":
            stable.append({"name": name, "event_count": len(rows),
                           "delay_median_ms": float(np.median([row["relative_delay_ms"] for _, row in rows])),
                           "relative_amplitude_median": float(np.median([row["relative_amplitude"] for _, row in rows])),
                           "by_event": {event: row for event, row in rows}})
    return sorted(stable, key=lambda row: row["delay_median_ms"])


def one_to_one_matches(estimated: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[tuple[int, int]]:
    delay = lambda row: float(row.get("delay_median_ms", row.get("relative_delay_ms")))
    pairs = [(abs(delay(e) - delay(t)), e_i, t_i)
             for e_i, e in enumerate(estimated) for t_i, t in enumerate(truth)
             if abs(delay(e) - delay(t)) <= TOLERANCE_MS]
    selected: list[tuple[int, int]] = []; used_e: set[int] = set(); used_t: set[int] = set()
    for _, e_i, t_i in sorted(pairs):
        if e_i not in used_e and t_i not in used_t:
            selected.append((e_i, t_i)); used_e.add(e_i); used_t.add(t_i)
    return selected


def structure_metrics(estimated: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, float]:
    matches = one_to_one_matches(estimated, truth)
    precision = len(matches) / len(estimated) if estimated else 0.0
    recall = len(matches) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delay = [abs(float(estimated[e]["delay_median_ms"]) - float(truth[t]["delay_median_ms"])) for e, t in matches]
    amplitude = [abs(20 * math.log10(float(estimated[e]["relative_amplitude_median"]) + EPSILON) - 20 * math.log10(float(truth[t]["relative_amplitude_median"]) + EPSILON)) for e, t in matches]
    return {"stable_structure_precision": precision, "stable_structure_recall": recall, "stable_structure_f1": f1,
            "stable_structure_delay_mae_ms": float(np.mean(delay)) if delay else float("nan"),
            "stable_structure_amplitude_mae_db": float(np.mean(amplitude)) if amplitude else float("nan"),
            "estimated_stable_path_count": len(estimated), "true_stable_path_count": len(truth)}


def event_metrics(estimated: list[dict[str, Any]], truth: list[dict[str, Any]], event: int) -> dict[str, float]:
    candidates = []
    for path in estimated:
        candidates.append({"delay_median_ms": float(path["event_delay_ms"][event]), "relative_amplitude_median": float(path["event_relative_amplitude"][event])})
    matches = one_to_one_matches(candidates, truth)
    precision = len(matches) / len(candidates) if candidates else 0.0
    recall = len(matches) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [abs(candidates[e]["delay_median_ms"] - truth[t]["relative_delay_ms"]) for e, t in matches]
    amplitude = [abs(20 * math.log10(candidates[e]["relative_amplitude_median"] + EPSILON) - 20 * math.log10(float(truth[t]["relative_amplitude"]) + EPSILON)) for e, t in matches]
    return {"path_precision": precision, "path_recall": recall, "path_f1": f1,
            "path_delay_mae_ms": float(np.mean(delays)) if delays else float("nan"),
            "path_relative_amplitude_mae_db": float(np.mean(amplitude)) if amplitude else float("nan")}


def calibration_rows(train_id: str, estimated: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in estimated:
        correct = any(abs(float(path["delay_median_ms"]) - float(target["delay_median_ms"])) <= TOLERANCE_MS for target in truth)
        rows.append({"train_id": train_id, "path_id": path["path_id"], "confidence": float(path["joint_confidence"]), "correct": int(correct)})
    return rows


def plot_train(train_id: str, estimated: list[dict[str, Any]], truth: list[dict[str, Any]], per_event_truth: list[list[dict[str, Any]]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.2, 4.0, 1.4))
    ax_structure = fig.add_subplot(grid[0, :])
    floor = -45
    tx = [row["delay_median_ms"] for row in truth]; ta = [20 * np.log10(row["relative_amplitude_median"] + EPSILON) for row in truth]
    ex = [row["delay_median_ms"] for row in estimated]; ea = [20 * np.log10(row["relative_amplitude_median"] + EPSILON) for row in estimated]
    ax_structure.vlines(tx, 0.55, .90, color="#222222", lw=1.5); ax_structure.scatter(tx, [.90] * len(tx), c="#222222", label="真值稳定路径")
    ax_structure.vlines(ex, .05, .40, color="#0072B2", lw=1.5); ax_structure.scatter(ex, [.40] * len(ex), c="#0072B2", marker="x", s=55, label="估计公共结构")
    for row in estimated:
        ax_structure.annotate(f"P{row['path_id']} c={row['joint_confidence']:.2f}\n[{row['delay_q025_ms']:.3f},{row['delay_q975_ms']:.3f}] ms", (row["delay_median_ms"], .34), fontsize=7, rotation=25)
    ax_structure.set_ylim(0, 1.05); ax_structure.set_yticks([.4, .9], ["估计", "真值"]); ax_structure.set_xlabel("相对时延 (ms)"); ax_structure.legend(loc="upper right", ncol=2); ax_structure.set_title(f"{train_id}：公共稳定结构")
    matrix_axes = [fig.add_subplot(grid[1, col]) for col in range(2)]
    for event in range(EVENT_COUNT):
        ax = matrix_axes[0]
        for target in per_event_truth[event]:
            ax.plot([target["relative_delay_ms"], target["relative_delay_ms"]], [event + .05, event + .38], color="#333333", lw=1)
        ax = matrix_axes[1]
        for path in estimated:
            delay = path["event_delay_ms"][event]
            ax.plot([delay, delay], [event + .05, event + .38], color="#0072B2", lw=1.2)
    for ax, title in zip(matrix_axes, ("十事件真值路径", "十事件公共结构应用结果")):
        ax.set_title(title); ax.set_xlabel("相对时延 (ms)"); ax.set_ylabel("事件编号"); ax.set_yticks(np.arange(EVENT_COUNT) + .2, [str(x + 1) for x in range(EVENT_COUNT)]); ax.set_ylim(10, 0)
    ax_table = fig.add_subplot(grid[2, :]); ax_table.axis("off")
    cell = [[f"P{row['path_id']}", f"{row['delay_median_ms']:.4f}", f"{row['joint_confidence']:.2f}", f"{row['wilson_lower_95']:.2f}", f"{row['event_coverage']:.2f}", f"{row['relative_amplitude_median']:.3f} [{row['relative_amplitude_q25']:.3f}, {row['relative_amplitude_q75']:.3f}]"] for row in estimated]
    ax_table.table(cellText=cell, colLabels=["路径", "时延 ms", "联合置信度", "Wilson 下界", "事件覆盖率", "相对幅度中位数 [IQR]"], loc="center", cellLoc="center")
    fig.savefig(destination, dpi=180); plt.close(fig)


def summary_figure(train_rows: list[dict[str, Any]], calibration: list[dict[str, Any]], destination: Path,
                   *, confidence_xlabel: str = "预测联合置信度", calibration_title: str = "置信度校准", calibration_label: str = "v6") -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    axes[0].hist([row["stable_structure_f1"] for row in train_rows], bins=12, color="#0072B2"); axes[0].set(title="稳定结构 F1", xlabel="F1")
    axes[1].hist([row["stable_structure_delay_mae_ms"] for row in train_rows if np.isfinite(row["stable_structure_delay_mae_ms"])], bins=12, color="#009E73"); axes[1].set(title="稳定结构时延 MAE", xlabel="ms")
    axes[2].hist([row["reconstruction_nrmse"] for row in train_rows], bins=12, color="#E69F00"); axes[2].set(title="重构 NRMSE", xlabel="NRMSE")
    axes[3].hist([row["estimated_stable_path_count"] for row in train_rows], bins=np.arange(0.5, 7.5), color="#CC79A7", alpha=.6, label="估计路径数"); axes[3].set(title="路径数分布", xlabel="路径数")
    confidence = np.asarray([row["confidence"] for row in calibration]); correct = np.asarray([row["correct"] for row in calibration])
    bins = np.linspace(0, 1, 6); centres = []; predicted = []; observed = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if np.any(mask): centres.append((low + high) / 2); predicted.append(float(np.mean(confidence[mask]))); observed.append(float(np.mean(correct[mask])))
    ax = axes[4]
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="理想校准"); ax.plot(predicted, observed, "o-", color="#0072B2", label=calibration_label); ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=confidence_xlabel, ylabel="实际正确率", title=calibration_title); ax.legend()
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=220); plt.close(fig)


def report(train_rows: list[dict[str, Any]], calibration: list[dict[str, Any]], output: Path) -> None:
    values = lambda key: np.asarray([row[key] for row in train_rows], dtype=float)
    brier = float(np.mean([(row["confidence"] - row["correct"]) ** 2 for row in calibration])) if calibration else float("nan")
    best = max(train_rows, key=lambda row: row["stable_structure_f1"]); hard = min(train_rows, key=lambda row: row["stable_structure_f1"])
    bootstrap_count = int(train_rows[0]["bootstrap_count"])
    text = f"""# 实验 v6：independent BCSC 的 clicktrain 稳定多径结构提取

## 范围

本报告覆盖 {len(train_rows)} 个 clicktrain。提取阶段仅复用 v4 independent BCSC 的盲提取源、十事件信道、重构和检测窗口，并在固定该源条件下进行残差块 bootstrap；真值仅在本评估阶段读取。因此，联合置信度是**固定已提取源的条件置信度**，不代表完整 BCSC 参数（尤其是源波形估计）的不确定性。

## 方法与固定参数

十事件候选按 0.03 ms 聚类，跨事件覆盖至少 6/10 的候选进入稳定结构池；直达径始终保留。每个样本进行 {bootstrap_count} 次固定种子的同步十事件条件残差块 bootstrap。BIC 在直达径与置信度最高的最多 5 条反射候选上枚举支持集；每个事件独立最小二乘重估幅度，复杂度项为 10K log N。

## 全量汇总

- 稳定结构 F1：中位数 {np.nanmedian(values('stable_structure_f1')):.3f}
- 稳定结构时延 MAE：中位数 {np.nanmedian(values('stable_structure_delay_mae_ms')):.5f} ms
- 逐事件路径 F1：中位数 {np.nanmedian(values('event_path_f1_median')):.3f}
- 重构 NRMSE：中位数 {np.nanmedian(values('reconstruction_nrmse')):.4f}
- 置信度 Brier score：{brier:.4f}

最佳样本：`{best['train_id']}`（稳定结构 F1={best['stable_structure_f1']:.3f}）。

困难样本：`{hard['train_id']}`（稳定结构 F1={hard['stable_structure_f1']:.3f}）。

## 局限性

该流程不重采样或重估盲源，故不能用于完整 BCSC 参数不确定性推断。稳定性还依赖 v4 第一阶段候选能覆盖真实路径；没有被第一阶段发现的路径无法由 bootstrap 新建为 v6 候选。BIC 的独立事件幅度重估降低了稀疏收缩偏差，但仍受固定时延和窗口对齐误差影响。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    artifacts = sorted(args.input_root.glob("*/consensus.json"))
    if args.limit is not None: artifacts = artifacts[:args.limit]
    if not artifacts: raise FileNotFoundError("no consensus artifacts available")
    train_rows: list[dict[str, Any]] = []; event_rows: list[dict[str, Any]] = []; calibration: list[dict[str, Any]] = []
    for index, artifact in enumerate(artifacts, start=1):
        result = json.loads(artifact.read_text(encoding="utf-8")); train_id = result["train_id"]; estimated = result["final_paths"]
        per_event_truth = load_truth(train_id); truth = stable_truth_paths(per_event_truth)
        summary = {"train_id": train_id, **structure_metrics(estimated, truth), "reconstruction_nrmse": float(result["reconstruction_nrmse"]), "bic": float(result["bic"]), "bootstrap_count": int(result["bootstrap_count"])}
        event_metrics_rows = []
        for event in range(EVENT_COUNT):
            row = {"train_id": train_id, "event_index": event + 1, **event_metrics(estimated, per_event_truth[event], event)}
            event_rows.append(row); event_metrics_rows.append(row)
        summary["event_path_f1_median"] = float(np.median([row["path_f1"] for row in event_metrics_rows]))
        train_rows.append(summary); calibration.extend(calibration_rows(train_id, estimated, truth))
        plot_train(train_id, estimated, truth, per_event_truth, args.output_root / "figures" / "per_clicktrain" / f"{train_id}.png")
        print(f"[评估与 PNG] {index:03d}/{len(artifacts)} | {train_id}", flush=True)
    write_csv(args.output_root / "train_metrics.csv", train_rows); write_csv(args.output_root / "event_metrics.csv", event_rows); write_csv(args.output_root / "confidence_calibration.csv", calibration)
    brier = float(np.mean([(row["confidence"] - row["correct"]) ** 2 for row in calibration])) if calibration else float("nan")
    atomic_json(args.output_root / "summary.json", {"n_clicktrains": len(train_rows), "n_paths_for_calibration": len(calibration), "brier_score": brier})
    summary_figure(train_rows, calibration, args.output_root / "figures" / "summary.png")
    report(train_rows, calibration, args.output_root / "实验报告_v6.md")


if __name__ == "__main__":
    main()
