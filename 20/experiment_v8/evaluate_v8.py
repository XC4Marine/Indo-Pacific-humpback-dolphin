#!/usr/bin/env python3
"""Truth-only evaluator and report generator for Experiment 20 v8."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from protocol_v8 import DATA_ROOT, EVENT_COUNT, OUTPUT_ROOT, condition_name, discover_inputs, load_prediction, write_csv

TOLERANCE_MS = .03
METHODS = ("cepstrum_cfar", "med", "sparse_gate")
EPS = 1e-18
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _truth_paths(train_id: str) -> list[list[dict[str, float]]]:
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as payload:
        arrivals = np.asarray(payload["event_synthesis_arrivals"]); first = float(payload["first_arrival_s"])
    result = []
    for event in arrivals:
        rows = []
        for top, bottom in sorted({(int(row["top_bounces"]), int(row["bottom_bounces"])) for row in event}):
            selected = event[(event["top_bounces"] == top) & (event["bottom_bounces"] == bottom)]
            rows.append({"name": "direct" if top == bottom == 0 else f"{top}_{bottom}",
                         "delay_ms": float(np.median(selected["delay_s"]) - first) * 1e3})
        direct = next(row["delay_ms"] for row in rows if row["name"] == "direct")
        result.append([{"name": row["name"], "delay_ms": float(row["delay_ms"] - direct)} for row in rows if row["name"] != "direct"])
    return result


def _stable_truth(per_event: list[list[dict[str, float]]]) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for paths in per_event:
        for path in paths: grouped.setdefault(path["name"], []).append(path["delay_ms"])
    return sorted(float(np.median(delays)) for delays in grouped.values() if len(delays) >= 6)


def _matches(estimated: list[float], truth: list[float]) -> list[tuple[int, int]]:
    pairs = [(abs(float(value) - float(target)), index, target_index) for index, value in enumerate(estimated)
             for target_index, target in enumerate(truth) if abs(float(value) - float(target)) <= TOLERANCE_MS]
    selected: list[tuple[int, int]] = []; used_estimated: set[int] = set(); used_truth: set[int] = set()
    for _, index, target_index in sorted(pairs):
        if index not in used_estimated and target_index not in used_truth:
            selected.append((index, target_index)); used_estimated.add(index); used_truth.add(target_index)
    return selected


def _metric(estimated: list[float], truth: list[float]) -> dict[str, float]:
    pairs = _matches(estimated, truth); tp = len(pairs); fp = len(estimated) - tp; fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [abs(estimated[index] - truth[target_index]) for index, target_index in pairs]
    return {"precision": precision, "recall": recall, "f1": f1, "delay_mae_ms": float(np.mean(errors)) if errors else float("nan"),
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "predicted_count": float(len(estimated)), "truth_count": float(len(truth)),
            "exact_count": float(len(estimated) == len(truth))}


def _bootstrap(values: np.ndarray, token: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return float("nan"), float("nan")
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % (2**32); rng = np.random.default_rng(seed)
    samples = [float(np.median(rng.choice(values, len(values), replace=True))) for _ in range(1000)]
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    metric_names = ("stable_precision", "stable_recall", "stable_f1", "stable_delay_mae_ms", "stable_exact_count",
                    "event_f1", "reconstruction_nrmse", "runtime_ms")
    for condition in sorted({str(row["condition"]) for row in rows}):
        for method in METHODS:
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            if not selected: continue
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in selected], dtype=float)
                finite = values[np.isfinite(values)]
                low, high = _bootstrap(finite, f"{condition}:{method}:{metric}")
                result.append({"condition": condition, "method": method, "metric": metric, "n_clicktrains": len(values),
                               "median": float(np.median(finite)) if len(finite) else float("nan"),
                               "q25": float(np.quantile(finite, .25)) if len(finite) else float("nan"),
                               "q75": float(np.quantile(finite, .75)) if len(finite) else float("nan"),
                               "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return result


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        for index, left in enumerate(METHODS):
            for right in METHODS[index + 1:]:
                left_rows = {str(row["train_id"]): row for row in rows if row["condition"] == condition and row["method"] == left}
                right_rows = {str(row["train_id"]): row for row in rows if row["condition"] == condition and row["method"] == right}
                common = sorted(set(left_rows) & set(right_rows))
                for metric in ("stable_f1", "stable_precision", "stable_recall", "runtime_ms"):
                    values = np.asarray([float(right_rows[key][metric]) - float(left_rows[key][metric]) for key in common])
                    low, high = _bootstrap(values, f"paired:{condition}:{left}:{right}:{metric}")
                    result.append({"condition": condition, "comparison": f"{right}_minus_{left}", "metric": metric, "n_clicktrains": len(values),
                                   "median_delta": float(np.nanmedian(values)), "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return result


def _plot_summary(summary: list[dict[str, Any]], output: Path) -> None:
    conditions = sorted({str(row["condition"]) for row in summary}); metrics = ("stable_precision", "stable_recall", "stable_f1")
    figure, axes = plt.subplots(1, len(conditions), figsize=(5.2 * len(conditions), 4.2), squeeze=False)
    for axis, condition in zip(axes[0], conditions):
        selected = {(row["method"], row["metric"]): row for row in summary if row["condition"] == condition}
        positions = np.arange(len(METHODS)); width = .23
        for offset, metric in enumerate(metrics):
            values = [selected.get((method, metric), {}).get("median", np.nan) for method in METHODS]
            axis.bar(positions + (offset - 1) * width, values, width, label=metric.replace("stable_", ""))
        axis.set(title=condition, ylim=(0, 1.05), xticks=positions, xticklabels=METHODS, ylabel="按 clicktrain 的中位数")
        axis.legend(fontsize=8)
    figure.suptitle("v8 稳定多径结构指标"); figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=180); plt.close(figure)


def _plot_cases(rows: list[dict[str, Any]], output: Path) -> None:
    clean = [row for row in rows if row["condition"] == "clean" and row["method"] == "cepstrum_cfar"]
    if not clean: return
    ordered = sorted(clean, key=lambda row: (float(row["stable_f1"]), str(row["train_id"])))
    cases = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for axis, selected in zip(axes, cases):
        train_id = str(selected["train_id"]); truth = selected["truth_stable"]
        axis.vlines(truth, .70, 1.0, color="#222222", label="Bellhop 真值")
        for offset, method in enumerate(METHODS):
            row = next(row for row in rows if row["condition"] == "clean" and row["train_id"] == train_id and row["method"] == method)
            axis.vlines(row["stable_delays"], .08 + .18 * offset, .20 + .18 * offset, label=method)
        axis.set(title=f"{train_id}（CFAR F1={selected['stable_f1']:.3f}）", ylim=(0, 1.08), ylabel="路径")
        axis.legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("相对时延 (ms)"); figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=180); plt.close(figure)


def _markdown_table(summary: list[dict[str, Any]], condition: str, metrics: tuple[str, ...]) -> list[str]:
    lookup = {(row["condition"], row["method"], row["metric"]): row for row in summary}; header = "| 方法 | " + " | ".join(metrics) + " |"
    lines = [header, "|---|" + "---:|" * len(metrics)]
    for method in METHODS:
        values = []
        for metric in metrics:
            row = lookup.get((condition, method, metric)); values.append("N/A" if row is None or not np.isfinite(row["median"]) else f"{row['median']:.4f}")
        lines.append("| " + method + " | " + " | ".join(values) + " |")
    return lines


def _write_report(summary: list[dict[str, Any]], paired: list[dict[str, Any]], rows: list[dict[str, Any]], output: Path) -> None:
    coverage = len({str(row["train_id"]) for row in rows})
    result_status = ("正式 OOF 汇总：覆盖全部 110 个 clicktrain。" if coverage == 110 else
                     f"开发冒烟汇总：当前仅覆盖 {coverage}/110 个 clicktrain，不能作为正式方法比较结论。")
    lines = ["# 第20实验 v8：全自动多径时延拓扑提取报告", "", "## 协议", "", "- 数据：第20实验 110 个 Bellhop 合成 clicktrain。",
             "- 条件：clean、+5 dB、−5 dB；评价阶段独立读取物理到达数据。", "- 协议：按 69 个 `train_id` 分组的 5 折 OOF；共享门控仅用训练组 clean 波形自监督训练。",
             "- 匹配：跨事件覆盖至少 6/10 的反射结构，±0.03 ms 一对一匹配；绝对幅值不计入主结论。", f"- 结果状态：{result_status}", "- 下表只由本次已生成预测计算，不预设任何方法优劣。", ""]
    for condition in sorted({str(row["condition"]) for row in rows}):
        lines.extend([f"## {condition}", "", "### 稳定结构", ""])
        lines.extend(_markdown_table(summary, condition, ("stable_precision", "stable_recall", "stable_f1", "stable_delay_mae_ms", "stable_exact_count")))
        lines.extend(["", "### 辅助指标", ""])
        lines.extend(_markdown_table(summary, condition, ("event_f1", "reconstruction_nrmse", "runtime_ms")))
        lines.append("")
    failures = {method: {"漏检": 0, "过检": 0, "数量正确但失配": 0} for method in METHODS}
    for row in rows:
        if row["condition"] != "clean": continue
        method = str(row["method"])
        if row["stable_recall"] < 1: failures[method]["漏检"] += 1
        if row["stable_precision"] < 1: failures[method]["过检"] += 1
        if row["stable_exact_count"] and row["stable_f1"] < 1: failures[method]["数量正确但失配"] += 1
    lines.extend(["## 观察到的失败模式（clean）", "", "| 方法 | 漏检样本 | 过检样本 | 数量正确但时延失配 |", "|---|---:|---:|---:|"])
    lines.extend(f"| {method} | {values['漏检']} | {values['过检']} | {values['数量正确但失配']} |" for method, values in failures.items())
    lines.extend(["", "## 配对比较", "", "`paired_deltas.csv` 保存全部条件、方法对和指标的配对中位数差及95% bootstrap区间。"
                  "当区间跨越零时，报告不将该差异解释为稳定优势。", "", "## 产物", "", "- `summary_metrics.csv`：逐条件、逐方法的中位数、四分位数和95%区间。",
                  "- `per_clicktrain_metrics.csv` 与 `event_metrics.csv`：可审计的逐样本结果。", "- `summary.png`：主指标图；`diagnostic_cases.png`：按 CFAR F1 自动选择的困难、中位和最佳案例。"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_from_runner(conditions: tuple[int | None, ...], train_ids: set[str] | None = None) -> None:
    paths = [path for path in discover_inputs() if train_ids is None or path.parent.name in train_ids]
    if not paths: raise ValueError("no clicktrains selected for evaluation")
    per_train: list[dict[str, Any]] = []; events: list[dict[str, Any]] = []
    expected = len(paths) * len(conditions) * len(METHODS); completed = 0
    for snr_db in conditions:
        condition = condition_name(snr_db)
        for path in paths:
            train_id = path.parent.name; event_truth = _truth_paths(train_id); stable_truth = _stable_truth(event_truth)
            for method in METHODS:
                artifact = OUTPUT_ROOT / "predictions" / condition / method / f"{train_id}.npz"
                if not artifact.exists(): raise FileNotFoundError(f"missing prediction artifact: {artifact}")
                prediction = load_prediction(artifact); stable = _metric(prediction["stable_delays"], stable_truth)
                event_scores = []
                for event_index, (estimated, truth) in enumerate(zip(prediction["event_delays"], event_truth), 1):
                    score = _metric(estimated, [item["delay_ms"] for item in truth]); event_scores.append(score)
                    events.append({"train_id": train_id, "condition": condition, "method": method, "event_index": event_index, **score})
                no_reflection = not stable_truth
                per_train.append({"train_id": train_id, "condition": condition, "method": method, "fold": prediction["fold"],
                                  **{f"stable_{key}": value for key, value in stable.items()}, "event_f1": float(np.median([item["f1"] for item in event_scores])),
                                  "reconstruction_nrmse": prediction["reconstruction_nrmse"], "runtime_ms": prediction["runtime_ms"],
                                  "no_reflection": int(no_reflection), "no_reflection_specificity": float(no_reflection and not prediction["stable_delays"]),
                                  "stable_delays": prediction["stable_delays"], "truth_stable": stable_truth})
                completed += 1; print(f"[总体进度：真值评估] {completed}/{expected} | {condition} | {method} | {train_id}", flush=True)
    summary = _summaries(per_train); paired = _paired(per_train); root = OUTPUT_ROOT / "evaluation"; root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "per_clicktrain_metrics.csv", per_train); write_csv(root / "event_metrics.csv", events); write_csv(root / "summary_metrics.csv", summary); write_csv(root / "paired_deltas.csv", paired)
    _plot_summary(summary, root / "summary.png"); _plot_cases(per_train, root / "diagnostic_cases.png"); _write_report(summary, paired, per_train, root / "report_zh.md")
    print(f"真值评估完成：{root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--conditions", default="clean,5,-5")
    parser.add_argument("--train-ids", default=None, help="comma-separated artifact IDs for a non-reporting smoke evaluation")
    args = parser.parse_args(); mapping = {"clean": None, "5": 5, "-5": -5}
    selected = None if args.train_ids is None else {token.strip() for token in args.train_ids.split(",") if token.strip()}
    main_from_runner(tuple(mapping[token.strip()] for token in args.conditions.split(",")), selected)


if __name__ == "__main__":
    main()
