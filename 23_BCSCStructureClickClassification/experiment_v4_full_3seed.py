#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总实验23-v4的C0--C5完整五折、三个随机种子结果。

每个种子先运行experiment_v4_minimal.py train-full，随后运行本脚本的aggregate。
三种子平均概率用于录音组配对Bootstrap；逐种子和逐折结果同时保留。
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
BASE_PATH = ROOT / "23_BCSCStructureClickClassification" / "experiment_v4_minimal.py"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v4_full_5fold_3seeds"
SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = ("C0_waveform", "C1_full_zero", "C2_v2_scalar", "C3_reliability", "C4_local_gate", "C5_structure_dropout")
STEP_PAIRS = (("reliability", "C2_v2_scalar", "C3_reliability"),
              ("local_gate", "C3_reliability", "C4_local_gate"),
              ("structure_dropout", "C4_local_gate", "C5_structure_dropout"))


def base() -> Any:
    name = "experiment23_v4_full_aggregate_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paths() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    return {"root": OUT, "figures": OUT / "figures"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    for seed in SEEDS:
        seed_root = OUT / f"seed_{seed}"
        metric_path, prediction_path = seed_root / "full_fold_metrics.csv", seed_root / "full_oof_predictions.csv"
        if not metric_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"种子{seed}尚未完成完整五折：{seed_root}")
        metric, prediction = pd.read_csv(metric_path), pd.read_csv(prediction_path)
        if len(metric) != 30 or len(prediction) != 42_000:
            raise RuntimeError(f"种子{seed}结果不完整：metrics={len(metric)}, predictions={len(prediction)}")
        if set(metric.seed.astype(int)) != {seed} or set(prediction.seed.astype(int)) != {seed}:
            raise RuntimeError(f"种子{seed}文件中的seed字段错误")
        metrics.append(metric); predictions.append(prediction)
    metric_all, prediction_all = pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)
    if len(metric_all) != 90 or len(prediction_all) != 126_000:
        raise RuntimeError("三种子合同失败")
    if prediction_all.duplicated(["seed", "condition", "sample_index"]).any():
        raise RuntimeError("三种子OOF含重复样本")
    return metric_all, prediction_all


def make_ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    fixed = ["label", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]
    aggregation: dict[str, str] = {name: "first" for name in fixed}
    aggregation.update({"p_click": "mean", "logit": "mean"})
    ensemble = predictions.groupby(["condition", "sample_index"], as_index=False).agg(aggregation)
    if len(ensemble) != 42_000 or ensemble.duplicated(["condition", "sample_index"]).any():
        raise RuntimeError("三种子平均OOF合同失败")
    return ensemble


def summary_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_summary = metrics.groupby(["condition", "seed"], as_index=False).agg(
        mean_low_snr_auroc=("low_snr_auroc", "mean"), mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"))
    overall = seed_summary.groupby("condition", as_index=False).agg(
        seeds=("seed", "nunique"), low_snr_auroc_mean=("mean_low_snr_auroc", "mean"),
        low_snr_auroc_seed_std=("mean_low_snr_auroc", "std"), auroc_mean=("mean_auroc", "mean"),
        auroc_seed_std=("mean_auroc", "std"), auprc_mean=("mean_auprc", "mean"),
        auprc_seed_std=("mean_auprc", "std"), balanced_accuracy_mean=("mean_balanced_accuracy", "mean"))
    return seed_summary, overall


def step_deltas(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for step, left, right in STEP_PAIRS:
        left_frame = metrics[metrics.condition == left].set_index(["seed", "test_fold"])
        right_frame = metrics[metrics.condition == right].set_index(["seed", "test_fold"])
        for key in left_frame.index:
            records.append({"step": step, "seed": int(key[0]), "test_fold": int(key[1]),
                            "from_condition": left, "to_condition": right,
                            "delta_low_snr_auroc": float(right_frame.loc[key, "low_snr_auroc"] - left_frame.loc[key, "low_snr_auroc"]),
                            "delta_auroc": float(right_frame.loc[key, "auroc"] - left_frame.loc[key, "auroc"])})
    details = pd.DataFrame(records)
    rows = []
    for step, item in details.groupby("step", sort=False):
        deltas = item.delta_low_snr_auroc.to_numpy()
        direction = "一致的改善方向" if np.all(deltas > 0) else "结果不稳定" if np.any(deltas > 0) else "当前配置下可能无效"
        rows.append({"step": step, "model_folds": len(item), "mean_delta_low_snr_auroc": float(np.mean(deltas)),
                     "std_delta_low_snr_auroc": float(np.std(deltas, ddof=1)), "positive_folds": int(np.sum(deltas > 0)),
                     "direction": direction})
    return details, pd.DataFrame(rows)


def plot_roc_pr(ensemble: pd.DataFrame, destination: Path, labels: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for condition in CONDITIONS:
        item = ensemble[ensemble.condition == condition]
        fpr, tpr, _ = roc_curve(item.label, item.p_click)
        precision, recall, _ = precision_recall_curve(item.label, item.p_click)
        axes[0].plot(fpr, tpr, label=f"{labels[condition]} ({roc_auc_score(item.label, item.p_click):.3f})")
        axes[1].plot(recall, precision, label=f"{labels[condition]} ({average_precision_score(item.label, item.p_click):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[0].set(xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1.02), title="三种子平均OOF ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="三种子平均OOF PR")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def plot_steps(details: pd.DataFrame, destination: Path) -> None:
    steps = ["reliability", "local_gate", "structure_dropout"]
    data = [details.loc[details.step == step, "delta_low_snr_auroc"].to_numpy() for step in steps]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.boxplot(data, tick_labels=["+Reliability", "+Local Gate", "+Structure Dropout"], showmeans=True)
    axis.axhline(0, color="black", lw=.8)
    axis.set(ylabel="低SNR AUROC增量", title="完整五折、三种子的逐步消融分布")
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns, rows = list(frame.columns), []
    for row in frame.itertuples(index=False, name=None):
        rows.append([f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) and math.isfinite(value) else str(value) for value in row])
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def aggregate(bootstrap_runs: int) -> None:
    local_base, all_metrics, all_predictions = base(), *load_inputs()
    output = paths()
    ensemble = make_ensemble(all_predictions)
    seed_summary, overall = summary_tables(all_metrics)
    step_detail, step_summary = step_deltas(all_metrics)
    comparisons = []
    pairs = (("C5_structure_dropout", "C0_waveform"), ("C5_structure_dropout", "C1_full_zero"),
             ("C3_reliability", "C2_v2_scalar"), ("C4_local_gate", "C3_reliability"),
             ("C5_structure_dropout", "C4_local_gate"))
    for index, (left, right) in enumerate(pairs):
        for endpoint in ("low_snr_auroc", "auroc"):
            row = local_base.paired_bootstrap(ensemble, left, right, endpoint, bootstrap_runs, 20260826 + index * 100 + len(endpoint))
            row["probability_aggregation"] = "three_seed_mean"
            comparisons.append(row)
    bootstrap = pd.DataFrame(comparisons)
    all_metrics.to_csv(output["root"] / "full_3seed_fold_metrics.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(output["root"] / "full_3seed_oof_predictions.csv", index=False, encoding="utf-8-sig")
    ensemble.to_csv(output["root"] / "full_3seed_mean_probability_oof.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(output["root"] / "full_3seed_seed_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output["root"] / "full_3seed_summary.csv", index=False, encoding="utf-8-sig")
    step_detail.to_csv(output["root"] / "full_3seed_step_deltas.csv", index=False, encoding="utf-8-sig")
    step_summary.to_csv(output["root"] / "full_3seed_step_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output["root"] / "full_3seed_paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    plot_roc_pr(ensemble, output["root"] / "full_3seed_roc_pr.png", local_base.LABELS)
    plot_steps(step_detail, output["root"] / "full_3seed_ablation_steps.png")
    main = bootstrap[(bootstrap.left == "C5_structure_dropout") & (bootstrap.right.isin(("C0_waveform", "C1_full_zero"))) & (bootstrap.endpoint == "low_snr_auroc")]
    supports = bool(main.supports_positive_gain.all())
    report = ["# 实验23-v4：完整五折、三随机种子确认", "",
              "## 设计", "", "C0--C5均完成五折，随机种子为20260826、20260827、20260828，共90个模型折。主统计以每个样本三个种子预测概率的均值计算，并采用34个录音组配对Bootstrap。逐种子五折均值另行保留。", "",
              "## 三种子五折汇总", "", markdown_table(overall), "",
              "## 顺序组件消融", "", markdown_table(step_summary), "",
              "## 三种子平均概率的配对Bootstrap", "", markdown_table(bootstrap), ""]
    report.extend(["## 核心结论", "", "C5相对C0和C1的主终点95%置信区间下限均大于0，BCSC结构增益在三种子平均概率的录音组Bootstrap下成立。" if supports else "至少一个C5核心比较的主终点95%置信区间下限不大于0；因此只能报告提升趋势，不能声称BCSC结构增益成立。", "",
                   "说明：此Bootstrap反映录音组重采样不确定性，三种子不被当作独立Bootstrap单元；种子间离散度见`full_3seed_seed_summary.csv`。"])
    (output["root"] / "research_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[总体进度 100%] 已汇总90个模型折：{output['root']}", flush=True)


def structural_gain(bootstrap_runs: int) -> None:
    """C2相对C0/C1的最小充分结构增益检验，不重训模型。"""
    local_base, all_metrics, all_predictions = base(), *load_inputs()
    output = paths()
    ensemble = make_ensemble(all_predictions)
    comparisons = []
    for index, right in enumerate(("C0_waveform", "C1_full_zero")):
        for endpoint in ("low_snr_auroc", "auroc"):
            row = local_base.paired_bootstrap(ensemble, "C2_v2_scalar", right, endpoint,
                                              bootstrap_runs, 20260926 + index * 100 + len(endpoint))
            row["probability_aggregation"] = "three_seed_mean"
            comparisons.append(row)
    bootstrap = pd.DataFrame(comparisons)
    seed_means = all_metrics.groupby(["condition", "seed"], as_index=False).agg(
        low_snr_auroc=("low_snr_auroc", "mean"), auroc=("auroc", "mean"))
    records: list[dict[str, Any]] = []
    for right in ("C0_waveform", "C1_full_zero"):
        left = seed_means[seed_means.condition == "C2_v2_scalar"].set_index("seed")
        baseline = seed_means[seed_means.condition == right].set_index("seed")
        for seed in SEEDS:
            records.append({"left": "C2_v2_scalar", "right": right, "seed": seed,
                            "delta_low_snr_auroc": float(left.loc[seed, "low_snr_auroc"] - baseline.loc[seed, "low_snr_auroc"]),
                            "delta_auroc": float(left.loc[seed, "auroc"] - baseline.loc[seed, "auroc"])})
    seed_deltas = pd.DataFrame(records)
    bootstrap.to_csv(output["root"] / "c2_structural_gain_bootstrap.csv", index=False, encoding="utf-8-sig")
    seed_deltas.to_csv(output["root"] / "c2_structural_gain_seed_deltas.csv", index=False, encoding="utf-8-sig")
    primary = bootstrap[bootstrap.endpoint == "low_snr_auroc"]
    supports = bool(primary.supports_positive_gain.all())
    text = ["# C2结构增益的三种子配对Bootstrap检验", "",
            "C2保留原始BCSC结构图并经Path-CNN与波形特征全局融合。C0仅使用波形；C1保留同构结构分支但将结构输入置零。概率先在三个随机种子间取均值，再以34个录音组为重采样单元进行2,000次配对Bootstrap。", "",
            "## Bootstrap结果", "", markdown_table(bootstrap), "",
            "## 种子级五折平均增量", "", markdown_table(seed_deltas), "",
            "## 结论", "",
            "C2相对C0和C1的低SNR AUROC配对Bootstrap 95%置信区间下限均大于0，支持原始BCSC结构图带来的增益。" if supports else "至少一个低SNR AUROC比较的95%置信区间下限不大于0。因此结果仅支持结构增益趋势，不能声称结构增益成立。"]
    (output["root"] / "c2_structural_gain_report_zh.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"[总体进度 100%] 已完成C2结构增益Bootstrap：{output['root']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("aggregate", "c2-structural-gain"))
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "aggregate":
        print("[总体进度 90%] 汇总完整五折、三随机种子结果", flush=True)
        aggregate(args.bootstrap_runs)
    else:
        print("[总体进度 90%] 执行C2相对C0/C1的结构增益Bootstrap", flush=True)
        structural_gain(args.bootstrap_runs)


if __name__ == "__main__":
    main()
