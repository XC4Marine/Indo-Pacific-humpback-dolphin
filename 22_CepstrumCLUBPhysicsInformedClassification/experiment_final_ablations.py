# -*- coding: utf-8 -*-
"""Experiment 22 final structured-factorization ablations.

This runner evaluates four matched conditions without mutual-information,
cross-consistency, or cepstral supervision.  It writes source-traceable OOF
predictions, latent diagnostics, source-bootstrap summaries, a Markdown table,
and a three-panel editable SVG figure.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import CCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from tqdm.auto import tqdm

import experiment as base
from experiment_cross_convolution_consistency import HDecoder, HEncoder, XDecoder, XEncoder, band_loss, compact_loss, h_loss

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs_final_structured_ablations_3seeds"
CONDITIONS = ("x_only_bce", "homogeneous_independent", "heterogeneous_no_priors", "heterogeneous_priors")
SEEDS = (20260826, 20260827, 20260828)
PRIOR_WEIGHTS = (0.3845476064627951, 0.013942945255098234, 0.013682163929311455)
SNR_BINS = ((-15., -10., "[-15, -10)"), (-10., -5., "[-10, -5)"), (-5., 0., "[-5, 0)"), (0., 5., "[0, 5)"), (5., 10., "[5, 10)"), (10., 15.000001, "[10, 15]"))

@dataclass(frozen=True)
class Config:
    implementation_version: str = "experiment22_final_structured_ablations_v1"
    output_dir: Path = DEFAULT_OUTPUT
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 5
    bootstrap_runs: int = 2000
    device: str = "auto"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    sample_rate: int = 576000
    sample_count: int = 5761
    channel_count: int = 2881
    latent_count: int = 64
    tukey_alpha: float = .04
    epsilon: float = 1e-8
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def paths(c: Config) -> dict[str, Path]:
    result = {"root": c.output_dir, "runs": c.output_dir / "runs", "metrics": c.output_dir / "metrics", "predictions": c.output_dir / "predictions", "latents": c.output_dir / "latent_vectors"}
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result


class XOnlyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.E_x = XEncoder(); self.classifier = nn.Linear(64, 1)
    def forward(self, received):
        z_click = self.E_x(received)
        return {"z_click": z_click, "logit": self.classifier(z_click).squeeze(1)}


class FactorModel(nn.Module):
    def __init__(self, c: Config, homogeneous: bool):
        super().__init__()
        self.E_x = XEncoder()
        self.E_h = XEncoder() if homogeneous else HEncoder()
        self.D_x = XDecoder(c)
        self.D_h = HDecoder()
        self.classifier = nn.Linear(64, 1)
    def forward(self, received):
        z_click, z_channel = self.E_x(received), self.E_h(received)
        x_hat, h_hat = self.D_x(z_click), self.D_h(z_channel)
        return {"z_click": z_click, "z_channel": z_channel, "x_hat": x_hat, "h_hat": h_hat, "r_hat": base.causal_fft_convolution(x_hat, h_hat), "logit": self.classifier(z_click).squeeze(1)}


def make_model(condition: str, c: Config) -> nn.Module:
    if condition == "x_only_bce": return XOnlyModel()
    return FactorModel(c, homogeneous=condition == "homogeneous_independent")


def metric(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= thresholds
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)), "auprc": float(average_precision_score(labels, probabilities)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "f1": float(f1_score(labels, predicted, zero_division=0)), "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp))}


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.r_[0., probabilities, 1.])
    scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    return float(candidates[np.argmax(scores)])


def prior_loss(out: dict[str, torch.Tensor], labels: torch.Tensor, c: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = labels > .5
    if not bool(positive.any()):
        zero = out["x_hat"].sum() * 0.; return zero, zero, zero
    return h_loss(out["h_hat"][positive], c), compact_loss(out["x_hat"][positive], c).mean(), band_loss(out["x_hat"][positive], c).mean()


def objective(out: dict[str, torch.Tensor], received: torch.Tensor, labels: torch.Tensor, condition: str, c: Config) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bce = nn.functional.binary_cross_entropy_with_logits(out["logit"], labels)
    if condition == "x_only_bce":
        return bce, {"loss_bce": bce, "loss_total": bce}
    reconstruction = base.reconstruction_loss(received, out["r_hat"], c.epsilon)
    total = reconstruction + bce
    if condition == "heterogeneous_no_priors":
        zero = total * 0.
        return total, {"loss_recon": reconstruction, "loss_bce": bce, "loss_h_raw": zero, "loss_compact_raw": zero, "loss_band_raw": zero, "loss_total": total}
    h, compact, band = prior_loss(out, labels, c)
    total = total + PRIOR_WEIGHTS[0] * h + PRIOR_WEIGHTS[1] * compact + PRIOR_WEIGHTS[2] * band
    return total, {"loss_recon": reconstruction, "loss_bce": bce, "loss_h_raw": h, "loss_compact_raw": compact, "loss_band_raw": band, "loss_total": total}


def predict(model: nn.Module, loader, device: torch.device, condition: str) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    model.eval(); frames, zx, zh = [], [], []
    with torch.no_grad():
        for received, labels, indices in loader:
            out = model(received.to(device)); frames.append(pd.DataFrame({"sample_index": indices.numpy(), "label": labels.numpy().astype(int), "probability": torch.sigmoid(out["logit"]).cpu().numpy()})); zx.append(out["z_click"].cpu().numpy())
            if condition != "x_only_bce": zh.append(out["z_channel"].cpu().numpy())
    latent = {"z_click": np.concatenate(zx)} if condition == "x_only_bce" else {"z_click": np.concatenate(zx), "z_channel": np.concatenate(zh)}
    return pd.concat(frames, ignore_index=True), latent


def fit(condition: str, seed: int, fold: int, waves: np.ndarray, manifest: pd.DataFrame, c: Config, device: torch.device, resume: bool, bar: tqdm) -> tuple[dict[str, Any], pd.DataFrame]:
    run = paths(c)["runs"] / condition / f"seed_{seed}" / f"fold_{fold}"; run.mkdir(parents=True, exist_ok=True)
    best, complete, history, predictions_file = run / "best_checkpoint.pt", run / "COMPLETE.json", run / "history.csv", run / "predictions.csv"
    if resume and complete.exists() and all(p.exists() for p in (best, history, predictions_file)):
        bar.update(); return json.loads((run / "metrics.json").read_text(encoding="utf-8")), pd.read_csv(predictions_file)
    seed_everything(seed + fold); model = make_model(condition, c).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    labels = manifest.label.to_numpy(np.float32); train, validation, test = base.split_indices(manifest, fold); records, best_score, stale = [], -np.inf, 0
    for epoch in range(1, c.max_epochs + 1):
        batch_values = []
        model.train()
        for received, y, _ in base.epoch_train_loader(waves, labels, train, c.batch_size, seed + fold, epoch, 0):
            received, y = received.to(device), y.to(device); optimizer.zero_grad(set_to_none=True); out = model(received); total, values = objective(out, received, y, condition, c); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); batch_values.append({key: float(value.detach().cpu()) for key, value in values.items()})
        validation_frame, _ = predict(model, base.evaluation_loader(waves, labels, validation, c.batch_size, 0), device, condition); score = float(roc_auc_score(validation_frame.label, validation_frame.probability)); records.append({"condition": condition, "seed": seed, "fold": fold, "epoch": epoch, "validation_auroc": score, **{key: float(np.mean([row[key] for row in batch_values])) for key in batch_values[0]}})
        if score > best_score + 1e-4:
            best_score, stale = score, 0; torch.save({"model": model.state_dict(), "epoch": epoch, "validation_auroc": score}, best)
        else: stale += 1
        pd.DataFrame(records).to_csv(history, index=False, encoding="utf-8-sig")
        if stale >= c.patience: break
    saved = torch.load(best, map_location=device, weights_only=True); model.load_state_dict(saved["model"])
    validation_frame, _ = predict(model, base.evaluation_loader(waves, labels, validation, c.batch_size, 0), device, condition); threshold = choose_threshold(validation_frame.label.to_numpy(), validation_frame.probability.to_numpy())
    frame, latent = predict(model, base.evaluation_loader(waves, labels, test, c.batch_size, 0), device, condition); frame["condition"], frame["seed"], frame["fold"], frame["threshold"] = condition, seed, fold, threshold
    frame = frame.merge(manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "target_snr_db"]], on="sample_index", how="left", validate="one_to_one")
    frame.to_csv(predictions_file, index=False, encoding="utf-8-sig")
    result = metric(frame.label.to_numpy(), frame.probability.to_numpy(), frame.threshold.to_numpy()) | {"condition": condition, "seed": seed, "fold": fold, "best_epoch": saved["epoch"], "validation_auroc": saved["validation_auroc"], "total_parameters": sum(p.numel() for p in model.parameters())}
    write_json(run / "metrics.json", result); write_json(complete, {"training_complete": True})
    if condition != "x_only_bce": np.savez_compressed(paths(c)["latents"] / f"{condition}_seed_{seed}_fold_{fold}.npz", sample_id=frame.sample_id.to_numpy(), label=frame.label.to_numpy(), ori_file_num=frame.ori_file_num.to_numpy(), z_click=latent["z_click"], z_channel=latent["z_channel"])
    bar.update(); return result, frame


def cca_summary(c: Config) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        if condition == "x_only_bce": continue
        for seed in SEEDS:
            arrays = [np.load(paths(c)["latents"] / f"{condition}_seed_{seed}_fold_{fold}.npz") for fold in range(5)]
            label = np.concatenate([z["label"] for z in arrays]); source = np.concatenate([z["ori_file_num"] for z in arrays]); zx = np.concatenate([z["z_click"] for z in arrays]); zh = np.concatenate([z["z_channel"] for z in arrays]); mask = label == 1; zx, zh, source = zx[mask], zh[mask], source[mask]
            for method in ("linear_cca", "kernel_cca_nystrom"):
                a, b = zx.copy(), zh.copy(); rng = np.random.default_rng(seed)
                if method == "kernel_cca_nystrom":
                    landmarks = min(32, len(a)); index = rng.choice(len(a), landmarks, replace=False); scale = max(float(np.median(np.linalg.norm(a[index,None] - a[None,index], axis=2))), 1e-6); a = Nystroem(gamma=1/(2*scale**2), n_components=landmarks, random_state=seed).fit_transform(a); b = Nystroem(gamma=1/(2*scale**2), n_components=landmarks, random_state=seed).fit_transform(b)
                u, v = CCA(n_components=10, max_iter=1000).fit_transform(a, b); corr = [float(np.corrcoef(u[:, i], v[:, i])[0, 1]) for i in range(10)]
                rows.append({"condition": condition, "seed": seed, "subset": "positive", "method": method, **{f"corr_{i+1}": corr[i] for i in range(10)}, "mean_top10": float(np.mean(corr)), "positive_samples": len(source)})
    result = pd.DataFrame(rows); result.to_csv(paths(c)["metrics"] / "cca_per_seed.csv", index=False, encoding="utf-8-sig"); return result


def source_bootstrap(frames: list[pd.DataFrame], runs: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    """Bootstrap recording sources within each seed and average seed estimates."""
    values = {key: [] for key in ("auroc", "auprc", "f1")}
    grouped = [list(frame.groupby("ori_file_num", sort=False)) for frame in frames]
    for _ in range(runs):
        estimates = {key: [] for key in values}
        for groups in grouped:
            selected = rng.integers(0, len(groups), len(groups))
            sampled = pd.concat([groups[index][1] for index in selected], ignore_index=True)
            current = metric(sampled.label.to_numpy(), sampled.probability.to_numpy(), sampled.threshold.to_numpy())
            for key in values:
                estimates[key].append(current[key])
        for key in values:
            values[key].append(float(np.mean(estimates[key])))
    return {key: (float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))) for key, value in values.items()}
def summarize(c: Config, predictions: pd.DataFrame) -> None:
    out = paths(c)["metrics"]
    predictions.to_csv(paths(c)["predictions"] / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    seed_rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            frame = predictions[(predictions.condition == condition) & (predictions.seed == seed)]
            seed_rows.append({"condition": condition, "seed": seed, **metric(frame.label.to_numpy(), frame.probability.to_numpy(), frame.threshold.to_numpy())})
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(out / "classification_per_seed.csv", index=False, encoding="utf-8-sig")

    bootstrap_file = out / "classification_source_bootstrap.csv"
    if bootstrap_file.exists():
        cached = pd.read_csv(bootstrap_file)
        bootstrap_map = {row.condition: {key: (float(row[f"{key}_ci_low"]), float(row[f"{key}_ci_high"])) for key in ("auroc", "auprc", "f1")} for _, row in cached.iterrows()}
    else:
        bootstrap_rows, bootstrap_map = [], {}
        for condition in CONDITIONS:
            frames = [predictions[(predictions.condition == condition) & (predictions.seed == seed)] for seed in SEEDS]
            intervals = source_bootstrap(frames, c.bootstrap_runs, np.random.default_rng(10_000 + CONDITIONS.index(condition)))
            bootstrap_map[condition] = intervals
            bootstrap_rows.append({"condition": condition, **{f"{key}_ci_{side}": interval[position] for key, interval in intervals.items() for side, position in (("low", 0), ("high", 1))}})
        pd.DataFrame(bootstrap_rows).to_csv(bootstrap_file, index=False, encoding="utf-8-sig")

    cca = cca_summary(c)
    table = []
    for condition in CONDITIONS:
        values = seed_metrics[seed_metrics.condition == condition]
        row = {"condition": condition, "random_seeds": len(SEEDS)}
        for key in ("auroc", "auprc", "f1"):
            row[f"{key}_mean"] = float(values[key].mean())
            row[f"{key}_sd"] = float(values[key].std(ddof=1))
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = bootstrap_map[condition][key]
        if condition == "x_only_bce":
            row["linear_cca_top10_mean"] = np.nan
            row["kernel_cca_top10_mean"] = np.nan
        else:
            for method, name in (("linear_cca", "linear_cca_top10_mean"), ("kernel_cca_nystrom", "kernel_cca_top10_mean")):
                current = cca[(cca.condition == condition) & (cca.method == method)].mean_top10
                row[name] = float(current.mean())
                row[name + "_sd"] = float(current.std(ddof=1))
        table.append(row)
    result = pd.DataFrame(table)
    result.to_csv(out / "ablation_main_table.csv", index=False, encoding="utf-8-sig")

    snr_rows, snr_frames = [], {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            current = predictions[(predictions.condition == condition) & (predictions.seed == seed)]
            positives = current[current.label == 1].copy()
            negatives = current[current.label == 0].copy()
            positives["target_snr_db"] = pd.to_numeric(positives.target_snr_db)
            for low, high, label in SNR_BINS:
                selected = positives[(positives.target_snr_db >= low) & (positives.target_snr_db < high)]
                neg = negatives[negatives.ori_file_num.isin(selected.ori_file_num.unique())]
                combined = pd.concat((selected, neg), ignore_index=True)
                if combined.label.nunique() != 2:
                    raise RuntimeError(f"SNR bin lacks both classes: {condition}, {seed}, {label}")
                snr_frames[(condition, label, seed)] = combined
                snr_rows.append({"condition": condition, "seed": seed, "snr_bin": label, "snr_center_db": (low + min(high, 15.)) / 2, "positive_count": len(selected), "negative_count": len(neg), **metric(combined.label.to_numpy(), combined.probability.to_numpy(), combined.threshold.to_numpy())})
    snr = pd.DataFrame(snr_rows)
    snr.to_csv(out / "snr_per_seed.csv", index=False, encoding="utf-8-sig")
    summary = snr.groupby(["condition", "snr_bin", "snr_center_db"], as_index=False).agg({"auroc": ["mean", "std"], "auprc": ["mean", "std"], "f1": ["mean", "std"], "positive_count": "mean", "negative_count": "mean"})
    summary.columns = ["_".join(filter(None, col)).rstrip("_") for col in summary.columns]
    snr_bootstrap_file = out / "snr_source_bootstrap.csv"
    if snr_bootstrap_file.exists():
        snr_intervals = pd.read_csv(snr_bootstrap_file)
    else:
        interval_rows = []
        labels = [item[2] for item in SNR_BINS]
        for condition in CONDITIONS:
            for label in labels:
                frames = [snr_frames[(condition, label, seed)] for seed in SEEDS]
                intervals = source_bootstrap(frames, c.bootstrap_runs, np.random.default_rng(20_000 + 100 * CONDITIONS.index(condition) + labels.index(label)))
                interval_rows.append({"condition": condition, "snr_bin": label, **{f"{key}_ci_{side}": interval[position] for key, interval in intervals.items() for side, position in (("low", 0), ("high", 1))}})
        snr_intervals = pd.DataFrame(interval_rows)
        snr_intervals.to_csv(snr_bootstrap_file, index=False, encoding="utf-8-sig")
    summary = summary.merge(snr_intervals, on=["condition", "snr_bin"], how="left", validate="one_to_one")
    summary.to_csv(out / "snr_summary.csv", index=False, encoding="utf-8-sig")
    report_table(c, result)

def report_table(c: Config, table: pd.DataFrame) -> None:
    labels = {"x_only_bce":"X-only BCE", "homogeneous_independent":"同构独立双分支", "heterogeneous_no_priors":"异构，无物理损失", "heterogeneous_priors":"异构，三项物理损失"}
    lines = ["# 最终模型消融结果", "", "| 条件 | 随机种子 | AUROC | AUPRC | F1 | 正类线性 CCA（前10均值） | 正类 Kernel CCA（前10均值） |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        cca = "不适用" if pd.isna(row.linear_cca_top10_mean) else f"{row.linear_cca_top10_mean:.4f} ± {row.linear_cca_top10_mean_sd:.4f}"
        kcca = "不适用" if pd.isna(row.kernel_cca_top10_mean) else f"{row.kernel_cca_top10_mean:.4f} ± {row.kernel_cca_top10_mean_sd:.4f}"
        lines.append(f"| {labels[row.condition]} | {row.random_seeds} | {row.auroc_mean:.4f} ± {row.auroc_sd:.4f} [{row.auroc_ci_low:.4f}, {row.auroc_ci_high:.4f}] | {row.auprc_mean:.4f} ± {row.auprc_sd:.4f} [{row.auprc_ci_low:.4f}, {row.auprc_ci_high:.4f}] | {row.f1_mean:.4f} ± {row.f1_sd:.4f} [{row.f1_ci_low:.4f}, {row.f1_ci_high:.4f}] | {cca} | {kcca} |")
    lines += ["", "性能数值为三个随机种子的 OOF 均值 ± 标准差，并附来源级 bootstrap 95% 区间。CCA 与 Kernel CCA 在真实正类上计算；较低相关只表示较弱的统计依赖。"]
    (paths(c)["root"] / "ablation_results_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blueprint(c: Config) -> None:
    """Write the Chinese paper blueprint only from the new, completed result files."""
    root = paths(c)["root"]
    table = pd.read_csv(paths(c)["metrics"] / "ablation_main_table.csv")
    snr = pd.read_csv(paths(c)["metrics"] / "snr_summary.csv")
    labels = {"x_only_bce": "X-only BCE", "homogeneous_independent": "同构独立双分支", "heterogeneous_no_priors": "异构，无物理损失", "heterogeneous_priors": "异构，三项物理损失"}
    final = table[table.condition == "heterogeneous_priors"].iloc[0]
    no_prior = table[table.condition == "heterogeneous_no_priors"].iloc[0]
    result_table = (root / "ablation_results_zh.md").read_text(encoding="utf-8")
    count_rows = snr.groupby("snr_bin", sort=False)[["positive_count_mean", "negative_count_mean"]].mean().round().astype(int)
    bin_counts = "；".join(f"{index}: 正类约 {row.positive_count_mean}，同来源负类约 {row.negative_count_mean}" for index, row in count_rows.iterrows())
    lines = [
        "# 第22实验：JASA 论文蓝图（最终模型消融与 SNR 证据）",
        "",
        "## 研究问题与证据边界",
        "",
        "本文检验异构 X/H 卷积分解是否在来源隔离的点击检测中带来可复现的检测表现，并以真实正类潜变量的线性 CCA 与 Kernel CCA 描述两潜变量之间的统计依赖。所有结果均来自本轮独立的四条件、三随机种子、五折来源隔离实验；不与不同训练协议的历史结果拼接或比较。",
        "",
        "## 固定实验设计",
        "",
        "四个条件共享数据划分、初始化映射、样本顺序、训练预算和在验证集上选择阈值的规则：(1) X-only BCE；(2) 同构独立双分支；(3) 异构、无物理损失；(4) 异构、三项物理损失。X-only 不产生信道潜变量，因此不报告 CCA 或 Kernel CCA。性能以 3 个随机种子的 OOF 均值±标准差报告；方括号为录音来源重采样的 95% 区间。",
        "",
        "## 图表编排",
        "",
        "- 图 1：最终异构 X/H 卷积分解模型结构图，见 `figures/jasa_experiment22_final_xh_architecture.svg`。",
        "- 图 2：六个 SNR 区间下的 AUROC、AUPRC 和 F1 三面板曲线，见 `figures/jasa_experiment22_figure2_snr_ablation.svg`。每一 SNR 区间均使用该区间正类与同录音来源的全部 OOF 负类；阴影为来源级 95% bootstrap 区间。",
        "- 本轮仅保留图 1 与图 2；图 3、图 4 待有独立的真实 CCA 谱与重建诊断后再编号。",
        "",
        "## 主消融结果",
        "",
        result_table.strip(),
        "",
        "## 结果阐释",
        "",
        f"异构、三项物理损失条件的 OOF AUROC、AUPRC 与 F1 分别为 {final.auroc_mean:.4f}、{final.auprc_mean:.4f} 和 {final.f1_mean:.4f}。相对异构、无物理损失条件，其三项均值差值分别为 {final.auroc_mean - no_prior.auroc_mean:+.4f}、{final.auprc_mean - no_prior.auprc_mean:+.4f} 和 {final.f1_mean - no_prior.f1_mean:+.4f}。这些差值应结合来源级 95% 区间解读，不将单一指标变化表述为因果性物理恢复结论。",
        "",
        "除 X-only 外，正类线性 CCA 与 Kernel CCA 的前十维均值用于描述 click latent 与 channel latent 的统计依赖；它们不是独立性的充分证明，也不能单独证明真实信道恢复。",
        "",
        "## SNR 条件下的检测表现",
        "",
        f"图 2 覆盖的有效样本池为：{bin_counts}。三面板分别报告 AUROC、AUPRC 与采用所属折验证阈值计算的 F1，因此展示的是同一固定实验协议下，模型条件与正类 SNR 之间的经验关系。",
        "",
        "## 可追溯工件",
        "",
        "- `metrics/ablation_main_table.csv`：主表的全部数值。",
        "- `metrics/classification_per_seed.csv` 与 `metrics/classification_source_bootstrap.csv`：性能种子汇总及来源级区间。",
        "- `metrics/cca_per_seed.csv`：正类 CCA 与 Kernel CCA。",
        "- `metrics/snr_per_seed.csv`、`metrics/snr_source_bootstrap.csv` 与 `metrics/snr_summary.csv`：SNR 曲线数值及区间。",
        "- `predictions/oof_predictions.csv`：84,000 条条件—种子—样本 OOF 预测。",
    ]
    (root / "JASA_experiment22_final_blueprint_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def plot(c: Config) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype":"none", "font.family":"DejaVu Sans", "font.size":10})
    d = pd.read_csv(paths(c)["metrics"] / "snr_summary.csv"); names = {"x_only_bce":"X-only BCE", "homogeneous_independent":"Homogeneous independent", "heterogeneous_no_priors":"Heterogeneous, no priors", "heterogeneous_priors":"Heterogeneous + priors"}; colors = {"x_only_bce":"#666666", "homogeneous_independent":"#E69F00", "heterogeneous_no_priors":"#009E73", "heterogeneous_priors":"#0072B2"}; markers = {"x_only_bce":"s", "homogeneous_independent":"^", "heterogeneous_no_priors":"D", "heterogeneous_priors":"o"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, metric, title in zip(axes, ("auroc", "auprc", "f1"), ("AUROC", "AUPRC", "F1")):
        for condition in CONDITIONS:
            x = d[d.condition == condition].sort_values("snr_center_db"); y = x[f"{metric}_mean"]
            ax.plot(x.snr_center_db, y, marker=markers[condition], color=colors[condition], label=names[condition], linewidth=2)
            ax.fill_between(x.snr_center_db, x[f"{metric}_ci_low"].clip(0,1), x[f"{metric}_ci_high"].clip(0,1), color=colors[condition], alpha=.13)
        ax.set_title(title); ax.set_xlabel("Positive-click SNR (dB)"); ax.set_ylim(0, 1.03); ax.set_xticks([-12.5,-7.5,-2.5,2.5,7.5,12.5]); ax.grid(True, alpha=.25)
    axes[0].set_ylabel("OOF performance"); axes[2].legend(loc="lower right", fontsize=8); fig.suptitle("SNR-stratified detection performance across final-model ablations", y=1.03); fig.text(.5, -.03, "Each SNR bin combines positive clicks with all OOF negative samples from the same recording sources; shaded bands are source-level bootstrap 95% intervals (n=3 seeds).", ha="center", fontsize=9); fig.tight_layout(); figure_dir = paths(c)["root"] / "figures"; figure_dir.mkdir(parents=True, exist_ok=True); fig.savefig(figure_dir / "jasa_experiment22_figure2_snr_ablation.svg", bbox_inches="tight"); plt.close(fig)


def self_test(c: Config, device: torch.device) -> None:
    received = torch.randn(4, c.sample_count, device=device)
    labels = torch.tensor([1., 1., 0., 0.], device=device)
    for condition in CONDITIONS:
        model = make_model(condition, c).to(device)
        out = model(received)
        total, losses = objective(out, received, labels, condition, c)
        assert torch.isfinite(total)
        if condition == "x_only_bce":
            assert "z_channel" not in out
        else:
            assert out["z_click"].shape == out["z_channel"].shape == (4, 64)
            assert out["h_hat"].shape == (4, 2881)
        if condition == "heterogeneous_no_priors":
            assert all(float(losses[key].detach().cpu()) == 0. for key in ("loss_h_raw", "loss_compact_raw", "loss_band_raw"))
    sample = pd.DataFrame({"label": [1, 1, 0, 0], "probability": [.9, .2, .1, .8], "threshold": [.5] * 4, "ori_file_num": [1, 1, 1, 1]})
    assert set(metric(sample.label.to_numpy(), sample.probability.to_numpy(), sample.threshold.to_numpy())) >= {"auroc", "auprc", "f1"}
    source_test = pd.DataFrame({"label": [1, 0, 0], "ori_file_num": [7, 7, 8], "target_snr_db": [-12., np.nan, np.nan]})
    pos = source_test[(source_test.label == 1) & (source_test.target_snr_db >= -15) & (source_test.target_snr_db < -10)]
    neg = source_test[(source_test.label == 0) & source_test.ori_file_num.isin(pos.ori_file_num.unique())]
    assert neg.ori_file_num.tolist() == [7]
    print("[总体进度 2%] 四个模型、目标函数、CCA 不适用逻辑与同来源 SNR 分层检查通过", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("integration","full","analyze"), default="full"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--device", default="auto"); parser.add_argument("--max-epochs", type=int, default=25); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--bootstrap-runs", type=int, default=2000); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    requested_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    c = Config(output_dir=args.output_dir.resolve(), device=requested_device, max_epochs=args.max_epochs, patience=args.patience, bootstrap_runs=args.bootstrap_runs); device = base.resolve_device(c.device); paths(c)
    if args.mode == "analyze":
        predictions = pd.read_csv(paths(c)["predictions"] / "oof_predictions.csv"); summarize(c, predictions); plot(c); write_blueprint(c); return
    self_test(c, device); manifest, waves, contract = base.load_locked_cache(c); write_json(paths(c)["root"] / "run_config.json", {**asdict(c), "output_dir": str(c.output_dir), "conditions": CONDITIONS, "seeds": SEEDS, "prior_weights": PRIOR_WEIGHTS, "snr_bins": SNR_BINS, "data_hash": base.EXPECTED_DATA_HASH}); write_json(paths(c)["root"] / "data_contract.json", contract)
    seeds = (SEEDS[0],) if args.mode == "integration" else SEEDS; folds = (0,) if args.mode == "integration" else tuple(range(5)); total = len(CONDITIONS)*len(seeds)*len(folds); print(f"[总体进度 5%] 启动 {total} 个最终模型消融训练", flush=True); bar = tqdm(total=total, desc="总体进度：最终模型消融", unit="模型"); rows, frames = [], []
    for seed in seeds:
        for fold in folds:
            for condition in CONDITIONS:
                row, frame = fit(condition, seed, fold, waves, manifest, c, device, args.resume, bar); rows.append(row); frames.append(frame)
    bar.close(); metrics = pd.DataFrame(rows); metrics.to_csv(paths(c)["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig"); predictions = pd.concat(frames, ignore_index=True)
    if args.mode == "full":
        if len(predictions) != 84000 or predictions.duplicated(["condition","seed","sample_id"]).any(): raise RuntimeError("Unexpected OOF prediction count or duplicate condition-seed-sample key")
        summarize(c, predictions); plot(c); write_blueprint(c)
    write_json(paths(c)["root"] / "run_summary.json", {"models":len(metrics), "predictions":len(predictions), "conditions":CONDITIONS, "seeds":seeds, "folds":folds, "device":str(device)}); print("[总体进度 100%] 训练与分析完成", flush=True)

if __name__ == "__main__": main()
