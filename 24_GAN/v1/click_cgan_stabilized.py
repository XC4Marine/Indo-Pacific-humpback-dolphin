#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定第 1 折的条件 GAN 稳定化实验。

本脚本不改动既有五折结果。它固定既有随机抽样和五折中的第 1 折，依次执行：
1. 仅用训练集真实波形预训练 9 维声学特征回归器；
2. 冻结回归器；
3. 以较低判别器学习率和 1:2 的 D:G 更新比训练条件 GAN；
4. 每 250 轮保存检查点并在固定测试集评估。
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from click_cgan_v1 import (
    DEFAULT_DATA_ROOT,
    EXPERIMENT_DIR,
    FEATURE_DIMENSION,
    NOISE_DIMENSION,
    RANDOM_SEED,
    WAVEFORM_LENGTH,
    ClickDataset,
    Generator,
    correlation,
    discover_samples,
    dtw_distance,
    extract_feature_vector,
    fit_waveform_length,
    json_dump,
    load_sampled_dataset,
    normalize_waveform,
    output_paths,
    resolve_device,
    seed_everything,
    spectral_correlation,
    five_fold_splits,
)


FEATURE_NAMES = (
    "Duration (us)", "Spectral centroid (Hz)", "Fpeak (Hz)",
    "BW-3 dB (Hz)", "BW-6 dB (Hz)", "BW-10 dB (Hz)",
    "BW-12 dB (Hz)", "BW-18 dB (Hz)", "BW-21 dB (Hz)",
)


@dataclass(frozen=True)
class StabilizationConfig:
    """单折稳定化实验的全部固定配置。"""

    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = EXPERIMENT_DIR / "outputs_stabilization_fold_1"
    sample_count: int = 2_000
    fold_count: int = 5
    fold_number: int = 1
    epochs: int = 3_000
    checkpoint_interval: int = 250
    pretrain_epochs: int = 200
    batch_size: int = 64
    generator_learning_rate: float = 0.001
    discriminator_learning_rate: float = 0.0002
    feature_regressor_learning_rate: float = 0.001
    generator_updates_per_discriminator: int = 2
    feature_loss_weight: float = 1.0
    feature_loss_warmup_epochs: int = 0
    feature_loss_ramp_epochs: int = 0
    torch_threads: int = 4
    seed: int = RANDOM_SEED
    device: str = "auto"


class FeatureRegressor(nn.Module):
    """从 117 点波形预测标准化后 9 维声学特征的独立可微代理模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Linear(64 * 30, FEATURE_DIMENSION)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.output(self.convolution(waveform).flatten(start_dim=1))


class StabilizedDiscriminator(nn.Module):
    """输出 logits 的条件判别器，避免 Sigmoid+BCE 的数值饱和。"""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 30 + FEATURE_DIMENSION, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 1),
        )

    def forward(self, waveform: Tensor, features: Tensor) -> Tensor:
        extracted = self.convolution(waveform).flatten(start_dim=1)
        return self.classifier(torch.cat((extracted, features), dim=1))


def make_paths(config: StabilizationConfig) -> dict[str, Path]:
    """创建稳定化实验的所有输出位置。"""
    root = config.output_dir
    paths = {
        "root": root,
        "models": root / "models",
        "logs": root / "logs",
        "metrics": root / "metrics",
        "generated": root / "generated",
        "checkpoints": root / "checkpoints",
        "pretrain": root / "pretrain",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def pretrain_feature_regressor(config: StabilizationConfig, train_loader: DataLoader,
                               test_waveforms: np.ndarray, test_features: np.ndarray,
                               feature_std: np.ndarray, device: torch.device,
                               paths: dict[str, Path]) -> FeatureRegressor:
    """在真实训练波形上训练特征代理，并只在冻结后把其梯度用于生成器。"""
    regressor = FeatureRegressor().to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=config.feature_regressor_learning_rate,
                                 betas=(0.5, 0.999))
    mse = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    progress = tqdm(range(1, config.pretrain_epochs + 1), desc="总体进度 5%：预训练特征回归器", unit="轮")
    for epoch in progress:
        regressor.train()
        losses = []
        for waveforms, features in train_loader:
            waveforms, features = waveforms.to(device), features.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(regressor(waveforms), features)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_feature_mse": mean_loss})
        progress.set_postfix(feature_mse=f"{mean_loss:.4f}")

    regressor.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_waveforms[:, None, :]).to(device)
        target_tensor = torch.from_numpy(test_features).to(device)
        squared_errors = (regressor(test_tensor) - target_tensor).square()
        per_feature_mse = squared_errors.mean(dim=0).cpu().numpy()
        test_mse = float(per_feature_mse.mean())
    feature_calibration = [
        {
            "feature": name,
            "normalized_mse": float(feature_mse),
            "original_unit_rmse": float(np.sqrt(feature_mse) * feature_scale),
        }
        for name, feature_mse, feature_scale in zip(FEATURE_NAMES, per_feature_mse, feature_std)
    ]
    with (paths["pretrain"] / "feature_regressor_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_feature_mse"])
        writer.writeheader()
        writer.writerows(history)
    json_dump(paths["pretrain"] / "feature_regressor_metrics.json", {
        "final_train_feature_mse": history[-1]["train_feature_mse"],
        "test_feature_mse": test_mse,
        "per_feature_calibration": feature_calibration,
    })
    torch.save({"feature_regressor_state_dict": regressor.state_dict()},
               paths["pretrain"] / "feature_regressor.pt")
    for parameter in regressor.parameters():
        parameter.requires_grad_(False)
    return regressor


@torch.no_grad()
def evaluate(generator: Generator, test_waveforms: np.ndarray, test_features: np.ndarray,
             device: torch.device, paths: dict[str, Path], seed: int, label: str) -> dict[str, float | int]:
    """在固定测试折生成并计算三项波形/频谱指标。"""
    generator.eval()
    rng = np.random.default_rng(seed)
    generated_batches: list[np.ndarray] = []
    for start in range(0, len(test_features), 256):
        feature_batch = torch.from_numpy(test_features[start:start + 256]).to(device)
        noise = torch.from_numpy(rng.standard_normal((len(feature_batch), NOISE_DIMENSION)).astype(np.float32)).to(device)
        generated_batches.append(generator(noise, feature_batch).squeeze(1).cpu().numpy())
    generated = np.concatenate(generated_batches, axis=0)
    np.save(paths["generated"] / f"{label}_generated_waveforms.npy", generated)
    wave_correlations = [correlation(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    spectral_correlations = [spectral_correlation(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    dtw_distances = [dtw_distance(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    metrics: dict[str, float | int] = {
        "Spe-CORR": float(np.mean(spectral_correlations)),
        "Wave-CORR": float(np.mean(wave_correlations)),
        "DTW-Distance": float(np.mean(dtw_distances)),
        "test_sample_count": int(len(test_waveforms)),
    }
    json_dump(paths["metrics"] / f"{label}_metrics.json", metrics)
    return metrics


def save_history(path: Path, history: list[dict[str, float | int]]) -> None:
    """在检查点时写入当前训练记录，避免后台中断丢失诊断证据。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "lambda_feat", "d_adv", "g_adv", "g_feat", "g_total"])
        writer.writeheader()
        writer.writerows(history)


def feature_loss_weight_at_epoch(config: StabilizationConfig, epoch: int) -> float:
    """返回本轮实际使用的特征损失权重；默认值保持原来的固定权重。"""
    if epoch <= config.feature_loss_warmup_epochs:
        return 0.0
    if config.feature_loss_ramp_epochs <= 0:
        return config.feature_loss_weight
    completed_ramp = epoch - config.feature_loss_warmup_epochs
    return config.feature_loss_weight * min(1.0, completed_ramp / config.feature_loss_ramp_epochs)


def train_stabilized(config: StabilizationConfig, train_loader: DataLoader,
                     test_waveforms: np.ndarray, test_features: np.ndarray,
                     regressor: FeatureRegressor, device: torch.device,
                     paths: dict[str, Path]) -> dict[str, float | int]:
    """以低学习率判别器和 1:2 D:G 更新比训练固定第 1 折。"""
    generator = Generator().to(device)
    discriminator = StabilizedDiscriminator().to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    feature_loss = nn.MSELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_learning_rate,
                                            betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate,
                                                betas=(0.5, 0.999))
    history: list[dict[str, float | int]] = []
    checkpoint_metrics: list[dict[str, float | int]] = []
    progress = tqdm(range(1, config.epochs + 1), desc="总体进度：固定第 1 折稳定化训练", unit="轮")

    for epoch in progress:
        generator.train()
        discriminator.train()
        d_losses, g_adversarial_losses, g_feature_losses, g_total_losses = [], [], [], []
        lambda_feat = feature_loss_weight_at_epoch(config, epoch)
        for real_waveforms, features in train_loader:
            real_waveforms, features = real_waveforms.to(device), features.to(device)
            batch_size = real_waveforms.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_loss = adversarial_loss(discriminator(real_waveforms, features), real_targets)
            noise = torch.randn(batch_size, NOISE_DIMENSION, device=device)
            fake_waveforms = generator(noise, features)
            fake_loss = adversarial_loss(discriminator(fake_waveforms.detach(), features), fake_targets)
            discriminator_loss = (real_loss + fake_loss) * 0.5
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 判别器每更新一次，生成器连续更新两次以抑制判别器单边优势。
            for _ in range(config.generator_updates_per_discriminator):
                generator_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, NOISE_DIMENSION, device=device)
                generated_waveforms = generator(noise, features)
                generator_adversarial_loss = adversarial_loss(
                    discriminator(generated_waveforms, features), real_targets
                )
                generator_feature_loss = feature_loss(regressor(generated_waveforms), features)
                generator_total_loss = generator_adversarial_loss + lambda_feat * generator_feature_loss
                generator_total_loss.backward()
                generator_optimizer.step()
                g_adversarial_losses.append(float(generator_adversarial_loss.detach().cpu()))
                g_feature_losses.append(float(generator_feature_loss.detach().cpu()))
                g_total_losses.append(float(generator_total_loss.detach().cpu()))
            d_losses.append(float(discriminator_loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "lambda_feat": lambda_feat,
            "d_adv": float(np.mean(d_losses)),
            "g_adv": float(np.mean(g_adversarial_losses)),
            "g_feat": float(np.mean(g_feature_losses)),
            "g_total": float(np.mean(g_total_losses)),
        }
        history.append(row)
        progress.set_postfix(lambda_feat=f"{lambda_feat:.3f}", d_adv=f"{row['d_adv']:.3f}", g_adv=f"{row['g_adv']:.3f}",
                             g_feat=f"{row['g_feat']:.3f}")

        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            torch.save({"epoch": epoch, "generator_state_dict": generator.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict()},
                       paths["checkpoints"] / f"epoch_{epoch}.pt")
            # 所有检查点复用同一组噪声，确保指标差异来自模型而非随机生成样本。
            metrics = evaluate(generator, test_waveforms, test_features, device, paths,
                               config.seed + 1, f"epoch_{epoch}")
            metrics["epoch"] = epoch
            checkpoint_metrics.append(metrics)
            save_history(paths["logs"] / "training_history.csv", history)

    json_dump(paths["metrics"] / "checkpoint_metrics.json", checkpoint_metrics)
    return checkpoint_metrics[-1]


def parse_args() -> StabilizationConfig:
    """接收单折稳定化配置；默认值即计划中待验证的唯一候选。"""
    parser = argparse.ArgumentParser(description="固定第 1 折条件 GAN 稳定化")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs_stabilization_fold_1")
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--generator-learning-rate", type=float, default=0.001)
    parser.add_argument("--discriminator-learning-rate", type=float, default=0.0002)
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--feature-loss-warmup-epochs", type=int, default=0)
    parser.add_argument("--feature-loss-ramp-epochs", type=int, default=0)
    parser.add_argument("--fold-number", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    return StabilizationConfig(output_dir=arguments.output_dir, epochs=arguments.epochs,
                               pretrain_epochs=arguments.pretrain_epochs,
                               checkpoint_interval=arguments.checkpoint_interval,
                               generator_learning_rate=arguments.generator_learning_rate,
                               discriminator_learning_rate=arguments.discriminator_learning_rate,
                               feature_loss_weight=arguments.feature_loss_weight,
                               feature_loss_warmup_epochs=arguments.feature_loss_warmup_epochs,
                               feature_loss_ramp_epochs=arguments.feature_loss_ramp_epochs,
                               fold_number=arguments.fold_number,
                               torch_threads=arguments.torch_threads,
                               device=arguments.device)


def main() -> None:
    config = parse_args()
    torch.set_num_threads(config.torch_threads)
    torch.set_num_interop_threads(1)
    seed_everything(config.seed)
    paths = make_paths(config)
    device = resolve_device(config.device)
    json_dump(paths["root"] / "config.json", asdict(config))
    print(f"[总体进度 0%] 固定第 {config.fold_number} 折；预训练 {config.pretrain_epochs} 轮；GAN 训练 {config.epochs} 轮；PyTorch 线程数 {config.torch_threads}。")

    # 复用原脚本的数据抽样逻辑，保证本实验与已完成基线使用同一随机清单和五折划分。
    data_config = type("DataConfig", (), {"data_root": config.data_root, "sample_count": config.sample_count,
                                            "seed": config.seed})()
    waveforms, features, _ = load_sampled_dataset(data_config, paths)
    split_config = type("SplitConfig", (), {"seed": config.seed, "fold_count": config.fold_count})()
    fold_splits = list(five_fold_splits(waveforms, features, split_config))
    train_waveforms, train_features, test_waveforms, test_features, mean, std, train_indices, test_indices = fold_splits[config.fold_number - 1]
    np.savez(paths["root"] / "feature_normalizer.npz", mean=mean, std=std,
             train_indices=train_indices, test_indices=test_indices)
    train_loader = DataLoader(ClickDataset(train_waveforms, train_features), batch_size=config.batch_size,
                              shuffle=True, drop_last=False)
    np.save(paths["generated"] / "test_real_waveforms.npy", test_waveforms)
    print(f"[总体进度 2%] 第 {config.fold_number} 折训练集 {len(train_waveforms)} 条；测试集 {len(test_waveforms)} 条。")

    regressor = pretrain_feature_regressor(config, train_loader, test_waveforms, test_features, std, device, paths)
    final_metrics = train_stabilized(config, train_loader, test_waveforms, test_features, regressor, device, paths)
    print("[总体进度 100%] 单折稳定化实验完成。最终检查点：")
    for name, value in final_metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
