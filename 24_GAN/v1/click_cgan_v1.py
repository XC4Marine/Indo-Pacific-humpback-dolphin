#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 v1：声学特征引导的条件 DCGAN，用于生成中华白海豚 click 信号。

运行示例：
    python click_cgan_v1.py

脚本从 ClickTrains/PulseTrain_* 内随机抽取 2,000 条 Pulse_*.wav，并使用五折
交叉验证。每一折的条件特征标准化参数只由该折训练部分拟合。PulseParameters.txt
中的 Duration(μs) 是条件向量第一维；其余频率特征由每个 WAV 的功率谱直接计算。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "00_Data" / "ClickTrains"
EXPERIMENT_DIR = PROJECT_ROOT / "24_GAN" / "v1"

# 2 次 stride=2、kernel_size=5 的转置卷积：27 -> 55 -> 117。
WAVEFORM_LENGTH = 117
FEATURE_DIMENSION = 9
NOISE_DIMENSION = 116
RANDOM_SEED = 20260917


@dataclass(frozen=True)
class Config:
    """实验中固定或可由命令行覆盖的超参数。"""

    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = EXPERIMENT_DIR / "outputs_five_fold_feature_loss"
    sample_count: int = 2_000
    fold_count: int = 5
    epochs: int = 3_000
    batch_size: int = 64
    learning_rate: float = 0.001
    beta1: float = 0.5
    feature_loss_weight: float = 1.0
    seed: int = RANDOM_SEED
    device: str = "auto"


def json_dump(path: Path, value: object) -> None:
    """以 UTF-8 写入结构化实验产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def seed_everything(seed: int) -> None:
    """固定抽样、网络初始化和 DataLoader 顺序。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    """解析 auto、cpu 或 cuda 设备参数。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 未检测到可用 CUDA 设备。")
    return device


def output_paths(config: Config) -> dict[str, Path]:
    """创建本实验的唯一输出位置。"""
    paths = {
        "root": config.output_dir,
        "folds": config.output_dir / "folds",
        "metrics": config.output_dir / "metrics",
        "logs": config.output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fold_paths(root: Path, fold_number: int) -> dict[str, Path]:
    """创建单个交叉验证折次的独立输出目录。"""
    fold_root = root / "folds" / f"fold_{fold_number}"
    paths = {
        "root": fold_root,
        "models": fold_root / "models",
        "metrics": fold_root / "metrics",
        "logs": fold_root / "logs",
        "generated": fold_root / "generated",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_parameter_table(path: Path) -> dict[int, tuple[float, float]]:
    """读取一个 PulseParameters.txt，返回“脉冲编号 -> (持续时间微秒, 峰值频率Hz)”。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"参数文件没有表头：{path}")
        required = {"No.Pulse", "Duration(μs)", "Fpeak(kHz)"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(f"参数文件缺少字段 {sorted(missing)}：{path}")
        parameters: dict[int, tuple[float, float]] = {}
        for row in reader:
            number_text = row["No.Pulse"].strip()
            duration_text = row["Duration(μs)"].strip()
            peak_frequency_text = row["Fpeak(kHz)"].strip()
            if number_text and duration_text and peak_frequency_text:
                parameters[int(number_text)] = (float(duration_text), float(peak_frequency_text) * 1_000.0)
    return parameters


def pulse_number(path: Path) -> int:
    """从 Pulse_1.wav 或 Pulse_001.wav 提取其整数编号。"""
    match = re.fullmatch(r"Pulse_(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"无法从文件名解析脉冲编号：{path}")
    return int(match.group(1))


def discover_samples(data_root: Path) -> list[tuple[Path, float, float]]:
    """发现具有有效 Duration(μs) 和 Fpeak(kHz) 记录的 click 音频。"""
    records: list[tuple[Path, float, float]] = []
    directories = sorted(data_root.glob("PulseTrain_*"))
    if not directories:
        raise FileNotFoundError(f"未找到 PulseTrain_* 目录：{data_root}")
    for directory in tqdm(directories, desc="总体进度 5%：读取参数表", unit="目录"):
        table_path = directory / "PulseParameters.txt"
        parameters = parse_parameter_table(table_path)
        for waveform_path in sorted(directory.glob("Pulse_*.wav")):
            number = pulse_number(waveform_path)
            if number in parameters:
                duration_us, peak_frequency_hz = parameters[number]
                records.append((waveform_path, duration_us, peak_frequency_hz))
    return records


def fit_waveform_length(waveform: np.ndarray) -> np.ndarray:
    """将切分 click 居中补零或居中裁剪至网络要求的 117 点。"""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == WAVEFORM_LENGTH:
        return waveform
    if waveform.size > WAVEFORM_LENGTH:
        start = (waveform.size - WAVEFORM_LENGTH) // 2
        return waveform[start:start + WAVEFORM_LENGTH]
    result = np.zeros(WAVEFORM_LENGTH, dtype=np.float32)
    start = (WAVEFORM_LENGTH - waveform.size) // 2
    result[start:start + waveform.size] = waveform
    return result


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """将单条波形峰值归一化到 [-1, 1]，以匹配生成器的 Tanh 输出。"""
    peak = float(np.max(np.abs(waveform)))
    if peak == 0.0:
        raise RuntimeError("发现全零 click 波形，无法执行峰值归一化。")
    return waveform / peak


def bandwidth_from_power_spectrum(power: np.ndarray, frequencies: np.ndarray, drop_db: int) -> float:
    """计算相对谱峰下降 drop_db 后，仍高于阈值频点的频带宽度（Hz）。"""
    threshold = float(np.max(power)) * 10.0 ** (-drop_db / 10.0)
    indices = np.flatnonzero(power >= threshold)
    if indices.size == 0:
        raise RuntimeError(f"无法计算 -{drop_db} dB 带宽。")
    return float(frequencies[indices[-1]] - frequencies[indices[0]])


def extract_feature_vector(waveform: np.ndarray, sample_rate: int, duration_us: float,
                           peak_frequency_hz: float) -> np.ndarray:
    """构建 V=[tau, fc, fpeak, BW-3, BW-6, BW-10, BW-12, BW-18, BW-21]。"""
    # 零填充仅提高频率栅格分辨率，不改变原 click 的物理时域长度。
    spectrum = np.fft.rfft(waveform, n=2048)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(2048, d=1.0 / sample_rate)
    total_power = float(np.sum(power))
    if total_power == 0.0:
        raise RuntimeError("全零波形不能计算频谱特征。")
    centroid_hz = float(np.sum(frequencies * power) / total_power)
    bandwidths = [bandwidth_from_power_spectrum(power, frequencies, level)
                  for level in (3, 6, 10, 12, 18, 21)]
    return np.asarray([duration_us, centroid_hz, peak_frequency_hz, *bandwidths], dtype=np.float32)


def load_sampled_dataset(config: Config, paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """随机抽取 2,000 条音频并完成波形、条件向量的预处理。"""
    records = discover_samples(config.data_root)
    if len(records) < config.sample_count:
        raise RuntimeError(f"有效音频只有 {len(records)} 条，少于要求的 {config.sample_count} 条。")
    rng = np.random.default_rng(config.seed)
    selected_indices = rng.choice(len(records), size=config.sample_count, replace=False)
    selected = [records[int(index)] for index in selected_indices]

    waveforms = np.empty((config.sample_count, WAVEFORM_LENGTH), dtype=np.float32)
    features = np.empty((config.sample_count, FEATURE_DIMENSION), dtype=np.float32)
    manifest_rows: list[dict[str, object]] = []
    expected_sample_rate: int | None = None
    for index, (waveform_path, duration_us, peak_frequency_hz) in enumerate(tqdm(selected, desc="总体进度 10%：提取波形与特征", unit="条")):
        waveform, sample_rate = sf.read(waveform_path, dtype="float32", always_2d=False)
        if waveform.ndim != 1:
            raise RuntimeError(f"只支持单声道 click 音频：{waveform_path}")
        if expected_sample_rate is None:
            expected_sample_rate = int(sample_rate)
        if sample_rate != expected_sample_rate:
            raise RuntimeError(f"采样率不一致：{waveform_path} 的采样率为 {sample_rate} Hz。")
        fitted = fit_waveform_length(waveform)
        features[index] = extract_feature_vector(fitted, sample_rate, duration_us, peak_frequency_hz)
        waveforms[index] = normalize_waveform(fitted)
        manifest_rows.append({"sample_index": index, "wav_path": str(waveform_path),
                              "duration_us": duration_us, "peak_frequency_hz": peak_frequency_hz,
                              "sample_rate": sample_rate})

    manifest_path = paths["root"] / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return waveforms, features, [row["wav_path"] for row in manifest_rows]


def five_fold_splits(waveforms: np.ndarray, features: np.ndarray, config: Config) -> Iterable[tuple[np.ndarray, ...]]:
    """生成五折数据；每一折只用训练部分的统计量标准化条件特征。"""
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(waveforms))
    folds = np.array_split(indices, config.fold_count)
    for fold_index, test_indices in enumerate(folds):
        train_indices = np.concatenate([fold for index, fold in enumerate(folds) if index != fold_index])
        feature_mean = features[train_indices].mean(axis=0)
        feature_std = features[train_indices].std(axis=0)
        if np.any(feature_std == 0.0):
            raise RuntimeError(f"第 {fold_index + 1} 折训练集存在方差为零的条件特征，无法进行标准化。")
        train_features = (features[train_indices] - feature_mean) / feature_std
        test_features = (features[test_indices] - feature_mean) / feature_std
        yield (waveforms[train_indices], train_features.astype(np.float32),
               waveforms[test_indices], test_features.astype(np.float32),
               feature_mean.astype(np.float32), feature_std.astype(np.float32), train_indices, test_indices)


class ClickDataset(Dataset[tuple[Tensor, Tensor]]):
    """返回一维时域波形与其标准化后的 9 维条件向量。"""

    def __init__(self, waveforms: np.ndarray, features: np.ndarray) -> None:
        self.waveforms = torch.from_numpy(waveforms[:, None, :])
        self.features = torch.from_numpy(features)

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.waveforms[index], self.features[index]


class Generator(nn.Module):
    """116 维噪声与 9 维声学条件拼接后的两层反卷积生成器。"""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(NOISE_DIMENSION + FEATURE_DIMENSION, 128 * 27),
            nn.BatchNorm1d(128 * 27),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 1, kernel_size=5, stride=2),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor, features: Tensor) -> Tensor:
        combined = torch.cat((noise, features), dim=1)
        projected = self.project(combined).view(-1, 128, 27)
        return self.upsample(projected)


class Discriminator(nn.Module):
    """联合判断 117 点时域形态和对应 9 维条件特征是否匹配。"""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            # padding=2 时，117 -> 59；随后 59 -> 30，均为近似二分下采样。
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 30 + FEATURE_DIMENSION, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(9, 1),
            nn.Sigmoid(),
        )
        # 只以真实波形监督该分支，使其成为“波形 -> 标准化物理特征”的连续代理。
        self.feature_regressor = nn.Linear(64 * 30, FEATURE_DIMENSION)

    def forward(self, waveform: Tensor, features: Tensor,
                return_predicted_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        extracted = self.convolution(waveform).flatten(start_dim=1)
        authenticity = self.classifier(torch.cat((extracted, features), dim=1))
        if return_predicted_features:
            return authenticity, self.feature_regressor(extracted)
        return authenticity


def train(config: Config, train_loader: DataLoader[tuple[Tensor, Tensor]], device: torch.device,
          paths: dict[str, Path], fold_number: int, overall_progress: tqdm) -> Generator:
    """交替训练判别器和生成器，完整运行该折的 3,000 轮。"""
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    adversarial_loss = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate,
                                            betas=(config.beta1, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate,
                                                betas=(config.beta1, 0.999))
    feature_loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()
        generator_losses, discriminator_losses = [], []
        generator_adversarial_losses, generator_feature_losses = [], []
        discriminator_feature_losses = []
        for real_waveforms, features in train_loader:
            real_waveforms, features = real_waveforms.to(device), features.to(device)
            batch_size = real_waveforms.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # 先训练 D：真实配对为真，生成波形与相同 V 的配对为假。
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_probability, predicted_real_features = discriminator(
                real_waveforms, features, return_predicted_features=True
            )
            real_loss = adversarial_loss(real_probability, real_targets)
            noise = torch.randn(batch_size, NOISE_DIMENSION, device=device)
            fake_waveforms = generator(noise, features)
            fake_loss = adversarial_loss(discriminator(fake_waveforms.detach(), features), fake_targets)
            discriminator_adversarial_loss = (real_loss + fake_loss) * 0.5
            discriminator_feature_loss = feature_loss_function(predicted_real_features, features)
            discriminator_loss = discriminator_adversarial_loss + config.feature_loss_weight * discriminator_feature_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 再训练 G：既欺骗 D，也要求特征回归头从生成波形恢复目标 V。
            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, NOISE_DIMENSION, device=device)
            generated_waveforms = generator(noise, features)
            generated_probability, predicted_generated_features = discriminator(
                generated_waveforms, features, return_predicted_features=True
            )
            generator_adversarial_loss = adversarial_loss(generated_probability, real_targets)
            generator_feature_loss = feature_loss_function(predicted_generated_features, features)
            generator_loss = generator_adversarial_loss + config.feature_loss_weight * generator_feature_loss
            generator_loss.backward()
            generator_optimizer.step()
            generator_losses.append(float(generator_loss.detach().cpu()))
            discriminator_losses.append(float(discriminator_loss.detach().cpu()))
            generator_adversarial_losses.append(float(generator_adversarial_loss.detach().cpu()))
            generator_feature_losses.append(float(generator_feature_loss.detach().cpu()))
            discriminator_feature_losses.append(float(discriminator_feature_loss.detach().cpu()))

        mean_generator_loss = float(np.mean(generator_losses))
        mean_discriminator_loss = float(np.mean(discriminator_losses))
        history.append({"epoch": epoch, "generator_loss": mean_generator_loss,
                        "generator_adversarial_loss": float(np.mean(generator_adversarial_losses)),
                        "generator_feature_loss": float(np.mean(generator_feature_losses)),
                        "discriminator_loss": mean_discriminator_loss,
                        "discriminator_feature_loss": float(np.mean(discriminator_feature_losses))})
        overall_progress.update(1)
        overall_progress.set_postfix(fold=f"{fold_number}/5", epoch=epoch,
                                     g_adv=f"{np.mean(generator_adversarial_losses):.3f}",
                                     g_feat=f"{np.mean(generator_feature_losses):.3f}",
                                     d_loss=f"{mean_discriminator_loss:.3f}")

    with (paths["logs"] / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "generator_loss", "generator_adversarial_loss",
                                                     "generator_feature_loss", "discriminator_loss",
                                                     "discriminator_feature_loss"])
        writer.writeheader()
        writer.writerows(history)
    torch.save({"epoch": config.epochs, "generator_state_dict": generator.state_dict()},
               paths["models"] / "final_generator.pt")
    generator.eval()
    return generator


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个一维向量的 Pearson 相关系数。"""
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator == 0.0 else float(np.dot(first, second) / denominator)


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个同长度 click 波形的经典 DTW 累积距离，并按路径长度归一化。"""
    rows, columns = len(first), len(second)
    cost = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            local_cost = abs(float(first[row - 1]) - float(second[column - 1]))
            cost[row, column] = local_cost + min(cost[row - 1, column], cost[row, column - 1], cost[row - 1, column - 1])
    return float(cost[rows, columns] / (rows + columns))


def spectral_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """以 2048 点功率谱计算 Spe-CORR。"""
    first_power = np.abs(np.fft.rfft(first, n=2048)) ** 2
    second_power = np.abs(np.fft.rfft(second, n=2048)) ** 2
    return correlation(first_power, second_power)


@torch.no_grad()
def evaluate(generator: Generator, test_waveforms: np.ndarray, test_features: np.ndarray,
             device: torch.device, paths: dict[str, Path], seed: int) -> dict[str, float]:
    """冻结生成器，在该折测试条件向量上生成并计算三项相似度指标。"""
    rng = np.random.default_rng(seed)
    generated_batches: list[np.ndarray] = []
    batch_size = 256
    for start in tqdm(range(0, len(test_features), batch_size), desc="总体进度 95%：测试集生成", unit="批"):
        feature_batch = torch.from_numpy(test_features[start:start + batch_size]).to(device)
        noise = torch.from_numpy(rng.standard_normal((len(feature_batch), NOISE_DIMENSION)).astype(np.float32)).to(device)
        generated_batches.append(generator(noise, feature_batch).squeeze(1).cpu().numpy())
    generated = np.concatenate(generated_batches, axis=0)
    np.save(paths["generated"] / "test_generated_waveforms.npy", generated)
    np.save(paths["generated"] / "test_real_waveforms.npy", test_waveforms)

    wave_correlations = [correlation(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    spectral_correlations = [spectral_correlation(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    dtw_distances = [dtw_distance(real, synthetic) for real, synthetic in zip(test_waveforms, generated)]
    metrics = {
        "Spe-CORR": float(np.mean(spectral_correlations)),
        "Wave-CORR": float(np.mean(wave_correlations)),
        "DTW-Distance": float(np.mean(dtw_distances)),
        "test_sample_count": int(len(test_waveforms)),
    }
    json_dump(paths["metrics"] / "test_metrics.json", metrics)
    return metrics


def parse_args() -> Config:
    """读取命令行参数；默认值为五折验证且每折完整训练 3,000 轮。"""
    parser = argparse.ArgumentParser(description="实验 v1：特征引导 click 条件 DCGAN")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs_five_fold_feature_loss")
    parser.add_argument("--epochs", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    return Config(data_root=arguments.data_root, output_dir=arguments.output_dir, epochs=arguments.epochs,
                  batch_size=arguments.batch_size, feature_loss_weight=arguments.feature_loss_weight,
                  device=arguments.device)


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    paths = output_paths(config)
    device = resolve_device(config.device)
    json_dump(paths["root"] / "config.json", asdict(config))
    print(f"[总体进度 0%] 使用设备：{device}；抽样数量：{config.sample_count}；五折验证；每折训练轮数：{config.epochs}；λ_feat={config.feature_loss_weight}")

    waveforms, features, _ = load_sampled_dataset(config, paths)
    all_metrics: list[dict[str, float]] = []
    overall_progress = tqdm(total=config.fold_count * config.epochs, desc="总体进度：五折 GAN 训练", unit="轮")
    for fold_number, split in enumerate(five_fold_splits(waveforms, features, config), start=1):
        train_waveforms, train_features, test_waveforms, test_features, mean, std, train_indices, test_indices = split
        current_paths = fold_paths(paths["root"], fold_number)
        np.savez(current_paths["root"] / "feature_normalizer.npz", mean=mean, std=std,
                 train_indices=train_indices, test_indices=test_indices)
        train_dataset = ClickDataset(train_waveforms, train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
        print(f"[总体进度] 第 {fold_number}/5 折：训练集 {len(train_dataset)} 条；测试集 {len(test_waveforms)} 条。")
        generator = train(config, train_loader, device, current_paths, fold_number, overall_progress)
        metrics = evaluate(generator, test_waveforms, test_features, device, current_paths, config.seed + fold_number)
        metrics["fold"] = fold_number
        all_metrics.append(metrics)
    overall_progress.close()

    metric_names = ("Spe-CORR", "Wave-CORR", "DTW-Distance")
    summary = {name: {"mean": float(np.mean([result[name] for result in all_metrics])),
                      "std": float(np.std([result[name] for result in all_metrics], ddof=1))}
               for name in metric_names}
    summary["fold_metrics"] = all_metrics
    json_dump(paths["metrics"] / "five_fold_metrics.json", summary)
    print("[总体进度 100%] 五折实验完成。指标均值：")
    for name in metric_names:
        print(f"  {name}: {summary[name]['mean']:.6f} ± {summary[name]['std']:.6f}")


if __name__ == "__main__":
    main()
