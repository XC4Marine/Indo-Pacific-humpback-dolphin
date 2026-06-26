#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
locate.py — 中华白海豚脉冲串（Pulse Train）定位与截取

复现 locate.m 的核心逻辑：
  1. 读取 PulseTrain.wav 与 PulseParameters.txt
  2. 基于 IPI 构造理想脉冲模板
  3. 计算 TEO 能量算子并归一化
  4. 全局互相关对齐，定位每个 click 的理论中心位置
  5. 截取 ±5ms 窗口 → 保存 .wav、波形图、TEO+自相关分析图

输出结构：
  09_Locate/
    ├── PulseTrain_001/
    │   ├── Click_001/
    │   │   ├── Pulse_001.wav
    │   │   ├── Waveform_001.png
    │   │   └── Analysis_001.png
    │   ├── Click_002/
    │   │   └── ...
    ...

运行方式：
  & "D:\Python_env\toothwhale\python.exe" locate.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pandas as pd
from scipy.signal import correlate
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # 非交互后端，允许无 GUI 渲染
import matplotlib.pyplot as plt

# ============================================================
# 全局常量
# ============================================================

# 输入数据根目录
DATA_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains")
# 输出根目录
OUTPUT_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate")

# 截取窗口半宽 (秒)
HALF_WINDOW_S = 5e-3

# 模板末尾余量 (秒)
TEMPLATE_MARGIN_S = 1e-3


# ============================================================
# 辅助函数
# ============================================================

def read_pulse_parameters(param_path: Path) -> pd.DataFrame:
    """
    读取 PulseParameters.txt，自动处理编码（UTF-8 / GBK）
    参数
    ----
    param_path : Path
        PulseParameters.txt 的完整路径
    返回
    ----
    df : DataFrame，包含所有参数列
    """
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            df = pd.read_csv(
                param_path,
                sep="\t",
                encoding=encoding,
                engine="python",
            )
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"无法解码文件: {param_path}")
    return df


def build_ideal_template(ipi_ms: np.ndarray, fs: int) -> tuple:
    """
    基于 IPI（脉冲间隔，毫秒）构造理想脉冲模板

    参数
    ----
    ipi_ms : 1-D ndarray
        IPI 序列，长度 = N-1（最后一条记录的 IPI 可能为空）
    fs : int
        采样率
    返回
    ----
    template : 1-D ndarray
        理想脉冲模板，非零位置 = 1
    pulse_rel_indices : 1-D ndarray
        每个脉冲在模板中的相对样本索引 (0-based)
    """
    ipi_samples = np.round(ipi_ms * fs / 1000).astype(int)
    pulse_rel_indices = np.concatenate(([0], np.cumsum(ipi_samples)))
    template_len = int(pulse_rel_indices[-1] + round(TEMPLATE_MARGIN_S * fs))
    template = np.zeros(template_len, dtype=np.float64)
    template[pulse_rel_indices] = 1.0
    return template, pulse_rel_indices


def compute_teo(signal: np.ndarray) -> tuple:
    """
    计算 Teager Energy Operator (TEO)
    TEO[x(n)] = x(n)^2 - x(n-1) * x(n+1)

    参数
    ----
    signal : 1-D ndarray
        输入音频信号
    返回
    ----
    teo : 1-D ndarray
        TEO 归一化结果 (已除以最大值)
    teo_raw : 1-D ndarray
        原始 TEO 值（未归一化）
    """
    teo_raw = np.zeros_like(signal)
    # TEO 核心公式：x(n)^2 - x(n-1)*x(n+1)
    teo_raw[1:-1] = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    # 归一化
    max_val = teo_raw.max()
    if max_val > 0:
        teo = teo_raw / max_val
    else:
        teo = teo_raw.copy()
    return teo, teo_raw


def align_template(teo_norm: np.ndarray, template: np.ndarray) -> int:
    """
    全局互相关对齐：TEO 能量曲线 vs 理想脉冲模板

    参数
    ----
    teo_norm : 1-D ndarray
        归一化 TEO 能量曲线
    template : 1-D ndarray
        理想脉冲模板
    返回
    ----
    best_lag : int
        最佳滞后样本数 (使互相关最大的偏移)
    """
    # scipy.signal.correlate 输出长度 = len(a) + len(b) - 1
    corr = correlate(teo_norm, template, mode="full")
    max_idx = np.argmax(corr)
    # lag = max_idx - (len(template) - 1)，即模板首位对准信号的位置
    best_lag = max_idx - (len(template) - 1)
    return best_lag


def extract_segment(
    signal: np.ndarray,
    center_idx: int,
    half_win: int,
) -> np.ndarray:
    """
    以 center_idx 为中心截取 ±half_win 样本，边界外补零

    参数
    ----
    signal : 1-D ndarray
        完整信号
    center_idx : int
        中心样本索引 (0-based)
    half_win : int
        截取窗口半宽（样本数）
    返回
    ----
    segment : 1-D ndarray
        截取后的片段
    """
    N = len(signal)
    start = center_idx - half_win
    end = center_idx + half_win
    seg_len = 2 * half_win + 1

    if start < 0:
        pad_left = -start
        actual_start = 0
    else:
        pad_left = 0
        actual_start = start

    if end > N:
        pad_right = end - N
        actual_end = N
    else:
        pad_right = 0
        actual_end = end

    middle = signal[actual_start:actual_end]
    segment = np.concatenate(
        [np.zeros(pad_left, dtype=signal.dtype), middle, np.zeros(pad_right, dtype=signal.dtype)]
    )
    # 确保精确长度
    if len(segment) < seg_len:
        segment = np.pad(segment, (0, seg_len - len(segment)))
    elif len(segment) > seg_len:
        segment = segment[:seg_len]
    return segment


def save_waveform_figure(
    segment: np.ndarray,
    fs: int,
    pulse_label: str,
    save_path: Path,
):
    """
    绘制并保存单个 click 的波形放大图 (灰曲线 + 红色中心虚线)

    参数
    ----
    segment : 1-D ndarray
        截取的 click 片段
    fs : int
        采样率
    pulse_label : str
        图片标题中的脉冲编号标签
    save_path : Path
        输出 .png 路径
    """
    half_win = len(segment) // 2
    t_ms = (np.arange(-half_win, half_win + 1) / fs) * 1000  # 毫秒

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_ms, segment, linewidth=0.8, color="#333333")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_title(f"{pulse_label} — Waveform")
    ax.set_xlabel("Relative Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3)
    # 对称 Y 轴范围
    y_max = np.max(np.abs(segment)) * 1.1
    if y_max == 0:
        y_max = 1.0
    ax.set_ylim(-y_max, y_max)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_analysis_figure(
    segment: np.ndarray,
    fs: int,
    pulse_label: str,
    save_path: Path,
):
    """
    绘制双面板分析图：TEO 能量算子 + 自相关函数

    参数
    ----
    segment : 1-D ndarray
        截取的 click 片段
    fs : int
        采样率
    pulse_label : str
        图片标题中的脉冲编号标签
    save_path : Path
        输出 .png 路径
    """
    half_win = len(segment) // 2
    t_ms = (np.arange(-half_win, half_win + 1) / fs) * 1000

    # 计算 TEO
    teo_raw = np.zeros_like(segment)
    teo_raw[1:-1] = segment[1:-1] ** 2 - segment[:-2] * segment[2:]

    # 计算自相关 (coefficient 模式)
    acf = np.correlate(segment, segment, mode="full")
    # 归一化：除以零滞后值
    zero_lag_val = acf[len(acf) // 2]
    if zero_lag_val != 0:
        acf = acf / zero_lag_val
    lags = np.arange(len(acf)) - (len(acf) // 2)
    lags_ms = (lags / fs) * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5))

    # 子图1：TEO 时域
    ax1.plot(t_ms, teo_raw, color="#D95319", linewidth=1.0)
    ax1.set_title(f"{pulse_label} — Teager Energy Operator")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("TEO Energy")
    ax1.set_xlim(-5, 5)
    ax1.grid(True, alpha=0.3)

    # 子图2：自相关
    ax2.plot(lags_ms, acf, color="#0072BD", linewidth=1.0)
    ax2.set_title(f"{pulse_label} — Autocorrelation")
    ax2.set_xlabel("Delay (ms)")
    ax2.set_ylabel("Correlation Coeff")
    ax2.set_xlim(-5, 5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def process_single_pulse_train(train_dir: Path) -> bool:
    """
    处理单个 PulseTrain 文件夹：定位所有 click 并保存

    参数
    ----
    train_dir : Path
        PulseTrain_XXX 文件夹路径
    返回
    ----
    success : bool
        处理是否成功
    """
    train_name = train_dir.name
    wav_path = train_dir / "PulseTrain.wav"
    param_path = train_dir / "PulseParameters.txt"

    if not wav_path.exists() or not param_path.exists():
        print(f"[跳过] {train_name}: 缺少 PulseTrain.wav 或 PulseParameters.txt")
        return False

    # ---- 1. 读数据 ----
    try:
        signal, fs = sf.read(str(wav_path))
    except Exception as e:
        print(f"[错误] {train_name}: 无法读取 wav — {e}")
        return False

    if signal.ndim > 1:
        signal = signal[:, 0]  # 取第一通道
    signal = signal.astype(np.float64)
    N = len(signal)

    try:
        df = read_pulse_parameters(param_path)
    except Exception as e:
        print(f"[错误] {train_name}: 无法读取参数文件 — {e}")
        return False

    # 提取 IPI 列（跳过最后一个 NaN）
    ipi_col = df["IPI(ms)"].values.astype(np.float64)
    ipi_valid = ipi_col[~np.isnan(ipi_col)]
    if len(ipi_valid) == 0:
        print(f"[跳过] {train_name}: IPI 列无有效值")
        return False

    # ---- 2. 构造理想模板 ----
    template, rel_indices = build_ideal_template(ipi_valid, fs)

    # ---- 3. 计算 TEO ----
    teo_norm, _ = compute_teo(signal)

    # ---- 4. 全局互相关对齐 ----
    best_lag = align_template(teo_norm, template)
    theoretical_samples = rel_indices + best_lag

    # 过滤超出信号范围的索引
    valid_mask = (theoretical_samples >= 0) & (theoretical_samples < N)
    theoretical_samples = theoretical_samples[valid_mask]
    if len(theoretical_samples) == 0:
        print(f"[警告] {train_name}: 对齐后无有效 click 位置")
        return False

    # ---- 5. 创建输出目录 ----
    out_train_dir = OUTPUT_DIR / train_name
    out_train_dir.mkdir(parents=True, exist_ok=True)

    half_win = round(HALF_WINDOW_S * fs)

    # ---- 6. 遍历每个 click ----
    num_pulses = len(theoretical_samples)
    for k in tqdm(
        range(num_pulses),
        desc=f"  {train_name}",
        leave=False,
        unit="click",
    ):
        center_idx = int(round(theoretical_samples[k]))
        click_name = f"Click_{k + 1:03d}"
        click_dir = out_train_dir / click_name
        click_dir.mkdir(parents=True, exist_ok=True)

        # 截取片段
        segment = extract_segment(signal, center_idx, half_win)

        # 保存 .wav
        wav_out = click_dir / f"Pulse_{k + 1:03d}.wav"
        sf.write(str(wav_out), segment, fs)

        # 保存波形图
        pulse_label = f"{train_name} / {click_name}"
        waveform_out = click_dir / f"Waveform_{k + 1:03d}.png"
        save_waveform_figure(segment, fs, pulse_label, waveform_out)

        # 保存分析图
        analysis_out = click_dir / f"Analysis_{k + 1:03d}.png"
        save_analysis_figure(segment, fs, pulse_label, analysis_out)

    return True


def main():
    """
    主入口：遍历 00_Data/ClickTrains 下所有 PulseTrain_XXX，逐个处理
    """
    print("=" * 60)
    print("Pulse Train Locate — MATLAB → Python")
    print(f"数据目录 : {DATA_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print("=" * 60)

    # 收集所有 PulseTrain 文件夹
    train_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("PulseTrain_")]
    )
    print(f"发现 {len(train_dirs)} 个 PulseTrain 文件夹\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for train_dir in tqdm(train_dirs, desc="总体进度", unit="train"):
        ok = process_single_pulse_train(train_dir)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n处理完成: 成功 {success_count}, 失败/跳过 {fail_count}")


if __name__ == "__main__":
    main()