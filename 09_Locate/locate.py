#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
locate.py — 中华白海豚脉冲串（Pulse Train）批量定位误差报告

复现 locate.m 的核心逻辑：
  1. 读取 PulseTrain.wav 与 PulseParameters.txt
  2. 基于 IPI 构造理想脉冲模板
  3. 计算 TEO 能量算子并归一化
  4. 在完整模板边界约束下进行全局互相关对齐
  5. 用原始 Pulse_N.wav 的精确位置评估定位误差

输出结构：
  09_Locate/Locate_Batch_Report/
    ├── pulse_location_pulse_report.csv
    ├── pulse_location_train_summary.csv
    ├── pulse_location_batch_summary.png
    ├── pulse_location_batch_summary.txt
    └── pulse_precise_positions.txt

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
# 批量 Report 输出目录
REPORT_DIR = OUTPUT_DIR / "Locate_Batch_Report"

# 文本与图表中展示的最高误差 PulseTrain 数量
TOP_REPORT_COUNT = 30

# 用户确认从精确位置清单中排除的高误差 PulseTrain；不删除原始文件。
EXCLUDED_PULSE_TRAINS = frozenset(
    {
        "PulseTrain_513",
        "PulseTrain_511",
        "PulseTrain_691",
        "PulseTrain_039",
        "PulseTrain_067",
        "PulseTrain_290",
        "PulseTrain_286",
    }
)

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


def align_template(
    teo_norm: np.ndarray,
    template: np.ndarray,
    pulse_rel_indices: np.ndarray,
) -> dict:
    """
    全局互相关对齐：TEO 能量曲线 vs 理想脉冲模板

    参数
    ----
    teo_norm : 1-D ndarray
        归一化 TEO 能量曲线
    template : 1-D ndarray
        理想脉冲模板
    返回包含完整模板约束 lag、无约束 lag 和相关得分的字典。
    """
    # scipy.signal.correlate 输出长度 = len(a) + len(b) - 1
    corr = correlate(teo_norm, template, mode="full")
    lag_offset = len(template) - 1

    unconstrained_idx = int(np.argmax(corr))
    unconstrained_lag = unconstrained_idx - lag_offset
    unconstrained_score = float(corr[unconstrained_idx])

    # 0-based 索引：所有 Pulse 模板点必须满足 0 <= rel + lag < len(signal)。
    min_valid_lag = int(-pulse_rel_indices[0])
    max_valid_lag = int(len(teo_norm) - 1 - pulse_rel_indices[-1])
    if min_valid_lag > max_valid_lag:
        raise ValueError("模板跨度超过 PulseTrain.wav，无法完整对齐")

    valid_start = min_valid_lag + lag_offset
    valid_stop = max_valid_lag + lag_offset + 1
    valid_corr = corr[valid_start:valid_stop]
    if valid_corr.size == 0:
        raise ValueError("没有可让全部 Pulse 模板点落入音频范围的 lag")

    constrained_local_idx = int(np.argmax(valid_corr))
    constrained_idx = valid_start + constrained_local_idx
    best_lag = constrained_idx - lag_offset

    return {
        "best_lag": int(best_lag),
        "best_score": float(corr[constrained_idx]),
        "unconstrained_lag": int(unconstrained_lag),
        "unconstrained_score": unconstrained_score,
        "partial_alignment_prevented": not (
            min_valid_lag <= unconstrained_lag <= max_valid_lag
        ),
    }


def find_exact_waveform_starts(
    signal: np.ndarray,
    reference: np.ndarray,
    anchor_count: int = 6,
) -> np.ndarray:
    """返回 reference 在 signal 中逐样本完全一致的所有 0-based 起点。"""
    signal_length = len(signal)
    reference_length = len(reference)
    if reference_length == 0 or reference_length > signal_length:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(np.abs(reference))[::-1]
    anchors = order[: min(anchor_count, reference_length)]
    first_anchor = int(anchors[0])
    candidates = np.flatnonzero(signal == reference[first_anchor]) - first_anchor
    candidates = candidates[
        (candidates >= 0) & (candidates + reference_length <= signal_length)
    ]

    for anchor in anchors[1:]:
        if candidates.size == 0:
            return candidates
        anchor = int(anchor)
        candidates = candidates[
            signal[candidates + anchor] == reference[anchor]
        ]

    exact_starts = [
        int(start)
        for start in candidates
        if np.array_equal(signal[start : start + reference_length], reference)
    ]
    return np.asarray(exact_starts, dtype=np.int64)


def find_local_ncc_reference(
    signal: np.ndarray,
    reference: np.ndarray,
    expected_start: int,
    search_radius: int,
    minimum_score: float = 0.99,
) -> tuple:
    """在有限邻域内用归一化互相关匹配量化格式不同的参考波形。"""
    reference_length = len(reference)
    max_start = len(signal) - reference_length
    if reference_length == 0 or max_start < 0:
        return None, np.nan

    start_min = max(0, int(expected_start) - int(search_radius))
    start_max = min(max_start, int(expected_start) + int(search_radius))
    if start_min > start_max:
        return None, np.nan

    search_signal = signal[start_min : start_max + reference_length]
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= np.finfo(np.float64).eps:
        return None, np.nan

    cross = correlate(search_signal, reference, mode="valid", method="fft")
    squared = search_signal**2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    window_energy = cumulative[reference_length:] - cumulative[:-reference_length]
    denominator = np.sqrt(np.maximum(window_energy, 0.0)) * reference_norm
    scores = np.divide(
        cross,
        denominator,
        out=np.full_like(cross, -np.inf, dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )

    best_local = int(np.argmax(scores))
    best_score = float(scores[best_local])
    if not np.isfinite(best_score) or best_score < minimum_score:
        return None, best_score
    return start_min + best_local, best_score


def calculate_error_metrics(errors_ms: np.ndarray) -> dict:
    """计算一个 PulseTrain 的定位误差统计。"""
    errors_ms = np.asarray(errors_ms, dtype=np.float64)
    errors_ms = errors_ms[np.isfinite(errors_ms)]
    if errors_ms.size == 0:
        return {
            "mean_bias_ms": np.nan,
            "mae_ms": np.nan,
            "median_abs_error_ms": np.nan,
            "rmse_ms": np.nan,
            "error_std_ms": np.nan,
            "p95_abs_error_ms": np.nan,
            "max_abs_error_ms": np.nan,
        }

    abs_errors = np.abs(errors_ms)
    return {
        "mean_bias_ms": float(np.mean(errors_ms)),
        "mae_ms": float(np.mean(abs_errors)),
        "median_abs_error_ms": float(np.median(abs_errors)),
        "rmse_ms": float(np.sqrt(np.mean(errors_ms**2))),
        "error_std_ms": float(np.std(errors_ms)),
        "p95_abs_error_ms": float(np.percentile(abs_errors, 95)),
        "max_abs_error_ms": float(np.max(abs_errors)),
    }


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


def process_single_pulse_train(train_dir: Path) -> dict:
    """处理一个 PulseTrain，返回逐 Train 汇总与逐 Pulse 误差记录。"""
    train_name = train_dir.name
    wav_path = train_dir / "PulseTrain.wav"
    param_path = train_dir / "PulseParameters.txt"

    if not wav_path.exists() or not param_path.exists():
        raise FileNotFoundError("缺少 PulseTrain.wav 或 PulseParameters.txt")

    # ---- 1. 读数据 ----
    signal, fs = sf.read(str(wav_path))

    if signal.ndim > 1:
        signal = signal[:, 0]  # 取第一通道
    signal = signal.astype(np.float64)
    N = len(signal)

    df = read_pulse_parameters(param_path)
    required_columns = {"No.Pulse", "IPI(ms)"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"参数表缺少字段: {sorted(missing_columns)}")

    pulse_numbers = pd.to_numeric(df["No.Pulse"], errors="coerce").to_numpy()
    if pulse_numbers.size == 0 or np.any(~np.isfinite(pulse_numbers)):
        raise ValueError("No.Pulse 列为空或包含非数字值")
    pulse_numbers = np.rint(pulse_numbers).astype(int)

    # 第 k 行 IPI 表示 Pulse k 到 k+1 的间隔，因此只使用前 N-1 项。
    expected_ipi_count = max(0, len(pulse_numbers) - 1)
    ipi_col = pd.to_numeric(df["IPI(ms)"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ipi_sequence = ipi_col[:expected_ipi_count]
    if len(ipi_sequence) != expected_ipi_count or np.any(~np.isfinite(ipi_sequence)):
        raise ValueError("IPI(ms) 前 N-1 项存在缺失或非数字值")

    # ---- 2. 构造理想模板 ----
    template, rel_indices = build_ideal_template(ipi_sequence, fs)

    # ---- 3. 计算 TEO ----
    teo_norm, teo_raw = compute_teo(signal)
    if not np.any(np.isfinite(teo_norm)) or np.max(teo_raw) <= 0:
        raise ValueError("PulseTrain.wav 没有有效的正 TEO 峰")

    # ---- 4. 完整模板约束下的全局互相关对齐 ----
    alignment = align_template(teo_norm, template, rel_indices)
    best_lag = alignment["best_lag"]
    theoretical_samples = rel_indices + best_lag
    if len(theoretical_samples) != len(pulse_numbers):
        raise ValueError("理论位置数量与参数表 Pulse 数量不一致")
    if np.any(theoretical_samples < 0) or np.any(theoretical_samples >= N):
        raise RuntimeError("完整模板约束失效：仍存在越界理论位置")

    # ---- 5. 旧版切片与逐 Click 图片输出已停用并保留为注释 ----
    # out_train_dir = OUTPUT_DIR / train_name
    # out_train_dir.mkdir(parents=True, exist_ok=True)
    # half_win = round(HALF_WINDOW_S * fs)
    # for k, center_idx in enumerate(theoretical_samples):
    #     click_name = f"Click_{k + 1:03d}"
    #     click_dir = out_train_dir / click_name
    #     click_dir.mkdir(parents=True, exist_ok=True)
    #     segment = extract_segment(signal, int(center_idx), half_win)
    #     wav_out = click_dir / f"Pulse_{k + 1:03d}.wav"
    #     sf.write(str(wav_out), segment, fs)
    #     pulse_label = f"{train_name} / {click_name}"
    #     waveform_out = click_dir / f"Waveform_{k + 1:03d}.png"
    #     save_waveform_figure(segment, fs, pulse_label, waveform_out)
    #     analysis_out = click_dir / f"Analysis_{k + 1:03d}.png"
    #     save_analysis_figure(segment, fs, pulse_label, analysis_out)

    # ---- 6. 用原始 Pulse_N.wav 的唯一精确位置生成误差记录 ----
    pulse_rows = []
    valid_errors_ms = []
    valid_error_pulse_numbers = []
    worst_pulse_number = np.nan
    worst_abs_error_ms = -np.inf
    exact_match_count = 0
    ncc_match_count = 0
    missing_reference_count = 0
    reference_failure_count = 0

    for index, pulse_number in enumerate(pulse_numbers):
        reference_name = f"Pulse_{pulse_number}.wav"
        reference_path = train_dir / reference_name
        algorithm_sample = int(theoretical_samples[index])
        reference_start = np.nan
        reference_peak = np.nan
        signed_error_samples = np.nan
        signed_error_ms = np.nan
        reference_match_method = "none"
        reference_match_score = np.nan
        match_status = "matched"

        if not reference_path.exists():
            match_status = "missing_reference_file"
        else:
            reference, reference_fs = sf.read(str(reference_path))
            if reference.ndim > 1:
                reference = reference[:, 0]
            reference = reference.astype(np.float64)

            if reference_fs != fs:
                match_status = "sample_rate_mismatch"
            elif len(reference) == 0 or len(reference) > N:
                match_status = "invalid_reference_length"
            else:
                reference_teo_norm, reference_teo = compute_teo(reference)
                del reference_teo_norm
                if np.max(reference_teo) <= 0:
                    match_status = "invalid_reference_teo"
                else:
                    exact_starts = find_exact_waveform_starts(signal, reference)
                    if len(exact_starts) == 0:
                        neighbor_gaps = []
                        if index > 0:
                            neighbor_gaps.append(
                                int(rel_indices[index] - rel_indices[index - 1])
                            )
                        if index + 1 < len(rel_indices):
                            neighbor_gaps.append(
                                int(rel_indices[index + 1] - rel_indices[index])
                            )
                        if neighbor_gaps:
                            search_radius = int(0.45 * min(neighbor_gaps))
                        else:
                            search_radius = int(round(0.05 * fs))
                        search_radius = max(
                            int(round(0.002 * fs)),
                            min(int(round(0.05 * fs)), search_radius),
                        )

                        ncc_start, ncc_score = find_local_ncc_reference(
                            signal,
                            reference,
                            algorithm_sample,
                            search_radius,
                        )
                        reference_match_score = ncc_score
                        if ncc_start is None:
                            match_status = "no_reference_match"
                        else:
                            local_peak = int(np.argmax(reference_teo))
                            reference_start = int(ncc_start)
                            reference_peak = reference_start + local_peak
                            signed_error_samples = algorithm_sample - reference_peak
                            signed_error_ms = signed_error_samples / fs * 1000.0
                            valid_errors_ms.append(signed_error_ms)
                            valid_error_pulse_numbers.append(int(pulse_number))
                            reference_match_method = "local_ncc"
                            match_status = "matched_ncc"
                            ncc_match_count += 1
                    elif len(exact_starts) > 1:
                        match_status = "multiple_exact_matches"
                    else:
                        local_peak = int(np.argmax(reference_teo))
                        reference_start = int(exact_starts[0])
                        reference_peak = reference_start + local_peak
                        signed_error_samples = algorithm_sample - reference_peak
                        signed_error_ms = signed_error_samples / fs * 1000.0
                        valid_errors_ms.append(signed_error_ms)
                        valid_error_pulse_numbers.append(int(pulse_number))
                        reference_match_method = "exact"
                        reference_match_score = 1.0
                        exact_match_count += 1

        if np.isfinite(signed_error_ms) and abs(signed_error_ms) > worst_abs_error_ms:
            worst_abs_error_ms = abs(signed_error_ms)
            worst_pulse_number = int(pulse_number)

        if match_status == "missing_reference_file":
            missing_reference_count += 1
        elif match_status not in {"matched", "matched_ncc"}:
            reference_failure_count += 1

        pulse_rows.append(
            {
                "pulse_train": train_name,
                "pulse_number": int(pulse_number),
                "sample_rate_hz": int(fs),
                "algorithm_sample_1based": algorithm_sample + 1,
                "algorithm_time_s": algorithm_sample / fs,
                "reference_start_sample_1based": (
                    reference_start + 1 if np.isfinite(reference_start) else np.nan
                ),
                "reference_peak_sample_1based": (
                    reference_peak + 1 if np.isfinite(reference_peak) else np.nan
                ),
                "reference_time_s": (
                    reference_peak / fs if np.isfinite(reference_peak) else np.nan
                ),
                "signed_error_samples": signed_error_samples,
                "abs_error_samples": abs(signed_error_samples),
                "signed_error_ms": signed_error_ms,
                "abs_error_ms": abs(signed_error_ms),
                "match_status": match_status,
                "reference_match_method": reference_match_method,
                "reference_match_score": reference_match_score,
                "reference_file": reference_name,
            }
        )

    metrics = calculate_error_metrics(np.asarray(valid_errors_ms))
    matched_count = len(valid_errors_ms)
    available_reference_count = len(pulse_numbers) - missing_reference_count
    if len(valid_errors_ms) >= 2:
        error_slope_ms_per_pulse = float(
            np.polyfit(valid_error_pulse_numbers, valid_errors_ms, 1)[0]
        )
    else:
        error_slope_ms_per_pulse = np.nan
    systematic_offset = bool(
        np.isfinite(metrics["mae_ms"])
        and metrics["mae_ms"] > 0
        and abs(metrics["mean_bias_ms"]) / metrics["mae_ms"] >= 0.9
    )
    reasons = []
    if alignment["partial_alignment_prevented"]:
        reasons.append("partial_alignment_prevented")
    if reference_failure_count:
        reasons.append(f"reference_match_failures={reference_failure_count}")

    summary = {
        "pulse_train": train_name,
        "status": "ok" if reference_failure_count == 0 else "reference_incomplete",
        "pulse_count": int(len(pulse_numbers)),
        "available_reference_count": int(available_reference_count),
        "ignored_missing_reference_count": int(missing_reference_count),
        "matched_count": int(matched_count),
        "exact_match_count": int(exact_match_count),
        "ncc_match_count": int(ncc_match_count),
        "match_rate": (
            matched_count / available_reference_count
            if available_reference_count > 0
            else np.nan
        ),
        "sample_rate_hz": int(fs),
        "unconstrained_lag": alignment["unconstrained_lag"],
        "constrained_lag": alignment["best_lag"],
        "unconstrained_score": alignment["unconstrained_score"],
        "constrained_score": alignment["best_score"],
        "partial_alignment_prevented": alignment["partial_alignment_prevented"],
        **metrics,
        "error_slope_ms_per_pulse": error_slope_ms_per_pulse,
        "systematic_offset": systematic_offset,
        "worst_pulse_number": worst_pulse_number,
        "problem_reason": ";".join(reasons),
    }
    return {"summary": summary, "pulses": pulse_rows}


def processing_error_summary(train_name: str, error: Exception) -> dict:
    """构造无法处理的 PulseTrain 汇总记录。"""
    return {
        "pulse_train": train_name,
        "status": "processing_error",
        "pulse_count": np.nan,
        "available_reference_count": np.nan,
        "ignored_missing_reference_count": 0,
        "matched_count": 0,
        "exact_match_count": 0,
        "ncc_match_count": 0,
        "match_rate": 0.0,
        "sample_rate_hz": np.nan,
        "unconstrained_lag": np.nan,
        "constrained_lag": np.nan,
        "unconstrained_score": np.nan,
        "constrained_score": np.nan,
        "partial_alignment_prevented": False,
        **calculate_error_metrics(np.asarray([])),
        "error_slope_ms_per_pulse": np.nan,
        "systematic_offset": False,
        "worst_pulse_number": np.nan,
        "problem_reason": f"{type(error).__name__}: {error}",
    }


def build_voyage_mapping() -> dict:
    """从 09_Locate/Voyage_Ori_* 目录结构建立 PulseTrain 到 Voyage 的映射。"""
    voyage_mapping = {}
    for voyage_dir in sorted(OUTPUT_DIR.glob("Voyage_Ori_*")):
        if not voyage_dir.is_dir():
            continue
        for train_dir in voyage_dir.iterdir():
            if not train_dir.is_dir() or not train_dir.name.startswith("PulseTrain_"):
                continue
            if train_dir.name in voyage_mapping:
                raise ValueError(
                    f"{train_dir.name} 同时出现在 {voyage_mapping[train_dir.name]} "
                    f"和 {voyage_dir.name}"
                )
            voyage_mapping[train_dir.name] = voyage_dir.name
    return voyage_mapping


def write_precise_positions(pulse_df: pd.DataFrame) -> dict:
    """写出排除问题 PulseTrain 后的逐 Pulse 精确位置 TSV 文本。"""
    output_path = REPORT_DIR / "pulse_precise_positions.txt"
    voyage_mapping = build_voyage_mapping()

    excluded_mask = pulse_df["pulse_train"].isin(EXCLUDED_PULSE_TRAINS)
    valid_status_mask = pulse_df["match_status"].isin({"matched", "matched_ncc"})
    finite_position_mask = np.isfinite(
        pd.to_numeric(pulse_df["reference_peak_sample_1based"], errors="coerce")
    )
    keep_mask = ~excluded_mask & valid_status_mask & finite_position_mask

    kept = pulse_df.loc[keep_mask].copy()
    kept["voyage_ori"] = kept["pulse_train"].map(voyage_mapping)
    missing_voyage = sorted(kept.loc[kept["voyage_ori"].isna(), "pulse_train"].unique())
    if missing_voyage:
        raise ValueError(f"以下 PulseTrain 缺少 Voyage_Ori 映射: {missing_voyage}")

    kept["train_number"] = pd.to_numeric(
        kept["pulse_train"].str.replace("PulseTrain_", "", regex=False),
        errors="raise",
    )
    kept = kept.sort_values(["train_number", "pulse_number"])

    excluded_pulse_count = int(excluded_mask.sum())
    unavailable_position_count = int((~excluded_mask & ~keep_mask).sum())
    exact_count = int((kept["reference_match_method"] == "exact").sum())
    ncc_count = int((kept["reference_match_method"] == "local_ncc").sum())

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("# Pulse 精确位置清单\n")
        handle.write("# 位置定义: 原始 Pulse_N.wav 的 TEO 峰在 PulseTrain.wav 中的 1-based 样本位置。\n")
        handle.write(
            "# local_ncc 表示原始 WAV 与 PulseTrain 的量化格式不同，位置由局部归一化互相关确定。\n"
        )
        handle.write(f"# 保留 Pulse 数: {len(kept)}\n")
        handle.write(f"# 逐样本精确匹配: {exact_count}\n")
        handle.write(f"# 局部 NCC 匹配: {ncc_count}\n")
        handle.write(f"# 被排除 PulseTrain 的 Pulse 数: {excluded_pulse_count}\n")
        handle.write(f"# 无可用参考位置而跳过的 Pulse 数: {unavailable_position_count}\n")
        handle.write(
            "# 排除的 PulseTrain: " + ", ".join(sorted(EXCLUDED_PULSE_TRAINS)) + "\n"
        )
        handle.write(
            "Voyage_Ori\tPulseTrain\tPulseNumber\tPositionSample_1based\t"
            "PositionTime_s\tPositionTime_ms\tSampleRate_Hz\tReferenceFile\t"
            "MatchMethod\tMatchScore\n"
        )

        for _, row in kept.iterrows():
            position_sample = int(row["reference_peak_sample_1based"])
            position_time_s = float(row["reference_time_s"])
            match_score = float(row["reference_match_score"])
            handle.write(
                f"{row['voyage_ori']}\t{row['pulse_train']}\t"
                f"{int(row['pulse_number'])}\t{position_sample}\t"
                f"{position_time_s:.12f}\t{position_time_s * 1000:.9f}\t"
                f"{int(row['sample_rate_hz'])}\t{row['reference_file']}\t"
                f"{row['reference_match_method']}\t{match_score:.12f}\n"
            )

    return {
        "path": output_path,
        "kept_count": int(len(kept)),
        "excluded_pulse_count": excluded_pulse_count,
        "unavailable_position_count": unavailable_position_count,
        "exact_count": exact_count,
        "ncc_count": ncc_count,
    }


def write_batch_report(train_summaries: list, pulse_rows: list) -> dict:
    """写出批量 CSV、文本摘要和汇总图。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_summaries)
    pulse_df = pd.DataFrame(pulse_rows)

    finite_mae = pd.to_numeric(train_df["mae_ms"], errors="coerce")
    finite_values = finite_mae[np.isfinite(finite_mae)].to_numpy()
    if finite_values.size >= 3:
        median_mae = float(np.median(finite_values))
        mad_mae = float(np.median(np.abs(finite_values - median_mae)))
        robust_sigma = 1.4826 * mad_mae
        outlier_threshold_ms = (
            median_mae + 3.0 * robust_sigma
            if robust_sigma > 0
            else float(np.nextafter(median_mae, np.inf))
        )
    else:
        median_mae = float(np.median(finite_values)) if finite_values.size else np.nan
        mad_mae = np.nan
        outlier_threshold_ms = np.inf

    train_df["mae_rank_desc"] = finite_mae.rank(
        method="min", ascending=False, na_option="bottom"
    )
    train_df["error_outlier"] = finite_mae > outlier_threshold_ms
    train_df["needs_review"] = (
        (train_df["status"] != "ok") | train_df["error_outlier"]
    )

    review_reasons = []
    for _, row in train_df.iterrows():
        reasons = [item for item in str(row["problem_reason"]).split(";") if item]
        if bool(row["error_outlier"]):
            reasons.append("high_mae_outlier")
            if bool(row["systematic_offset"]):
                reasons.append("systematic_offset")
        review_reasons.append(";".join(reasons))
    train_df["review_reason"] = review_reasons

    train_df = train_df.sort_values(
        ["needs_review", "mae_ms"], ascending=[False, False], na_position="last"
    )
    if not pulse_df.empty:
        pulse_df = pulse_df.sort_values(["pulse_train", "pulse_number"])

    pulse_csv = REPORT_DIR / "pulse_location_pulse_report.csv"
    train_csv = REPORT_DIR / "pulse_location_train_summary.csv"
    summary_png = REPORT_DIR / "pulse_location_batch_summary.png"
    summary_txt = REPORT_DIR / "pulse_location_batch_summary.txt"

    pulse_df.to_csv(pulse_csv, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    precise_positions = write_precise_positions(pulse_df)

    valid_train_df = train_df[np.isfinite(pd.to_numeric(train_df["mae_ms"], errors="coerce"))]
    processing_errors = train_df[train_df["status"] == "processing_error"]
    incomplete = train_df[train_df["status"] == "reference_incomplete"]
    ignored_missing = train_df[
        pd.to_numeric(train_df["ignored_missing_reference_count"], errors="coerce") > 0
    ]
    prevented = train_df[train_df["partial_alignment_prevented"] == True]  # noqa: E712
    outliers = train_df[train_df["error_outlier"] == True]  # noqa: E712
    systematic_outliers = outliers[outliers["systematic_offset"] == True]  # noqa: E712
    total_exact_matches = int(pd.to_numeric(train_df["exact_match_count"]).sum())
    total_ncc_matches = int(pd.to_numeric(train_df["ncc_match_count"]).sum())
    total_ignored_missing = int(
        pd.to_numeric(train_df["ignored_missing_reference_count"]).sum()
    )

    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("PulseTrain 批量定位误差 Report\n")
        handle.write("================================\n")
        handle.write(f"PulseTrain 总数: {len(train_df)}\n")
        handle.write(f"成功生成误差统计: {len(valid_train_df)}\n")
        handle.write(f"处理失败: {len(processing_errors)}\n")
        handle.write(f"参考匹配不完整: {len(incomplete)}\n")
        handle.write(f"忽略的缺失参考 WAV: {total_ignored_missing}\n")
        handle.write(f"逐样本精确参考匹配: {total_exact_matches}\n")
        handle.write(f"局部 NCC 后备参考匹配: {total_ncc_matches}\n")
        handle.write(f"阻止旧逻辑部分模板越界: {len(prevented)}\n")
        handle.write(f"MAE 中位数: {median_mae:.9f} ms\n")
        handle.write(f"MAE MAD: {mad_mae:.9f} ms\n")
        handle.write(f"高 MAE 离群阈值: {outlier_threshold_ms:.9f} ms\n")
        handle.write(f"高 MAE 离群 PulseTrain: {len(outliers)}\n\n")
        handle.write(f"其中系统性固定偏移: {len(systematic_outliers)}\n")
        handle.write(
            "判读: systematic_offset 表示正负误差没有相互抵消，通常说明模板稳定选择了主峰之后或之前的另一组重复 TEO 结构，而不是 IPI 累积漂移。\n\n"
        )

        handle.write(f"按 MAE 排序的前 {TOP_REPORT_COUNT} 个 PulseTrain\n")
        handle.write("--------------------------------\n")
        top = valid_train_df.nlargest(TOP_REPORT_COUNT, "mae_ms")
        for _, row in top.iterrows():
            handle.write(
                f"{row['pulse_train']}: MAE={row['mae_ms']:.9f} ms, "
                f"Bias={row['mean_bias_ms']:.9f} ms, "
                f"Std={row['error_std_ms']:.9f} ms, "
                f"Max={row['max_abs_error_ms']:.9f} ms, "
                f"Matched={int(row['matched_count'])}/"
                f"{int(row['available_reference_count'])} available, "
                f"Exact={int(row['exact_match_count'])}, "
                f"NCC={int(row['ncc_match_count'])}, "
                f"Review={bool(row['needs_review'])}, "
                f"Reason={row['review_reason']}\n"
            )

        if len(outliers):
            handle.write("\n全部高 MAE 离群 PulseTrain\n")
            handle.write("--------------------------------\n")
            for _, row in outliers.sort_values("mae_ms", ascending=False).iterrows():
                handle.write(
                    f"{row['pulse_train']}: MAE={row['mae_ms']:.9f} ms, "
                    f"Bias={row['mean_bias_ms']:.9f} ms, "
                    f"Std={row['error_std_ms']:.9f} ms, "
                    f"Max={row['max_abs_error_ms']:.9f} ms\n"
                )

        if len(prevented):
            handle.write("\n旧逻辑会发生部分模板越界的 PulseTrain（已被修正）\n")
            handle.write("--------------------------------\n")
            for _, row in prevented.sort_values("pulse_train").iterrows():
                handle.write(
                    f"{row['pulse_train']}: old_lag={int(row['unconstrained_lag'])}, "
                    f"new_lag={int(row['constrained_lag'])}, "
                    f"MAE={row['mae_ms']:.9f} ms\n"
                )

        if len(incomplete):
            handle.write("\n参考文件不完整的 PulseTrain\n")
            handle.write("--------------------------------\n")
            for _, row in incomplete.iterrows():
                handle.write(
                    f"{row['pulse_train']}: Matched={int(row['matched_count'])}/"
                    f"{int(row['available_reference_count'])} available, "
                    f"Reason={row['problem_reason']}\n"
                )

        if len(ignored_missing):
            handle.write("\n已忽略缺失参考 WAV 的 PulseTrain\n")
            handle.write("--------------------------------\n")
            for _, row in ignored_missing.iterrows():
                handle.write(
                    f"{row['pulse_train']}: IgnoredMissing="
                    f"{int(row['ignored_missing_reference_count'])}, "
                    f"Matched={int(row['matched_count'])}/"
                    f"{int(row['available_reference_count'])} available\n"
                )

        if len(processing_errors):
            handle.write("\n处理失败的 PulseTrain\n")
            handle.write("--------------------------------\n")
            for _, row in processing_errors.iterrows():
                handle.write(f"{row['pulse_train']}: {row['problem_reason']}\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    top_plot = valid_train_df.nlargest(TOP_REPORT_COUNT, "mae_ms").sort_values(
        "mae_ms"
    )
    colors = ["#D95319" if flag else "#0072BD" for flag in top_plot["needs_review"]]
    axes[0, 0].barh(top_plot["pulse_train"], top_plot["mae_ms"], color=colors)
    axes[0, 0].set_title(f"Top {TOP_REPORT_COUNT} PulseTrains by MAE")
    axes[0, 0].set_xlabel("MAE (ms)")
    axes[0, 0].grid(axis="x", alpha=0.3)

    prevented_mask = valid_train_df["partial_alignment_prevented"].astype(bool)
    axes[0, 1].scatter(
        valid_train_df.loc[~prevented_mask, "pulse_count"],
        valid_train_df.loc[~prevented_mask, "mae_ms"],
        s=16,
        alpha=0.65,
        label="Unconstrained lag already valid",
    )
    axes[0, 1].scatter(
        valid_train_df.loc[prevented_mask, "pulse_count"],
        valid_train_df.loc[prevented_mask, "mae_ms"],
        s=26,
        color="#D95319",
        label="Partial alignment prevented",
    )
    axes[0, 1].set_title("MAE vs. Pulse Count")
    axes[0, 1].set_xlabel("Pulse count")
    axes[0, 1].set_ylabel("MAE (ms)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(valid_train_df["mae_ms"], bins=40, color="#ED7D31", alpha=0.8)
    if np.isfinite(outlier_threshold_ms):
        axes[1, 0].axvline(
            outlier_threshold_ms,
            color="red",
            linestyle="--",
            label=f"Outlier threshold {outlier_threshold_ms:.6f} ms",
        )
        axes[1, 0].legend()
    axes[1, 0].set_title("PulseTrain MAE Distribution")
    axes[1, 0].set_xlabel("MAE (ms)")
    axes[1, 0].set_ylabel("PulseTrain count")
    axes[1, 0].grid(alpha=0.3)

    max_plot = valid_train_df.nlargest(
        TOP_REPORT_COUNT, "max_abs_error_ms"
    ).sort_values("max_abs_error_ms")
    axes[1, 1].barh(
        max_plot["pulse_train"], max_plot["max_abs_error_ms"], color="#70AD47"
    )
    axes[1, 1].set_title(f"Top {TOP_REPORT_COUNT} PulseTrains by Max Absolute Error")
    axes[1, 1].set_xlabel("Max absolute error (ms)")
    axes[1, 1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Pulse Location Batch Report | Trains {len(train_df)} | "
        f"Review {int(train_df['needs_review'].sum())} | "
        f"Partial-alignments prevented {len(prevented)}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(summary_png, dpi=180)
    plt.close(fig)

    return {
        "pulse_csv": pulse_csv,
        "train_csv": train_csv,
        "summary_png": summary_png,
        "summary_txt": summary_txt,
        "precise_positions": precise_positions,
        "train_df": train_df,
    }


def main():
    """
    主入口：遍历所有 PulseTrain，生成一份批量误差 Report。
    """
    print("=" * 60)
    print("Pulse Train Locate — Batch Accuracy Report")
    print(f"数据目录 : {DATA_DIR}")
    print(f"Report目录: {REPORT_DIR}")
    print("=" * 60)

    # 收集所有 PulseTrain 文件夹
    train_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("PulseTrain_")]
    )
    print(f"发现 {len(train_dirs)} 个 PulseTrain 文件夹\n")

    train_summaries = []
    pulse_rows = []

    for train_dir in tqdm(
        train_dirs,
        desc="总体进度",
        unit="train",
        mininterval=0.5,
    ):
        try:
            result = process_single_pulse_train(train_dir)
            train_summaries.append(result["summary"])
            pulse_rows.extend(result["pulses"])
        except Exception as error:
            train_summaries.append(processing_error_summary(train_dir.name, error))
            tqdm.write(f"[错误] {train_dir.name}: {type(error).__name__}: {error}")

    report = write_batch_report(train_summaries, pulse_rows)
    failed_count = sum(row["status"] == "processing_error" for row in train_summaries)
    print(f"\n处理完成: PulseTrain {len(train_summaries)}, 失败 {failed_count}")
    print(f"逐 Pulse 报告 : {report['pulse_csv']}")
    print(f"逐 Train 汇总  : {report['train_csv']}")
    print(f"汇总图         : {report['summary_png']}")
    print(f"文本摘要       : {report['summary_txt']}")
    print(f"精确位置清单   : {report['precise_positions']['path']}")
    print(
        f"位置清单统计   : 保留 {report['precise_positions']['kept_count']}, "
        f"排除 {report['precise_positions']['excluded_pulse_count']}, "
        f"无参考跳过 {report['precise_positions']['unavailable_position_count']}"
    )


if __name__ == "__main__":
    main()
