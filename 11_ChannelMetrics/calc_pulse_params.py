#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calc_pulse_params.py — 脉冲多径信道参数计算

对每个 PulseTrain 组内所有 click 波形做 Z-Score 归一化后，
计算每一径（直达 + 多径 1/2/3）的三个信道指标：
  1. 时延 (Delay, μs)
  2. 幅度比例 (AmpRatio): RMS(直达) / RMS(该径)
  3. 相干系数 (Coherence): 该径窗与 click 同位置窗的 Pearson 相关系数
"""

import sys, gc, struct
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ============================================================
# 路径常量
# ============================================================
BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
PULSE_TIMES_PATH = BASE_DIR / "10_AutoCorrMultipath" / "all_pulse_times.txt"
OUTPUT_DIR = BASE_DIR / "11_ChannelMetrics"
OUTPUT_PATH = OUTPUT_DIR / "pulse_params.txt"

# ============================================================
# 参数
# ============================================================
WINDOW_US = 50.0          # ±50μs 窗宽 (总共100μs)
DIRECT_TIME_MS = 5.0       # 直达脉冲时间 (ms)
MAX_LAG_MS = 5.0            # 最大多径延迟 (超过5ms的不纳入考虑)
MAX_LAG_S = 4.99e-3         # 严格排除 >= 5.0ms 的边界伪影


def parse_pulse_times(path):
    """解析 all_pulse_times.txt，返回 [(train, click, [mp1, mp2, mp3]), ...]。

    每个元素: (pulse_train_name, click_name, [float_or_NaN, ...])
    直达时间固定 5.0ms，因此不单独存储。
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            train = parts[0]
            click = parts[1]
            mp_times = []
            for val in parts[3:]:
                try:
                    mp_times.append(float(val))
                except (ValueError, TypeError):
                    pass
            records.append((train, click, mp_times))
    return records


def read_wav(path):
    """读取 wav 文件，返回 (signal_float64, sample_rate)。纯 numpy + struct 实现。"""
    with open(str(path), "rb") as f:
        riff, size, wave = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"不是有效的 WAV 文件: {path}")
        bits_per_sample = 16
        fs_val = 576000
        nchannels = 1
        data_bytes = b""
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            chunk_data = f.read(chunk_size)
            if chunk_id == b"fmt ":
                audio_format, nchannels, fs_val, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", chunk_data[:16]
                )
            elif chunk_id == b"data":
                data_bytes = chunk_data
        if bits_per_sample == 16:
            dtype = np.int16
        elif bits_per_sample == 24:
            dtype = np.int32
        elif bits_per_sample == 32:
            dtype = np.int32
        else:
            raise ValueError(f"不支持的位深: {bits_per_sample}")
        n_samples = len(data_bytes) // (bits_per_sample // 8)
        signal = np.frombuffer(data_bytes, dtype=dtype, count=n_samples)
        if nchannels > 1:
            signal = signal.reshape(-1, nchannels)[:, 0]
        return signal.astype(np.float64), fs_val


def rms_in_window(signal, center_s, fs, half_us=WINDOW_US):
    """计算 signal 在 center_s 前后 ±half_us μs 窗内的 RMS。

    signal: 归一化后的 1D 数组
    center_s: 窗中心时间 (秒)
    fs: 采样率
    """
    n = len(signal)
    half_s = half_us * 1e-6
    start_s = center_s - half_s
    end_s = center_s + half_s
    idx_start = max(0, int(round(start_s * fs)))
    idx_end = min(n, int(round(end_s * fs)))
    if idx_end <= idx_start:
        return 0.0
    segment = signal[idx_start:idx_end]
    return float(np.sqrt(np.mean(segment ** 2)))


def pearson_r(signal_a, signal_b):
    """计算两个等长一维信号的 Pearson 相关系数。"""
    if len(signal_a) < 2:
        return 0.0
    ma, mb = np.mean(signal_a), np.mean(signal_b)
    sa = signal_a - ma
    sb = signal_b - mb
    num = np.sum(sa * sb)
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(num / den) if den > 0 else 0.0


def extract_window(signal, center_s, fs, half_us=WINDOW_US):
    """从 signal 中提取 center_s ± half_us μs 的片段。"""
    n = len(signal)
    half_s = half_us * 1e-6
    idx_start = max(0, int(round((center_s - half_s) * fs)))
    idx_end = min(n, int(round((center_s + half_s) * fs)))
    return signal[idx_start:idx_end], idx_start, idx_end


def main():
    print("=" * 60)
    print("11_ChannelMetrics — 脉冲多径信道参数计算 (窗宽±25μs)")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. 读取脉冲定位
    # -------------------------------------------------------
    print("\n[1/4] 读取脉冲定位...")
    records = parse_pulse_times(PULSE_TIMES_PATH)
    print(f"  共 {len(records)} 条脉冲定位记录")

    # 按 PulseTrain 分组
    train_groups = {}
    for train, click, mp_times in records:
        train_groups.setdefault(train, []).append((click, mp_times))
    print(f"  共 {len(train_groups)} 个 PulseTrain")

    # -------------------------------------------------------
    # 2. 读取所有波形 & Z-Score 归一化（按组）
    # -------------------------------------------------------
    print("\n[2/4] 读取波形 & Z-Score 归一化...")

    # 收集缺失路径（该 click 存在但 wav 不存在）
    missing = []
    # 存储: { (train, click): (signal_z, fs, mp_times) }
    all_data = {}

    for train, entries in tqdm(train_groups.items(), desc="PulseTrain", unit="组"):
        click_signals = []          # (click_name, signal, fs, mp_times)
        for click, mp_times in entries:
            wav_path = LOCATE_DIR / train / click / f"Pulse_{click.split('_')[1]}.wav"
            if not wav_path.exists():
                missing.append(str(wav_path))
                continue
            signal, fs = read_wav(wav_path)
            click_signals.append((click, signal, fs, mp_times))

        if not click_signals:
            continue

        # 收集该组所有采样点，计算 μ 和 σ
        all_samples = np.concatenate([sig for _, sig, _, _ in click_signals])
        mu = np.mean(all_samples)
        sigma = np.std(all_samples)
        if sigma == 0:
            sigma = 1e-12

        # 归一化并存储
        for click, sig, fs_val, mp_times in click_signals:
            sig_z = (sig - mu) / sigma

            all_data[(train, click)] = (sig_z, fs_val, mp_times)

    if missing:
        print(f"  警告: {len(missing)} 个 wav 文件缺失，已跳过")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... 还有 {len(missing) - 5} 个")

    print(f"  成功加载 {len(all_data)} 个 click")

    # -------------------------------------------------------
    # 3. 计算三个指标
    # -------------------------------------------------------
    print("\n[3/4] 计算信道参数 (窗宽=±25μs)...")

    output_rows = []
    pbar = tqdm(total=len(all_data), desc="计算", unit="click")

    for (train, click), (sig_z, fs_val, mp_times) in all_data.items():
        # 直达径 RMS（±25μs 窗，中心 5.0ms）
        direct_center_s = DIRECT_TIME_MS / 1000.0
        direct_rms = rms_in_window(sig_z, direct_center_s, fs_val)

        # ---- 直达径行 ----
        output_rows.append({
            "train": train,
            "click": click,
            "path_type": "Direct",
            "delay_us": 0.0,
            "amp_ratio": 1.0,
            "coherence": 1.0,
        })

        # ---- 多径行 ----
        for i, mp_ms in enumerate(mp_times):
            if mp_ms <= 0:
                continue
            # >= 5.0ms 的边界伪影不纳入信道参数计算
            mp_delay_ms = mp_ms - DIRECT_TIME_MS
            if mp_delay_ms >= MAX_LAG_S * 1000.0:
                continue
            mp_center_s = mp_ms / 1000.0

            # 幅度比例 = RMS(直达) / RMS(该多径)
            mp_rms = rms_in_window(sig_z, mp_center_s, fs_val)
            amp_ratio = direct_rms / mp_rms if mp_rms > 0 else np.inf

            # 相干系数 = Pearson r(多径窗, 直达径窗)
            mp_seg, _, _ = extract_window(sig_z, mp_center_s, fs_val)
            direct_seg, _, _ = extract_window(sig_z, direct_center_s, fs_val)
            min_len = min(len(mp_seg), len(direct_seg))
            if min_len < 2:
                coherence = 0.0
            else:
                coherence = pearson_r(mp_seg[:min_len], direct_seg[:min_len])

            output_rows.append({
                "train": train,
                "click": click,
                "path_type": f"Multipath_{i + 1}",
                "delay_us": mp_delay_ms * 1000.0,
                "amp_ratio": min(amp_ratio, 1e6),  # 防溢出
                "coherence": max(-1.0, min(1.0, coherence)),
            })

        pbar.update(1)
    pbar.close()

    # -------------------------------------------------------
    # 4. 写入输出
    # -------------------------------------------------------
    print("\n[4/4] 写入结果...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("PulseTrain\tClickName\tPathType\tDelay_us\tAmpRatio\tCoherence\n")
        for row in output_rows:
            f.write(
                f"{row['train']}\t{row['click']}\t{row['path_type']}\t"
                f"{row['delay_us']:.2f}\t{row['amp_ratio']:.6f}\t"
                f"{row['coherence']:.6f}\n"
            )

    print(f"\n输出文件: {OUTPUT_PATH}")
    print(f"总行数 (含表头): {len(output_rows) + 1}")

    # 统计
    direct_count = sum(1 for r in output_rows if r["path_type"] == "Direct")
    mp_count = len(output_rows) - direct_count
    print(f"直达径: {direct_count}  多径: {mp_count}")

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
