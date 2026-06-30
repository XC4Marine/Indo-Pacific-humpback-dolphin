#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_anomaly.py — 绘制8个AmpRatio异常click的波形图，标注直达径与多径位置 (窗宽±25μs)"""

from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
PULSE_TIMES_PATH = BASE_DIR / "10_AutoCorrMultipath" / "all_pulse_times.txt"
OUTPUT_DIR = BASE_DIR / "11_ChannelMetrics" / "anomaly_plots"
WINDOW_US = 50.0
DIRECT_TIME_MS = 5.0
MAX_LAG_MS = 5.0
MAX_LAG_S = 4.99e-3         # 严格排除 >= 5.0ms 的边界伪影

# 8个异常click
ANOMALIES = [
    ("PulseTrain_011", "Click_009", 6840.3),
    ("PulseTrain_414", "Click_003", 6123.3),
    ("PulseTrain_421", "Click_009", 8892.4),
    ("PulseTrain_565", "Click_002", 5123.3),
    ("PulseTrain_641", "Click_019", 8814.2),
    ("PulseTrain_655", "Click_003", 5100.7),
    ("PulseTrain_709", "Click_001", 7024.3),
]

# 解析 all_pulse_times.txt，建立索引
pulse_map = {}
with open(PULSE_TIMES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        train, click = parts[0], parts[1]
        mp_times = []
        for val in parts[3:]:
            mp_times.append(float(val) if val.strip().lower() != "nan" else np.nan)
        pulse_map[(train, click)] = mp_times

def read_wav(path):
    signal, fs = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal.astype(np.float64), fs

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for train, click, _ in ANOMALIES:
    wav_path = LOCATE_DIR / train / click / f"Pulse_{click.split('_')[1]}.wav"
    signal, fs = read_wav(wav_path)
    n = len(signal)
    direct_sample = int(round(DIRECT_TIME_MS / 1000.0 * fs))
    t_ms = (np.arange(n) - direct_sample) / fs * 1000.0

    mp_times = pulse_map.get((train, click), [])
    valid_mp = [(i+1, t) for i, t in enumerate(mp_times) if not np.isnan(t) and (t - DIRECT_TIME_MS) < MAX_LAG_S * 1000.0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Raw Waveform
    ax = axes[0]
    ax.plot(t_ms, signal, color="#333333", linewidth=0.5)
    ax.axvline(x=0, color="#2E8B57", linewidth=2, linestyle="-")
    ax.axvspan(-WINDOW_US / 1000.0, WINDOW_US / 1000.0, alpha=0.15, color="#2E8B57")
    colors = ["#1E90FF", "#FF6347", "#FFD700"]
    for idx, (mp_i, mp_ms) in enumerate(valid_mp):
        t_mp = mp_ms - DIRECT_TIME_MS
        c = colors[min(idx, 2)]
        ax.axvline(x=t_mp, color=c, linewidth=1.5, linestyle="--")
        half = WINDOW_US / 1000.0
        ax.axvspan(t_mp - half, t_mp + half, alpha=0.10, color=c)

    ax.set_title(f"{train} / {click}  — Raw Waveform", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time re. Direct (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    legends = [mpatches.Patch(color="#2E8B57", alpha=0.3, label=f"Direct ±{WINDOW_US:.0f}μs")]
    for idx, (mp_i, mp_ms) in enumerate(valid_mp):
        c = colors[min(idx, 2)]
        legends.append(mpatches.Patch(color=c, alpha=0.2, label=f"MP{mp_i}: {mp_ms:.4f}ms ±{WINDOW_US:.0f}μs"))
    ax.legend(handles=legends, loc="upper right", fontsize=8)

    # Z-Score 归一化后
    mu = np.mean(signal)
    sigma = np.std(signal)
    sig_z = (signal - mu) / sigma if sigma > 0 else signal

    ax2 = axes[1]
    ax2.plot(t_ms, sig_z, color="#555555", linewidth=0.5)
    ax2.axvline(x=0, color="#2E8B57", linewidth=2, linestyle="-")
    ax2.axvspan(-WINDOW_US / 1000.0, WINDOW_US / 1000.0, alpha=0.15, color="#2E8B57")
    for idx, (mp_i, mp_ms) in enumerate(valid_mp):
        t_mp = mp_ms - DIRECT_TIME_MS
        c = colors[min(idx, 2)]
        ax2.axvline(x=t_mp, color=c, linewidth=1.5, linestyle="--")
        half = WINDOW_US / 1000.0
        ax2.axvspan(t_mp - half, t_mp + half, alpha=0.10, color=c)
    ax2.set_title(f"{train} / {click}  — Z-Score Z-Score Normalized (μ={mu:.4f}, σ={sigma:.4f})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time re. Direct (ms)")
    ax2.set_ylabel("Z-Score")
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legends, loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{train}_{click}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"已生成: {out_path.name}")

print("\n完成。")
