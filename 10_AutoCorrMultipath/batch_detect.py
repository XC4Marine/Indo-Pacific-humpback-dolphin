#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_detect.py — 对全部 Click 片段做自相关次峰法直达/多径检测，输出 txt"""

import time, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import find_peaks
from tqdm import tqdm

BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
OUTPUT_DIR = BASE_DIR / "10_AutoCorrMultipath"
TXT_PATH = OUTPUT_DIR / "all_pulse_times.txt"

# ── 参数 ──
MIN_LAG_S = 0.1e-3           # 最小滞后 (避开直达峰)
THRESHOLD = 0.10             # 10% 阈值
MAX_MULTIPATH = 3            # 最多多径数
FS = 576000                  # 采样率
WINDOW_SAMPLES = 5761        # 窗口长度
DIRECT_OFFSET_S = 5e-3       # 直达脉冲位于窗口中心 (+5ms)

def autocorr(signal):
    n = len(signal)
    result = np.correlate(signal, signal, mode="full")
    acf_raw = result[n - 1:]
    acf_max = acf_raw[0]
    return acf_raw / acf_max if acf_max > 0 else np.ones_like(acf_raw)

def detect_peaks(acf, lag_samples, threshold, min_lag_samples, max_count, fs):
    pos_mask = lag_samples >= min_lag_samples
    acf_valid = acf[pos_mask]
    lag_valid = lag_samples[pos_mask]
    if len(acf_valid) == 0:
        return [], []
    min_distance = max(1, int(30e-6 * fs))
    peaks_idx, props = find_peaks(acf_valid, height=threshold, distance=min_distance,
                                   prominence=threshold * 0.5)
    if len(peaks_idx) == 0:
        return [], []
    p_lags = lag_valid[peaks_idx]
    p_vals = props["peak_heights"]
    si = np.argsort(p_vals)[::-1]
    p_lags, p_vals = p_lags[si], p_vals[si]
    if len(p_lags) > max_count:
        p_lags, p_vals = p_lags[:max_count], p_vals[:max_count]
    return list(p_lags), list(p_vals)

def read_wav(path):
    signal, fs = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal.astype(np.float64), fs

def main():
    print("=" * 70)
    print("10_AutoCorrMultipath — 全量批处理")
    print("=" * 70)
    t_start = time.time()

    all_wavs = sorted(LOCATE_DIR.rglob("Pulse_*.wav"))
    total = len(all_wavs)
    print(f"\n# 发现 {total} 个 Click 片段")
    print(f"# 阈值: {THRESHOLD*100:.0f}%, 最大多径数: {MAX_MULTIPATH}, 最小滞后: {MIN_LAG_S*1e3:.2f}ms")
    print(f"# 输出: {TXT_PATH}\n")

    min_ls = int(round(MIN_LAG_S * FS))

    with open(str(TXT_PATH), "w", encoding="utf-8") as f_out:
        f_out.write("# 10_AutoCorrMultipath — 全量直达/多径脉冲时间\n")
        f_out.write(f"# 检测参数: threshold={THRESHOLD*100:.0f}%, min_lag={MIN_LAG_S*1e3:.2f}ms, max_mp={MAX_MULTIPATH}\n")
        f_out.write("# 格式: PulseTrain  ClickName  直达时间(ms)  多径1时间(ms)  多径2...  多径3\n")
        f_out.write("# 直达时间固定为 +5.0000ms (窗口中心)\n")
        f_out.write("#\n")

        stats = {"total": 0, "has_mp": 0, "total_mp": 0}

        pbar = tqdm(total=total, desc="检测进度", unit="f", ncols=90)
        for wav_path in all_wavs:
            signal, fs = read_wav(wav_path)
            acf = autocorr(signal)
            pl, pv = detect_peaks(acf, np.arange(len(acf)), THRESHOLD, min_ls, MAX_MULTIPATH, fs)

            pt = wav_path.parent.parent.name          # PulseTrain_XXX
            cn = wav_path.parent.name                  # Click_YYY
            direct_ms = DIRECT_OFFSET_S * 1000         # 5.0 ms

            # 多径时间 = 直达时间 + lag_offset
            mp_times_ms = [direct_ms + (l / fs) * 1000 for l in pl]

            stats["total"] += 1
            stats["total_mp"] += len(mp_times_ms)
            if len(mp_times_ms) > 0:
                stats["has_mp"] += 1

            # 构建每行: PT Click direct_ms mp1 mp2 mp3 (空位用 NaN)
            parts = [pt, cn, f"{direct_ms:.4f}"]
            for i in range(MAX_MULTIPATH):
                if i < len(mp_times_ms):
                    parts.append(f"{mp_times_ms[i]:.4f}")
                else:
                    parts.append("NaN")
            f_out.write("\t".join(parts) + "\n")

            pbar.update(1)
        pbar.close()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    print(f"  总片段数:        {stats['total']}")
    print(f"  检出≥1个多径:    {stats['has_mp']}/{stats['total']} ({stats['has_mp']/stats['total']*100:.1f}%)")
    print(f"  多径总数:        {stats['total_mp']}")
    print(f"  平均每片段:      {stats['total_mp']/stats['total']:.2f}")
    print(f"  耗时:            {elapsed:.1f}s")
    print(f"\n# 输出文件: {TXT_PATH}")
    print("=" * 70)

    # 追加统计到 txt 末尾
    with open(str(TXT_PATH), "a", encoding="utf-8") as f_out:
        f_out.write(f"\n# -- 统计 --\n")
        f_out.write(f"# 总片段数: {stats['total']}\n")
        f_out.write(f"# 检出≥1个多径: {stats['has_mp']}/{stats['total']} ({stats['has_mp']/stats['total']*100:.1f}%)\n")
        f_out.write(f"# 多径总数: {stats['total_mp']}\n")
        f_out.write(f"# 平均每片段: {stats['total_mp']/stats['total']:.2f}\n")

if __name__ == "__main__":
    main()
