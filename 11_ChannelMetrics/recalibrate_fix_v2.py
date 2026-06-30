# -*- coding: utf-8 -*-
"""recalibrate_fix_v2.py — RMS质心法校准 AmpRatio<0.9 异常 Click，值替换异常行"""

import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
PULSE_TIMES = BASE_DIR / "10_AutoCorrMultipath" / "all_pulse_times.txt"
PULSE_PARAMS = BASE_DIR / "11_ChannelMetrics" / "pulse_params.txt"
ANOMALY_OUT = BASE_DIR / "11_ChannelMetrics" / "anomalous_ampratio_final.txt"

FS = 576000
WINDOW_SAMPLES = 5761
MIN_LAG_S = 0.1e-3
THRESHOLD = 0.10
MAX_MULTIPATH = 3
MAX_LAG_S = 4.99e-3
MIN_ABS_COHERENCE = 0.02
WINDOW_US = 50.0

def read_wav(path):
    sig, fs_val = sf.read(str(path))
    if sig.ndim > 1:
        sig = sig[:, 0]
    return sig.astype(np.float64), fs_val

def autocorr(signal):
    n = len(signal)
    r = np.correlate(signal, signal, mode="full")
    acf_raw = r[n - 1:]
    acf_max = acf_raw[0]
    return acf_raw / acf_max if acf_max > 0 else np.ones_like(acf_raw)

def pearson_r(a, b):
    if len(a) < 2:
        return 0.0
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = a - ma, b - mb
    num = np.sum(sa * sb)
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(num / den) if den > 0 else 0.0

def rms_in_window(sig, center_s, fs_val, half_us=50.0):
    n = len(sig)
    hw = max(1, int(round(half_us * 1e-6 * fs_val)))
    c = int(round(center_s * fs_val))
    lo = max(0, c - hw)
    hi = min(n, c + hw + 1)
    win = sig[lo:hi]
    return np.sqrt(np.mean(win ** 2)) if len(win) > 0 else 0.0

def detect_peaks_v2(acf, threshold, min_lag_samp, max_lag_samp, max_count):
    slice_acf = acf[min_lag_samp:max_lag_samp]
    lag_offset = min_lag_samp
    if len(slice_acf) == 0:
        return [], []
    min_dist = max(1, int(50e-6 * FS))
    peaks_idx, props = find_peaks(slice_acf, height=threshold, distance=min_dist,
                                   prominence=threshold * 0.5)
    if len(peaks_idx) == 0:
        return [], []
    p_lags = (peaks_idx + lag_offset).tolist()
    p_vals = props["peak_heights"].tolist()
    si = np.argsort(p_vals)[::-1]
    p_lags = [p_lags[i] for i in si]
    p_vals = [p_vals[i] for i in si]
    if len(p_lags) > max_count:
        p_lags, p_vals = p_lags[:max_count], p_vals[:max_count]
    return p_lags, p_vals

def find_best_anchor(sig_raw, fs_val):
    squared = sig_raw ** 2
    hw = max(1, int(round(50e-6 * fs_val)))
    kernel = np.ones(2 * hw + 1) / (2 * hw + 1)
    rms_sq = np.convolve(squared, kernel, mode="same")
    rms_sq[:hw] = 0
    rms_sq[-hw:] = 0
    return int(np.argmax(rms_sq))

def find_anomalies(params_lines):
    """从 pulse_params.txt 行列表中找出异常 Click"""
    keys = set()
    for line in params_lines:
        parts = line.strip().split("\t")
        if len(parts) < 6:
            continue
        pt = parts[2]
        amp = float(parts[4])
        if pt.startswith("Multipath") and amp < 0.9:
            keys.add((parts[0], parts[1]))
    return sorted(keys, key=lambda x: (x[0], x[1]))

def main():
    t0 = time.time()
    print("=" * 70)
    print("recalibrate_fix_v2 — RMS质心法校准异常 Click (值替换, 10%阈值, MAX=3)")
    print("=" * 70)

    # 读取 pulse_params.txt
    with open(PULSE_PARAMS, "r", encoding="utf-8") as f:
        pp_header = f.readline()
        pp_lines = []
        for line in f:
            line_stripped = line.strip()
            pp_lines.append(line)
    anomaly_keys = find_anomalies(pp_lines)
    print(f"\n异常 Click: {len(anomaly_keys)}")

    # 构建 wav 映射
    wav_map = {}
    for w in sorted(LOCATE_DIR.rglob("Pulse_*.wav")):
        cn = w.parent.name
        pt = w.parent.parent.name
        wav_map[(pt, cn)] = w

    # 处理异常 Click
    new_times_map = {}
    new_params_map = {}
    stats = {"processed": 0, "failed": 0, "total_mp": 0}

    for key in anomaly_keys:
        train, click = key
        if key not in wav_map:
            stats["failed"] += 1
            continue
        sig_raw, fs_val = read_wav(wav_map[key])
        if len(sig_raw) < WINDOW_SAMPLES:
            stats["failed"] += 1
            continue

        anchor_idx = find_best_anchor(sig_raw, fs_val)
        anchor_s = anchor_idx / fs_val
        direct_ms = anchor_s * 1000

        acf = autocorr(sig_raw)
        min_ls = int(round(MIN_LAG_S * fs_val))
        max_ls = int(round(MAX_LAG_S * fs_val))
        pl, pv = detect_peaks_v2(acf, THRESHOLD, min_ls, max_ls, MAX_MULTIPATH)

        # 边界过滤
        valid = []
        for j, lag in enumerate(pl):
            if (anchor_idx + lag) < len(sig_raw):
                valid.append(j)
        pl = [pl[j] for j in valid]
        pv = [pv[j] for j in valid]

        # Pearson 相干性过滤
        if MIN_ABS_COHERENCE > 0 and len(pl) > 0:
            n_sig = len(sig_raw)
            hw_samp = max(1, int(round(50e-6 * fs_val)))
            ds = min(n_sig - 1, max(0, anchor_idx))
            dw = sig_raw[max(0, ds - hw_samp): min(n_sig, ds + hw_samp + 1)]
            cf = []
            for j, lag in enumerate(pl):
                ts = ds + lag
                mw = sig_raw[max(0, ts - hw_samp): min(n_sig, ts + hw_samp + 1)]
                ml = min(len(dw), len(mw))
                if ml >= 2 and abs(pearson_r(dw[:ml], mw[:ml])) >= MIN_ABS_COHERENCE:
                    cf.append(j)
                elif ml < 2:
                    cf.append(j)
            pl = [pl[j] for j in cf]
            pv = [pv[j] for j in cf]

        mp_ms = [direct_ms + (l / fs_val) * 1000 for l in pl]

        # 存储 times
        new_times_map[key] = (direct_ms, mp_ms)

        # 计算信道参数
        sig_z = sig_raw.copy()
        direct_rms = rms_in_window(sig_z, anchor_s, fs_val, WINDOW_US)
        param_rows = [
            f"{train}\t{click}\tDirect\t0.00\t1.000000\t1.000000"
        ]
        for i, (lag, ms) in enumerate(zip(pl, mp_ms)):
            delay_us = (ms - direct_ms) * 1000
            mp_s = ms / 1000
            mp_rms = rms_in_window(sig_z, mp_s, fs_val, WINDOW_US)
            amp_ratio = direct_rms / mp_rms if mp_rms > 0 else float('inf')
            hw_s = max(1, int(round(50e-6 * fs_val)))
            ds = min(len(sig_z) - 1, max(0, anchor_idx))
            ts = min(len(sig_z) - 1, max(0, int(round(mp_s * fs_val))))
            dw = sig_z[max(0, ds - hw_s): min(len(sig_z), ds + hw_s + 1)]
            mw = sig_z[max(0, ts - hw_s): min(len(sig_z), ts + hw_s + 1)]
            ml = min(len(dw), len(mw))
            coherence = pearson_r(dw[:ml], mw[:ml]) if ml >= 2 else 0.0
            param_rows.append(
                f"{train}\t{click}\tMultipath_{i+1}\t{delay_us:.2f}\t{amp_ratio:.6f}\t{coherence:.6f}"
            )
        new_params_map[key] = param_rows
        stats["processed"] += 1
        stats["total_mp"] += len(pl)

    # 写入 all_pulse_times.txt (仅替换异常行)
    print("\n写入 all_pulse_times.txt...")
    with open(PULSE_TIMES, "r", encoding="utf-8") as f:
        times_lines = f.readlines()

    new_times_lines = []
    for line in times_lines:
        if line.startswith("#"):
            new_times_lines.append(line)
            continue
        parts = line.strip().split("\t")
        if len(parts) < 3:
            new_times_lines.append(line)
            continue
        key = (parts[0], parts[1])
        if key in new_times_map:
            direct_ms, mp_ms = new_times_map[key]
            parts[2] = f"{direct_ms:.4f}"
            remaining = [f"{t:.4f}" for t in mp_ms]
            new_line = "\t".join(parts[:2] + [parts[2]] + remaining) + "\n"
            new_times_lines.append(new_line)
        else:
            new_times_lines.append(line)

    with open(PULSE_TIMES, "w", encoding="utf-8") as f:
        f.writelines(new_times_lines)

    # 写入 pulse_params.txt (仅替换异常行)
    print("写入 pulse_params.txt...")
    new_pp_lines = [pp_header]
    for line in pp_lines:
        parts = line.strip().split("\t")
        if len(parts) < 6:
            new_pp_lines.append(line)
            continue
        key = (parts[0], parts[1])
        if key not in new_params_map:
            new_pp_lines.append(line)
    # 追加异常 Click 的新行
    for key, rows in new_params_map.items():
        for r in rows:
            new_pp_lines.append(r + "\n")

    with open(PULSE_PARAMS, "w", encoding="utf-8") as f:
        f.writelines(new_pp_lines)

    # 验证并检测残留异常
    print("验证 & 检测残留异常...")
    total_direct = 0
    total_mp = 0
    still_anomaly = []
    with open(PULSE_PARAMS, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            pt = parts[2]
            amp = float(parts[4])
            if pt == "Direct":
                total_direct += 1
            elif pt.startswith("Multipath"):
                total_mp += 1
                if amp < 0.9:
                    still_anomaly.append((parts[0], parts[1], pt, float(parts[3]), amp, float(parts[5])))

    # 写入异常文件
    with open(ANOMALY_OUT, "w", encoding="utf-8") as f:
        f.write(f"# 校准后仍 AmpRatio < 0.9 的异常径 (10% 阈值, MAX=3)\n")
        f.write(f"# 异常径数: {len(still_anomaly)}\n")
        f.write(f"# PulseTrain\tClickName\tPathType\tDelay_us\tAmpRatio\tCoherence\n")
        for train, click, pt, md, ar, co in still_anomaly:
            f.write(f"{train}\t{click}\t{pt}\t{md:.2f}\t{ar:.6f}\t{co:.6f}\n")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("校准结果")
    print("=" * 70)
    print(f"  校准前异常 Click: {len(anomaly_keys)}")
    print(f"  成功校准:         {stats['processed']}")
    print(f"  失败:             {stats['failed']}")
    print(f"  新检出多径总数:   {stats['total_mp']}")
    print(f"  直达径总数:       {total_direct}")
    print(f"  多径总数:         {total_mp}")
    print(f"  校准后仍异常:     {len(still_anomaly)}")
    print(f"  耗时:             {elapsed:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
