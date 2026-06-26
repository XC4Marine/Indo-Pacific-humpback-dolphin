#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""autocorr_detect.py — 自相关次峰法直达/多径脉冲检测实验"""

import sys, time, gc
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
OUTPUT_DIR = BASE_DIR / "10_AutoCorrMultipath"
PLOTS_DIR = OUTPUT_DIR / "plots"
RANDOM_SEED = 12025
SAMPLE_COUNT = 100
MIN_LAG_S = 0.1e-3
CANDIDATE_THRESHOLDS = [0.05, 0.10]
MAX_MULTIPATH = 3
MARK_RANGE_S = 100e-6

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
    peaks_idx, props = find_peaks(acf_valid, height=threshold,
                                   distance=min_distance, prominence=threshold * 0.5)
    if len(peaks_idx) == 0:
        return [], []
    p_lags = lag_valid[peaks_idx]
    p_vals = props["peak_heights"]
    si = np.argsort(p_vals)[::-1]
    p_lags, p_vals = p_lags[si], p_vals[si]
    if len(p_lags) > max_count:
        p_lags, p_vals = p_lags[:max_count], p_vals[:max_count]
    return list(p_lags), list(p_vals)

def lag_to_offset(l, fs):
    return l / fs

def generate_plot(signal, fs, acf, direct_lag, mp_lags, mp_vals, save_path, label, thresh):
    n = len(signal)
    t_ms = (np.arange(n) - direct_lag) / fs * 1000
    lag_ms = np.arange(len(acf)) / fs * 1000
    mp_offsets = [lag_to_offset(ml, fs) for ml in mp_lags]
    mp_samps = [direct_lag + int(round(ms * fs)) for ms in mp_offsets]
    valid = [(i, s) for i, s in enumerate(mp_samps) if 0 <= s < n]
    mp_samps = [s for _, s in valid]
    mp_lags_v = [mp_lags[i] for i, _ in valid]
    mp_vals_v = [mp_vals[i] for i, _ in valid]
    mr = MARK_RANGE_S * 1000
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(t_ms, signal, color="#333333", linewidth=0.6)
    ax1.axvline(x=0, color="#2E8B57", linewidth=1.5)
    ax1.axvspan(-mr, mr, alpha=0.15, color="#2E8B57")
    for mp_s in mp_samps:
        t_mp = (mp_s - direct_lag) / fs * 1000
        ax1.axvline(x=t_mp, color="#1E90FF", linewidth=1.2, linestyle="--")
        ax1.axvspan(t_mp - mr, t_mp + mr, alpha=0.12, color="#1E90FF")
    ax1.set_title(label + " — Waveform", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Amplitude")
    ax1.set_xlim(t_ms[0], t_ms[-1]); ax1.grid(True, alpha=0.3)
    le = [mpatches.Patch(color="#2E8B57", alpha=0.3, label="Direct pulse ±100μs")]
    if len(mp_samps) > 0:
        le.append(mpatches.Patch(color="#1E90FF", alpha=0.3, label="Multipath ±100μs"))
    ax1.legend(handles=le, loc="upper right", fontsize=9)
    ax2.plot(lag_ms, acf, color="#0072BD", linewidth=1.0)
    ax2.set_title(f"Autocorrelation — Thresh={thresh*100:.0f}%, MP: {len(mp_samps)}",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Lag (ms)"); ax2.set_ylabel("Normalized Correlation")
    ax2.plot(0, 1.0, marker="*", color="#2E8B57", markersize=15,
             markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    for ml, mv in zip(mp_lags_v, mp_vals_v):
        lp = ml / fs * 1000
        ax2.plot(lp, mv, marker="o", color="#1E90FF", markersize=8,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax2.annotate(f"{mv:.3f}", (lp, mv), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#1E90FF")
    ax2.axhline(y=thresh, color="#FF4500", linewidth=1.0, linestyle=":",
                label=f"Threshold ({thresh*100:.0f}%)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(lag_ms[0], lag_ms[-1]); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)

def generate_html(results, out_path, plot_files, best_th, n_display=20):
    total = len(results)
    has_mp = sum(1 for r in results if len(r.get("mp_lags_ms", [])) > 0)
    total_mp = sum(len(r.get("mp_lags_ms", [])) for r in results)
    avg_mp = total_mp / total if total > 0 else 0
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>10_AutoCorrMultipath — 实验摘要</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;margin:30px;background:#f5f5f5}}
h1{{color:#333;border-bottom:3px solid #2E8B57;padding-bottom:10px}}
h2{{color:#555;margin-top:30px}}
.stats{{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.1);margin:15px 0}}
.stats table{{border-collapse:collapse;width:100%}}
.stats td{{padding:8px 12px;border-bottom:1px solid #eee}}
.stats td:first-child{{font-weight:bold;color:#555;width:250px}}
.gallery{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-top:20px}}
.gallery img{{width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15)}}
.gallery figure{{margin:0}}
.gallery figcaption{{text-align:center;font-size:12px;color:#666;margin-top:6px}}
</style></head><body>
<h1>10_AutoCorrMultipath — 实验摘要</h1>
<div class="stats"><h2>统计摘要</h2><table>
<tr><td>实验样本总数</td><td>{total}</td></tr>
<tr><td>随机种子</td><td>{RANDOM_SEED}</td></tr>
<tr><td>最佳阈值</td><td>{best_th*100:.1f}%</td></tr>
<tr><td>检出≥1个多径</td><td>{has_mp}/{total}({has_mp/total*100:.1f}%)</td></tr>
<tr><td>多径总数</td><td>{total_mp}</td></tr>
<tr><td>平均每片段</td><td>{avg_mp:.2f}</td></tr>
</table></div>
<h2>结果图（前{min(n_display, len(plot_files))}张）</h2>
<div class="gallery">"""
    for pf in plot_files[:n_display]:
        html += f'<figure><img src="plots/{pf}" loading="lazy"><figcaption>{pf}</figcaption></figure>'
    html += "</div></body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def read_wav(path):
    signal, fs = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal.astype(np.float64), fs

def main():
    print("=" * 60)
    print("10_AutoCorrMultipath")
    print("=" * 60)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    all_wavs = sorted(LOCATE_DIR.rglob("Pulse_*.wav"))
    print(f"\n发现 {len(all_wavs)} 个 Click 片段")
    if len(all_wavs) < SAMPLE_COUNT:
        selected = all_wavs
    else:
        indices = np.random.choice(len(all_wavs), size=SAMPLE_COUNT, replace=False)
        selected = [all_wavs[i] for i in indices]
    print(f"选取 {len(selected)} 个样本\n")

    # 阶段 1: 计算 ACF
    print("阶段 1/2: 计算自相关...")
    acf_list = []
    all_peaks = []
    for wav_path in tqdm(selected, desc="处理", unit="f"):
        signal, fs = read_wav(wav_path)
        n = len(signal)
        acf = autocorr(signal)
        min_ls = int(round(MIN_LAG_S * fs))
        pm = np.arange(len(acf)) >= min_ls
        acf_p = acf[pm]
        if len(acf_p) > 0:
            pks, _ = find_peaks(acf_p, height=0, distance=1)
            for pi in pks:
                all_peaks.append(float(acf_p[pi]))
        pt = wav_path.parent.parent.name
        cn = wav_path.parent.name
        acf_list.append((acf, fs, pt, cn, str(wav_path)))
    print(f"完成 {len(acf_list)} 个\n")

    # 阈值判定
    print("阈值判定：")
    best_th = None
    best_d = float("inf")
    for th in CANDIDATE_THRESHOLDS:
        cnt = []
        for acf, fs, pt, cn, wp in acf_list:
            min_ls = int(round(MIN_LAG_S * fs))
            pl, _ = detect_peaks(acf, np.arange(len(acf)), th, min_ls, MAX_MULTIPATH, fs)
            cnt.append(len(pl))
        avg = np.mean(cnt) if cnt else 0
        nz = sum(1 for c in cnt if c > 0) / len(cnt) if cnt else 0
        print(f"  {th*100:.0f}%: avg={avg:.2f}, 非零={nz:.1%}")
        if 0 < avg <= MAX_MULTIPATH and abs(avg - 1.5) < abs(best_d - 1.5):
            best_d = abs(avg - 1.5); best_th = th
    if best_th is None:
        best_th = max(CANDIDATE_THRESHOLDS)
    print(f"  >> {best_th*100:.1f}%\n")

    # 阶段 2: 可视化
    print("阶段 2/2: 生成图表...")
    results = []
    plot_files = []
    pbar = tqdm(total=len(acf_list), desc="总进度", unit="f")
    for idx, (acf, fs, pt, cn, wav_path) in enumerate(acf_list):
        signal, _ = read_wav(Path(wav_path))
        ds = len(signal) // 2
        min_ls = int(round(MIN_LAG_S * fs))
        pl, pv = detect_peaks(acf, np.arange(len(acf)), best_th, min_ls, MAX_MULTIPATH, fs)
        plot_fn = f"{idx+1:03d}_{pt}_{cn}.png"
        pp = PLOTS_DIR / plot_fn
        generate_plot(signal, fs, acf, ds, pl, pv, pp, f"{pt}/{cn}", best_th)
        plot_files.append(plot_fn)
        results.append({
            "pt": pt, "cn": cn, "n_mp": len(pl),
            "mp_lags_ms": [l/fs*1000 for l in pl],
            "mp_vals": pv,
        })
        pbar.update(1)
    pbar.close()

    # CSV
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("pulse_train,click_name,num_multipath,mp1_lag_ms,mp1_acorr,mp2_lag_ms,mp2_acorr,mp3_lag_ms,mp3_acorr\n")
        for r in results:
            ml = r["mp_lags_ms"]; mv = r["mp_vals"]
            row = f"{r['pt']},{r['cn']},{r['n_mp']}"
            for i in range(3):
                row += f",{ml[i]:.6f},{mv[i]:.4f}" if i < len(ml) else ",,"
            f.write(row + "\n")
    print(f"\nCSV: {csv_path}")

    html_path = OUTPUT_DIR / "summary.html"
    generate_html(results, html_path, plot_files, best_th)
    print(f"HTML: {html_path}")

    tot = len(results); hmp = sum(1 for r in results if r["n_mp"] > 0)
    tmp = sum(r["n_mp"] for r in results)
    print("\n" + "=" * 60)
    print(f"样本: {tot}  |  阈值: {best_th*100:.1f}%  |  ≥1多径: {hmp}/{tot}({hmp/tot*100:.1f}%)")
    print(f"多径总数: {tmp}  |  平均: {tmp/tot:.2f}  |  图片: {len(plot_files)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
