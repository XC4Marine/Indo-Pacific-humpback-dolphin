# -*- coding: utf-8 -*-
"""recalibrate_fix_v2.py — RMS质心法全量前置直达锚点校准"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
LOCATE_DIR = BASE_DIR / "09_Locate"
OUTPUT_DIR = BASE_DIR / "10_AutoCorrMultipath"
ANCHOR_PATH = OUTPUT_DIR / "calibrated_anchors.txt"

FS = 576000
ANCHOR_WINDOW_US = 50.0


def read_wav(path):
    sig, fs_val = sf.read(str(path))
    if sig.ndim > 1:
        sig = sig[:, 0]
    return sig.astype(np.float64), fs_val


def find_best_anchor(sig_raw, fs_val):
    """Return the sample index with maximum local RMS energy."""
    squared = sig_raw ** 2
    hw = max(1, int(round(ANCHOR_WINDOW_US * 1e-6 * fs_val)))
    kernel = np.ones(2 * hw + 1) / (2 * hw + 1)
    rms_sq = np.convolve(squared, kernel, mode="same")
    rms_sq[:hw] = 0
    rms_sq[-hw:] = 0
    return int(np.argmax(rms_sq))


def main():
    t0 = time.time()
    print("=" * 70)
    print("recalibrate_fix_v2 — RMS质心法全量前置锚点校准")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_wavs = sorted(LOCATE_DIR.rglob("Pulse_*.wav"))
    total = len(all_wavs)
    print(f"\n# 发现 {total} 个 Click 片段")
    print(f"# 输出: {ANCHOR_PATH}\n")

    stats = {"total": 0, "failed": 0}
    with open(ANCHOR_PATH, "w", encoding="utf-8") as f:
        f.write("# RMS质心法前置直达锚点校准\n")
        f.write(f"# anchor_window_us={ANCHOR_WINDOW_US:.1f}\n")
        f.write("# 格式: PulseTrain\tClickName\tDirectTime_ms\tAnchorSample\tSampleRate\n")

        pbar = tqdm(total=total, desc="锚点校准进度", unit="click", ncols=90)
        for wav_path in all_wavs:
            train = wav_path.parent.parent.name
            click = wav_path.parent.name
            try:
                sig_raw, fs_val = read_wav(wav_path)
                anchor_idx = find_best_anchor(sig_raw, fs_val)
                direct_ms = anchor_idx / fs_val * 1000.0
                f.write(f"{train}\t{click}\t{direct_ms:.4f}\t{anchor_idx}\t{fs_val}\n")
                stats["total"] += 1
            except Exception as exc:
                stats["failed"] += 1
                f.write(f"# FAILED\t{train}\t{click}\t{exc}\n")
            pbar.update(1)
        pbar.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("锚点校准结果")
    print("=" * 70)
    print(f"  成功校准: {stats['total']}")
    print(f"  失败:     {stats['failed']}")
    print(f"  耗时:     {elapsed:.1f}s")
    print(f"\n# 输出文件: {ANCHOR_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
