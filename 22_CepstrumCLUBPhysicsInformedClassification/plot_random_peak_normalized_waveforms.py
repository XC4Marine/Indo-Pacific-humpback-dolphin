"""Plot reproducibly sampled, per-waveform peak-normalized input waveforms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    EXPERIMENT_DIR.parent / "21_PhysicsInformedClickClassification"
    / "outputs_snr_m15_p15_source_only_5seeds"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing dataset_manifest.csv and cache/preprocessed_waveforms.npy")
    parser.add_argument("--output-dir", type=Path,
                        default=EXPERIMENT_DIR / "figures" / "random_peak_normalized_waveforms")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--samples-per-class", type=int, default=10)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    manifest_path = data_dir / "dataset_manifest.csv"
    array_path = data_dir / "cache" / "preprocessed_waveforms.npy"

    with tqdm(total=4, desc="Overall progress", unit="stage") as progress:
        manifest = pd.read_csv(manifest_path)
        waveforms = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if waveforms.ndim != 2 or waveforms.shape != (len(manifest), 5761):
            raise ValueError(f"Manifest/cache shape mismatch: {waveforms.shape}")
        if manifest["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample IDs in manifest")
        # The pipeline writes cache rows in manifest order; verify against its audit.
        audit = pd.read_csv(data_dir / "waveform_audit.csv")
        if not np.array_equal(audit["sample_id"].to_numpy(), manifest["sample_id"].to_numpy()):
            raise ValueError("Waveform audit and manifest sample order differ")
        if not np.array_equal(audit["sample_index"].to_numpy(), np.arange(len(manifest))):
            raise ValueError("Waveform audit indices do not match cache row indices")
        rates = audit["sample_rate_hz"].unique()
        if len(rates) != 1 or rates[0] <= 0:
            raise ValueError("Expected a single positive sample rate")
        sample_rate = float(rates[0])
        progress.update(1)

        rng = np.random.default_rng(args.seed)
        selected_parts = []
        for label in (1, 0):
            pool = np.flatnonzero(manifest["label"].to_numpy() == label)
            if len(pool) < args.samples_per_class:
                raise ValueError(f"Class {label} has only {len(pool)} samples")
            indices = rng.choice(pool, args.samples_per_class, replace=False)
            part = manifest.iloc[indices].copy()
            part.insert(0, "sample_index", indices)
            part.insert(1, "plot_number", np.arange(1, len(part) + 1))
            selected_parts.append(part)
        selected = pd.concat(selected_parts, ignore_index=True)
        values = np.asarray(waveforms[selected["sample_index"].to_numpy()], dtype=np.float64)
        peaks = np.max(np.abs(values), axis=1)
        if not np.isfinite(values).all() or np.any(peaks == 0):
            raise ValueError("Selected waveforms contain non-finite values or zero peaks")
        normalized = values / peaks[:, None]
        normalized_peaks = np.max(np.abs(normalized), axis=1)
        if not np.allclose(normalized_peaks, 1.0, rtol=0, atol=1e-12):
            raise ValueError("Peak normalization verification failed")
        selected["peak_before"] = peaks
        selected["peak_after"] = normalized_peaks
        selected["sampling_seed"] = args.seed
        output_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output_dir / "selected_samples.csv", index=False, encoding="utf-8-sig")
        progress.update(1)

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})
        time_ms = np.arange(waveforms.shape[1]) / sample_rate * 1000
        image_paths = []
        for label, class_name, color in ((1, "positive", "#0072B2"),
                                          (0, "negative", "#D55E00")):
            positions = np.flatnonzero(selected["label"].to_numpy() == label)
            rows = (len(positions) + 1) // 2
            figure, axes = plt.subplots(rows, 2, figsize=(13, 2.4 * rows),
                                        sharex=True, sharey=True, squeeze=False,
                                        layout="constrained")
            for axis, position in zip(axes.flat, positions):
                row = selected.iloc[position]
                axis.plot(time_ms, normalized[position], color=color, linewidth=0.65)
                axis.axhline(0, color="#888888", linewidth=0.4, zorder=0)
                axis.set_ylim(-1.05, 1.05)
                axis.set_xlim(time_ms[0], time_ms[-1])
                axis.set_yticks([-1, 0, 1])
                axis.grid(alpha=0.18)
                subtitle = f"{int(row['plot_number']):02d} | {row['sample_id']}"
                if label == 1 and pd.notna(row.get("target_snr_db")):
                    subtitle += f" | SNR {row['target_snr_db']:+.2f} dB"
                axis.set_title(subtitle, loc="left", fontsize=9)
            for axis in list(axes.flat)[len(positions):]:
                axis.set_visible(False)
            figure.suptitle(f"{class_name.capitalize()} input waveforms | n={len(positions)} | seed={args.seed}",
                            fontsize=15, fontweight="bold")
            figure.supxlabel("Time from window start (ms)")
            figure.supylabel("Peak-normalized amplitude")
            image_path = output_dir / f"{class_name}_{len(positions)}_peak_normalized.png"
            figure.savefig(image_path, dpi=180)
            plt.close(figure)
            image_paths.append(str(image_path))
            progress.update(1)

    metadata = {"manifest": str(manifest_path), "waveform_cache": str(array_path),
                "sample_rate_hz": sample_rate, "seed": args.seed,
                "samples_per_class": args.samples_per_class,
                "normalization": "Each selected waveform divided by its maximum absolute amplitude",
                "peak_normalization_verified": True, "images": image_paths}
    (output_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
