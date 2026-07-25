from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

from src.click_detection import DetectionSettings, bandpass_filter, read_wav_mono
from src.fpod_pdf_pipeline import (
    FpodPdfSettings,
    run_fpod_pdf_pipeline,
    write_fpod_records_csv,
)
from src.visualization import plot_multipath_cluster, write_features_csv


BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
DEFAULT_WAV = BASE_DIR / "00_Data" / "ClickTrains" / "PulseTrain_651" / "PulseTrain.wav"
DEFAULT_OUTPUT_DIR = BASE_DIR / "16_FpodStyleMultipath"


def progress(step: int, total: int, message: str) -> None:
    width = 32
    filled = round(width * step / total)
    bar = "#" * filled + "." * (width - filled)
    percent = int(round(100 * step / total))
    print(f"[{bar}] {percent:3d}% {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F-POD KERNO-F-style click multipath cluster analysis."
    )
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="Input PulseTrain.wav path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_path = args.wav.resolve()
    output_dir = args.output_dir.resolve()
    png_path = output_dir / "multipath_cluster_visualization.png"
    cropped_png_path = output_dir / "multipath_cluster_visualization_80_120ms.png"
    cropped_30_130_png_path = output_dir / "multipath_cluster_visualization_30_130ms.png"
    csv_path = output_dir / "multipath_features.csv"
    records_csv_path = output_dir / "fpod_click_records.csv"

    progress(0, 6, "Preparing input")
    if not wav_path.exists():
        raise FileNotFoundError(f"Input WAV not found: {wav_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(1, 6, "Reading WAV")
    sample_rate, signal = read_wav_mono(wav_path)

    progress(2, 6, "Bandpass filtering above 20 kHz and below 220 kHz")
    detection_settings = DetectionSettings()
    filtered = bandpass_filter(signal, sample_rate, detection_settings)

    progress(3, 6, "Running PDF-structured F-POD pipeline")
    fpod_settings = FpodPdfSettings()
    result = run_fpod_pdf_pipeline(filtered, sample_rate, fpod_settings)
    if not result.click_records:
        raise RuntimeError("No FP1-like click records were extracted.")
    if not result.features:
        raise RuntimeError("No multipath features were built from FP1-like click records.")

    progress(4, 6, "Writing FP1-like records and feature CSV")
    write_fpod_records_csv(result.click_records, records_csv_path)
    write_features_csv(result.features, csv_path)

    progress(5, 6, "Plotting FP3-like multipath cluster views")
    plot_multipath_cluster(
        result.features,
        png_path,
        title="PDF-structured F-POD click multipath cluster visualization",
    )
    plot_multipath_cluster(
        result.features,
        cropped_png_path,
        title="PDF-structured F-POD click multipath cluster visualization (80-120 ms)",
        time_start_ms=80.0,
        time_end_ms=120.0,
    )
    plot_multipath_cluster(
        result.features,
        cropped_30_130_png_path,
        title="PDF-structured F-POD click multipath cluster visualization (30-130 ms)",
        time_start_ms=30.0,
        time_end_ms=130.0,
    )

    progress(6, 6, "Done")
    print(f"Input WAV: {wav_path}")
    print(f"Original sample rate: {sample_rate} Hz")
    print(f"F-POD-style sample rate: {result.sample_rate} Hz")
    print(
        "FP1-like click records: "
        f"{int(result.detection_choice['click_records'])}, "
        f"amplitude p{result.detection_choice['amplitude_percentile']:.0f}, "
        f"IPI change tolerance {result.detection_choice['ipi_change_tolerance']:.2f}"
    )
    print(
        "FP3-like structure: "
        f"trains={result.train_count}, "
        f"multipath clusters={result.multipath_cluster_count}, "
        f"feature rows={len(result.features)}"
    )
    print(f"FP1-like records CSV: {records_csv_path}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"Cropped PNG: {cropped_png_path}")
    print(f"Cropped PNG 30-130 ms: {cropped_30_130_png_path}")


if __name__ == "__main__":
    main()
