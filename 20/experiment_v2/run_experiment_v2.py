#!/usr/bin/env python3
"""Reproducible Bellhop 20--190 kHz shallow-water click synthesis.

This Python entry point replaces the MATLAB runtime requirement.  It keeps the
Bellhop solver, parses its ASCII arrivals natively, and writes only the
artifacts needed to reproduce and use each synthesized multipath click.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "00_Data"
FREQUENCIES_HZ = np.arange(20_000, 190_001, 5_000, dtype=float)
SEED = 20260907


@dataclass(frozen=True)
class Config:
    input_csv: str
    ssp_csv: str
    bellhop_exe: str
    output_root: str
    seed: int = SEED
    water_depth_m: float = 24.0
    receiver_depth_m: float = 20.0
    bottom_speed_mps: float = 1700.0
    bottom_density_gcc: float = 1.8
    bottom_atten_db_per_lambda: float = 0.8
    ray_count: int = 4001
    retry_ray_count: int = 8001
    first_angles: tuple[float, float] = (-89.0, 89.0)
    retry_angles: tuple[float, float] = (-89.9, 89.9)
    arrival_window_s: float = 0.010
    relative_amplitude_threshold: float = 10 ** (-80 / 20)


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()


def pressure_to_depth_0deg(pressure_dbar: np.ndarray) -> np.ndarray:
    p = pressure_dbar.astype(float)
    return (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p / 9.780318


def read_ssp(cfg: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    lines = Path(cfg.ssp_csv).read_text(encoding="utf-8").splitlines()
    data_line = next((i for i, line in enumerate(lines) if line.strip() == "[data]"), None)
    if data_line is None:
        raise ValueError("SSP 文件不含 [data] 段。")
    raw = pd.read_csv(cfg.ssp_csv, skiprows=data_line + 1)
    pressure = pd.to_numeric(raw["Pressure (dBar)"], errors="coerce").to_numpy()
    speed = pd.to_numeric(raw["SV (m/s)"], errors="coerce").to_numpy()
    valid = np.isfinite(pressure) & np.isfinite(speed)
    pressure, speed = pressure[valid], speed[valid]
    end = int(np.argmax(pressure)) + 1  # first maximum: downcast only
    pressure, speed = pressure[:end], speed[:end]
    median5 = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy()
    keep = np.abs(speed - median5) <= 2.0
    removed = int((~keep).sum())
    surface = float(np.median(speed[keep & (pressure >= 0) & (pressure <= 0.5)]))
    depth = pressure_to_depth_0deg(pressure[keep])
    velocity = speed[keep]
    order = np.argsort(depth)
    depth, velocity = depth[order], velocity[order]
    depth, unique_i = np.unique(depth, return_index=True)
    velocity = velocity[unique_i]
    depth = np.r_[0.0, depth, cfg.water_depth_m]
    velocity = np.r_[surface, velocity, velocity[-1]]
    depth, unique_i = np.unique(depth, return_index=True)
    velocity = velocity[unique_i]
    query = np.unique(np.r_[depth, np.arange(0, cfg.water_depth_m + 1e-9, 0.1)])
    c = PchipInterpolator(depth, velocity)(query)
    if not (np.all(np.diff(query) > 0) and np.all(np.isfinite(c)) and np.all((1400 < c) & (c < 1600))):
        raise ValueError("清洗后的 SSP 无法形成合理的 0--24 m PCHIP 剖面。")
    meta = {"downcast_points": end, "removed_outliers": removed, "surface_speed_mps": surface}
    return query, c, meta


def stable_manifest(cfg: Config, output_root: Path) -> pd.DataFrame:
    source = pd.read_csv(cfg.input_csv)
    selected = source[pd.to_numeric(source["SNR_dB"], errors="coerce") >= 90].copy()
    selected = selected.sort_values(["TrainNumber", "PulseNumber"], kind="stable").reset_index(drop=True)
    if len(selected) != 110 or selected["TrainNumber"].nunique() != 69:
        raise ValueError(f"SNR 筛选异常：{len(selected)} 条、{selected['TrainNumber'].nunique()} 个 PulseTrain。")
    if not selected["PulseWavExists"].astype(bool).all() or not selected["PulseWavPath"].map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("筛选清单存在缺失 WAV。")
    rng = np.random.Generator(np.random.MT19937(cfg.seed))
    source_depth = rng.uniform(1, 22, len(selected))
    ranges = rng.uniform(50, 200, len(selected))
    manifest = pd.DataFrame({
        "click_id": [f"click_{i + 1:03d}_train_{int(r.TrainNumber):03d}_pulse_{int(r.PulseNumber):03d}" for i, r in selected.iterrows()],
        "train_number": selected["TrainNumber"].to_numpy(), "pulse_number": selected["PulseNumber"].to_numpy(),
        "snr_db": selected["SNR_dB"].to_numpy(), "input_wav": selected["PulseWavPath"].to_numpy(),
        "source_depth_m": source_depth, "receiver_depth_m": cfg.receiver_depth_m, "range_m": ranges,
        "random_seed": cfg.seed, "status": "pending", "successful_frequency_count": 0,
        "output_path": [str(output_root / "clicks" / click_id) for click_id in [f"click_{i + 1:03d}_train_{int(r.TrainNumber):03d}_pulse_{int(r.PulseNumber):03d}" for i, r in selected.iterrows()]],
        "error_message": pd.Series([""] * len(selected), dtype="object"), "runtime_s": np.nan,
        "first_arrival_s": np.nan, "arrivals_10ms": np.nan, "raw_rms_gain": np.nan, "preview_scale": np.nan,
    })
    return manifest


def validate_wavs(manifest: pd.DataFrame) -> None:
    for path in manifest["input_wav"]:
        info = sf.info(path)
        if info.samplerate != 576000 or info.channels != 1 or info.frames != 116 or info.subtype != "FLOAT":
            raise ValueError(f"WAV 格式不符：{path} ({info.samplerate} Hz, {info.channels} ch, {info.frames} samples, {info.subtype})")


def write_env(path: Path, frequency_hz: float, row: pd.Series, z: np.ndarray, c: np.ndarray,
              cfg: Config, ray_count: int, angles: tuple[float, float], run_type: str = "A") -> None:
    lines = ["'Experiment v2 shallow-water channel'", f"{frequency_hz:.9f}", "1", "'PVWT'",
             f"{len(z)} 0.0 {cfg.water_depth_m:.3f}"]
    lines.extend(f"{depth:.6f} {speed:.6f} /" for depth, speed in zip(z, c))
    lines.extend(["'A' 0.0", f"{cfg.water_depth_m:.3f} {cfg.bottom_speed_mps:.1f} {cfg.bottom_atten_db_per_lambda:.3f} {cfg.bottom_density_gcc:.3f} 0.0 0.0 /",
                  "1", f"{row.source_depth_m:.6f} /", "1", f"{row.receiver_depth_m:.6f} /", "1", f"{row.range_m / 1000:.9f} /",
                  f"'{run_type}'", str(ray_count), f"{angles[0]:.3f} {angles[1]:.3f} /", f"0.10 {cfg.water_depth_m:.3f} {max(0.2, row.range_m / 1000 + 0.01):.6f}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_ascii_arrivals(path: Path, frequency_hz: float) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    tokens = path.read_text(encoding="ascii").split()
    if len(tokens) < 8 or tokens[0] != "'2D'":
        raise ValueError(f"不是有效的 Bellhop 2D ASCII arrivals：{path}")
    cursor = 2  # flag, frequency
    nsz = int(tokens[cursor]); cursor += 1 + nsz
    nrz = int(tokens[cursor]); cursor += 1 + nrz
    nrr = int(tokens[cursor]); cursor += 1 + nrr
    rows = []
    for _ in range(nsz):
        cursor += 1  # maximum number of arrivals
        for _ in range(nrz * nrr):
            narr = int(tokens[cursor]); cursor += 1
            for _ in range(narr):
                magnitude, phase_deg, delay, delay_imag, src_angle, rcv_angle, top, bottom = map(float, tokens[cursor:cursor + 8])
                cursor += 8
                amp = magnitude * np.exp(1j * np.deg2rad(phase_deg))
                effective = amp * np.exp(2 * np.pi * frequency_hz * delay_imag)
                rows.append((frequency_hz, delay, delay_imag, effective, abs(effective), src_angle, rcv_angle, int(top), int(bottom)))
    return pd.DataFrame(rows, columns=["frequency_hz", "delay_s", "delay_imag_s", "complex_gain", "effective_gain", "source_angle_deg", "receiver_angle_deg", "top_bounces", "bottom_bounces"]).sort_values("delay_s", kind="stable")


def run_frequency(task: tuple[int, Path, pd.Series, np.ndarray, np.ndarray, Config]) -> tuple[int, pd.DataFrame, dict]:
    index, click_dir, row, z, c, cfg = task
    frequency_hz = FREQUENCIES_HZ[index]
    directory = click_dir / "bellhop" / f"f_{frequency_hz / 1000:.0f}kHz"
    directory.mkdir(parents=True, exist_ok=True)
    base = f"channel_{frequency_hz / 1000:.0f}kHz"
    env = directory / f"{base}.env"
    log = directory / f"{base}.log"
    status = {"frequency_hz": frequency_hz, "status": "failed", "retried": False, "environment_file": str(env), "arrivals_file": str(directory / f"{base}.arr"), "error_message": ""}
    for retry, rays, angles in ((False, cfg.ray_count, cfg.first_angles), (True, cfg.retry_ray_count, cfg.retry_angles)):
        write_env(env, frequency_hz, row, z, c, cfg, rays, angles)
        proc = subprocess.run([cfg.bellhop_exe, base], cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        if proc.returncode:
            status["error_message"] = f"Bellhop return code {proc.returncode}; see {log}"
            continue
        arrivals = parse_ascii_arrivals(directory / f"{base}.arr", frequency_hz)
        if not arrivals.empty:
            status.update(status="success", retried=retry)
            return index, arrivals, status
        status["error_message"] = "Bellhop completed but returned no arrivals after retry." if retry else "No arrivals on first run."
    return index, pd.DataFrame(), status


def subband_synthesis(x: np.ndarray, fs: int, arrivals: list[pd.DataFrame], first_arrival: float,
                      output_samples: int | None = None) -> np.ndarray:
    """Synthesize bandwise arrivals, optionally returning a shorter safe output.

    ``output_samples`` must include the input click duration plus the desired
    excess-delay window.  The FFT size is still rounded up independently, so
    fractional-delay synthesis cannot wrap into the returned interval.
    """
    if output_samples is None:
        output_samples = round(0.012 * fs)
    n = 1 << math.ceil(math.log2(max(len(x), output_samples)))
    spectrum = np.fft.rfft(np.pad(x, (0, n - len(x))))
    frequency = np.fft.rfftfreq(n, 1 / fs)
    mask = np.maximum(np.cos(np.pi / 2 * (frequency[:, None] - FREQUENCIES_HZ[None, :]) / 10_000), 0)
    mask[(frequency < 20_000) | (frequency > 190_000), :] = 0
    mask /= np.maximum(np.sqrt(np.sum(mask ** 2, axis=1, keepdims=True)), np.finfo(float).eps)
    time_frequency = np.fft.fftfreq(n, 1 / fs)
    received = np.zeros(n, dtype=float)
    for k, table in enumerate(arrivals):
        band = np.fft.irfft(spectrum * mask[:, k], n=n)
        analytic = hilbert(band)
        for arrival in table.itertuples(index=False):
            delay = arrival.delay_s - first_arrival
            delayed = np.fft.ifft(np.fft.fft(analytic) * np.exp(-2j * np.pi * time_frequency * delay))
            received += np.real(arrival.complex_gain * delayed)
    return received[:output_samples]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def process_click(i: int, row: pd.Series, z: np.ndarray, c: np.ndarray, cfg: Config, workers: int) -> dict:
    started = time.monotonic(); click_dir = Path(row.output_path); click_dir.mkdir(parents=True, exist_ok=True)
    x, fs = sf.read(row.input_wav, dtype="float64", always_2d=False)
    tasks = [(k, click_dir, row, z, c, cfg) for k in range(len(FREQUENCIES_HZ))]
    results: list[pd.DataFrame | None] = [None] * len(FREQUENCIES_HZ); statuses: list[dict | None] = [None] * len(FREQUENCIES_HZ)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_frequency, task) for task in tasks]
        for future in as_completed(futures):
            k, arrivals, status = future.result(); results[k], statuses[k] = arrivals, status
    status_frame = pd.DataFrame(statuses)
    atomic_csv(status_frame, click_dir / "frequency_status.csv")
    if any(table is None or table.empty for table in results):
        raise RuntimeError(f"{sum(table is None or table.empty for table in results)} 个频点失败；详见 {click_dir / 'frequency_status.csv'}")
    all_arrivals = [table for table in results if table is not None]
    first = min(float(table.delay_s.min()) for table in all_arrivals)
    gated = []
    for table in all_arrivals:
        threshold = float(table.effective_gain.max()) * cfg.relative_amplitude_threshold
        kept = table[(table.delay_s >= first) & (table.delay_s <= first + cfg.arrival_window_s) & (table.effective_gain >= threshold)].copy()
        if kept.empty:
            raise RuntimeError("到达门限筛选后出现空频带。")
        gated.append(kept)
    rx = subband_synthesis(np.asarray(x, dtype=float), int(fs), gated, first)
    preview_scale = 1.0 / max(float(np.max(np.abs(rx))), np.finfo(float).eps)
    np.save(click_dir / "rx_raw.npy", rx)
    sf.write(click_dir / "rx_preview.wav", rx * preview_scale, int(fs), subtype="FLOAT")
    flat_all = pd.concat(all_arrivals, ignore_index=True); flat_gate = pd.concat(gated, ignore_index=True)
    np.savez_compressed(click_dir / "channel.npz", frequencies_hz=FREQUENCIES_HZ, first_arrival_s=first,
                        all_arrivals=flat_all.to_records(index=False), gated_arrivals=flat_gate.to_records(index=False),
                        ssp_depth_m=z, ssp_speed_mps=c, source_depth_m=row.source_depth_m, receiver_depth_m=row.receiver_depth_m,
                        range_m=row.range_m, random_seed=SEED, config_hash=config_hash(cfg))
    return {"status": "success", "successful_frequency_count": len(FREQUENCIES_HZ), "runtime_s": time.monotonic() - started,
            "first_arrival_s": first, "arrivals_10ms": len(flat_gate), "raw_rms_gain": float(np.sqrt(np.mean(rx ** 2)) / max(np.sqrt(np.mean(x ** 2)), np.finfo(float).eps)),
            "preview_scale": preview_scale, "error_message": ""}


def self_test(cfg: Config) -> None:
    arr = parse_ascii_arrivals(ROOT / "self_test" / "shallow_arrivals.arr", 100_000)
    if arr.empty or not np.all(np.isfinite(arr.delay_s)):
        raise RuntimeError("Bellhop self_test ASCII arrivals 解析失败。")
    print(f"Python arrivals parser: {len(arr)} valid arrivals.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 条，用于验收；省略则运行全部 110 条。")
    args = parser.parse_args()
    cfg = Config(str(DATA_ROOT / "SNR_ge_60dB_click_paths.csv"), str(DATA_ROOT / "浅海信道24m.csv"),
                 str(ROOT / "third_party" / "Acoustics-Toolbox" / "Bellhop" / "bellhop.exe"), str(ROOT / "python_results"))
    if not Path(cfg.bellhop_exe).is_file():
        raise FileNotFoundError("找不到 bellhop.exe；请先运行 setup_bellhop.ps1。")
    if args.self_test:
        self_test(cfg); return 0
    output = Path(cfg.output_root); output.mkdir(parents=True, exist_ok=True); (output / "clicks").mkdir(exist_ok=True)
    cfg_file = output / "config.json"; digest = config_hash(cfg)
    if cfg_file.exists() and json.loads(cfg_file.read_text()).get("hash") != digest:
        raise RuntimeError("配置哈希不一致，拒绝续跑。请移走 python_results 或检查配置。")
    cfg_file.write_text(json.dumps({"hash": digest, "config": asdict(cfg)}, indent=2), encoding="utf-8")
    z, c, ssp_meta = read_ssp(cfg); atomic_csv(pd.DataFrame({"depth_m": z, "sound_speed_mps": c}), output / "clean_ssp.csv")
    manifest = stable_manifest(cfg, output); validate_wavs(manifest)
    manifest_file = output / "click_manifest.csv"
    if args.resume and manifest_file.exists() and not args.force:
        old = pd.read_csv(manifest_file)
        if list(old.click_id) == list(manifest.click_id):
            for column in ["status", "successful_frequency_count", "error_message", "runtime_s"]:
                manifest[column] = old[column]
    manifest["error_message"] = manifest["error_message"].astype("object")
    active = manifest if args.limit is None else manifest.head(args.limit)
    total = len(active) * len(FREQUENCIES_HZ); done = 0; overall = time.monotonic()
    for i, row in active.iterrows():
        click_dir = Path(row.output_path)
        completed = row.status == "success" and (click_dir / "rx_raw.npy").is_file() and (click_dir / "channel.npz").is_file()
        if completed and args.resume and not args.force:
            done += len(FREQUENCIES_HZ); continue
        try:
            result = process_click(i, row, z, c, cfg, args.workers)
        except Exception as exc:  # preserve diagnostic and continue to the next click
            result = {"status": "failed", "successful_frequency_count": 0, "runtime_s": np.nan, "error_message": repr(exc)}
            click_dir.mkdir(parents=True, exist_ok=True); (click_dir / "failure.log").write_text(repr(exc), encoding="utf-8")
        for key, value in result.items(): manifest.loc[i, key] = value
        done += len(FREQUENCIES_HZ); elapsed = time.monotonic() - overall; eta = elapsed / done * (total - done)
        print(f"[{done / total:6.2%}] click {i + 1:3d}/{len(manifest)} | success={sum(manifest.status == 'success')} failed={sum(manifest.status == 'failed')} | elapsed={elapsed / 60:.1f} min ETA={eta / 60:.1f} min", flush=True)
        atomic_csv(manifest, manifest_file)
    atomic_csv(manifest, output / "result_summary.csv")
    print(f"Done. Synthesized signals: {output / 'clicks'}; SSP metadata: {ssp_meta}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
