#!/usr/bin/env python3
"""Generate reproducible perturbed click trains from the final v2 channels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from run_experiment_v2 import FREQUENCIES_HZ, subband_synthesis


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CLICKTRAINS = REPO_ROOT / "00_Data" / "ClickTrains"
SOURCE_ROOT = ROOT / "python_results_delay4ms_top3_extra10dB_hf30dB_direct10dB"
OUTPUT_ROOT = ROOT / "clicktrains_10click_perturbed"
SEED = 20260907


@dataclass(frozen=True)
class TrainConfig:
    source_root: str
    output_root: str
    click_count: int = 10
    seed: int = SEED
    source_amplitude_jitter_db: float = 1.0
    ipi_jitter_fraction: float = 0.05
    reflection_gain_jitter_db: float = 1.0
    reflection_delay_jitter_s: float = 10e-6
    minimum_direct_over_reflection_db: float = 10.0


def digest_config(config: TrainConfig, source_hash: str) -> str:
    payload = {"config": asdict(config), "source_config_hash": source_hash}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    temporary = path.with_name(path.stem + ".tmp.wav")
    sf.write(temporary, data, sample_rate, subtype="FLOAT")
    os.replace(temporary, path)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    """Materialize a named artifact without duplicating same-volume waveform data."""
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        import shutil
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def materialize_source_assets(row: pd.Series) -> None:
    """Add explicitly named train, source click, and source-channel files."""
    directory = Path(row.output_path)
    base_directory = Path(row.base_output_path)
    required = {
        "pulsetrain_raw.npy": directory / "clicktrain_raw.npy",
        "pulsetrain_preview.wav": directory / "clicktrain_preview.wav",
        "original_click.wav": Path(row.input_wav),
        "original_channel.npz": base_directory / "channel.npz",
    }
    for name, source in required.items():
        if not source.is_file():
            raise FileNotFoundError(f"无法实体化所需文件：{source}")
        hardlink_or_copy(source, directory / name)


def click_seed(click_id: str, seed: int) -> int:
    token = hashlib.sha256(f"{seed}:clicktrain:{click_id}".encode("utf-8")).digest()
    return int.from_bytes(token[:8], "little") % (2 ** 32)


def load_ipi_ms(train_number: int) -> np.ndarray:
    path = CLICKTRAINS / f"PulseTrain_{train_number:03d}" / "PulseParameters.txt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 IPI 参考文件：{path}")
    table = pd.read_csv(path, sep="\t")
    ipi_ms = pd.to_numeric(table["IPI(ms)"], errors="coerce").to_numpy(dtype=float)
    ipi_ms = ipi_ms[np.isfinite(ipi_ms) & (ipi_ms > 0)]
    if not len(ipi_ms):
        raise ValueError(f"IPI 参考文件没有正的 IPI：{path}")
    return ipi_ms


def is_direct(frame: pd.DataFrame) -> pd.Series:
    return (frame["top_bounces"] == 0) & (frame["bottom_bounces"] == 0)


def perturb_arrivals(base: pd.DataFrame, first_arrival_s: float, rng: np.random.Generator,
                     config: TrainConfig) -> pd.DataFrame:
    """Perturb reflected paths and then restore the direct-path advantage."""
    event = base.copy()
    direct_mask = is_direct(event)
    reflected_mask = ~direct_mask
    gain_jitter_db = np.zeros(len(event), dtype=float)
    delay_jitter_s = np.zeros(len(event), dtype=float)
    reflection_constraint_db = np.zeros(len(event), dtype=float)
    gain_jitter_db[reflected_mask.to_numpy()] = rng.uniform(-config.reflection_gain_jitter_db,
                                                            config.reflection_gain_jitter_db,
                                                            reflected_mask.sum())
    delay_jitter_s[reflected_mask.to_numpy()] = rng.uniform(-config.reflection_delay_jitter_s,
                                                            config.reflection_delay_jitter_s,
                                                            reflected_mask.sum())
    event.loc[reflected_mask, "delay_s"] = np.clip(
        event.loc[reflected_mask, "delay_s"].to_numpy() + delay_jitter_s[reflected_mask.to_numpy()],
        first_arrival_s, first_arrival_s + 0.004,
    )
    gain_scale = 10 ** (gain_jitter_db / 20)
    event["complex_gain"] *= gain_scale
    event["effective_gain"] *= gain_scale
    required_ratio = 10 ** (config.minimum_direct_over_reflection_db / 20)
    for frequency_hz in FREQUENCIES_HZ:
        mask = np.isclose(event["frequency_hz"], frequency_hz)
        local_direct = event[mask & direct_mask]
        if local_direct.empty:
            raise RuntimeError(f"{frequency_hz / 1000:g} kHz 缺少直达径。")
        direct_gain = float(local_direct["effective_gain"].max())
        local_reflected = event[mask & reflected_mask]
        if local_reflected.empty:
            continue
        allowed_gain = direct_gain / required_ratio
        local_scale = np.minimum(1.0, allowed_gain / local_reflected["effective_gain"].to_numpy())
        event.loc[local_reflected.index, "complex_gain"] *= local_scale
        event.loc[local_reflected.index, "effective_gain"] *= local_scale
        reflection_constraint_db[local_reflected.index.to_numpy()] = -20 * np.log10(local_scale)
    event["reflection_gain_jitter_db"] = gain_jitter_db
    event["reflection_delay_jitter_s"] = delay_jitter_s
    event["direct_constraint_extra_attenuation_db"] = reflection_constraint_db
    return event


def event_tables(arrivals: pd.DataFrame) -> list[pd.DataFrame]:
    tables = [arrivals[np.isclose(arrivals["frequency_hz"], frequency_hz)].copy() for frequency_hz in FREQUENCIES_HZ]
    if any(table.empty for table in tables):
        raise RuntimeError("事件扰动后出现空频带。")
    return tables


def generate_train(row: pd.Series, config: TrainConfig) -> dict:
    started = time.monotonic()
    directory = Path(row.output_path)
    directory.mkdir(parents=True, exist_ok=True)
    base_directory = Path(row.base_output_path)
    source, sample_rate = sf.read(row.input_wav, dtype="float64", always_2d=False)
    ipi_reference = load_ipi_ms(int(row.train_number))
    event_seed = click_seed(str(row.click_id), config.seed)
    rng = np.random.Generator(np.random.MT19937(event_seed))
    with np.load(base_directory / "channel.npz", allow_pickle=True) as channel:
        base_arrivals = pd.DataFrame.from_records(channel["synthesis_arrivals"])
        first_arrival_s = float(channel["first_arrival_s"])
    event_waveforms: list[np.ndarray] = []
    event_arrivals: list[np.ndarray] = []
    event_rows = []
    ipi_raw = rng.choice(ipi_reference, size=config.click_count - 1, replace=True)
    ipi_jitter_fraction = rng.uniform(-config.ipi_jitter_fraction, config.ipi_jitter_fraction, config.click_count - 1)
    ipi_perturbed = ipi_raw * (1.0 + ipi_jitter_fraction)
    start_samples = np.zeros(config.click_count, dtype=int)
    start_samples[1:] = np.rint(np.cumsum(ipi_perturbed) * sample_rate / 1e3).astype(int)
    for event_index in range(config.click_count):
        source_gain_db = float(rng.uniform(-config.source_amplitude_jitter_db, config.source_amplitude_jitter_db))
        source_scale = 10 ** (source_gain_db / 20)
        arrivals = perturb_arrivals(base_arrivals, first_arrival_s, rng, config)
        tables = event_tables(arrivals)
        max_delay_samples = int(np.ceil(max(float(table["delay_s"].max()) - first_arrival_s for table in tables) * sample_rate))
        waveform = subband_synthesis(source * source_scale, int(sample_rate), tables, first_arrival_s,
                                     len(source) + max_delay_samples)
        event_waveforms.append(waveform)
        event_arrivals.append(arrivals.to_records(index=False))
        reflection = arrivals[~is_direct(arrivals)]
        event_rows.append({
            "event_index": event_index + 1,
            "start_sample": int(start_samples[event_index]),
            "start_time_ms": float(start_samples[event_index] / sample_rate * 1e3),
            "ipi_reference_ms": np.nan if event_index == 0 else float(ipi_raw[event_index - 1]),
            "ipi_jitter_fraction": np.nan if event_index == 0 else float(ipi_jitter_fraction[event_index - 1]),
            "ipi_perturbed_ms": np.nan if event_index == 0 else float(ipi_perturbed[event_index - 1]),
            "source_gain_db": source_gain_db,
            "max_reflection_constraint_extra_attenuation_db": float(reflection["direct_constraint_extra_attenuation_db"].max()) if not reflection.empty else 0.0,
        })
    total_samples = max(start + len(waveform) for start, waveform in zip(start_samples, event_waveforms))
    clicktrain = np.zeros(total_samples, dtype=float)
    for start, waveform in zip(start_samples, event_waveforms):
        clicktrain[start:start + len(waveform)] += waveform
    preview_scale = 1.0 / max(float(np.max(np.abs(clicktrain))), np.finfo(float).eps)
    np.save(directory / "clicktrain_raw.npy", clicktrain)
    atomic_wav(directory / "clicktrain_preview.wav", clicktrain * preview_scale, int(sample_rate))
    atomic_csv(pd.DataFrame(event_rows), directory / "event_manifest.csv")
    event_lengths = np.asarray([len(waveform) for waveform in event_waveforms], dtype=int)
    event_rx_raw = np.zeros((config.click_count, int(event_lengths.max())), dtype=float)
    for event_index, waveform in enumerate(event_waveforms):
        event_rx_raw[event_index, :len(waveform)] = waveform
    atomic_npz(directory / "event_channels.npz", event_synthesis_arrivals=np.stack(event_arrivals),
               event_rx_raw=event_rx_raw, event_rx_lengths=event_lengths, event_start_samples=start_samples,
               ipi_reference_ms=ipi_raw, ipi_jitter_fraction=ipi_jitter_fraction,
               ipi_perturbed_ms=ipi_perturbed, train_seed=event_seed,
               base_channel_path=str(base_directory / "channel.npz"), first_arrival_s=first_arrival_s)
    materialize_source_assets(row)
    return {"status": "success", "event_count": config.click_count, "duration_s": total_samples / sample_rate,
            "preview_scale": preview_scale, "runtime_s": time.monotonic() - started, "error_message": ""}


def source_manifest(source_root: Path, output_root: Path) -> tuple[pd.DataFrame, str]:
    manifest = pd.read_csv(source_root / "click_manifest.csv")
    successful = manifest[manifest["status"] == "success"].copy().reset_index(drop=True)
    if len(successful) != 110:
        raise ValueError(f"期望 110 条源信道，实际为 {len(successful)}。")
    source_hash = str(json.loads((source_root / "config.json").read_text(encoding="utf-8"))["hash"])
    successful["base_output_path"] = successful["output_path"]
    successful["output_path"] = [str(output_root / "clicktrains" / click_id) for click_id in successful["click_id"]]
    successful["status"] = "pending"
    successful["event_count"] = 0
    successful["duration_s"] = np.nan
    successful["preview_scale"] = np.nan
    successful["runtime_s"] = np.nan
    successful["error_message"] = pd.Series([""] * len(successful), dtype="object")
    return successful, source_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--materialize-assets-only", action="store_true", help="仅补充具名 waveform、原始 click 与原始信道文件。")
    args = parser.parse_args()
    config = TrainConfig(str(args.source_root.resolve()), str(args.output_root.resolve()))
    manifest, source_hash = source_manifest(args.source_root, args.output_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "clicktrains").mkdir(exist_ok=True)
    digest = digest_config(config, source_hash)
    config_file = output_root / "config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")).get("hash") != digest:
        raise RuntimeError("目标 clicktrain 目录的配置哈希不一致，拒绝续跑。")
    config_file.write_text(json.dumps({"hash": digest, "config": asdict(config), "source_config_hash": source_hash}, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_file = output_root / "clicktrain_manifest.csv"
    if args.resume and manifest_file.is_file() and not args.force:
        old = pd.read_csv(manifest_file)
        if list(old.get("click_id", [])) == list(manifest["click_id"]):
            for column in ["status", "event_count", "duration_s", "preview_scale", "runtime_s", "error_message"]:
                if column in old:
                    manifest[column] = old[column]
    active = manifest if args.limit is None else manifest.head(args.limit)
    overall = time.monotonic()
    for completed, (index, row) in enumerate(active.iterrows(), start=1):
        directory = Path(row.output_path)
        if args.materialize_assets_only:
            if row.status != "success":
                raise RuntimeError(f"{row.click_id} 尚未成功生成，不能仅实体化资产。")
            materialize_source_assets(row)
            print(f"[{completed / len(active):6.2%}] assets {completed:3d}/{len(active)}", flush=True)
            continue
        finished = row.status == "success" and (directory / "clicktrain_raw.npy").is_file() and (directory / "event_channels.npz").is_file()
        if not (finished and args.resume and not args.force):
            try:
                result = generate_train(row, config)
            except Exception as exc:
                result = {"status": "failed", "event_count": 0, "duration_s": np.nan, "preview_scale": np.nan,
                          "runtime_s": np.nan, "error_message": repr(exc)}
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "failure.log").write_text(repr(exc), encoding="utf-8")
            for key, value in result.items():
                manifest.loc[index, key] = value
        elapsed = time.monotonic() - overall
        eta = elapsed / completed * (len(active) - completed)
        print(f"[{completed / len(active):6.2%}] train {completed:3d}/{len(active)} | success={sum(manifest.status == 'success')} failed={sum(manifest.status == 'failed')} | elapsed={elapsed:.1f} s ETA={eta:.1f} s", flush=True)
        atomic_csv(manifest, manifest_file)
    atomic_csv(manifest, output_root / "result_summary.csv")
    print(f"Done. Click trains: {output_root / 'clicktrains'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
