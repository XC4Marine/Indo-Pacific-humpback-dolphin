#!/usr/bin/env python3
"""Extract v7 physical features without opening any simulation truth asset."""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
import numpy as np
from core_v7 import (PROTOCOL, ROOT, atomic_json, candidate_feature_rows, input_paths, load_segments,
                     load_v4_fit, serialise_features, source_ensemble, write_synthetic_checks)


def assert_truth_isolation() -> None:
    forbidden = ("event_" + "channels.npz", "original_" + "click.wav", "event_" + "manifest.csv", "original_" + "channel.npz")
    sources = (ROOT / "core_v7.py", Path(__file__).resolve())
    leaked = [f"{p.name}:{name}" for p in sources for name in forbidden if name in p.read_text(encoding="utf-8")]
    if leaked: raise RuntimeError(f"truth-isolation violation: {leaked}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int); parser.add_argument("--start", type=int, default=0, help="zero-based inclusive input offset for a resumable shard")
    parser.add_argument("--stop", type=int, help="zero-based exclusive input offset for a resumable shard")
    parser.add_argument("--ensemble", type=int, default=20)
    parser.add_argument("--rebuild-features", action="store_true", help="reuse cached source_ensemble.npz and recompute candidate features")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); assert_truth_isolation()
    if args.ensemble != 20: raise ValueError("protocol requires --ensemble 20")
    all_paths = input_paths(); paths = all_paths[args.start:args.stop]
    if args.limit is not None: paths = paths[:args.limit]
    if not paths: raise ValueError("selected shard contains no clicktrains")
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Shards never share a mutable metadata file.  The deterministic mechanism
    # check is written once by the first shard; feature directories are unique.
    if args.start == 0:
        write_synthetic_checks(args.output_root / "synthetic_mechanism_checks.json")
    manifest_path = args.output_root / f"manifest_{args.start:03d}_{(args.stop if args.stop is not None else 110):03d}.json"
    manifest = []
    for index, path in enumerate(paths, 1):
        train_id = path.parent.name; directory = args.output_root / train_id
        features_path, ensemble_path = directory / "candidate_features.json", directory / "source_ensemble.npz"
        if features_path.exists() and ensemble_path.exists() and not args.force and not args.rebuild_features:
            manifest.append({"train_id": train_id, "status": "reused"}); print(f"[总体进度] {index:03d}/{len(paths)} | {train_id} | reused", flush=True); continue
        try:
            source, channels, starts, config = load_v4_fit(train_id); segments = load_segments(path, starts)
            if args.rebuild_features:
                if not ensemble_path.exists(): raise FileNotFoundError(f"missing cached source ensemble for {train_id}")
                with np.load(ensemble_path, allow_pickle=False) as cached:
                    ensemble_sources, ensemble_channels = np.asarray(cached["sources"]), np.asarray(cached["channels"])
                rows, _ = candidate_feature_rows(train_id, source, channels, segments, ensemble_sources, ensemble_channels)
                directory.mkdir(parents=True, exist_ok=True); atomic_json(features_path, serialise_features(rows, train_id, config))
                manifest.append({"train_id": train_id, "status": "rebuilt", "candidate_count": len(rows)})
                atomic_json(manifest_path, {"protocol": PROTOCOL, "truth_isolation": True, "shard": [args.start, args.stop], "items": manifest})
                print(f"[总体进度] {index:03d}/{len(paths)} | {train_id} | rebuilt", flush=True)
                continue
            def report_refit(done: int, total: int) -> None:
                print(f"[源重估进度] {index:03d}/{len(paths)} | {train_id} | {done:02d}/{total}", flush=True)
            def run_refits():
                return source_ensemble(segments, config, source, train_id, args.ensemble, report_refit)
            print(f"[源重估进度] {index:03d}/{len(paths)} | {train_id} | 00/20", flush=True)
            ensemble_sources, ensemble_channels = run_refits()
            print(f"[源重估进度] {index:03d}/{len(paths)} | {train_id} | 20/20", flush=True)
            rows, payload = candidate_feature_rows(train_id, source, channels, segments, ensemble_sources, ensemble_channels)
            directory.mkdir(parents=True, exist_ok=True); atomic_json(features_path, serialise_features(rows, train_id, config)); np.savez_compressed(ensemble_path, **payload)
            manifest.append({"train_id": train_id, "status": "complete", "candidate_count": len(rows)})
        except Exception as error:
            manifest.append({"train_id": train_id, "status": "failed", "error": str(error)}); traceback.print_exc()
        atomic_json(manifest_path, {"protocol": PROTOCOL, "truth_isolation": True, "shard": [args.start, args.stop], "items": manifest})
        print(f"[总体进度] {index:03d}/{len(paths)} | {train_id} | {manifest[-1]['status']}", flush=True)


if __name__ == "__main__": main()
