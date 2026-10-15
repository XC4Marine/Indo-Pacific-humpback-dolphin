"""Shared, truth-free protocol utilities for Experiment 20 v5."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
OUTPUT_ROOT = ROOT / "outputs"
SPLIT_SEED = 20260908
GROUP_RE = re.compile(r"^click_\d+_train_(\d+)_pulse_\d+$")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def discover_inputs() -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if len(paths) != 110:
        raise RuntimeError(f"expected 110 clicktrains, found {len(paths)}")
    return paths


def source_group(path: Path) -> str:
    match = GROUP_RE.match(path.parent.name)
    if not match:
        raise ValueError(f"invalid clicktrain directory name: {path.parent.name}")
    return f"train_{int(match.group(1)):03d}"


def group_split(paths: list[Path]) -> list[dict[str, str]]:
    """Return the locked 48/10/11 source-group split without inspecting truth."""
    groups = sorted({source_group(path) for path in paths})
    if len(groups) != 69:
        raise RuntimeError(f"expected 69 source groups, found {len(groups)}")
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{SPLIT_SEED}:{group}".encode()).hexdigest())
    labels = {group: "train" for group in ranked[:48]}
    labels.update({group: "validation" for group in ranked[48:58]})
    labels.update({group: "test" for group in ranked[58:]})
    rows = [{"train_id": path.parent.name, "source_group": source_group(path), "split": labels[source_group(path)]}
            for path in paths]
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    if any(not counts[name] for name in counts):
        raise RuntimeError(f"an empty split was generated: {counts}")
    return rows


def config_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
