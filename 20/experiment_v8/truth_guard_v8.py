"""Static guard ensuring v8 extraction code never references simulator truth assets."""
from __future__ import annotations

from pathlib import Path

FORBIDDEN = ("event_channels" + ".npz", "original_channel" + ".npz", "event_manifest" + ".csv", "original_click" + ".wav")


def assert_extraction_truth_isolation(root: Path) -> None:
    sources = [root / name for name in ("protocol_v8.py", "extractors_v8.py", "model_v8.py", "run_v8.py")]
    leaked = [f"{source.name}:{name}" for source in sources for name in FORBIDDEN if name in source.read_text(encoding="utf-8")]
    if leaked:
        raise RuntimeError(f"truth-isolation violation: {leaked}")
