"""Static truth-isolation guard for v7 inference modules."""
from __future__ import annotations

from pathlib import Path


def assert_inference_truth_isolation(root: Path) -> None:
    forbidden = ("event_" + "channels.npz", "original_" + "click.wav", "event_" + "manifest.csv", "original_" + "channel.npz")
    sources = (root / "model_v7.py", root / "infer_v7.py")
    leaked = [f"{path.name}:{name}" for path in sources for name in forbidden if name in path.read_text(encoding="utf-8")]
    if leaked: raise RuntimeError(f"truth-isolation violation: {leaked}")
