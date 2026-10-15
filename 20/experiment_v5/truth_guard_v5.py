"""Static truth-isolation guard; this module is not imported by numerical extraction cores."""
from __future__ import annotations

from pathlib import Path


FORBIDDEN = ("event_manifest.csv", "original_click.wav", "original_channel.npz", "event_channels.npz")


def assert_inference_truth_isolation(root: Path) -> None:
    sources = [root / "unrolled_bcsc.py", root / "infer_v5.py", root / "protocol_v5.py"]
    leaked: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        leaked.extend(f"{source.name}:{name}" for name in FORBIDDEN if name in text)
    if leaked:
        raise RuntimeError(f"truth-isolation violation: {leaked}")
