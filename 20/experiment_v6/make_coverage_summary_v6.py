#!/usr/bin/env python3
"""Render a v6 summary that treats original ten-event coverage as confidence.

No bootstrap samples are read or recomputed.  The script joins the already
truth-evaluated correctness labels with each final path's original event coverage.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluate_v6 import EVALUATION_ROOT, OUTPUT_ROOT, summary_figure, write_csv


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    correctness = read_csv(EVALUATION_ROOT / "confidence_calibration.csv")
    coverage: dict[tuple[str, int], float] = {}
    for artifact in OUTPUT_ROOT.glob("*/consensus.json"):
        result = json.loads(artifact.read_text(encoding="utf-8"))
        for path in result["final_paths"]:
            coverage[(str(result["train_id"]), int(path["path_id"]))] = float(path["event_coverage"])
    calibration = []
    for row in correctness:
        key = (str(row["train_id"]), int(row["path_id"]))
        if key not in coverage:
            raise KeyError(f"missing final-path coverage for {key}")
        calibration.append({"train_id": key[0], "path_id": key[1], "coverage_confidence": coverage[key], "correct": int(row["correct"])})
    train_rows = []
    for row in read_csv(EVALUATION_ROOT / "train_metrics.csv"):
        train_rows.append({key: float(value) if key not in {"train_id"} else value for key, value in row.items()})
    brier = sum((row["coverage_confidence"] - row["correct"]) ** 2 for row in calibration) / len(calibration)
    write_csv(EVALUATION_ROOT / "coverage_confidence_calibration.csv", calibration)
    summary_figure(train_rows, [{"confidence": row["coverage_confidence"], "correct": row["correct"]} for row in calibration],
                   EVALUATION_ROOT / "figures" / "summary_coverage_confidence.png",
                   confidence_xlabel="事件覆盖率（作为置信度）", calibration_title="覆盖率置信度校准", calibration_label="v6 覆盖率")
    print(f"覆盖率置信度 Brier score: {brier:.6f} | paths={len(calibration)}")


if __name__ == "__main__":
    main()
