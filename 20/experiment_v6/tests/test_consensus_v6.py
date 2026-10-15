from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from consensus_v6 import (EVENT_COUNT, CHANNEL_LENGTH, FS, SEGMENT_LENGTH, build_candidates, bic_select, finalise_paths,
                          reconstruct, summarise_candidates, wilson_lower)


def path(sample: int, delay_ms: float, amplitude: float = 1.0) -> dict[str, float]:
    return {"sample": sample, "delay_ms": delay_ms, "coefficient": amplitude, "relative_amplitude": abs(amplitude)}


class ConsensusV6Tests(unittest.TestCase):
    def test_cluster_and_six_of_ten_rule(self) -> None:
        paths = []
        for event in range(EVENT_COUNT):
            rows = [path(80 + event, 0.0)]
            if event < 6:
                rows.append(path(260 + event, .300 + (event - 3) * .002))
            if event < 5:
                rows.append(path(450 + event, .600))
            paths.append(rows)
        candidates = build_candidates(paths)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[0]["is_direct"])
        self.assertEqual(candidates[1]["event_detected_count"], 6)
        self.assertAlmostEqual(candidates[1]["event_coverage"], .6)

    def test_joint_confidence_and_wilson_lower(self) -> None:
        candidates = [{"is_direct": True, "delay_median_ms": 0.0, "event_coverage": 1.0, "event_detected_count": 10},
                      {"is_direct": False, "delay_median_ms": .3, "event_coverage": .6, "event_detected_count": 6}]
        detected = np.zeros((10, 2, EVENT_COUNT), dtype=bool)
        detected[:, 0, :] = True
        detected[:7, 1, :6] = True
        bootstrap = {"detected": detected, "delay_ms": np.full((10, 2, EVENT_COUNT), .3), "relative_amplitude": np.ones((10, 2, EVENT_COUNT))}
        summarise_candidates(candidates, bootstrap, 10)
        self.assertAlmostEqual(candidates[1]["joint_confidence"], .7)
        self.assertAlmostEqual(candidates[1]["wilson_lower_95"], wilson_lower(7, 10))
        self.assertLess(candidates[1]["wilson_lower_95"], .7)

    def test_bic_selects_sparse_reflection_and_refits_amplitudes(self) -> None:
        source = np.zeros(116); source[20] = 1.0; source[21] = -.4
        direct, reflected = 80, 300
        channels = np.zeros((EVENT_COUNT, CHANNEL_LENGTH))
        channels[:, direct] = 1.0
        channels[:, reflected] = .3
        segments = np.stack([reconstruct(source, channel, SEGMENT_LENGTH) for channel in channels])
        direct_candidate = {"path_id": 1, "is_direct": True, "delay_median_ms": 0.0, "joint_confidence": 1.0, "event_coverage": 1.0, "event_detected_count": 10, "event_paths": {event: path(direct, 0.0) for event in range(EVENT_COUNT)}}
        reflection = {"path_id": 2, "is_direct": False, "delay_median_ms": (reflected - direct) / FS * 1e3, "joint_confidence": .9, "event_coverage": 1.0, "event_detected_count": 10, "event_paths": {event: path(reflected, (reflected - direct) / FS * 1e3, .3) for event in range(EVENT_COUNT)}}
        selected, _, coefficients, _ = bic_select(source, segments, [direct_candidate, reflection])
        final = finalise_paths(selected, [direct_candidate, reflection], coefficients)
        self.assertEqual([row["candidate_path_id"] for row in final], [1, 2])
        self.assertAlmostEqual(final[1]["relative_amplitude_median"], .3, places=8)

    def test_extraction_entry_has_no_truth_asset_literals(self) -> None:
        source = (ROOT / "consensus_v6.py").read_text(encoding="utf-8")
        for name in ("event_channels.npz", "original_click.wav", "event_manifest.csv", "original_channel.npz"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
