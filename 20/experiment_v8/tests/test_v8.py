from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractors_v8 import _bh_fdr, cepstrum_cfar_event, fuse_event_candidates, med_event
from model_v8 import SparseGateNet, gate_loss
from protocol_v8 import CHANNEL_SAMPLES, EVENT_COUNT, SEGMENT_SAMPLES, discover_inputs, fold_rows
from truth_guard_v8 import assert_extraction_truth_isolation


class V8Tests(unittest.TestCase):
    def test_group_folds_are_disjoint_and_complete(self) -> None:
        rows = fold_rows(discover_inputs())
        by_group: dict[str, set[int]] = {}
        for row in rows: by_group.setdefault(str(row["source_group"]), set()).add(int(row["fold"]))
        self.assertEqual(len(rows), 110); self.assertTrue(all(len(values) == 1 for values in by_group.values()))
        self.assertEqual({row["fold"] for row in rows}, set(range(5)))

    def test_fusion_requires_six_event_support(self) -> None:
        events = [[.24] for _ in range(6)] + [[.91] for _ in range(4)]
        stable, mask = fuse_event_candidates(events)
        self.assertEqual(len(stable), 1); self.assertAlmostEqual(stable[0], .24, places=5); self.assertEqual(len(mask), CHANNEL_SAMPLES)

    def test_fdr_handles_noise_and_spike(self) -> None:
        values = np.r_[np.zeros(20), 30.0]
        detected = _bh_fdr(values)
        self.assertEqual(detected.shape, values.shape); self.assertTrue(detected[-1])

    def test_known_echo_is_processed_without_manual_thresholds(self) -> None:
        rng = np.random.default_rng(1); segment = np.zeros(SEGMENT_SAMPLES)
        source = rng.normal(size=116) * np.hanning(116); segment[120:236] = source; segment[350:466] += .5 * source
        expected_ms = 230 / 576_000 * 1e3
        cepstrum = cepstrum_cfar_event(segment, 116); med = med_event(segment, 116)
        self.assertTrue(any(abs(value - expected_ms) < .03 for value in cepstrum))
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in med))

    def test_gate_shape_and_bounded_mask(self) -> None:
        model = SparseGateNet(); values = torch.randn(1, EVENT_COUNT, SEGMENT_SAMPLES)
        result = model(values); loss, _ = gate_loss(result, values)
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(tuple(result["mask"].shape), (1, CHANNEL_SAMPLES))
        self.assertTrue(torch.all(result["mask"] >= 0)); self.assertTrue(torch.all(result["mask"] <= 1)); self.assertEqual(float(result["mask"][0, 0].detach()), 1.0)

    def test_truth_guard(self) -> None:
        assert_extraction_truth_isolation(ROOT)


if __name__ == "__main__":
    unittest.main()
