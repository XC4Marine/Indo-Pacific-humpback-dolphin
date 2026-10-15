from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol_v5 import discover_inputs, group_split, source_group
from truth_guard_v5 import assert_inference_truth_isolation
from unrolled_bcsc import CHANNEL_LENGTH, EVENT_COUNT, SEGMENT_LENGTH, SOURCE_LENGTH, UnrolledBCSC, causal_fft_convolution

V4_ROOT = ROOT.parent / "experiment_v4"
sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import canonicalise, measure_3db_bandwidth


class UnrolledBCSCTests(unittest.TestCase):
    def test_fft_convolution_matches_scipy(self) -> None:
        rng = np.random.default_rng(14); source = rng.normal(size=(1, SOURCE_LENGTH)).astype(np.float32)
        channels = rng.normal(size=(1, EVENT_COUNT, CHANNEL_LENGTH)).astype(np.float32)
        actual = causal_fft_convolution(torch.from_numpy(source), torch.from_numpy(channels)).numpy()[0]
        expected = np.stack([fftconvolve(source[0], channel) for channel in channels[0]])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_six_stage_forward_and_gradients_are_finite(self) -> None:
        torch.manual_seed(15); segments = torch.randn(1, EVENT_COUNT, SEGMENT_LENGTH, requires_grad=True)
        model = UnrolledBCSC(6, learnable=True); outputs = model(segments)
        self.assertEqual(tuple(outputs["source"].shape), (1, SOURCE_LENGTH))
        self.assertEqual(tuple(outputs["channels"].shape), (1, EVENT_COUNT, CHANNEL_LENGTH))
        self.assertEqual(tuple(outputs["reconstructions"].shape), (1, EVENT_COUNT, SEGMENT_LENGTH))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(outputs["source"], dim=1), torch.ones(1), atol=1e-5))
        outputs["reconstructions"].square().mean().backward()
        self.assertTrue(torch.isfinite(segments.grad).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_learned_scalars_stay_positive(self) -> None:
        values = UnrolledBCSC(6, learnable=True).positive_parameters()
        self.assertTrue(all(torch.all(item > 0) for item in values.values()))

    def test_group_split_is_complete_and_has_no_source_leakage(self) -> None:
        paths = discover_inputs(); rows = group_split(paths)
        self.assertEqual(len(rows), 110)
        labels: dict[str, set[str]] = {}
        for row in rows: labels.setdefault(row["source_group"], set()).add(row["split"])
        self.assertEqual(len(labels), 69); self.assertTrue(all(len(value) == 1 for value in labels.values()))
        self.assertEqual({name: sum(name in values for values in labels.values()) for name in ("train", "validation", "test")},
                         {"train": 48, "validation": 10, "test": 11})
        self.assertEqual({name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")},
                         {"train": 80, "validation": 14, "test": 16})
        self.assertEqual(source_group(paths[0]), next(row["source_group"] for row in rows if row["train_id"] == paths[0].parent.name))

    def test_inference_sources_are_truth_isolated(self) -> None:
        assert_inference_truth_isolation(ROOT)

    def test_inference_canonicalisation_enforces_source_constraints(self) -> None:
        torch.manual_seed(16); output = UnrolledBCSC(6)(torch.randn(1, EVENT_COUNT, SEGMENT_LENGTH))
        source, _ = canonicalise(output["source"][0].detach().numpy(), output["channels"][0].detach().numpy())
        bandwidth, _, _, _ = measure_3db_bandwidth(source)
        self.assertAlmostEqual(float(np.linalg.norm(source)), 1.0, places=7)
        self.assertGreaterEqual(bandwidth, 6_000); self.assertLessEqual(bandwidth, 72_000)


if __name__ == "__main__":
    unittest.main()
