from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcsc_v4 import (CHANNEL_LENGTH, FS, SEGMENT_LENGTH, SOURCE_LENGTH, FitConfig, detect_event_segments, fit_segments, fit_single_segment, measure_3db_bandwidth, project_bandwidth, reconstruct)
from bootstrap_high_confidence_v4 import adaptive_bic_paths
from run_noise_robustness_v4 import add_awgn, noise_seed


class BCSCV4Tests(unittest.TestCase):
    def synthetic_source(self) -> np.ndarray:
        time = (np.arange(SOURCE_LENGTH) - 57) / FS
        source = np.cos(2 * np.pi * 100_000 * time) * np.exp(-(time / 18e-6) ** 2)
        return project_bandwidth(source)

    def test_projection_enforces_length_norm_and_bandwidth(self) -> None:
        source = self.synthetic_source()
        bandwidth, _, _, _ = measure_3db_bandwidth(source)
        self.assertEqual(len(source), SOURCE_LENGTH)
        self.assertAlmostEqual(float(np.linalg.norm(source)), 1.0, places=8)
        self.assertGreaterEqual(bandwidth, 6_000)
        self.assertLessEqual(bandwidth, 72_000)

    def test_detector_uses_waveform_only(self) -> None:
        source = self.synthetic_source()
        train = np.zeros(70_000)
        for index, start in enumerate(range(500, 60_000, 6_000)):
            channel = np.zeros(CHANNEL_LENGTH); channel[80] = 1.0; channel[260] = .15
            waveform = reconstruct(source, channel, SEGMENT_LENGTH)
            train[start:start + len(waveform)] += waveform
        segments, starts = detect_event_segments(train)
        self.assertEqual(segments.shape, (10, SEGMENT_LENGTH))
        self.assertEqual(len(starts), 10)

    def test_two_solvers_return_constrained_sources(self) -> None:
        source = self.synthetic_source()
        channels = np.zeros((10, CHANNEL_LENGTH))
        for event in range(10):
            channels[event, 80] = 1.0 + event * .01
            channels[event, 260 + (event % 3)] = .15
        segments = np.stack([reconstruct(source, channel, SEGMENT_LENGTH) for channel in channels])
        for config in (FitConfig(outer_iterations=5, code_iterations=15, source_iterations=2), FitConfig(group_ratio=.5, outer_iterations=5, admm_iterations=4, source_iterations=2)):
            result = fit_segments(segments, config)
            self.assertEqual(len(result["source"]), SOURCE_LENGTH)
            self.assertTrue(result["constraint"]["bandwidth_pass"])
            self.assertEqual(result["channels"].shape, (10, CHANNEL_LENGTH))

    def test_single_click_has_no_cross_event_state(self) -> None:
        source = self.synthetic_source(); channel = np.zeros(CHANNEL_LENGTH); channel[80] = 1.0; channel[260] = .15
        result = fit_single_segment(reconstruct(source, channel, SEGMENT_LENGTH), FitConfig(outer_iterations=5, code_iterations=15, source_iterations=2))
        self.assertEqual(result["channels"].shape, (1, CHANNEL_LENGTH))
        self.assertEqual(len(result["source"]), SOURCE_LENGTH)
        self.assertTrue(result["constraint"]["bandwidth_pass"])

    def test_awgn_is_deterministic_and_rms_calibrated(self) -> None:
        clicktrain = np.linspace(-2.0, 2.0, 20_000)
        seed = noise_seed("synthetic_train", -5)
        first, metadata = add_awgn(clicktrain, -5, seed)
        second, _ = add_awgn(clicktrain, -5, seed)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(metadata["realised_snr_db"]), -5.0, places=9)

    def test_adaptive_bic_selects_known_sparse_support_without_truth(self) -> None:
        source = self.synthetic_source()
        direct, reflection = 80, 330
        channel = np.zeros(CHANNEL_LENGTH); channel[direct] = 1.0; channel[reflection] = .30
        segment = reconstruct(source, channel, SEGMENT_LENGTH)
        candidates = [
            {"path_index": 0, "sample": direct, "delay_ms": 0.0, "delay_median_ms": 0.0, "relative_amplitude": 1.0, "relative_amplitude_median": 1.0, "coefficient_median": 1.0, "confidence": 1.0, "wilson_lower_95": .9, "is_direct": True},
            {"path_index": 1, "sample": reflection, "delay_ms": (reflection - direct) / FS * 1e3, "delay_median_ms": (reflection - direct) / FS * 1e3, "relative_amplitude": .30, "relative_amplitude_median": .30, "coefficient_median": .30, "confidence": .75, "wilson_lower_95": .6, "is_direct": False},
            {"path_index": 2, "sample": 700, "delay_ms": (700 - direct) / FS * 1e3, "delay_median_ms": (700 - direct) / FS * 1e3, "relative_amplitude": .02, "relative_amplitude_median": .02, "coefficient_median": .02, "confidence": .20, "wilson_lower_95": .1, "is_direct": False},
        ]
        selected, _ = adaptive_bic_paths(candidates, source, segment)
        self.assertEqual([item["sample"] for item in selected], [direct, reflection])


if __name__ == "__main__":
    unittest.main()
