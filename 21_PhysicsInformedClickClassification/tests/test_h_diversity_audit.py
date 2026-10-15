#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h 多样性审计的单元、数据契约和统计回归测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = EXPERIMENT_DIR / "h_diversity_audit.py"
SPEC = importlib.util.spec_from_file_location("h_diversity_audit", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class DescriptorTests(unittest.TestCase):
    def test_identical_edp_has_zero_hellinger_distance(self) -> None:
        rng = np.random.default_rng(7)
        response = rng.normal(size=(8, 1153))
        response[:, 0] = 1.0
        descriptor = audit.energy_delay_descriptor(response)
        np.testing.assert_allclose(audit.hellinger(descriptor, descriptor), 0.0, atol=1e-15)

    def test_unit_impulse_is_tail_collapse(self) -> None:
        response = np.zeros((3, 1153), dtype=np.float64)
        response[:, 0] = 1.0
        ratio, collapsed = audit.collapse_diagnostics(response)
        np.testing.assert_array_equal(ratio, 0.0)
        self.assertTrue(collapsed.all())

    def test_own_channel_swap_equals_normal_reconstruction_error(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(4, 5761)).astype(np.float32)
        response = np.zeros((4, 1153), dtype=np.float32)
        response[:, 0] = 1.0
        response[:, 17] = 0.12
        response[:, 101] = -0.04
        reconstruction = audit.causal_fft_convolution(
            torch.from_numpy(source), torch.from_numpy(response)
        ).numpy()
        observed = audit.relative_nrmse_batch(reconstruction, source, response)
        numerator = np.sum((reconstruction - reconstruction) ** 2, axis=1)
        denominator = np.sum(reconstruction**2, axis=1)
        expected = np.sqrt(numerator / denominator)
        np.testing.assert_allclose(observed, expected, atol=1e-7)

    def test_output_directory_cannot_contain_checkpoint_directory(self) -> None:
        config = audit.AuditConfig(
            mode="smoke",
            checkpoint_output=EXPERIMENT_DIR / "outputs_snr_m15_p15_full_5seeds",
            output_dir=EXPERIMENT_DIR,
            seeds=(20260826,),
            bootstrap_runs=10,
            permutation_runs=10,
            device="cpu",
            force=True,
        )
        with self.assertRaisesRegex(ValueError, "上级目录"):
            audit.prepare_output(config)


def synthetic_pair_frame(value: float, samples: int = 120) -> pd.DataFrame:
    rows = []
    for condition in ("recon_bce", "recon_physics_bce"):
        for index in range(samples):
            rows.append(
                {
                    "condition": condition,
                    "seed": 20260826 + index % 5,
                    "recording_id": index % 10,
                    "mix_noise_type": f"noise_{index % 2}",
                    "snr_bin_low_db": float(-15 + 5 * (index % 6)),
                    "m_train": value,
                    "m_noise": value,
                    "m_swap": value,
                }
            )
    return pd.DataFrame(rows)


class StatisticalTests(unittest.TestCase):
    def make_config(self, runs: int = 199) -> audit.AuditConfig:
        return audit.AuditConfig(
            mode="full",
            checkpoint_output=EXPERIMENT_DIR / "outputs_snr_m15_p15_full_5seeds",
            output_dir=EXPERIMENT_DIR / ".test_h_audit_unused",
            seeds=audit.EXPECTED_SEEDS,
            bootstrap_runs=runs,
            permutation_runs=runs,
            device="cpu",
            force=False,
        )

    def test_fixed_permutation_seed_is_exactly_reproducible(self) -> None:
        frame = synthetic_pair_frame(0.2).query("condition == 'recon_bce'")
        first = audit.stratified_sign_permutation(frame, "m_train", 50, np.random.default_rng(1234))
        second = audit.stratified_sign_permutation(frame, "m_train", 50, np.random.default_rng(1234))
        np.testing.assert_array_equal(first, second)

    def test_all_identical_responses_fail_main_acceptance(self) -> None:
        summary = audit.summarize_main_metrics(synthetic_pair_frame(0.0), self.make_config())
        self.assertFalse(summary["passed"].any())
        self.assertTrue((summary["ci_low"] == 0.0).all())

    def test_train_stable_between_train_distinct_responses_pass(self) -> None:
        summary = audit.summarize_main_metrics(synthetic_pair_frame(0.25), self.make_config())
        self.assertTrue(summary["passed"].all())
        self.assertTrue((summary["ci_low"] > 0.0).all())
        self.assertTrue((summary["bootstrap_runs_valid"] == 199).all())
        self.assertTrue((summary["permutation_runs_valid"] == 199).all())

    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = audit.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.5})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertEqual(adjusted["c"], 0.5)


class RealDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = EXPERIMENT_DIR / "outputs_snr_m15_p15_full_5seeds" / "dataset_manifest.csv"
        if not path.exists():
            raise unittest.SkipTest(f"缺少真实低SNR清单：{path}")
        cls.manifest = audit.parse_sample_ids(pd.read_csv(path))
        cls.pairs = audit.build_pair_manifest(cls.manifest.query("label == 1"))

    def test_fixed_click_train_counts(self) -> None:
        checks = audit.validate_dataset_contract(self.manifest, self.pairs)
        self.assertEqual(checks["train_count"], 640)
        self.assertEqual(checks["multi_click_train_count"], 475)
        self.assertEqual(checks["comparable_click_count"], 3335)
        self.assertEqual(checks["matched_click_count"], 3285)
        self.assertAlmostEqual(checks["matched_coverage"], 3285 / 3335)

    def test_pairing_never_crosses_test_fold(self) -> None:
        checks = audit.validate_dataset_contract(self.manifest, self.pairs)
        self.assertEqual(checks["same_pair_fold_mismatch_count"], 0)
        self.assertEqual(checks["different_pair_fold_mismatch_count"], 0)
        self.assertTrue(checks["passed"])

    def test_pair_keys_are_unique(self) -> None:
        self.assertFalse(self.pairs["sample_id"].duplicated().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
