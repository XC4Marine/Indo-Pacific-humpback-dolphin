#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无残差特征消融的结构、梯度、损失和初始化种子测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("click_pipeline_no_residual_tests", PIPELINE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class NoResidualAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260826)
        self.model = pipeline.PhysicalClassifier(include_residual=False)
        self.waveform = torch.from_numpy(np.random.default_rng(20260826).normal(
            size=(2, 5761)).astype(np.float32))
        self.labels = torch.tensor([0.0, 1.0])

    def test_classifier_is_64_to_64_to_1_without_residual_encoder(self) -> None:
        self.assertFalse(hasattr(self.model, "residual_classifier"))
        self.assertEqual(self.model.classification_head[0].in_features, 64)
        self.assertEqual(self.model.classification_head[0].out_features, 64)
        self.assertEqual(self.model.classification_head[-1].out_features, 1)

    def test_residual_is_diagnostic_only_and_bce_reaches_both_decoders(self) -> None:
        outputs = self.model(self.waveform)
        outputs["residual"].retain_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["logit"], self.labels).backward()
        self.assertIsNone(outputs["residual"].grad)
        self.assertGreater(pipeline.gradient_norm(self.model.source_decoder), 0.0)
        self.assertGreater(pipeline.gradient_norm(self.model.response_decoder), 0.0)

    def test_recon_and_physics_losses_are_finite_and_backward(self) -> None:
        config = pipeline.Config(conditions=pipeline.NO_RESIDUAL_CONDITIONS)
        for condition in pipeline.NO_RESIDUAL_CONDITIONS:
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(self.waveform)
            losses = pipeline.model_losses(outputs, self.waveform, self.labels, condition, config)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            losses["total"].backward()
            self.assertGreater(pipeline.gradient_norm(self.model.shared_encoder), 0.0)
        self.assertEqual(
            pipeline.INITIALIZATION_OFFSETS["recon_bce_no_residual"],
            pipeline.INITIALIZATION_OFFSETS["recon_bce"])
        self.assertEqual(
            pipeline.INITIALIZATION_OFFSETS["recon_physics_bce_no_residual"],
            pipeline.INITIALIZATION_OFFSETS["recon_physics_bce"])


if __name__ == "__main__":
    unittest.main()
