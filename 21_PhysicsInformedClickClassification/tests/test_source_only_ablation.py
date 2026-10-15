#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信源主分支单独分类消融的结构、梯度、损失和初始化种子测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("click_pipeline_source_only_tests", PIPELINE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class SourceOnlyAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260826)
        self.model = pipeline.PhysicalClassifier(include_residual=False, include_response_feature=False)
        self.waveform = torch.from_numpy(np.random.default_rng(20260826).normal(
            size=(2, 5761)).astype(np.float32))
        self.labels = torch.tensor([0.0, 1.0])
        self.config = pipeline.Config(conditions=pipeline.SOURCE_ONLY_CONDITIONS)

    def test_classifier_is_32_to_64_to_1_without_response_or_residual_encoders(self) -> None:
        self.assertFalse(hasattr(self.model, "response_classifier"))
        self.assertFalse(hasattr(self.model, "residual_classifier"))
        self.assertEqual(self.model.classification_head[0].in_features, 32)
        self.assertEqual(self.model.classification_head[0].out_features, 64)
        self.assertEqual(self.model.classification_head[-1].out_features, 1)

    def test_bce_uses_source_path_but_not_response_decoder(self) -> None:
        outputs = self.model(self.waveform)
        torch.nn.functional.binary_cross_entropy_with_logits(outputs["logit"], self.labels).backward()
        for module in (self.model.source_decoder, self.model.shared_encoder, self.model.classification_head):
            self.assertTrue(np.isfinite(pipeline.gradient_norm(module)))
            self.assertGreater(pipeline.gradient_norm(module), 0.0)
        self.assertEqual(pipeline.gradient_norm(self.model.response_decoder), 0.0)

    def test_reconstruction_and_physics_reach_response_decoder(self) -> None:
        # 响应解码器的尾部按设计零初始化；为测试 |h| 的非零独立梯度，先构造非零尾部。
        with torch.no_grad():
            self.model.response_decoder.layers[-1].weight.normal_(0.0, 1e-3)
        for condition in pipeline.SOURCE_ONLY_CONDITIONS:
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(self.waveform)
            losses = pipeline.model_losses(outputs, self.waveform, self.labels, condition, self.config)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
            losses["recon"].backward(retain_graph=True)
            self.assertGreater(pipeline.gradient_norm(self.model.response_decoder), 0.0)
            if condition == "recon_physics_bce_source_only":
                self.model.zero_grad(set_to_none=True)
                pipeline.physics_loss_for_condition(outputs["response"], condition, self.config).backward()
                self.assertGreater(pipeline.gradient_norm(self.model.response_decoder), 0.0)
        self.assertEqual(pipeline.INITIALIZATION_OFFSETS["recon_bce_source_only"], 1_000)
        self.assertEqual(pipeline.INITIALIZATION_OFFSETS["recon_physics_bce_source_only"], 2_000)


if __name__ == "__main__":
    unittest.main()
