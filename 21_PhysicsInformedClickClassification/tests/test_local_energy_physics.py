#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局部能量Physics损失的数值、范围、梯度与哈希测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("click_pipeline_local_energy_tests", PIPELINE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class LocalEnergyPhysicsTests(unittest.TestCase):
    """验证局部能量损失的数值、约束范围和梯度路径。"""

    def test_avg_pool_matches_manual_and_zero_tail_is_zero(self) -> None:
        response = torch.zeros(2, 1153, dtype=torch.float64)
        response[:, 0] = 1.0
        response[1, 1:58] = torch.arange(57, dtype=torch.float64) / 57.0
        curve = pipeline.local_energy_curve(response, window=57, stride=1, epsilon=1e-8)
        expected = torch.sqrt(torch.mean(response[1, 1:58].square()) + 1e-8) - 1e-4
        self.assertEqual(curve.shape, (2, 1096))
        self.assertTrue(torch.equal(curve[0], torch.zeros(1096, dtype=torch.float64)))
        self.assertTrue(torch.allclose(curve[1, 0], expected, atol=1e-14, rtol=0.0))

    def test_h0_is_excluded_and_patterns_have_finite_gradients(self) -> None:
        response = torch.zeros(4, 1153, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            response[:, 0] = 1.0
            response[1, 20:90] = 0.5
            response[2, 200] = 1.0
            response[3, 1:] = 0.1
        curve = pipeline.local_energy_curve(response)
        curve.mean().backward()
        self.assertTrue(torch.isfinite(curve).all())
        self.assertTrue(torch.isfinite(response.grad).all())
        self.assertTrue(torch.equal(response.grad[:, 0], torch.zeros(4, dtype=torch.float64)))
        changed_h0 = response.detach().clone(); changed_h0[:, 0] = -1234.0
        self.assertTrue(torch.equal(curve.detach(), pipeline.local_energy_curve(changed_h0)))

    def test_local_energy_only_reaches_response_decoder_and_shared_encoder(self) -> None:
        torch.manual_seed(20260826)
        model = pipeline.PhysicalClassifier(sample_count=5761, response_count=1153)
        # 解码器历史设计为严格零尾部初始化；模拟完成一次总损失更新后的非零状态。
        with torch.no_grad():
            model.response_decoder.layers[-1].weight.normal_(0.0, 1e-3)
        waveform = torch.from_numpy(np.random.default_rng(20260826).normal(
            size=(2, 5761)).astype(np.float32))
        outputs = model(waveform)
        pipeline.local_energy_physics_loss(outputs["response"]).backward()
        for module in (model.response_decoder, model.shared_encoder):
            norm = pipeline.gradient_norm(module)
            self.assertTrue(np.isfinite(norm) and norm > 0.0)

    def test_config_hash_separates_local_energy_geometry(self) -> None:
        base = pipeline.Config(conditions=pipeline.ALL_CONDITIONS)
        changed_window = pipeline.Config(conditions=pipeline.ALL_CONDITIONS, local_energy_window=55)
        changed_stride = pipeline.Config(conditions=pipeline.ALL_CONDITIONS, local_energy_stride=2)
        self.assertEqual(len({base.config_hash, changed_window.config_hash, changed_stride.config_hash}), 3)


if __name__ == "__main__":
    unittest.main()
