from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("experiment22_orthogonal", ROOT / "experiment_orthogonal.py")
orthogonal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = orthogonal
SPEC.loader.exec_module(orthogonal)


def test_cosine_orthogonality_loss_has_expected_geometry():
    labels = torch.tensor([1.0, 1.0])
    z_click = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    parallel = torch.tensor([[3.0, 0.0], [0.0, -2.0]])
    orthogonal_latent = torch.tensor([[0.0, 4.0], [5.0, 0.0]])
    parallel_loss, count, applied = orthogonal.positive_cosine_orthogonality_loss(z_click, parallel, labels)
    orthogonal_loss, _, _ = orthogonal.positive_cosine_orthogonality_loss(z_click, orthogonal_latent, labels)
    assert count == 2 and applied
    assert torch.allclose(parallel_loss, torch.tensor(1.0))
    assert torch.allclose(orthogonal_loss, torch.tensor(0.0))


def test_negative_latents_do_not_affect_positive_only_loss():
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    z_click = torch.randn(4, 64)
    z_channel = torch.randn(4, 64)
    first, _, _ = orthogonal.positive_cosine_orthogonality_loss(z_click, z_channel, labels)
    changed = z_channel.clone()
    changed[labels == 0] *= 1_000
    second, _, _ = orthogonal.positive_cosine_orthogonality_loss(z_click, changed, labels)
    assert torch.allclose(first, second)


def test_all_negative_loss_is_differentiable_zero():
    z_click = torch.randn(3, 64, requires_grad=True)
    z_channel = torch.randn(3, 64, requires_grad=True)
    loss, count, applied = orthogonal.positive_cosine_orthogonality_loss(
        z_click, z_channel, torch.zeros(3)
    )
    assert loss.item() == 0.0 and count == 0 and not applied
    loss.backward()
    assert z_click.grad is not None and z_channel.grad is not None
    assert torch.count_nonzero(z_click.grad) == 0 and torch.count_nonzero(z_channel.grad) == 0


def test_weight_zero_excludes_orthogonal_term_from_total_loss():
    model = orthogonal.base.CepstrumCLUBModel()
    optimizer = torch.optim.AdamW(model.parameters())
    config = orthogonal.OrthConfig(batch_size=2)
    losses = orthogonal.train_orthogonal_batch(
        model, optimizer, torch.randn(2, 5761), torch.tensor([1.0, 0.0]), 0.0, config
    )
    assert losses["loss_orth_weighted"] == 0.0
    assert abs(losses["loss_total"] - losses["loss_recon"] - losses["loss_bce"]) < 1e-5
    assert all("mim" not in key and "q" not in key for key in losses)


def test_config_hash_includes_orthogonal_weight_grid():
    first = orthogonal.OrthConfig(orth_weights=(0.0, 0.1))
    second = orthogonal.OrthConfig(orth_weights=(0.0, 0.3))
    assert first.config_hash != second.config_hash


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
def test_gpu_batch_is_finite_and_orthogonal_gradients_reach_both_encoders():
    device = torch.device("cuda")
    model = orthogonal.base.CepstrumCLUBModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    losses = orthogonal.train_orthogonal_batch(
        model, optimizer, torch.randn(2, 5761, device=device), torch.tensor([1.0, 1.0], device=device),
        1.0, orthogonal.OrthConfig(batch_size=2, device="cuda")
    )
    assert torch.isfinite(torch.tensor(list(losses.values()))).all()
    assert any(parameter.grad is not None for parameter in model.E_click.parameters())
    assert any(parameter.grad is not None for parameter in model.E_channel.parameters())


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
