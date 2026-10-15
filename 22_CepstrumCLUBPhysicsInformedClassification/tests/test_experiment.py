from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


EXPERIMENT_PATH = Path(__file__).resolve().parents[1] / "experiment.py"
SPEC = importlib.util.spec_from_file_location("experiment22", EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_channel_parameterization_is_exact_and_has_no_direct_parameter():
    decoder = experiment.CausalChannelDecoder()
    response = decoder(torch.randn(3, 64))
    assert response.shape == (3, 2881)
    assert torch.equal(response[:, 0], torch.ones(3))
    assert all("direct" not in name for name, _ in decoder.named_parameters())
    assert torch.all(response[:, 1:].abs() <= 0.25)


def test_unit_impulse_channel_recovers_source():
    source = torch.randn(4, 5761, dtype=torch.float64)
    channel = torch.zeros(4, 2881, dtype=torch.float64)
    channel[:, 0] = 1.0
    assert torch.max(torch.abs(source - experiment.causal_fft_convolution(source, channel))).item() <= 1e-5


def test_positive_only_mim_ignores_negative_latents_and_deranges_positive_subset():
    torch.manual_seed(7)
    q_network = experiment.CLUBConditional()
    z_click = torch.randn(4, 64)
    z_channel = torch.randn(4, 64)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    positive_click, positive_channel, count = experiment.positive_latents(z_click, z_channel, labels)
    assert count == 2
    torch.manual_seed(13)
    first = experiment.club_mi_loss(q_network, positive_click, positive_channel)
    changed = z_click.clone()
    changed[labels == 0] = torch.randn_like(changed[labels == 0]) * 1_000
    changed_click, changed_channel, _ = experiment.positive_latents(changed, z_channel, labels)
    torch.manual_seed(13)
    second = experiment.club_mi_loss(q_network, changed_click, changed_channel)
    assert torch.allclose(first, second)
    index = experiment.random_derangement(count, torch.device("cpu"))
    assert index.numel() == count and torch.all(index != torch.arange(count))


def test_fewer_than_two_positive_clicks_skips_mim_safely():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = experiment.CepstrumCLUBModel().to(device)
    q_network = experiment.CLUBConditional().to(device)
    active = experiment.active_main_parameters(model, "full")
    losses = experiment.train_condition_batch(
        model, q_network, torch.optim.AdamW(active), torch.optim.AdamW(q_network.parameters()),
        torch.randn(2, 5761, device=device), torch.tensor([1.0, 0.0], device=device),
        "full", experiment.Config(batch_size=2),
    )
    assert losses["mim_positive_count"] == 1.0
    assert losses["mim_applied"] == 0.0
    assert losses["loss_q"] == 0.0 and losses["loss_mim"] == 0.0
    assert torch.isfinite(torch.tensor(list(losses.values()))).all()


def test_q_and_main_gradient_isolation_with_positive_only_mim():
    torch.manual_seed(7)
    model, q_network = experiment.CepstrumCLUBModel(), experiment.CLUBConditional()
    config = experiment.Config(batch_size=4)
    main_optimizer, q_optimizer = experiment.make_optimizers(model, q_network, config)
    losses, audit = experiment.train_two_optimizer_batch(
        model, q_network, main_optimizer, q_optimizer, torch.randn(4, 5761),
        torch.tensor([0.0, 1.0, 0.0, 1.0]), config,
    )
    assert losses["mim_positive_count"] == 2.0 and losses["mim_applied"] == 1.0
    assert audit["encoder_after_q_backward"]["parameter_tensors_with_gradient"] == 0
    assert audit["q_after_q_update"]["nonzero"]
    assert audit["q_after_main_backward"]["parameter_tensors_with_gradient"] == 0
    assert audit["latent_encoder_gradients"]["E_click"]["nonzero"]
    assert audit["latent_encoder_gradients"]["E_channel"]["nonzero"]


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
def test_single_batch_gpu_forward_backward_is_finite():
    device = torch.device("cuda")
    model, q_network = experiment.CepstrumCLUBModel().to(device), experiment.CLUBConditional().to(device)
    config = experiment.Config(batch_size=2)
    main_optimizer, q_optimizer = experiment.make_optimizers(model, q_network, config)
    losses, audit = experiment.train_two_optimizer_batch(
        model, q_network, main_optimizer, q_optimizer, torch.randn(2, 5761, device=device),
        torch.tensor([0.0, 1.0], device=device), config,
    )
    assert torch.isfinite(torch.tensor(list(losses.values()))).all()
    assert audit["outputs_finite"] and audit["losses_finite"] and audit["h0_exact"]


def test_lightweight_parameter_caps_and_all_output_shapes():
    model, q_network = experiment.CepstrumCLUBModel(), experiment.CLUBConditional()
    assert sum(parameter.numel() for parameter in model.parameters()) <= 900_000
    assert sum(parameter.numel() for parameter in model.parameters()) + sum(parameter.numel() for parameter in q_network.parameters()) <= 950_000
    outputs = model(torch.randn(2, 5761))
    assert {name: list(value.shape) for name, value in outputs.items()} == {
        "z_click": [2, 64], "z_channel": [2, 64], "click_temporal": [2, 32, 181],
        "channel_temporal": [2, 32, 181], "x_hat": [2, 5761], "h_hat": [2, 2881],
        "r_hat": [2, 5761], "logit": [2],
    }


def test_four_condition_gradient_routing_and_q_creation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for condition in experiment.CONDITIONS:
        model = experiment.CepstrumCLUBModel().to(device)
        active = experiment.active_main_parameters(model, condition)
        q_network = experiment.CLUBConditional().to(device) if experiment.condition_weights(condition)["mim"] else None
        assert (q_network is not None) == (condition in ("full", "without_recon"))
        losses = experiment.train_condition_batch(
            model, q_network, torch.optim.AdamW(active),
            torch.optim.AdamW(q_network.parameters()) if q_network else None,
            torch.randn(2, 5761, device=device), torch.tensor([0.0, 1.0], device=device),
            condition, experiment.Config(batch_size=2),
        )
        assert torch.isfinite(torch.tensor(list(losses.values()))).all()
        active_ids = {id(parameter) for parameter in active}
        assert all(parameter.grad is None for parameter in model.parameters() if id(parameter) not in active_ids)


def test_best_checkpoint_restoration_uses_saved_best_not_last_state():
    model, q_network = experiment.CepstrumCLUBModel(), experiment.CLUBConditional()
    expected = next(model.parameters()).detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        torch.save({"model": model.state_dict(), "q_network": q_network.state_dict(), "best_epoch": 2, "validation_auroc": 0.8}, path)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        saved = experiment.restore_best_checkpoint(path, model, q_network, torch.device("cpu"))
    assert saved["best_epoch"] == 2 and torch.equal(next(model.parameters()), expected)


def test_interrupted_training_state_restores_latest_epoch_and_optimizers():
    model, q_network = experiment.CepstrumCLUBModel(), experiment.CLUBConditional()
    config = experiment.Config(batch_size=2)
    active = experiment.active_main_parameters(model, "full")
    main_optimizer, q_optimizer = torch.optim.AdamW(active), torch.optim.AdamW(q_network.parameters())
    expected = next(model.parameters()).detach().clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training_state.pt"
        experiment.save_training_state(path, model, q_network, main_optimizer, q_optimizer, 3, 0.77, 2, 1, "full", 20260826, 0, config)
        with torch.no_grad():
            next(model.parameters()).add_(2.0)
        restored = experiment.restore_training_state(path, model, q_network, main_optimizer, q_optimizer, config, torch.device("cpu"))
    assert restored == (4, 0.77, 2, 1) and torch.equal(next(model.parameters()), expected)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite
