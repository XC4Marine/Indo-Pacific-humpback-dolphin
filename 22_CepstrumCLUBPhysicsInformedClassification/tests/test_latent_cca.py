import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
import analyze_latent_cca as cca  # noqa: E402


def test_linear_cca_detects_shared_linear_factor():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(600, 3))
    x = np.c_[latent, rng.normal(size=(600, 3))]
    y = np.c_[2.0 * latent + 0.02 * rng.normal(size=(600, 3)), rng.normal(size=(600, 3))]
    values = cca.regularized_cca(x, y, components=3)
    assert values[0] > 0.99
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_kernel_cca_detects_nonlinear_relation():
    rng = np.random.default_rng(9)
    base = rng.uniform(-1.0, 1.0, size=(700, 1))
    x = np.c_[base, rng.normal(size=(700, 2))]
    y = np.c_[base ** 2 + 0.02 * rng.normal(size=(700, 1)), rng.normal(size=(700, 2))]
    linear = cca.regularized_cca(x, y, components=1)[0]
    kernel, _, _ = cca.kernel_cca(x, y, components=1, landmarks=128, seed=11)
    assert kernel[0] > linear + 0.25


def test_kernel_features_and_cca_are_deterministic_and_finite():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(100, 8)), rng.normal(size=(100, 8))
    first = cca.kernel_cca(x, y, components=5, landmarks=32, seed=19)
    second = cca.kernel_cca(x, y, components=5, landmarks=32, seed=19)
    assert np.allclose(first[0], second[0])
    assert np.isfinite(first[0]).all()
    assert 0.0 <= first[0].min() <= first[0].max() <= 1.0
