"""Shared test fixtures for diffusion-policy-mlx."""

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import diffusers  # noqa: F401

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

import mlx.core as mx


def check_close(mlx_result, reference, atol=1e-5, rtol=1e-5):
    """Compare MLX result against reference (numpy, torch, or mx.array).

    Uses relative tolerance scaling matching PyTorch test patterns.
    """
    actual = np.array(mlx_result) if isinstance(mlx_result, mx.array) else np.asarray(mlx_result)
    if HAS_TORCH and isinstance(reference, torch.Tensor):
        expected = reference.detach().cpu().numpy()
    elif isinstance(reference, mx.array):
        expected = np.array(reference)
    else:
        expected = np.asarray(reference)

    scale = max(1.0, np.abs(expected).max())
    scaled_atol = atol * scale
    np.testing.assert_allclose(actual, expected, atol=scaled_atol, rtol=rtol)
