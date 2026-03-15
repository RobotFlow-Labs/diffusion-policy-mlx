"""Tests for conv1d components and sinusoidal positional embedding."""

import math

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from diffusion_policy_mlx.model.diffusion.positional_embedding import SinusoidalPosEmb

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Conv1dBlock
# ---------------------------------------------------------------------------


class TestConv1dBlock:
    def test_shape_preserved(self):
        """Conv1dBlock should preserve spatial dimension L."""
        block = Conv1dBlock(16, 32, kernel_size=3, n_groups=8)
        x = mx.random.normal((2, 16, 50))
        out = block(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_different_kernel_sizes(self):
        """Padding should always preserve L."""
        for ks in [1, 3, 5, 7]:
            block = Conv1dBlock(8, 16, kernel_size=ks, n_groups=8)
            x = mx.random.normal((1, 8, 20))
            out = block(x)
            mx.eval(out)
            assert out.shape == (1, 16, 20), f"kernel_size={ks} changed spatial dim"

    def test_single_sample(self):
        """Works with batch size 1."""
        block = Conv1dBlock(4, 8, kernel_size=3, n_groups=4)
        x = mx.random.normal((1, 4, 10))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 8, 10)


# ---------------------------------------------------------------------------
# Downsample1d
# ---------------------------------------------------------------------------


class TestDownsample1d:
    def test_halves_length(self):
        """Downsample1d should halve the spatial dimension."""
        ds = Downsample1d(32)
        x = mx.random.normal((2, 32, 50))
        out = ds(x)
        mx.eval(out)
        assert out.shape == (2, 32, 25)

    def test_preserves_channels(self):
        ds = Downsample1d(64)
        x = mx.random.normal((3, 64, 40))
        out = ds(x)
        mx.eval(out)
        assert out.shape[1] == 64

    def test_even_length(self):
        ds = Downsample1d(16)
        x = mx.random.normal((1, 16, 16))
        out = ds(x)
        mx.eval(out)
        assert out.shape == (1, 16, 8)


# ---------------------------------------------------------------------------
# Upsample1d
# ---------------------------------------------------------------------------


class TestUpsample1d:
    def test_doubles_length(self):
        """Upsample1d should double the spatial dimension."""
        us = Upsample1d(32)
        x = mx.random.normal((2, 32, 25))
        out = us(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_preserves_channels(self):
        us = Upsample1d(64)
        x = mx.random.normal((3, 64, 10))
        out = us(x)
        mx.eval(out)
        assert out.shape[1] == 64

    def test_roundtrip(self):
        """Downsample then upsample should recover original length."""
        ds = Downsample1d(16)
        us = Upsample1d(16)
        x = mx.random.normal((1, 16, 20))
        down = ds(x)
        mx.eval(down)
        up = us(down)
        mx.eval(up)
        assert up.shape == x.shape


# ---------------------------------------------------------------------------
# SinusoidalPosEmb
# ---------------------------------------------------------------------------


class TestSinusoidalPosEmb:
    def test_output_shape(self):
        emb = SinusoidalPosEmb(256)
        t = mx.array([0, 50, 999])
        out = emb(t)
        mx.eval(out)
        assert out.shape == (3, 256)

    def test_output_shape_small(self):
        emb = SinusoidalPosEmb(64)
        t = mx.array([0, 1, 2, 3])
        out = emb(t)
        mx.eval(out)
        assert out.shape == (4, 64)

    def test_different_timesteps_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        emb = SinusoidalPosEmb(128)
        t = mx.array([0, 100])
        out = emb(t)
        mx.eval(out)
        diff = mx.sum(mx.abs(out[0] - out[1])).item()
        assert diff > 0.0

    def test_sin_cos_split(self):
        """First half should be sin, second half cos."""
        dim = 64
        emb = SinusoidalPosEmb(dim)
        t = mx.array([42])
        out = emb(t)
        mx.eval(out)
        out_np = np.array(out[0])

        # Manually compute expected
        half = dim // 2
        freqs = np.exp(np.arange(half) * -(math.log(10000) / (half - 1)))
        expected_sin = np.sin(42.0 * freqs)
        expected_cos = np.cos(42.0 * freqs)

        np.testing.assert_allclose(out_np[:half], expected_sin, atol=1e-5)
        np.testing.assert_allclose(out_np[half:], expected_cos, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_matches_upstream(self):
        """Output should match upstream PyTorch SinusoidalPosEmb."""
        import sys

        sys.path.insert(0, "repositories/diffusion-policy-upstream")
        from diffusion_policy.model.diffusion.positional_embedding import (
            SinusoidalPosEmb as TorchSinusoidalPosEmb,
        )

        dim = 256
        mlx_emb = SinusoidalPosEmb(dim)
        torch_emb = TorchSinusoidalPosEmb(dim)

        timesteps = [0, 1, 50, 100, 999]
        t_mlx = mx.array(timesteps)
        t_torch = torch.tensor(timesteps, dtype=torch.long)

        out_mlx = mlx_emb(t_mlx)
        mx.eval(out_mlx)
        out_torch = torch_emb(t_torch)

        np.testing.assert_allclose(np.array(out_mlx), out_torch.detach().numpy(), atol=1e-4)
