"""Tests for ConditionalUnet1D."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest

from diffusion_policy_mlx.model.diffusion.conditional_unet1d import (
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# ConditionalResidualBlock1D
# ---------------------------------------------------------------------------

class TestConditionalResidualBlock1D:
    def test_same_channels(self):
        """When in_channels == out_channels, identity residual."""
        block = ConditionalResidualBlock1D(32, 32, cond_dim=64, n_groups=8)
        x = mx.random.normal((2, 32, 16))
        cond = mx.random.normal((2, 64))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 32, 16)

    def test_different_channels(self):
        """When in_channels != out_channels, 1x1 residual conv."""
        block = ConditionalResidualBlock1D(16, 32, cond_dim=64, n_groups=8)
        x = mx.random.normal((2, 16, 20))
        cond = mx.random.normal((2, 64))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 32, 20)

    def test_film_scale_bias(self):
        """FiLM with cond_predict_scale=True."""
        block = ConditionalResidualBlock1D(
            16, 32, cond_dim=64, n_groups=8, cond_predict_scale=True
        )
        x = mx.random.normal((2, 16, 10))
        cond = mx.random.normal((2, 64))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 32, 10)

    def test_film_bias_only(self):
        """FiLM with cond_predict_scale=False (default)."""
        block = ConditionalResidualBlock1D(
            16, 32, cond_dim=64, n_groups=8, cond_predict_scale=False
        )
        x = mx.random.normal((2, 16, 10))
        cond = mx.random.normal((2, 64))
        out = block(x, cond)
        mx.eval(out)
        assert out.shape == (2, 32, 10)


# ---------------------------------------------------------------------------
# ConditionalUnet1D — shape tests
# ---------------------------------------------------------------------------

class TestConditionalUnet1DShape:
    def test_basic_shape(self):
        """Output shape should be (B, T, input_dim)."""
        unet = ConditionalUnet1D(input_dim=2, down_dims=[32, 64, 128])
        x = mx.random.normal((4, 16, 2))  # (B, T, input_dim)
        t = mx.array([10, 20, 30, 40])
        out = unet(x, t)
        mx.eval(out)
        assert out.shape == (4, 16, 2)

    def test_with_global_cond(self):
        """Should work with global conditioning."""
        unet = ConditionalUnet1D(
            input_dim=2, global_cond_dim=64, down_dims=[32, 64]
        )
        x = mx.random.normal((4, 16, 2))
        t = mx.array([10, 20, 30, 40])
        cond = mx.random.normal((4, 64))
        out = unet(x, t, global_cond=cond)
        mx.eval(out)
        assert out.shape == (4, 16, 2)

    def test_scalar_timestep(self):
        """Should handle scalar timestep input."""
        unet = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        x = mx.random.normal((2, 8, 2))
        out = unet(x, 5)
        mx.eval(out)
        assert out.shape == (2, 8, 2)

    def test_single_element_timestep(self):
        """Should handle 0-dim array timestep."""
        unet = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        x = mx.random.normal((2, 8, 2))
        t = mx.array(5)
        out = unet(x, t)
        mx.eval(out)
        assert out.shape == (2, 8, 2)

    def test_different_input_dims(self):
        """Should work with various input_dim values."""
        for input_dim in [2, 7, 16]:
            unet = ConditionalUnet1D(input_dim=input_dim, down_dims=[16, 32])
            x = mx.random.normal((2, 8, input_dim))
            t = mx.array([1, 2])
            out = unet(x, t)
            mx.eval(out)
            assert out.shape == (2, 8, input_dim)

    def test_two_level_unet(self):
        """Should work with only 2 downsampling levels."""
        unet = ConditionalUnet1D(input_dim=4, down_dims=[32, 64])
        x = mx.random.normal((2, 16, 4))
        t = mx.array([1, 2])
        out = unet(x, t)
        mx.eval(out)
        assert out.shape == (2, 16, 4)

    def test_with_cond_predict_scale(self):
        """FiLM scale+bias mode should produce correct shape."""
        unet = ConditionalUnet1D(
            input_dim=2, down_dims=[16, 32], cond_predict_scale=True
        )
        x = mx.random.normal((2, 8, 2))
        t = mx.array([1, 2])
        out = unet(x, t)
        mx.eval(out)
        assert out.shape == (2, 8, 2)

    def test_with_local_cond(self):
        """Local conditioning path should produce correct shape."""
        unet = ConditionalUnet1D(
            input_dim=2, local_cond_dim=8, down_dims=[32, 64]
        )
        x = mx.random.normal((2, 16, 2))
        t = mx.array([1, 2])
        local = mx.random.normal((2, 16, 8))
        out = unet(x, t, local_cond=local)
        mx.eval(out)
        assert out.shape == (2, 16, 2)

    def test_with_local_and_global_cond(self):
        """Both local and global conditioning paths together."""
        unet = ConditionalUnet1D(
            input_dim=2, local_cond_dim=8, global_cond_dim=16, down_dims=[32, 64]
        )
        x = mx.random.normal((2, 16, 2))
        t = mx.array([1, 2])
        local = mx.random.normal((2, 16, 8))
        glob = mx.random.normal((2, 16))
        out = unet(x, t, local_cond=local, global_cond=glob)
        mx.eval(out)
        assert out.shape == (2, 16, 2)


# ---------------------------------------------------------------------------
# ConditionalUnet1D — gradient flow
# ---------------------------------------------------------------------------

class TestConditionalUnet1DGradient:
    def test_gradient_flow(self):
        """Verify gradients flow through all parameters."""
        unet = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        x = mx.random.normal((2, 8, 2))
        t = mx.array([5, 10])

        def loss_fn(model, x, t):
            out = model(x, t)
            return mx.mean(out ** 2)

        loss, grads = nn.value_and_grad(unet, loss_fn)(unet, x, t)
        mx.eval(loss, grads)

        # Check that loss is finite
        assert np.isfinite(loss.item())

        # Check that at least some grads are non-zero
        flat_grads = mlx.utils.tree_flatten(grads)
        has_nonzero = any(mx.any(g != 0).item() for _, g in flat_grads if isinstance(g, mx.array))
        assert has_nonzero, "All gradients are zero — gradient flow is broken"

    def test_gradient_flow_with_global_cond(self):
        """Gradients should flow through global conditioning path."""
        unet = ConditionalUnet1D(
            input_dim=2, global_cond_dim=32, down_dims=[16, 32]
        )
        x = mx.random.normal((2, 8, 2))
        t = mx.array([5, 10])
        cond = mx.random.normal((2, 32))

        def loss_fn(model, x, t, cond):
            out = model(x, t, global_cond=cond)
            return mx.mean(out ** 2)

        loss, grads = nn.value_and_grad(unet, loss_fn)(unet, x, t, cond)
        mx.eval(loss, grads)

        assert np.isfinite(loss.item())
        flat_grads = mlx.utils.tree_flatten(grads)
        has_nonzero = any(mx.any(g != 0).item() for _, g in flat_grads if isinstance(g, mx.array))
        assert has_nonzero


# ---------------------------------------------------------------------------
# ConditionalUnet1D — determinism
# ---------------------------------------------------------------------------

class TestConditionalUnet1DDeterminism:
    def test_deterministic_output(self):
        """Same input should produce same output (no dropout)."""
        unet = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        x = mx.random.normal((2, 8, 2))
        t = mx.array([5, 10])

        out1 = unet(x, t)
        mx.eval(out1)
        out2 = unet(x, t)
        mx.eval(out2)

        np.testing.assert_array_equal(np.array(out1), np.array(out2))


# ---------------------------------------------------------------------------
# Cross-framework match (optional, requires torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestConditionalUnet1DMatchUpstream:
    def test_sinusoidal_emb_matches(self):
        """Timestep embeddings should match upstream exactly."""
        import sys
        sys.path.insert(0, "repositories/diffusion-policy-upstream")
        from diffusion_policy.model.diffusion.positional_embedding import (
            SinusoidalPosEmb as TorchSinPosEmb,
        )
        from diffusion_policy_mlx.model.diffusion.positional_embedding import (
            SinusoidalPosEmb as MLXSinPosEmb,
        )

        dim = 256
        t_vals = [0, 1, 50, 100, 999]

        torch_emb = TorchSinPosEmb(dim)
        mlx_emb = MLXSinPosEmb(dim)

        t_torch = torch.tensor(t_vals, dtype=torch.long)
        t_mlx = mx.array(t_vals)

        out_torch = torch_emb(t_torch).detach().numpy()
        out_mlx = np.array(mlx_emb(t_mlx))
        mx.eval(mlx_emb(t_mlx))

        np.testing.assert_allclose(out_mlx, out_torch, atol=1e-4)
