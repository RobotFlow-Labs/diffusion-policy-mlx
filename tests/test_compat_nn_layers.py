"""Tests for diffusion_policy_mlx.compat.nn_layers and related modules."""

import numpy as np
import pytest
import mlx.core as mx

from diffusion_policy_mlx.compat import (
    Conv1d,
    ConvTranspose1d,
    Conv2d,
    GroupNorm,
    BatchNorm2d,
    Sequential,
    Identity,
    Mish,
    SiLU,
    mish,
    silu,
    pad_1d,
    interpolate_1d,
    rearrange_b_h_t_to_b_t_h,
    rearrange_b_t_h_to_b_h_t,
    rearrange_batch_t_to_batch_t_1,
    ModuleAttrMixin,
    DictOfTensorMixin,
)

try:
    import torch
    import torch.nn as torch_nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Conv1d shape tests
# ---------------------------------------------------------------------------

class TestConv1dShape:
    def test_basic_shape(self):
        conv = Conv1d(16, 32, 3, padding=1)
        x = mx.random.normal((2, 16, 50))  # (B, C, L)
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_stride_shape(self):
        conv = Conv1d(16, 32, 3, stride=2, padding=1)
        x = mx.random.normal((2, 16, 50))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 32, 25)

    def test_no_padding_shape(self):
        conv = Conv1d(8, 16, 5)
        x = mx.random.normal((1, 8, 20))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16)  # 20 - 5 + 1 = 16

    def test_kernel1_shape(self):
        """1x1 conv (used for residual projection)."""
        conv = Conv1d(16, 32, 1)
        x = mx.random.normal((2, 16, 50))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)


# ---------------------------------------------------------------------------
# Conv1d numerics vs torch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestConv1dNumerics:
    def test_matches_torch(self):
        """Conv1d output matches PyTorch given identical weights."""
        in_c, out_c, kernel = 8, 16, 3
        padding = 1

        # Create MLX conv
        mlx_conv = Conv1d(in_c, out_c, kernel, padding=padding, bias=True)

        # Create PyTorch conv
        torch_conv = torch_nn.Conv1d(in_c, out_c, kernel, padding=padding, bias=True)

        # Copy weights: torch weight is (C_out, C_in, K)
        # MLX internal conv weight is (C_out, K, C_in)
        torch_w = torch_conv.weight.detach().numpy()  # (C_out, C_in, K)
        torch_b = torch_conv.bias.detach().numpy()     # (C_out,)

        # MLX Conv1d weight shape: (out, kernel, in)
        mlx_w = np.transpose(torch_w, (0, 2, 1))  # (C_out, K, C_in)
        mlx_conv._conv.weight = mx.array(mlx_w)
        mlx_conv._conv.bias = mx.array(torch_b)

        # Forward pass with same input
        x_np = np.random.randn(2, in_c, 20).astype(np.float32)
        torch_out = torch_conv(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_conv(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=2e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# ConvTranspose1d
# ---------------------------------------------------------------------------

class TestConvTranspose1d:
    def test_upsample_2x_shape(self):
        ct = ConvTranspose1d(32, 32, 4, stride=2, padding=1)
        x = mx.random.normal((2, 32, 25))
        out = ct(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_different_channels(self):
        ct = ConvTranspose1d(64, 32, 4, stride=2, padding=1)
        x = mx.random.normal((1, 64, 10))
        out = ct(x)
        mx.eval(out)
        assert out.shape == (1, 32, 20)


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class TestConv2d:
    def test_basic_shape(self):
        conv = Conv2d(3, 16, 3, padding=1)
        x = mx.random.normal((1, 3, 32, 32))  # (B, C, H, W)
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 32, 32)

    def test_stride_shape(self):
        conv = Conv2d(3, 16, 3, stride=2, padding=1)
        x = mx.random.normal((1, 3, 32, 32))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 16)


# ---------------------------------------------------------------------------
# GroupNorm
# ---------------------------------------------------------------------------

class TestGroupNorm:
    def test_ncl_shape(self):
        gn = GroupNorm(8, 32)
        x = mx.random.normal((2, 32, 50))
        out = gn(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_nchw_shape(self):
        gn = GroupNorm(4, 16)
        x = mx.random.normal((1, 16, 8, 8))
        out = gn(x)
        mx.eval(out)
        assert out.shape == (1, 16, 8, 8)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_matches_torch_ncl(self):
        """GroupNorm output matches PyTorch on (B, C, L) input."""
        B, C, L = 2, 16, 20
        num_groups = 4

        mlx_gn = GroupNorm(num_groups, C)
        torch_gn = torch_nn.GroupNorm(num_groups, C)

        # Copy weights
        torch_w = torch_gn.weight.detach().numpy()
        torch_b = torch_gn.bias.detach().numpy()
        mlx_gn._gn.weight = mx.array(torch_w)
        mlx_gn._gn.bias = mx.array(torch_b)

        x_np = np.random.randn(B, C, L).astype(np.float32)
        torch_out = torch_gn(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_gn(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# BatchNorm2d
# ---------------------------------------------------------------------------

class TestBatchNorm2d:
    def test_shape(self):
        bn = BatchNorm2d(16)
        x = mx.random.normal((2, 16, 8, 8))
        out = bn(x)
        mx.eval(out)
        assert out.shape == (2, 16, 8, 8)


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

class TestSequential:
    def test_chain(self):
        seq = Sequential(
            Conv1d(16, 32, 3, padding=1),
            GroupNorm(8, 32),
        )
        x = mx.random.normal((2, 16, 50))
        out = seq(x)
        mx.eval(out)
        assert out.shape == (2, 32, 50)

    def test_with_identity(self):
        seq = Sequential(
            Conv1d(16, 16, 1),
            Identity(),
        )
        x = mx.random.normal((1, 16, 10))
        out = seq(x)
        mx.eval(out)
        assert out.shape == (1, 16, 10)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_passthrough(self):
        ident = Identity()
        x = mx.array([1.0, 2.0, 3.0])
        out = ident(x)
        np.testing.assert_array_equal(np.array(out), np.array(x))

    def test_ignores_extra_args(self):
        ident = Identity()
        x = mx.array([1.0])
        out = ident(x, "ignored", key="ignored")
        np.testing.assert_array_equal(np.array(out), np.array(x))


# ---------------------------------------------------------------------------
# Functional: mish, silu
# ---------------------------------------------------------------------------

class TestActivations:
    def test_mish_shape(self):
        x = mx.random.normal((2, 16))
        out = mish(x)
        assert out.shape == x.shape

    def test_silu_shape(self):
        x = mx.random.normal((2, 16))
        out = silu(x)
        assert out.shape == x.shape

    def test_mish_module(self):
        m = Mish()
        x = mx.array([0.0, 1.0, -1.0])
        out = m(x)
        mx.eval(out)
        assert out.shape == (3,)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_mish_matches_torch(self):
        x_np = np.random.randn(10).astype(np.float32)
        torch_out = torch.nn.functional.mish(torch.tensor(x_np)).numpy()
        mlx_out = np.array(mish(mx.array(x_np)))
        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_silu_matches_torch(self):
        x_np = np.random.randn(10).astype(np.float32)
        torch_out = torch.nn.functional.silu(torch.tensor(x_np)).numpy()
        mlx_out = np.array(silu(mx.array(x_np)))
        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-5)


# ---------------------------------------------------------------------------
# Functional: pad_1d
# ---------------------------------------------------------------------------

class TestPad1d:
    def test_constant_pad(self):
        x = mx.ones((1, 2, 5))
        out = pad_1d(x, (2, 3), mode="constant", value=0.0)
        mx.eval(out)
        assert out.shape == (1, 2, 10)
        out_np = np.array(out)
        # Left padding should be zeros
        np.testing.assert_array_equal(out_np[0, 0, :2], [0.0, 0.0])
        # Right padding should be zeros
        np.testing.assert_array_equal(out_np[0, 0, 7:], [0.0, 0.0, 0.0])

    def test_replicate_pad(self):
        x = mx.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        out = pad_1d(x, (2, 1), mode="replicate")
        mx.eval(out)
        assert out.shape == (1, 1, 6)
        expected = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
        np.testing.assert_array_equal(np.array(out).flatten(), expected)


# ---------------------------------------------------------------------------
# Functional: interpolate_1d
# ---------------------------------------------------------------------------

class TestInterpolate1d:
    def test_upsample_2x(self):
        x = mx.ones((1, 4, 10))
        out = interpolate_1d(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 4, 20)

    def test_upsample_by_size(self):
        x = mx.ones((1, 4, 10))
        out = interpolate_1d(x, size=30)
        mx.eval(out)
        assert out.shape == (1, 4, 30)


# ---------------------------------------------------------------------------
# Einops helpers
# ---------------------------------------------------------------------------

class TestEinopsHelpers:
    def test_b_h_t_to_b_t_h(self):
        x = mx.random.normal((2, 3, 5))
        out = rearrange_b_h_t_to_b_t_h(x)
        assert out.shape == (2, 5, 3)

    def test_b_t_h_to_b_h_t(self):
        x = mx.random.normal((2, 5, 3))
        out = rearrange_b_t_h_to_b_h_t(x)
        assert out.shape == (2, 3, 5)

    def test_roundtrip(self):
        x = mx.random.normal((2, 3, 5))
        out = rearrange_b_t_h_to_b_h_t(rearrange_b_h_t_to_b_t_h(x))
        np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-7)

    def test_batch_t_to_batch_t_1(self):
        x = mx.random.normal((4, 8))
        out = rearrange_batch_t_to_batch_t_1(x)
        assert out.shape == (4, 8, 1)


# ---------------------------------------------------------------------------
# Module mixins
# ---------------------------------------------------------------------------

class TestModuleAttrMixin:
    def test_device_property(self):
        class MyModule(ModuleAttrMixin):
            def __init__(self):
                super().__init__()
                self.w = mx.zeros((3, 3))
            def __call__(self, x):
                return x
        m = MyModule()
        assert m.device == "mlx"

    def test_dtype_property(self):
        class MyModule(ModuleAttrMixin):
            def __init__(self):
                super().__init__()
                self.w = mx.zeros((3, 3), dtype=mx.float32)
            def __call__(self, x):
                return x
        m = MyModule()
        assert m.dtype == mx.float32


class TestDictOfTensorMixin:
    def test_init_empty(self):
        m = DictOfTensorMixin()
        assert m.params_dict == {}

    def test_load_params_dict(self):
        m = DictOfTensorMixin()
        flat = {
            "params_dict.mean": mx.array([1.0, 2.0]),
            "params_dict.std": mx.array([0.5, 0.5]),
        }
        m.load_params_dict(flat)
        assert "mean" in m.params_dict
        assert "std" in m.params_dict
        np.testing.assert_allclose(np.array(m.params_dict["mean"]), [1.0, 2.0])

    def test_load_params_dict_nested(self):
        m = DictOfTensorMixin()
        flat = {
            "params_dict.obs.mean": mx.array([1.0]),
            "params_dict.obs.std": mx.array([2.0]),
        }
        m.load_params_dict(flat)
        assert "obs" in m.params_dict
        assert "mean" in m.params_dict["obs"]


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_compat_import_star():
    """Verify that `from diffusion_policy_mlx.compat import *` works."""
    import diffusion_policy_mlx.compat as compat
    # Spot-check a few names
    assert hasattr(compat, "Conv1d")
    assert hasattr(compat, "GroupNorm")
    assert hasattr(compat, "is_tensor")
    assert hasattr(compat, "mish")
    assert hasattr(compat, "rearrange_b_h_t_to_b_t_h")
