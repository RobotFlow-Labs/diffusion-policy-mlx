"""Tests for diffusion_policy_mlx.compat.nn_layers and related modules."""

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.compat import (
    BatchNorm2d,
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    DictOfTensorMixin,
    GroupNorm,
    Identity,
    Mish,
    ModuleAttrMixin,
    Sequential,
    interpolate_1d,
    mish,
    pad_1d,
    rearrange_b_h_t_to_b_t_h,
    rearrange_b_t_h_to_b_h_t,
    rearrange_batch_t_to_batch_t_1,
    silu,
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

    def test_groups_shape(self):
        """Conv1d with groups parameter produces correct output shape."""
        conv = Conv1d(16, 16, 3, padding=1, groups=4)
        x = mx.random.normal((2, 16, 50))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 16, 50)
        assert conv.groups == 4

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_groups_matches_torch(self):
        """Conv1d with groups matches PyTorch numerically."""
        in_c, out_c, kernel, groups = 16, 16, 3, 4
        padding = 1

        mlx_conv = Conv1d(in_c, out_c, kernel, padding=padding, groups=groups, bias=True)
        torch_conv = torch_nn.Conv1d(in_c, out_c, kernel, padding=padding, groups=groups, bias=True)

        torch_w = torch_conv.weight.detach().numpy()  # (C_out, C_in/groups, K)
        torch_b = torch_conv.bias.detach().numpy()
        mlx_w = np.transpose(torch_w, (0, 2, 1))  # (C_out, K, C_in/groups)
        mlx_conv._conv.weight = mx.array(mlx_w)
        mlx_conv._conv.bias = mx.array(torch_b)

        x_np = np.random.randn(2, in_c, 20).astype(np.float32)
        torch_out = torch_conv(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_conv(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=2e-3, rtol=1e-4)


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
        torch_b = torch_conv.bias.detach().numpy()  # (C_out,)

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

    def test_output_padding_stored(self):
        ct = ConvTranspose1d(16, 16, 3, stride=2, padding=1, output_padding=1)
        assert ct.output_padding == 1

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_matches_torch(self):
        """ConvTranspose1d output matches PyTorch given identical weights."""
        in_c, out_c, kernel = 8, 16, 4
        stride, padding = 2, 1

        mlx_ct = ConvTranspose1d(in_c, out_c, kernel, stride=stride, padding=padding, bias=True)
        torch_ct = torch_nn.ConvTranspose1d(
            in_c, out_c, kernel, stride=stride, padding=padding, bias=True
        )

        # Copy weights: torch weight is (C_in, C_out, K)
        # MLX ConvTranspose1d weight is (C_out, K, C_in)
        torch_w = torch_ct.weight.detach().numpy()  # (C_in, C_out, K)
        torch_b = torch_ct.bias.detach().numpy()  # (C_out,)

        mlx_w = np.transpose(torch_w, (1, 2, 0))  # (C_out, K, C_in)
        mlx_ct._conv.weight = mx.array(mlx_w)
        mlx_ct._conv.bias = mx.array(torch_b)

        x_np = np.random.randn(2, in_c, 10).astype(np.float32)
        torch_out = torch_ct(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_ct(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=2e-3, rtol=1e-4)


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

    def test_invalid_input_dims(self):
        """GroupNorm wrapper should reject 2-D and 5-D inputs."""
        gn = GroupNorm(4, 16)
        with pytest.raises(ValueError, match="3-D or 4-D"):
            gn(mx.random.normal((16,)))  # 1-D
        with pytest.raises(ValueError, match="3-D or 4-D"):
            gn(mx.random.normal((2, 16)))  # 2-D
        with pytest.raises(ValueError, match="3-D or 4-D"):
            gn(mx.random.normal((1, 2, 16, 4, 4)))  # 5-D


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


# ---------------------------------------------------------------------------
# Conv2d numerics vs torch (P0)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestConv2dNumerics:
    def test_matches_torch(self):
        """Conv2d output matches PyTorch given same weights."""
        in_c, out_c, kernel = 3, 16, 3
        padding = 1

        mlx_conv = Conv2d(in_c, out_c, kernel, padding=padding, bias=True)
        torch_conv = torch_nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=True)

        # Copy weights: torch weight is (C_out, C_in, H, W) = OIHW
        # MLX Conv2d internal weight is (C_out, H, W, C_in) = OHWI
        torch_w = torch_conv.weight.detach().numpy()  # (C_out, C_in, kH, kW)
        torch_b = torch_conv.bias.detach().numpy()

        mlx_w = np.transpose(torch_w, (0, 2, 3, 1))  # (C_out, kH, kW, C_in)
        mlx_conv._conv.weight = mx.array(mlx_w)
        mlx_conv._conv.bias = mx.array(torch_b)

        x_np = np.random.randn(2, in_c, 8, 8).astype(np.float32)
        torch_out = torch_conv(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_conv(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-3, rtol=1e-4)

    def test_stride_matches_torch(self):
        """Conv2d with stride=2 matches PyTorch."""
        in_c, out_c, kernel = 3, 8, 3
        stride, padding = 2, 1

        mlx_conv = Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=True)
        torch_conv = torch_nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=True)

        torch_w = torch_conv.weight.detach().numpy()
        torch_b = torch_conv.bias.detach().numpy()
        mlx_w = np.transpose(torch_w, (0, 2, 3, 1))
        mlx_conv._conv.weight = mx.array(mlx_w)
        mlx_conv._conv.bias = mx.array(torch_b)

        x_np = np.random.randn(1, in_c, 16, 16).astype(np.float32)
        torch_out = torch_conv(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_conv(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-3, rtol=1e-4)

    def test_nonzero_output(self):
        """Conv2d produces non-trivial output."""
        conv = Conv2d(3, 16, 3, padding=1)
        x = mx.random.normal((2, 3, 8, 8))
        out = conv(x)
        mx.eval(out)
        assert float(mx.std(out)) > 0.01, "Conv2d output is trivially zero"


# ---------------------------------------------------------------------------
# BatchNorm2d numerics vs torch (P0)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBatchNorm2dNumerics:
    def test_matches_torch_eval_mode(self):
        """BatchNorm2d output matches PyTorch in eval mode."""
        num_features = 16
        B, H, W = 2, 4, 4

        mlx_bn = BatchNorm2d(num_features)
        torch_bn = torch_nn.BatchNorm2d(num_features)

        # Copy weights and running stats
        torch_bn.eval()
        torch_w = torch_bn.weight.detach().numpy()
        torch_b = torch_bn.bias.detach().numpy()
        torch_rm = torch_bn.running_mean.detach().numpy()
        torch_rv = torch_bn.running_var.detach().numpy()

        mlx_bn._bn.weight = mx.array(torch_w)
        mlx_bn._bn.bias = mx.array(torch_b)
        mlx_bn._bn.running_mean = mx.array(torch_rm)
        mlx_bn._bn.running_var = mx.array(torch_rv)
        mlx_bn.eval()

        x_np = np.random.randn(B, num_features, H, W).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_bn(torch.tensor(x_np)).numpy()

        mlx_out = np.array(mlx_bn(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_nonzero_output(self):
        """BatchNorm2d produces non-trivial output."""
        bn = BatchNorm2d(8)
        x = mx.random.normal((4, 8, 4, 4))
        out = bn(x)
        mx.eval(out)
        assert float(mx.std(out)) > 0.01, "BatchNorm2d output is trivially zero"


# ---------------------------------------------------------------------------
# GroupNorm NCHW cross-framework test (P1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGroupNormNCHW:
    def test_matches_torch_nchw(self):
        """GroupNorm on (B,C,H,W) matches PyTorch."""
        B, C, H, W = 2, 16, 8, 8
        num_groups = 4

        mlx_gn = GroupNorm(num_groups, C)
        torch_gn = torch_nn.GroupNorm(num_groups, C)

        # Copy weights
        torch_w = torch_gn.weight.detach().numpy()
        torch_b = torch_gn.bias.detach().numpy()
        mlx_gn._gn.weight = mx.array(torch_w)
        mlx_gn._gn.bias = mx.array(torch_b)

        x_np = np.random.randn(B, C, H, W).astype(np.float32)
        torch_out = torch_gn(torch.tensor(x_np)).detach().numpy()
        mlx_out = np.array(mlx_gn(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Numerical stability tests (P1)
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_conv1d_nan_propagation(self):
        """NaN inputs produce NaN outputs (no silent corruption)."""
        conv = Conv1d(4, 8, 3, padding=1)
        x = mx.random.normal((1, 4, 10))
        # Inject NaN at one position
        x_np = np.array(x)
        x_np[0, 0, 3] = float("nan")
        x_nan = mx.array(x_np)
        out = conv(x_nan)
        mx.eval(out)
        out_np = np.array(out)
        # At least some outputs should be NaN (the NaN propagates through convolution)
        assert np.any(np.isnan(out_np)), "NaN input did not propagate through Conv1d"

    def test_mish_large_input(self):
        """Mish handles large inputs without overflow (threshold at x>20)."""
        x = mx.array([0.0, 20.0, 50.0, 100.0, 1000.0])
        out = mish(x)
        mx.eval(out)
        out_np = np.array(out)
        assert not np.any(np.isnan(out_np)), f"NaN in mish output: {out_np}"
        assert not np.any(np.isinf(out_np)), f"Inf in mish output: {out_np}"
        # For large x, mish(x) ≈ x (since tanh(softplus(x)) → 1)
        np.testing.assert_allclose(out_np[3], 100.0, atol=1e-2)
        np.testing.assert_allclose(out_np[4], 1000.0, atol=1e-2)

    def test_mish_negative_large_input(self):
        """Mish handles large negative inputs without issues."""
        x = mx.array([-50.0, -100.0, -1000.0])
        out = mish(x)
        mx.eval(out)
        out_np = np.array(out)
        assert not np.any(np.isnan(out_np)), f"NaN in mish output: {out_np}"
        assert not np.any(np.isinf(out_np)), f"Inf in mish output: {out_np}"
        # For large negative x, mish(x) → 0
        np.testing.assert_allclose(out_np, 0.0, atol=1e-5)

    def test_groupnorm_nan_propagation(self):
        """GroupNorm with NaN input produces NaN (doesn't silently zero)."""
        gn = GroupNorm(4, 16)
        x = mx.random.normal((1, 16, 10))
        x_np = np.array(x)
        x_np[0, 0, 0] = float("nan")
        x_nan = mx.array(x_np)
        out = gn(x_nan)
        mx.eval(out)
        out_np = np.array(out)
        # NaN in one group should propagate to that group's outputs
        assert np.any(np.isnan(out_np)), "NaN input did not propagate through GroupNorm"

    def test_batchnorm2d_nan_propagation(self):
        """BatchNorm2d with NaN input produces NaN (doesn't silently zero)."""
        bn = BatchNorm2d(4)
        x = mx.random.normal((2, 4, 4, 4))
        x_np = np.array(x)
        x_np[0, 0, 0, 0] = float("nan")
        x_nan = mx.array(x_np)
        out = bn(x_nan)
        mx.eval(out)
        out_np = np.array(out)
        assert np.any(np.isnan(out_np)), "NaN input did not propagate through BatchNorm2d"


# ---------------------------------------------------------------------------
# interpolate_1d floor correctness (P1 fix #7)
# ---------------------------------------------------------------------------


class TestInterpolate1dFloor:
    def test_floor_vs_truncate_non_integer_ratio(self):
        """interpolate_1d with non-integer ratio uses floor (not truncate).

        For nearest-neighbor, index = floor(i * L / target_len).
        With negative intermediates (not applicable here since i>=0, L>0),
        floor and truncate differ. But we also verify the general correctness
        of the nearest-neighbor index computation.
        """
        # Create a signal where each sample has a unique value
        x_np = np.arange(1, 8, dtype=np.float32).reshape(1, 1, 7)  # L=7
        x = mx.array(x_np)

        # Upsample to 10: ratio = 7/10 = 0.7
        # floor(i * 7 / 10) for i=0..9: floor([0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3])
        #                                  = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]
        out = interpolate_1d(x, size=10)
        mx.eval(out)
        out_np = np.array(out).flatten()

        expected_indices = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]
        expected_values = np.array([x_np.flatten()[i] for i in expected_indices])
        np.testing.assert_array_equal(out_np, expected_values)

    def test_upsample_preserves_values(self):
        """interpolate_1d nearest-neighbor: output values are from input."""
        x_np = np.array([[[10.0, 20.0, 30.0, 40.0]]])  # (1,1,4)
        x = mx.array(x_np)
        out = interpolate_1d(x, scale_factor=3)  # 4 -> 12
        mx.eval(out)
        out_np = np.array(out).flatten()
        # Every output value must be one of the input values
        for v in out_np:
            assert v in [10.0, 20.0, 30.0, 40.0], f"Unexpected interpolated value {v}"

    def test_downsample_by_size(self):
        """interpolate_1d can downsample."""
        x = mx.random.normal((2, 4, 20))
        out = interpolate_1d(x, size=5)
        mx.eval(out)
        assert out.shape == (2, 4, 5)


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
