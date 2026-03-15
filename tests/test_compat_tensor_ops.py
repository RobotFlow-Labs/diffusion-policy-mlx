"""Tests for diffusion_policy_mlx.compat.tensor_ops."""

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.compat import (
    cat,
    clamp,
    detach,
    expand_as_batch,
    flatten,
    is_tensor,
    ones_like,
    randn_like,
    squeeze,
    stack,
    tensor_to_float,
    tensor_to_long,
    unsqueeze,
    zeros_like,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# is_tensor
# ---------------------------------------------------------------------------


class TestIsTensor:
    def test_mx_array_is_tensor(self):
        assert is_tensor(mx.array([1.0]))

    def test_list_is_not_tensor(self):
        assert not is_tensor([1.0])

    def test_numpy_is_not_tensor(self):
        assert not is_tensor(np.array([1.0]))

    def test_scalar_is_not_tensor(self):
        assert not is_tensor(42)


# ---------------------------------------------------------------------------
# tensor_to_long / tensor_to_float
# ---------------------------------------------------------------------------


class TestCasts:
    def test_to_long(self):
        x = mx.array([1.5, 2.7])
        out = tensor_to_long(x)
        assert out.dtype == mx.int32

    def test_to_float(self):
        x = mx.array([1, 2], dtype=mx.int32)
        out = tensor_to_float(x)
        assert out.dtype == mx.float32


# ---------------------------------------------------------------------------
# expand_as_batch
# ---------------------------------------------------------------------------


class TestExpandAsBatch:
    def test_scalar(self):
        t = mx.array(5)
        out = expand_as_batch(t, 4)
        assert out.shape == (4,)
        assert np.all(np.array(out) == 5)

    def test_single_element(self):
        t = mx.array([7])
        out = expand_as_batch(t, 3)
        assert out.shape == (3,)
        assert np.all(np.array(out) == 7)

    def test_already_batched(self):
        t = mx.array([1, 2, 3])
        out = expand_as_batch(t, 3)
        assert out.shape == (3,)
        np.testing.assert_array_equal(np.array(out), [1, 2, 3])


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_both_bounds(self):
        x = mx.array([-2.0, 0.5, 3.0])
        out = clamp(x, 0.0, 1.0)
        np.testing.assert_allclose(np.array(out), [0.0, 0.5, 1.0])

    def test_min_only(self):
        x = mx.array([-2.0, 0.5])
        out = clamp(x, min_val=0.0)
        assert np.array(out)[0] == 0.0

    def test_max_only(self):
        x = mx.array([0.5, 5.0])
        out = clamp(x, max_val=1.0)
        assert np.array(out)[1] == 1.0


# ---------------------------------------------------------------------------
# zeros_like / ones_like / randn_like
# ---------------------------------------------------------------------------


class TestLikeOps:
    def test_zeros_like(self):
        x = mx.ones((2, 3))
        out = zeros_like(x)
        assert out.shape == (2, 3)
        assert np.all(np.array(out) == 0.0)

    def test_ones_like(self):
        x = mx.zeros((2, 3))
        out = ones_like(x)
        assert out.shape == (2, 3)
        assert np.all(np.array(out) == 1.0)

    def test_randn_like_shape_dtype(self):
        x = mx.zeros((4, 5), dtype=mx.float32)
        out = randn_like(x)
        assert out.shape == (4, 5)
        assert out.dtype == mx.float32


# ---------------------------------------------------------------------------
# unsqueeze / squeeze
# ---------------------------------------------------------------------------


class TestUnsqueezeSqueeze:
    def test_unsqueeze(self):
        x = mx.zeros((2, 3))
        assert unsqueeze(x, 0).shape == (1, 2, 3)
        assert unsqueeze(x, 1).shape == (2, 1, 3)
        assert unsqueeze(x, -1).shape == (2, 3, 1)

    def test_squeeze_specific(self):
        x = mx.zeros((2, 1, 3))
        assert squeeze(x, 1).shape == (2, 3)

    def test_squeeze_all(self):
        x = mx.zeros((1, 2, 1, 3))
        assert squeeze(x).shape == (2, 3)


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


class TestFlatten:
    def test_flatten_default(self):
        x = mx.zeros((2, 3, 4))
        out = flatten(x)
        assert out.shape == (24,)

    def test_flatten_start_dim(self):
        x = mx.zeros((2, 3, 4))
        out = flatten(x, start_dim=1)
        assert out.shape == (2, 12)

    def test_flatten_range(self):
        x = mx.zeros((2, 3, 4, 5))
        out = flatten(x, start_dim=1, end_dim=2)
        assert out.shape == (2, 12, 5)


# ---------------------------------------------------------------------------
# detach
# ---------------------------------------------------------------------------


class TestDetach:
    def test_detach_returns_array(self):
        x = mx.array([1.0, 2.0])
        out = detach(x)
        assert isinstance(out, mx.array)
        np.testing.assert_array_equal(np.array(out), np.array(x))


# ---------------------------------------------------------------------------
# cat / stack
# ---------------------------------------------------------------------------


class TestCatStack:
    def test_cat_dim0(self):
        a = mx.ones((2, 3))
        b = mx.zeros((2, 3))
        out = cat([a, b], dim=0)
        assert out.shape == (4, 3)

    def test_cat_dim1(self):
        a = mx.ones((2, 3))
        b = mx.zeros((2, 4))
        out = cat([a, b], dim=1)
        assert out.shape == (2, 7)

    def test_stack_dim0(self):
        a = mx.ones((2, 3))
        b = mx.zeros((2, 3))
        out = stack([a, b], dim=0)
        assert out.shape == (2, 2, 3)

    def test_stack_dim1(self):
        a = mx.ones((2, 3))
        b = mx.zeros((2, 3))
        out = stack([a, b], dim=1)
        assert out.shape == (2, 2, 3)


# ---------------------------------------------------------------------------
# Cross-framework numeric tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestCrossFramework:
    def test_cat_matches_torch(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)
        torch_out = torch.cat([torch.tensor(a_np), torch.tensor(b_np)], dim=0)
        mlx_out = cat([mx.array(a_np), mx.array(b_np)], dim=0)
        np.testing.assert_allclose(np.array(mlx_out), torch_out.numpy(), atol=1e-6)

    def test_stack_matches_torch(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)
        torch_out = torch.stack([torch.tensor(a_np), torch.tensor(b_np)], dim=1)
        mlx_out = stack([mx.array(a_np), mx.array(b_np)], dim=1)
        np.testing.assert_allclose(np.array(mlx_out), torch_out.numpy(), atol=1e-6)

    def test_clamp_matches_torch(self):
        x_np = np.random.randn(10).astype(np.float32)
        torch_out = torch.clamp(torch.tensor(x_np), -0.5, 0.5)
        mlx_out = clamp(mx.array(x_np), -0.5, 0.5)
        np.testing.assert_allclose(np.array(mlx_out), torch_out.numpy(), atol=1e-6)
