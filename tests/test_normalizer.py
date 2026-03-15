"""Tests for normalizer module (PRD-05).

Tests SingleFieldLinearNormalizer and LinearNormalizer:
  - Limits mode normalization
  - Gaussian mode normalization
  - Round-trip fidelity
  - Identity normalizer
  - Manual normalizer
  - Dict-based LinearNormalizer
  - State dict serialization
"""

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


# ---------------------------------------------------------------------------
# SingleFieldLinearNormalizer
# ---------------------------------------------------------------------------

class TestSingleFieldLinearNormalizerLimits:
    """Tests for limits mode."""

    def test_output_range(self):
        """Normalized data should lie in [-1, 1]."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(100, 2)) * 10 - 5
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="limits")
        normed = norm.normalize(data)
        assert float(mx.min(normed)) >= -1.01
        assert float(mx.max(normed)) <= 1.01

    def test_roundtrip(self):
        """normalize -> unnormalize should recover original data."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(100, 2)) * 10 - 5
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="limits")
        normed = norm.normalize(data)
        recovered = norm.unnormalize(normed)
        np.testing.assert_allclose(
            np.array(recovered), np.array(data), atol=1e-5
        )

    def test_custom_range(self):
        """Custom output range [0, 1]."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(50, 3)) * 100
        norm = SingleFieldLinearNormalizer.create_fit(
            data, mode="limits", output_min=0.0, output_max=1.0
        )
        normed = norm.normalize(data)
        assert float(mx.min(normed)) >= -0.01
        assert float(mx.max(normed)) <= 1.01

    def test_multidim_shape_preserved(self):
        """Normalization should preserve the input shape."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(10, 16, 2)) * 5
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="limits")
        normed = norm.normalize(data)
        assert normed.shape == data.shape
        recovered = norm.unnormalize(normed)
        assert recovered.shape == data.shape

    def test_no_offset(self):
        """fit_offset=False should use zero offset."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(100, 2)) * 10 - 5
        norm = SingleFieldLinearNormalizer.create_fit(
            data, mode="limits", fit_offset=False
        )
        assert float(mx.max(mx.abs(norm.offset))) == 0.0


class TestSingleFieldLinearNormalizerGaussian:
    """Tests for gaussian mode."""

    def test_stats(self):
        """Normalized data should have ~mean=0, ~std=1."""
        mx.random.seed(42)
        data = mx.random.normal((1000, 3)) * 5 + 10
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="gaussian")
        normed = norm.normalize(data)
        assert abs(float(mx.mean(normed))) < 0.15
        assert abs(float(mx.std(normed)) - 1.0) < 0.15

    def test_roundtrip(self):
        """normalize -> unnormalize should recover original data."""
        mx.random.seed(42)
        data = mx.random.normal((200, 4)) * 3 + 7
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="gaussian")
        normed = norm.normalize(data)
        recovered = norm.unnormalize(normed)
        np.testing.assert_allclose(
            np.array(recovered), np.array(data), atol=1e-4
        )


class TestSingleFieldLinearNormalizerFactories:
    """Tests for identity and manual constructors."""

    def test_identity(self):
        """Identity normalizer should be a no-op."""
        norm = SingleFieldLinearNormalizer.create_identity(shape=(3,))
        data = mx.array([1.0, 2.0, 3.0])
        normed = norm.normalize(data)
        np.testing.assert_allclose(np.array(normed), np.array(data), atol=1e-7)

    def test_manual(self):
        """Manual scale/offset should apply correctly."""
        norm = SingleFieldLinearNormalizer.create_manual(
            scale=mx.array([2.0, 0.5]),
            offset=mx.array([1.0, -1.0]),
        )
        data = mx.array([[1.0, 2.0], [3.0, 4.0]])
        normed = norm.normalize(data)
        expected = data * mx.array([2.0, 0.5]) + mx.array([1.0, -1.0])
        np.testing.assert_allclose(np.array(normed), np.array(expected), atol=1e-6)

    def test_state_dict_roundtrip(self):
        """state_dict -> load_state_dict should preserve normalizer."""
        mx.random.seed(42)
        data = mx.random.uniform(shape=(50, 2))
        norm = SingleFieldLinearNormalizer.create_fit(data, mode="limits")
        sd = norm.state_dict()

        norm2 = SingleFieldLinearNormalizer(
            scale=sd["scale"], offset=sd["offset"]
        )
        test_data = mx.random.uniform(shape=(5, 2))
        np.testing.assert_allclose(
            np.array(norm.normalize(test_data)),
            np.array(norm2.normalize(test_data)),
            atol=1e-7,
        )


# ---------------------------------------------------------------------------
# LinearNormalizer
# ---------------------------------------------------------------------------

class TestLinearNormalizer:
    """Tests for the dict-based LinearNormalizer."""

    def test_flat_dict(self):
        """Normalize a flat dict of tensors."""
        mx.random.seed(42)
        norm = LinearNormalizer()
        norm.fit({
            "action": mx.random.uniform(shape=(100, 16, 2)),
            "agent_pos": mx.random.uniform(shape=(100, 16, 2)),
        })
        batch = {
            "action": mx.random.uniform(shape=(4, 16, 2)),
            "agent_pos": mx.random.uniform(shape=(4, 16, 2)),
        }
        out = norm.normalize(batch)
        assert out["action"].shape == (4, 16, 2)
        assert out["agent_pos"].shape == (4, 16, 2)

    def test_nested_dict(self):
        """Normalize a nested dict (obs with sub-keys)."""
        mx.random.seed(42)
        norm = LinearNormalizer()
        norm.fit({
            "obs": {
                "agent_pos": mx.random.uniform(shape=(100, 2)) * 10,
            },
            "action": mx.random.uniform(shape=(100, 2)) * 5,
        })
        batch = {
            "obs": {"agent_pos": mx.random.uniform(shape=(4, 2)) * 10},
            "action": mx.random.uniform(shape=(4, 2)) * 5,
        }
        out = norm.normalize(batch)
        assert out["obs"]["agent_pos"].shape == (4, 2)
        assert out["action"].shape == (4, 2)

    def test_roundtrip(self):
        """normalize -> unnormalize roundtrip for dict."""
        mx.random.seed(42)
        norm = LinearNormalizer()
        fit_data = {
            "action": mx.random.uniform(shape=(100, 16, 2)) * 512,
            "obs": mx.random.uniform(shape=(100, 16, 4)) * 512,
        }
        norm.fit(fit_data)

        batch = {
            "action": mx.random.uniform(shape=(4, 16, 2)) * 512,
            "obs": mx.random.uniform(shape=(4, 16, 4)) * 512,
        }
        normed = norm.normalize(batch)
        recovered = norm.unnormalize(normed)
        np.testing.assert_allclose(
            np.array(recovered["action"]),
            np.array(batch["action"]),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(recovered["obs"]),
            np.array(batch["obs"]),
            atol=1e-3,
        )

    def test_getitem(self):
        """Indexing should return a SingleFieldLinearNormalizer."""
        mx.random.seed(42)
        norm = LinearNormalizer()
        norm.fit({"action": mx.random.uniform(shape=(50, 2))})
        sub = norm["action"]
        assert isinstance(sub, SingleFieldLinearNormalizer)

    def test_contains(self):
        """__contains__ should work."""
        norm = LinearNormalizer()
        norm.fit({"action": mx.random.uniform(shape=(10, 2))})
        assert "action" in norm
        assert "missing" not in norm

    def test_state_dict_roundtrip(self):
        """state_dict -> load_state_dict preserves LinearNormalizer."""
        mx.random.seed(42)
        norm = LinearNormalizer()
        norm.fit({
            "action": mx.random.uniform(shape=(50, 2)) * 100,
            "obs": mx.random.uniform(shape=(50, 4)) * 100,
        })
        sd = norm.state_dict()

        norm2 = LinearNormalizer()
        norm2.load_state_dict(sd)

        batch = {
            "action": mx.random.uniform(shape=(3, 2)) * 100,
            "obs": mx.random.uniform(shape=(3, 4)) * 100,
        }
        out1 = norm.normalize(batch)
        out2 = norm2.normalize(batch)
        np.testing.assert_allclose(
            np.array(out1["action"]), np.array(out2["action"]), atol=1e-6
        )
        np.testing.assert_allclose(
            np.array(out1["obs"]), np.array(out2["obs"]), atol=1e-6
        )

    def test_passthrough_unknown_keys(self):
        """Keys not in the normalizer should pass through unchanged."""
        norm = LinearNormalizer()
        norm.fit({"action": mx.random.uniform(shape=(10, 2))})
        batch = {
            "action": mx.random.uniform(shape=(2, 2)),
            "extra": mx.array([1.0, 2.0]),
        }
        out = norm.normalize(batch)
        np.testing.assert_allclose(
            np.array(out["extra"]), np.array(batch["extra"]), atol=1e-7
        )
