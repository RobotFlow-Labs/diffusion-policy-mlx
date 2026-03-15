"""Tests for the full policy assembly (PRD-05).

Tests DiffusionUnetHybridImagePolicy:
  - predict_action output shapes
  - compute_loss returns scalar > 0
  - Loss is differentiable (gradients flow)
  - LowdimMaskGenerator
  - BaseImagePolicy interface
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from diffusion_policy_mlx.compat.schedulers import DDPMScheduler
from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_mlx.policy.base_image_policy import BaseImagePolicy
from diffusion_policy_mlx.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shape_meta():
    return {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }


def _make_policy(
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    num_inference_steps=3,
    num_train_timesteps=10,
    down_dims=(32, 64),
    n_groups=4,
):
    """Create a small policy for testing."""
    shape_meta = _make_shape_meta()
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=32,
        down_dims=down_dims,
        kernel_size=3,
        n_groups=n_groups,
        cond_predict_scale=True,
    )
    return policy


def _make_identity_normalizer():
    """Build an identity normalizer matching the default shape_meta."""
    normalizer = LinearNormalizer()
    # For obs (nested dict)
    normalizer.params_dict["obs"] = {
        "image": SingleFieldLinearNormalizer.create_identity(shape=(3,)),
        "agent_pos": SingleFieldLinearNormalizer.create_identity(shape=(2,)),
    }
    # For action
    normalizer.params_dict["action"] = SingleFieldLinearNormalizer.create_identity(shape=(2,))
    return normalizer


# ---------------------------------------------------------------------------
# LowdimMaskGenerator
# ---------------------------------------------------------------------------


class TestLowdimMaskGenerator:
    def test_shape(self):
        """Mask shape should match input shape."""
        gen = LowdimMaskGenerator(action_dim=2, obs_dim=10, max_n_obs_steps=3)
        mask = gen((4, 16, 12))
        assert mask.shape == (4, 16, 12)
        assert mask.dtype == mx.bool_

    def test_zero_obs_dim(self):
        """When obs_dim=0, mask should be all False."""
        gen = LowdimMaskGenerator(action_dim=2, obs_dim=0, max_n_obs_steps=2)
        mask = gen((4, 16, 2))
        assert mask.shape == (4, 16, 2)
        assert float(mx.sum(mask.astype(mx.int32))) == 0.0

    def test_obs_visible(self):
        """Obs dims should be visible for first n_obs timesteps."""
        gen = LowdimMaskGenerator(action_dim=2, obs_dim=3, max_n_obs_steps=2, fix_obs_steps=True)
        mask = gen((1, 8, 5))
        mask_np = np.array(mask)
        # Action dims (0, 1) should be all False
        assert mask_np[0, :, 0].sum() == 0
        assert mask_np[0, :, 1].sum() == 0
        # Obs dims (2, 3, 4) should be True for t=0, t=1
        assert mask_np[0, 0, 2]
        assert mask_np[0, 1, 2]
        assert not mask_np[0, 2, 2]

    def test_action_visible(self):
        """With action_visible=True, actions visible for t < obs_steps - 1."""
        gen = LowdimMaskGenerator(
            action_dim=2,
            obs_dim=3,
            max_n_obs_steps=3,
            fix_obs_steps=True,
            action_visible=True,
        )
        mask = gen((1, 8, 5))
        mask_np = np.array(mask)
        # Action dims visible for t < 2 (3 - 1)
        assert mask_np[0, 0, 0]
        assert mask_np[0, 1, 0]
        assert not mask_np[0, 2, 0]

    def test_assert_wrong_dim(self):
        """Should raise when D != action_dim + obs_dim."""
        gen = LowdimMaskGenerator(action_dim=2, obs_dim=3)
        with pytest.raises(AssertionError):
            gen((1, 8, 10))


# ---------------------------------------------------------------------------
# BaseImagePolicy
# ---------------------------------------------------------------------------


class TestBaseImagePolicy:
    def test_abstract(self):
        """predict_action and set_normalizer should raise NotImplementedError."""
        policy = BaseImagePolicy()
        with pytest.raises(NotImplementedError):
            policy.predict_action({})
        with pytest.raises(NotImplementedError):
            policy.set_normalizer(None)

    def test_reset(self):
        """reset() should not raise."""
        policy = BaseImagePolicy()
        policy.reset()


# ---------------------------------------------------------------------------
# DiffusionUnetHybridImagePolicy
# ---------------------------------------------------------------------------


class TestPolicyConstruction:
    def test_instantiation(self):
        """Policy should instantiate without error."""
        policy = _make_policy()
        assert isinstance(policy, BaseImagePolicy)
        assert isinstance(policy, nn.Module)

    def test_set_normalizer(self):
        """set_normalizer should work."""
        policy = _make_policy()
        normalizer = _make_identity_normalizer()
        policy.set_normalizer(normalizer)
        assert "action" in policy.normalizer


class TestPredictAction:
    def test_shapes(self):
        """predict_action should return correct shapes."""
        mx.random.seed(42)
        policy = _make_policy(
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=2,
            num_train_timesteps=5,
        )
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        B = 2
        obs = {
            "image": mx.random.normal((B, 2, 3, 96, 96)),
            "agent_pos": mx.random.normal((B, 2, 2)),
        }
        result = policy.predict_action(obs)
        mx.eval(result["action"], result["action_pred"])

        assert result["action"].shape == (B, 8, 2), (
            f"Expected (2, 8, 2), got {result['action'].shape}"
        )
        assert result["action_pred"].shape == (B, 16, 2), (
            f"Expected (2, 16, 2), got {result['action_pred'].shape}"
        )

    def test_no_nan(self):
        """Output should not contain NaN or Inf."""
        mx.random.seed(42)
        policy = _make_policy(num_inference_steps=2, num_train_timesteps=5)
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        obs = {
            "image": mx.random.normal((1, 2, 3, 96, 96)),
            "agent_pos": mx.random.normal((1, 2, 2)),
        }
        result = policy.predict_action(obs)
        action = result["action"]
        mx.eval(action)
        action_np = np.array(action)
        assert not np.any(np.isnan(action_np)), "NaN detected in output"
        assert not np.any(np.isinf(action_np)), "Inf detected in output"


class TestComputeLoss:
    def test_scalar(self):
        """compute_loss should return a scalar > 0."""
        mx.random.seed(42)
        policy = _make_policy()
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": {
                "image": mx.random.normal((2, 16, 3, 96, 96)),
                "agent_pos": mx.random.normal((2, 16, 2)),
            },
            "action": mx.random.normal((2, 16, 2)),
        }
        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"
        assert float(loss) > 0, f"Loss should be > 0, got {float(loss)}"

    def test_differentiable(self):
        """Gradients should flow through compute_loss."""
        mx.random.seed(42)
        policy = _make_policy()
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": {
                "image": mx.random.normal((2, 16, 3, 96, 96)),
                "agent_pos": mx.random.normal((2, 16, 2)),
            },
            "action": mx.random.normal((2, 16, 2)),
        }

        def loss_fn(model, batch):
            return model.compute_loss(batch)

        loss, grads = nn.value_and_grad(policy, loss_fn)(policy, batch)
        mx.eval(loss)

        assert float(loss) > 0, f"Loss should be > 0, got {float(loss)}"

        # Check that at least some gradients are non-zero
        flat_grads = nn.utils.tree_flatten(grads)
        has_nonzero = False
        for name, g in flat_grads:
            if g is not None and g.size > 0:
                mx.eval(g)
                if float(mx.max(mx.abs(g))) > 0:
                    has_nonzero = True
                    break
        assert has_nonzero, "All gradients are zero — no gradient flow!"
