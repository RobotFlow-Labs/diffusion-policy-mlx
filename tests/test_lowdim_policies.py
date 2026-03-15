"""Tests for low-dim policies and image-only policy.

Tests:
  - BaseLowdimPolicy abstract interface
  - DiffusionUnetLowdimPolicy: predict_action shape, compute_loss scalar, gradient flow
  - DiffusionUnetImagePolicy: predict_action shape, compute_loss scalar
  - PushTLowdimDataset: shapes, normalizer, round-trip (synthetic data)
  - Synthetic data training for each policy
"""


import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np
import pytest

from diffusion_policy_mlx.compat.schedulers import DDPMScheduler
from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_mlx.policy.base_image_policy import BaseImagePolicy
from diffusion_policy_mlx.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy_mlx.policy.diffusion_unet_image_policy import (
    DiffusionUnetImagePolicy,
)
from diffusion_policy_mlx.policy.diffusion_unet_lowdim_policy import (
    DiffusionUnetLowdimPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 5
ACTION_DIM = 2
HORIZON = 16
N_OBS_STEPS = 2
N_ACTION_STEPS = 8


def _make_scheduler(num_train_timesteps=10):
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


def _make_lowdim_policy(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    horizon=HORIZON,
    n_obs_steps=N_OBS_STEPS,
    n_action_steps=N_ACTION_STEPS,
    obs_as_global_cond=True,
    num_inference_steps=3,
    num_train_timesteps=10,
):
    """Create a small low-dim policy for testing."""
    scheduler = _make_scheduler(num_train_timesteps)

    if obs_as_global_cond:
        input_dim = action_dim
        global_cond_dim = obs_dim * n_obs_steps
    else:
        input_dim = action_dim + obs_dim
        global_cond_dim = None

    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        cond_predict_scale=True,
    )

    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=obs_as_global_cond,
    )
    return policy


def _make_lowdim_normalizer(obs_dim=OBS_DIM, action_dim=ACTION_DIM):
    """Build an identity normalizer for low-dim policy."""
    normalizer = LinearNormalizer()
    normalizer["obs"] = SingleFieldLinearNormalizer.create_identity(shape=(obs_dim,))
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity(shape=(action_dim,))
    return normalizer


def _make_image_only_shape_meta():
    """Shape meta with only image observations (no agent_pos)."""
    return {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
        },
        "action": {"shape": (2,)},
    }


def _make_image_only_policy(
    horizon=HORIZON,
    n_action_steps=N_ACTION_STEPS,
    n_obs_steps=N_OBS_STEPS,
    num_inference_steps=2,
    num_train_timesteps=5,
):
    """Create a small image-only policy for testing."""
    shape_meta = _make_image_only_shape_meta()
    scheduler = _make_scheduler(num_train_timesteps)
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        cond_predict_scale=True,
    )
    return policy


def _make_image_only_normalizer():
    """Build an identity normalizer for image-only policy."""
    normalizer = LinearNormalizer()
    normalizer["obs"] = {
        "image": SingleFieldLinearNormalizer.create_identity(shape=(3,)),
    }
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity(shape=(2,))
    return normalizer


# ---------------------------------------------------------------------------
# BaseLowdimPolicy
# ---------------------------------------------------------------------------


class TestBaseLowdimPolicy:
    def test_abstract(self):
        """predict_action and set_normalizer should raise NotImplementedError."""
        policy = BaseLowdimPolicy()
        with pytest.raises(NotImplementedError):
            policy.predict_action({})
        with pytest.raises(NotImplementedError):
            policy.set_normalizer(None)

    def test_reset(self):
        """reset() should not raise."""
        policy = BaseLowdimPolicy()
        policy.reset()

    def test_is_nn_module(self):
        """BaseLowdimPolicy should be an nn.Module."""
        policy = BaseLowdimPolicy()
        assert isinstance(policy, nn.Module)


# ---------------------------------------------------------------------------
# DiffusionUnetLowdimPolicy
# ---------------------------------------------------------------------------


class TestLowdimPolicyConstruction:
    def test_instantiation(self):
        """Policy should instantiate without error."""
        policy = _make_lowdim_policy()
        assert isinstance(policy, BaseLowdimPolicy)
        assert isinstance(policy, nn.Module)

    def test_set_normalizer(self):
        """set_normalizer should work."""
        policy = _make_lowdim_policy()
        normalizer = _make_lowdim_normalizer()
        policy.set_normalizer(normalizer)
        assert "action" in policy.normalizer
        assert "obs" in policy.normalizer

    def test_invalid_cond_combination(self):
        """Cannot use both local and global conditioning."""
        with pytest.raises(AssertionError):
            scheduler = _make_scheduler()
            model = ConditionalUnet1D(
                input_dim=ACTION_DIM,
                global_cond_dim=OBS_DIM * N_OBS_STEPS,
                diffusion_step_embed_dim=32,
                down_dims=(32,),
                kernel_size=3,
                n_groups=4,
            )
            DiffusionUnetLowdimPolicy(
                model=model,
                noise_scheduler=scheduler,
                horizon=HORIZON,
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                n_action_steps=N_ACTION_STEPS,
                n_obs_steps=N_OBS_STEPS,
                obs_as_local_cond=True,
                obs_as_global_cond=True,
            )


class TestLowdimPredictAction:
    def test_shapes_global_cond(self):
        """predict_action should return correct shapes with global cond."""
        mx.random.seed(42)
        policy = _make_lowdim_policy(
            obs_as_global_cond=True,
            num_inference_steps=2,
            num_train_timesteps=5,
        )
        policy.set_normalizer(_make_lowdim_normalizer())
        mx.eval(policy.parameters())

        B = 2
        obs_dict = {"obs": mx.random.normal((B, N_OBS_STEPS, OBS_DIM))}
        result = policy.predict_action(obs_dict)
        mx.eval(result["action"], result["action_pred"])

        assert result["action"].shape == (B, N_ACTION_STEPS, ACTION_DIM), (
            f"Expected ({B}, {N_ACTION_STEPS}, {ACTION_DIM}), got {result['action'].shape}"
        )
        assert result["action_pred"].shape == (B, HORIZON, ACTION_DIM), (
            f"Expected ({B}, {HORIZON}, {ACTION_DIM}), got {result['action_pred'].shape}"
        )

    def test_no_nan(self):
        """Output should not contain NaN or Inf."""
        mx.random.seed(42)
        policy = _make_lowdim_policy(num_inference_steps=2, num_train_timesteps=5)
        policy.set_normalizer(_make_lowdim_normalizer())
        mx.eval(policy.parameters())

        obs_dict = {"obs": mx.random.normal((1, N_OBS_STEPS, OBS_DIM))}
        result = policy.predict_action(obs_dict)
        action = result["action"]
        mx.eval(action)
        action_np = np.array(action)
        assert not np.any(np.isnan(action_np)), "NaN detected in output"
        assert not np.any(np.isinf(action_np)), "Inf detected in output"


class TestLowdimComputeLoss:
    def test_scalar(self):
        """compute_loss should return a scalar > 0."""
        mx.random.seed(42)
        policy = _make_lowdim_policy()
        policy.set_normalizer(_make_lowdim_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": mx.random.normal((2, HORIZON, OBS_DIM)),
            "action": mx.random.normal((2, HORIZON, ACTION_DIM)),
        }
        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"
        assert float(loss) > 0, f"Loss should be > 0, got {float(loss)}"

    def test_differentiable(self):
        """Gradients should flow through compute_loss."""
        mx.random.seed(42)
        policy = _make_lowdim_policy()
        policy.set_normalizer(_make_lowdim_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": mx.random.normal((2, HORIZON, OBS_DIM)),
            "action": mx.random.normal((2, HORIZON, ACTION_DIM)),
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


class TestLowdimSyntheticTraining:
    def test_training_runs(self):
        """Multiple optimizer steps should run without error."""
        mx.random.seed(42)
        policy = _make_lowdim_policy(
            obs_as_global_cond=True,
            num_train_timesteps=10,
        )
        policy.set_normalizer(_make_lowdim_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": mx.random.normal((4, HORIZON, OBS_DIM)),
            "action": mx.random.normal((4, HORIZON, ACTION_DIM)),
        }

        def loss_fn(model, batch):
            return model.compute_loss(batch)

        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        losses = []
        for step in range(5):
            loss, grads = nn.value_and_grad(policy, loss_fn)(policy, batch)
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state, loss)
            losses.append(float(loss))

        # All losses should be finite positive numbers
        for i, loss_val in enumerate(losses):
            assert np.isfinite(loss_val), f"Loss at step {i} is not finite: {loss_val}"
            assert loss_val > 0, f"Loss at step {i} should be > 0, got {loss_val}"


# ---------------------------------------------------------------------------
# DiffusionUnetImagePolicy
# ---------------------------------------------------------------------------


class TestImageOnlyPolicyConstruction:
    def test_instantiation(self):
        """Policy should instantiate without error."""
        policy = _make_image_only_policy()
        assert isinstance(policy, BaseImagePolicy)
        assert isinstance(policy, nn.Module)

    def test_set_normalizer(self):
        """set_normalizer should work."""
        policy = _make_image_only_policy()
        normalizer = _make_image_only_normalizer()
        policy.set_normalizer(normalizer)
        assert "action" in policy.normalizer


class TestImageOnlyPredictAction:
    def test_shapes(self):
        """predict_action should return correct shapes."""
        mx.random.seed(42)
        policy = _make_image_only_policy(
            num_inference_steps=2,
            num_train_timesteps=5,
        )
        policy.set_normalizer(_make_image_only_normalizer())
        mx.eval(policy.parameters())

        B = 2
        obs = {
            "image": mx.random.normal((B, N_OBS_STEPS, 3, 96, 96)),
        }
        result = policy.predict_action(obs)
        mx.eval(result["action"], result["action_pred"])

        assert result["action"].shape == (B, N_ACTION_STEPS, 2), (
            f"Expected ({B}, {N_ACTION_STEPS}, 2), got {result['action'].shape}"
        )
        assert result["action_pred"].shape == (B, HORIZON, 2), (
            f"Expected ({B}, {HORIZON}, 2), got {result['action_pred'].shape}"
        )


class TestImageOnlyComputeLoss:
    def test_scalar(self):
        """compute_loss should return a scalar > 0."""
        mx.random.seed(42)
        policy = _make_image_only_policy()
        policy.set_normalizer(_make_image_only_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": {
                "image": mx.random.normal((2, HORIZON, 3, 96, 96)),
            },
            "action": mx.random.normal((2, HORIZON, 2)),
        }
        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"
        assert float(loss) > 0, f"Loss should be > 0, got {float(loss)}"

    def test_differentiable(self):
        """Gradients should flow through compute_loss."""
        mx.random.seed(42)
        policy = _make_image_only_policy()
        policy.set_normalizer(_make_image_only_normalizer())
        mx.eval(policy.parameters())

        batch = {
            "obs": {
                "image": mx.random.normal((2, HORIZON, 3, 96, 96)),
            },
            "action": mx.random.normal((2, HORIZON, 2)),
        }

        def loss_fn(model, batch):
            return model.compute_loss(batch)

        loss, grads = nn.value_and_grad(policy, loss_fn)(policy, batch)
        mx.eval(loss)

        assert float(loss) > 0, f"Loss should be > 0, got {float(loss)}"

        flat_grads = nn.utils.tree_flatten(grads)
        has_nonzero = False
        for name, g in flat_grads:
            if g is not None and g.size > 0:
                mx.eval(g)
                if float(mx.max(mx.abs(g))) > 0:
                    has_nonzero = True
                    break
        assert has_nonzero, "All gradients are zero — no gradient flow!"


# ---------------------------------------------------------------------------
# PushTLowdimDataset (synthetic zarr)
# ---------------------------------------------------------------------------


def _create_synthetic_zarr(tmp_path: str, n_episodes: int = 3, ep_length: int = 50):
    """Create a synthetic PushT zarr archive for testing."""
    import zarr

    root = zarr.open(tmp_path, mode="w")
    total = n_episodes * ep_length
    rng = np.random.default_rng(42)

    # Create data arrays matching PushT format
    root.create_array(
        "data/keypoint",
        data=rng.standard_normal((total, 9, 2)).astype(np.float32),
    )
    root.create_array(
        "data/state",
        data=rng.standard_normal((total, 5)).astype(np.float32),
    )
    root.create_array(
        "data/action",
        data=rng.standard_normal((total, 2)).astype(np.float32),
    )

    episode_ends = np.arange(1, n_episodes + 1) * ep_length
    root.create_array("meta/episode_ends", data=episode_ends.astype(np.int64))

    return tmp_path


class TestPushTLowdimDataset:
    @pytest.fixture
    def zarr_path(self, tmp_path):
        return _create_synthetic_zarr(str(tmp_path / "test.zarr"))

    def test_len(self, zarr_path):
        """Dataset length should be > 0."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        ds = PushTLowdimDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        assert len(ds) > 0

    def test_shapes(self, zarr_path):
        """Each sample should have correct shapes."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        horizon = 16
        ds = PushTLowdimDataset(zarr_path, horizon=horizon, pad_before=1, pad_after=7)
        sample = ds[0]

        assert "obs" in sample
        assert "action" in sample

        # obs = keypoints (9*2=18) + agent_pos (2) = 20
        assert sample["obs"].shape == (horizon, 20), (
            f"Expected obs shape ({horizon}, 20), got {sample['obs'].shape}"
        )
        assert sample["action"].shape == (horizon, 2), (
            f"Expected action shape ({horizon}, 2), got {sample['action'].shape}"
        )

    def test_dtypes(self, zarr_path):
        """Data should be float32."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        ds = PushTLowdimDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        sample = ds[0]
        assert sample["obs"].dtype == np.float32
        assert sample["action"].dtype == np.float32

    def test_normalizer(self, zarr_path):
        """get_normalizer should return a valid normalizer."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        ds = PushTLowdimDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")

        assert "obs" in normalizer
        assert "action" in normalizer

        # Test round-trip
        sample = ds[0]
        obs_mx = mx.array(sample["obs"])
        norm_obs = normalizer["obs"].normalize(obs_mx)
        unnorm_obs = normalizer["obs"].unnormalize(norm_obs)
        mx.eval(unnorm_obs)
        np.testing.assert_allclose(
            np.array(unnorm_obs), sample["obs"], atol=1e-4
        )

    def test_validation_split(self, zarr_path):
        """get_validation_dataset should return a valid dataset."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        ds = PushTLowdimDataset(
            zarr_path, horizon=16, pad_before=1, pad_after=7, val_ratio=0.33
        )
        val_ds = ds.get_validation_dataset()
        # With 3 episodes and 0.33 ratio, 1 should be val
        assert len(val_ds) > 0
        sample = val_ds[0]
        assert "obs" in sample
        assert "action" in sample

    def test_get_all_actions(self, zarr_path):
        """get_all_actions should return all actions."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        ds = PushTLowdimDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        actions = ds.get_all_actions()
        assert actions.shape == (150, 2)  # 3 episodes * 50 steps
        assert actions.dtype == np.float32


class TestPushTLowdimWithPolicy:
    """Test that a PushTLowdimDataset can be used with the lowdim policy."""

    @pytest.fixture
    def zarr_path(self, tmp_path):
        return _create_synthetic_zarr(str(tmp_path / "test.zarr"))

    def test_end_to_end(self, zarr_path):
        """Dataset -> normalizer -> policy compute_loss should work."""
        from diffusion_policy_mlx.dataset.pusht_lowdim_dataset import PushTLowdimDataset

        mx.random.seed(42)

        ds = PushTLowdimDataset(zarr_path, horizon=HORIZON, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")

        # obs_dim = 20 (18 keypoints + 2 agent_pos)
        obs_dim = 20
        policy = _make_lowdim_policy(obs_dim=obs_dim, obs_as_global_cond=True)
        policy.set_normalizer(normalizer)
        mx.eval(policy.parameters())

        # Create a batch from dataset
        samples = [ds[i] for i in range(2)]
        batch = {
            "obs": mx.array(np.stack([s["obs"] for s in samples])),
            "action": mx.array(np.stack([s["action"] for s in samples])),
        }

        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0
        assert float(loss) > 0
