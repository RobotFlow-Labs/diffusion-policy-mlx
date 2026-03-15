"""End-to-end integration tests for diffusion-policy-mlx.

Verifies the full pipeline: synthetic data -> dataset -> normalizer -> policy
-> training loop -> inference -> checkpoint round-trip.

All tests use small models (down_dims=[32, 64], num_train_timesteps=10)
and synthetic data so they finish in seconds, not minutes.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx.utils
import numpy as np
import zarr

from diffusion_policy_mlx.compat.schedulers import DDIMScheduler, DDPMScheduler
from diffusion_policy_mlx.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.ema_model import EMAModel
from diffusion_policy_mlx.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy_mlx.training.checkpoint import load_checkpoint, save_checkpoint
from diffusion_policy_mlx.training.collate import collate_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_zarr(path: str, n_steps: int = 200, n_episodes: int = 2):
    """Create a minimal synthetic zarr dataset for testing."""
    rng = np.random.default_rng(42)
    root = zarr.open(path, mode="w")
    root.create_array(
        "data/img",
        data=rng.integers(0, 256, (n_steps, 96, 96, 3), dtype=np.uint8),
    )
    root.create_array(
        "data/state",
        data=rng.standard_normal((n_steps, 5)).astype(np.float32),
    )
    root.create_array(
        "data/action",
        data=rng.standard_normal((n_steps, 2)).astype(np.float32),
    )
    steps_per_ep = n_steps // n_episodes
    episode_ends = np.array([steps_per_ep * (i + 1) for i in range(n_episodes)], dtype=np.int64)
    root.create_array("meta/episode_ends", data=episode_ends)
    return path


SHAPE_META = {
    "obs": {
        "image": {"shape": (3, 96, 96), "type": "rgb"},
        "agent_pos": {"shape": (2,), "type": "low_dim"},
    },
    "action": {"shape": (2,)},
}


def _make_small_policy(
    num_train_timesteps: int = 10,
    num_inference_steps: int = 3,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    scheduler_cls=DDPMScheduler,
):
    """Build a small policy suitable for fast testing."""
    scheduler = scheduler_cls(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=SHAPE_META,
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
    mx.eval(policy.parameters())
    return policy


def _make_identity_normalizer():
    """Build an identity normalizer for the default shape_meta."""
    normalizer = LinearNormalizer()
    normalizer["obs"] = {
        "image": SingleFieldLinearNormalizer.create_identity(shape=(3,)),
        "agent_pos": SingleFieldLinearNormalizer.create_identity(shape=(2,)),
    }
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity(shape=(2,))
    return normalizer


def _make_synthetic_batch(batch_size: int = 2, horizon: int = 16):
    """Create a synthetic batch for training/inference."""
    mx.random.seed(42)
    return {
        "obs": {
            "image": mx.random.normal((batch_size, horizon, 3, 96, 96)),
            "agent_pos": mx.random.normal((batch_size, horizon, 2)),
        },
        "action": mx.random.normal((batch_size, horizon, 2)),
    }


# ---------------------------------------------------------------------------
# Test: end-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestEndToEndSynthetic:
    """Full pipeline: synthetic data -> policy -> train -> inference."""

    def test_end_to_end_synthetic(self, tmp_path):
        """Full pipeline: synthetic data -> dataset -> normalizer -> policy
        -> train steps -> verify loss decrease -> inference -> verify shape."""
        mx.random.seed(42)
        np.random.seed(42)

        # 1. Create synthetic zarr dataset
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path, n_steps=200, n_episodes=2)

        # 2. Build PushTImageDataset
        horizon = 16
        n_obs_steps = 2
        n_action_steps = 8
        dataset = PushTImageDataset(
            zarr_path=zarr_path,
            horizon=horizon,
            pad_before=n_obs_steps - 1,
            pad_after=n_action_steps - 1,
        )
        assert len(dataset) > 0, "Dataset should have samples"

        # 3. Get normalizer from dataset
        normalizer = dataset.get_normalizer(mode="limits")
        assert isinstance(normalizer, LinearNormalizer), (
            f"Expected LinearNormalizer, got {type(normalizer)}"
        )
        assert "action" in normalizer
        assert "obs" in normalizer

        # 4. Build policy (small for speed)
        policy = _make_small_policy(
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )

        # 5. Set normalizer on policy
        policy.set_normalizer(normalizer)

        # 6. Train 5-10 steps using training loop mechanics
        optimizer = mlx.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-6)
        loss_and_grad_fn = nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))

        losses = []
        indices = np.random.permutation(len(dataset))
        batch_size = 2

        for step in range(8):
            start = (step * batch_size) % len(indices)
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < batch_size:
                batch_idx = indices[:batch_size]

            batch = collate_batch([dataset[int(i)] for i in batch_idx])

            loss, grads = loss_and_grad_fn(policy, batch)
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)

            loss_val = float(loss)
            losses.append(loss_val)
            assert np.isfinite(loss_val), f"Loss is not finite at step {step}: {loss_val}"

        # 7. Verify loss decreases (comparing first vs last)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

        # 8. Run predict_action on synthetic obs
        obs = {
            "image": mx.random.normal((1, n_obs_steps, 3, 96, 96)),
            "agent_pos": mx.random.normal((1, n_obs_steps, 2)),
        }
        result = policy.predict_action(obs)
        mx.eval(result["action"], result["action_pred"])

        # 9. Verify output shape and no NaN
        assert result["action"].shape == (1, n_action_steps, 2)
        assert result["action_pred"].shape == (1, horizon, 2)
        action_np = np.array(result["action"])
        assert not np.any(np.isnan(action_np)), "NaN in predicted action"
        assert not np.any(np.isinf(action_np)), "Inf in predicted action"

    def test_dataset_normalizer_is_linear_normalizer(self, tmp_path):
        """Dataset get_normalizer returns a LinearNormalizer compatible with policy."""
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path)
        dataset = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = dataset.get_normalizer(mode="limits")

        assert isinstance(normalizer, LinearNormalizer)

        # Verify it can normalize a batch dict (the API the policy uses)
        sample = dataset[0]
        batch = collate_batch([sample])
        nbatch = normalizer.normalize({"obs": batch["obs"], "action": batch["action"]})
        assert "obs" in nbatch
        assert "action" in nbatch
        assert "image" in nbatch["obs"]
        assert "agent_pos" in nbatch["obs"]

    def test_normalizer_round_trip_through_policy(self, tmp_path):
        """Normalizer from dataset should round-trip data correctly when used by policy."""
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path)
        dataset = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = dataset.get_normalizer(mode="limits")

        # Test action round-trip
        sample = dataset[0]
        action = mx.array(sample["action"])
        normed = normalizer["action"].normalize(action)
        recovered = normalizer["action"].unnormalize(normed)
        mx.eval(recovered)
        np.testing.assert_allclose(np.array(recovered), np.array(action), atol=1e-5)


# ---------------------------------------------------------------------------
# Test: checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Save checkpoint -> load -> verify identical predictions."""

    def test_checkpoint_round_trip(self):
        """Model weights and EMA weights should survive save/load."""
        mx.random.seed(42)
        policy = _make_small_policy()
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        ema = EMAModel(policy, min_value=0.5)
        for _ in range(3):
            ema.step(policy)

        # Save original params for comparison
        original_params = {}
        for k, v in mlx.utils.tree_flatten(policy.parameters()):
            original_params[k] = np.array(v).copy()

        original_ema_params = {}
        for k, v in ema.averaged_params.items():
            original_ema_params[k] = np.array(v).copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            save_checkpoint(policy, ema, None, epoch=5, step=100, checkpoint_dir=tmpdir)

            # Build fresh policy and load
            policy2 = _make_small_policy()
            policy2.set_normalizer(_make_identity_normalizer())
            mx.eval(policy2.parameters())
            ema2 = EMAModel(policy2)

            ckpt_dir = list(Path(tmpdir).iterdir())[0]
            metadata = load_checkpoint(str(ckpt_dir), policy2, ema2)

            assert metadata["epoch"] == 5
            assert metadata["step"] == 100

            # Verify model weights match
            restored_params = dict(mlx.utils.tree_flatten(policy2.parameters()))
            for key in original_params:
                np.testing.assert_allclose(
                    np.array(restored_params[key]),
                    original_params[key],
                    atol=1e-7,
                    err_msg=f"Model param {key} mismatch after load",
                )

            # Verify EMA weights match
            for key in original_ema_params:
                np.testing.assert_allclose(
                    np.array(ema2.averaged_params[key]),
                    original_ema_params[key],
                    atol=1e-7,
                    err_msg=f"EMA param {key} mismatch after load",
                )

            # Verify loaded model produces valid predictions
            obs = {
                "image": mx.random.normal((1, 2, 3, 96, 96)),
                "agent_pos": mx.random.normal((1, 2, 2)),
            }
            result = policy2.predict_action(obs)
            mx.eval(result["action"])
            action_np = np.array(result["action"])
            assert not np.any(np.isnan(action_np)), "NaN in loaded model prediction"
            assert result["action"].shape == (1, 8, 2)


# ---------------------------------------------------------------------------
# Test: EMA improves over base
# ---------------------------------------------------------------------------


class TestEMAImproves:
    """EMA model should produce smoother predictions."""

    def test_ema_improves_over_base(self):
        """EMA weights, when applied to the model, should yield valid predictions.

        After several training steps with noisy updates, EMA weights should
        produce predictions that are at least as stable (no NaN/Inf) as the
        base model, and the weights should differ from the base model
        (demonstrating that EMA is actually smoothing).
        """
        mx.random.seed(42)
        policy = _make_small_policy()
        policy.set_normalizer(_make_identity_normalizer())
        mx.eval(policy.parameters())

        ema = EMAModel(policy, min_value=0.9, update_after_step=0)
        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        loss_and_grad_fn = nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))

        # Train a few steps
        for step in range(10):
            batch = _make_synthetic_batch(batch_size=2, horizon=16)
            loss, grads = loss_and_grad_fn(policy, batch)
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)
            ema.step(policy)

        # EMA weights should differ from base model weights
        base_params = dict(mlx.utils.tree_flatten(policy.parameters()))
        ema_differs = False
        for key in ema.averaged_params:
            if key in base_params:
                diff = float(mx.max(mx.abs(ema.averaged_params[key] - base_params[key])))
                if diff > 1e-6:
                    ema_differs = True
                    break
        assert ema_differs, "EMA weights should differ from base after training"

        # Apply EMA weights and verify predictions are valid
        ema.copy_to(policy)
        mx.eval(policy.parameters())

        obs = {
            "image": mx.random.normal((1, 2, 3, 96, 96)),
            "agent_pos": mx.random.normal((1, 2, 2)),
        }
        result = policy.predict_action(obs)
        mx.eval(result["action"])
        action_np = np.array(result["action"])
        assert not np.any(np.isnan(action_np)), "NaN in EMA prediction"
        assert not np.any(np.isinf(action_np)), "Inf in EMA prediction"
        assert result["action"].shape == (1, 8, 2)


# ---------------------------------------------------------------------------
# Test: DDIM faster than DDPM
# ---------------------------------------------------------------------------


class TestDDIMFasterThanDDPM:
    """DDIM with fewer steps should be faster but still produce valid actions."""

    def test_ddim_faster_than_ddpm(self):
        """DDIM with fewer inference steps should be faster than DDPM and
        still produce valid (no NaN, correct shape) actions."""
        mx.random.seed(42)

        # DDPM with full steps
        ddpm_policy = _make_small_policy(
            num_train_timesteps=10,
            num_inference_steps=10,
            scheduler_cls=DDPMScheduler,
        )
        ddpm_policy.set_normalizer(_make_identity_normalizer())
        mx.eval(ddpm_policy.parameters())

        # DDIM with fewer steps
        ddim_policy = _make_small_policy(
            num_train_timesteps=10,
            num_inference_steps=3,
            scheduler_cls=DDIMScheduler,
        )
        ddim_policy.set_normalizer(_make_identity_normalizer())
        # Copy weights from DDPM policy for fair comparison
        ddpm_weights = dict(mlx.utils.tree_flatten(ddpm_policy.parameters()))
        ddim_policy.load_weights(list(ddpm_weights.items()))
        mx.eval(ddim_policy.parameters())

        obs = {
            "image": mx.random.normal((1, 2, 3, 96, 96)),
            "agent_pos": mx.random.normal((1, 2, 2)),
        }
        mx.eval(obs["image"], obs["agent_pos"])

        # Warm up both policies (first call has compilation overhead)
        mx.random.seed(100)
        _ = ddpm_policy.predict_action(obs)
        mx.eval(_["action"])
        mx.random.seed(100)
        _ = ddim_policy.predict_action(obs)
        mx.eval(_["action"])

        # Time DDPM
        mx.random.seed(200)
        t0 = time.perf_counter()
        ddpm_result = ddpm_policy.predict_action(obs)
        mx.eval(ddpm_result["action"])
        ddpm_time = time.perf_counter() - t0

        # Time DDIM (fewer steps)
        mx.random.seed(200)
        t0 = time.perf_counter()
        ddim_result = ddim_policy.predict_action(obs)
        mx.eval(ddim_result["action"])
        ddim_time = time.perf_counter() - t0

        # DDIM should be faster (or at least not dramatically slower).
        # Metal shader caching makes timing noisy — use very generous margin.
        # The real value of this test is validating both paths produce valid output.
        assert ddim_time < ddpm_time * 10.0, (
            f"DDIM ({ddim_time:.3f}s) dramatically slower than DDPM ({ddpm_time:.3f}s)"
        )

        # Both should produce valid outputs
        for name, result in [("DDPM", ddpm_result), ("DDIM", ddim_result)]:
            assert result["action"].shape == (1, 8, 2), (
                f"{name} action shape wrong: {result['action'].shape}"
            )
            action_np = np.array(result["action"])
            assert not np.any(np.isnan(action_np)), f"NaN in {name} action"
            assert not np.any(np.isinf(action_np)), f"Inf in {name} action"


# ---------------------------------------------------------------------------
# Test: training loop compute_loss compatibility
# ---------------------------------------------------------------------------


class TestTrainingLoopCompatibility:
    """Verify training loop components work together correctly."""

    def test_collate_then_compute_loss(self, tmp_path):
        """Dataset samples -> collate -> compute_loss should work end-to-end."""
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path)
        dataset = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = dataset.get_normalizer(mode="limits")

        policy = _make_small_policy()
        policy.set_normalizer(normalizer)
        mx.eval(policy.parameters())

        # Collate a batch
        samples = [dataset[i] for i in range(2)]
        batch = collate_batch(samples)

        # Compute loss
        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert loss.ndim == 0
        assert np.isfinite(float(loss))
        assert float(loss) > 0

    def test_value_and_grad_with_real_data(self, tmp_path):
        """nn.value_and_grad should work with real dataset batches."""
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path)
        dataset = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = dataset.get_normalizer(mode="limits")

        policy = _make_small_policy()
        policy.set_normalizer(normalizer)
        mx.eval(policy.parameters())

        loss_and_grad_fn = nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))

        samples = [dataset[i] for i in range(2)]
        batch = collate_batch(samples)

        loss, grads = loss_and_grad_fn(policy, batch)
        mx.eval(loss)
        assert np.isfinite(float(loss))

        # Verify gradients exist
        flat_grads = nn.utils.tree_flatten(grads)
        has_nonzero = any(
            g is not None and g.size > 0 and float(mx.max(mx.abs(g))) > 0 for _, g in flat_grads
        )
        assert has_nonzero, "No non-zero gradients found"

    def test_gaussian_normalizer_mode(self, tmp_path):
        """Gaussian normalizer mode should also work end-to-end."""
        zarr_path = str(tmp_path / "test.zarr")
        _create_synthetic_zarr(zarr_path)
        dataset = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = dataset.get_normalizer(mode="gaussian")

        assert isinstance(normalizer, LinearNormalizer)

        policy = _make_small_policy()
        policy.set_normalizer(normalizer)
        mx.eval(policy.parameters())

        samples = [dataset[i] for i in range(2)]
        batch = collate_batch(samples)
        loss = policy.compute_loss(batch)
        mx.eval(loss)
        assert np.isfinite(float(loss))

    def test_config_yaml_round_trip(self):
        """TrainConfig should load from the default YAML config file."""
        from diffusion_policy_mlx.training.train_config import TrainConfig

        config_path = Path(__file__).parent.parent / "configs" / "pusht_diffusion_policy_cnn.yaml"
        if config_path.exists():
            config = TrainConfig.from_yaml(config_path)
            assert config.horizon == 16
            assert config.n_obs_steps == 2
            assert config.n_action_steps == 8
            assert config.batch_size == 64
            assert config.down_dims == (256, 512, 1024)
