"""Tests for PRD-06: Training Loop components.

Tests cover:
    - EMA decay schedule (matches upstream formula)
    - EMA step updates and copy_to
    - LR schedulers (cosine, linear, constant, constant_with_warmup)
    - get_scheduler factory
    - Checkpoint save/load round-trip
    - Training step: loss decreases on synthetic data
    - TrainConfig YAML serialization
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx.utils
import numpy as np
import pytest

from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_mlx.model.diffusion.ema_model import EMAModel
from diffusion_policy_mlx.model.common.lr_scheduler import (
    CosineAnnealingLR,
    ConstantLR,
    ConstantWithWarmupLR,
    LinearLR,
    get_scheduler,
)
from diffusion_policy_mlx.training.checkpoint import (
    TopKCheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from diffusion_policy_mlx.training.train_config import TrainConfig


# ---------------------------------------------------------------------------
# EMA Tests
# ---------------------------------------------------------------------------

class TestEMADecaySchedule:
    """EMA decay ramps from 0 toward ~0.9999 following upstream formula."""

    def test_decay_at_step_zero(self):
        """Decay should be 0.0 at step 0 (before update_after_step)."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model)
        assert ema.get_decay(0) == 0.0

    def test_decay_at_step_one(self):
        """Decay should be 0.0 at step 1 (step - update_after_step - 1 = 0)."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model)
        assert ema.get_decay(1) == 0.0

    def test_decay_increases_over_time(self):
        """Decay should increase monotonically."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model)
        decays = [ema.get_decay(s) for s in range(0, 10001, 100)]
        # Filter out the initial zero values
        nonzero = [d for d in decays if d > 0]
        assert len(nonzero) > 0
        for i in range(1, len(nonzero)):
            assert nonzero[i] >= nonzero[i - 1]

    def test_decay_at_high_step(self):
        """Decay should approach max_value (0.9999) at high steps."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model)
        decay = ema.get_decay(10000)
        assert decay > 0.99, f"Expected decay > 0.99 at step 10000, got {decay}"
        assert decay <= 0.9999, f"Expected decay <= 0.9999, got {decay}"

    def test_decay_respects_update_after_step(self):
        """No update when step < update_after_step."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model, update_after_step=100)
        # Steps 0 through 101 should return 0.0
        for s in range(102):
            assert ema.get_decay(s) == 0.0
        # Step 102 should be non-zero
        assert ema.get_decay(102) > 0.0

    def test_decay_formula_exact(self):
        """Verify the exact upstream formula: 1 - (1 + step/inv_gamma)^(-power)."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model, inv_gamma=1.0, power=2 / 3)
        step_val = 1000
        # step = max(0, optimization_step - update_after_step - 1)
        effective_step = step_val - 0 - 1  # 999
        expected = 1.0 - (1.0 + effective_step / 1.0) ** (-2 / 3)
        expected = max(0.0, min(expected, 0.9999))
        actual = ema.get_decay(step_val)
        assert abs(actual - expected) < 1e-10

    def test_decay_clamps_to_max(self):
        """Very high step values should clamp to max_value."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema = EMAModel(model, max_value=0.95)
        decay = ema.get_decay(1_000_000)
        assert decay == 0.95


class TestEMAStep:
    """EMA step() should update shadow parameters."""

    def test_ema_step_updates_params(self):
        """After modifying model and calling step, EMA params should change."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())

        # Use min_value > 0 so decay is nonzero from first step
        ema = EMAModel(model, update_after_step=0, min_value=0.5)

        # Record initial EMA params
        before = {}
        for k, v in ema.averaged_params.items():
            before[k] = np.array(v).copy()

        # Perturb model parameters significantly
        model_params = dict(mlx.utils.tree_flatten(model.parameters()))
        new_weights = []
        for key, param in model_params.items():
            new_weights.append((key, param + 10.0))
        model.load_weights(new_weights)
        mx.eval(model.parameters())

        # Step EMA
        ema.step(model)

        # Check that at least some EMA params changed
        changed = False
        for k in before:
            if k in ema.averaged_params:
                diff = np.abs(np.array(ema.averaged_params[k]) - before[k]).max()
                if diff > 1e-6:
                    changed = True
                    break
        assert changed, "EMA params did not change after step"

    def test_ema_step_count_increments(self):
        """Step count should increment after each step call."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())
        ema = EMAModel(model)
        assert ema.step_count == 0
        ema.step(model)
        assert ema.step_count == 1
        ema.step(model)
        assert ema.step_count == 2


class TestEMACopyTo:
    """EMA copy_to should transfer weights to a model."""

    def test_copy_to_transfers_weights(self):
        """After copy_to, model params should match EMA params."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())

        ema = EMAModel(model, min_value=0.9)

        # Perturb model
        model_params = dict(mlx.utils.tree_flatten(model.parameters()))
        new_weights = [(k, v + 5.0) for k, v in model_params.items()]
        model.load_weights(new_weights)
        mx.eval(model.parameters())

        # Step EMA a few times
        for _ in range(5):
            ema.step(model)

        # Create a fresh model and copy EMA weights to it
        eval_model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        ema.copy_to(eval_model)
        mx.eval(eval_model.parameters())

        # Verify eval_model has EMA params
        eval_params = dict(mlx.utils.tree_flatten(eval_model.parameters()))
        for key in ema.averaged_params:
            if key in eval_params:
                np.testing.assert_allclose(
                    np.array(eval_params[key]),
                    np.array(ema.averaged_params[key]),
                    atol=1e-7,
                )


class TestEMAStateDictRoundTrip:
    """EMA state_dict / load_state_dict round-trip."""

    def test_state_dict_round_trip(self):
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())
        ema = EMAModel(model, min_value=0.5)

        # Do some steps
        for _ in range(3):
            ema.step(model)

        state = ema.state_dict()

        # Create new EMA and restore
        ema2 = EMAModel(model)
        ema2.load_state_dict(state)

        assert ema2.step_count == ema.step_count
        for key in ema.averaged_params:
            np.testing.assert_allclose(
                np.array(ema2.averaged_params[key]),
                np.array(ema.averaged_params[key]),
                atol=1e-7,
            )


# ---------------------------------------------------------------------------
# LR Scheduler Tests
# ---------------------------------------------------------------------------

class TestCosineAnnealingLR:
    def test_warmup_increases(self):
        """LR should increase during warmup."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = CosineAnnealingLR(opt, num_training_steps=1000, num_warmup_steps=100)

        sched.step()
        lr1 = float(opt.learning_rate)

        for _ in range(99):
            sched.step()
        lr100 = float(opt.learning_rate)

        assert lr100 > lr1, f"LR at step 100 ({lr100}) should be > step 1 ({lr1})"

    def test_cosine_decreases(self):
        """LR should decrease during cosine phase."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = CosineAnnealingLR(opt, num_training_steps=1000, num_warmup_steps=100)

        for _ in range(100):
            sched.step()
        lr_at_warmup_end = float(opt.learning_rate)

        for _ in range(900):
            sched.step()
        lr_at_end = float(opt.learning_rate)

        assert lr_at_end < lr_at_warmup_end

    def test_warmup_reaches_base_lr(self):
        """At the end of warmup, LR should approximately equal base_lr."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = CosineAnnealingLR(opt, num_training_steps=1000, num_warmup_steps=100)

        for _ in range(100):
            sched.step()
        lr = float(opt.learning_rate)
        assert abs(lr - 1e-3) < 1e-6

    def test_end_approaches_min_lr(self):
        """At the end, LR should approach min_lr."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = CosineAnnealingLR(
            opt, num_training_steps=1000, num_warmup_steps=0, min_lr=1e-5
        )
        for _ in range(1000):
            sched.step()
        lr = float(opt.learning_rate)
        assert abs(lr - 1e-5) < 1e-6

    def test_no_warmup(self):
        """Without warmup, LR should start at base and decrease."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = CosineAnnealingLR(opt, num_training_steps=100, num_warmup_steps=0)
        sched.step()
        lr = float(opt.learning_rate)
        # After one step of cosine from base_lr, it should be slightly below
        assert lr <= 1e-3


class TestLinearLR:
    def test_warmup_increases(self):
        """LR should increase during warmup."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = LinearLR(opt, num_training_steps=1000, num_warmup_steps=100)

        sched.step()
        lr1 = float(opt.learning_rate)
        for _ in range(99):
            sched.step()
        lr100 = float(opt.learning_rate)
        assert lr100 > lr1

    def test_linear_decreases_to_zero(self):
        """LR should linearly decrease to 0."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = LinearLR(opt, num_training_steps=1000, num_warmup_steps=0)

        for _ in range(1000):
            sched.step()
        lr = float(opt.learning_rate)
        assert lr < 1e-6, f"Expected LR ~0.0, got {lr}"

    def test_monotonic_decrease_after_warmup(self):
        """LR should decrease monotonically after warmup."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = LinearLR(opt, num_training_steps=200, num_warmup_steps=20)

        for _ in range(20):
            sched.step()

        prev_lr = float(opt.learning_rate)
        for _ in range(180):
            sched.step()
            lr = float(opt.learning_rate)
            assert lr <= prev_lr + 1e-10  # allow tiny float error
            prev_lr = lr


class TestConstantLR:
    def test_lr_stays_constant(self):
        """LR should not change."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = ConstantLR(opt)
        for _ in range(100):
            sched.step()
        assert abs(float(opt.learning_rate) - 1e-3) < 1e-7


class TestConstantWithWarmupLR:
    def test_warmup_then_constant(self):
        """LR should increase during warmup, then stay constant."""
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = ConstantWithWarmupLR(opt, num_warmup_steps=50)

        sched.step()
        lr1 = float(opt.learning_rate)

        for _ in range(49):
            sched.step()
        lr50 = float(opt.learning_rate)
        assert lr50 > lr1

        # After warmup, should stay at base_lr
        for _ in range(100):
            sched.step()
        lr_later = float(opt.learning_rate)
        assert abs(lr_later - 1e-3) < 1e-7


class TestGetSchedulerFactory:
    """get_scheduler factory should return correct types."""

    def test_cosine(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = get_scheduler("cosine", opt, num_training_steps=1000, num_warmup_steps=100)
        assert isinstance(sched, CosineAnnealingLR)

    def test_linear(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = get_scheduler("linear", opt, num_training_steps=1000)
        assert isinstance(sched, LinearLR)

    def test_constant(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = get_scheduler("constant", opt)
        assert isinstance(sched, ConstantLR)

    def test_constant_with_warmup(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        sched = get_scheduler("constant_with_warmup", opt, num_warmup_steps=100)
        assert isinstance(sched, ConstantWithWarmupLR)

    def test_unknown_raises(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("nonexistent", opt)


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------

class TestCheckpointSaveLoad:
    """Checkpoint save/load round-trip."""

    def test_round_trip(self):
        """Weights should be preserved after save + load."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())
        ema = EMAModel(model, min_value=0.5)

        # Do some EMA steps
        for _ in range(3):
            ema.step(model)

        # Save original params for comparison
        original_params = {}
        for k, v in mlx.utils.tree_flatten(model.parameters()):
            original_params[k] = np.array(v).copy()

        original_ema_params = {}
        for k, v in ema.averaged_params.items():
            original_ema_params[k] = np.array(v).copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, ema, None, epoch=5, step=500, checkpoint_dir=tmpdir)

            # Create fresh model and EMA
            model2 = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
            mx.eval(model2.parameters())
            ema2 = EMAModel(model2)

            # Load
            ckpt_dir = list(Path(tmpdir).iterdir())[0]
            metadata = load_checkpoint(str(ckpt_dir), model2, ema2)

            assert metadata["epoch"] == 5
            assert metadata["step"] == 500

            # Verify model weights match
            restored_params = dict(mlx.utils.tree_flatten(model2.parameters()))
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

    def test_save_without_ema(self):
        """Should handle ema=None gracefully."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(model, None, None, epoch=0, step=0, checkpoint_dir=tmpdir)
            assert (path / "model.npz").exists()
            assert not (path / "ema.npz").exists()


class TestTopKCheckpointManager:
    def test_keeps_only_k(self):
        """Should prune checkpoints beyond k."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TopKCheckpointManager(save_dir=tmpdir, k=3, mode="min")

            # Save 5 checkpoints with different losses
            for i, loss in enumerate([0.5, 0.3, 0.8, 0.1, 0.4]):
                manager.save(
                    metric=loss,
                    policy=model, ema=None, optimizer=None,
                    epoch=i, step=i * 100,
                )

            # Should have exactly 3 checkpoints
            assert len(manager._checkpoints) == 3
            # Best should be 0.1
            assert manager.best_metric == 0.1

    def test_mode_max(self):
        """In 'max' mode, should keep highest metric checkpoints."""
        model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = TopKCheckpointManager(save_dir=tmpdir, k=2, mode="max")

            for i, score in enumerate([0.5, 0.9, 0.3, 0.7]):
                manager.save(
                    metric=score,
                    policy=model, ema=None, optimizer=None,
                    epoch=i, step=i * 100,
                )

            assert len(manager._checkpoints) == 2
            # Best (highest) should be 0.9
            assert manager.best_metric == 0.9


# ---------------------------------------------------------------------------
# TrainConfig Tests
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_default_values(self):
        """TrainConfig should have sensible defaults."""
        cfg = TrainConfig()
        assert cfg.batch_size == 64
        assert cfg.num_epochs == 300
        assert cfg.lr == 1e-4
        assert cfg.lr_scheduler == "cosine"

    def test_yaml_round_trip(self):
        """Config should survive YAML save/load round-trip."""
        cfg = TrainConfig(
            batch_size=32,
            num_epochs=10,
            lr=5e-4,
            down_dims=(128, 256),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            cfg.to_yaml(yaml_path)

            loaded = TrainConfig.from_yaml(yaml_path)
            assert loaded.batch_size == 32
            assert loaded.num_epochs == 10
            assert loaded.lr == 5e-4
            assert loaded.down_dims == (128, 256)

    def test_to_dict(self):
        """to_dict should return a plain dict."""
        cfg = TrainConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["batch_size"] == 64


# ---------------------------------------------------------------------------
# Training Step: Loss Decreases
# ---------------------------------------------------------------------------

class TestTrainingStepLossDecreases:
    """Verify that a training step on synthetic data reduces loss."""

    def test_loss_decreases_over_steps(self):
        """Loss should decrease over 100 training steps on fixed synthetic data.

        Uses ConditionalUnet1D directly with DDPM noise schedule.
        """
        from diffusion_policy_mlx.compat.schedulers import DDPMScheduler

        # Small model
        input_dim = 2
        horizon = 8
        batch_size = 4

        unet = ConditionalUnet1D(input_dim=input_dim, down_dims=[16, 32])
        mx.eval(unet.parameters())

        scheduler = DDPMScheduler(num_train_timesteps=100)
        optimizer = mlx.optimizers.Adam(learning_rate=1e-3)

        # Fixed synthetic data
        mx.random.seed(42)
        actions = mx.random.normal((batch_size, horizon, input_dim))
        noise = mx.random.normal((batch_size, horizon, input_dim))
        timesteps = mx.array([10, 30, 50, 70])

        # Precompute noisy actions (these are constant)
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)
        mx.eval(actions, noise, noisy_actions)

        def loss_fn(model, noisy_actions, timesteps, noise):
            pred = model(noisy_actions, timesteps)
            return mx.mean((pred - noise) ** 2)

        grad_fn = nn.value_and_grad(unet, loss_fn)

        losses = []
        for _ in range(100):
            loss, grads = grad_fn(unet, noisy_actions, timesteps, noise)
            optimizer.update(unet, grads)
            mx.eval(unet.parameters(), optimizer.state)
            losses.append(float(loss))

        # Loss should trend downward significantly
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f} "
            f"(ratio: {losses[-1]/losses[0]:.2f}, expected < 0.5)"
        )

    def test_single_training_step_finite_loss(self):
        """A single training step should produce a finite loss."""
        input_dim = 2
        unet = ConditionalUnet1D(input_dim=input_dim, down_dims=[16, 32])
        mx.eval(unet.parameters())

        mx.random.seed(123)
        x = mx.random.normal((2, 8, input_dim))
        t = mx.array([5, 10])

        def loss_fn(model, x, t):
            pred = model(x, t)
            return mx.mean(pred ** 2)

        loss, grads = nn.value_and_grad(unet, loss_fn)(unet, x, t)
        mx.eval(loss, grads)
        assert np.isfinite(float(loss))


# ---------------------------------------------------------------------------
# Collate batch test
# ---------------------------------------------------------------------------

class TestCollateBatch:
    def test_collate_flat(self):
        """Collate flat samples."""
        from diffusion_policy_mlx.training.train_diffusion import collate_batch

        samples = [
            {"action": np.array([1.0, 2.0]), "obs": np.array([3.0, 4.0])},
            {"action": np.array([5.0, 6.0]), "obs": np.array([7.0, 8.0])},
        ]
        batch = collate_batch(samples)
        assert batch["action"].shape == (2, 2)
        assert batch["obs"].shape == (2, 2)

    def test_collate_nested(self):
        """Collate nested dict samples."""
        from diffusion_policy_mlx.training.train_diffusion import collate_batch

        samples = [
            {"obs": {"image": np.zeros((3, 4, 4)), "pos": np.array([1.0, 2.0])}},
            {"obs": {"image": np.ones((3, 4, 4)), "pos": np.array([3.0, 4.0])}},
        ]
        batch = collate_batch(samples)
        assert batch["obs"]["image"].shape == (2, 3, 4, 4)
        assert batch["obs"]["pos"].shape == (2, 2)

    def test_collate_empty(self):
        """Collating empty list should return empty dict."""
        from diffusion_policy_mlx.training.train_diffusion import collate_batch

        assert collate_batch([]) == {}
