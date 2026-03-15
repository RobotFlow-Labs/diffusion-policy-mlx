"""Tests for common utilities, training validator, wandb logger, and gradient clipping.

Tests cover:
    - dict_apply on nested dicts
    - dict_apply with mx.array values
    - dict_apply_split and dict_apply_reduce
    - JsonLogger write/read round-trip
    - param_count on known model
    - pad_remaining_dims
    - TrainingValidator runs without error
    - WandbLogger disabled mode (no wandb import needed)
    - Gradient clipping works
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx.utils
import numpy as np

from diffusion_policy_mlx.common.dict_util import (
    dict_apply,
    dict_apply_reduce,
    dict_apply_split,
)
from diffusion_policy_mlx.common.json_logger import JsonLogger
from diffusion_policy_mlx.common.pytorch_util import (
    optimizer_to,
    pad_remaining_dims,
    param_count,
)
from diffusion_policy_mlx.training.train_diffusion import clip_grad_norm
from diffusion_policy_mlx.training.validator import TrainingValidator
from diffusion_policy_mlx.training.wandb_logger import WandbLogger

# ---------------------------------------------------------------------------
# dict_apply Tests
# ---------------------------------------------------------------------------


class TestDictApply:
    """dict_apply should recursively transform leaves."""

    def test_flat_dict(self):
        """Apply func to flat dict values."""
        d = {"a": 1, "b": 2, "c": 3}
        result = dict_apply(d, lambda x: x * 10)
        assert result == {"a": 10, "b": 20, "c": 30}

    def test_nested_dict(self):
        """Apply func to nested dict values."""
        d = {"x": {"y": 1, "z": 2}, "w": 3}
        result = dict_apply(d, lambda x: x + 100)
        assert result == {"x": {"y": 101, "z": 102}, "w": 103}

    def test_deeply_nested(self):
        """Apply func to deeply nested structure."""
        d = {"a": {"b": {"c": 5}}}
        result = dict_apply(d, lambda x: x * 2)
        assert result == {"a": {"b": {"c": 10}}}

    def test_with_mx_arrays(self):
        """Apply func to mx.array leaf values."""
        d = {
            "obs": {
                "image": mx.zeros((2, 3)),
                "pos": mx.ones((4,)),
            },
            "action": mx.ones((2,)) * 3,
        }
        result = dict_apply(d, lambda x: x + 1)
        mx.eval(result["obs"]["image"], result["obs"]["pos"], result["action"])

        np.testing.assert_allclose(np.array(result["obs"]["image"]), 1.0)
        np.testing.assert_allclose(np.array(result["obs"]["pos"]), 2.0)
        np.testing.assert_allclose(np.array(result["action"]), 4.0)

    def test_preserves_structure(self):
        """Result should have the exact same key structure."""
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        result = dict_apply(d, lambda x: x)
        assert set(result.keys()) == {"a", "d"}
        assert set(result["a"].keys()) == {"b", "c"}

    def test_empty_dict(self):
        """Apply on empty dict returns empty dict."""
        assert dict_apply({}, lambda x: x) == {}


class TestDictApplySplit:
    """dict_apply_split should transpose dict-of-dicts."""

    def test_basic_split(self):
        """Split values and transpose."""
        d = {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}
        result = dict_apply_split(d, lambda v: {"first": v[:2], "second": v[2:]})
        assert result["first"]["x"] == [1, 2]
        assert result["second"]["y"] == [7, 8]

    def test_with_numpy(self):
        """Split numpy arrays."""
        d = {"a": np.array([1, 2, 3, 4]), "b": np.array([5, 6, 7, 8])}
        result = dict_apply_split(d, lambda v: {"lo": v[:2], "hi": v[2:]})
        np.testing.assert_array_equal(result["lo"]["a"], [1, 2])
        np.testing.assert_array_equal(result["hi"]["b"], [7, 8])


class TestDictApplyReduce:
    """dict_apply_reduce should reduce a list of dicts."""

    def test_mean_reduce(self):
        """Reduce with mean."""
        dicts = [
            {"loss": 0.5, "acc": 0.8},
            {"loss": 0.3, "acc": 0.9},
        ]
        result = dict_apply_reduce(dicts, lambda vals: sum(vals) / len(vals))
        assert abs(result["loss"] - 0.4) < 1e-10
        assert abs(result["acc"] - 0.85) < 1e-10

    def test_sum_reduce(self):
        """Reduce with sum."""
        dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = dict_apply_reduce(dicts, sum)
        assert result == {"a": 4, "b": 6}

    def test_empty_list(self):
        """Empty list returns empty dict."""
        assert dict_apply_reduce([], lambda x: x) == {}


# ---------------------------------------------------------------------------
# JsonLogger Tests
# ---------------------------------------------------------------------------


class TestJsonLogger:
    """JsonLogger should write/read JSONL entries."""

    def test_write_read_round_trip(self):
        """Written entries should be readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            with JsonLogger(path) as logger:
                logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
                logger.log({"loss": 0.3, "lr": 5e-5}, step=200)

            entries = JsonLogger.read(path)
            assert len(entries) == 2
            assert entries[0]["step"] == 100
            assert entries[0]["loss"] == 0.5
            assert entries[1]["step"] == 200
            assert abs(entries[1]["lr"] - 5e-5) < 1e-10

    def test_append_mode(self):
        """Multiple open/close cycles should append, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            logger = JsonLogger(path)
            logger.start()
            logger.log({"loss": 0.5}, step=1)
            logger.stop()

            logger2 = JsonLogger(path)
            logger2.start()
            logger2.log({"loss": 0.3}, step=2)
            logger2.stop()

            entries = JsonLogger.read(path)
            assert len(entries) == 2

    def test_filters_non_numeric(self):
        """Non-numeric values should be filtered out by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            with JsonLogger(path) as logger:
                logger.log({"loss": 0.5, "name": "test", "count": 42}, step=1)

            entries = JsonLogger.read(path)
            assert len(entries) == 1
            assert "loss" in entries[0]
            assert "count" in entries[0]
            assert "name" not in entries[0]

    def test_last_log_property(self):
        """last_log should return the most recent entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            with JsonLogger(path) as logger:
                logger.log({"loss": 0.5}, step=1)
                logger.log({"loss": 0.3}, step=2)
                assert logger.last_log["loss"] == 0.3
                assert logger.last_log["step"] == 2

    def test_read_nonexistent_file(self):
        """Reading a nonexistent file returns empty list."""
        entries = JsonLogger.read("/nonexistent/path.jsonl")
        assert entries == []

    def test_numpy_scalar_serialization(self):
        """numpy scalar types should be serialized correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            with JsonLogger(path) as logger:
                logger.log({"loss": np.float64(0.5), "epoch": np.int64(3)}, step=1)

            entries = JsonLogger.read(path)
            assert entries[0]["loss"] == 0.5
            assert entries[0]["epoch"] == 3

    def test_auto_write_without_start(self):
        """log() without start() should still write (auto-open)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonLogger(path)
            logger.log({"loss": 0.5}, step=1)

            entries = JsonLogger.read(path)
            assert len(entries) == 1


# ---------------------------------------------------------------------------
# pytorch_util Tests
# ---------------------------------------------------------------------------


class TestParamCount:
    """param_count should return total number of scalar parameters."""

    def test_linear(self):
        """Linear(10, 5) has 10*5 + 5 = 55 params."""
        model = nn.Linear(10, 5)
        mx.eval(model.parameters())
        assert param_count(model) == 55

    def test_nested_model(self):
        """Count params in a model with multiple layers."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 8)  # 4*8 + 8 = 40
                self.l2 = nn.Linear(8, 2)  # 8*2 + 2 = 18

            def __call__(self, x):
                return self.l2(nn.relu(self.l1(x)))

        model = TinyModel()
        mx.eval(model.parameters())
        assert param_count(model) == 40 + 18

    def test_zero_param_model(self):
        """Model with no parameters should return 0."""

        class NoParamModule(nn.Module):
            def __call__(self, x):
                return x

        model = NoParamModule()
        assert param_count(model) == 0


class TestPadRemainingDims:
    """pad_remaining_dims should add trailing size-1 dims."""

    def test_1d_to_3d(self):
        """Pad 1D to match 3D target."""
        x = mx.array([1.0, 2.0])
        target = mx.zeros((2, 4, 4))
        result = pad_remaining_dims(x, target)
        assert result.shape == (2, 1, 1)

    def test_2d_to_4d(self):
        """Pad 2D to match 4D target."""
        x = mx.ones((3, 4))
        target = mx.zeros((3, 4, 5, 6))
        result = pad_remaining_dims(x, target)
        assert result.shape == (3, 4, 1, 1)

    def test_same_dims(self):
        """No padding needed when dims match."""
        x = mx.ones((2, 3))
        target = mx.zeros((2, 3))
        result = pad_remaining_dims(x, target)
        assert result.shape == (2, 3)


class TestOptimizerTo:
    """optimizer_to should be a no-op in MLX."""

    def test_returns_same_optimizer(self):
        opt = mlx.optimizers.Adam(learning_rate=1e-3)
        result = optimizer_to(opt, "cuda")
        assert result is opt


# ---------------------------------------------------------------------------
# TrainingValidator Tests
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal dataset for testing the validator."""

    def __init__(self, size: int = 100, input_dim: int = 2, horizon: int = 8):
        self.size = size
        self.input_dim = input_dim
        self.horizon = horizon

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        return {
            "obs": {
                "agent_pos": rng.randn(self.horizon, self.input_dim).astype(np.float32),
            },
            "action": rng.randn(self.horizon, self.input_dim).astype(np.float32),
        }


class _FakePolicy(nn.Module):
    """Minimal policy for testing that has compute_loss."""

    def __init__(self, input_dim: int = 2, horizon: int = 8):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def compute_loss(self, batch):
        action = batch["action"]  # (B, T, D)
        obs = batch["obs"]["agent_pos"]  # (B, T, D)
        pred = self.linear(obs)
        return mx.mean((pred - action) ** 2)


class TestTrainingValidator:
    """TrainingValidator should run validation without error."""

    def test_should_validate(self):
        """should_validate returns True at correct epochs."""
        policy = _FakePolicy()
        mx.eval(policy.parameters())
        dataset = _FakeDataset(size=10)
        validator = TrainingValidator(
            policy=policy,
            val_dataset=dataset,
            eval_every_n_epochs=5,
        )
        # epoch is 0-indexed, triggers at end of epoch 4, 9, 14...
        assert not validator.should_validate(0)
        assert not validator.should_validate(3)
        assert validator.should_validate(4)
        assert validator.should_validate(9)

    def test_validate_returns_metrics(self):
        """validate() should return a dict with val_loss."""
        policy = _FakePolicy()
        mx.eval(policy.parameters())
        dataset = _FakeDataset(size=20)
        validator = TrainingValidator(
            policy=policy,
            val_dataset=dataset,
            eval_every_n_epochs=1,
            n_val_batches=2,
            batch_size=4,
        )
        metrics = validator.validate()
        assert "val_loss" in metrics
        assert "val_steps" in metrics
        assert "val_best" in metrics
        assert np.isfinite(metrics["val_loss"])
        assert metrics["val_steps"] > 0
        assert metrics["val_best"] == metrics["val_loss"]  # first call

    def test_validate_empty_dataset(self):
        """validate() with empty dataset should return nan."""
        policy = _FakePolicy()
        mx.eval(policy.parameters())
        dataset = _FakeDataset(size=0)
        validator = TrainingValidator(
            policy=policy,
            val_dataset=dataset,
        )
        metrics = validator.validate()
        assert np.isnan(metrics["val_loss"])
        assert metrics["val_steps"] == 0

    def test_best_val_loss_tracks(self):
        """best_val_loss should track the minimum."""
        policy = _FakePolicy()
        mx.eval(policy.parameters())
        dataset = _FakeDataset(size=20)
        validator = TrainingValidator(
            policy=policy,
            val_dataset=dataset,
            n_val_batches=2,
            batch_size=4,
        )
        m1 = validator.validate()
        # best should be set
        assert validator.best_val_loss == m1["val_loss"]

    def test_check_early_stopping(self):
        """check_early_stopping should return True when no improvement."""
        policy = _FakePolicy()
        mx.eval(policy.parameters())
        dataset = _FakeDataset(size=20)
        validator = TrainingValidator(
            policy=policy,
            val_dataset=dataset,
            n_val_batches=2,
            batch_size=4,
        )
        # Before any validation, should not stop
        assert not validator.check_early_stopping(1.0)

        # After validation, check
        validator.validate()
        best = validator.best_val_loss
        # Much worse loss should trigger
        assert validator.check_early_stopping(best + 1.0)
        # Equal or better should not trigger
        assert not validator.check_early_stopping(best)


# ---------------------------------------------------------------------------
# WandbLogger Tests
# ---------------------------------------------------------------------------


class TestWandbLogger:
    """WandbLogger disabled mode should be silent no-ops."""

    def test_disabled_mode(self):
        """With enabled=False, all methods should be no-ops."""
        logger = WandbLogger(enabled=False)
        assert not logger.is_active
        # These should not raise
        logger.log({"loss": 0.5}, step=1)
        logger.log_config({"lr": 1e-4})
        logger.finish()

    def test_context_manager(self):
        """Context manager should work in disabled mode."""
        with WandbLogger(enabled=False) as logger:
            logger.log({"loss": 0.5}, step=1)
            assert not logger.is_active

    def test_missing_wandb_import(self):
        """If wandb is not installed, should degrade gracefully.

        This test works regardless of whether wandb is actually installed,
        because we test the disabled path which doesn't try to import.
        """
        logger = WandbLogger(enabled=False)
        logger.log({"loss": 0.5}, step=1)
        logger.finish()
        # No exception = pass


# ---------------------------------------------------------------------------
# Gradient Clipping Tests
# ---------------------------------------------------------------------------


class TestClipGradNorm:
    """clip_grad_norm should limit gradient magnitudes."""

    def test_clips_large_grads(self):
        """Large gradients should be scaled down."""
        # Create gradients with known large norm
        grads = {"layer": {"weight": mx.ones((10, 10)) * 100.0}}
        # Norm would be sqrt(10000 * 10000) = 10000
        clipped = clip_grad_norm(grads, max_norm=1.0)

        flat = mlx.utils.tree_flatten(clipped)
        total_norm_sq = sum(float(mx.sum(g * g)) for _, g in flat)
        total_norm = total_norm_sq**0.5

        # Should be close to max_norm
        assert total_norm < 1.1, f"Clipped norm {total_norm} > 1.1"

    def test_small_grads_unchanged(self):
        """Small gradients should not be modified."""
        grads = {"layer": {"weight": mx.ones((2, 2)) * 0.01}}
        clipped = clip_grad_norm(grads, max_norm=100.0)

        orig_flat = mlx.utils.tree_flatten(grads)
        clip_flat = mlx.utils.tree_flatten(clipped)

        for (_, g_orig), (_, g_clip) in zip(orig_flat, clip_flat):
            mx.eval(g_orig, g_clip)
            np.testing.assert_allclose(
                np.array(g_clip), np.array(g_orig), atol=1e-6
            )

    def test_zero_max_norm_is_noop(self):
        """max_norm=0 should be a no-op (disabled)."""
        grads = {"layer": {"weight": mx.ones((3, 3)) * 999.0}}
        clipped = clip_grad_norm(grads, max_norm=0.0)

        flat = mlx.utils.tree_flatten(clipped)
        for _, g in flat:
            mx.eval(g)
            np.testing.assert_allclose(np.array(g), 999.0)

    def test_negative_max_norm_is_noop(self):
        """Negative max_norm should be a no-op."""
        grads = {"layer": {"weight": mx.ones((3, 3)) * 50.0}}
        clipped = clip_grad_norm(grads, max_norm=-1.0)

        flat = mlx.utils.tree_flatten(clipped)
        for _, g in flat:
            mx.eval(g)
            np.testing.assert_allclose(np.array(g), 50.0)

    def test_nested_grads(self):
        """Should work with deeply nested gradient dicts."""
        grads = {
            "encoder": {
                "conv": {"weight": mx.ones((4, 4)) * 100.0},
                "bn": {"weight": mx.ones((4,)) * 100.0},
            },
            "decoder": {"linear": {"weight": mx.ones((2, 4)) * 100.0}},
        }
        clipped = clip_grad_norm(grads, max_norm=1.0)

        flat = mlx.utils.tree_flatten(clipped)
        total_norm_sq = sum(float(mx.sum(g * g)) for _, g in flat)
        total_norm = total_norm_sq**0.5

        assert total_norm < 1.1

    def test_gradient_clipping_in_training_step(self):
        """Gradient clipping should work in a real training step."""
        model = nn.Linear(4, 2)
        mx.eval(model.parameters())
        optimizer = mlx.optimizers.Adam(learning_rate=1e-2)

        x = mx.random.normal((2, 4))
        target = mx.random.normal((2, 2))

        def loss_fn(model, x, target):
            return mx.mean((model(x) - target) ** 2)

        grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = grad_fn(model, x, target)

        # Clip and update
        clipped_grads = clip_grad_norm(grads, max_norm=0.5)
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)

        # Should not error; loss should be finite
        assert np.isfinite(float(loss))


# ---------------------------------------------------------------------------
# Import / Re-export Tests
# ---------------------------------------------------------------------------


class TestCommonImports:
    """Verify public API re-exports work."""

    def test_common_package_imports(self):
        """All public names should be importable from common."""
        from diffusion_policy_mlx.common import (  # noqa: F401
            JsonLogger,
            dict_apply,
            dict_apply_reduce,
            dict_apply_split,
            optimizer_to,
            param_count,
            replace_submodules,
        )

    def test_training_package_imports(self):
        """Updated training package exports."""
        from diffusion_policy_mlx.training import (  # noqa: F401
            TrainConfig,
            TrainingValidator,
            WandbLogger,
            train,
        )
