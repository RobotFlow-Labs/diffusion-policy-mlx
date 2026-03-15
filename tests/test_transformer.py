"""Tests for TransformerForDiffusion and transformer-based policies."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import pytest

from diffusion_policy_mlx.compat.schedulers import DDPMScheduler
from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy_mlx.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)

# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = dict(
    input_dim=16,
    output_dim=16,
    horizon=8,
    n_obs_steps=4,
    n_layer=2,
    n_head=2,
    n_emb=32,
    p_drop_emb=0.0,
    p_drop_attn=0.0,
)


def _make_transformer(**overrides):
    cfg = {**_SMALL_CFG, **overrides}
    return TransformerForDiffusion(**cfg)


# ---------------------------------------------------------------------------
# TransformerForDiffusion — output shape
# ---------------------------------------------------------------------------


class TestTransformerForDiffusionShape:
    """Verify output shapes for various configurations."""

    def test_encoder_only_time_token(self):
        """BERT-style: time_as_cond=False, no obs cond."""
        model = _make_transformer(time_as_cond=False, cond_dim=0)
        sample = mx.random.normal((4, 8, 16))
        timestep = mx.array(0)
        out = model(sample, timestep)
        mx.eval(out)
        assert out.shape == (4, 8, 16), f"Expected (4,8,16), got {out.shape}"

    def test_decoder_time_cond_only(self):
        """Encoder-decoder with time_as_cond=True, no obs cond."""
        model = _make_transformer(time_as_cond=True, cond_dim=0)
        sample = mx.random.normal((4, 8, 16))
        timestep = mx.array(0)
        out = model(sample, timestep)
        mx.eval(out)
        assert out.shape == (4, 8, 16)

    def test_decoder_obs_cond(self):
        """Encoder-decoder with obs conditioning (no cond encoder layers)."""
        model = _make_transformer(cond_dim=10, time_as_cond=True, n_cond_layers=0)
        sample = mx.random.normal((4, 8, 16))
        timestep = mx.array(0)
        cond = mx.random.normal((4, 4, 10))
        out = model(sample, timestep, cond)
        mx.eval(out)
        assert out.shape == (4, 8, 16)

    def test_decoder_obs_cond_with_encoder(self):
        """Encoder-decoder with obs conditioning and cond encoder layers."""
        model = _make_transformer(cond_dim=10, time_as_cond=True, n_cond_layers=2)
        sample = mx.random.normal((4, 8, 16))
        timestep = mx.array(0)
        cond = mx.random.normal((4, 4, 10))
        out = model(sample, timestep, cond)
        mx.eval(out)
        assert out.shape == (4, 8, 16)

    def test_different_input_output_dim(self):
        """input_dim != output_dim."""
        model = _make_transformer(input_dim=10, output_dim=20)
        sample = mx.random.normal((2, 8, 10))
        timestep = mx.array(5)
        out = model(sample, timestep)
        mx.eval(out)
        assert out.shape == (2, 8, 20)

    def test_batch_timesteps(self):
        """Timestep as a batch vector."""
        model = _make_transformer(cond_dim=0, time_as_cond=True)
        sample = mx.random.normal((4, 8, 16))
        timesteps = mx.array([0, 5, 10, 15])
        out = model(sample, timesteps)
        mx.eval(out)
        assert out.shape == (4, 8, 16)

    def test_scalar_int_timestep(self):
        """Timestep as a plain Python int."""
        model = _make_transformer(cond_dim=0, time_as_cond=True)
        sample = mx.random.normal((2, 8, 16))
        out = model(sample, 3)
        mx.eval(out)
        assert out.shape == (2, 8, 16)


# ---------------------------------------------------------------------------
# Causal vs non-causal attention
# ---------------------------------------------------------------------------


class TestCausalAttention:
    def test_causal_mask_shape(self):
        """Causal model creates a mask of the right size."""
        model = _make_transformer(causal_attn=True, cond_dim=0, time_as_cond=True)
        assert model._mask is not None
        # T = horizon = 8
        assert model._mask.shape == (8, 8)

    def test_noncausal_no_mask(self):
        """Non-causal model has no mask."""
        model = _make_transformer(causal_attn=False)
        assert model._mask is None
        assert model._memory_mask is None

    def test_causal_with_obs_cond_memory_mask(self):
        """Causal + obs_as_cond creates a memory_mask."""
        model = _make_transformer(
            causal_attn=True, cond_dim=10, time_as_cond=True
        )
        assert model._mask is not None
        assert model._memory_mask is not None
        # T = 8, T_cond = 1 (time) + 4 (obs) = 5
        assert model._memory_mask.shape == (8, 5)

    def test_causal_output_runs(self):
        """Causal attention model produces valid output."""
        model = _make_transformer(
            causal_attn=True, cond_dim=10, time_as_cond=True, n_cond_layers=2
        )
        sample = mx.random.normal((2, 8, 16))
        cond = mx.random.normal((2, 4, 10))
        out = model(sample, mx.array(0), cond)
        mx.eval(out)
        assert out.shape == (2, 8, 16)
        assert mx.isfinite(out).all().item()

    def test_encoder_only_causal(self):
        """Encoder-only BERT with causal attention."""
        model = _make_transformer(
            causal_attn=True, cond_dim=0, time_as_cond=False
        )
        # T = horizon + 1 = 9 (time token prepended)
        assert model._mask is not None
        assert model._mask.shape == (9, 9)
        sample = mx.random.normal((2, 8, 16))
        out = model(sample, mx.array(0))
        mx.eval(out)
        assert out.shape == (2, 8, 16)


# ---------------------------------------------------------------------------
# time_as_cond and obs_as_cond modes
# ---------------------------------------------------------------------------


class TestCondModes:
    def test_time_as_cond_true_encoder_decoder(self):
        """time_as_cond=True -> encoder-decoder architecture."""
        model = _make_transformer(time_as_cond=True, cond_dim=0)
        assert not model.encoder_only
        assert model.decoder is not None

    def test_time_as_cond_false_encoder_only(self):
        """time_as_cond=False -> encoder-only BERT."""
        model = _make_transformer(time_as_cond=False, cond_dim=0)
        assert model.encoder_only
        assert model.decoder is None

    def test_obs_as_cond_requires_time_as_cond(self):
        """obs_as_cond with cond_dim > 0 requires time_as_cond=True."""
        with pytest.raises(AssertionError):
            _make_transformer(cond_dim=10, time_as_cond=False)

    def test_obs_cond_has_cond_obs_emb(self):
        """When cond_dim > 0, cond_obs_emb is created."""
        model = _make_transformer(cond_dim=10, time_as_cond=True)
        assert model.cond_obs_emb is not None
        assert model.obs_as_cond is True

    def test_no_obs_cond_no_cond_obs_emb(self):
        """When cond_dim = 0, cond_obs_emb is None."""
        model = _make_transformer(cond_dim=0)
        assert model.cond_obs_emb is None


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradient_encoder_decoder(self):
        """Gradients flow through encoder-decoder transformer."""
        model = _make_transformer(cond_dim=10, time_as_cond=True, n_cond_layers=2)

        def loss_fn(model, sample, timestep, cond):
            out = model(sample, timestep, cond)
            return mx.mean(out ** 2)

        sample = mx.random.normal((2, 8, 16))
        cond = mx.random.normal((2, 4, 10))
        timestep = mx.array([0, 5])

        loss, grads = nn.value_and_grad(model, loss_fn)(model, sample, timestep, cond)
        mx.eval(loss, grads)

        assert loss.item() > 0
        # Check that at least some gradients are non-zero
        flat_grads = [v for _, v in mlx.utils.tree_flatten(grads)]
        has_nonzero = any(
            mx.any(mx.abs(g) > 0).item() for g in flat_grads if isinstance(g, mx.array)
        )
        assert has_nonzero, "All gradients are zero"

    def test_gradient_encoder_only(self):
        """Gradients flow through encoder-only (BERT) transformer."""
        model = _make_transformer(time_as_cond=False, cond_dim=0)

        def loss_fn(model, sample, timestep):
            out = model(sample, timestep)
            return mx.mean(out ** 2)

        sample = mx.random.normal((2, 8, 16))
        timestep = mx.array([0, 5])

        loss, grads = nn.value_and_grad(model, loss_fn)(model, sample, timestep)
        mx.eval(loss, grads)

        assert loss.item() > 0
        flat_grads = [v for _, v in mlx.utils.tree_flatten(grads)]
        has_nonzero = any(
            mx.any(mx.abs(g) > 0).item() for g in flat_grads if isinstance(g, mx.array)
        )
        assert has_nonzero, "All gradients are zero"


# ---------------------------------------------------------------------------
# Transformer Lowdim Policy
# ---------------------------------------------------------------------------


def _make_lowdim_policy(obs_as_cond=True, pred_action_steps_only=False):
    """Create a small lowdim policy for testing."""
    obs_dim = 10
    action_dim = 4
    horizon = 8
    n_obs_steps = 2
    n_action_steps = 4

    if obs_as_cond:
        input_dim = action_dim
        output_dim = action_dim
        cond_dim = obs_dim
    else:
        input_dim = action_dim + obs_dim
        output_dim = input_dim
        cond_dim = 0

    # When pred_action_steps_only, the model only sees n_action_steps tokens,
    # so causal mask must match. Use causal_attn=False for pred_action_steps_only
    # (consistent with upstream usage).
    causal_attn = not pred_action_steps_only

    model = TransformerForDiffusion(
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        cond_dim=cond_dim,
        n_layer=2,
        n_head=2,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=causal_attn,
        time_as_cond=True,
        obs_as_cond=obs_as_cond,
        n_cond_layers=0,
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    policy = DiffusionTransformerLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=3,
        obs_as_cond=obs_as_cond,
        pred_action_steps_only=pred_action_steps_only,
    )
    return policy


def _make_lowdim_normalizer(obs_dim=10, action_dim=4):
    """Build an identity normalizer for lowdim policy."""
    normalizer = LinearNormalizer()
    normalizer["obs"] = SingleFieldLinearNormalizer.create_identity(obs_dim)
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity(action_dim)
    return normalizer


class TestTransformerLowdimPolicy:
    def test_predict_action_shape_obs_cond(self):
        """predict_action returns correct shapes with obs_as_cond."""
        policy = _make_lowdim_policy(obs_as_cond=True)
        policy.set_normalizer(_make_lowdim_normalizer())

        obs = mx.random.normal((2, 3, 10))
        result = policy.predict_action({"obs": obs})
        mx.eval(result["action"])

        assert result["action"].shape == (2, 4, 4), f"Got {result['action'].shape}"
        assert result["action_pred"].shape == (2, 8, 4)

    def test_predict_action_shape_inpaint(self):
        """predict_action returns correct shapes without obs_as_cond."""
        policy = _make_lowdim_policy(obs_as_cond=False)
        policy.set_normalizer(_make_lowdim_normalizer())

        obs = mx.random.normal((2, 3, 10))
        result = policy.predict_action({"obs": obs})
        mx.eval(result["action"])

        assert result["action"].shape == (2, 4, 4)
        assert "obs_pred" in result

    def test_compute_loss_scalar(self):
        """compute_loss returns a scalar > 0."""
        policy = _make_lowdim_policy(obs_as_cond=True)
        policy.set_normalizer(_make_lowdim_normalizer())

        batch = {
            "obs": mx.random.normal((4, 8, 10)),
            "action": mx.random.normal((4, 8, 4)),
        }
        loss = policy.compute_loss(batch)
        mx.eval(loss)

        assert loss.ndim == 0, f"Expected scalar, got ndim={loss.ndim}"
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_compute_loss_gradient(self):
        """Loss is differentiable through the policy."""
        policy = _make_lowdim_policy(obs_as_cond=True)
        policy.set_normalizer(_make_lowdim_normalizer())

        batch = {
            "obs": mx.random.normal((2, 8, 10)),
            "action": mx.random.normal((2, 8, 4)),
        }

        def loss_fn(policy):
            return policy.compute_loss(batch)

        loss, grads = nn.value_and_grad(policy, loss_fn)(policy)
        mx.eval(loss, grads)

        assert loss.item() > 0
        flat_grads = [v for _, v in mlx.utils.tree_flatten(grads)]
        has_nonzero = any(
            mx.any(mx.abs(g) > 0).item()
            for g in flat_grads
            if isinstance(g, mx.array)
        )
        assert has_nonzero, "All gradients are zero"

    def test_pred_action_steps_only(self):
        """pred_action_steps_only mode returns correct shapes."""
        policy = _make_lowdim_policy(obs_as_cond=True, pred_action_steps_only=True)
        policy.set_normalizer(_make_lowdim_normalizer())

        obs = mx.random.normal((2, 3, 10))
        result = policy.predict_action({"obs": obs})
        mx.eval(result["action"])

        # When pred_action_steps_only, action_pred has n_action_steps length
        assert result["action_pred"].shape == (2, 4, 4)
        assert result["action"].shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# Transformer Hybrid Image Policy
# ---------------------------------------------------------------------------


class TestTransformerHybridImagePolicy:
    """Test the transformer hybrid image policy.

    Uses the same approach as test_policy.py for the UNet variant.
    """

    def _make_shape_meta(self):
        return {
            "obs": {
                "image": {"shape": (3, 96, 96), "type": "rgb"},
                "agent_pos": {"shape": (2,), "type": "low_dim"},
            },
            "action": {"shape": (2,)},
        }

    def _make_policy(self):
        from diffusion_policy_mlx.policy.diffusion_transformer_hybrid_image_policy import (
            DiffusionTransformerHybridImagePolicy,
        )

        shape_meta = self._make_shape_meta()
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        policy = DiffusionTransformerHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=2,
            num_inference_steps=3,
            crop_shape=(76, 76),
            n_layer=2,
            n_cond_layers=0,
            n_head=2,
            n_emb=32,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
        )
        return policy

    def _make_identity_normalizer(self):
        normalizer = LinearNormalizer()
        # For obs (nested dict)
        obs_normalizer = LinearNormalizer()
        obs_normalizer["image"] = SingleFieldLinearNormalizer.create_identity(3 * 96 * 96)
        obs_normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity(2)
        normalizer["obs"] = obs_normalizer
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity(2)
        return normalizer

    def test_predict_action_shape(self):
        """predict_action returns correct shapes."""
        policy = self._make_policy()
        policy.set_normalizer(self._make_identity_normalizer())

        obs_dict = {
            "image": mx.random.normal((1, 2, 3, 96, 96)),
            "agent_pos": mx.random.normal((1, 2, 2)),
        }
        result = policy.predict_action(obs_dict)
        mx.eval(result["action"])

        assert result["action"].shape == (1, 8, 2), f"Got {result['action'].shape}"
        assert result["action_pred"].shape == (1, 16, 2)

    def test_compute_loss_scalar(self):
        """compute_loss returns a scalar > 0."""
        policy = self._make_policy()
        policy.set_normalizer(self._make_identity_normalizer())

        batch = {
            "obs": {
                "image": mx.random.normal((2, 16, 3, 96, 96)),
                "agent_pos": mx.random.normal((2, 16, 2)),
            },
            "action": mx.random.normal((2, 16, 2)),
        }
        loss = policy.compute_loss(batch)
        mx.eval(loss)

        assert loss.ndim == 0, f"Expected scalar, got ndim={loss.ndim}"
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_compute_loss_gradient(self):
        """Loss is differentiable (gradients flow)."""
        policy = self._make_policy()
        policy.set_normalizer(self._make_identity_normalizer())

        batch = {
            "obs": {
                "image": mx.random.normal((1, 16, 3, 96, 96)),
                "agent_pos": mx.random.normal((1, 16, 2)),
            },
            "action": mx.random.normal((1, 16, 2)),
        }

        def loss_fn(policy):
            return policy.compute_loss(batch)

        loss, grads = nn.value_and_grad(policy, loss_fn)(policy)
        mx.eval(loss, grads)

        assert loss.item() > 0
        flat_grads = [v for _, v in mlx.utils.tree_flatten(grads)]
        has_nonzero = any(
            mx.any(mx.abs(g) > 0).item()
            for g in flat_grads
            if isinstance(g, mx.array)
        )
        assert has_nonzero, "All gradients are zero"
