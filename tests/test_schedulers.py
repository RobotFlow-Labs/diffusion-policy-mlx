"""Tests for DDPM and DDIM schedulers (MLX implementation)."""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.compat.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    SchedulerOutput,
)

# ---------------------------------------------------------------------------
# Optional dependency flag
# ---------------------------------------------------------------------------

try:
    import diffusers  # noqa: F401

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===================================================================
# DDPM tests
# ===================================================================


class TestDDPMBetaSchedule:
    def test_linear_shape_and_order(self):
        s = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
        assert s.betas.shape == (100,)
        betas = np.array(s.betas)
        assert betas[0] > 0
        assert betas[-1] > betas[0]  # increasing

    def test_cosine_shape_and_clip(self):
        s = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
        assert s.betas.shape == (100,)
        assert float(mx.max(s.betas)) <= 0.999 + 1e-6  # allow float32 rounding

    def test_cosine_alias(self):
        s1 = DDPMScheduler(num_train_timesteps=50, beta_schedule="squaredcos_cap_v2")
        s2 = DDPMScheduler(num_train_timesteps=50, beta_schedule="cosine")
        np.testing.assert_allclose(np.array(s1.betas), np.array(s2.betas))

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown beta schedule"):
            DDPMScheduler(beta_schedule="magic")


class TestDDPMAlphasCumprod:
    def test_monotonic_decreasing(self):
        s = DDPMScheduler(num_train_timesteps=100)
        ac = np.array(s.alphas_cumprod)
        assert np.all(ac[1:] <= ac[:-1]), "alphas_cumprod must be monotonically decreasing"

    def test_range(self):
        s = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        ac = np.array(s.alphas_cumprod)
        assert ac[0] > 0.99, f"alpha_bar[0] should be ~1, got {ac[0]}"
        assert ac[-1] < 0.1, f"alpha_bar[-1] should be ~0, got {ac[-1]}"


class TestDDPMAddNoise:
    def test_shape_preserved(self):
        s = DDPMScheduler(num_train_timesteps=100)
        x0 = mx.random.normal((4, 16, 2))
        noise = mx.random.normal((4, 16, 2))
        t = mx.array([10, 20, 30, 40])
        xt = s.add_noise(x0, noise, t)
        assert xt.shape == (4, 16, 2)

    def test_t0_nearly_clean(self):
        s = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        x0 = mx.ones((1, 4))
        noise = mx.zeros((1, 4))
        xt = s.add_noise(x0, noise, mx.array([0]))
        np.testing.assert_allclose(np.array(xt), np.array(x0), atol=0.01)

    def test_tmax_nearly_noise(self):
        s = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        x0 = mx.ones((1, 4))
        noise = mx.random.normal((1, 4))
        xt = s.add_noise(x0, noise, mx.array([999]))
        # At t_max, alpha_bar is very small, so xt ≈ noise
        ab = float(s.alphas_cumprod[999])
        expected = np.sqrt(ab) * np.array(x0) + np.sqrt(1 - ab) * np.array(noise)
        np.testing.assert_allclose(np.array(xt), expected, atol=1e-5)

    def test_4d_input(self):
        """Ensure broadcasting works for image-like tensors."""
        s = DDPMScheduler(num_train_timesteps=100)
        x0 = mx.random.normal((2, 3, 8, 8))
        noise = mx.random.normal((2, 3, 8, 8))
        t = mx.array([10, 50])
        xt = s.add_noise(x0, noise, t)
        assert xt.shape == (2, 3, 8, 8)


class TestDDPMStep:
    def test_produces_valid_samples(self):
        s = DDPMScheduler(num_train_timesteps=100)
        s.set_timesteps(100)
        sample = mx.random.normal((1, 4))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample)
            sample = out.prev_sample
        arr = np.array(sample)
        assert not np.any(np.isnan(arr)), "NaN in output"
        assert not np.any(np.isinf(arr)), "Inf in output"

    def test_full_reverse_process(self):
        """After 100 reverse steps the sample should not explode."""
        s = DDPMScheduler(num_train_timesteps=100)
        s.set_timesteps(100)
        sample = mx.random.normal((2, 8))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample)
            sample = out.prev_sample
        assert mx.all(mx.isfinite(sample)).item()
        assert float(mx.max(mx.abs(sample))) < 100.0

    def test_returns_scheduler_output(self):
        s = DDPMScheduler(num_train_timesteps=10)
        s.set_timesteps(10)
        sample = mx.random.normal((1, 2))
        t = s.timesteps[0]
        out = s.step(mx.zeros_like(sample), t, sample)
        assert isinstance(out, SchedulerOutput)
        assert out.prev_sample.shape == sample.shape

    def test_prediction_type_sample(self):
        s = DDPMScheduler(num_train_timesteps=20, prediction_type="sample")
        s.set_timesteps(20)
        sample = mx.random.normal((1, 4))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample)
            sample = out.prev_sample
        assert mx.all(mx.isfinite(sample)).item()

    def test_prediction_type_v(self):
        s = DDPMScheduler(num_train_timesteps=20, prediction_type="v_prediction")
        s.set_timesteps(20)
        sample = mx.random.normal((1, 4))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample)
            sample = out.prev_sample
        assert mx.all(mx.isfinite(sample)).item()


class TestDDPMSetTimesteps:
    def test_length(self):
        s = DDPMScheduler(num_train_timesteps=1000)
        s.set_timesteps(100)
        assert len(s.timesteps) == 100

    def test_reversed_order(self):
        s = DDPMScheduler(num_train_timesteps=1000)
        s.set_timesteps(50)
        ts = np.array(s.timesteps)
        assert np.all(ts[:-1] >= ts[1:]), "timesteps should be in descending order"


# ===================================================================
# DDIM tests
# ===================================================================


class TestDDIMSetTimesteps:
    def test_fewer_steps(self):
        s = DDIMScheduler(num_train_timesteps=1000)
        s.set_timesteps(50)
        assert len(s.timesteps) == 50

    def test_step_spacing(self):
        s = DDIMScheduler(num_train_timesteps=1000)
        s.set_timesteps(10)
        ts = np.array(s.timesteps)
        # Should be evenly spaced with ratio 100
        diffs = np.diff(ts)
        assert np.all(diffs == -100), f"Expected uniform spacing of 100, got diffs={diffs}"


class TestDDIMDeterministic:
    def test_eta0_deterministic(self):
        """DDIM with eta=0 should give identical results on repeated runs."""
        s = DDIMScheduler(num_train_timesteps=100)
        s.set_timesteps(10)

        start = mx.array(np.random.randn(1, 4).astype(np.float32))

        def run_once(init):
            sample = init
            for t in s.timesteps:
                out = s.step(mx.zeros_like(sample), t, sample, eta=0.0)
                sample = out.prev_sample
            mx.eval(sample)
            return np.array(sample)

        r1 = run_once(start)
        r2 = run_once(start)
        np.testing.assert_array_equal(r1, r2)


class TestDDIMStep:
    def test_produces_valid_samples(self):
        s = DDIMScheduler(num_train_timesteps=100)
        s.set_timesteps(20)
        sample = mx.random.normal((1, 4))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample, eta=0.0)
            sample = out.prev_sample
        arr = np.array(sample)
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))

    def test_eta_positive(self):
        """With eta > 0, step should still produce valid (but stochastic) output."""
        s = DDIMScheduler(num_train_timesteps=100)
        s.set_timesteps(20)
        sample = mx.random.normal((1, 4))
        for t in s.timesteps:
            out = s.step(mx.zeros_like(sample), t, sample, eta=1.0)
            sample = out.prev_sample
        arr = np.array(sample)
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))

    def test_returns_scheduler_output(self):
        s = DDIMScheduler(num_train_timesteps=50)
        s.set_timesteps(10)
        sample = mx.random.normal((1, 2))
        t = s.timesteps[0]
        out = s.step(mx.zeros_like(sample), t, sample, eta=0.0)
        assert isinstance(out, SchedulerOutput)
        assert out.prev_sample.shape == sample.shape


class TestDDIMAddNoise:
    def test_shape_preserved(self):
        s = DDIMScheduler(num_train_timesteps=100)
        x0 = mx.random.normal((4, 16, 2))
        noise = mx.random.normal((4, 16, 2))
        t = mx.array([10, 20, 30, 40])
        xt = s.add_noise(x0, noise, t)
        assert xt.shape == (4, 16, 2)


# ===================================================================
# Cross-framework tests (vs HuggingFace diffusers)
# ===================================================================


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
class TestCrossFrameworkDDPM:
    def test_alphas_match_diffusers_cosine(self):
        from diffusers import DDPMScheduler as DiffusersDDPM

        ours = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
        )
        theirs = DiffusersDDPM(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
        )
        np.testing.assert_allclose(
            np.array(ours.alphas_cumprod),
            theirs.alphas_cumprod.numpy(),
            atol=1e-6,
        )

    def test_alphas_match_diffusers_linear(self):
        from diffusers import DDPMScheduler as DiffusersDDPM

        ours = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="linear"
        )
        theirs = DiffusersDDPM(
            num_train_timesteps=100, beta_schedule="linear"
        )
        np.testing.assert_allclose(
            np.array(ours.alphas_cumprod),
            theirs.alphas_cumprod.numpy(),
            atol=1e-6,
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_add_noise_matches_diffusers(self):
        import torch
        from diffusers import DDPMScheduler as DiffusersDDPM

        T = 100
        ours = DDPMScheduler(
            num_train_timesteps=T, beta_schedule="squaredcos_cap_v2"
        )
        theirs = DiffusersDDPM(
            num_train_timesteps=T, beta_schedule="squaredcos_cap_v2"
        )

        np_x0 = np.random.randn(4, 16, 2).astype(np.float32)
        np_noise = np.random.randn(4, 16, 2).astype(np.float32)
        np_t = np.array([0, 25, 50, 99], dtype=np.int64)

        mx_xt = ours.add_noise(
            mx.array(np_x0), mx.array(np_noise), mx.array(np_t)
        )
        th_xt = theirs.add_noise(
            torch.tensor(np_x0),
            torch.tensor(np_noise),
            torch.tensor(np_t),
        )

        np.testing.assert_allclose(
            np.array(mx_xt), th_xt.numpy(), atol=1e-5
        )


    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_ddpm_step_matches_diffusers(self):
        """DDPM step() output should match diffusers for a single reverse step."""
        import torch
        from diffusers import DDPMScheduler as DiffusersDDPM

        T = 100
        ours = DDPMScheduler(
            num_train_timesteps=T,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )
        theirs = DiffusersDDPM(
            num_train_timesteps=T,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

        np.random.seed(42)
        np_sample = np.random.randn(2, 8, 2).astype(np.float32)
        np_model_out = np.random.randn(2, 8, 2).astype(np.float32)
        t = 50

        # Use deterministic step (t=0 has no noise)
        # For t>0, we can't compare due to random noise.
        # Instead compare the predicted x0 and posterior mean by testing at t=0.
        mx_out = ours.step(mx.array(np_model_out), 0, mx.array(np_sample))
        th_out = theirs.step(torch.tensor(np_model_out), 0, torch.tensor(np_sample))

        np.testing.assert_allclose(
            np.array(mx_out.prev_sample),
            th_out.prev_sample.numpy(),
            atol=1e-4,
        )


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
class TestCrossFrameworkDDIM:
    def test_alphas_match_diffusers(self):
        from diffusers import DDIMScheduler as DiffusersDDIM

        ours = DDIMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
        )
        theirs = DiffusersDDIM(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
        )
        np.testing.assert_allclose(
            np.array(ours.alphas_cumprod),
            theirs.alphas_cumprod.numpy(),
            atol=1e-6,
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_ddim_step_matches_diffusers_eta0(self):
        """DDIM step() with eta=0 should match diffusers (deterministic)."""
        import torch
        from diffusers import DDIMScheduler as DiffusersDDIM

        T = 100
        ours = DDIMScheduler(
            num_train_timesteps=T,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )
        theirs = DiffusersDDIM(
            num_train_timesteps=T,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            clip_sample_range=1.0,
            prediction_type="epsilon",
        )

        ours.set_timesteps(10)
        theirs.set_timesteps(10)

        np.random.seed(42)
        np_sample = np.random.randn(2, 8, 2).astype(np.float32)
        np_model_out = np.random.randn(2, 8, 2).astype(np.float32)

        t = int(ours.timesteps[0])

        mx_out = ours.step(mx.array(np_model_out), t, mx.array(np_sample), eta=0.0)
        th_out = theirs.step(torch.tensor(np_model_out), t, torch.tensor(np_sample), eta=0.0)

        np.testing.assert_allclose(
            np.array(mx_out.prev_sample),
            th_out.prev_sample.numpy(),
            atol=1e-4,
        )
