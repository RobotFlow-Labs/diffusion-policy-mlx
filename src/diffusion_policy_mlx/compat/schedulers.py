"""DDPM and DDIM noise schedulers implemented in pure MLX.

Drop-in replacements for diffusers.DDPMScheduler and diffusers.DDIMScheduler,
implementing only the API surface used by the upstream diffusion_policy codebase.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Scheduler output container
# ---------------------------------------------------------------------------

class SchedulerOutput:
    """Simple container matching diffusers ``SchedulerOutput``."""

    def __init__(self, prev_sample: mx.array):
        self.prev_sample = prev_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(num_train_timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine schedule from *Improved DDPM* (Nichol & Dhariwal, 2021).

    ``squaredcos_cap_v2`` in diffusers.
    """
    steps = np.arange(num_train_timesteps + 1, dtype=np.float64)
    f_t = np.cos(((steps / num_train_timesteps) + s) / (1.0 + s) * (math.pi * 0.5)) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0, 0.999).astype(np.float32)


def _linear_beta_schedule(
    num_train_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> np.ndarray:
    return np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)


def _get_betas(
    beta_schedule: str,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> np.ndarray:
    if beta_schedule == "linear":
        return _linear_beta_schedule(num_train_timesteps, beta_start, beta_end)
    elif beta_schedule in ("squaredcos_cap_v2", "cosine"):
        return _cosine_beta_schedule(num_train_timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")


def _add_noise(
    alphas_cumprod: mx.array,
    original_samples: mx.array,
    noise: mx.array,
    timesteps: mx.array,
) -> mx.array:
    """Forward diffusion: q(x_t | x_0). Shared by DDPM and DDIM."""
    alpha_bar = alphas_cumprod[timesteps]
    while alpha_bar.ndim < original_samples.ndim:
        alpha_bar = mx.expand_dims(alpha_bar, axis=-1)
    sqrt_ab = mx.sqrt(alpha_bar)
    sqrt_one_minus_ab = mx.sqrt(1.0 - alpha_bar)
    return sqrt_ab * original_samples + sqrt_one_minus_ab * noise


def _predict_x0(
    prediction_type: str,
    model_output: mx.array,
    sample: mx.array,
    alpha_bar_t: mx.array,
) -> mx.array:
    """Recover predicted x_0 from the model output."""
    if prediction_type == "epsilon":
        return (sample - mx.sqrt(1.0 - alpha_bar_t) * model_output) / mx.sqrt(alpha_bar_t)
    elif prediction_type == "sample":
        return model_output
    elif prediction_type == "v_prediction":
        return mx.sqrt(alpha_bar_t) * sample - mx.sqrt(1.0 - alpha_bar_t) * model_output
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")


def _predict_epsilon(
    prediction_type: str,
    model_output: mx.array,
    sample: mx.array,
    alpha_bar_t: mx.array,
    pred_x0: mx.array,
) -> mx.array:
    """Recover predicted noise epsilon from model output (needed for DDIM)."""
    if prediction_type == "epsilon":
        return model_output
    elif prediction_type == "sample":
        return (sample - mx.sqrt(alpha_bar_t) * pred_x0) / mx.sqrt(1.0 - alpha_bar_t)
    elif prediction_type == "v_prediction":
        return mx.sqrt(alpha_bar_t) * model_output + mx.sqrt(1.0 - alpha_bar_t) * sample
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")


# ---------------------------------------------------------------------------
# DDPMScheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler.

    Implements:
      - Forward process: q(x_t | x_0)
      - Reverse process: p(x_{t-1} | x_t) with learned noise prediction

    Constructor matches the diffusers API subset used by upstream diffusion_policy.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.variance_type = variance_type

        # Expose a config-like attribute so that upstream code accessing
        # ``scheduler.config.num_train_timesteps`` keeps working.
        self.config = _SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )

        # Compute schedule --------------------------------------------------
        betas_np = _get_betas(beta_schedule, num_train_timesteps, beta_start, beta_end)
        self.betas = mx.array(betas_np, dtype=mx.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        # Default timesteps (reversed, for full schedule)
        self.timesteps = mx.arange(num_train_timesteps - 1, -1, -1)

    # -- forward diffusion --------------------------------------------------

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """Forward diffusion: q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        return _add_noise(self.alphas_cumprod, original_samples, noise, timesteps)

    # -- inference -----------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Configure reverse schedule for inference."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps_np = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int32)
        self.timesteps = mx.array(timesteps_np[::-1].copy(), dtype=mx.int32)
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
        generator=None,
    ) -> SchedulerOutput:
        """Reverse diffusion step: p(x_{t-1} | x_t)."""
        t = int(timestep)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]

        # Predict x_0
        pred_x0 = _predict_x0(self.prediction_type, model_output, sample, alpha_bar_t)

        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # Posterior mean  q(x_{t-1} | x_t, x_0)
        coeff1 = mx.sqrt(alpha_bar_t_prev) * beta_t / (1.0 - alpha_bar_t)
        coeff2 = mx.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
        pred_mean = coeff1 * pred_x0 + coeff2 * sample

        # Posterior variance
        if t > 0:
            if self.variance_type == "fixed_small":
                variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                # Floor to prevent NaN from sqrt of denormalized/negative values
                variance = mx.maximum(variance, mx.array(1e-20))
            else:  # fixed_large
                variance = beta_t
            noise = mx.random.normal(sample.shape)
            prev_sample = pred_mean + mx.sqrt(variance) * noise
        else:
            prev_sample = pred_mean

        return SchedulerOutput(prev_sample=prev_sample)


# ---------------------------------------------------------------------------
# DDIMScheduler
# ---------------------------------------------------------------------------

class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler.

    Accelerated sampling: can use 10-50 steps instead of 1000.
    Deterministic when eta=0.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        set_alpha_to_one: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        self.config = _SchedulerConfig(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
        )

        betas_np = _get_betas(beta_schedule, num_train_timesteps, beta_start, beta_end)
        self.betas = mx.array(betas_np, dtype=mx.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        self.final_alpha_cumprod = (
            mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # Default full schedule
        self.timesteps = mx.arange(num_train_timesteps - 1, -1, -1)
        self._step_ratio: int = 1

    # -- forward diffusion (same as DDPM) ------------------------------------

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        return _add_noise(self.alphas_cumprod, original_samples, noise, timesteps)

    # -- inference -----------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        step_ratio = self.num_train_timesteps // num_inference_steps
        self._step_ratio = step_ratio
        timesteps_np = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int32)
        self.timesteps = mx.array(timesteps_np[::-1].copy(), dtype=mx.int32)
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: mx.array,
        timestep,
        sample: mx.array,
        eta: float = 0.0,
        generator=None,
    ) -> SchedulerOutput:
        """DDIM reverse step.  Deterministic when ``eta=0``."""
        t = int(timestep)
        prev_t = self._get_previous_timestep(t)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )

        # Predict x_0 and epsilon
        pred_x0 = _predict_x0(self.prediction_type, model_output, sample, alpha_bar_t)
        pred_eps = _predict_epsilon(
            self.prediction_type, model_output, sample, alpha_bar_t, pred_x0
        )

        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # DDIM sigma — floor to prevent sqrt of negative from float rounding
        sigma_sq = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_prev)
        )
        sigma = eta * mx.sqrt(mx.maximum(sigma_sq, mx.array(0.0)))

        pred_direction = mx.sqrt(1.0 - alpha_bar_prev - sigma**2) * pred_eps
        prev_sample = mx.sqrt(alpha_bar_prev) * pred_x0 + pred_direction

        if eta > 0 and prev_t > 0:
            noise = mx.random.normal(sample.shape)
            prev_sample = prev_sample + sigma * noise

        return SchedulerOutput(prev_sample=prev_sample)

    def _get_previous_timestep(self, t: int) -> int:
        """Return the previous timestep in the current inference schedule."""
        prev_t = t - self._step_ratio
        return prev_t


# ---------------------------------------------------------------------------
# Minimal config object
# ---------------------------------------------------------------------------

class _SchedulerConfig:
    """Tiny namespace so ``scheduler.config.num_train_timesteps`` works."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
