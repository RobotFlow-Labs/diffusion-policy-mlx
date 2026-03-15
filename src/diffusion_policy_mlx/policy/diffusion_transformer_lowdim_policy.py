"""Diffusion Transformer Low-Dim Policy.

Upstream: diffusion_policy/policy/diffusion_transformer_lowdim_policy.py

Uses TransformerForDiffusion as the denoiser for low-dimensional
observation spaces (no vision encoder needed).

Two conditioning modes:
  - ``obs_as_cond=True``: observations passed as conditioning tokens
    to the transformer decoder via cross-attention.
  - ``obs_as_cond=False``: observations concatenated with noisy actions
    along feature dim; inpainting mask conditions on observed timesteps.
"""

from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from diffusion_policy_mlx.model.common.normalizer import LinearNormalizer
from diffusion_policy_mlx.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_mlx.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


class DiffusionTransformerLowdimPolicy(nn.Module):
    """Low-dimensional diffusion policy using a Transformer denoiser.

    Args:
        model: TransformerForDiffusion instance.
        noise_scheduler: A DDPMScheduler or DDIMScheduler instance.
        horizon: Full trajectory length (T).
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        n_action_steps: Number of action steps to execute (Ta).
        n_obs_steps: Number of observation steps (To).
        num_inference_steps: Number of denoising steps at inference.
        obs_as_cond: Whether to use observation conditioning.
        pred_action_steps_only: Only predict action steps (not full horizon).
    """

    def __init__(
        self,
        model: TransformerForDiffusion,
        noise_scheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_cond: bool = False,
        pred_action_steps_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond, "pred_action_steps_only requires obs_as_cond=True"

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # -- Normalizer ------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the data normalizer."""
        self.normalizer = normalizer

    # -- Conditional sampling (inference) --------------------------------------

    def conditional_sample(
        self,
        condition_data: mx.array,
        condition_mask: mx.array,
        cond: Optional[mx.array] = None,
    ) -> mx.array:
        """Run iterative denoising with optional inpainting.

        Args:
            condition_data: ``(B, T, D)`` conditioning values.
            condition_mask: ``(B, T, D)`` boolean mask (True = observed).
            cond: ``(B, To, cond_dim)`` observation conditioning (optional).

        Returns:
            Denoised trajectory ``(B, T, D)``.
        """
        trajectory = mx.random.normal(condition_data.shape)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Apply conditioning (inpainting)
            trajectory = mx.where(condition_mask, condition_data, trajectory)

            # Predict noise / x0
            model_output = self.model(trajectory, t, cond)

            # Scheduler step
            out = self.noise_scheduler.step(model_output, t, trajectory)
            trajectory = out.prev_sample

        # Final conditioning enforcement
        trajectory = mx.where(condition_mask, condition_data, trajectory)

        return trajectory

    # -- Inference -------------------------------------------------------------

    def predict_action(self, obs_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Generate actions via iterative denoising.

        Args:
            obs_dict: Must include ``'obs'`` key with shape ``(B, T, obs_dim)``.

        Returns:
            Dict with ``'action'``, ``'action_pred'``, and optionally
            ``'obs_pred'`` and ``'action_obs_pred'``.
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict, "past_action not implemented"

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        cond = None

        if self.obs_as_cond:
            cond = nobs[:, :To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = mx.zeros(shape)
            cond_mask = mx.zeros(shape, dtype=mx.bool_)
        else:
            # Condition through inpainting
            cond_data = mx.zeros((B, T, Da + Do))
            cond_mask = mx.zeros((B, T, Da + Do), dtype=mx.bool_)

            # Fill in observed obs for first To timesteps
            obs_block = mx.concatenate(
                [mx.zeros((B, To, Da)), nobs[:, :To]], axis=-1
            )
            pad_block = mx.zeros((B, T - To, Da + Do))
            cond_data = mx.concatenate([obs_block, pad_block], axis=1)

            # Build mask
            mask_obs = mx.concatenate(
                [
                    mx.zeros((B, To, Da), dtype=mx.bool_),
                    mx.ones((B, To, Do), dtype=mx.bool_),
                ],
                axis=-1,
            )
            mask_pad = mx.zeros((B, T - To, Da + Do), dtype=mx.bool_)
            cond_mask = mx.concatenate([mask_obs, mask_pad], axis=1)

        # Run sampling
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond)

        # Materialize denoising result before post-processing.
        # Without this, the entire lazy computation graph from all denoising
        # steps would be kept alive, causing unbounded memory growth.
        mx.eval(nsample)

        # Extract action dims and unnormalize
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Get action window
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}

        if not self.obs_as_cond:
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred

        return result

    # -- Training loss ---------------------------------------------------------

    def compute_loss(self, batch: Dict) -> mx.array:
        """Compute training loss (MSE between predicted and true noise).

        Args:
            batch: ``{'obs': (B, T, obs_dim), 'action': (B, T, action_dim)}``

        Returns:
            Scalar loss.
        """
        # Normalize
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        cond = None
        trajectory = action

        if self.obs_as_cond:
            cond = obs[:, : self.n_obs_steps, :]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = mx.concatenate([action, obs], axis=-1)

        # Generate inpainting mask
        if self.pred_action_steps_only:
            condition_mask = mx.zeros(trajectory.shape, dtype=mx.bool_)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise
        noise = mx.random.normal(trajectory.shape)
        B = trajectory.shape[0]

        # Sample random timesteps
        timesteps = mx.random.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,)
        )

        # Forward diffusion: add noise
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Compute loss mask
        loss_mask = ~condition_mask

        # Apply conditioning (inpainting)
        noisy_trajectory = mx.where(condition_mask, trajectory, noisy_trajectory)

        # Predict noise
        pred = self.model(noisy_trajectory, timesteps, cond)

        # Determine target based on prediction type
        pred_type = self.noise_scheduler.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

        # MSE loss with mask
        loss = (pred - target) ** 2
        loss = loss * loss_mask.astype(loss.dtype)
        loss = loss.reshape(B, -1).mean(axis=-1).mean()

        return loss
