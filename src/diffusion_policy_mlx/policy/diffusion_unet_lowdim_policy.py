"""Diffusion UNet Low-Dim Policy — denoising policy for low-dimensional observations.

Upstream: diffusion_policy/policy/diffusion_unet_lowdim_policy.py

No vision encoder is used.  Raw state vectors (joint positions, keypoints,
etc.) are fed directly as conditioning to the UNet denoiser.

Three conditioning modes:
  - ``obs_as_global_cond=True`` (default): observations flattened and passed
    as a global conditioning vector.  UNet input is ``(B, T, action_dim)``.
  - ``obs_as_local_cond=True``: observations passed as per-timestep local
    conditioning.  UNet input is ``(B, T, action_dim)``.
  - Neither: observations concatenated with actions along feature dim
    (inpainting mode).  UNet input is ``(B, T, action_dim + obs_dim)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx

from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_mlx.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_mlx.policy.base_lowdim_policy import BaseLowdimPolicy


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    """Diffusion policy using a 1D UNet denoiser with low-dim observations.

    Args:
        model: A ``ConditionalUnet1D`` instance.
        noise_scheduler: A DDPMScheduler or DDIMScheduler instance.
        horizon: Full trajectory length (T).
        obs_dim: Observation feature dimension (Do).
        action_dim: Action dimension (Da).
        n_action_steps: Number of action steps to execute (Ta).
        n_obs_steps: Number of observation steps (To).
        num_inference_steps: Number of denoising steps at inference.
        obs_as_local_cond: Use per-timestep local conditioning.
        obs_as_global_cond: Use global conditioning vector.
        pred_action_steps_only: Only predict action steps (requires global cond).
        oa_step_convention: Offset action start by -1.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond), (
            "Cannot use both local and global conditioning"
        )
        if pred_action_steps_only:
            assert obs_as_global_cond, (
                "pred_action_steps_only requires obs_as_global_cond"
            )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
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
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
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
        local_cond: Optional[mx.array] = None,
        global_cond: Optional[mx.array] = None,
    ) -> mx.array:
        """Run iterative denoising with optional inpainting.

        Args:
            condition_data: ``(B, T, D)`` conditioning values.
            condition_mask: ``(B, T, D)`` boolean mask (True = observed).
            local_cond: Optional per-timestep conditioning.
            global_cond: Optional global conditioning vector.

        Returns:
            Denoised trajectory ``(B, T, D)``.
        """
        trajectory = mx.random.normal(condition_data.shape)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Apply conditioning (inpainting)
            trajectory = mx.where(condition_mask, condition_data, trajectory)

            # Predict noise / x0
            model_output = self.model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # Scheduler step
            out = self.noise_scheduler.step(model_output, t, trajectory)
            trajectory = out.prev_sample

        # Final conditioning enforcement
        trajectory = mx.where(condition_mask, condition_data, trajectory)

        return trajectory

    # -- Inference -------------------------------------------------------------

    def predict_action(
        self, obs_dict: Dict[str, mx.array]
    ) -> Dict[str, mx.array]:
        """Generate actions via iterative denoising.

        Args:
            obs_dict: Must contain ``'obs'`` key with shape ``(B, To, Do)``.

        Returns:
            Dict with ``'action'`` and ``'action_pred'`` keys.
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B = nobs.shape[0]
        Do = nobs.shape[2]
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim
        assert Do == self.obs_dim

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None

        if self.obs_as_local_cond:
            # Condition through local feature: obs padded with zeros after To
            local_cond = mx.zeros((B, T, Do))
            # Set first To timesteps
            obs_slice = nobs[:, :To]
            local_cond = mx.concatenate(
                [obs_slice, mx.zeros((B, T - To, Do))], axis=1
            )
            cond_data = mx.zeros((B, T, Da))
            cond_mask = mx.zeros((B, T, Da), dtype=mx.bool_)

        elif self.obs_as_global_cond:
            # Condition through global feature: flatten obs
            global_cond = nobs[:, :To].reshape(B, -1)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            else:
                shape = (B, T, Da)
            cond_data = mx.zeros(shape)
            cond_mask = mx.zeros(shape, dtype=mx.bool_)

        else:
            # Condition through inpainting
            cond_data = mx.zeros((B, T, Da + Do))
            cond_mask = mx.zeros((B, T, Da + Do), dtype=mx.bool_)

            # Fill observed obs into the obs portion of cond_data for t < To
            obs_block = mx.concatenate(
                [mx.zeros((B, To, Da)), nobs[:, :To]], axis=-1
            )
            pad_block = mx.zeros((B, T - To, Da + Do))
            cond_data = mx.concatenate([obs_block, pad_block], axis=1)

            # Build mask: obs dims visible for first To steps
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
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Get action window
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}

        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action_obs_pred = obs_pred[:, start:end]
            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred

        return result

    # -- Training loss ---------------------------------------------------------

    def compute_loss(self, batch: Dict) -> mx.array:
        """Compute training loss (MSE between predicted and true noise).

        Args:
            batch: ``{'obs': (B, T, Do), 'action': (B, T, Da)}``

        Returns:
            Scalar loss.
        """
        # Normalize input
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        B = action.shape[0]

        # Handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            # Zero out observations after n_obs_steps
            local_cond = mx.concatenate(
                [
                    obs[:, : self.n_obs_steps, :],
                    mx.zeros((B, obs.shape[1] - self.n_obs_steps, obs.shape[2])),
                ],
                axis=1,
            )

        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(B, -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
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

        # Sample random timesteps
        timesteps = mx.random.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,)
        )

        # Forward diffusion: add noise
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # Compute loss mask (invert condition mask)
        loss_mask = ~condition_mask

        # Apply conditioning (inpainting)
        noisy_trajectory = mx.where(condition_mask, trajectory, noisy_trajectory)

        # Predict noise
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

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
