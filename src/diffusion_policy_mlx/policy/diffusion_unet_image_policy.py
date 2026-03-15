"""Diffusion UNet Image Policy — image-only policy (no hybrid low-dim features).

Upstream: diffusion_policy/policy/diffusion_unet_image_policy.py

Uses only RGB observations through a vision encoder (MultiImageObsEncoder).
Unlike the hybrid policy, no low-dim features (agent_pos, etc.) are
concatenated.  This is the simpler image-based variant.

Two conditioning modes:
  - ``obs_as_global_cond=True`` (default): encoded observations as global
    conditioning.  UNet input is ``(B, horizon, action_dim)``.
  - ``obs_as_global_cond=False``: encoded observations concatenated with
    noisy actions; inpainting mask conditions on observed timesteps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mlx.core as mx

from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
)
from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_mlx.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_mlx.model.vision.model_getter import get_resnet
from diffusion_policy_mlx.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from diffusion_policy_mlx.policy.base_image_policy import BaseImagePolicy


class DiffusionUnetImagePolicy(BaseImagePolicy):
    """Image-only diffusion policy using a 1D UNet denoiser.

    Unlike ``DiffusionUnetHybridImagePolicy``, the ``shape_meta`` for
    observations should contain only RGB keys (no ``low_dim`` keys like
    ``agent_pos``).

    Args:
        shape_meta: Dict describing observation and action shapes::

            {
                'obs': {
                    'image': {'shape': (3, 96, 96), 'type': 'rgb'},
                },
                'action': {'shape': (2,)},
            }

        noise_scheduler: A DDPMScheduler or DDIMScheduler instance.
        horizon: Full trajectory length (T).
        n_action_steps: Number of action steps to execute (Ta).
        n_obs_steps: Number of observation steps (To).
        num_inference_steps: Number of denoising steps at inference.
        obs_as_global_cond: If True, encode observations as global conditioning.
        crop_shape: Random crop size for RGB inputs.
        diffusion_step_embed_dim: Timestep embedding dimension.
        down_dims: Channel dimensions per UNet level.
        kernel_size: Conv kernel size for UNet.
        n_groups: GroupNorm groups in UNet.
        cond_predict_scale: Use FiLM scale+bias (True) vs bias-only (False).
        obs_encoder_group_norm: Replace BatchNorm with GroupNorm in vision encoder.
        eval_fixed_crop: Use fixed center crop at eval time.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_global_cond: bool = True,
        crop_shape: Optional[Tuple[int, int]] = (76, 76),
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        **kwargs,
    ):
        super().__init__()

        # -- Parse shape_meta --------------------------------------------------
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1, "action shape must be 1-D"
        action_dim = action_shape[0]

        # -- Build observation encoder -----------------------------------------
        obs_encoder = self._build_obs_encoder(
            shape_meta=shape_meta,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
        )
        self.obs_encoder = obs_encoder

        # Determine obs feature dim via a dummy forward pass
        obs_feature_dim = obs_encoder.output_shape()[0]

        # -- Build UNet --------------------------------------------------------
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # -- Mask generator (for non-global-cond mode) -------------------------
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        # -- Scheduler & normalizer --------------------------------------------
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        # -- Store config ------------------------------------------------------
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # -- Obs encoder builder ---------------------------------------------------

    @staticmethod
    def _build_obs_encoder(
        shape_meta: dict,
        crop_shape: Optional[Tuple[int, int]],
        obs_encoder_group_norm: bool,
        eval_fixed_crop: bool,
    ) -> MultiImageObsEncoder:
        """Build a ``MultiImageObsEncoder`` from shape_meta."""
        rgb_model = get_resnet("resnet18")
        return MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=crop_shape,
            use_group_norm=obs_encoder_group_norm,
            share_rgb_model=True,
            imagenet_norm=True,
        )

    # -- Normalizer ------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the data normalizer."""
        self.normalizer = normalizer

    # -- Observation encoding helper -------------------------------------------

    def _encode_obs(self, nobs: Dict[str, mx.array], B: int) -> mx.array:
        """Encode observations, flattening temporal dim into batch.

        Args:
            nobs: Normalized observation dict ``{key: (B, To, ...)}``.
            B: Batch size.

        Returns:
            Encoded features.  Shape depends on ``obs_as_global_cond``:
              - True:  ``(B, To * feat_dim)``
              - False: ``(B, To, feat_dim)``
        """
        To = self.n_obs_steps

        # Flatten (B, To, ...) -> (B*To, ...) for the encoder
        obs_flat = {}
        for k, v in nobs.items():
            v_sliced = v[:, :To]  # (B, To, ...)
            obs_flat[k] = v_sliced.reshape(B * To, *v.shape[2:])

        features = self.obs_encoder(obs_flat)  # (B*To, feat_dim)

        if self.obs_as_global_cond:
            return features.reshape(B, -1)  # (B, To * feat_dim)
        else:
            return features.reshape(B, To, -1)  # (B, To, feat_dim)

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
            obs_dict: ``{'image': (B, To, C, H, W)}``

        Returns:
            ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon, Da)}``
        """
        assert "past_action" not in obs_dict  # not implemented yet

        # Normalize observations
        nobs = self.normalizer.normalize({"obs": obs_dict})["obs"]

        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            # Encode observations as global conditioning
            global_cond = self._encode_obs(nobs, B)  # (B, To * feat_dim)

            cond_data = mx.zeros((B, T, Da))
            cond_mask = mx.zeros((B, T, Da), dtype=mx.bool_)
        else:
            # Encode observations for inpainting
            Do = self.obs_feature_dim
            nobs_features = self._encode_obs(nobs, B)  # (B, To, Do)

            obs_block = mx.concatenate(
                [mx.zeros((B, To, Da)), nobs_features], axis=-1
            )
            pad_block = mx.zeros((B, T - To, Da + Do))
            cond_data = mx.concatenate([obs_block, pad_block], axis=1)

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

        # Materialize denoising result before post-processing.
        # Without this, the entire lazy computation graph from all denoising
        # steps would be kept alive, causing unbounded memory growth.
        mx.eval(nsample)

        # Extract action dims and unnormalize
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract the action window
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    # -- Training loss ---------------------------------------------------------

    def compute_loss(self, batch: Dict) -> mx.array:
        """Compute training loss (MSE between predicted and true noise).

        Args:
            batch: ``{'obs': {key: (B, T, ...)}, 'action': (B, T, Da)}``

        Returns:
            Scalar loss.
        """
        # Normalize
        nbatch = self.normalizer.normalize(
            {"obs": batch["obs"], "action": batch["action"]}
        )
        nobs = nbatch["obs"]
        nactions = nbatch["action"]

        B = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions  # (B, horizon, action_dim)

        if self.obs_as_global_cond:
            # Encode observations as global conditioning
            global_cond = self._encode_obs(nobs, B)  # (B, To * feat_dim)
            cond_data = trajectory
        else:
            # Encode obs and concatenate with actions
            obs_flat = {}
            for k, v in nobs.items():
                obs_flat[k] = v.reshape(B * horizon, *v.shape[2:])
            nobs_features = self.obs_encoder(obs_flat)
            nobs_features = nobs_features.reshape(B, horizon, -1)

            cond_data = mx.concatenate([nactions, nobs_features], axis=-1)
            trajectory = mx.stop_gradient(cond_data)

        # Generate inpainting mask
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
        noisy_trajectory = mx.where(condition_mask, cond_data, noisy_trajectory)

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
