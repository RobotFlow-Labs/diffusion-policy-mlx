"""Diffusion Transformer Hybrid Image Policy.

Upstream: diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py

Ties together:
  - ``MultiImageObsEncoder``: encodes RGB + low-dim observations
  - ``TransformerForDiffusion``: denoises action trajectories
  - ``DDPMScheduler`` / ``DDIMScheduler``: noise schedule
  - ``LinearNormalizer``: data normalization

Two conditioning modes:
  - ``obs_as_cond=True`` (default): observations encoded and passed as
    conditioning tokens to the transformer decoder via cross-attention.
  - ``obs_as_cond=False``: observations concatenated with noisy actions
    along feature dim; inpainting mask conditions on observed timesteps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mlx.core as mx

from diffusion_policy_mlx.model.common.normalizer import LinearNormalizer
from diffusion_policy_mlx.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_mlx.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy_mlx.model.vision.model_getter import get_resnet
from diffusion_policy_mlx.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)
from diffusion_policy_mlx.policy.base_image_policy import BaseImagePolicy


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    """Vision-conditioned diffusion policy using a Transformer denoiser.

    Args:
        shape_meta: Dict describing observation and action shapes.
        noise_scheduler: A DDPMScheduler or DDIMScheduler instance.
        horizon: Full trajectory length (T).
        n_action_steps: Number of action steps to execute (Ta).
        n_obs_steps: Number of observation steps (To).
        num_inference_steps: Number of denoising steps at inference.
        crop_shape: Random crop size for RGB inputs.
        obs_encoder_group_norm: Replace BatchNorm with GroupNorm in vision encoder.
        eval_fixed_crop: Use fixed center crop at eval time.
        n_layer: Number of transformer layers.
        n_cond_layers: Number of conditioning encoder layers.
        n_head: Number of attention heads.
        n_emb: Embedding dimension.
        p_drop_emb: Embedding dropout.
        p_drop_attn: Attention/FFN dropout.
        causal_attn: Use causal attention mask.
        time_as_cond: Pass timestep as conditioning token.
        obs_as_cond: Use observation conditioning via cross-attention.
        pred_action_steps_only: Only predict action steps (not full horizon).
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        # task params
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        # image
        crop_shape: Optional[Tuple[int, int]] = (76, 76),
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        # arch
        n_layer: int = 8,
        n_cond_layers: int = 0,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
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

        # Determine obs feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # -- Build Transformer -------------------------------------------------
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
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
            Encoded features ``(B, To, feat_dim)``.
        """
        To = self.n_obs_steps

        # Flatten (B, To, ...) -> (B*To, ...) for the encoder
        obs_flat = {}
        for k, v in nobs.items():
            v_sliced = v[:, :To]
            obs_flat[k] = v_sliced.reshape(B * To, *v.shape[2:])

        features = self.obs_encoder(obs_flat)  # (B*To, feat_dim)
        return features.reshape(B, To, -1)  # (B, To, feat_dim)

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
            obs_dict: ``{'image': (B, To, C, H, W), 'agent_pos': (B, To, D), ...}``

        Returns:
            ``{'action': (B, n_action_steps, Da), 'action_pred': (B, T, Da)}``
        """
        # Normalize observations
        nobs = self.normalizer.normalize({"obs": obs_dict})["obs"]

        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        cond = None

        if self.obs_as_cond:
            # Encode observations as conditioning tokens
            cond = self._encode_obs(nobs, B)  # (B, To, feat_dim)

            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = mx.zeros(shape)
            cond_mask = mx.zeros(shape, dtype=mx.bool_)
        else:
            # Condition through inpainting
            Do = self.obs_feature_dim
            nobs_features = self._encode_obs(nobs, B)  # (B, To, Do)

            cond_data = mx.zeros((B, T, Da + Do))
            cond_mask = mx.zeros((B, T, Da + Do), dtype=mx.bool_)

            # Fill in observed obs features for first To timesteps
            obs_block = mx.concatenate(
                [mx.zeros((B, To, Da)), nobs_features], axis=-1
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
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond)

        # Extract action dims and unnormalize
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract the action window
        if self.pred_action_steps_only:
            action = action_pred
        else:
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
        To = self.n_obs_steps

        cond = None
        trajectory = nactions

        if self.obs_as_cond:
            # Encode observations as conditioning tokens
            cond = self._encode_obs(nobs, B)  # (B, To, feat_dim)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            # Encode obs and concatenate with actions
            obs_flat = {}
            for k, v in nobs.items():
                obs_flat[k] = v.reshape(B * horizon, *v.shape[2:])
            nobs_features = self.obs_encoder(obs_flat)
            nobs_features = nobs_features.reshape(B, horizon, -1)
            trajectory = mx.stop_gradient(
                mx.concatenate([nactions, nobs_features], axis=-1)
            )

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
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Compute loss mask (invert condition mask)
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
