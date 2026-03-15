"""Low-dimensional mask generator for diffusion inpainting.

Upstream: diffusion_policy/model/diffusion/mask_generator.py

Creates boolean masks of shape ``(B, T, D)`` indicating which dimensions
are observed (True) vs generated (False) during diffusion training.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class LowdimMaskGenerator:
    """Generate observation/action masks for diffusion inpainting.

    The trajectory has shape ``(B, T, D)`` where ``D = action_dim + obs_dim``.
    Actions occupy dims ``[:action_dim]``, observations occupy ``[action_dim:]``.

    Args:
        action_dim: Number of action dimensions.
        obs_dim: Number of observation dimensions (0 when obs_as_global_cond).
        max_n_obs_steps: How many timesteps of observations are visible.
        fix_obs_steps: If True, always use ``max_n_obs_steps``; else random in
                       ``[1, max_n_obs_steps]``.
        action_visible: If True, past actions (before obs steps) are also visible.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int = 0,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    def __call__(self, shape) -> mx.array:
        """Generate mask.

        Args:
            shape: ``(B, T, D)`` tuple.

        Returns:
            Boolean array of shape ``(B, T, D)``.  ``True`` = observed/visible.
        """
        B, T, D = shape
        assert D == self.action_dim + self.obs_dim, (
            f"D={D} != action_dim={self.action_dim} + obs_dim={self.obs_dim}"
        )

        # When obs_dim == 0 (obs_as_global_cond), everything is False (nothing
        # is conditioned via inpainting — conditioning goes through global_cond).
        if self.obs_dim == 0:
            return mx.zeros((B, T, D), dtype=mx.bool_)

        # Determine number of obs steps per batch element
        if self.fix_obs_steps:
            n_obs = self.max_n_obs_steps
            # obs_steps: (B,) all equal to n_obs
            obs_steps = mx.full((B,), n_obs, dtype=mx.int32)
        else:
            obs_steps = mx.array(
                np.random.randint(1, self.max_n_obs_steps + 1, size=(B,)),
                dtype=mx.int32,
            )

        # Build temporal mask: (B, T) — True for t < obs_steps[b]
        steps = mx.broadcast_to(mx.arange(T).reshape(1, T), (B, T))  # (B, T)
        obs_steps_expanded = mx.expand_dims(obs_steps, axis=1)  # (B, 1)
        time_mask = steps < obs_steps_expanded  # (B, T)
        time_mask = mx.expand_dims(time_mask, axis=2)  # (B, T, 1)
        time_mask = mx.broadcast_to(time_mask, (B, T, D))  # (B, T, D)

        # Build dim mask: obs dims are the last obs_dim dimensions
        dim_flag = mx.concatenate(
            [
                mx.zeros((D - self.obs_dim,), dtype=mx.bool_),  # action dims
                mx.ones((self.obs_dim,), dtype=mx.bool_),  # obs dims
            ]
        )
        is_obs_dim = mx.broadcast_to(dim_flag.reshape(1, 1, D), (B, T, D))

        obs_mask = time_mask & is_obs_dim

        # Action mask (optional): actions visible for t < obs_steps - 1
        if self.action_visible:
            action_steps = mx.maximum(obs_steps - 1, mx.zeros_like(obs_steps))
            action_steps_expanded = mx.expand_dims(action_steps, axis=1)
            action_time_mask = steps < action_steps_expanded
            action_time_mask = mx.expand_dims(action_time_mask, axis=2)
            action_time_mask = mx.broadcast_to(action_time_mask, (B, T, D))

            is_action_dim = mx.broadcast_to(
                mx.concatenate(
                    [
                        mx.ones((self.action_dim,), dtype=mx.bool_),
                        mx.zeros((self.obs_dim,), dtype=mx.bool_),
                    ]
                ).reshape(1, 1, D),
                (B, T, D),
            )
            action_mask = action_time_mask & is_action_dim
            return obs_mask | action_mask

        return obs_mask
