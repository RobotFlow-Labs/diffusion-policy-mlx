"""Abstract base class for low-dimensional observation policies.

Upstream: diffusion_policy/policy/base_lowdim_policy.py

Low-dim policies work with raw state vectors (joint positions, forces,
keypoints, etc.) instead of image observations.  No vision encoder
is needed.
"""

from __future__ import annotations

from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class BaseLowdimPolicy(nn.Module):
    """Abstract base class for low-dimensional observation policies.

    Subclasses must implement ``predict_action`` and ``set_normalizer``.

    Unlike ``BaseImagePolicy``, observations are raw state vectors
    (e.g. joint positions, forces) rather than images.
    """

    def predict_action(self, obs_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Generate actions from low-dimensional observations.

        Args:
            obs_dict: Must contain ``'obs'`` key with shape ``(B, To, Do)``.

        Returns:
            Dict with at least ``'action': (B, Ta, Da)`` and
            ``'action_pred': (B, T, Da)``.

        Horizon diagram::

            |o|o|o|
            | | |a|a|a|a|
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state (for recurrent / stateful policies)."""
        pass

    def set_normalizer(self, normalizer):
        """Set the data normalizer."""
        raise NotImplementedError
