"""Abstract base class for image-conditioned policies.

Upstream: diffusion_policy/policy/base_image_policy.py
"""

from __future__ import annotations

from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class BaseImagePolicy(nn.Module):
    """Abstract base class for image-conditioned diffusion policies.

    Subclasses must implement ``predict_action`` and ``set_normalizer``.
    """

    def predict_action(self, obs_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Generate actions from observations.

        Args:
            obs_dict: ``{key: (B, To, *shape)}`` observation dict.

        Returns:
            Dict with at least ``'action': (B, Ta, Da)`` and
            ``'action_pred': (B, T, Da)``.
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state (for recurrent / stateful policies)."""
        pass

    def set_normalizer(self, normalizer):
        """Set the data normalizer."""
        raise NotImplementedError
