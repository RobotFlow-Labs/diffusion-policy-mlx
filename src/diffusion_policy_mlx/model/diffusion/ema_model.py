"""Exponential Moving Average of model parameters.

Upstream: diffusion_policy/model/diffusion/ema_model.py

Maintains a shadow copy of model weights, updated each step:
    ema_param = decay * ema_param + (1-decay) * new_param

Decay schedule ramps from min_value to max_value over training:
    decay = 1 - (1 + step/inv_gamma)^(-power)

This is NOT an nn.Module — it manages a separate copy of model parameters.
"""

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


class EMAModel:
    """Exponential Moving Average of model parameters.

    Args:
        model: The MLX nn.Module whose parameters to track.
        update_after_step: Don't update EMA until this many steps have passed.
        inv_gamma: Inverse multiplicative factor of EMA warmup. Default: 1.0.
        power: Exponential factor of EMA warmup. Default: 2/3.
        min_value: Minimum EMA decay rate. Default: 0.0.
        max_value: Maximum EMA decay rate. Default: 0.9999.

    Notes on decay schedule (from @crowsonkb):
        gamma=1, power=2/3 are good for models trained 1M+ steps
        (reaches decay 0.999 at 31.6K steps, 0.9999 at 1M steps).
        gamma=1, power=3/4 for shorter training
        (reaches decay 0.999 at 10K steps, 0.9999 at 215.4K steps).
    """

    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.step_count = 0

        # Clone model parameters for EMA shadow.
        # We cannot use copy.deepcopy on MLX arrays, so we flatten to a dict
        # and create fresh mx.arrays from the data.
        self.averaged_params: Dict[str, mx.array] = {}
        for key, value in mlx.utils.tree_flatten(model.parameters()):
            self.averaged_params[key] = mx.array(value)
        mx.eval(*self.averaged_params.values())

    def get_decay(self, optimization_step: int) -> float:
        """Compute the decay factor for the given optimization step.

        Matches upstream formula:
            step = max(0, optimization_step - update_after_step - 1)
            if step <= 0: return 0.0
            decay = 1 - (1 + step / inv_gamma) ^ (-power)
            return clamp(decay, min_value, max_value)
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    def step(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters.

        EMA update rule:
            ema_param = decay * ema_param + (1 - decay) * model_param
        """
        decay = self.get_decay(self.step_count)
        self.step_count += 1

        model_params = dict(mlx.utils.tree_flatten(model.parameters()))
        for key in self.averaged_params:
            if key in model_params:
                self.averaged_params[key] = (
                    decay * self.averaged_params[key] + (1.0 - decay) * model_params[key]
                )

        # Materialize the updated EMA arrays
        mx.eval(*self.averaged_params.values())

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into a model for evaluation."""
        model.load_weights(list(self.averaged_params.items()))

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA state for checkpointing."""
        return {
            "averaged_params": self.averaged_params,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore EMA state from a checkpoint."""
        self.averaged_params = state["averaged_params"]
        self.step_count = state["step_count"]
