"""Learning rate schedulers for MLX optimizers.

Upstream: diffusion_policy/model/common/lr_scheduler.py (wraps diffusers).

We implement the schedulers directly, operating on the MLX optimizer's
``learning_rate`` attribute. Each scheduler has a ``.step()`` method that
advances the schedule by one step and updates the optimizer's LR in place.
"""

from __future__ import annotations

import math
from typing import Union


class CosineAnnealingLR:
    """Cosine annealing with optional linear warmup.

    Schedule:
        warmup phase:  linear from 0 to base_lr over ``num_warmup_steps``
        cosine phase:  base_lr -> min_lr following cosine curve

    Args:
        optimizer: MLX optimizer with a ``learning_rate`` attribute.
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps. Default: 0.
        min_lr: Minimum learning rate at the end of cosine decay. Default: 0.0.
    """

    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.learning_rate)
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self) -> None:
        """Advance the schedule by one step and update the optimizer LR."""
        self.current_step += 1
        lr = self._compute_lr()
        self.optimizer.learning_rate = lr

    def _compute_lr(self) -> float:
        step = self.current_step
        if step < self.num_warmup_steps:
            return self.base_lr * step / max(1, self.num_warmup_steps)
        progress = (step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )


class LinearLR:
    """Linear decay with optional linear warmup.

    Schedule:
        warmup phase: linear from 0 to base_lr over ``num_warmup_steps``
        decay phase:  linear from base_lr to 0 over remaining steps

    Args:
        optimizer: MLX optimizer with a ``learning_rate`` attribute.
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps. Default: 0.
    """

    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
    ):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.learning_rate)
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.current_step = 0

    def step(self) -> None:
        """Advance the schedule by one step and update the optimizer LR."""
        self.current_step += 1
        lr = self._compute_lr()
        self.optimizer.learning_rate = lr

    def _compute_lr(self) -> float:
        step = self.current_step
        if step < self.num_warmup_steps:
            return self.base_lr * step / max(1, self.num_warmup_steps)
        progress = (step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps
        )
        return self.base_lr * max(0.0, 1.0 - progress)


class ConstantLR:
    """Constant learning rate (no-op scheduler).

    Args:
        optimizer: MLX optimizer with a ``learning_rate`` attribute.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self) -> None:
        """No-op: LR stays constant."""
        pass


class ConstantWithWarmupLR:
    """Constant learning rate with linear warmup.

    Schedule:
        warmup phase: linear from 0 to base_lr over ``num_warmup_steps``
        constant phase: base_lr forever after

    Args:
        optimizer: MLX optimizer with a ``learning_rate`` attribute.
        num_warmup_steps: Number of warmup steps.
    """

    def __init__(self, optimizer, num_warmup_steps: int):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.learning_rate)
        self.num_warmup_steps = num_warmup_steps
        self.current_step = 0

    def step(self) -> None:
        """Advance the schedule by one step and update the optimizer LR."""
        self.current_step += 1
        lr = self._compute_lr()
        self.optimizer.learning_rate = lr

    def _compute_lr(self) -> float:
        step = self.current_step
        if step < self.num_warmup_steps:
            return self.base_lr * step / max(1, self.num_warmup_steps)
        return self.base_lr


def get_scheduler(
    name: str,
    optimizer,
    num_training_steps: int = 0,
    num_warmup_steps: int = 0,
    **kwargs,
) -> Union[CosineAnnealingLR, LinearLR, ConstantLR, ConstantWithWarmupLR]:
    """Factory for LR schedulers.

    Args:
        name: One of 'cosine', 'linear', 'constant', 'constant_with_warmup'.
        optimizer: MLX optimizer.
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps. Default: 0.
        **kwargs: Extra keyword arguments forwarded to the scheduler.

    Returns:
        A scheduler instance with a ``.step()`` method.

    Raises:
        ValueError: If ``name`` is not a recognized scheduler type.
    """
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer, num_training_steps, num_warmup_steps, **kwargs
        )
    elif name == "linear":
        return LinearLR(optimizer, num_training_steps, num_warmup_steps)
    elif name == "constant":
        return ConstantLR(optimizer)
    elif name == "constant_with_warmup":
        return ConstantWithWarmupLR(optimizer, num_warmup_steps)
    else:
        raise ValueError(
            f"Unknown scheduler: {name!r}. "
            f"Choose from: 'cosine', 'linear', 'constant', 'constant_with_warmup'."
        )
