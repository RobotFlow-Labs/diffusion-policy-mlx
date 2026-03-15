"""Validation during training.

Provides :class:`TrainingValidator` which runs periodic validation passes
over a held-out dataset and returns aggregate loss metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from diffusion_policy_mlx.training.collate import collate_batch


class TrainingValidator:
    """Run validation at regular intervals during training.

    Evaluates the policy's ``compute_loss`` on batches drawn from a
    validation dataset. Intended to be called from the training loop.

    Example::

        validator = TrainingValidator(
            policy=policy,
            val_dataset=val_dataset,
            eval_every_n_epochs=10,
        )

        for epoch in range(num_epochs):
            # ... training ...
            if validator.should_validate(epoch):
                val_metrics = validator.validate()
                print(f"val_loss: {val_metrics['val_loss']:.4f}")

    Args:
        policy: The policy module (must have ``compute_loss(batch) -> mx.array``).
        val_dataset: A dataset supporting ``__len__`` and ``__getitem__``.
        eval_every_n_epochs: Run validation every N epochs. Default: 10.
        n_val_batches: Maximum number of batches to evaluate. Default: 10.
        batch_size: Validation batch size. Default: 64.
    """

    def __init__(
        self,
        policy: nn.Module,
        val_dataset: Any,
        eval_every_n_epochs: int = 10,
        n_val_batches: int = 10,
        batch_size: int = 64,
    ):
        self.policy = policy
        self.val_dataset = val_dataset
        self.eval_every_n_epochs = eval_every_n_epochs
        self.n_val_batches = n_val_batches
        self.batch_size = batch_size
        self._best_val_loss: Optional[float] = None

    def should_validate(self, epoch: int) -> bool:
        """Check whether validation should run at this epoch.

        Returns True if ``epoch`` is a multiple of ``eval_every_n_epochs``
        (1-indexed, so epoch 9 triggers for eval_every_n_epochs=10, i.e.
        after the 10th epoch).

        Args:
            epoch: Zero-based epoch index.

        Returns:
            True if validation should run.
        """
        return (epoch + 1) % self.eval_every_n_epochs == 0

    def validate(self) -> Dict[str, float]:
        """Run a validation pass and return aggregate metrics.

        Draws up to ``n_val_batches`` batches from the validation dataset
        (using a fixed random permutation for reproducibility within the
        call) and computes the mean loss.

        Returns:
            Dictionary with keys:
                - ``val_loss``: Mean validation loss across batches.
                - ``val_steps``: Number of batches evaluated.
                - ``val_best``: Best validation loss seen so far.
        """
        dataset_size = len(self.val_dataset)
        if dataset_size == 0:
            return {"val_loss": float("nan"), "val_steps": 0, "val_best": float("nan")}

        # Deterministic shuffle for validation
        rng = np.random.RandomState(seed=0)
        indices = rng.permutation(dataset_size)

        losses: List[float] = []
        batches_done = 0

        for batch_start in range(0, len(indices), self.batch_size):
            if batches_done >= self.n_val_batches:
                break

            batch_end = min(batch_start + self.batch_size, len(indices))
            batch_idx = indices[batch_start:batch_end]

            if len(batch_idx) < 2:
                continue

            batch = collate_batch(
                [self.val_dataset[int(i)] for i in batch_idx]
            )

            # Compute loss without gradient tracking
            loss = self.policy.compute_loss(batch)
            mx.eval(loss)
            losses.append(float(loss))
            batches_done += 1

        if not losses:
            return {"val_loss": float("nan"), "val_steps": 0, "val_best": float("nan")}

        mean_loss = float(np.mean(losses))

        # Track best
        if self._best_val_loss is None or mean_loss < self._best_val_loss:
            self._best_val_loss = mean_loss

        return {
            "val_loss": mean_loss,
            "val_steps": batches_done,
            "val_best": self._best_val_loss,
        }

    @property
    def best_val_loss(self) -> Optional[float]:
        """Return the best validation loss seen so far, or None."""
        return self._best_val_loss

    def check_early_stopping(
        self,
        val_loss: float,
        patience: int = 20,
        min_delta: float = 1e-5,
    ) -> bool:
        """Check if training should stop early based on validation loss.

        This is a stateless check — the caller is responsible for counting
        how many epochs have passed without improvement.

        Args:
            val_loss: Current validation loss.
            patience: Not used in this stateless check (kept for API
                compatibility). The caller tracks patience.
            min_delta: Minimum improvement to count as "better".

        Returns:
            True if the current val_loss is worse than the best by more
            than ``min_delta`` (i.e., no improvement).
        """
        if self._best_val_loss is None:
            return False
        return val_loss > (self._best_val_loss + min_delta)
