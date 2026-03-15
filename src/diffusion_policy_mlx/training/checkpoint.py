"""Checkpointing utilities for Diffusion Policy MLX training.

Provides:
    - ``save_checkpoint`` / ``load_checkpoint`` for full training state
    - ``TopKCheckpointManager`` for keeping only the best K checkpoints
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


def save_checkpoint(
    policy: nn.Module,
    ema,
    optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint.

    Creates a directory containing:
        - model.npz: model parameters
        - ema.npz: EMA shadow parameters
        - train_state.json: epoch, step, and any extra metadata

    Args:
        policy: The model being trained.
        ema: EMAModel instance (or None to skip EMA saving).
        optimizer: MLX optimizer (metadata only; MLX optimizer state is
            recomputed on resume since it's cheap).
        epoch: Current epoch number.
        step: Current global step number.
        checkpoint_dir: Base directory for checkpoints.
        extra_metadata: Optional extra metadata to save.

    Returns:
        Path to the saved checkpoint directory.
    """
    path = Path(checkpoint_dir) / f"epoch_{epoch:04d}_step_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)

    # Model weights
    model_params = dict(mlx.utils.tree_flatten(policy.parameters()))
    mx.savez(str(path / "model.npz"), **model_params)

    # EMA weights
    if ema is not None:
        mx.savez(str(path / "ema.npz"), **ema.averaged_params)

    # Training state metadata
    metadata: Dict[str, Any] = {
        "epoch": epoch,
        "step": step,
    }
    if ema is not None:
        metadata["ema_step_count"] = ema.step_count
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(path / "train_state.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    policy: nn.Module,
    ema=None,
    optimizer=None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint directory.
        policy: Model to load weights into.
        ema: Optional EMAModel to restore EMA weights into.
        optimizer: Unused (MLX optimizer state is recomputed). Kept for API
            compatibility.

    Returns:
        Dictionary with training metadata (epoch, step, etc.).
    """
    ckpt_path = Path(path)

    # Load model weights
    model_file = ckpt_path / "model.npz"
    if model_file.exists():
        weights = dict(mx.load(str(model_file)))
        policy.load_weights(list(weights.items()))

    # Load EMA weights
    ema_file = ckpt_path / "ema.npz"
    if ema is not None and ema_file.exists():
        ema_weights = dict(mx.load(str(ema_file)))
        ema.averaged_params = ema_weights

    # Load training state
    state_file = ckpt_path / "train_state.json"
    metadata: Dict[str, Any] = {}
    if state_file.exists():
        with open(state_file, "r") as f:
            metadata = json.load(f)

    # Restore EMA step count
    if ema is not None and "ema_step_count" in metadata:
        ema.step_count = metadata["ema_step_count"]

    print(f"Checkpoint loaded: {ckpt_path}")
    return metadata


class TopKCheckpointManager:
    """Keep only the top-K checkpoints by a tracked metric.

    Automatically removes the worst checkpoint when a new one is saved and
    the total exceeds ``k``.

    Args:
        save_dir: Base directory for checkpoints.
        k: Maximum number of checkpoints to keep. Default: 5.
        mode: 'min' to keep lowest metric values, 'max' to keep highest.
            Default: 'min' (for loss).
    """

    def __init__(
        self,
        save_dir: str,
        k: int = 5,
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.mode = mode
        # List of (metric_value, checkpoint_path) sorted by metric
        self._checkpoints: List[Tuple[float, Path]] = []

    def should_save(self, metric: float) -> bool:
        """Check if a checkpoint with the given metric should be saved.

        Returns True if there's room or the metric is better than the worst
        saved checkpoint.
        """
        if len(self._checkpoints) < self.k:
            return True
        worst_metric = self._checkpoints[-1][0]
        if self.mode == "min":
            return metric < worst_metric
        else:
            return metric > worst_metric

    def save(
        self,
        metric: float,
        policy: nn.Module,
        ema,
        optimizer,
        epoch: int,
        step: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Conditionally save a checkpoint and prune if over capacity.

        Args:
            metric: The metric value (e.g. validation loss) associated with
                this checkpoint.
            policy, ema, optimizer, epoch, step: Forwarded to ``save_checkpoint``.
            extra_metadata: Optional extra metadata.

        Returns:
            Path to the saved checkpoint, or None if not saved.
        """
        if not self.should_save(metric):
            return None

        # Save
        meta = extra_metadata or {}
        meta["topk_metric"] = metric
        ckpt_path = save_checkpoint(
            policy,
            ema,
            optimizer,
            epoch,
            step,
            str(self.save_dir),
            extra_metadata=meta,
        )

        # Insert and sort
        self._checkpoints.append((metric, ckpt_path))
        reverse = self.mode == "max"
        self._checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Prune excess
        while len(self._checkpoints) > self.k:
            _, worst_path = self._checkpoints.pop()
            if worst_path.exists():
                shutil.rmtree(worst_path)
                print(f"Pruned checkpoint: {worst_path}")

        return ckpt_path

    @property
    def best_metric(self) -> Optional[float]:
        """Return the best metric value, or None if no checkpoints."""
        if not self._checkpoints:
            return None
        return self._checkpoints[0][0]

    @property
    def best_path(self) -> Optional[Path]:
        """Return the path to the best checkpoint, or None if no checkpoints."""
        if not self._checkpoints:
            return None
        return self._checkpoints[0][1]
