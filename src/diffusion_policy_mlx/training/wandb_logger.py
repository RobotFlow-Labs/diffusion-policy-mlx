"""Optional Weights & Biases integration for training.

Provides :class:`WandbLogger` — a thin wrapper around the ``wandb`` Python
package that degrades gracefully when wandb is not installed or is disabled.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class WandbLogger:
    """Optional Weights & Biases logger.

    Wraps ``wandb.init``, ``wandb.log``, and ``wandb.finish`` with
    graceful degradation:

    - If ``wandb`` is not installed, all methods are silent no-ops.
    - If ``enabled=False``, no run is created.

    Example::

        logger = WandbLogger(project="my-project", config={"lr": 1e-4})
        logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
        logger.finish()

    Args:
        project: W&B project name. Default: ``"diffusion-policy-mlx"``.
        config: Optional config dict to attach to the W&B run.
        enabled: If False, wandb is not initialized even if installed.
            Default: True.
        **kwargs: Additional keyword arguments forwarded to ``wandb.init``.
    """

    def __init__(
        self,
        project: str = "diffusion-policy-mlx",
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        **kwargs: Any,
    ):
        self._wandb = None
        self._enabled = enabled

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, config=config, **kwargs)
        except ImportError:
            # wandb not installed — degrade silently
            self._wandb = None
        except Exception as exc:
            # wandb installed but init failed (e.g., no API key)
            import logging

            logging.getLogger(__name__).warning("wandb init failed, logging disabled: %s", exc)
            self._wandb = None

    @property
    def is_active(self) -> bool:
        """Return True if wandb is initialized and logging."""
        return self._wandb is not None and self._wandb.run is not None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Silent no-op if wandb is not available or disabled.

        Args:
            data: Dictionary of metric names to values.
            step: Optional global step number.
        """
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.log(data, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Update the W&B run config.

        Args:
            config: Dictionary of config values to add/update.
        """
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.config.update(config)

    def finish(self) -> None:
        """Finish the W&B run.

        Safe to call multiple times or when wandb is not active.
        """
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()
