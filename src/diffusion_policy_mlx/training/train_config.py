"""Training configuration dataclass with YAML serialization.

Provides ``TrainConfig`` — a dataclass with all training hyperparameters and
defaults matching the upstream PushT configuration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union


@dataclass
class TrainConfig:
    """Training configuration for Diffusion Policy MLX.

    Defaults match the upstream PushT image diffusion policy configuration.
    """

    # -- Data ------------------------------------------------------------------
    dataset_path: str = "data/pusht_image.zarr"
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8

    # -- Model -----------------------------------------------------------------
    down_dims: Tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 256
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    crop_shape: Tuple[int, ...] = (76, 76)

    # -- Training --------------------------------------------------------------
    batch_size: int = 64
    num_epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    ema_power: float = 0.75
    seed: int = 42

    # -- Checkpointing ---------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 50
    top_k: int = 5

    # -- Logging ---------------------------------------------------------------
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "diffusion-policy-mlx"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        """Load configuration from a YAML file.

        Only keys matching dataclass fields are used; unknown keys are ignored.
        Tuple fields stored as lists in YAML are converted back to tuples.
        """
        import yaml  # lazy import to avoid hard dependency at module level

        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        # Filter to only known fields
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in raw.items():
            if k in known_fields:
                filtered[k] = v

        # Convert lists to tuples for tuple-typed fields
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        for k, v in filtered.items():
            if isinstance(v, list) and "Tuple" in str(field_types.get(k, "")):
                filtered[k] = tuple(v)

        return cls(**filtered)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        import yaml  # lazy import

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict, tuples to lists for YAML
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)

        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return dataclasses.asdict(self)
