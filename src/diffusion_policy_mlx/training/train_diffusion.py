"""Training loop for Diffusion Policy MLX.

Upstream: diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py

Replaces Hydra-based workspace with a clean, self-contained training function
using MLX's functional gradient API (``mlx.nn.value_and_grad``).
"""

from __future__ import annotations

import time
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import mlx.utils
import numpy as np

from diffusion_policy_mlx.model.common.lr_scheduler import get_scheduler
from diffusion_policy_mlx.model.diffusion.ema_model import EMAModel
from diffusion_policy_mlx.training.checkpoint import (
    TopKCheckpointManager,
    save_checkpoint,
)
from diffusion_policy_mlx.training.collate import collate_batch  # noqa: F401
from diffusion_policy_mlx.training.train_config import TrainConfig


def train(config: TrainConfig) -> None:
    """Main training function for Diffusion Policy.

    Implements the full training loop:
        1. Load dataset and build normalizer
        2. Build policy, optimizer, EMA, LR scheduler
        3. Training loop with value_and_grad
        4. Periodic checkpointing and logging

    Args:
        config: Training configuration dataclass.
    """
    from diffusion_policy_mlx.compat.schedulers import DDPMScheduler
    from diffusion_policy_mlx.dataset.pusht_image_dataset import PushTImageDataset
    from diffusion_policy_mlx.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    # Set random seed
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    print(f"Training config: {config}")

    # 1. Load dataset
    dataset = PushTImageDataset(
        zarr_path=config.dataset_path,
        horizon=config.horizon,
        pad_before=config.n_obs_steps - 1,
        pad_after=config.n_action_steps - 1,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # 2. Build normalizer from dataset
    normalizer = dataset.get_normalizer(mode="limits")

    # 3. Build policy
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=config.horizon,
        n_action_steps=config.n_action_steps,
        n_obs_steps=config.n_obs_steps,
        num_inference_steps=config.num_inference_steps,
        down_dims=config.down_dims,
        kernel_size=config.kernel_size,
        n_groups=config.n_groups,
        diffusion_step_embed_dim=config.diffusion_step_embed_dim,
        crop_shape=config.crop_shape,
    )
    policy.set_normalizer(normalizer)
    print("Policy built.")

    # 4. Optimizer
    optimizer = mlx.optimizers.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )

    # 5. EMA
    ema = EMAModel(policy, power=config.ema_power)

    # 6. LR scheduler
    steps_per_epoch = max(1, len(dataset) // config.batch_size)
    total_steps = config.num_epochs * steps_per_epoch
    lr_sched = get_scheduler(config.lr_scheduler, optimizer, total_steps, config.lr_warmup_steps)

    # 7. Loss + grad function (MLX functional pattern)
    loss_and_grad_fn = nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))

    # 8. Checkpoint manager
    topk_manager = TopKCheckpointManager(save_dir=config.checkpoint_dir, k=config.top_k, mode="min")

    # 9. Training loop
    global_step = 0
    print(
        f"Starting training: {config.num_epochs} epochs, "
        f"~{steps_per_epoch} steps/epoch, {total_steps} total steps"
    )

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_losses: List[float] = []

        # Shuffle indices
        indices = np.random.permutation(len(dataset))

        for batch_start in range(0, len(indices), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(indices))
            batch_idx = indices[batch_start:batch_end]

            # Skip incomplete batches
            if len(batch_idx) < 2:
                continue

            batch = collate_batch([dataset[int(i)] for i in batch_idx])

            # Forward + backward (MLX functional paradigm)
            loss, grads = loss_and_grad_fn(policy, batch)

            # Update parameters
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)

            # EMA update
            ema.step(policy)

            # LR schedule step
            lr_sched.step()

            global_step += 1
            loss_val = float(loss)
            epoch_losses.append(loss_val)

            # Logging
            if global_step % config.log_every_n_steps == 0:
                lr_val = float(optimizer.learning_rate)
                print(
                    f"epoch {epoch:4d} | step {global_step:6d} | "
                    f"loss {loss_val:.6f} | lr {lr_val:.2e}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"Epoch {epoch:4d} done in {epoch_time:.1f}s | avg_loss {avg_loss:.6f}")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_checkpoint(
                policy,
                ema,
                optimizer,
                epoch,
                global_step,
                config.checkpoint_dir,
            )

        # Top-K checkpoint based on epoch loss
        topk_manager.save(
            metric=avg_loss,
            policy=policy,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )

    print(f"Training complete. Total steps: {global_step}")


def main() -> None:
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Diffusion Policy (MLX)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Start from config file or defaults
    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    # CLI overrides
    if args.dataset is not None:
        config.dataset_path = args.dataset
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None:
        config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
