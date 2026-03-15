# PRD-06: Training Loop

**Status:** Complete
**Depends on:** PRD-05 (Policy Assembly)
**Blocks:** PRD-08 (Evaluation)

---

## Objective

Build the MLX-native training loop: EMA model tracking, LR scheduling, checkpointing, and the training script. Replaces upstream's Hydra workspace with a clean, modern Python training loop using MLX's functional gradient API.

---

## Upstream Reference

| File | What |
|------|------|
| `workspace/train_diffusion_unet_hybrid_workspace.py` | Hydra training orchestration |
| `model/diffusion/ema_model.py` | EMAModel |
| `model/common/lr_scheduler.py` | get_scheduler (wraps diffusers) |
| `common/checkpoint_util.py` | TopKCheckpointManager |
| `common/json_logger.py` | JsonLogger |

---

## Design Decision: No Hydra

Upstream uses Hydra + OmegaConf for config management. We replace this with:
- **YAML config files** parsed by `pyyaml` — simple, no magic
- **Dataclass configs** for type safety
- **CLI args** for overrides

This eliminates the Hydra dependency and makes the training loop self-contained.

---

## Deliverables

### 1. `model/diffusion/ema_model.py` — EMA

```python
class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights, updated each step:
        ema_param = decay * ema_param + (1-decay) * new_param

    Decay schedule ramps from min_value to max_value over training:
        decay = 1 - (1 + step/inv_gamma)^(-power)

    Upstream: not an nn.Module — manages a separate model copy.
    """

    def __init__(self, model, update_after_step=0,
                 inv_gamma=1.0, power=2/3,
                 min_value=0.0, max_value=0.9999):
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.step_count = 0

        # Deep copy model parameters for EMA shadow
        import copy
        self.averaged_params = copy.deepcopy(
            dict(mlx.utils.tree_flatten(model.parameters()))
        )

    def get_decay(self, step):
        """Compute decay factor for current step."""
        if step < self.update_after_step:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    def step(self, model):
        """Update EMA parameters with current model."""
        self.step_count += 1
        decay = self.get_decay(self.step_count)

        model_params = dict(mlx.utils.tree_flatten(model.parameters()))
        for key in self.averaged_params:
            if key in model_params:
                self.averaged_params[key] = (
                    decay * self.averaged_params[key]
                    + (1 - decay) * model_params[key]
                )

        mx.eval(*self.averaged_params.values())

    def copy_to(self, model):
        """Copy EMA parameters into a model for evaluation."""
        model.load_weights(list(self.averaged_params.items()))

    def state_dict(self):
        return {
            'averaged_params': self.averaged_params,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state):
        self.averaged_params = state['averaged_params']
        self.step_count = state['step_count']
```

### 2. `model/common/lr_scheduler.py` — Learning Rate Scheduling

```python
class CosineAnnealingLR:
    """Cosine annealing with optional warmup.

    LR schedule:
      warmup: linear from 0 to base_lr over warmup_steps
      cosine: base_lr → min_lr following cosine curve
    """
    def __init__(self, optimizer, num_training_steps, num_warmup_steps=0,
                 min_lr=0.0):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        self.optimizer.learning_rate = lr

    def _compute_lr(self):
        step = self.current_step
        if step < self.num_warmup_steps:
            return self.base_lr * step / max(1, self.num_warmup_steps)
        progress = (step - self.num_warmup_steps) / max(
            1, self.num_training_steps - self.num_warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress))


def get_scheduler(name, optimizer, num_training_steps, num_warmup_steps=0):
    """Factory for LR schedulers.

    Args:
        name: 'cosine', 'linear', 'constant', 'constant_with_warmup'
    """
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, num_training_steps, num_warmup_steps)
    elif name == 'constant':
        return ConstantLR(optimizer)
    elif name == 'constant_with_warmup':
        return ConstantWithWarmupLR(optimizer, num_warmup_steps)
    elif name == 'linear':
        return LinearLR(optimizer, num_training_steps, num_warmup_steps)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
```

### 3. `training/train_diffusion.py` — Training Loop

```python
@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    dataset_path: str = "data/pusht_image.zarr"
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8

    # Model
    down_dims: tuple = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 256
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    crop_shape: tuple = (76, 76)

    # Training
    batch_size: int = 64
    num_epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    ema_power: float = 0.75

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 50
    top_k: int = 5

    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "diffusion-policy-mlx"


def train(config: TrainConfig):
    """Main training function."""

    # 1. Load dataset
    dataset = PushTImageDataset(
        zarr_path=config.dataset_path,
        horizon=config.horizon,
        pad_before=config.n_obs_steps - 1,
        pad_after=config.n_action_steps - 1,
    )

    # 2. Build normalizer from dataset
    normalizer = dataset.get_normalizer(mode='limits')

    # 3. Build policy
    shape_meta = {
        'obs': {
            'image': {'shape': (3, 96, 96), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)}
    }
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
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

    # 4. Optimizer
    optimizer = mlx.optimizers.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )

    # 5. EMA
    ema = EMAModel(policy, power=config.ema_power)

    # 6. LR scheduler
    total_steps = config.num_epochs * len(dataset) // config.batch_size
    lr_sched = get_scheduler(
        config.lr_scheduler, optimizer, total_steps, config.lr_warmup_steps)

    # 7. Loss + grad function (MLX functional pattern)
    loss_and_grad_fn = mlx.nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))

    # 8. Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        # Shuffle and batch
        indices = np.random.permutation(len(dataset))
        for batch_start in range(0, len(indices), config.batch_size):
            batch_idx = indices[batch_start:batch_start + config.batch_size]
            batch = collate_batch([dataset[i] for i in batch_idx])

            # Forward + backward
            loss, grads = loss_and_grad_fn(policy, batch)

            # Update
            optimizer.update(policy, grads)
            mx.eval(policy.parameters(), optimizer.state)

            # EMA
            ema.step(policy)

            # LR schedule
            lr_sched.step()

            global_step += 1

            # Logging
            if global_step % config.log_every_n_steps == 0:
                print(f"step {global_step} | loss {float(loss):.6f} | "
                      f"lr {optimizer.learning_rate:.2e}")

        # Checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_checkpoint(policy, ema, optimizer, epoch, global_step,
                           config.checkpoint_dir)


def collate_batch(samples):
    """Collate list of dataset samples into batched mx.arrays."""
    batch = {}
    for key in samples[0]:
        if isinstance(samples[0][key], dict):
            batch[key] = {}
            for subkey in samples[0][key]:
                batch[key][subkey] = mx.array(
                    np.stack([s[key][subkey] for s in samples]))
        else:
            batch[key] = mx.array(np.stack([s[key] for s in samples]))
    return batch


def save_checkpoint(policy, ema, optimizer, epoch, step, checkpoint_dir):
    """Save model checkpoint."""
    path = Path(checkpoint_dir) / f"epoch_{epoch:04d}_step_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)

    # Model weights
    mx.savez(str(path / "model.npz"), **dict(mlx.utils.tree_flatten(policy.parameters())))

    # EMA weights
    mx.savez(str(path / "ema.npz"), **ema.averaged_params)

    # Training state
    import json
    with open(path / "train_state.json", "w") as f:
        json.dump({'epoch': epoch, 'step': step}, f)

    print(f"Checkpoint saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/pusht_image.zarr")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    config = TrainConfig(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train(config)
```

---

## Tests

### `tests/test_training.py`

```python
def test_ema_decay_schedule():
    """EMA decay ramps from 0 to ~0.999."""
    model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
    ema = EMAModel(model)
    assert ema.get_decay(0) == 0.0
    assert ema.get_decay(10000) > 0.99
    assert ema.get_decay(10000) < 1.0

def test_ema_step_updates():
    """EMA params change after step."""
    model = ConditionalUnet1D(input_dim=2, down_dims=[16, 32])
    ema = EMAModel(model, update_after_step=0, min_value=0.5)
    before = {k: mx.array(v) for k, v in ema.averaged_params.items()}

    # Modify model
    params = model.parameters()
    # ... perturb weights

    ema.step(model)
    # Check params changed
    ...

def test_cosine_lr_schedule():
    optimizer = mlx.optimizers.Adam(learning_rate=1e-3)
    sched = CosineAnnealingLR(optimizer, num_training_steps=1000, num_warmup_steps=100)

    # Warmup: should increase
    sched.step()
    lr1 = optimizer.learning_rate
    for _ in range(99):
        sched.step()
    lr100 = optimizer.learning_rate
    assert lr100 > lr1

    # Cosine: should decrease
    for _ in range(900):
        sched.step()
    lr1000 = optimizer.learning_rate
    assert lr1000 < lr100

def test_training_step_loss_decreases():
    """Loss should decrease over multiple training steps on fixed data."""
    # Use tiny model and synthetic data
    model = ... # small policy
    optimizer = mlx.optimizers.Adam(learning_rate=1e-3)
    loss_fn = mlx.nn.value_and_grad(model, lambda m, b: m.compute_loss(b))

    batch = ... # fixed synthetic batch

    losses = []
    for _ in range(100):
        loss, grads = loss_fn(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(float(loss))

    # Loss should trend downward
    assert losses[-1] < losses[0] * 0.5  # at least 50% reduction
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | EMA decay schedule matches upstream formula | exact |
| 2 | Cosine LR schedule with warmup works correctly | visual + numeric check |
| 3 | Training step: loss decreases over 100 steps on synthetic data | losses[-1] < losses[0] * 0.5 |
| 4 | `mx.eval()` called after every optimizer step | code review |
| 5 | Checkpoint save/load round-trip preserves weights | atol=1e-7 |
| 6 | Training loop runs without OOM on 8GB M-series (small config) | no crash |

---

## Upstream Sync Notes

**Files to watch:**
- `workspace/train_diffusion_unet_hybrid_workspace.py` — training hyperparameters, evaluation schedule
- `model/diffusion/ema_model.py` — EMA formula, decay schedule
- `model/common/lr_scheduler.py` — scheduler types (currently wraps diffusers)

**Key difference from upstream:** We use `mlx.nn.value_and_grad()` instead of `loss.backward()` + `optimizer.step()`. This is the MLX functional paradigm and is non-negotiable.
