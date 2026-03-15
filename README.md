# Diffusion Policy MLX

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-%E2%89%A50.22-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-308%20passing-brightgreen.svg)](#testing)

The first native Apple Silicon port of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (Chi et al., RSS 2023 Best Paper) to [Apple MLX](https://github.com/ml-explore/mlx). Trains and runs visuomotor diffusion policies entirely on M-series hardware -- no CUDA, no cloud, no CPU-GPU transfer overhead. Leverages unified memory for real-time robot control at sub-100ms inference latency.

## Key Features

- **Native Apple Silicon** -- runs on M1/M2/M3/M4 via Metal GPU, no CUDA required
- **Sub-100ms inference** -- real-time action generation for robot control loops
- **Unified memory** -- zero-copy CPU/GPU data sharing eliminates transfer bottlenecks
- **Full training pipeline** -- train from scratch on PushT or custom datasets
- **Weight conversion** -- load pretrained PyTorch checkpoints directly
- **DDPM + DDIM scheduling** -- standard and accelerated diffusion sampling
- **308 tests passing** -- cross-framework validation against PyTorch reference
- **Drop-in architecture** -- mirrors upstream class names and interfaces

## Architecture

```
Observations                          Actions
     |                                   ^
     v                                   |
 [RGB Image] -----> [ResNet18] -----> [features]     [action trajectory]
                         |                ^                  ^
                         v                |                  |
                    [MultiImage     [global cond]     [x_0 predicted]
                     ObsEncoder]         |                  |
                         |               v                  |
                         +-------> [ConditionalUnet1D] -----+
                                        ^
                                        |
                                   [t, noise]
                                        |
                                  [DDPM Scheduler]
                                        |
                                   x_T ~ N(0,I)

Forward (training):
  1. Encode observations through ResNet vision backbone
  2. Sample random timestep t, add noise to ground-truth actions
  3. UNet predicts noise conditioned on (timestep, obs features)
  4. MSE loss between predicted and actual noise

Reverse (inference):
  1. Encode observations through ResNet vision backbone
  2. Initialize x_T ~ N(0, I) of shape (B, horizon, action_dim)
  3. For t in T..0: x_{t-1} = UNet_denoise(x_t, t, obs_features)
  4. Return first n_action_steps of denoised trajectory x_0
```

## Quick Start

```bash
# 1. Install
git clone https://github.com/AIFLOW-LABS/diffusion-policy-mlx.git
cd diffusion-policy-mlx
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Download PushT dataset
python scripts/download_pusht.py

# 3. Train
python -m diffusion_policy_mlx.training.train_diffusion \
    --config configs/pusht_diffusion_policy_cnn.yaml

# 4. Evaluate
python scripts/eval_pusht.py \
    --checkpoint checkpoints/latest.safetensors
```

### Convert Pretrained PyTorch Weights

```bash
python scripts/convert_weights.py \
    --checkpoint path/to/pytorch_checkpoint.ckpt \
    --output checkpoints/pusht_mlx.safetensors
```

## Performance

Benchmarks on Apple M-series hardware (batch size 1, 100 diffusion steps):

| Metric | Apple M2 Pro | Apple M3 Max | Notes |
|--------|-------------|-------------|-------|
| Inference latency | ~85 ms | ~60 ms | Single action trajectory |
| Training throughput | ~12 batch/s | ~18 batch/s | batch_size=64 |
| Peak memory | ~2.1 GB | ~2.1 GB | Unified memory |
| DDIM (10 steps) | ~12 ms | ~8 ms | Accelerated sampling |

> Unified memory eliminates the CPU-GPU transfer overhead that dominates PyTorch+CUDA latency on small batch sizes typical of real-time robot control.

## Project Structure

```
diffusion-policy-mlx/
  src/diffusion_policy_mlx/
    compat/                     # PyTorch -> MLX translation layer
      tensor_ops.py             #   torch.* function equivalents
      nn_modules.py             #   Module base class with .train()/.eval()
      nn_layers.py              #   Conv1d, GroupNorm, Linear, Mish, etc.
      functional.py             #   F.mish, F.silu, padding ops
      vision.py                 #   ResNet18/34/50 in MLX (NCHW <-> NHWC)
      schedulers.py             #   DDPMScheduler, DDIMScheduler
      einops_mlx.py             #   rearrange for common patterns
    model/
      diffusion/
        conditional_unet1d.py   #   1D UNet denoiser with FiLM conditioning
        conv1d_components.py    #   Downsample1d, Upsample1d, Conv1dBlock
        positional_embedding.py #   Sinusoidal timestep encoding
        ema_model.py            #   Exponential moving average
        mask_generator.py       #   Action/observation masking
      vision/
        model_getter.py         #   get_resnet() factory
        multi_image_obs_encoder.py  # Multi-camera observation encoder
        crop_randomizer.py      #   Spatial crop augmentation
      common/
        normalizer.py           #   LinearNormalizer (fit/normalize/unnormalize)
        lr_scheduler.py         #   Cosine and linear warmup schedulers
    policy/
      base_image_policy.py      # Abstract base with predict_action()
      diffusion_unet_hybrid_image_policy.py  # Main policy (vision + UNet)
    training/
      train_diffusion.py        # MLX-native training loop
      train_config.py           # TrainConfig dataclass with YAML support
      checkpoint.py             # TopK checkpoint management
      collate.py                # Batch collation for mx.array
    dataset/
      base_dataset.py           # Base dataset class
      pusht_image_dataset.py    # PushT zarr-backed image dataset
      replay_buffer.py          # Replay buffer utilities
  tests/                        # 308 tests (cross-framework validation)
  scripts/
    convert_weights.py          # PyTorch checkpoint -> MLX safetensors
    download_pusht.py           # Download PushT dataset
    eval_pusht.py               # PushT evaluation loop
    benchmark.py                # Inference latency benchmarks
  configs/                      # Training configuration files
  prds/                         # PRD documents (build plan)
  repositories/
    diffusion-policy-upstream/  # Read-only upstream reference
```

## Development

### Prerequisites

- macOS with Apple Silicon (M1 or later)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific component tests
pytest tests/test_compat_tensor_ops.py -v
pytest tests/test_unet.py -v
pytest tests/test_policy.py -v

# Run benchmarks
pytest tests/test_benchmark.py -v
```

### Linting

```bash
ruff check src/
ruff format src/
```

## Upstream Sync

This port tracks [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy). The upstream repository is cloned as a read-only reference in `repositories/diffusion-policy-upstream/`.

To update when upstream changes:

```bash
# 1. Pull latest upstream
cd repositories/diffusion-policy-upstream
git fetch && git pull

# 2. Check what changed
git diff HEAD~1 --name-only

# 3. If model/* changed:
#    - Update mirrored classes in src/diffusion_policy_mlx/
#    - Add new torch.* calls to compat/
#    - Update convert_weights.py for new weight shapes
#    - Re-run cross-framework tests

# 4. Update UPSTREAM_VERSION.md with the new commit hash
```

The port stays sync-friendly because all PyTorch-to-MLX translation is isolated in the `compat/` layer, and no upstream files are ever modified.

## Citation

If you use this work, please cite the original Diffusion Policy paper:

```bibtex
@inproceedings{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric
          and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2023}
}
```

## License

MIT

## Built By

[AIFLOW LABS](https://aiflowlabs.io) | [RobotFlow Labs](https://robotflowlabs.com)

Part of the AIFLOW LABS Apple Silicon robotics ML stack, alongside [pointelligence-mlx](https://github.com/AIFLOW-LABS/pointelligence-mlx) (3D perception) and LeRobot-mlx (policy framework).
