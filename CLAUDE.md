# Diffusion Policy MLX — Claude Code Project Config

## Project Overview
Port of Diffusion Policy (RSS 2023, real-stanford/diffusion_policy, 5k+ GitHub stars) from PyTorch to Apple MLX.
Pure PyTorch project — no CUDA kernels. The port replaces `torch.nn` with `mlx.nn` and `torch.*` with `mx.*`.

Built by [AIFLOW LABS](https://aiflowlabs.io) / [RobotFlow Labs](https://robotflowlabs.com).

## Key Architecture

### Upstream Structure (in `repositories/diffusion-policy-upstream/`)
```
diffusion_policy/
  model/diffusion/          # ConditionalUnet1D, TransformerForDiffusion, Conv1dBlock, EMA
  model/vision/             # ResNet backbone (torchvision), MultiImageObsEncoder, CropRandomizer
  model/common/             # LinearNormalizer, LR schedulers, ModuleAttrMixin
  model/bet/                # Behavior Transformer (secondary priority)
  policy/                   # DiffusionUnetHybridImagePolicy (main target), Transformer variants
  dataset/                  # PushT, Kitchen dataset loaders (zarr/HDF5)
  workspace/                # Hydra-based training orchestration
  env/                      # PushT, Kitchen gym environments
  common/                   # pytorch_util, checkpoint_util, json_logger
```

### MLX Port Structure (our code)
```
src/diffusion_policy_mlx/
  compat/                   # torch→mlx translation layer (THE foundation)
    tensor_ops.py           # mx.array wrappers matching torch.* signatures
    nn_modules.py           # Module base class with .to()/.train()/.eval() no-ops
    nn_layers.py            # Conv1d, ConvTranspose1d, GroupNorm, Linear, Mish, etc.
    functional.py           # F.mish, F.silu, padding ops
    vision.py               # ResNet18/34/50 in MLX, NCHW↔NHWC handling
    schedulers.py           # DDPMScheduler, DDIMScheduler (replaces diffusers)
    einops_mlx.py           # rearrange/reduce for common patterns
  model/
    diffusion/              # ConditionalUnet1D, Conv1dBlock, SinusoidalPosEmb, EMA
    vision/                 # MultiImageObsEncoder, model_getter
    common/                 # LinearNormalizer, ModuleAttrMixin
  policy/                   # DiffusionUnetHybridImagePolicy
  training/                 # MLX-native training loop (replaces Hydra workspace)
  dataset/                  # Data loading returning mx.array
tests/                      # Cross-framework validation tests
```

## Critical Design Rules
1. **NEVER modify upstream files** in `repositories/` — read-only reference
2. **Mirror upstream structure** — same class names, method signatures, tensor shapes
3. **Use the compat layer** — all torch→mlx translation goes through `compat/`
4. **Channel format** — upstream uses NCHW (PyTorch); MLX Conv2d uses NHWC. Handle in compat/vision.py
5. **Conv1d format** — upstream uses NCL (batch, channels, length); MLX Conv1d uses NLC. Handle in compat/nn_layers.py
6. **Cross-framework tests** — compare MLX output vs PyTorch reference with tolerance
7. **PRD-driven** — write PRD in `prds/` before implementing each component

## Build Order (PRD Sequence)
1. **PRD-00: Dev Environment** — uv, mlx, project scaffolding
2. **PRD-01: Compat Foundation** — tensor_ops, nn_modules, nn_layers, functional
3. **PRD-02: Vision Encoder** — ResNet18/34 in MLX, MultiImageObsEncoder
4. **PRD-03: UNet Denoiser** — Conv1dBlock, ConditionalUnet1D, SinusoidalPosEmb
5. **PRD-04: DDPM/DDIM Scheduler** — noise schedule, forward/reverse process
6. **PRD-05: Policy Assembly** — DiffusionUnetHybridImagePolicy, LinearNormalizer
7. **PRD-06: Training Loop** — MLX-native train loop, EMA, checkpointing
8. **PRD-07: PushT Dataset** — zarr loading, data pipeline
9. **PRD-08: Evaluation** — PushT env, inference loop, benchmarks

## Dev Commands
```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
ruff format src/
```

## Key torch→mlx Mappings
| PyTorch | MLX |
|---------|-----|
| `torch.Tensor` | `mx.array` |
| `nn.Conv1d` | `mx.nn.Conv1d` (NLC not NCL — transpose needed) |
| `nn.ConvTranspose1d` | Custom impl or `mx.nn.ConvTranspose1d` |
| `nn.GroupNorm` | `mx.nn.GroupNorm` |
| `nn.Linear` | `mx.nn.Linear` |
| `nn.Mish` | `mx.nn.Mish` or `mx.nn.mish` |
| `nn.MultiheadAttention` | `mx.nn.MultiHeadAttention` |
| `nn.ModuleList` | Python list (MLX traces dynamically) |
| `nn.Sequential` | `mx.nn.Sequential` or manual chain |
| `F.interpolate` | Custom using `mx.reshape` / repeat |
| `DDPMScheduler` (diffusers) | Custom MLX implementation |
| `torchvision.models.resnet18` | Custom MLX ResNet |
| `einops.rearrange` | `mx.reshape` + `mx.transpose` |

## MLX Gotchas
- No int64 on GPU — cast to int32 for index ops
- `mx.eval()` is MANDATORY after optimizer step (prevents lazy graph buildup)
- No boolean indexing — use `mx.where` or `mx.take`
- Conv weights: PyTorch OIHW → MLX OHWI (Conv2d), PyTorch OIL → MLX OLI (Conv1d)
- `mlx.__version__` doesn't exist — use `importlib.metadata.version("mlx")`
- No `.to(device)` — MLX handles device placement automatically
- No `torch.no_grad()` — MLX uses functional transforms, no grad context needed

## Conventions
- Package manager: `uv`
- Linter/formatter: `ruff`
- Use `rg` (ripgrep) instead of `grep` in Bash

# currentDate
Today's date is 2026-03-15.
