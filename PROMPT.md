# Diffusion Policy MLX — Master Build Prompt

> Copy this entire file and paste it as a prompt when working in the `/Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx` directory.

---

## Mission

Port **Diffusion Policy** (Chi et al., RSS 2023, `real-stanford/diffusion_policy`, 5k+ GitHub stars) from **PyTorch** to **Apple MLX**, creating the first native Apple Silicon implementation of the most widely adopted robot visuomotor policy framework.

**Built by [AIFLOW LABS](https://aiflowlabs.io) / [RobotFlow Labs](https://robotflowlabs.com)**

---

## Context and Prior Art

We have successfully ported two major frameworks to MLX:

1. **PointCNN++ (CVPR 2026)** — PyTorch + Triton + CUDA to MLX. 5 custom kernels rewritten as MLX ops. 344 tests passing. Repo: `pointelligence-mlx`.
2. **LeRobot v0.5.1** — HuggingFace's robotics framework. Pure PyTorch to MLX. 19-PRD build plan. Repo: `LeRobot-mlx`.

The proven methodology:

1. PRD-driven development (one PRD per component, strict build order)
2. Compat/bridge pattern (keep upstream structure, replace torch ops with mlx ops)
3. Cross-framework tests (compare MLX output vs PyTorch reference within tolerance)
4. Bottom-up build order (primitives -> layers -> models -> policy -> training)

**Key characteristic of this port**: Diffusion Policy is pure PyTorch — no CUDA kernels, no custom C++ extensions. The port is about systematically replacing `torch.*` / `torch.nn.*` / `torchvision.*` / `diffusers.*` calls with `mlx.*` / `mlx.nn.*` equivalents.

---

## Reference Repositories

The upstream is already cloned:

```
repositories/diffusion-policy-upstream/    # read-only, never modify
```

Source: https://github.com/real-stanford/diffusion_policy

---

## Upstream Architecture Map

### Core Neural Network Components

```
diffusion_policy/
  model/
    diffusion/
      conditional_unet1d.py       # THE denoiser — ConditionalUnet1D, ConditionalResidualBlock1D
                                  #   Conv1d residual blocks + FiLM conditioning + sinusoidal time embed
                                  #   Down/mid/up architecture with skip connections
      conv1d_components.py        # Downsample1d (strided Conv1d), Upsample1d (ConvTranspose1d), Conv1dBlock (Conv1d+GroupNorm+Mish)
      positional_embedding.py     # SinusoidalPosEmb — timestep encoding
      transformer_for_diffusion.py # TransformerForDiffusion — alternative denoiser (TransformerEncoder-based)
      ema_model.py                # EMAModel — exponential moving average of model weights
      mask_generator.py           # LowdimMaskGenerator — action/obs masking for inpainting

    vision/
      model_getter.py             # get_resnet() — loads torchvision ResNet18/34/50 with optional R3M weights
      multi_image_obs_encoder.py  # MultiImageObsEncoder — handles multiple camera inputs
                                  #   Per-camera ResNet backbone + optional crop augmentation
                                  #   Shared or separate backbones, optional GroupNorm replacement
      crop_randomizer.py          # CropRandomizer — spatial crop augmentation for visual inputs

    common/
      normalizer.py               # LinearNormalizer — fits min/max or gaussian stats, normalizes/unnormalizes
      module_attr_mixin.py        # ModuleAttrMixin — adds .device and .dtype property to nn.Module
      lr_scheduler.py             # get_scheduler() — cosine, linear warmup LR schedulers
      dict_of_tensor_mixin.py     # DictOfTensorMixin — dict-like module for storing tensor params
      rotation_transformer.py     # Rotation representations (axis-angle, 6D, quaternion)
      shape_util.py               # Shape manipulation utilities
      tensor_util.py              # Tensor manipulation utilities

    bet/                          # Behavior Transformer (secondary priority)
      action_ae/                  #   Action autoencoder (VQ-VAE for action discretization)
      latent_generators/          #   MinGPT-based latent sequence generator
      libraries/                  #   MinGPT implementation
      utils.py                    #   BeT utilities
```

### Policy Layer

```
  policy/
    base_image_policy.py                          # BaseImagePolicy — abstract base with predict_action()
    base_lowdim_policy.py                         # BaseLowdimPolicy — low-dimensional observation base
    diffusion_unet_hybrid_image_policy.py         # PRIMARY TARGET — UNet + ResNet vision encoder
                                                  #   Uses DDPMScheduler from diffusers
                                                  #   obs_as_global_cond: vision features as global conditioning
                                                  #   Handles multi-camera RGB + low-dim observations
    diffusion_unet_image_policy.py                # UNet with image-only obs (no hybrid)
    diffusion_unet_lowdim_policy.py               # UNet with low-dim obs only (no vision)
    diffusion_transformer_hybrid_image_policy.py  # Transformer denoiser variant
    diffusion_transformer_lowdim_policy.py        # Transformer with low-dim obs
    diffusion_unet_video_policy.py                # Video-conditioned variant
    bet_lowdim_policy.py                          # Behavior Transformer policy
    ibc_dfo_hybrid_image_policy.py                # Implicit BC with DFO
    ibc_dfo_lowdim_policy.py                      # IBC low-dim variant
    robomimic_image_policy.py                     # RoboMimic wrapper
    robomimic_lowdim_policy.py                    # RoboMimic low-dim wrapper
```

### Training and Data

```
  workspace/
    base_workspace.py                             # BaseWorkspace — Hydra-based training orchestration
    train_diffusion_unet_hybrid_workspace.py      # PRIMARY — trains UNet hybrid policy
                                                  #   DataLoader, optimizer, EMA, checkpointing, eval
    train_diffusion_unet_lowdim_workspace.py      # Low-dim UNet training
    train_diffusion_transformer_*_workspace.py    # Transformer variant training
    train_bet_lowdim_workspace.py                 # BeT training

  dataset/
    base_dataset.py                               # BaseImageDataset — zarr-backed
    pusht_image_dataset.py                        # PushT with images — THE demo dataset
    pusht_dataset.py                              # PushT low-dim
    kitchen_lowdim_dataset.py                     # Franka Kitchen
    robomimic_replay_image_dataset.py             # RoboMimic replay buffer

  env/
    pusht/                                        # PushT gym environment
    kitchen/                                      # Franka Kitchen environment
    block_pushing/                                # Block pushing environment

  common/
    pytorch_util.py                               # dict_apply, replace_submodules, optimizer_to
    checkpoint_util.py                            # TopKCheckpointManager
    json_logger.py                                # Training metrics logging
    robomimic_config_util.py                      # RoboMimic config helpers
```

### External Dependencies to Replace

| Upstream Dependency | What It Provides | MLX Replacement |
|---------------------|-----------------|-----------------|
| `torch` | Tensors, autograd, nn | `mlx` core |
| `torch.nn` | Conv1d, Linear, GroupNorm, Mish, ModuleList | `mlx.nn` + compat layer |
| `torch.nn.functional` | Activation functions, padding | `mlx.nn` functional |
| `torchvision.models` | ResNet18/34/50 backbones | Custom MLX ResNet |
| `diffusers.DDPMScheduler` | DDPM noise schedule, step function | Custom MLX scheduler |
| `diffusers.DDIMScheduler` | DDIM accelerated sampling | Custom MLX scheduler |
| `einops` | Tensor rearrangement | `mx.reshape` + `mx.transpose` |
| `hydra` | Config management | Simple dataclass configs or YAML |
| `wandb` | Experiment tracking | Keep as-is (numpy-compatible) |
| `robomimic` | Vision encoder backbone (used in hybrid policy) | Replace with direct ResNet |

---

## Port Strategy

### Phase 1: Foundation (Compat Layer)

Build the `compat/` translation layer that makes MLX look like PyTorch from the caller's perspective. This is the foundation everything else builds on.

**Files:**
- `compat/tensor_ops.py` — `zeros`, `ones`, `cat`, `stack`, `where`, `arange`, `linspace`, `exp`, `sin`, `cos`
- `compat/nn_modules.py` — `Module` base class with `.parameters()`, `.train()`, `.eval()`, `.state_dict()`
- `compat/nn_layers.py` — `Conv1d`, `ConvTranspose1d`, `GroupNorm`, `Linear`, `Mish`, `SiLU`, `Identity`, `ModuleList`, `Sequential`
- `compat/functional.py` — `mish`, `silu`, `pad` (reflect, replicate, constant), `interpolate`
- `compat/einops_mlx.py` — `rearrange('batch t -> batch t 1')` and other patterns used in upstream

**Critical: Conv1d Layout Translation**
- PyTorch Conv1d: input `(N, C, L)`, weight `(C_out, C_in, K)`
- MLX Conv1d: input `(N, L, C)`, weight `(C_out, K, C_in)`
- The compat Conv1d must transpose inputs/outputs to maintain upstream NCL convention externally while using NLC internally.

### Phase 2: Vision Encoder

Port the ResNet visual backbone and observation encoder.

**Files:**
- `compat/vision.py` — ResNet18/34/50 from scratch in MLX (BasicBlock, Bottleneck, full ResNet)
  - Weight conversion from torchvision format (OIHW -> OHWI for Conv2d)
  - ImageNet normalization
  - NCHW input -> NHWC internal -> NCHW output (maintain upstream convention)
- `model/vision/model_getter.py` — `get_resnet()` returning MLX ResNet
- `model/vision/multi_image_obs_encoder.py` — MultiImageObsEncoder handling multiple cameras
- `model/vision/crop_randomizer.py` — CropRandomizer for data augmentation

### Phase 3: UNet Denoiser

Port the core 1D UNet that denoises action trajectories.

**Files:**
- `model/diffusion/positional_embedding.py` — SinusoidalPosEmb (simple, port first)
- `model/diffusion/conv1d_components.py` — Downsample1d, Upsample1d, Conv1dBlock
- `model/diffusion/conditional_unet1d.py` — ConditionalUnet1D with:
  - Sinusoidal timestep embedding
  - FiLM conditioning (scale + bias modulation)
  - Down blocks -> mid block -> up blocks with skip connections
  - Global conditioning concatenation

**Architecture of ConditionalUnet1D:**
```
Input: (B, action_dim, horizon) + conditioning (B, cond_dim)
  |
  v
[SinusoidalPosEmb] -> timestep embedding
  |
  v
[MLP] -> diffusion_step_encoder (Linear -> Mish -> Linear)
  |
  v
[Concat global_cond if obs_as_global_cond]
  |
  v
[Conv1d] -> input projection
  |
  v
[DownBlocks] -> ConditionalResidualBlock1D x2 + Downsample1d (per level)
  |  (save skip connections)
  v
[MidBlocks] -> ConditionalResidualBlock1D x2
  |
  v
[UpBlocks] -> ConditionalResidualBlock1D x2 + Upsample1d (per level)
  |  (concat skip connections)
  v
[Conv1d] -> final projection -> (B, action_dim, horizon)
```

### Phase 4: DDPM/DDIM Scheduler

Implement the diffusion noise schedulers (replacing HuggingFace `diffusers`).

**Files:**
- `compat/schedulers.py` — DDPMScheduler and DDIMScheduler
  - Linear/cosine/squaredcos beta schedule
  - `add_noise(x_0, noise, timesteps)` — forward diffusion
  - `step(model_output, timestep, sample)` — reverse denoising step
  - `set_timesteps(num_inference_steps)` — configure inference schedule
  - Prediction types: `epsilon` (predict noise), `sample` (predict clean), `v_prediction`
  - Clip sample, thresholding options

### Phase 5: Policy Assembly

Assemble the full policy that ties vision encoder + UNet + scheduler together.

**Files:**
- `model/common/normalizer.py` — LinearNormalizer (fits data stats, normalizes obs/actions)
- `model/common/module_attr_mixin.py` — ModuleAttrMixin
- `model/diffusion/mask_generator.py` — LowdimMaskGenerator
- `policy/base_image_policy.py` — BaseImagePolicy
- `policy/diffusion_unet_hybrid_image_policy.py` — THE main policy:
  - Takes shape_meta (obs shapes, action dim)
  - Creates ResNet vision encoder via MultiImageObsEncoder
  - Creates ConditionalUnet1D denoiser
  - `predict_action(obs_dict)`:
    1. Encode observations through vision backbone
    2. Initialize noise: `x_T ~ N(0, I)` of shape `(B, horizon, action_dim)`
    3. For t in reverse schedule: `x_{t-1} = denoise(x_t, t, obs_features)`
    4. Return action chunk `x_0[:, :n_action_steps, :]`
  - `compute_loss(batch)`:
    1. Normalize obs and actions
    2. Sample random timestep t
    3. Add noise to actions: `x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise`
    4. Predict noise: `noise_pred = UNet(x_t, t, obs_features)`
    5. Return MSE(noise_pred, noise)

### Phase 6: Training Loop

Replace Hydra workspace with a clean MLX training loop.

**Files:**
- `model/diffusion/ema_model.py` — EMAModel (exponential moving average)
- `model/common/lr_scheduler.py` — Cosine/linear warmup schedulers
- `training/train_diffusion.py` — Training loop:
  - Load dataset, create data iterator
  - Initialize policy, optimizer, EMA
  - Training step: `loss = policy.compute_loss(batch); loss.backward(); optimizer.step()`
  - In MLX: use `mx.grad()` / `nn.value_and_grad()` functional pattern
  - EMA update after each step
  - Checkpoint saving (TopK by validation loss)
  - Logging (JSON + optional wandb)

### Phase 7: Dataset and Evaluation

**Files:**
- `dataset/pusht_image_dataset.py` — PushT dataset (zarr-backed image + low-dim)
- `dataset/base_dataset.py` — Base dataset returning mx.array batches
- Evaluation against PushT environment

---

## Build Order (PRD Sequence)

Each PRD is a self-contained deliverable. Build strictly in order — each depends on the previous.

```
PRD-00: Dev Environment Setup
  - uv project with pyproject.toml
  - Dependencies: mlx, numpy, zarr, Pillow
  - Directory scaffold: src/diffusion_policy_mlx/{compat,model,policy,training,dataset}/
  - Smoke test: import mlx, create mx.array

PRD-01: Compat Foundation
  - tensor_ops.py: 20+ torch.* function equivalents
  - nn_modules.py: Module base class
  - nn_layers.py: Conv1d, ConvTranspose1d, GroupNorm, Linear, Mish, Identity
  - functional.py: mish, silu, pad
  - einops_mlx.py: rearrange for 3 upstream patterns
  - Tests: each op matches torch output within 1e-5

PRD-02: Vision Encoder (ResNet)
  - vision.py: ResNet18/34/50 in MLX from scratch
  - Weight loading from torchvision state_dict (OIHW->OHWI)
  - model_getter.py: get_resnet() returning MLX model
  - multi_image_obs_encoder.py: MultiImageObsEncoder
  - crop_randomizer.py: CropRandomizer
  - Tests: feature vector shape matches, forward pass within tolerance

PRD-03: UNet Denoiser
  - positional_embedding.py: SinusoidalPosEmb
  - conv1d_components.py: Downsample1d, Upsample1d, Conv1dBlock
  - conditional_unet1d.py: ConditionalUnet1D
  - Tests: output shape (B, action_dim, horizon), gradient flow

PRD-04: DDPM/DDIM Scheduler
  - schedulers.py: DDPMScheduler, DDIMScheduler
  - Beta schedules: linear, cosine, squaredcos_cap_v2
  - add_noise, step, set_timesteps
  - Tests: noise schedule values match diffusers, forward/reverse roundtrip

PRD-05: Policy Assembly
  - normalizer.py: LinearNormalizer
  - mask_generator.py: LowdimMaskGenerator
  - diffusion_unet_hybrid_image_policy.py: Full policy
  - Tests: predict_action shape, compute_loss scalar output

PRD-06: Training Loop
  - ema_model.py: EMAModel
  - lr_scheduler.py: get_scheduler()
  - train_diffusion.py: Full training loop
  - Tests: loss decreases over 100 steps on synthetic data

PRD-07: PushT Dataset
  - pusht_image_dataset.py: zarr-backed data loading
  - base_dataset.py: Base class
  - Data download script
  - Tests: batch shapes correct, data range valid

PRD-08: End-to-End Evaluation
  - PushT inference benchmark
  - Comparison: MLX vs PyTorch inference speed
  - Weight conversion script (PyTorch checkpoint -> MLX safetensors)
  - Documentation and usage examples
```

---

## torch -> mlx Mapping Table

### Tensor Operations

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.tensor(data)` | `mx.array(data)` | |
| `torch.zeros(shape)` | `mx.zeros(shape)` | |
| `torch.ones(shape)` | `mx.ones(shape)` | |
| `torch.randn(shape)` | `mx.random.normal(shape)` | |
| `torch.arange(n)` | `mx.arange(n)` | |
| `torch.cat(tensors, dim)` | `mx.concatenate(tensors, axis)` | |
| `torch.stack(tensors, dim)` | `mx.stack(tensors, axis)` | |
| `torch.where(cond, x, y)` | `mx.where(cond, x, y)` | |
| `torch.exp(x)` | `mx.exp(x)` | |
| `torch.sin(x)` / `torch.cos(x)` | `mx.sin(x)` / `mx.cos(x)` | |
| `torch.clamp(x, min, max)` | `mx.clip(x, min, max)` | |
| `x.unsqueeze(dim)` | `mx.expand_dims(x, axis)` | |
| `x.squeeze(dim)` | `mx.squeeze(x, axis)` | |
| `x.reshape(shape)` | `mx.reshape(x, shape)` | |
| `x.permute(dims)` | `mx.transpose(x, axes)` | |
| `x.contiguous()` | no-op | MLX handles layout |
| `x.to(device)` | no-op | MLX auto device |
| `x.detach()` | `mx.stop_gradient(x)` | |
| `x.float()` | `x.astype(mx.float32)` | |

### Neural Network Layers

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `nn.Conv1d(C_in, C_out, K)` | `nn.Conv1d(C_in, C_out, K)` | Input layout differs: NCL vs NLC |
| `nn.ConvTranspose1d(C_in, C_out, K, S, P)` | Custom or `nn.ConvTranspose1d` | Check MLX version |
| `nn.Conv2d(C_in, C_out, K)` | `nn.Conv2d(C_in, C_out, K)` | NCHW vs NHWC |
| `nn.Linear(in, out)` | `nn.Linear(in, out)` | Same API |
| `nn.GroupNorm(G, C)` | `nn.GroupNorm(G, C)` | |
| `nn.BatchNorm2d(C)` | `nn.BatchNorm(C)` | |
| `nn.LayerNorm(C)` | `nn.LayerNorm(C)` | |
| `nn.Mish()` | `nn.Mish()` | Available in recent MLX |
| `nn.SiLU()` | `nn.SiLU()` | |
| `nn.ReLU()` | `nn.ReLU()` | |
| `nn.GELU()` | `nn.GELU()` | |
| `nn.Dropout(p)` | `nn.Dropout(p)` | |
| `nn.Identity()` | `nn.Identity()` | |
| `nn.Embedding(N, D)` | `nn.Embedding(N, D)` | |
| `nn.MultiheadAttention` | `nn.MultiHeadAttention` | Note capitalization |
| `nn.ModuleList([...])` | Python `list` | MLX traces dynamically |
| `nn.Sequential(...)` | Manual chain or custom | |

### Training

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `loss.backward()` | `mx.grad(loss_fn)(params)` | Functional grad |
| `optimizer.step()` | `optimizer.update(model, grads)` | |
| `optimizer.zero_grad()` | Not needed | Functional paradigm |
| `torch.no_grad()` | Not needed | No grad context |
| `model.train()` | `model.train()` | |
| `model.eval()` | `model.eval()` | |
| `model.parameters()` | `model.parameters()` | Returns dict tree |
| `model.state_dict()` | `model.parameters()` | Flat dict via `tree_flatten` |
| `torch.save(state, path)` | `mx.savez(path, **params)` | Or use safetensors |
| `torch.load(path)` | `mx.load(path)` | |
| `DataLoader(dataset, batch_size)` | Custom batching | `mx.array` from numpy |
| `model.to('cuda')` | No-op | MLX auto Metal GPU |

### Diffusers Replacement

| diffusers | Our Implementation | Notes |
|-----------|-------------------|-------|
| `DDPMScheduler(num_train_timesteps, beta_schedule, ...)` | `DDPMScheduler(...)` in `compat/schedulers.py` | |
| `scheduler.add_noise(x_0, noise, t)` | Same API | `sqrt(alpha) * x_0 + sqrt(1-alpha) * noise` |
| `scheduler.step(model_output, t, x_t)` | Same API | Reverse diffusion step |
| `scheduler.set_timesteps(N)` | Same API | Configure inference schedule |
| `DDIMScheduler` | `DDIMScheduler(...)` | Accelerated sampling (10-50 steps vs 1000) |

---

## Success Criteria

### Minimum Viable Port (Demo Day Target)
1. `DiffusionUnetHybridImagePolicy` fully functional in MLX
2. Forward pass (inference) produces valid action trajectories
3. Training loop runs on PushT dataset, loss converges
4. Weight conversion from PyTorch checkpoint works
5. Inference speed within 2x of PyTorch on Apple M-series

### Full Port
6. DDIM accelerated inference (10-50 step generation)
7. EMA model tracking
8. Multi-camera support via MultiImageObsEncoder
9. Transformer denoiser variant (`TransformerForDiffusion`)
10. Complete test suite with cross-framework validation
11. PushT evaluation matching upstream performance (>0.7 success rate with pretrained weights)

---

## File Tree (Target)

```
diffusion-policy-mlx/
  .gitignore
  .claude/CLAUDE.md
  PROMPT.md
  UPSTREAM_VERSION.md
  pyproject.toml

  repositories/
    diffusion-policy-upstream/          # read-only reference (gitignored)

  prds/
    PRD-00-dev-environment.md
    PRD-01-compat-foundation.md
    PRD-02-vision-encoder.md
    PRD-03-unet-denoiser.md
    PRD-04-ddpm-scheduler.md
    PRD-05-policy-assembly.md
    PRD-06-training-loop.md
    PRD-07-pusht-dataset.md
    PRD-08-evaluation.md

  src/
    diffusion_policy_mlx/
      __init__.py
      compat/
        __init__.py
        tensor_ops.py
        nn_modules.py
        nn_layers.py
        functional.py
        vision.py
        schedulers.py
        einops_mlx.py
      model/
        __init__.py
        diffusion/
          __init__.py
          conditional_unet1d.py
          conv1d_components.py
          positional_embedding.py
          transformer_for_diffusion.py
          ema_model.py
          mask_generator.py
        vision/
          __init__.py
          model_getter.py
          multi_image_obs_encoder.py
          crop_randomizer.py
        common/
          __init__.py
          normalizer.py
          module_attr_mixin.py
          lr_scheduler.py
      policy/
        __init__.py
        base_image_policy.py
        diffusion_unet_hybrid_image_policy.py
      training/
        __init__.py
        train_diffusion.py
      dataset/
        __init__.py
        base_dataset.py
        pusht_image_dataset.py

  tests/
    test_compat_tensor_ops.py
    test_compat_nn_layers.py
    test_vision_encoder.py
    test_conv1d_components.py
    test_unet.py
    test_schedulers.py
    test_policy.py
    test_training.py

  scripts/
    convert_weights.py                  # PyTorch -> MLX weight conversion
    download_pusht.py                   # Download PushT dataset
```

---

## Quick Start (After Setup)

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx

# Create environment
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Convert pretrained weights
python scripts/convert_weights.py --checkpoint path/to/pytorch.ckpt --output checkpoints/pusht_mlx.safetensors

# Train on PushT
python -m diffusion_policy_mlx.training.train_diffusion --config configs/pusht_unet_hybrid.yaml

# Inference
python -m diffusion_policy_mlx.scripts.eval --checkpoint checkpoints/pusht_mlx.safetensors
```

---

## Notes for Investors

This port demonstrates AIFLOW LABS' systematic approach to framework translation — a core competency for deploying state-of-the-art robotics research on edge hardware. Diffusion Policy is the dominant paradigm in robot learning (RSS 2023 Best Paper, adopted by Toyota Research, Stanford, Berkeley, Google DeepMind). Running natively on Apple Silicon via MLX enables:

1. **On-device training** — fine-tune policies on the robot's own Apple Silicon compute
2. **Real-time inference** — sub-100ms action generation on M-series chips
3. **Unified memory** — no CPU-GPU transfer overhead, critical for high-frequency control
4. **Edge deployment** — no cloud dependency, works offline in the field

Combined with our PointCNN++ MLX port (3D perception) and LeRobot MLX port (policy framework), this creates a complete Apple Silicon robotics ML stack: **perception -> policy -> action**.

---

*Built by AIFLOW LABS / RobotFlow Labs — March 2026*
