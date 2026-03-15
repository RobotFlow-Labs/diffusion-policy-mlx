# Session Restart Guide

This document captures the full build session context so any future Claude Code session can pick up where we left off.

## Session Summary

**Date:** 2026-03-15
**Duration:** Single continuous session
**Model:** Claude Opus 4.6 (1M context)
**Agents used:** 28 parallel subagents across 6 waves
**Result:** Ship-ready port of Diffusion Policy to Apple MLX

## Repository

- **URL:** https://github.com/RobotFlow-Labs/diffusion-policy-mlx
- **Branch:** main
- **Commits:** 11
- **Working directory:** `/Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx`

## How to Restart a Session

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx
source .venv/bin/activate

# Verify environment
python -c "import diffusion_policy_mlx; print(f'v{diffusion_policy_mlx.__version__}')"
python -c "import mlx.core as mx; print(mx.default_device())"

# Run tests (should be 472 passing)
pytest tests/ --tb=short -q

# Check lint
ruff check src/ tests/ scripts/ examples/
```

## What Was Built (Wave by Wave)

### Wave 1: PRD Creation
- Read PROMPT.md (master build prompt) and upstream source
- Created 9 PRDs in `prds/` with BUILD_ORDER.md
- Set up project scaffold (PRD-00): pyproject.toml, directories, conftest.py

### Wave 2: Core Components (4 parallel agents)
- **Agent 1 (PRD-01):** Compat foundation — Conv1d NCL/NLC, Conv2d NCHW/NHWC, GroupNorm, tensor_ops, einops
- **Agent 2 (PRD-02):** Vision encoder — ResNet18/34/50 in MLX, MultiImageObsEncoder, CropRandomizer
- **Agent 3 (PRD-03):** UNet denoiser — ConditionalUnet1D, FiLM conditioning, skip connections
- **Agent 4 (PRD-04):** Schedulers — DDPMScheduler, DDIMScheduler, cross-validated vs diffusers

### Wave 3: Integration (4 parallel agents)
- **Agent 5 (PRD-05):** Policy assembly — DiffusionUnetHybridImagePolicy, LinearNormalizer, LowdimMaskGenerator
- **Agent 6 (PRD-06):** Training loop — EMAModel, LR schedulers, checkpointing, train_diffusion.py
- **Agent 7 (PRD-07):** PushT dataset — zarr loading, SequenceSampler, collate_batch
- **Agent 8 (PRD-08):** Evaluation — weight converter, benchmark, eval scaffold

### Wave 4: Code Review + Hardening (5 parallel agents)
- **Agent 9:** Fix PRD-02 bugs (deepcopy → clone_module)
- **Agent 10:** Code review PRD-01 (found: clamp guard, Conv2d dilation/groups, interpolate_1d)
- **Agent 11:** Code review PRD-03+04 (found: variance floor, DRY add_noise, DDIM clip)
- **Agent 12:** Integration wiring (unified normalizers, end-to-end tests)
- **Agent 13:** Code quality hardening (deduplicate Conv1d, vectorize CropRandomizer, in-memory clone)

### Wave 5: Polish + Gap Closing (6 parallel agents)
- **Agent 14:** README with Mermaid diagrams
- **Agent 15:** 6 working examples with tests
- **Agent 16:** Lint cleanup + API exports
- **Agent 17:** TransformerForDiffusion + transformer policies
- **Agent 18:** Low-dim policies + PushTLowdimDataset
- **Agent 19:** PushT environment + eval runner
- **Agent 20:** Training utils (dict_util, JsonLogger, validator, wandb)

### Wave 6: Final Review + Fixes (6 parallel agents)
- **Agent 21:** Correctness review (found: GroupNorm pytorch_compatible, DDIM clip ordering)
- **Agent 22:** Security review (found: torch.load pickle, zip slip, unbounded history)
- **Agent 23:** Test quality review (found: shape-only gaps, missing numerical tests)
- **Agent 24:** P0/P1 test fixes (23 new numerical tests, interpolate_1d floor)
- **Agent 25:** MLX Metal GPU verification (mx.eval in all policies, metal_utils, CPU audit)
- **Agent 26:** Remaining fixes (checksum, PIL compat, bounds checking, dead code)

## Key Files to Read First

| File | Why |
|------|-----|
| `PROMPT.md` | Master build prompt — full upstream architecture map and port strategy |
| `CLAUDE.md` | Project config — key design rules, torch→mlx mappings, MLX gotchas |
| `prds/BUILD_ORDER.md` | Dependency graph and build phases |
| `README.md` | User-facing docs with Mermaid diagrams |

## Key Architecture Decisions

1. **Compat layer pattern:** All torch→mlx translation in `src/diffusion_policy_mlx/compat/`. No scattered conditionals.
2. **NCL↔NLC at Conv boundaries:** Compat Conv1d accepts (B,C,L), transposes internally to (B,L,C) for MLX, transposes back.
3. **NCHW↔NHWC at ResNet boundaries:** Same pattern for Conv2d. Internal ResNet processing is NHWC.
4. **No Hydra:** Replaced with YAML + dataclass configs (simpler, explicit).
5. **No diffusers:** Custom DDPM/DDIM schedulers in pure MLX.
6. **mx.eval() strategy:** After optimizer step (training), after denoising loop (inference), after EMA update.
7. **Upstream bug preservation:** ConditionalUnet1D line 307 always-False condition kept for checkpoint compatibility.
8. **clone_module via save/load:** MLX modules can't be deepcopied (nanobind), so we flatten→copy→rebuild.

## Upstream Sync Protocol

```bash
cd repositories/diffusion-policy-upstream && git fetch && git pull
# Check what changed:
git diff HEAD~1 --name-only
# If model/* changed: update mirrored classes, compat, convert_weights
# If config/* changed: update TrainConfig defaults
# Update UPSTREAM_VERSION.md
```

## What's Left (Intentionally Deferred)

| Item | Why deferred |
|------|-------------|
| Kitchen/RoboMimic datasets | Require D4RL/robomimic external deps |
| IBC/BET policies | Different algorithms (not diffusion-based) |
| Video observations | Architectural extension |
| Distributed training | Single-machine MLX focus |
| Real-world hardware | Camera/robot integration — deploy when needed |

## Environment Details

```
Python: 3.12.12
MLX: >=0.22.0 (uses Metal GPU)
PyTorch: 2.10.0 (dev dependency for cross-framework tests)
torchvision: 0.25.0
diffusers: >=0.25.0 (dev dependency for scheduler validation)
OS: macOS (Apple Silicon M-series)
Package manager: uv
Linter: ruff
```

## Useful Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific component
pytest tests/test_unet.py -v
pytest tests/test_policy.py -v
pytest tests/test_integration.py -v

# Run examples
python examples/01_quickstart.py

# Check Metal GPU status
python -c "from diffusion_policy_mlx.common.metal_utils import print_metal_status; print_metal_status()"

# Train on synthetic data (no download needed)
python examples/03_train_synthetic.py

# Benchmark
python scripts/benchmark.py --num-runs 50

# Lint
ruff check src/ tests/ scripts/ examples/
ruff format src/ tests/ scripts/ examples/
```
