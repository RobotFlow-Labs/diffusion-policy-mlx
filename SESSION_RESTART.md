# Session Restart Guide

This document captures the full build session context so any future Claude Code session can pick up exactly where we left off.

---

## How to Resume This Session

Paste this block at the start of a new Claude Code conversation:

```
cd /Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx

Read SESSION_RESTART.md, PROMPT.md, CLAUDE.md, and prds/BUILD_ORDER.md.
This is a port of Diffusion Policy (RSS 2023) from PyTorch to Apple MLX.
The project has 472 tests passing, 82 Python files, ~19.4k LOC.
All 9 PRDs are complete. 6 policy variants shipped. Metal GPU verified.
Resume development from the current state on the main branch.
```

### Quick Environment Check

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx
source .venv/bin/activate

# Verify everything works
python -c "import diffusion_policy_mlx; print(f'v{diffusion_policy_mlx.__version__}')"
python -c "import mlx.core as mx; print(f'Device: {mx.default_device()}')"
pytest tests/ --tb=short -q
ruff check src/ tests/ scripts/ examples/
```

Expected output: `v0.1.0`, `Device: Device(gpu, 0)`, `472 passed`, `All checks passed!`

---

## Session Identity

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Duration** | Single continuous session (~3 hours) |
| **Model** | Claude Opus 4.6 (1M context) |
| **Skill loaded** | `/port-to-mlx` (MLX porting patterns from pointelligence, ZED, triton ports) |
| **Agents used** | 28 parallel subagents across 6 waves |
| **Final git SHA** | `bd8ab2f` (check with `git rev-parse HEAD`) |
| **Branch** | `main` |
| **Commits** | 12 |
| **Working directory** | `/Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx` |

## Repository

| Field | Value |
|-------|-------|
| **GitHub** | https://github.com/RobotFlow-Labs/diffusion-policy-mlx |
| **Website** | https://robotflowlabs.com/ |
| **Upstream** | https://github.com/real-stanford/diffusion_policy |
| **Organization** | AIFLOW LABS / RobotFlow Labs |

---

## Final Stats

| Category | Count | Details |
|----------|-------|---------|
| **Source code** | 52 files | 9,208 lines of Python |
| **Tests** | 20 files | 7,530 lines, 472 test cases |
| **Examples** | 6 files | 657 lines, all runnable standalone |
| **Scripts** | 4 files | 2,028 lines (convert, download, eval, benchmark) |
| **Configs** | 3 files | CNN, Transformer, LowDim YAML |
| **PRDs** | 10 files | 9 component specs + build order |
| **Total Python** | 82 files | **19,423 LOC** |

### Policy Variants Shipped

| Policy | Denoiser | Observation | File |
|--------|----------|-------------|------|
| `DiffusionUnetHybridImagePolicy` | UNet | RGB + low-dim | Primary target |
| `DiffusionUnetImagePolicy` | UNet | RGB only | Image-only |
| `DiffusionUnetLowdimPolicy` | UNet | Low-dim only | No vision encoder |
| `DiffusionTransformerHybridImagePolicy` | Transformer | RGB + low-dim | Alternative denoiser |
| `DiffusionTransformerLowdimPolicy` | Transformer | Low-dim only | Alternative denoiser |

### Quality Gates Passed

| Gate | Result |
|------|--------|
| 472 pytest tests | All green |
| ruff check | 0 issues |
| Cross-framework validation | Conv1d, Conv2d, GroupNorm, BatchNorm, ResNet18/34/50, DDPM, DDIM vs PyTorch/diffusers |
| Security audit | torch.load safe, zip slip protected, SHA-256 download |
| Metal GPU audit | Zero CPU fallbacks in hot paths, mx.eval at all sync points |
| 3x code review | Correctness, security, test quality |
| NaN/Inf stability | Mish overflow, variance floors, sigma floors |

---

## What Was Built (Wave by Wave)

### Wave 1: PRD Creation (sequential)
- Read `PROMPT.md` (master build prompt) and upstream source code
- Explored all 19 upstream files for exact API signatures
- Created 9 PRDs in `prds/` with `BUILD_ORDER.md`
- Set up project scaffold (PRD-00): `pyproject.toml`, directories, `conftest.py`
- Installed environment with `uv venv` + `uv pip install -e ".[dev]"`

### Wave 2: Core Components (4 parallel agents)
| Agent | PRD | What it built | Tests |
|-------|-----|---------------|-------|
| 1 | PRD-01 | Compat foundation (Conv1d NCL/NLC, Conv2d NCHW/NHWC, GroupNorm, tensor_ops, einops) | 65 |
| 2 | PRD-02 | Vision encoder (ResNet18/34/50, MultiImageObsEncoder, CropRandomizer) | 18 |
| 3 | PRD-03 | UNet denoiser (ConditionalUnet1D, FiLM conditioning, skip connections) | 17 |
| 4 | PRD-04 | Schedulers (DDPMScheduler, DDIMScheduler, cross-validated vs diffusers) | 30 |

### Wave 3: Integration (4 parallel agents)
| Agent | PRD | What it built | Tests |
|-------|-----|---------------|-------|
| 5 | PRD-05 | Policy assembly (DiffusionUnetHybridImagePolicy, LinearNormalizer, LowdimMaskGenerator) | 30 |
| 6 | PRD-06 | Training loop (EMAModel, LR schedulers, checkpointing, TopK, train_diffusion.py) | 38 |
| 7 | PRD-07 | PushT dataset (zarr loading, SequenceSampler, collate_batch, download script) | 32 |
| 8 | PRD-08 | Evaluation (weight converter with key mapping, benchmark, eval scaffold) | 64 |

### Wave 4: Code Review + Hardening (5 parallel agents)
| Agent | Task | Findings/Fixes |
|-------|------|----------------|
| 9 | Fix PRD-02 bugs | deepcopy → clone_module (nanobind can't pickle MLX modules) |
| 10 | Code review PRD-01 | clamp None guard, Conv2d dilation/groups, interpolate_1d float division, mish softplus |
| 11 | Code review PRD-03+04 | Variance floor, DRY add_noise, DDIM clip behavior vs diffusers |
| 12 | Integration wiring | Unified normalizers, end-to-end tests, fixed compute_loss obs normalization |
| 13 | Code quality hardening | Deduplicate Conv1d wrappers (-77 LOC), vectorize CropRandomizer, in-memory clone_module |

### Wave 5: Polish + Gap Closing (7 parallel agents)
| Agent | Task | Deliverables |
|-------|------|-------------|
| 14 | README + Mermaid | 5 diagrams (architecture, build order, training flow, inference, module map) |
| 15 | Working examples | 6 runnable scripts + tests/test_examples.py |
| 16 | Lint + API exports | ruff format, 7 __init__.py with clean public API, py.typed marker |
| 17 | Transformer denoiser | TransformerForDiffusion + 2 transformer policy variants |
| 18 | Low-dim policies | BaseLowdimPolicy + UNet lowdim/image + PushTLowdimDataset |
| 19 | PushT environment | PushTEnv (pymunk + numpy fallback), PushTImageRunner |
| 20 | Training utils | dict_util, JsonLogger, WandbLogger, TrainingValidator, gradient clipping |

### Wave 6: Final Review + Fixes (6 parallel agents)
| Agent | Task | Findings/Fixes |
|-------|------|----------------|
| 21 | Correctness review | GroupNorm missing pytorch_compatible=True (2 sites), DDIM pred_eps ordering |
| 22 | Security review | torch.load pickle risk, zip slip, unbounded history, silent wandb |
| 23 | Test quality review | Shape-only gaps (Conv2d, BatchNorm2d, Conv1dBlock), missing numerical tests |
| 24 | P0/P1 test fixes | 23 new numerical tests, interpolate_1d floor fix, weight conversion integration |
| 25 | MLX Metal GPU | mx.eval in all 5 policies, metal_utils module, CPU fallback audit, mx.compile eval |
| 26 | Remaining fixes | Download checksum, PIL compat, bounds checking, dead code, deque popleft |

---

## Git Commit History

```
bd8ab2f docs: project stats in README, SESSION_RESTART.md for continuity
d886137 docs: ship-ready README — Metal GPU section, updated stats, full module map
c1911b5 fix: all review items — numerical tests, Metal GPU, remaining fixes
4f2b842 fix: code review round 2 — GroupNorm compat, DDIM clip ordering
3e23702 fix: security hardening — torch.load safety, zip slip, bounded history
399842a feat: close upstream gaps — transformer, low-dim, env, training utils
4b7024d feat: Mermaid diagrams, 6 working examples, polished README
678e4f0 feat: integration, hardening, lint, docs — project ship-ready
442dab7 feat: PRD-07 — PushT dataset with zarr replay buffer and sequence sampler
3bd2ae7 feat: Phase 3-4 — policy assembly, training loop, evaluation scripts
206cb63 fix: address code review blockers — mish overflow, DDIM sigma floor, local_cond warning
beb9b0f feat: Phase 1-2 complete — compat layer, vision encoder, UNet denoiser, DDPM/DDIM schedulers
```

---

## Key Files to Read First

| File | Why | Priority |
|------|-----|----------|
| `PROMPT.md` | Master build prompt — full upstream architecture map, port strategy, success criteria | Must read |
| `CLAUDE.md` | Project config — key design rules, torch→mlx mappings, MLX gotchas | Must read |
| `.claude/CLAUDE.md` | Same as above (loaded automatically by Claude Code) | Auto-loaded |
| `prds/BUILD_ORDER.md` | Dependency graph and build phases | Reference |
| `README.md` | User-facing docs with Mermaid diagrams, stats, Metal GPU section | Reference |
| `SESSION_RESTART.md` | This file — full session context | You're reading it |

---

## Key Architecture Decisions

1. **Compat layer pattern:** All torch→mlx translation in `src/diffusion_policy_mlx/compat/`. No scattered conditionals.
2. **NCL↔NLC at Conv1d boundaries:** Compat Conv1d accepts (B,C,L), transposes internally to (B,L,C) for MLX, transposes back. Same for ConvTranspose1d.
3. **NCHW↔NHWC at Conv2d/ResNet boundaries:** Same pattern for 2D convolutions. Internal ResNet processing is NHWC.
4. **GroupNorm `pytorch_compatible=True`:** Required everywhere — MLX default uses unbiased variance which differs from PyTorch.
5. **No Hydra:** Replaced with YAML + dataclass configs (simpler, explicit).
6. **No diffusers:** Custom DDPM/DDIM schedulers in pure MLX.
7. **`mx.eval()` strategy:** After optimizer step (training), after denoising loop (inference), after EMA update. Prevents lazy graph memory explosion.
8. **Upstream bug preservation:** ConditionalUnet1D always-False condition on local_cond up-path kept for checkpoint compatibility with published weights.
9. **`clone_module` via flatten→copy→rebuild:** MLX modules can't be `copy.deepcopy`'d (nanobind objects not picklable).
10. **`__getitem__` returns numpy, `collate_batch` converts to mx.array:** Avoids creating many small mx.arrays during data loading.

---

## Upstream Sync Protocol

```bash
cd repositories/diffusion-policy-upstream && git fetch && git pull
# Check what changed:
git diff HEAD~1 --name-only
# If model/* changed: update mirrored classes, compat/, convert_weights.py
# If config/* changed: update TrainConfig defaults
# If dataset/* changed: update PushTImageDataset
# Update UPSTREAM_VERSION.md with new sync date
```

---

## What's Left (Intentionally Deferred)

| Item | Why deferred | Effort to add |
|------|-------------|---------------|
| Kitchen/RoboMimic datasets | Require D4RL/robomimic external deps | 3-4 hours |
| IBC/BET policies | Different algorithms (not diffusion-based) | 5+ hours |
| Video observations | Architectural extension (temporal modeling) | 4+ hours |
| Distributed training | Single-machine MLX focus | Not applicable |
| Real-world hardware integration | Camera/robot — deploy when needed | 20+ hours |
| Wandb integration testing | Requires wandb account | 1 hour |
| CI/CD pipeline | GitHub Actions with macOS runners | 2 hours |

---

## Environment Details

```
Python:       3.12.12
MLX:          >=0.22.0 (Metal GPU backend)
PyTorch:      2.10.0 (dev dependency for cross-framework tests only)
torchvision:  0.25.0 (dev dependency)
diffusers:    >=0.25.0 (dev dependency for scheduler validation)
OS:           macOS (Apple Silicon M-series)
Package mgr:  uv
Linter:       ruff (0 issues)
Test runner:  pytest (472 passing)
```

---

## Useful Commands

```bash
# === Quick Health Check ===
source .venv/bin/activate
pytest tests/ --tb=short -q        # 472 passed
ruff check src/ tests/ scripts/    # All checks passed

# === Run by component ===
pytest tests/test_compat_nn_layers.py -v   # Compat layer (71 tests)
pytest tests/test_unet.py -v               # UNet denoiser (17 tests)
pytest tests/test_transformer.py -v        # Transformer (27 tests)
pytest tests/test_policy.py -v             # Policy (13 tests)
pytest tests/test_schedulers.py -v         # Schedulers (30 tests)
pytest tests/test_integration.py -v        # End-to-end (10 tests)

# === Run examples ===
python examples/01_quickstart.py           # ~2s, no data needed
python examples/03_train_synthetic.py      # ~5s, full training loop

# === Metal GPU ===
python -c "from diffusion_policy_mlx.common.metal_utils import print_metal_status; print_metal_status()"

# === Full training (requires dataset) ===
python scripts/download_pusht.py --output data/
python -m diffusion_policy_mlx.training.train_diffusion --config configs/pusht_diffusion_policy_cnn.yaml

# === Weight conversion ===
python scripts/convert_weights.py --checkpoint path/to/file.ckpt --output checkpoints/mlx/

# === Benchmark ===
python scripts/benchmark.py --num-runs 50

# === Lint ===
ruff check src/ tests/ scripts/ examples/
ruff format src/ tests/ scripts/ examples/
```
