# Diffusion Policy MLX — Build Order & Dependency Graph

## Dependency Graph

```
PRD-00: Dev Environment
  │
  └─→ PRD-01: Compat Foundation
        │
        ├─→ PRD-02: Vision Encoder (ResNet + MultiImageObsEncoder)
        │     │
        ├─→ PRD-03: UNet Denoiser (ConditionalUnet1D)
        │     │
        ├─→ PRD-04: DDPM/DDIM Scheduler
        │     │
        │     └─────────────────────────────┐
        │                                   │
        └─→ PRD-05: Policy Assembly ←───────┘
              │           (combines PRD-02 + PRD-03 + PRD-04)
              │
              ├─→ PRD-06: Training Loop
              │     │
              │     └─→ PRD-08: End-to-End Evaluation
              │           │
              └─→ PRD-07: PushT Dataset ──────┘
```

## Build Phases

### Phase 1: Foundation (Sequential)
```
PRD-00 → PRD-01
```
Must be sequential. Everything depends on the compat layer.

### Phase 2: Core Components (Parallel)
```
PRD-02 ─┐
PRD-03 ─┼─ can build in parallel
PRD-04 ─┘
```
All three depend only on PRD-01. Independent of each other.
**This is the optimal parallelization point — run 3 agents simultaneously.**

### Phase 3: Integration (Sequential)
```
PRD-05 (needs PRD-02 + PRD-03 + PRD-04)
```
Assembly requires all three core components.

### Phase 4: Training & Data (Parallel)
```
PRD-06 ─┐
PRD-07 ─┘─ can build in parallel (both need PRD-05)
```

### Phase 5: Evaluation (Sequential)
```
PRD-08 (needs PRD-06 + PRD-07)
```

## Estimated Effort

| PRD | LOC (est.) | Test Files | Complexity |
|-----|-----------|-----------|------------|
| PRD-00 | ~50 | 1 | Low |
| PRD-01 | ~430 | 2 | Medium |
| PRD-02 | ~600 | 1 | High |
| PRD-03 | ~500 | 2 | High |
| PRD-04 | ~400 | 1 | Medium |
| PRD-05 | ~700 | 2 | High |
| PRD-06 | ~500 | 1 | Medium |
| PRD-07 | ~300 | 1 | Medium |
| PRD-08 | ~400 | 1 | Medium |
| **Total** | **~3,880** | **12** | |

## Upstream Sync Strategy

The fork stays sync-friendly because:

1. **No upstream files modified** — `repositories/` is read-only
2. **Mirrored structure** — our module hierarchy mirrors upstream
3. **Single translation layer** — all torch→mlx in `compat/`
4. **Key mapping in one place** — weight conversion in `scripts/convert_weights.py`

### When upstream updates:

```bash
# 1. Pull upstream
cd repositories/diffusion-policy-upstream
git fetch && git pull

# 2. Check what changed
git diff HEAD~1 --name-only

# 3. If model/* changed:
#    - Check class signatures → update our mirrored classes
#    - Check new torch.* calls → add to compat/
#    - Check weight shapes → update convert_weights.py
#    - Re-run cross-framework tests

# 4. If config/* changed:
#    - Update our TrainConfig defaults

# 5. If dataset/* changed:
#    - Update PushTImageDataset

# 6. Update UPSTREAM_VERSION.md
```

### Sync risk by PRD:

| PRD | Sync Risk | Why |
|-----|-----------|-----|
| PRD-00 | None | Pure scaffolding |
| PRD-01 | Low | Compat layer adapts to API changes |
| PRD-02 | Medium | ResNet arch is stable but torchvision weights change |
| PRD-03 | Low | UNet arch hasn't changed since paper |
| PRD-04 | Low | DDPM/DDIM math is fixed |
| PRD-05 | Medium | Policy is the integration point — most likely to change |
| PRD-06 | Low | Training loop is our own design |
| PRD-07 | Low | PushT zarr format is stable |
| PRD-08 | Medium | Weight key paths depend on upstream module names |
