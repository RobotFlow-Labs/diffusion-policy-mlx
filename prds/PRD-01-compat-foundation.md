# PRD-01: Compat Foundation

**Status:** Not Started
**Depends on:** PRD-00
**Blocks:** PRD-02, PRD-03, PRD-04, PRD-05, PRD-06

---

## Objective

Build the `compat/` translation layer — the single foundation that makes MLX behave like PyTorch from the caller's perspective. Every subsequent PRD builds on this.

---

## Design Principle: Transparent Translation

The compat layer does NOT wrap MLX in a PyTorch-like API. It provides drop-in replacements for the specific torch operations used in upstream diffusion_policy. Callers use MLX conventions (`mx.array`, `mlx.nn.Module`) but get helpers for the patterns that differ.

---

## Deliverables

### 1. `compat/tensor_ops.py` — Tensor Operation Helpers

Functions that bridge torch→mlx gaps used in upstream:

```python
# Mapping of upstream calls to what we provide:

# torch.tensor([x]) → mx.array([x])                    # direct
# torch.zeros/ones/randn → mx.zeros/ones/random.normal  # direct
# torch.cat(tensors, dim) → mx.concatenate(tensors, axis)  # direct
# torch.stack(tensors, dim) → mx.stack(tensors, axis)      # direct
# torch.is_tensor(x) → isinstance(x, mx.array)

# These need thin wrappers:
def is_tensor(x) -> bool:
    """torch.is_tensor() replacement."""
    return isinstance(x, mx.array)

def tensor_to_long(x):
    """x.long() → x.astype(mx.int32) (no int64 on Metal)."""
    return x.astype(mx.int32)

def expand_as_batch(timesteps, batch_size):
    """timesteps.expand(batch_size) — broadcast scalar/1D to batch."""
    if timesteps.ndim == 0:
        return mx.broadcast_to(timesteps, (batch_size,))
    if timesteps.shape[0] == 1:
        return mx.broadcast_to(timesteps, (batch_size,))
    return timesteps

def pad_1d(x, pad, mode='constant', value=0.0):
    """F.pad for 1D: x is (B, C, L), pad is (left, right)."""
    # MLX mx.pad works on any axis
    ...

def clamp(x, min_val=None, max_val=None):
    """torch.clamp → mx.clip."""
    return mx.clip(x, min_val, max_val)
```

**Upstream usage scan** (actual patterns found in source):

| Pattern | Files Using It | Count |
|---------|---------------|-------|
| `torch.is_tensor(x)` | conditional_unet1d.py | 2 |
| `timesteps.expand(B)` | conditional_unet1d.py | 1 |
| `torch.tensor([x], dtype=torch.long)` | conditional_unet1d.py | 1 |
| `torch.cat([a, b], axis=-1)` | conditional_unet1d.py, policy | 5+ |
| `torch.clamp(x, min, max)` | normalizer.py, schedulers | 3+ |
| `einops.rearrange('b h t -> b t h')` | conditional_unet1d.py | 3 |
| `x.unsqueeze(dim)` / `x.squeeze(dim)` | normalizer.py, policy | 5+ |
| `x.flatten(start_dim)` | multi_image_obs_encoder.py | 2 |
| `x.reshape(shape)` | conditional_unet1d.py | 1 |
| `torch.zeros_like(x)` | policy, schedulers | 5+ |
| `torch.randn_like(x)` | policy, schedulers | 3+ |
| `x.to(device)` | throughout | No-op in MLX |
| `x.detach()` | ema_model.py | 2 |

### 2. `compat/nn_modules.py` — Module Utilities

MLX `mlx.nn.Module` already provides `parameters()`, `__call__`, and `train()`/`eval()`. We need utilities for patterns that differ:

```python
class ModuleAttrMixin(mlx.nn.Module):
    """Upstream ModuleAttrMixin — adds .device/.dtype properties.
    In MLX, device is always Metal (automatic), dtype from parameters."""

    @property
    def device(self):
        return "mlx"  # MLX has no device concept

    @property
    def dtype(self):
        params = self.parameters()
        # Walk param tree to find first leaf
        ...
        return leaf.dtype

class DictOfTensorMixin(mlx.nn.Module):
    """Store named tensor parameters as a dict tree."""

    def __init__(self):
        super().__init__()
        self.params_dict = {}

    # fit/normalize/unnormalize added by subclasses
```

### 3. `compat/nn_layers.py` — Layer Wrappers with Layout Translation

**Critical: Conv1d NCL↔NLC translation.**

Upstream uses PyTorch convention: input `(B, C, L)`, weight `(C_out, C_in, K)`.
MLX Conv1d expects: input `(B, L, C)`, weight `(C_out, K, C_in)`.

```python
class Conv1d(mlx.nn.Module):
    """Drop-in for torch.nn.Conv1d with NCL interface.

    Caller sends (B, C_in, L), gets back (B, C_out, L').
    Internally transposes to (B, L, C_in) for MLX Conv1d,
    then transposes output back to (B, C_out, L').
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self._conv = mlx.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def __call__(self, x):
        # x: (B, C, L) → (B, L, C) → conv → (B, L', C') → (B, C', L')
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        x = mx.transpose(x, axes=(0, 2, 1))
        return x

class ConvTranspose1d(mlx.nn.Module):
    """Drop-in for torch.nn.ConvTranspose1d with NCL interface."""
    # Same transpose pattern as Conv1d
    ...

class GroupNorm(mlx.nn.Module):
    """torch.nn.GroupNorm — same API, MLX expects channels-last."""
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self._gn = mlx.nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def __call__(self, x):
        # Upstream sends (B, C, L) — MLX GroupNorm expects channels-last
        x = mx.transpose(x, axes=(0, 2, 1))  # (B, L, C)
        x = self._gn(x)
        x = mx.transpose(x, axes=(0, 2, 1))  # (B, C, L)
        return x

# These are direct pass-through (no layout issues):
# Linear = mlx.nn.Linear
# Mish = mlx.nn.Mish (or lambda: nn.Mish())
# SiLU = mlx.nn.SiLU
# Identity = lambda x: x (or mlx.nn.Identity if available)
# Dropout = mlx.nn.Dropout
# Embedding = mlx.nn.Embedding

class Sequential(mlx.nn.Module):
    """torch.nn.Sequential — sequential module composition."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Conv2d translation (for ResNet in PRD-02):**

```python
class Conv2d(mlx.nn.Module):
    """Drop-in for torch.nn.Conv2d with NCHW interface.

    Caller sends (B, C, H, W), gets back (B, C', H', W').
    Internally uses NHWC for MLX.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self._conv = mlx.nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def __call__(self, x):
        # x: (B, C, H, W) → (B, H, W, C) → conv → (B, H', W', C') → (B, C', H', W')
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self._conv(x)
        x = mx.transpose(x, axes=(0, 3, 1, 2))
        return x
```

### 4. `compat/functional.py` — Functional Operations

```python
def mish(x):
    """F.mish activation."""
    return x * mx.tanh(mx.softplus(x))
    # Or use mlx.nn.mish if available

def silu(x):
    """F.silu / swish activation."""
    return x * mx.sigmoid(x)

def interpolate_1d(x, scale_factor=None, size=None, mode='nearest'):
    """F.interpolate for 1D signals. x: (B, C, L)."""
    # Used by Upsample1d if ConvTranspose1d not available
    ...

def pad_1d(x, padding, mode='constant', value=0.0):
    """F.pad for 3D tensors (B, C, L). padding = (left, right)."""
    if mode == 'constant':
        return mx.pad(x, [(0,0), (0,0), (padding[0], padding[1])],
                       constant_values=value)
    elif mode == 'replicate':
        # Manual edge replication
        ...
```

### 5. `compat/einops_mlx.py` — Rearrange Patterns

Only 3 patterns are used in upstream:

```python
def rearrange_b_h_t_to_b_t_h(x):
    """einops.rearrange(x, 'b h t -> b t h') — transpose last 2 dims."""
    return mx.transpose(x, axes=(0, 2, 1))

def rearrange_b_t_h_to_b_h_t(x):
    """einops.rearrange(x, 'b t h -> b h t') — transpose last 2 dims."""
    return mx.transpose(x, axes=(0, 2, 1))

def rearrange_batch_t_to_batch_t_1(x):
    """Rearrange('batch t -> batch t 1') — used in FiLM cond_encoder."""
    return mx.expand_dims(x, axis=-1)
```

---

## File Summary

| File | LOC (est.) | Key Classes/Functions |
|------|-----------|----------------------|
| `compat/tensor_ops.py` | ~80 | `is_tensor`, `expand_as_batch`, `pad_1d`, `clamp` |
| `compat/nn_modules.py` | ~60 | `ModuleAttrMixin`, `DictOfTensorMixin` |
| `compat/nn_layers.py` | ~200 | `Conv1d`, `ConvTranspose1d`, `Conv2d`, `GroupNorm`, `Sequential` |
| `compat/functional.py` | ~60 | `mish`, `silu`, `interpolate_1d`, `pad_1d` |
| `compat/einops_mlx.py` | ~30 | 3 rearrange functions |
| `compat/__init__.py` | ~20 | Re-exports |

---

## Tests

### `tests/test_compat_tensor_ops.py`

```python
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_is_tensor():
    assert is_tensor(mx.array([1.0]))
    assert not is_tensor([1.0])

def test_expand_as_batch_scalar():
    t = mx.array(5)
    out = expand_as_batch(t, 4)
    assert out.shape == (4,)

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_cat_matches_torch():
    a_np, b_np = np.random.randn(2, 3), np.random.randn(2, 3)
    torch_out = torch.cat([torch.tensor(a_np), torch.tensor(b_np)], dim=0)
    mlx_out = mx.concatenate([mx.array(a_np), mx.array(b_np)], axis=0)
    np.testing.assert_allclose(np.array(mlx_out), torch_out.numpy(), atol=1e-6)
```

### `tests/test_compat_nn_layers.py`

```python
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_conv1d_shape():
    """Our Conv1d takes NCL and returns NCL, matching PyTorch."""
    conv_mlx = compat.Conv1d(16, 32, 3, padding=1)
    x = mx.random.normal((2, 16, 50))  # (B, C, L)
    out = conv_mlx(x)
    assert out.shape == (2, 32, 50)

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_conv1d_numerics():
    """Conv1d output matches PyTorch given same weights."""
    # Create both, copy weights, compare output
    ...

def test_groupnorm_ncl():
    """GroupNorm takes (B,C,L) and returns (B,C,L)."""
    gn = compat.GroupNorm(8, 32)
    x = mx.random.normal((2, 32, 50))
    out = gn(x)
    assert out.shape == (2, 32, 50)

def test_sequential():
    seq = compat.Sequential(
        compat.Conv1d(16, 32, 3, padding=1),
        compat.GroupNorm(8, 32),
    )
    x = mx.random.normal((2, 16, 50))
    out = seq(x)
    assert out.shape == (2, 32, 50)

def test_conv_transpose1d_shape():
    ct = compat.ConvTranspose1d(32, 32, 4, stride=2, padding=1)
    x = mx.random.normal((2, 32, 25))
    out = ct(x)
    assert out.shape == (2, 32, 50)  # 2x upsample
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | `Conv1d` accepts (B,C,L), returns (B,C',L') matching PyTorch | shape exact, values atol=1e-5 |
| 2 | `ConvTranspose1d` 2x upsamples correctly | shape exact, values atol=1e-5 |
| 3 | `GroupNorm` on (B,C,L) matches PyTorch GroupNorm | atol=1e-5 |
| 4 | `Sequential` chains layers correctly | shape exact |
| 5 | `rearrange` functions match einops output | exact |
| 6 | All compat imports work: `from diffusion_policy_mlx.compat import *` | no errors |
| 7 | Cross-framework numeric tests pass (when torch available) | per-test tolerance |

---

## Upstream Sync Notes

If upstream adds new `torch.*` calls in the files we port, we add the corresponding compat function here. The compat layer is the **single place** where torch→mlx translation lives — no scattered conditionals.

**Files to watch on upstream sync:**
- `model/diffusion/conditional_unet1d.py` — uses `einops.rearrange`, `torch.is_tensor`, `torch.cat`
- `model/diffusion/conv1d_components.py` — uses `nn.Conv1d`, `nn.ConvTranspose1d`, `nn.GroupNorm`, `nn.Mish`
- `model/common/normalizer.py` — uses `torch.clamp`, `torch.zeros_like`, `unsqueeze`
- `policy/diffusion_unet_hybrid_image_policy.py` — uses `torch.randn`, `torch.cat`, `torch.zeros`
