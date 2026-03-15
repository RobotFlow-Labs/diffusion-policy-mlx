# PRD-00: Dev Environment Setup

**Status:** Complete
**Depends on:** Nothing
**Blocks:** PRD-01 through PRD-08

---

## Objective

Scaffold the project structure, configure `uv` packaging, install MLX and core dependencies, and verify the development environment works on Apple Silicon.

---

## Deliverables

### 1. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "diffusion-policy-mlx"
version = "0.1.0"
description = "Diffusion Policy for Apple Silicon via MLX"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.22.0",
    "numpy>=1.24.0",
    "Pillow>=9.0.0",
    "zarr>=2.12.0",
    "h5py>=3.7.0",
    "scipy>=1.10.0",
    "safetensors>=0.4.0",
    "tqdm>=4.60.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "ruff>=0.4.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "diffusers>=0.25.0",
    "einops>=0.4.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Design note:** `dev` extras include PyTorch + diffusers for cross-framework validation tests only. The production package needs only MLX.

### 2. Directory Structure

```
src/
  diffusion_policy_mlx/
    __init__.py
    compat/
      __init__.py
    model/
      __init__.py
      diffusion/
        __init__.py
      vision/
        __init__.py
      common/
        __init__.py
    policy/
      __init__.py
    training/
      __init__.py
    dataset/
      __init__.py
tests/
  __init__.py
  conftest.py
scripts/
prds/
configs/
```

### 3. `tests/conftest.py`

Shared fixtures:
- `has_torch` — skip marker for cross-framework tests when torch not installed
- `has_diffusers` — skip marker for scheduler comparison tests
- `device_info` — print MLX device info
- `check_close(mlx_result, reference, atol, rtol)` — standard comparison helper

### 4. `src/diffusion_policy_mlx/__init__.py`

```python
"""Diffusion Policy for Apple Silicon via MLX."""
__version__ = "0.1.0"
```

### 5. Smoke Test

```python
# tests/test_smoke.py
def test_mlx_import():
    import mlx.core as mx
    import mlx.nn as nn
    x = mx.ones((2, 3))
    assert x.shape == (2, 3)

def test_package_import():
    import diffusion_policy_mlx
    assert diffusion_policy_mlx.__version__ == "0.1.0"
```

---

## Setup Commands

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/diffusion-policy-mlx
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/test_smoke.py -v
ruff check src/
```

---

## Acceptance Criteria

| # | Criterion | Validation |
|---|-----------|------------|
| 1 | `uv pip install -e ".[dev]"` completes without errors | Run command |
| 2 | `import diffusion_policy_mlx` works | `pytest tests/test_smoke.py` |
| 3 | `import mlx.core as mx; mx.ones((2,3))` works | Smoke test |
| 4 | `ruff check src/` reports 0 errors | Run command |
| 5 | All `__init__.py` files exist in every package | Glob check |
| 6 | `pytest tests/ -v` runs (even if only smoke tests) | Run command |

---

## Upstream Sync Notes

- The `repositories/` directory is `.gitignore`d — upstream stays read-only
- No upstream code is copied; all our code lives in `src/diffusion_policy_mlx/`
- Upstream dependency versions are recorded in `UPSTREAM_VERSION.md`
- When upstream updates, we only need to check if API signatures changed in the files we mirror
