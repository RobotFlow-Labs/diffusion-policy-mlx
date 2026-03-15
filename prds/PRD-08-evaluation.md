# PRD-08: End-to-End Evaluation

**Status:** Complete
**Depends on:** PRD-05 (Policy), PRD-06 (Training), PRD-07 (Dataset)
**Blocks:** Nothing (final PRD)

---

## Objective

End-to-end evaluation: weight conversion from PyTorch checkpoints, inference benchmarks, PushT environment evaluation, and MLX vs PyTorch performance comparison.

---

## Deliverables

### 1. `scripts/convert_weights.py` — PyTorch → MLX Weight Conversion

```python
"""Convert PyTorch Diffusion Policy checkpoint to MLX format.

Usage:
    python scripts/convert_weights.py \
        --checkpoint path/to/pytorch.ckpt \
        --output checkpoints/pusht_mlx

Handles:
  - Policy model weights (UNet + ResNet)
  - EMA model weights
  - Normalizer state (scale/offset)
"""

import torch
import mlx.core as mx
import numpy as np
from pathlib import Path


def convert_checkpoint(checkpoint_path: str, output_dir: str):
    """Convert a full PyTorch checkpoint to MLX format."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load PyTorch checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (depends on workspace format)
    if 'state_dicts' in ckpt:
        model_state = ckpt['state_dicts']['model']
        ema_state = ckpt['state_dicts'].get('ema_model', None)
    elif 'model' in ckpt:
        model_state = ckpt['model']
        ema_state = ckpt.get('ema', None)
    else:
        model_state = ckpt
        ema_state = None

    # Convert model weights
    mlx_weights = convert_state_dict(model_state)
    mx.savez(str(output / "model.npz"), **mlx_weights)

    # Convert EMA weights
    if ema_state is not None:
        mlx_ema = convert_state_dict(ema_state)
        mx.savez(str(output / "ema.npz"), **mlx_ema)

    # Extract and save normalizer
    normalizer_state = extract_normalizer(model_state)
    if normalizer_state:
        mx.savez(str(output / "normalizer.npz"), **normalizer_state)

    print(f"Converted checkpoint saved to {output}")


def convert_state_dict(state_dict: dict) -> dict:
    """Convert PyTorch state dict to MLX parameter dict.

    Key transformations:
      1. Conv2d weights: (C_out, C_in, H, W) → (C_out, H, W, C_in)
      2. Conv1d weights: (C_out, C_in, K) → (C_out, K, C_in)
      3. ConvTranspose1d weights: same transpose
      4. Key path mapping: upstream module hierarchy → our hierarchy
      5. All tensors: torch.Tensor → numpy → mx.array
    """
    mlx_params = {}
    for key, value in state_dict.items():
        # Skip non-parameter entries
        if not isinstance(value, torch.Tensor):
            continue

        np_value = value.numpy()

        # Conv2d weight transpose
        if _is_conv2d_weight(key, np_value):
            np_value = np.transpose(np_value, (0, 2, 3, 1))  # OIHW → OHWI

        # Conv1d weight transpose
        elif _is_conv1d_weight(key, np_value):
            np_value = np.transpose(np_value, (0, 2, 1))  # OIK → OKI

        # Map key path
        mlx_key = map_key_path(key)
        mlx_params[mlx_key] = mx.array(np_value)

    return mlx_params


def map_key_path(pytorch_key: str) -> str:
    """Map PyTorch parameter path to our MLX module path.

    Examples:
        'obs_encoder.nets.image.backbone.conv1.weight'
        → 'obs_encoder.rgb_models.image.conv1.weight'

        'model.down_modules.0.0.blocks.0.block.0.weight'
        → 'model.down_modules.0.0.blocks.0.conv.weight'
    """
    # This needs to be built incrementally during implementation
    # as we discover the exact path mappings
    return pytorch_key  # placeholder


def _is_conv2d_weight(key: str, value: np.ndarray) -> bool:
    return 'weight' in key and value.ndim == 4

def _is_conv1d_weight(key: str, value: np.ndarray) -> bool:
    return 'weight' in key and value.ndim == 3 and 'embedding' not in key

def extract_normalizer(state_dict: dict) -> dict:
    """Extract normalizer parameters from state dict."""
    norm_params = {}
    for key, value in state_dict.items():
        if 'normalizer' in key:
            np_value = value.numpy() if isinstance(value, torch.Tensor) else value
            norm_params[key] = mx.array(np_value)
    return norm_params
```

### 2. `scripts/eval_pusht.py` — PushT Environment Evaluation

```python
"""Evaluate a trained Diffusion Policy on PushT environment.

Usage:
    python scripts/eval_pusht.py \
        --checkpoint checkpoints/pusht_mlx \
        --n_episodes 50 \
        --render
"""

def evaluate(checkpoint_dir: str, n_episodes: int = 50, render: bool = False):
    """Run policy in PushT environment and compute success metrics."""

    # Load policy
    policy = load_policy(checkpoint_dir)
    policy.eval()

    # Create environment
    from diffusion_policy_mlx.env.pusht import PushTEnv
    env = PushTEnv(render=render)

    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        obs_history = []

        while not done:
            # Build observation dict with history
            obs_history.append(obs)
            if len(obs_history) > policy.n_obs_steps:
                obs_history = obs_history[-policy.n_obs_steps:]

            obs_dict = prepare_obs(obs_history, policy.n_obs_steps)

            # Predict action
            with_timer = time.time()
            result = policy.predict_action(obs_dict)
            inference_time = time.time() - with_timer

            action = np.array(result['action'][0])  # first action step

            # Execute in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

        results.append({
            'episode': ep,
            'reward': total_reward,
            'success': info.get('success', total_reward > 0.9),
            'inference_time_ms': inference_time * 1000,
        })
        print(f"Episode {ep}: reward={total_reward:.3f}, "
              f"success={results[-1]['success']}, "
              f"inference={inference_time*1000:.1f}ms")

    # Summary
    success_rate = np.mean([r['success'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_inference = np.mean([r['inference_time_ms'] for r in results])

    print(f"\n{'='*50}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Avg inference: {avg_inference:.1f}ms")
    print(f"{'='*50}")

    return results
```

### 3. `scripts/benchmark.py` — MLX Inference Benchmark

```python
"""Benchmark Diffusion Policy inference on Apple Silicon.

Measures:
  - Single inference latency
  - Throughput (inferences/second)
  - Memory usage (Metal GPU memory)
  - Comparison with PyTorch (if available)
"""

def benchmark_inference(checkpoint_dir: str, n_warmup: int = 5, n_runs: int = 100):
    """Benchmark inference latency and throughput."""
    policy = load_policy(checkpoint_dir)
    policy.eval()

    # Dummy observation
    obs = {
        'image': mx.random.normal((1, 2, 3, 96, 96)),
        'agent_pos': mx.random.normal((1, 2, 2)),
    }

    # Warmup
    for _ in range(n_warmup):
        _ = policy.predict_action(obs)
        mx.eval()

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = policy.predict_action(obs)
        mx.eval()  # Force synchronization
        latencies.append(time.perf_counter() - start)

    # Memory
    peak_mem = mx.get_peak_memory() / 1e9

    p50 = np.percentile(latencies, 50) * 1000
    p95 = np.percentile(latencies, 95) * 1000
    p99 = np.percentile(latencies, 99) * 1000
    throughput = 1.0 / np.mean(latencies)

    print(f"MLX Inference Benchmark ({n_runs} runs)")
    print(f"  Latency p50: {p50:.1f}ms")
    print(f"  Latency p95: {p95:.1f}ms")
    print(f"  Latency p99: {p99:.1f}ms")
    print(f"  Throughput: {throughput:.1f} inferences/sec")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def benchmark_pytorch_comparison():
    """Compare MLX vs PyTorch inference speed."""
    # Load same model in both frameworks
    # Run same input through both
    # Report ratio
    ...
```

### 4. PushT Environment (Minimal)

The PushT environment is a simple 2D pushing task. For evaluation, we need a minimal environment wrapper. Options:

**Option A: Use upstream gym env** (requires mujoco):
```python
# If mujoco is installed on macOS
from diffusion_policy.env.pusht.pusht_env import PushTEnv
```

**Option B: Pure numpy/pygame env** (no mujoco):
```python
# Simplified PushT using shapely + pygame for rendering
# This is sufficient for policy evaluation
```

We choose **Option B** for macOS compatibility. The PushT task is 2D and doesn't need mujoco.

---

## Tests

### `tests/test_weight_conversion.py`

```python
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_conv2d_weight_conversion():
    """Conv2d weights properly transposed OIHW → OHWI."""
    torch_weight = torch.randn(64, 3, 7, 7)
    converted = convert_state_dict({'conv.weight': torch_weight})
    mlx_weight = converted['conv.weight']
    assert mlx_weight.shape == (64, 7, 7, 3)  # OHWI

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_conv1d_weight_conversion():
    """Conv1d weights properly transposed OIK → OKI."""
    torch_weight = torch.randn(256, 128, 5)
    converted = convert_state_dict({'conv.weight': torch_weight})
    mlx_weight = converted['conv.weight']
    assert mlx_weight.shape == (256, 5, 128)  # OKI

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_full_checkpoint_conversion(tmp_path):
    """Full checkpoint conversion produces loadable weights."""
    # Create dummy PyTorch checkpoint
    # Convert
    # Load into MLX model
    # Verify forward pass works
    ...

def test_benchmark_runs():
    """Benchmark completes without error."""
    # Small model, few runs
    ...
```

---

## Success Criteria (End-to-End)

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Weight conversion from PyTorch checkpoint works | no errors |
| 2 | Converted model produces valid action predictions | no NaN/Inf |
| 3 | Inference latency on M-series (100 diffusion steps) | < 500ms |
| 4 | Inference latency with DDIM (10 steps) | < 50ms |
| 5 | PushT success rate with converted weights | > 0.7 |
| 6 | Memory usage for single inference | < 2GB |
| 7 | Training loss converges on PushT dataset | decreasing trend |
| 8 | MLX inference within 2x of PyTorch on same hardware | ratio check |

---

## Upstream Sync Notes

**Weight format dependency:** The `map_key_path` function is the most fragile part — it depends on exact upstream module naming. If upstream restructures their module hierarchy, this needs updating.

**Strategy for sync-resilient weight conversion:**
1. Load upstream checkpoint → get key list
2. Load our model → get parameter name list
3. Match by shape and position, not just name
4. Fall back to manual mapping for ambiguous cases

**Files to watch:**
- Any upstream model file that changes class structure or parameter names
- `workspace/*.py` — checkpoint saving format
- PushT environment — if evaluation protocol changes
