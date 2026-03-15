"""Benchmark Diffusion Policy inference on Apple Silicon.

Measures:
  - Single inference latency (p50/p95/p99)
  - Throughput (inferences/second)
  - Memory usage (Metal GPU memory)
  - Optional comparison with PyTorch (if available)

Usage:
    python scripts/benchmark.py --num-runs 100 --ddim-steps 10
    python scripts/benchmark.py --checkpoint dir/ --num-runs 50
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from diffusion_policy_mlx.compat.schedulers import DDIMScheduler, DDPMScheduler
from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    mode: str  # "ddpm" or "ddim"
    num_runs: int
    num_diffusion_steps: int
    latencies_ms: List[float] = field(default_factory=list)
    peak_memory_gb: float = 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms))

    @property
    def throughput(self) -> float:
        """Inferences per second."""
        mean_s = self.mean_ms / 1000.0
        return 1.0 / mean_s if mean_s > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"  Mode: {self.mode.upper()} ({self.num_diffusion_steps} steps)",
            f"  Runs: {self.num_runs}",
            f"  Latency p50: {self.p50_ms:.1f}ms",
            f"  Latency p95: {self.p95_ms:.1f}ms",
            f"  Latency p99: {self.p99_ms:.1f}ms",
            f"  Latency mean: {self.mean_ms:.1f}ms",
            f"  Throughput: {self.throughput:.1f} inferences/sec",
            f"  Peak GPU memory: {self.peak_memory_gb:.3f} GB",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_model(
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
    down_dims: tuple = (256, 512, 1024),
    diffusion_step_embed_dim: int = 256,
    kernel_size: int = 5,
    n_groups: int = 8,
    cond_predict_scale: bool = True,
) -> ConditionalUnet1D:
    """Create a ConditionalUnet1D model with PushT-like config."""
    global_cond_dim = obs_dim * n_obs_steps

    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )

    return model


def create_dummy_inputs(
    batch_size: int = 1,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
) -> Dict[str, mx.array]:
    """Create dummy inputs for benchmarking."""
    return {
        "sample": mx.random.normal((batch_size, horizon, action_dim)),
        "global_cond": mx.random.normal((batch_size, obs_dim * n_obs_steps)),
    }


# ---------------------------------------------------------------------------
# Single-inference benchmark
# ---------------------------------------------------------------------------


def run_single_inference(
    model: ConditionalUnet1D,
    sample: mx.array,
    timestep: int,
    global_cond: mx.array,
) -> mx.array:
    """Run a single UNet forward pass."""
    result = model(
        sample,
        mx.array([timestep], dtype=mx.int32),
        global_cond=global_cond,
    )
    mx.eval(result)
    return result


def benchmark_single_forward(
    model: Optional[ConditionalUnet1D] = None,
    num_warmup: int = 5,
    num_runs: int = 100,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
) -> BenchmarkResult:
    """Benchmark a single UNet forward pass (no diffusion loop)."""
    if model is None:
        model = create_model(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
        )

    inputs = create_dummy_inputs(
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )

    # Warmup
    for _ in range(num_warmup):
        run_single_inference(model, inputs["sample"], 0, inputs["global_cond"])

    # Reset peak memory tracking
    mx.get_peak_memory()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_single_inference(model, inputs["sample"], 0, inputs["global_cond"])
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    peak_mem = mx.get_peak_memory() / 1e9

    result = BenchmarkResult(
        mode="single_forward",
        num_runs=num_runs,
        num_diffusion_steps=1,
        latencies_ms=latencies,
        peak_memory_gb=peak_mem,
    )

    return result


# ---------------------------------------------------------------------------
# Full diffusion loop benchmark
# ---------------------------------------------------------------------------


def run_diffusion_loop(
    model: ConditionalUnet1D,
    scheduler,
    sample: mx.array,
    global_cond: mx.array,
    num_inference_steps: int,
) -> mx.array:
    """Run full diffusion denoising loop."""
    scheduler.set_timesteps(num_inference_steps)

    trajectory = mx.random.normal(sample.shape)

    for t in scheduler.timesteps:
        t_val = int(t.item()) if isinstance(t, mx.array) else int(t)
        model_output = model(
            trajectory,
            mx.array([t_val], dtype=mx.int32),
            global_cond=global_cond,
        )
        mx.eval(model_output)
        result = scheduler.step(model_output, t_val, trajectory)
        trajectory = result.prev_sample
        mx.eval(trajectory)

    return trajectory


def benchmark_diffusion(
    model: Optional[ConditionalUnet1D] = None,
    mode: str = "ddpm",
    num_inference_steps: Optional[int] = None,
    num_warmup: int = 2,
    num_runs: int = 10,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
) -> BenchmarkResult:
    """Benchmark full diffusion inference loop.

    Args:
        model: ConditionalUnet1D model (created if None)
        mode: 'ddpm' (full steps) or 'ddim' (accelerated)
        num_inference_steps: Number of diffusion steps.
            Default: 100 for DDPM, 10 for DDIM
        num_warmup: Warmup runs
        num_runs: Benchmark runs
    """
    if model is None:
        model = create_model(
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
        )

    # Create scheduler
    if mode == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        if num_inference_steps is None:
            num_inference_steps = 100
    elif mode == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        if num_inference_steps is None:
            num_inference_steps = 10
    else:
        raise ValueError(f"Unknown mode: {mode}")

    inputs = create_dummy_inputs(
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )

    # Warmup
    for _ in range(num_warmup):
        run_diffusion_loop(
            model,
            scheduler,
            inputs["sample"],
            inputs["global_cond"],
            num_inference_steps,
        )

    # Reset peak memory
    mx.get_peak_memory()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_diffusion_loop(
            model,
            scheduler,
            inputs["sample"],
            inputs["global_cond"],
            num_inference_steps,
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms

    peak_mem = mx.get_peak_memory() / 1e9

    result = BenchmarkResult(
        mode=mode,
        num_runs=num_runs,
        num_diffusion_steps=num_inference_steps,
        latencies_ms=latencies,
        peak_memory_gb=peak_mem,
    )

    return result


# ---------------------------------------------------------------------------
# PyTorch comparison (optional)
# ---------------------------------------------------------------------------


def benchmark_pytorch_forward(
    num_warmup: int = 5,
    num_runs: int = 100,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
) -> Optional[BenchmarkResult]:
    """Benchmark PyTorch forward pass for comparison."""
    if not HAS_TORCH:
        logger.warning("PyTorch not available, skipping comparison")
        return None

    try:
        # Import upstream model
        from diffusion_policy.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D as TorchUnet,
        )

        global_cond_dim = obs_dim * n_obs_steps
        torch_model = TorchUnet(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        torch_model.eval()

        sample = torch.randn(1, horizon, action_dim)
        cond = torch.randn(1, global_cond_dim)
        t = torch.tensor([0])

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                torch_model(sample, t, global_cond=cond)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                torch_model(sample, t, global_cond=cond)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed * 1000)

        return BenchmarkResult(
            mode="pytorch_forward",
            num_runs=num_runs,
            num_diffusion_steps=1,
            latencies_ms=latencies,
            peak_memory_gb=0.0,
        )
    except Exception as e:
        logger.warning("PyTorch comparison failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    checkpoint_dir: Optional[str] = None,
    num_runs: int = 100,
    ddim_steps: int = 10,
    ddpm_steps: int = 100,
    compare_pytorch: bool = False,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
) -> Dict[str, BenchmarkResult]:
    """Run full benchmark suite.

    Args:
        checkpoint_dir: Optional directory with converted weights
        num_runs: Number of benchmark runs
        ddim_steps: Number of DDIM inference steps
        ddpm_steps: Number of DDPM inference steps
        compare_pytorch: Whether to run PyTorch comparison
    """
    results: Dict[str, BenchmarkResult] = {}

    # Create or load model
    model = create_model(
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )

    if checkpoint_dir is not None:
        from pathlib import Path

        ckpt_path = Path(checkpoint_dir)
        weights_path = ckpt_path / "model.npz"
        if weights_path.exists():
            weights = dict(mx.load(str(weights_path)))
            # Filter to only model.* keys and strip prefix
            model_weights = {}
            for k, v in weights.items():
                if k.startswith("model."):
                    model_weights[k[len("model.") :]] = v
            if model_weights:
                model.load_weights(list(model_weights.items()))
                logger.info("Loaded model weights from %s", weights_path)

    print("=" * 60)
    print("MLX Diffusion Policy Inference Benchmark")
    print("=" * 60)

    # 1. Single forward pass
    print("\n--- Single Forward Pass ---")
    fwd_result = benchmark_single_forward(
        model=model,
        num_runs=num_runs,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )
    results["single_forward"] = fwd_result
    print(fwd_result.summary())

    # 2. DDIM diffusion loop
    print(f"\n--- DDIM Diffusion ({ddim_steps} steps) ---")
    ddim_runs = max(1, num_runs // 10)  # fewer runs for full loops
    ddim_result = benchmark_diffusion(
        model=model,
        mode="ddim",
        num_inference_steps=ddim_steps,
        num_runs=ddim_runs,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )
    results["ddim"] = ddim_result
    print(ddim_result.summary())

    # 3. DDPM diffusion loop (fewer runs, it's slow)
    print(f"\n--- DDPM Diffusion ({ddpm_steps} steps) ---")
    ddpm_runs = max(1, num_runs // 20)
    ddpm_result = benchmark_diffusion(
        model=model,
        mode="ddpm",
        num_inference_steps=ddpm_steps,
        num_runs=ddpm_runs,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )
    results["ddpm"] = ddpm_result
    print(ddpm_result.summary())

    # 4. PyTorch comparison (optional)
    if compare_pytorch:
        print("\n--- PyTorch Comparison ---")
        pt_result = benchmark_pytorch_forward(
            num_runs=num_runs,
            action_dim=action_dim,
            obs_dim=obs_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
        )
        if pt_result is not None:
            results["pytorch_forward"] = pt_result
            print(pt_result.summary())
            speedup = pt_result.mean_ms / fwd_result.mean_ms
            print(f"\n  MLX/PyTorch speedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Diffusion Policy inference on Apple Silicon"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to converted checkpoint directory",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps (default: 10)",
    )
    parser.add_argument(
        "--ddpm-steps",
        type=int,
        default=100,
        help="Number of DDPM inference steps (default: 100)",
    )
    parser.add_argument(
        "--compare-pytorch",
        action="store_true",
        help="Run PyTorch comparison benchmark",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_benchmark(
        checkpoint_dir=args.checkpoint,
        num_runs=args.num_runs,
        ddim_steps=args.ddim_steps,
        ddpm_steps=args.ddpm_steps,
        compare_pytorch=args.compare_pytorch,
    )


if __name__ == "__main__":
    main()
