"""Tests for the inference benchmark module.

Tests cover:
  - Benchmark function runs without error on tiny model
  - Memory reporting works (mx.get_peak_memory returns >= 0)
  - Latency measurement is reasonable (> 0, < 60s for tiny model)
  - BenchmarkResult dataclass properties
  - Single forward benchmark
  - Full diffusion loop benchmark (DDIM, small steps)
"""

import sys
import os

import numpy as np
import pytest

import mlx.core as mx

# Ensure scripts/ is importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "scripts"
    ),
)

from benchmark import (
    BenchmarkResult,
    benchmark_diffusion,
    benchmark_single_forward,
    create_dummy_inputs,
    create_model,
    run_single_inference,
)


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_percentiles(self):
        result = BenchmarkResult(
            mode="test",
            num_runs=10,
            num_diffusion_steps=1,
            latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        assert result.p50_ms == pytest.approx(5.5, abs=0.1)
        assert result.p95_ms > result.p50_ms
        assert result.p99_ms >= result.p95_ms

    def test_throughput(self):
        result = BenchmarkResult(
            mode="test",
            num_runs=5,
            num_diffusion_steps=1,
            latencies_ms=[10.0, 10.0, 10.0, 10.0, 10.0],
        )
        # 10ms mean -> 100 inferences/sec
        assert result.throughput == pytest.approx(100.0, rel=0.01)

    def test_mean(self):
        result = BenchmarkResult(
            mode="test",
            num_runs=3,
            num_diffusion_steps=1,
            latencies_ms=[10.0, 20.0, 30.0],
        )
        assert result.mean_ms == pytest.approx(20.0)

    def test_summary_string(self):
        result = BenchmarkResult(
            mode="ddim",
            num_runs=5,
            num_diffusion_steps=10,
            latencies_ms=[5.0, 5.0, 5.0, 5.0, 5.0],
        )
        summary = result.summary()
        assert "DDIM" in summary
        assert "10 steps" in summary
        assert "Latency p50" in summary
        assert "Throughput" in summary


# ---------------------------------------------------------------------------
# Model creation tests
# ---------------------------------------------------------------------------


class TestModelCreation:
    """Test model and input creation utilities."""

    def test_create_model(self):
        model = create_model(action_dim=2, obs_dim=64, n_obs_steps=2)
        assert model is not None

    def test_create_dummy_inputs(self):
        inputs = create_dummy_inputs(
            batch_size=2, action_dim=2, obs_dim=64, horizon=8, n_obs_steps=2
        )
        assert "sample" in inputs
        assert "global_cond" in inputs
        assert inputs["sample"].shape == (2, 8, 2)
        assert inputs["global_cond"].shape == (2, 128)  # 64 * 2


# ---------------------------------------------------------------------------
# Single inference tests
# ---------------------------------------------------------------------------


class TestSingleInference:
    """Test single forward pass."""

    def test_runs_without_error(self):
        """Single inference should complete without error."""
        model = create_model(
            action_dim=2,
            obs_dim=32,
            n_obs_steps=2,
            down_dims=(32, 64),
            diffusion_step_embed_dim=32,
        )
        inputs = create_dummy_inputs(
            action_dim=2, obs_dim=32, horizon=8, n_obs_steps=2
        )
        result = run_single_inference(
            model, inputs["sample"], 0, inputs["global_cond"]
        )
        assert result.shape == inputs["sample"].shape
        # Check no NaN
        assert not mx.any(mx.isnan(result)).item()

    def test_output_shape(self):
        """Output should match input shape."""
        model = create_model(
            action_dim=4,
            obs_dim=16,
            n_obs_steps=1,
            down_dims=(16, 32),
            diffusion_step_embed_dim=16,
        )
        inputs = create_dummy_inputs(
            action_dim=4, obs_dim=16, horizon=4, n_obs_steps=1
        )
        result = run_single_inference(
            model, inputs["sample"], 5, inputs["global_cond"]
        )
        assert result.shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# Benchmark function tests
# ---------------------------------------------------------------------------


class TestBenchmarkSingleForward:
    """Test benchmark_single_forward function."""

    def test_runs_and_returns_result(self):
        """Benchmark should complete and return valid results."""
        result = benchmark_single_forward(
            num_warmup=1,
            num_runs=3,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.num_runs == 3
        assert len(result.latencies_ms) == 3

    def test_latencies_positive(self):
        """All latencies should be positive."""
        result = benchmark_single_forward(
            num_warmup=1,
            num_runs=5,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert all(l > 0 for l in result.latencies_ms)

    def test_latencies_reasonable(self):
        """Latencies for a tiny model should be under 60 seconds."""
        result = benchmark_single_forward(
            num_warmup=1,
            num_runs=3,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert all(l < 60000 for l in result.latencies_ms)  # < 60s


class TestBenchmarkDiffusion:
    """Test full diffusion loop benchmark."""

    def test_ddim_runs(self):
        """DDIM benchmark should complete."""
        result = benchmark_diffusion(
            mode="ddim",
            num_inference_steps=2,
            num_warmup=1,
            num_runs=2,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.mode == "ddim"
        assert result.num_diffusion_steps == 2
        assert len(result.latencies_ms) == 2

    def test_ddpm_runs(self):
        """DDPM benchmark should complete."""
        result = benchmark_diffusion(
            mode="ddpm",
            num_inference_steps=3,
            num_warmup=1,
            num_runs=2,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.mode == "ddpm"

    def test_ddim_faster_than_ddpm(self):
        """DDIM with fewer steps should generally be faster than DDPM."""
        ddim = benchmark_diffusion(
            mode="ddim",
            num_inference_steps=2,
            num_warmup=1,
            num_runs=3,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        ddpm = benchmark_diffusion(
            mode="ddpm",
            num_inference_steps=10,
            num_warmup=1,
            num_runs=3,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        # DDIM with 2 steps should be faster than DDPM with 10
        assert ddim.mean_ms < ddpm.mean_ms


# ---------------------------------------------------------------------------
# Memory reporting tests
# ---------------------------------------------------------------------------


class TestMemoryReporting:
    """Test GPU memory reporting."""

    def test_peak_memory_non_negative(self):
        """mx.get_peak_memory should return a non-negative value."""
        # Run a small computation to ensure memory is allocated
        x = mx.random.normal((100, 100))
        y = x @ x.T
        mx.eval(y)

        mem = mx.get_peak_memory()
        assert mem >= 0

    def test_peak_memory_after_benchmark(self):
        """Peak memory should be reported in benchmark results."""
        result = benchmark_single_forward(
            num_warmup=1,
            num_runs=2,
            action_dim=2,
            obs_dim=32,
            horizon=8,
            n_obs_steps=2,
        )
        assert result.peak_memory_gb >= 0
