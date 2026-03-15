"""Metal GPU utilities for Apple Silicon.

Provides helpers for querying Metal device status and memory usage.
Useful for debugging performance issues, monitoring GPU memory during
training, and verifying that MLX is running on the GPU.

Usage::

    from diffusion_policy_mlx.common.metal_utils import print_metal_status
    print_metal_status()
"""

from __future__ import annotations

from typing import Dict

import mlx.core as mx


def get_metal_info() -> Dict[str, object]:
    """Return Metal device info and memory statistics.

    Returns:
        Dictionary with keys:
          - ``device``: Current default device string (e.g. ``"Device(gpu, 0)"``)
          - ``metal_available``: Whether Metal GPU is available
          - ``active_memory_gb``: Currently allocated GPU memory in GB
          - ``peak_memory_gb``: Peak GPU memory usage in GB
          - ``cache_memory_gb``: Cached (recyclable) GPU memory in GB
    """
    return {
        "device": str(mx.default_device()),
        "metal_available": mx.metal.is_available(),
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "cache_memory_gb": mx.get_cache_memory() / 1e9,
    }


def reset_peak_memory() -> None:
    """Reset peak memory counter for a fresh measurement window.

    Call this before a section of code you want to profile, then
    read ``mx.get_peak_memory()`` after to get the peak for that section.
    """
    mx.reset_peak_memory()


def log_memory(label: str = "") -> None:
    """Log current GPU memory stats to stdout.

    Args:
        label: Optional label prefix for the log line.
    """
    info = get_metal_info()
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}GPU mem: "
        f"active={info['active_memory_gb']:.3f}GB  "
        f"peak={info['peak_memory_gb']:.3f}GB  "
        f"cache={info['cache_memory_gb']:.3f}GB"
    )


def print_metal_status() -> None:
    """Print Metal GPU status summary (useful for debugging)."""
    info = get_metal_info()
    print(f"Device: {info['device']}")
    print(f"Metal: {'available' if info['metal_available'] else 'NOT available'}")
    print(f"Active memory: {info['active_memory_gb']:.2f} GB")
    print(f"Peak memory:   {info['peak_memory_gb']:.2f} GB")
    print(f"Cache memory:  {info['cache_memory_gb']:.2f} GB")
