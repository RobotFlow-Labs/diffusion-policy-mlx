"""Collation utility for batching dataset samples into mx.arrays.

Usage::

    from diffusion_policy_mlx.training.collate import collate_batch

    samples = [dataset[i] for i in batch_indices]
    batch = collate_batch(samples)
    # batch['obs']['image'] is mx.array of shape (B, T, C, H, W)
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import mlx.core as mx
import numpy as np


def collate_batch(
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stack a list of dataset samples into a batched dict of ``mx.array``.

    Each sample is a (possibly nested) dict whose leaves are numpy
    arrays.  All arrays with the same key path are stacked along a new
    leading batch dimension and converted to ``mx.array``.

    Parameters
    ----------
    samples : list[dict]
        Output of ``dataset[i]`` calls.

    Returns
    -------
    dict
        Same structure as a single sample but with an extra leading
        batch dimension and all leaves as ``mx.array``.
    """
    if len(samples) == 0:
        return {}
    return _collate_recursive(samples)


def _collate_recursive(
    items: Sequence[Any],
) -> Any:
    """Recursively collate a list of identically-structured objects."""
    first = items[0]

    if isinstance(first, dict):
        result: Dict[str, Any] = {}
        for key in first:
            result[key] = _collate_recursive([item[key] for item in items])
        return result

    if isinstance(first, np.ndarray):
        stacked = np.stack(items, axis=0)
        return mx.array(stacked)

    if isinstance(first, mx.array):
        # Already mx — stack via numpy round-trip for safety
        stacked = np.stack([np.asarray(item) for item in items], axis=0)
        return mx.array(stacked)

    # Scalar or other — return as list
    return items
