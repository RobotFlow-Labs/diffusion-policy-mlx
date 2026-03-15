"""MLX equivalents of upstream pytorch_util functions.

Upstream: diffusion_policy/common/pytorch_util.py

Provides API-compatible replacements for the subset of pytorch_util that
is used across the codebase. Functions that don't apply to MLX (like
``optimizer_to``) are kept as no-ops for compatibility.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

# Re-export dict_apply from dict_util for API compatibility with upstream
from diffusion_policy_mlx.common.dict_util import dict_apply  # noqa: F401


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace all submodules matching *predicate* with the result of *func*.

    Re-exported from ``compat.vision`` where the full implementation lives.
    This wrapper exists so callers can import from ``common.pytorch_util``
    just like in upstream code.

    Args:
        root_module: The root module to modify in-place.
        predicate: Returns True if a submodule should be replaced.
        func: Given a matching submodule, returns its replacement.

    Returns:
        The root module (possibly replaced if it matches predicate itself).
    """
    from diffusion_policy_mlx.compat.vision import (
        replace_submodules as _replace_submodules,
    )

    return _replace_submodules(root_module, predicate, func)


def param_count(model: nn.Module) -> int:
    """Count the total number of scalar parameters in an MLX module.

    Example::

        >>> import mlx.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> param_count(linear)  # 10*5 + 5 = 55
        55

    Args:
        model: An MLX module.

    Returns:
        Total number of scalar parameters (sum of all parameter array sizes).
    """
    total = 0
    for _, v in mlx.utils.tree_flatten(model.parameters()):
        total += v.size
    return total


def optimizer_to(optimizer: Any, device: Any) -> Any:
    """No-op in MLX — there is no device management.

    Kept for API compatibility with upstream code that calls
    ``optimizer_to(optimizer, 'cuda')`` etc.

    Args:
        optimizer: Any optimizer instance.
        device: Ignored.

    Returns:
        The optimizer, unchanged.
    """
    return optimizer


def pad_remaining_dims(x: mx.array, target: mx.array) -> mx.array:
    """Pad *x* with trailing size-1 dimensions to match *target*'s ndim.

    Upstream: ``pytorch_util.pad_remaining_dims``

    Used when broadcasting a per-sample scalar (e.g., noise schedule
    coefficients) to match a higher-dimensional tensor shape.

    Args:
        x: Array whose shape is a prefix of target's shape.
        target: Array whose ndim determines the desired output ndim.

    Returns:
        *x* reshaped with trailing dimensions of size 1.
    """
    assert x.shape == target.shape[: len(x.shape)]
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))
