"""Tensor operation helpers bridging PyTorch conventions to MLX.

Provides thin wrappers for torch.* calls used in upstream diffusion_policy
so that ported code can use MLX arrays with minimal changes.
"""

from typing import Optional, Sequence, Union

import mlx.core as mx


def is_tensor(x) -> bool:
    """Drop-in for ``torch.is_tensor(x)``."""
    return isinstance(x, mx.array)


def tensor_to_long(x: mx.array) -> mx.array:
    """``x.long()`` — cast to int32 (Metal has no int64)."""
    return x.astype(mx.int32)


def tensor_to_float(x: mx.array) -> mx.array:
    """``x.float()`` — cast to float32."""
    return x.astype(mx.float32)


def expand_as_batch(timesteps: mx.array, batch_size: int) -> mx.array:
    """``timesteps.expand(batch_size)`` — broadcast scalar/1-D to batch dim."""
    if timesteps.ndim == 0:
        return mx.broadcast_to(timesteps, (batch_size,))
    if timesteps.shape[0] == 1:
        return mx.broadcast_to(timesteps, (batch_size,))
    return timesteps


def clamp(
    x: mx.array,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> mx.array:
    """``torch.clamp`` → ``mx.clip``."""
    if min_val is None and max_val is None:
        return x
    return mx.clip(x, min_val, max_val)


def zeros_like(x: mx.array) -> mx.array:
    """``torch.zeros_like(x)``."""
    return mx.zeros_like(x)


def ones_like(x: mx.array) -> mx.array:
    """``torch.ones_like(x)``."""
    return mx.ones_like(x)


def randn_like(x: mx.array) -> mx.array:
    """``torch.randn_like(x)`` — normal samples with same shape/dtype."""
    return mx.random.normal(x.shape, dtype=x.dtype)


def unsqueeze(x: mx.array, dim: int) -> mx.array:
    """``x.unsqueeze(dim)``."""
    return mx.expand_dims(x, axis=dim)


def squeeze(x: mx.array, dim: Optional[int] = None) -> mx.array:
    """``x.squeeze(dim)``."""
    if dim is None:
        return mx.squeeze(x)
    return mx.squeeze(x, axis=dim)


def flatten(x: mx.array, start_dim: int = 0, end_dim: int = -1) -> mx.array:
    """``x.flatten(start_dim, end_dim)``."""
    shape = x.shape
    ndim = len(shape)
    # Normalise negative dims
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    new_shape = (
        list(shape[:start_dim])
        + [-1]
        + list(shape[end_dim + 1 :])
    )
    return x.reshape(new_shape)


def detach(x: mx.array) -> mx.array:
    """``x.detach()`` — MLX uses ``mx.stop_gradient``."""
    return mx.stop_gradient(x)


def cat(arrays: Sequence[mx.array], dim: int = 0) -> mx.array:
    """``torch.cat(tensors, dim)``."""
    return mx.concatenate(arrays, axis=dim)


def stack(arrays: Sequence[mx.array], dim: int = 0) -> mx.array:
    """``torch.stack(tensors, dim)``."""
    return mx.stack(arrays, axis=dim)
