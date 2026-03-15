"""Functional operations mirroring ``torch.nn.functional`` patterns.

Only the subset actually used by upstream diffusion_policy is implemented.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def mish(x: mx.array) -> mx.array:
    """``F.mish`` activation:  x * tanh(softplus(x)).

    Uses thresholded softplus to avoid float32 overflow for x > ~88.
    """
    softplus = mx.where(x > 20.0, x, mx.log(1.0 + mx.exp(x)))
    return x * mx.tanh(softplus)


def silu(x: mx.array) -> mx.array:
    """``F.silu`` / swish activation:  x * sigmoid(x)."""
    return x * mx.sigmoid(x)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def pad_1d(
    x: mx.array,
    padding: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> mx.array:
    """``F.pad`` for 3-D tensors ``(B, C, L)``.

    Parameters
    ----------
    x : mx.array
        Input of shape ``(B, C, L)``.
    padding : tuple[int, int]
        ``(pad_left, pad_right)`` applied to the last dimension.
    mode : str
        ``'constant'`` or ``'replicate'``.
    value : float
        Fill value when *mode* is ``'constant'``.
    """
    pad_left, pad_right = padding

    if mode == "constant":
        return mx.pad(
            x,
            pad_width=[(0, 0), (0, 0), (pad_left, pad_right)],
            constant_values=value,
        )

    if mode == "replicate":
        parts = []
        if pad_left > 0:
            left_edge = x[:, :, :1]  # (B, C, 1)
            parts.append(mx.broadcast_to(left_edge, (x.shape[0], x.shape[1], pad_left)))
        parts.append(x)
        if pad_right > 0:
            right_edge = x[:, :, -1:]  # (B, C, 1)
            parts.append(mx.broadcast_to(right_edge, (x.shape[0], x.shape[1], pad_right)))
        return mx.concatenate(parts, axis=2)

    raise ValueError(f"pad_1d: unsupported mode '{mode}'")


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate_1d(
    x: mx.array,
    scale_factor: Optional[Union[int, float]] = None,
    size: Optional[int] = None,
    mode: str = "nearest",
) -> mx.array:
    """``F.interpolate`` for 1-D signals.  x: ``(B, C, L)``.

    Only ``mode='nearest'`` is implemented (used by Upsample1d fallback).
    """
    if mode != "nearest":
        raise NotImplementedError(f"interpolate_1d: mode '{mode}' not implemented")

    B, C, L = x.shape

    if size is not None:
        target_len = size
    elif scale_factor is not None:
        target_len = int(L * scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be provided")

    # Nearest-neighbour via index repeat
    indices = mx.arange(target_len)
    src_indices = mx.floor(indices.astype(mx.float32) * L / target_len).astype(mx.int32)

    # Transpose to (B, L, C), index, transpose back
    x_nlc = mx.transpose(x, axes=(0, 2, 1))  # (B, L, C)
    out_nlc = x_nlc[:, src_indices, :]  # (B, target_len, C)
    return mx.transpose(out_nlc, axes=(0, 2, 1))  # (B, C, target_len)
