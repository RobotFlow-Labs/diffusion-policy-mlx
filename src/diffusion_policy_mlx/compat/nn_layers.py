"""Neural-network layer wrappers with automatic layout translation.

PyTorch uses channels-first (NCL / NCHW).  MLX uses channels-last (NLC / NHWC).
The wrappers here accept the PyTorch convention on input/output while using
MLX operations internally.
"""

from typing import Union

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Conv1d  (NCL ↔ NLC)
# ---------------------------------------------------------------------------


class Conv1d(nn.Module):
    """Drop-in for ``torch.nn.Conv1d`` with an NCL interface.

    *   Caller sends ``(B, C_in, L)``, gets back ``(B, C_out, L')``.
    *   Internally transposes to ``(B, L, C_in)`` for ``mlx.nn.Conv1d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # Store for inspection / debugging
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        # (B, C, L) → (B, L, C)
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        # (B, L', C') → (B, C', L')
        x = mx.transpose(x, axes=(0, 2, 1))
        return x


# ---------------------------------------------------------------------------
# ConvTranspose1d  (NCL ↔ NLC)
# ---------------------------------------------------------------------------


class ConvTranspose1d(nn.Module):
    """Drop-in for ``torch.nn.ConvTranspose1d`` with an NCL interface."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        # (B, C, L) → (B, L, C)
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        # (B, L', C') → (B, C', L')
        x = mx.transpose(x, axes=(0, 2, 1))
        return x


# ---------------------------------------------------------------------------
# Conv2d  (NCHW ↔ NHWC)
# ---------------------------------------------------------------------------


class Conv2d(nn.Module):
    """Drop-in for ``torch.nn.Conv2d`` with an NCHW interface.

    *   Caller sends ``(B, C, H, W)``, gets back ``(B, C', H', W')``.
    *   Internally converts to ``(B, H, W, C)`` for ``mlx.nn.Conv2d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.groups = groups

    def __call__(self, x: mx.array) -> mx.array:
        # (B, C, H, W) → (B, H, W, C)
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self._conv(x)
        # (B, H', W', C') → (B, C', H', W')
        x = mx.transpose(x, axes=(0, 3, 1, 2))
        return x


# ---------------------------------------------------------------------------
# GroupNorm  (channels-first wrapper)
# ---------------------------------------------------------------------------


class GroupNorm(nn.Module):
    """``torch.nn.GroupNorm`` — channels-first wrapper around MLX GroupNorm.

    Upstream sends ``(B, C, *)`` (channels-first).  MLX GroupNorm expects
    channels-last.  We transpose in and out.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self._gn = nn.GroupNorm(
            num_groups,
            num_channels,
            eps=eps,
            affine=affine,
            pytorch_compatible=True,
        )
        self.num_groups = num_groups
        self.num_channels = num_channels

    def __call__(self, x: mx.array) -> mx.array:
        ndim = x.ndim
        if ndim == 3:
            # (B, C, L) → (B, L, C)
            x = mx.transpose(x, axes=(0, 2, 1))
            x = self._gn(x)
            x = mx.transpose(x, axes=(0, 2, 1))
            return x
        if ndim == 4:
            # (B, C, H, W) → (B, H, W, C)
            x = mx.transpose(x, axes=(0, 2, 3, 1))
            x = self._gn(x)
            x = mx.transpose(x, axes=(0, 3, 1, 2))
            return x
        raise ValueError(f"GroupNorm wrapper expects 3-D or 4-D input, got {ndim}-D")


# ---------------------------------------------------------------------------
# BatchNorm2d  (channels-first wrapper)
# ---------------------------------------------------------------------------


class BatchNorm2d(nn.Module):
    """``torch.nn.BatchNorm2d`` — channels-first wrapper around MLX BatchNorm."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self._bn = nn.BatchNorm(num_features, eps=eps, momentum=momentum, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        # (B, C, H, W) → (B, H, W, C)
        x = mx.transpose(x, axes=(0, 2, 3, 1))
        x = self._bn(x)
        x = mx.transpose(x, axes=(0, 3, 1, 2))
        return x


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


class Sequential(nn.Module):
    """``torch.nn.Sequential`` — sequential module composition."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
            # After first layer, don't forward extra args
            args = ()
            kwargs = {}
        return x


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Identity(nn.Module):
    """``torch.nn.Identity``."""

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return x


# ---------------------------------------------------------------------------
# Re-exports of MLX layers that need no wrapping
# ---------------------------------------------------------------------------

Linear = nn.Linear
Dropout = nn.Dropout
Embedding = nn.Embedding
Mish = nn.Mish
SiLU = nn.SiLU
