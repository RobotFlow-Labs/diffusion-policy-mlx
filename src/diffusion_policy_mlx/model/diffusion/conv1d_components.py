"""1D convolution components for diffusion UNet.

Upstream: diffusion_policy/model/diffusion/conv1d_components.py

All public classes operate on NCL (batch, channels, length) convention,
matching the PyTorch upstream interface. Internal MLX ops use NLC
(batch, length, channels), so wrappers handle the transpose.
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# NCL wrappers — MLX native layers expect NLC (channels-last)
# ---------------------------------------------------------------------------

class _Conv1d(nn.Module):
    """Conv1d with NCL (channels-first) interface.

    MLX's nn.Conv1d expects NLC input. This wrapper transposes
    NCL -> NLC before the conv and NLC -> NCL after.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self._conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, L) -> (B, L, C)
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        # (B, L', C') -> (B, C', L')
        return mx.transpose(x, axes=(0, 2, 1))


class _ConvTranspose1d(nn.Module):
    """ConvTranspose1d with NCL (channels-first) interface.

    MLX's nn.ConvTranspose1d expects NLC input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self._conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, L) -> (B, L, C)
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._conv(x)
        # (B, L', C') -> (B, C', L')
        return mx.transpose(x, axes=(0, 2, 1))


class _GroupNorm(nn.Module):
    """GroupNorm with NCL (channels-first) interface.

    MLX's nn.GroupNorm normalises over the last axis (channels-last).
    This wrapper transposes NCL -> NLC, applies GroupNorm, then
    transposes back.
    """

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self._norm = nn.GroupNorm(num_groups, num_channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, L) -> (B, L, C)
        x = mx.transpose(x, axes=(0, 2, 1))
        x = self._norm(x)
        # (B, L, C) -> (B, C, L)
        return mx.transpose(x, axes=(0, 2, 1))


# ---------------------------------------------------------------------------
# Identity — drop-in replacement for nn.Identity on the up/down paths
# ---------------------------------------------------------------------------

class _Identity(nn.Module):
    """Identity module (no-op), used where upstream uses nn.Identity()."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


# ---------------------------------------------------------------------------
# Public components (NCL interface)
# ---------------------------------------------------------------------------

class Downsample1d(nn.Module):
    """Strided Conv1d for 2x spatial downsampling.

    Input:  (B, C, L)
    Output: (B, C, L // 2)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = _Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample1d(nn.Module):
    """ConvTranspose1d for 2x spatial upsampling.

    Input:  (B, C, L)
    Output: (B, C, 2*L)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = _ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish.

    Input:  (B, C_in, L)
    Output: (B, C_out, L)   (padding preserves spatial dim)
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.conv = _Conv1d(
            inp_channels, out_channels, kernel_size, padding=kernel_size // 2,
        )
        self.group_norm = _GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.mish(x)
        return x
