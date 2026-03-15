"""1D convolution components for diffusion UNet.

Upstream: diffusion_policy/model/diffusion/conv1d_components.py

All public classes operate on NCL (batch, channels, length) convention,
matching the PyTorch upstream interface. Internal MLX ops use NLC
(batch, length, channels), so wrappers handle the transpose.
"""

import mlx.core as mx
import mlx.nn as nn

from diffusion_policy_mlx.compat.nn_layers import (
    Conv1d as _Conv1d,
)
from diffusion_policy_mlx.compat.nn_layers import (
    ConvTranspose1d as _ConvTranspose1d,
)
from diffusion_policy_mlx.compat.nn_layers import (
    GroupNorm as _GroupNorm,
)

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
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.group_norm = _GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.mish(x)
        return x
