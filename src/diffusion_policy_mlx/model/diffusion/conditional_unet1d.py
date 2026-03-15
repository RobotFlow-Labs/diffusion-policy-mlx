"""Conditional 1D UNet for diffusion policy denoising.

Upstream: diffusion_policy/model/diffusion/conditional_unet1d.py

Architecture:
  Input  (B, T, input_dim)
    -> rearrange to (B, input_dim, T) = NCL for conv blocks
    -> SinusoidalPosEmb + MLP for timestep encoding
    -> Concat global_cond if provided
    -> Down blocks: [ResBlock, ResBlock, Downsample] x N
    -> Mid blocks:  [ResBlock, ResBlock]
    -> Up blocks:   [cat skip, ResBlock, ResBlock, Upsample] x N
    -> Final Conv1d
  Output (B, T, input_dim)
"""

import logging
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from diffusion_policy_mlx.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
    _Conv1d,
    _Identity,
)
from diffusion_policy_mlx.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning.

    Input:
        x:    (B, in_channels, L)  -- NCL format
        cond: (B, cond_dim)        -- conditioning vector

    Output: (B, out_channels, L)

    FiLM modulation:
        if cond_predict_scale: out = scale * conv(x) + bias
        else:                  out = conv(x) + bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        self.blocks = [
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ]

        # FiLM modulation (https://arxiv.org/abs/1709.07871)
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        # cond_encoder: Mish -> Linear -> unsqueeze last dim
        self.cond_mish = nn.Mish()
        self.cond_linear = nn.Linear(cond_dim, cond_channels)

        # Residual projection (1x1 conv when channel dims differ, identity otherwise)
        self.residual_conv: Union[_Conv1d, _Identity]
        if in_channels != out_channels:
            self.residual_conv = _Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = _Identity()

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        """
        x:    (B, in_channels, L)
        cond: (B, cond_dim)
        returns: (B, out_channels, L)
        """
        out = self.blocks[0](x)

        # FiLM conditioning
        embed = self.cond_mish(cond)
        embed = self.cond_linear(embed)
        embed = mx.expand_dims(embed, axis=-1)  # (B, cond_channels, 1)

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)

        # Residual connection
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D UNet with conditional residual blocks for diffusion denoising.

    Constructor args:
        input_dim:                action dimension
        local_cond_dim:           optional per-timestep conditioning dim
        global_cond_dim:          optional global conditioning dim (e.g. obs features)
        diffusion_step_embed_dim: timestep embedding size (default 256)
        down_dims:                channel dims per level (default [256, 512, 1024])
        kernel_size:              conv kernel size (default 3)
        n_groups:                 GroupNorm groups (default 8)
        cond_predict_scale:       FiLM scale+bias vs bias-only (default False)
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim

        # Timestep encoder: SinPosEmb -> Linear -> Mish -> Linear
        self.diffusion_step_encoder_posemb = SinusoidalPosEmb(dsed)
        self.diffusion_step_encoder_linear1 = nn.Linear(dsed, dsed * 4)
        self.diffusion_step_encoder_mish = nn.Mish()
        self.diffusion_step_encoder_linear2 = nn.Linear(dsed * 4, dsed)

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Local cond encoder (optional)
        # WARNING: Due to a bug in the upstream implementation (line 233),
        # the local_cond up-path contribution is dead code (the condition
        # `idx == len(self.up_modules)` is always False). We preserve this
        # for checkpoint compatibility. The down-path contribution at idx=0
        # IS applied. See https://github.com/real-stanford/diffusion_policy
        self.local_cond_encoder: Optional[list] = None
        if local_cond_dim is not None:
            logger.warning(
                "local_cond_dim is set, but the up-path local conditioning "
                "is dead code (upstream bug preserved for checkpoint compat). "
                "Only the down-path contribution at idx=0 is applied."
            )
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = [
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in,
                    dim_out,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in,
                    dim_out,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]

        # Mid blocks
        mid_dim = all_dims[-1]
        self.mid_modules = [
            ConditionalResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
            ConditionalResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            ),
        ]

        # Down path
        self.down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        dim_out,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    Downsample1d(dim_out) if not is_last else _Identity(),
                ]
            )

        # Up path
        self.up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                [
                    ConditionalResidualBlock1D(
                        dim_out * 2,
                        dim_in,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_in,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    Upsample1d(dim_in) if not is_last else _Identity(),
                ]
            )

        # Final conv
        self.final_block = Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size)
        self.final_conv = _Conv1d(start_dim, input_dim, kernel_size=1)

    def __call__(
        self,
        sample: mx.array,
        timestep: Union[mx.array, float, int],
        local_cond: Optional[mx.array] = None,
        global_cond: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            sample:      (B, T, input_dim) action trajectory
            timestep:    (B,) or scalar diffusion step
            local_cond:  (B, T, local_cond_dim) or None
            global_cond: (B, global_cond_dim) or None

        Returns:
            (B, T, input_dim) denoised prediction
        """
        # Rearrange: (B, T, input_dim) -> (B, input_dim, T) = NCL
        # Upstream: einops.rearrange(sample, 'b h t -> b t h')
        # where input is (B, h=T, t=input_dim) -> (B, t=input_dim, h=T)
        sample = mx.transpose(sample, axes=(0, 2, 1))

        # 1. Timestep encoding
        timesteps = timestep
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array([timesteps], dtype=mx.int32)
        elif timesteps.ndim == 0:
            timesteps = mx.expand_dims(timesteps, axis=0)
        # Broadcast to batch dimension
        timesteps = mx.broadcast_to(timesteps, (sample.shape[0],))

        global_feature = self.diffusion_step_encoder_posemb(timesteps)
        global_feature = self.diffusion_step_encoder_linear1(global_feature)
        global_feature = self.diffusion_step_encoder_mish(global_feature)
        global_feature = self.diffusion_step_encoder_linear2(global_feature)

        if global_cond is not None:
            global_feature = mx.concatenate([global_feature, global_cond], axis=-1)

        # Encode local features
        h_local: list = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = mx.transpose(local_cond, axes=(0, 2, 1))
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        # Down path
        x = sample
        h: list = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Up path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = mx.concatenate([x, h.pop()], axis=1)  # skip connection on C dim
            x = resnet(x, global_feature)
            # BUG PRESERVED from upstream (line 233): this condition is always False.
            # The correct condition should be:
            #   if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
            # However changing it would break compatibility with published checkpoints.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_block(x)
        x = self.final_conv(x)

        # Rearrange back: NCL -> (B, T, input_dim)
        # Upstream: einops.rearrange(x, 'b t h -> b h t')
        x = mx.transpose(x, axes=(0, 2, 1))
        return x
