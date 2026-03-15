"""Sinusoidal positional embedding for diffusion timesteps.

Upstream: diffusion_policy/model/diffusion/positional_embedding.py
"""

import math

import mlx.core as mx
import mlx.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Input:  (B,) integer or float timesteps
    Output: (B, dim) float embeddings

    Uses the standard transformer sin/cos encoding with base 10000.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = x[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb
