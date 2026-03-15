"""Einops rearrange replacements for the 3 patterns used in upstream.

Only the exact patterns found in ``conditional_unet1d.py`` and
``conv1d_components.py`` are provided — no general-purpose parser.
"""

import mlx.core as mx


def rearrange_b_h_t_to_b_t_h(x: mx.array) -> mx.array:
    """``einops.rearrange(x, 'b h t -> b t h')`` — transpose last two dims."""
    return mx.transpose(x, axes=(0, 2, 1))


def rearrange_b_t_h_to_b_h_t(x: mx.array) -> mx.array:
    """``einops.rearrange(x, 'b t h -> b h t')`` — transpose last two dims."""
    return mx.transpose(x, axes=(0, 2, 1))


def rearrange_batch_t_to_batch_t_1(x: mx.array) -> mx.array:
    """``Rearrange('batch t -> batch t 1')`` — used in FiLM cond_encoder."""
    return mx.expand_dims(x, axis=-1)
