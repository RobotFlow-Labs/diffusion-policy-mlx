"""Transformer denoiser for diffusion policy.

Upstream: diffusion_policy/model/diffusion/transformer_for_diffusion.py

Architecture:
  - Input embedding (Linear) + learnable position embeddings
  - Sinusoidal timestep embedding
  - Optional observation conditioning (obs_as_cond)
  - Two modes:
    1. Encoder-only (BERT-style): timestep token prepended to sequence
    2. Encoder-decoder: conditioning tokens (time + obs) processed by encoder,
       action tokens decoded with cross-attention
  - Causal attention mask support
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from diffusion_policy_mlx.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Transformer building blocks
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (self-attention + FFN).

    Matches PyTorch nn.TransformerEncoderLayer with:
      - norm_first=True
      - activation='gelu'
      - batch_first=True
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def __call__(self, src: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            src: (B, T, d_model)
            mask: (T, T) additive attention mask or None
        Returns:
            (B, T, d_model)
        """
        # Pre-norm self-attention
        x = self.norm1(src)
        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout1(x)
        src = src + x

        # Pre-norm FFN
        x = self.norm2(src)
        x = nn.gelu(self.linear1(x))
        x = self.dropout2(self.linear2(x))
        src = src + x

        return src


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer (self-attn + cross-attn + FFN).

    Matches PyTorch nn.TransformerDecoderLayer with:
      - norm_first=True
      - activation='gelu'
      - batch_first=True
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead)
        self.cross_attn = nn.MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            tgt: (B, T, d_model) target (action) sequence
            memory: (B, S, d_model) encoder output (conditioning)
            tgt_mask: (T, T) additive causal mask for self-attention
            memory_mask: (T, S) additive mask for cross-attention
        Returns:
            (B, T, d_model)
        """
        # Pre-norm self-attention
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.dropout1(x)
        tgt = tgt + x

        # Pre-norm cross-attention
        x = self.norm2(tgt)
        x = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.dropout2(x)
        tgt = tgt + x

        # Pre-norm FFN
        x = self.norm3(tgt)
        x = nn.gelu(self.linear1(x))
        x = self.dropout3(self.linear2(x))
        tgt = tgt + x

        return tgt


# ---------------------------------------------------------------------------
# TransformerForDiffusion
# ---------------------------------------------------------------------------


class TransformerForDiffusion(nn.Module):
    """Transformer-based denoiser for diffusion policy.

    Supports two architectures:
      1. **Encoder-decoder** (default when time_as_cond=True): Conditioning
         tokens (timestep + optional obs) are processed by an encoder, and
         the noisy action sequence is decoded with cross-attention.
      2. **Encoder-only** (BERT-style, when time_as_cond=False and no obs cond):
         Timestep token is prepended to the action sequence; a standard
         Transformer encoder processes the concatenated sequence.

    Args:
        input_dim: Action/input dimension per timestep.
        output_dim: Output dimension per timestep.
        horizon: Number of action timesteps (T).
        n_obs_steps: Number of observation timesteps (defaults to horizon).
        cond_dim: Observation conditioning dimension (0 = no obs cond).
        n_layer: Number of decoder (or encoder-only) layers.
        n_head: Number of attention heads.
        n_emb: Embedding dimension.
        p_drop_emb: Dropout probability for embeddings.
        p_drop_attn: Dropout probability for attention/FFN layers.
        causal_attn: Whether to use causal (autoregressive) attention.
        time_as_cond: Whether to pass timestep as a conditioning token
                      (True) vs prepend to the action sequence (False).
        obs_as_cond: Whether to use observation conditioning (auto-set
                     from cond_dim > 0).
        n_cond_layers: Number of encoder layers for conditioning. If 0,
                       uses a simple MLP instead.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: Optional[int] = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        # Compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1  # at least the timestep token
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond, "obs_as_cond requires time_as_cond=True"
            T_cond += n_obs_steps

        # Input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = mx.zeros((1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Conditioning encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = mx.zeros((1, T_cond, n_emb))
            if n_cond_layers > 0:
                self.encoder = [
                    TransformerEncoderLayer(
                        d_model=n_emb,
                        nhead=n_head,
                        dim_feedforward=4 * n_emb,
                        dropout=p_drop_attn,
                    )
                    for _ in range(n_cond_layers)
                ]
            else:
                self.encoder_linear1 = nn.Linear(n_emb, 4 * n_emb)
                self.encoder_mish = nn.Mish()
                self.encoder_linear2 = nn.Linear(4 * n_emb, n_emb)

            # Decoder
            self.decoder = [
                TransformerDecoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                )
                for _ in range(n_layer)
            ]
        else:
            # Encoder-only BERT
            encoder_only = True
            self.encoder = [
                TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                )
                for _ in range(n_layer)
            ]

        # Attention masks
        if causal_attn:
            # Causal mask: upper triangle = -inf, diagonal and below = 0
            # MLX MultiHeadAttention additive mask convention
            sz = T
            # Create upper triangular mask with -inf
            mask = mx.where(
                mx.triu(mx.ones((sz, sz)), k=1).astype(mx.bool_),
                mx.full((sz, sz), float("-inf")),
                mx.zeros((sz, sz)),
            )
            self._mask = mask

            if time_as_cond and obs_as_cond:
                S = T_cond
                # memory_mask[i, j]: target position i attending to memory position j
                # t >= (s - 1): allow attending to time token (s=0) always,
                # and obs token s only if target position t >= s-1
                import numpy as np
                t_idx, s_idx = np.meshgrid(np.arange(T), np.arange(S), indexing="ij")
                allowed = t_idx >= (s_idx - 1)  # True means allowed
                mem_mask_np = np.where(allowed, 0.0, float("-inf")).astype(np.float32)
                self._memory_mask = mx.array(mem_mask_np)
            else:
                self._memory_mask = None
        else:
            self._mask = None
            self._memory_mask = None

        # Decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # Constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only
        self.n_cond_layers = n_cond_layers if T_cond > 0 else -1

        # Initialize weights
        self._init_weights()

        # Count parameters
        n_params = sum(p.size for _, p in self.parameters().items() if isinstance(p, mx.array))
        logger.info("number of parameters: %e", n_params)

    def _init_weights(self):
        """Initialize weights following upstream GPT-style initialization."""
        # Initialize pos_emb with small normal values
        self.pos_emb = mx.random.normal(self.pos_emb.shape) * 0.02
        if self.cond_pos_emb is not None:
            self.cond_pos_emb = mx.random.normal(self.cond_pos_emb.shape) * 0.02

    def __call__(
        self,
        sample: mx.array,
        timestep: Union[mx.array, float, int],
        cond: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Forward pass.

        Args:
            sample: (B, T, input_dim) noisy action trajectory.
            timestep: (B,) or scalar diffusion timestep.
            cond: (B, T', cond_dim) observation conditioning (optional).

        Returns:
            (B, T, output_dim) predicted noise or sample.
        """
        # 1. Timestep embedding
        timesteps = timestep
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array([timesteps], dtype=mx.int32)
        elif timesteps.ndim == 0:
            timesteps = mx.expand_dims(timesteps, axis=0)
        # Broadcast to batch dimension
        timesteps = mx.broadcast_to(timesteps, (sample.shape[0],))
        time_emb = mx.expand_dims(self.time_emb(timesteps), axis=1)
        # (B, 1, n_emb)

        # Process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT-style: prepend time token
            token_embeddings = mx.concatenate([time_emb, input_emb], axis=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            # (B, T+1, n_emb)
            for layer in self.encoder:
                x = layer(x, mask=self._mask)
            # (B, T+1, n_emb) -> remove time token
            x = x[:, 1:, :]
            # (B, T, n_emb)
        else:
            # Encoder-decoder mode
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B, To, n_emb)
                cond_embeddings = mx.concatenate([cond_embeddings, cond_obs_emb], axis=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[:, :tc, :]
            x = self.drop(cond_embeddings + position_embeddings)

            # Encode conditioning
            if self.n_cond_layers > 0:
                for layer in self.encoder:
                    x = layer(x)
            else:
                x = self.encoder_linear1(x)
                x = self.encoder_mish(x)
                x = self.encoder_linear2(x)
            memory = x
            # (B, T_cond, n_emb)

            # Decode action sequence
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + position_embeddings)
            # (B, T, n_emb)
            for layer in self.decoder:
                x = layer(
                    x,
                    memory,
                    tgt_mask=self._mask,
                    memory_mask=self._memory_mask,
                )
            # (B, T, n_emb)

        # Head
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, output_dim)
        return x
