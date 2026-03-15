"""Compat layer — drop-in replacements for torch patterns used by upstream diffusion_policy.

Usage::

    from diffusion_policy_mlx.compat import Conv1d, GroupNorm, Sequential, is_tensor
"""

# -- tensor ops --
# -- einops helpers --
from diffusion_policy_mlx.compat.einops_mlx import (
    rearrange_b_h_t_to_b_t_h,
    rearrange_b_t_h_to_b_h_t,
    rearrange_batch_t_to_batch_t_1,
)

# -- functional --
from diffusion_policy_mlx.compat.functional import (
    interpolate_1d,
    mish,
    pad_1d,
    silu,
)

# -- nn layers --
from diffusion_policy_mlx.compat.nn_layers import (
    BatchNorm2d,
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    Dropout,
    Embedding,
    GroupNorm,
    Identity,
    Linear,
    Mish,
    Sequential,
    SiLU,
)

# -- nn modules (mixins) --
from diffusion_policy_mlx.compat.nn_modules import (
    DictOfTensorMixin,
    ModuleAttrMixin,
)
from diffusion_policy_mlx.compat.tensor_ops import (
    cat,
    clamp,
    detach,
    expand_as_batch,
    flatten,
    is_tensor,
    ones_like,
    randn_like,
    squeeze,
    stack,
    tensor_to_float,
    tensor_to_long,
    unsqueeze,
    zeros_like,
)

__all__ = [
    # tensor ops
    "is_tensor",
    "tensor_to_long",
    "tensor_to_float",
    "expand_as_batch",
    "clamp",
    "zeros_like",
    "ones_like",
    "randn_like",
    "unsqueeze",
    "squeeze",
    "flatten",
    "detach",
    "cat",
    "stack",
    # nn modules
    "ModuleAttrMixin",
    "DictOfTensorMixin",
    # nn layers
    "Conv1d",
    "ConvTranspose1d",
    "Conv2d",
    "GroupNorm",
    "BatchNorm2d",
    "Sequential",
    "Identity",
    "Linear",
    "Dropout",
    "Embedding",
    "Mish",
    "SiLU",
    # functional
    "mish",
    "silu",
    "pad_1d",
    "interpolate_1d",
    # einops
    "rearrange_b_h_t_to_b_t_h",
    "rearrange_b_t_h_to_b_h_t",
    "rearrange_batch_t_to_batch_t_1",
]
