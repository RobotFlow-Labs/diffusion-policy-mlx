"""Common utilities for diffusion-policy-mlx.

Provides dict manipulation helpers, JSON logging, and MLX equivalents
of upstream pytorch_util functions.
"""

from diffusion_policy_mlx.common.dict_util import (
    dict_apply,
    dict_apply_reduce,
    dict_apply_split,
)
from diffusion_policy_mlx.common.json_logger import JsonLogger
from diffusion_policy_mlx.common.pytorch_util import (
    optimizer_to,
    param_count,
    replace_submodules,
)

__all__ = [
    "dict_apply",
    "dict_apply_reduce",
    "dict_apply_split",
    "JsonLogger",
    "optimizer_to",
    "param_count",
    "replace_submodules",
]
