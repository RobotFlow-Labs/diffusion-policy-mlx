"""Module utility mixins ported from upstream diffusion_policy.

* ``ModuleAttrMixin`` — adds ``.device`` / ``.dtype`` properties.
* ``DictOfTensorMixin`` — stores named tensors as a nested dict tree.
"""

from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_leaf(tree):
    """Return the first ``mx.array`` leaf found via DFS in *tree*."""
    if isinstance(tree, mx.array):
        return tree
    if isinstance(tree, dict):
        for v in tree.values():
            leaf = _first_leaf(v)
            if leaf is not None:
                return leaf
    if isinstance(tree, (list, tuple)):
        for v in tree:
            leaf = _first_leaf(v)
            if leaf is not None:
                return leaf
    return None


# ---------------------------------------------------------------------------
# ModuleAttrMixin
# ---------------------------------------------------------------------------


class ModuleAttrMixin(nn.Module):
    """Port of ``diffusion_policy.model.common.module_attr_mixin.ModuleAttrMixin``.

    In PyTorch this exposes ``.device`` and ``.dtype`` from a dummy parameter.
    In MLX there is no device concept; dtype is inferred from first parameter.
    """

    @property
    def device(self) -> str:
        return "mlx"

    @property
    def dtype(self):
        leaf = _first_leaf(self.parameters())
        if leaf is not None:
            return leaf.dtype
        return mx.float32


# ---------------------------------------------------------------------------
# DictOfTensorMixin
# ---------------------------------------------------------------------------


class DictOfTensorMixin(nn.Module):
    """Port of ``diffusion_policy.model.common.dict_of_tensor_mixin.DictOfTensorMixin``.

    Stores named tensor parameters in ``self.params_dict``.
    MLX modules auto-discover ``mx.array`` attributes, so we keep them
    in a plain dict that is assigned as an attribute.
    """

    def __init__(self, params_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.params_dict: Dict[str, Any] = params_dict if params_dict is not None else {}

    @property
    def device(self) -> str:
        return "mlx"

    # ------------------------------------------------------------------
    # Weight-loading helper (mirrors upstream _load_from_state_dict)
    # ------------------------------------------------------------------

    def load_params_dict(self, flat_dict: Dict[str, mx.array], prefix: str = "params_dict."):
        """Reconstruct a nested ``params_dict`` from a flat key→array map.

        Keys are expected to have the form ``params_dict.a.b.c`` — the
        *prefix* is stripped and the remaining dot-separated parts form the
        nested dict path.
        """

        def _dfs_add(dest: dict, keys: list, value: mx.array):
            if len(keys) == 1:
                dest[keys[0]] = value
                return
            if keys[0] not in dest:
                dest[keys[0]] = {}
            _dfs_add(dest[keys[0]], keys[1:], value)

        out: Dict[str, Any] = {}
        for key, value in flat_dict.items():
            if key.startswith(prefix):
                param_keys = key[len(prefix) :].split(".")
                _dfs_add(out, param_keys, value)
        self.params_dict = out
