"""Dict manipulation utilities for nested observation/action dictionaries.

Upstream: diffusion_policy/common/pytorch_util.py (dict_apply, dict_apply_split,
dict_apply_reduce).

These are framework-agnostic — they operate on dict structures whose leaves
can be any type (mx.array, np.ndarray, scalars, etc.).
"""

from __future__ import annotations

import collections
from typing import Any, Callable, Dict, List


def dict_apply(
    x: Dict[str, Any],
    func: Callable[[Any], Any],
) -> Dict[str, Any]:
    """Recursively apply *func* to all leaf values in a nested dict.

    A "leaf" is any value that is not itself a ``dict``.

    Example::

        >>> import mlx.core as mx
        >>> d = {"obs": {"image": mx.zeros((3, 96, 96)), "pos": mx.ones((2,))}}
        >>> d2 = dict_apply(d, lambda v: v * 2)
        >>> float(d2["obs"]["pos"][0])
        2.0

    Args:
        x: A (possibly nested) dictionary.
        func: A callable to apply to each leaf value.

    Returns:
        A new dictionary with the same structure, leaf values transformed.
    """
    result: Dict[str, Any] = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_apply_split(
    x: Dict[str, Any],
    split_func: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Apply a function that returns a dict to each leaf, then transpose the result.

    Given a dict ``{a: v1, b: v2}`` and a *split_func* that maps each value
    to ``{k1: ..., k2: ...}``, returns ``{k1: {a: ..., b: ...}, k2: {a: ..., b: ...}}``.

    This is useful for splitting a batch dict into train/val portions, or
    splitting observations into different modality groups.

    Example::

        >>> import numpy as np
        >>> d = {"x": np.array([1, 2, 3, 4]), "y": np.array([5, 6, 7, 8])}
        >>> result = dict_apply_split(d, lambda v: {"first": v[:2], "second": v[2:]})
        >>> list(result["first"]["x"])
        [1, 2]

    Args:
        x: A flat dictionary (non-nested — each value is a leaf).
        split_func: A callable that takes a leaf value and returns a dict of
            named splits.

    Returns:
        A dict of dicts, transposed from the input structure.
    """
    results: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    for key, value in x.items():
        split_result = split_func(value)
        for split_key, split_value in split_result.items():
            results[split_key][key] = split_value
    return dict(results)


def dict_apply_reduce(
    x_list: List[Dict[str, Any]],
    reduce_func: Callable[[List[Any]], Any],
) -> Dict[str, Any]:
    """Reduce a list of dicts into a single dict by applying *reduce_func* per key.

    All dicts in *x_list* must have the same top-level keys.

    Example::

        >>> import numpy as np
        >>> dicts = [{"loss": np.float64(0.5)}, {"loss": np.float64(0.3)}]
        >>> result = dict_apply_reduce(dicts, lambda vals: sum(vals) / len(vals))
        >>> float(result["loss"])
        0.4

    Args:
        x_list: A list of dicts with identical key sets.
        reduce_func: A callable that takes a list of values (one per dict)
            and returns a single reduced value.

    Returns:
        A single dict with the same keys, each value reduced.

    Raises:
        IndexError: If *x_list* is empty.
        KeyError: If dicts have inconsistent keys.
    """
    if not x_list:
        return {}
    result: Dict[str, Any] = {}
    for key in x_list[0].keys():
        result[key] = reduce_func([d[key] for d in x_list])
    return result
