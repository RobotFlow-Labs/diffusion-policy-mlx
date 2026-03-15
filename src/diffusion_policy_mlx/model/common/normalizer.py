"""Linear normalizer for diffusion policy data pipelines.

Upstream: diffusion_policy/model/common/normalizer.py

Provides:
  - ``SingleFieldLinearNormalizer``: normalizes a single tensor field (scale/offset).
  - ``LinearNormalizer``: dict-based normalizer with one sub-normalizer per key.

Usage::

    norm = SingleFieldLinearNormalizer.create_fit(data, mode='limits')
    x_norm = norm.normalize(x)
    x_orig = norm.unnormalize(x_norm)

    ln = LinearNormalizer()
    ln.fit({'action': action_data, 'obs': obs_data})
    batch_norm = ln.normalize(batch)
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# SingleFieldLinearNormalizer
# ---------------------------------------------------------------------------


class SingleFieldLinearNormalizer:
    """Normalizes a single tensor field via an affine transform ``y = x * scale + offset``.

    Modes:
      - ``'limits'``: map data range ``[min, max]`` to ``[output_min, output_max]``.
      - ``'gaussian'``: standardize to ``mean=0, std=1``.
    """

    def __init__(
        self,
        scale: mx.array,
        offset: mx.array,
        input_stats: Optional[Dict[str, mx.array]] = None,
    ):
        self.scale = scale
        self.offset = offset
        self.input_stats = input_stats or {}

    # -- factory methods -------------------------------------------------------

    @classmethod
    def create_fit(
        cls,
        data: Union[mx.array, np.ndarray],
        last_n_dims: int = 1,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> "SingleFieldLinearNormalizer":
        """Fit normalizer parameters from data.

        Args:
            data: Array of shape ``(..., D)`` where the last ``last_n_dims``
                  dimensions are normalized together.
            last_n_dims: Number of trailing dims to treat as a single feature
                         vector.  ``0`` means flatten everything.
            mode: ``'limits'`` or ``'gaussian'``.
            output_max / output_min: Target range for limits mode.
            range_eps: Minimum range to avoid division by zero.
            fit_offset: Whether to compute an offset term.
        """
        assert mode in ("limits", "gaussian"), f"Unknown mode: {mode}"
        assert output_max > output_min

        if isinstance(data, np.ndarray):
            data = mx.array(data, dtype=mx.float32)

        # Flatten all but last_n_dims
        if last_n_dims > 0:
            dim = 1
            for d in data.shape[-last_n_dims:]:
                dim *= d
        else:
            dim = 1
            for d in data.shape:
                dim *= d
            dim = max(dim // max(data.shape[0], 1), 1)
            # Actually: flatten everything into (N, dim)
            dim = int(np.prod(data.shape)) // data.shape[0] if data.ndim > 0 else 1

        if last_n_dims > 0:
            flat = data.reshape(-1, *data.shape[-last_n_dims:])
            flat = flat.reshape(-1, dim)
        else:
            flat = data.reshape(-1, dim)

        # Compute stats
        input_min = mx.min(flat, axis=0)
        input_max = mx.max(flat, axis=0)
        input_mean = mx.mean(flat, axis=0)
        input_std = mx.std(flat, axis=0)

        input_stats = {
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        }

        if mode == "limits":
            if fit_offset:
                input_range = input_max - input_min
                # For constant channels: set range to target range so scale = 1
                ignore_mask = input_range < range_eps
                # Replace small ranges with target range
                input_range = mx.where(
                    ignore_mask,
                    mx.full(input_range.shape, output_max - output_min),
                    input_range,
                )
                scale = (output_max - output_min) / input_range
                offset = output_min - scale * input_min
                # For ignored dims, center the output
                center = (output_max + output_min) / 2.0
                offset = mx.where(
                    ignore_mask,
                    center - input_min,
                    offset,
                )
            else:
                assert output_max > 0 and output_min < 0
                output_abs = min(abs(output_min), abs(output_max))
                input_abs = mx.maximum(mx.abs(input_min), mx.abs(input_max))
                ignore_mask = input_abs < range_eps
                input_abs = mx.where(
                    ignore_mask,
                    mx.full(input_abs.shape, output_abs),
                    input_abs,
                )
                scale = output_abs / input_abs
                offset = mx.zeros_like(input_mean)
        elif mode == "gaussian":
            ignore_mask = input_std < range_eps
            safe_std = mx.where(
                ignore_mask,
                mx.ones_like(input_std),
                input_std,
            )
            scale = 1.0 / safe_std
            if fit_offset:
                offset = -input_mean * scale
            else:
                offset = mx.zeros_like(input_mean)

        return cls(scale=scale, offset=offset, input_stats=input_stats)

    @classmethod
    def create_identity(cls, shape=None) -> "SingleFieldLinearNormalizer":
        """Create an identity normalizer (no-op)."""
        if shape is None:
            shape = (1,)
        if isinstance(shape, int):
            shape = (shape,)
        return cls(
            scale=mx.ones(shape),
            offset=mx.zeros(shape),
            input_stats={
                "min": mx.full(shape, -1.0),
                "max": mx.full(shape, 1.0),
                "mean": mx.zeros(shape),
                "std": mx.ones(shape),
            },
        )

    @classmethod
    def create_manual(
        cls,
        scale: Union[mx.array, np.ndarray, float],
        offset: Union[mx.array, np.ndarray, float],
        input_stats: Optional[Dict] = None,
    ) -> "SingleFieldLinearNormalizer":
        """Create a normalizer with manually specified scale and offset."""
        if not isinstance(scale, mx.array):
            scale = mx.array(np.array(scale, dtype=np.float32))
        if not isinstance(offset, mx.array):
            offset = mx.array(np.array(offset, dtype=np.float32))
        scale = scale.reshape(-1)
        offset = offset.reshape(-1)
        return cls(scale=scale, offset=offset, input_stats=input_stats or {})

    # -- normalize / unnormalize -----------------------------------------------

    def normalize(self, x: Union[mx.array, np.ndarray]) -> mx.array:
        """Forward: ``y = x * scale + offset``."""
        if isinstance(x, np.ndarray):
            x = mx.array(x, dtype=mx.float32)
        src_shape = x.shape
        x = x.reshape(-1, self.scale.shape[0])
        x = x * self.scale + self.offset
        return x.reshape(src_shape)

    def unnormalize(self, x: Union[mx.array, np.ndarray]) -> mx.array:
        """Inverse: ``x = (y - offset) / scale``."""
        if isinstance(x, np.ndarray):
            x = mx.array(x, dtype=mx.float32)
        src_shape = x.shape
        x = x.reshape(-1, self.scale.shape[0])
        x = (x - self.offset) / self.scale
        return x.reshape(src_shape)

    def __call__(self, x):
        return self.normalize(x)

    # -- serialization ---------------------------------------------------------

    def state_dict(self) -> dict:
        d = {"scale": self.scale, "offset": self.offset}
        if self.input_stats:
            d["input_stats"] = dict(self.input_stats)
        return d

    def load_state_dict(self, d: dict):
        self.scale = d["scale"]
        self.offset = d["offset"]
        if "input_stats" in d:
            self.input_stats = dict(d["input_stats"])


# ---------------------------------------------------------------------------
# LinearNormalizer
# ---------------------------------------------------------------------------


class LinearNormalizer:
    """Dict-based normalizer: one ``SingleFieldLinearNormalizer`` per key.

    Supports flat dicts and nested observation dicts::

        normalizer = LinearNormalizer()
        normalizer.fit({
            'action': action_data,          # mx.array
            'obs': {                        # nested dict
                'image': image_data,
                'agent_pos': pos_data,
            },
        })
    """

    def __init__(self):
        self.params_dict: Dict[
            str, Union[SingleFieldLinearNormalizer, Dict[str, SingleFieldLinearNormalizer]]
        ] = {}

    def fit(
        self,
        data: Dict[str, Union[mx.array, np.ndarray, Dict]],
        last_n_dims: int = 1,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ):
        """Fit normalizer for each key in *data*.

        If a value is itself a dict, creates a sub-normalizer per sub-key.
        """
        kw = dict(
            last_n_dims=last_n_dims,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )
        for key, value in data.items():
            if isinstance(value, dict):
                sub = {}
                for subkey, subvalue in value.items():
                    sub[subkey] = SingleFieldLinearNormalizer.create_fit(subvalue, **kw)
                self.params_dict[key] = sub
            else:
                self.params_dict[key] = SingleFieldLinearNormalizer.create_fit(value, **kw)

    # -- item access -----------------------------------------------------------

    def __getitem__(
        self, key: str
    ) -> Union[SingleFieldLinearNormalizer, Dict[str, SingleFieldLinearNormalizer]]:
        return self.params_dict[key]

    def __setitem__(self, key: str, value):
        self.params_dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.params_dict

    def keys(self):
        return self.params_dict.keys()

    # -- normalize / unnormalize -----------------------------------------------

    def _apply(self, x, forward: bool):
        """Apply normalize or unnormalize to a dict or single tensor."""
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if key in self.params_dict:
                    entry = self.params_dict[key]
                    if isinstance(value, dict) and isinstance(entry, dict):
                        result[key] = {}
                        for k, v in value.items():
                            if k in entry:
                                result[key][k] = (
                                    entry[k].normalize(v) if forward else entry[k].unnormalize(v)
                                )
                            else:
                                result[key][k] = v
                    elif isinstance(entry, SingleFieldLinearNormalizer):
                        result[key] = (
                            entry.normalize(value) if forward else entry.unnormalize(value)
                        )
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return result
        else:
            # single tensor — check for _default key
            if "_default" in self.params_dict:
                entry = self.params_dict["_default"]
                return entry.normalize(x) if forward else entry.unnormalize(x)
            raise RuntimeError("LinearNormalizer not initialized for single-tensor input")

    def normalize(self, x):
        return self._apply(x, forward=True)

    def unnormalize(self, x):
        return self._apply(x, forward=False)

    def __call__(self, x):
        return self.normalize(x)

    # -- serialization ---------------------------------------------------------

    def state_dict(self) -> dict:
        result = {}
        for key, entry in self.params_dict.items():
            if isinstance(entry, dict):
                result[key] = {k: v.state_dict() for k, v in entry.items()}
            else:
                result[key] = entry.state_dict()
        return result

    def load_state_dict(self, d: dict):
        for key, entry in d.items():
            # Check if it's a nested dict of normalizer state dicts
            if isinstance(entry, dict) and "scale" not in entry:
                sub = {}
                for subkey, subentry in entry.items():
                    n = SingleFieldLinearNormalizer(
                        scale=subentry["scale"],
                        offset=subentry["offset"],
                        input_stats=subentry.get("input_stats"),
                    )
                    sub[subkey] = n
                self.params_dict[key] = sub
            else:
                self.params_dict[key] = SingleFieldLinearNormalizer(
                    scale=entry["scale"],
                    offset=entry["offset"],
                    input_stats=entry.get("input_stats"),
                )
