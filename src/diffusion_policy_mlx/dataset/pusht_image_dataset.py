"""PushT image dataset for diffusion policy training.

Loads a zarr replay buffer with structure::

    data/img           (N, 96, 96, 3) uint8
    data/state         (N, 5)         float32
    data/action        (N, 2)         float32
    meta/episode_ends  (num_eps,)     int64

Each ``__getitem__`` call returns a horizon-length sequence of numpy
arrays.  Conversion to ``mx.array`` happens in ``collate_batch``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diffusion_policy_mlx.dataset.base_dataset import BaseImageDataset


# ---------------------------------------------------------------------------
# Index creation (pure numpy, no numba dependency)
# ---------------------------------------------------------------------------

def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    episode_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute valid sampling indices respecting episode boundaries.

    Returns an (M, 4) int64 array where each row is::

        [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]

    ``buffer_*`` are indices into the global data arrays.
    ``sample_*`` are indices into the output array of length
    ``sequence_length``.  When the sequence extends past an episode
    boundary the gap is filled by replicating the edge frame (handled
    in ``SequenceSampler.sample_sequence``).
    """
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    if episode_mask is None:
        episode_mask = np.ones(len(episode_ends), dtype=bool)

    indices: List[List[int]] = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx,
                buffer_end_idx,
                sample_start_idx,
                sample_end_idx,
            ])

    if len(indices) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return np.array(indices, dtype=np.int64)


def get_val_mask(
    n_episodes: int, val_ratio: float, seed: int = 0
) -> np.ndarray:
    """Create a boolean mask selecting validation episodes."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(
    mask: np.ndarray, max_n: Optional[int], seed: int = 0
) -> np.ndarray:
    """Optionally subsample True entries in *mask* down to *max_n*."""
    if max_n is not None and int(np.sum(mask)) > max_n:
        n_train = int(max_n)
        curr_idxs = np.nonzero(mask)[0]
        rng = np.random.default_rng(seed=seed)
        chosen = rng.choice(len(curr_idxs), size=n_train, replace=False)
        new_mask = np.zeros_like(mask)
        new_mask[curr_idxs[chosen]] = True
        return new_mask
    return mask


# ---------------------------------------------------------------------------
# Sequence sampler
# ---------------------------------------------------------------------------

class SequenceSampler:
    """Episode-aware sequence sampler with edge-replicate padding."""

    def __init__(
        self,
        data: Dict[str, Any],
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys: Optional[List[str]] = None,
        episode_mask: Optional[np.ndarray] = None,
    ):
        assert sequence_length >= 1
        if keys is None:
            keys = list(data.keys())

        self.indices = create_indices(
            episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.data = data

    def __len__(self) -> int:
        return len(self.indices)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """Return a dict of padded numpy arrays for sample *idx*."""
        buffer_start, buffer_end, sample_start, sample_end = self.indices[idx]

        result: Dict[str, np.ndarray] = {}
        for key in self.keys:
            input_arr = self.data[key]
            sample = np.asarray(input_arr[buffer_start:buffer_end])

            if sample_start > 0 or sample_end < self.sequence_length:
                # Need padding — replicate edge frames
                shape = (self.sequence_length,) + sample.shape[1:]
                data_out = np.zeros(shape, dtype=sample.dtype)
                # Fill before with first frame
                if sample_start > 0:
                    data_out[:sample_start] = sample[0]
                # Fill after with last frame
                if sample_end < self.sequence_length:
                    data_out[sample_end:] = sample[-1]
                # Copy actual data
                data_out[sample_start:sample_end] = sample
                sample = data_out

            result[key] = sample
        return result


# ---------------------------------------------------------------------------
# Minimal normalizer (standalone, swappable with PRD-05 normalizer later)
# ---------------------------------------------------------------------------

class _SingleFieldNormalizer:
    """Minimal linear normalizer for a single field.

    Supports ``limits`` mode (scale to [-1, 1]) and ``gaussian`` mode
    (zero-mean, unit-variance).  Also supports ``identity`` (no-op).
    """

    def __init__(
        self,
        offset: np.ndarray,
        scale: np.ndarray,
        input_range_low: np.ndarray,
        input_range_high: np.ndarray,
    ):
        self.offset = offset.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.input_range_low = input_range_low.astype(np.float32)
        self.input_range_high = input_range_high.astype(np.float32)

    @classmethod
    def create_fit(
        cls,
        data: np.ndarray,
        mode: str = "limits",
        output_min: float = -1.0,
        output_max: float = 1.0,
        eps: float = 1e-7,
    ) -> "_SingleFieldNormalizer":
        """Fit normalizer to data."""
        # Flatten to 2D: (N, D)
        flat = data.reshape(-1, data.shape[-1]).astype(np.float32)

        if mode == "limits":
            input_min = flat.min(axis=0)
            input_max = flat.max(axis=0)
            input_range = input_max - input_min
            input_range = np.maximum(input_range, eps)
            # scale to [output_min, output_max]
            scale = (output_max - output_min) / input_range
            offset = output_min - input_min * scale
            return cls(
                offset=offset,
                scale=scale,
                input_range_low=input_min,
                input_range_high=input_max,
            )
        elif mode == "gaussian":
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            std = np.maximum(std, eps)
            scale = 1.0 / std
            offset = -mean * scale
            return cls(
                offset=offset,
                scale=scale,
                input_range_low=mean - 3 * std,
                input_range_high=mean + 3 * std,
            )
        else:
            raise ValueError(f"Unknown normalizer mode: {mode}")

    @classmethod
    def create_identity(cls, shape: Tuple[int, ...] = ()) -> "_SingleFieldNormalizer":
        """No-op normalizer (for images already in [0,1])."""
        d = max(shape[-1] if shape else 1, 1)
        return cls(
            offset=np.zeros(d, dtype=np.float32),
            scale=np.ones(d, dtype=np.float32),
            input_range_low=np.zeros(d, dtype=np.float32),
            input_range_high=np.ones(d, dtype=np.float32),
        )

    def normalize(self, x):
        """Normalize ``x``.  Works with numpy or mx.array."""
        return x * self.scale + self.offset

    def unnormalize(self, x):
        """Inverse of ``normalize``."""
        return (x - self.offset) / self.scale


class _DictNormalizer(dict):
    """Dict-like container for per-field normalizers.

    Supports nested access: ``normalizer['obs']['agent_pos']``.
    """
    pass


# ---------------------------------------------------------------------------
# PushTImageDataset
# ---------------------------------------------------------------------------

class PushTImageDataset(BaseImageDataset):
    """PushT image dataset from zarr replay buffer.

    Returns sequences of length ``horizon`` with images, agent positions,
    and actions.  Handles episode boundaries with replicate-edge padding.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr archive directory.
    horizon : int
        Sequence length for each sample.
    pad_before : int
        Number of timesteps the sequence may extend before the episode
        start (filled by replicating the first frame).
    pad_after : int
        Number of timesteps the sequence may extend after the episode
        end (filled by replicating the last frame).
    seed : int
        RNG seed for train/val splitting.
    val_ratio : float
        Fraction of episodes reserved for validation.
    max_train_episodes : int or None
        Cap on training episodes (``None`` = use all).
    """

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        import zarr as _zarr

        self._zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._seed = seed
        self._val_ratio = val_ratio

        root = _zarr.open(zarr_path, mode="r")

        # Eagerly read meta; keep data arrays lazy (zarr-backed).
        self._images = root["data/img"]          # (N, 96, 96, 3) uint8
        self._states = root["data/state"]        # (N, 5) float32
        self._actions = root["data/action"]      # (N, 2) float32
        self._episode_ends: np.ndarray = np.asarray(
            root["meta/episode_ends"][:]
        ).astype(np.int64)

        # Train/val split ------------------------------------------------
        val_mask = get_val_mask(
            n_episodes=len(self._episode_ends),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        self._train_mask = train_mask

        # Build sampler --------------------------------------------------
        data_dict = {
            "img": self._images,
            "state": self._states,
            "action": self._actions,
        }
        self.sampler = SequenceSampler(
            data=data_dict,
            episode_ends=self._episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=["img", "state", "action"],
            episode_mask=train_mask,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.sampler.sample_sequence(idx)

        # Image: (T, 96, 96, 3) uint8 -> (T, 3, 96, 96) float32 [0,1]
        images = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0

        # Agent position: first 2 dims of state
        agent_pos = sample["state"][:, :2].astype(np.float32)

        # Action
        actions = sample["action"].astype(np.float32)

        return {
            "obs": {
                "image": images,       # (T, 3, 96, 96)
                "agent_pos": agent_pos, # (T, 2)
            },
            "action": actions,          # (T, 2)
        }

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------

    def get_normalizer(self, mode: str = "limits", **kwargs) -> _DictNormalizer:
        """Fit normalizer on all data (actions + agent_pos).

        Returns a dict-like normalizer with keys ``action`` and
        ``obs`` (which itself is a dict with ``agent_pos`` and
        ``image``).
        """
        all_actions = np.asarray(self._actions[:]).astype(np.float32)
        all_agent_pos = np.asarray(self._states[:, :2]).astype(np.float32)

        normalizer = _DictNormalizer()
        normalizer["action"] = _SingleFieldNormalizer.create_fit(
            all_actions, mode=mode, **kwargs
        )

        obs_normalizer = _DictNormalizer()
        obs_normalizer["agent_pos"] = _SingleFieldNormalizer.create_fit(
            all_agent_pos, mode=mode, **kwargs
        )
        obs_normalizer["image"] = _SingleFieldNormalizer.create_identity((3,))
        normalizer["obs"] = obs_normalizer

        return normalizer

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------

    def get_validation_dataset(self) -> "PushTImageDataset":
        """Return a dataset instance using the validation episodes."""
        val_set = copy.copy(self)
        val_mask = ~self._train_mask
        data_dict = {
            "img": self._images,
            "state": self._states,
            "action": self._actions,
        }
        val_set.sampler = SequenceSampler(
            data=data_dict,
            episode_ends=self._episode_ends,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=["img", "state", "action"],
            episode_mask=val_mask,
        )
        val_set._train_mask = val_mask
        return val_set

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_all_actions(self) -> np.ndarray:
        return np.asarray(self._actions[:]).astype(np.float32)
