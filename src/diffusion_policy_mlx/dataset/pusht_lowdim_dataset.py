"""PushT low-dimensional dataset for diffusion policy training.

Upstream: diffusion_policy/dataset/pusht_dataset.py

Loads a zarr replay buffer with structure::

    data/keypoint      (N, 9, 2) float32   -- 9 keypoints x 2D
    data/state         (N, 5)   float32    -- [agent_x, agent_y, block_x, block_y, block_angle]
    data/action        (N, 2)   float32    -- [dx, dy]
    meta/episode_ends  (num_eps,) int64

Each ``__getitem__`` call returns a horizon-length sequence with::

    obs:    (T, Do)  -- concatenation of flattened keypoints + agent_pos
    action: (T, Da)  -- action vectors

This is much faster to load than the image variant since no image
processing is needed.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np

from diffusion_policy_mlx.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy_mlx.dataset.pusht_image_dataset import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy_mlx.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class PushTLowdimDataset(BaseLowdimDataset):
    """PushT low-dim dataset from zarr replay buffer.

    Returns sequences of length ``horizon`` with low-dimensional
    observations (keypoints + agent position) and actions.

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
    obs_key : str
        Key for keypoint data in the zarr archive.
    state_key : str
        Key for state data in the zarr archive.
    action_key : str
        Key for action data in the zarr archive.
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
        obs_key: str = "keypoint",
        state_key: str = "state",
        action_key: str = "action",
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
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key

        root = _zarr.open(zarr_path, mode="r")

        # Load data arrays (lazy zarr-backed)
        self._keypoints = root[f"data/{obs_key}"]      # (N, 9, 2) or similar
        self._states = root[f"data/{state_key}"]        # (N, 5) float32
        self._actions = root[f"data/{action_key}"]      # (N, 2) float32
        self._episode_ends: np.ndarray = np.asarray(
            root["meta/episode_ends"][:]
        ).astype(np.int64)

        # Train/val split
        val_mask = get_val_mask(
            n_episodes=len(self._episode_ends),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_train_episodes, seed=seed)
        self._train_mask = train_mask

        # Build sampler
        data_dict = {
            obs_key: self._keypoints,
            state_key: self._states,
            action_key: self._actions,
        }
        self.sampler = SequenceSampler(
            data=data_dict,
            episode_ends=self._episode_ends,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=[obs_key, state_key, action_key],
            episode_mask=train_mask,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert raw sample to obs/action dict.

        Observations are the concatenation of flattened keypoints and
        agent position (first 2 dims of state).
        """
        keypoint = sample[self.obs_key]  # (T, 9, 2) or similar
        state = sample[self.state_key]   # (T, 5)
        agent_pos = state[:, :2]

        # Flatten keypoints: (T, 9, 2) -> (T, 18)
        keypoint_flat = keypoint.reshape(keypoint.shape[0], -1)

        obs = np.concatenate(
            [keypoint_flat, agent_pos], axis=-1
        ).astype(np.float32)

        data = {
            "obs": obs,                                          # (T, Do)
            "action": sample[self.action_key].astype(np.float32),  # (T, Da)
        }
        return data

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """Fit normalizer on all data (actions + obs).

        Returns a ``LinearNormalizer`` with keys ``action`` and ``obs``.
        """
        # Build the full obs array
        all_keypoints = np.asarray(self._keypoints[:]).astype(np.float32)
        all_states = np.asarray(self._states[:]).astype(np.float32)
        all_actions = np.asarray(self._actions[:]).astype(np.float32)

        agent_pos = all_states[:, :2]
        keypoint_flat = all_keypoints.reshape(all_keypoints.shape[0], -1)
        all_obs = np.concatenate([keypoint_flat, agent_pos], axis=-1)

        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            all_actions, mode=mode, **kwargs
        )
        normalizer["obs"] = SingleFieldLinearNormalizer.create_fit(
            all_obs, mode=mode, **kwargs
        )

        return normalizer

    # ------------------------------------------------------------------
    # Validation split
    # ------------------------------------------------------------------

    def get_validation_dataset(self) -> "PushTLowdimDataset":
        """Return a dataset instance using the validation episodes."""
        val_set = copy.copy(self)
        val_mask = ~self._train_mask
        data_dict = {
            self.obs_key: self._keypoints,
            self.state_key: self._states,
            self.action_key: self._actions,
        }
        val_set.sampler = SequenceSampler(
            data=data_dict,
            episode_ends=self._episode_ends,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=[self.obs_key, self.state_key, self.action_key],
            episode_mask=val_mask,
        )
        val_set._train_mask = val_mask
        return val_set

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_all_actions(self) -> np.ndarray:
        return np.asarray(self._actions[:]).astype(np.float32)
