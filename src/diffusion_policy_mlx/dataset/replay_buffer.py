"""Lightweight replay buffer backed by zarr for read-only dataset access.

This is a simplified version of the upstream ``ReplayBuffer`` tailored
for the MLX port.  It supports reading zarr archives produced by the
original Diffusion Policy codebase and sampling horizon-length
sequences with episode-boundary padding.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr


class ReplayBuffer:
    """Read-only zarr-backed replay buffer.

    Parameters
    ----------
    root : zarr.Group
        Opened zarr group with ``data/`` and ``meta/episode_ends``.
    keys : list[str] or None
        Data keys to expose.  ``None`` means all keys under ``data/``.
    """

    def __init__(self, root: zarr.Group, keys: Optional[List[str]] = None):
        self.root = root
        self._data = root["data"]
        self._meta = root["meta"]

        if keys is not None:
            self._keys = list(keys)
        else:
            # zarr v3: .members() returns (name, node) pairs
            self._keys = sorted(self._list_children(self._data))

        self._episode_ends: np.ndarray = np.asarray(
            self._meta["episode_ends"][:]
        )

    # ------------------------------------------------------------------
    # zarr v3 compat: list child names
    # ------------------------------------------------------------------
    @staticmethod
    def _list_children(group: zarr.Group) -> List[str]:
        """Return child names, compatible with zarr v2 and v3."""
        if hasattr(group, "members"):
            # zarr v3
            return [name for name, _ in group.members()]
        # zarr v2 fallback
        return list(group.keys())  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def create_from_path(
        cls,
        zarr_path: str,
        keys: Optional[List[str]] = None,
    ) -> "ReplayBuffer":
        """Open an on-disk zarr archive read-only."""
        group = zarr.open(os.path.expanduser(zarr_path), mode="r")
        return cls(root=group, keys=keys)

    @classmethod
    def copy_from_path(
        cls,
        zarr_path: str,
        keys: Optional[List[str]] = None,
    ) -> "ReplayBuffer":
        """Load zarr data into memory (numpy dict backend).

        This mirrors the upstream ``copy_from_path`` behaviour: all
        requested arrays are eagerly read into RAM so that subsequent
        access is fast.
        """
        src = zarr.open(os.path.expanduser(zarr_path), mode="r")

        if keys is None:
            keys = sorted(cls._list_children(src["data"]))

        data: Dict[str, np.ndarray] = {}
        for key in keys:
            data[key] = np.asarray(src["data"][key][:])

        episode_ends = np.asarray(src["meta"]["episode_ends"][:])

        return cls._from_numpy(data, episode_ends, keys)

    @classmethod
    def _from_numpy(
        cls,
        data: Dict[str, np.ndarray],
        episode_ends: np.ndarray,
        keys: List[str],
    ) -> "ReplayBuffer":
        """Construct from in-memory numpy arrays (dict backend)."""
        buf = object.__new__(cls)
        buf._data = data  # type: ignore[assignment]
        buf._meta = {"episode_ends": episode_ends}
        buf._keys = list(keys)
        buf._episode_ends = episode_ends
        buf.root = {"data": data, "meta": buf._meta}  # type: ignore[assignment]
        return buf

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def episode_ends(self) -> np.ndarray:
        return self._episode_ends

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ends)

    @property
    def n_steps(self) -> int:
        if len(self._episode_ends) == 0:
            return 0
        return int(self._episode_ends[-1])

    # ------------------------------------------------------------------
    # dict-like access
    # ------------------------------------------------------------------
    def keys(self) -> List[str]:
        return self._keys

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    # ------------------------------------------------------------------
    # episode / sequence access
    # ------------------------------------------------------------------
    def get_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """Return all data for episode ``idx``."""
        start = 0 if idx == 0 else int(self._episode_ends[idx - 1])
        end = int(self._episode_ends[idx])
        result: Dict[str, np.ndarray] = {}
        for key in self._keys:
            arr = self._data[key]
            result[key] = np.asarray(arr[start:end])
        return result

    def get_episode_slice(self, idx: int) -> slice:
        start = 0 if idx == 0 else int(self._episode_ends[idx - 1])
        end = int(self._episode_ends[idx])
        return slice(start, end)
