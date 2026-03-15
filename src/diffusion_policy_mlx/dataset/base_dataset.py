"""Abstract base classes for image and low-dim datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseImageDataset(ABC):
    """Abstract base for image + action datasets.

    Subclasses must implement ``__len__``, ``__getitem__``,
    ``get_normalizer``, and optionally ``get_validation_dataset``.

    ``__getitem__`` returns plain numpy arrays.  The training loop is
    responsible for batching and converting to ``mx.array`` via
    ``collate_batch``.
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single training sample.

        Returns
        -------
        dict
            ``obs``  -- dict with ``image`` (T, C, H, W) float32 [0,1]
                        and ``agent_pos`` (T, D) float32.
            ``action`` -- (T, Da) float32.
        """
        raise NotImplementedError

    @abstractmethod
    def get_normalizer(self, mode: str = "limits", **kwargs):
        """Return a normalizer fitted on this dataset."""
        raise NotImplementedError

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation split.  Default: empty dataset."""
        return _EmptyImageDataset()

    def get_all_actions(self) -> np.ndarray:
        """Return all raw actions as a single numpy array."""
        raise NotImplementedError


class _EmptyImageDataset(BaseImageDataset):
    """Fallback empty dataset returned by ``get_validation_dataset``."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise IndexError("Empty dataset")

    def get_normalizer(self, mode: str = "limits", **kwargs):
        raise RuntimeError("Cannot fit normalizer on empty dataset")


# ---------------------------------------------------------------------------
# Low-dim base
# ---------------------------------------------------------------------------


class BaseLowdimDataset(ABC):
    """Abstract base for low-dimensional observation datasets.

    Similar to ``BaseImageDataset`` but ``__getitem__`` returns flat
    observation vectors instead of image dicts::

        {
            'obs': (T, Do) float32,
            'action': (T, Da) float32,
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single training sample.

        Returns
        -------
        dict
            ``obs``    -- (T, Do) float32 observation vector.
            ``action`` -- (T, Da) float32 action vector.
        """
        raise NotImplementedError

    @abstractmethod
    def get_normalizer(self, mode: str = "limits", **kwargs):
        """Return a normalizer fitted on this dataset."""
        raise NotImplementedError

    def get_validation_dataset(self) -> "BaseLowdimDataset":
        """Return a validation split.  Default: empty dataset."""
        return _EmptyLowdimDataset()

    def get_all_actions(self) -> np.ndarray:
        """Return all raw actions as a single numpy array."""
        raise NotImplementedError


class _EmptyLowdimDataset(BaseLowdimDataset):
    """Fallback empty dataset returned by ``get_validation_dataset``."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise IndexError("Empty dataset")

    def get_normalizer(self, mode: str = "limits", **kwargs):
        raise RuntimeError("Cannot fit normalizer on empty dataset")
