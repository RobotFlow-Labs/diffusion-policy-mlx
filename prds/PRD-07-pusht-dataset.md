# PRD-07: PushT Dataset

**Status:** Complete
**Depends on:** PRD-00 (dev environment), PRD-05 (normalizer)
**Blocks:** PRD-08 (Evaluation)

---

## Objective

Implement the PushT image dataset loader: zarr-backed data reading, numpy→mx.array conversion, normalization, and a download script.

---

## Upstream Reference

| File | Classes |
|------|---------|
| `dataset/pusht_image_dataset.py` | `PushTImageDataset` |
| `dataset/base_dataset.py` | `BaseImageDataset` |
| `common/replay_buffer.py` | `ReplayBuffer` — zarr-based storage |
| `common/sampler.py` | `SequenceSampler` — episode-aware sampling |

---

## PushT Dataset Format

The dataset is a zarr archive with structure:

```
pusht_image.zarr/
  data/
    img/       # (N, 96, 96, 3) uint8 — RGB images
    state/     # (N, 5) float32 — [agent_x, agent_y, block_x, block_y, block_angle]
    action/    # (N, 2) float32 — [dx, dy] agent actions
  meta/
    episode_ends/  # (num_episodes,) int — cumulative episode lengths
```

**Total size:** ~2.5GB for the full dataset (~200 episodes).

---

## Deliverables

### 1. `dataset/base_dataset.py`

```python
class BaseImageDataset:
    """Abstract base for image + action datasets."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns:
            {
                'obs': {
                    'image': np.ndarray (T, 3, 96, 96) float32 [0,1],
                    'agent_pos': np.ndarray (T, 2) float32,
                },
                'action': np.ndarray (T, 2) float32,
            }
        """
        raise NotImplementedError

    def get_normalizer(self, mode='limits', **kwargs):
        """Return LinearNormalizer fitted on this dataset."""
        raise NotImplementedError

    def get_validation_dataset(self):
        """Return a validation split."""
        raise NotImplementedError
```

### 2. `dataset/pusht_image_dataset.py`

```python
class PushTImageDataset(BaseImageDataset):
    """PushT image dataset from zarr replay buffer.

    Returns sequences of length `horizon` with images, agent positions,
    and actions. Handles episode boundaries with padding.
    """

    def __init__(self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,     # n_obs_steps - 1
        pad_after: int = 7,      # n_action_steps - 1
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int = None):

        # Load zarr
        import zarr
        root = zarr.open(zarr_path, 'r')
        self.images = root['data/img']           # (N, 96, 96, 3)
        self.states = root['data/state']          # (N, 5)
        self.actions = root['data/action']        # (N, 2)
        self.episode_ends = root['meta/episode_ends'][:]  # (num_episodes,)

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Compute valid sampling indices
        self.indices = self._compute_indices()

        # Train/val split
        if val_ratio > 0:
            n_val = int(len(self.episode_ends) * val_ratio)
            ...

    def _compute_indices(self):
        """Compute valid (episode_idx, start_idx) pairs for sampling.

        Each index yields a sequence of length `horizon` from a single episode.
        Sequences near episode boundaries are padded (replicate edge frames).
        """
        indices = []
        episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        for ep_idx, (start, end) in enumerate(zip(episode_starts, self.episode_ends)):
            ep_len = end - start
            # Can sample starting positions from -pad_before to ep_len-1+pad_after
            for offset in range(ep_len):
                indices.append((start + offset, start, end))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center, ep_start, ep_end = self.indices[idx]

        # Compute sequence range with padding
        seq_start = center - self.pad_before
        seq_end = center + self.horizon - self.pad_before

        # Sample with episode-boundary clamping
        actual_start = max(seq_start, ep_start)
        actual_end = min(seq_end, ep_end)

        # Load data
        images = self.images[actual_start:actual_end]  # (T', 96, 96, 3)
        states = self.states[actual_start:actual_end]   # (T', 5)
        actions = self.actions[actual_start:actual_end]  # (T', 2)

        # Pad if needed (replicate edge frames)
        pad_before = actual_start - seq_start
        pad_after = seq_end - actual_end
        if pad_before > 0 or pad_after > 0:
            images = self._pad_sequence(images, pad_before, pad_after)
            states = self._pad_sequence(states, pad_before, pad_after)
            actions = self._pad_sequence(actions, pad_before, pad_after)

        # Process images: (T, 96, 96, 3) uint8 → (T, 3, 96, 96) float32 [0,1]
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))  # HWC → CHW

        # Agent position: first 2 dims of state
        agent_pos = states[:, :2].astype(np.float32)

        return {
            'obs': {
                'image': images,         # (T, 3, 96, 96)
                'agent_pos': agent_pos,  # (T, 2)
            },
            'action': actions.astype(np.float32),  # (T, 2)
        }

    @staticmethod
    def _pad_sequence(seq, pad_before, pad_after):
        """Replicate-pad sequence at boundaries."""
        pads = []
        if pad_before > 0:
            pads.append(np.repeat(seq[:1], pad_before, axis=0))
        pads.append(seq)
        if pad_after > 0:
            pads.append(np.repeat(seq[-1:], pad_after, axis=0))
        return np.concatenate(pads, axis=0)

    def get_normalizer(self, mode='limits', **kwargs):
        """Fit normalizer on all data."""
        normalizer = LinearNormalizer()

        # Fit on actions
        all_actions = self.actions[:].astype(np.float32)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            mx.array(all_actions), mode=mode, **kwargs)

        # Fit on agent_pos
        all_agent_pos = self.states[:, :2].astype(np.float32)
        normalizer['obs'] = {}
        normalizer['obs']['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
            mx.array(all_agent_pos), mode=mode, **kwargs)

        # Image: identity normalizer (already in [0,1])
        normalizer['obs']['image'] = SingleFieldLinearNormalizer.create_identity((3,))

        return normalizer

    def get_validation_dataset(self):
        """Return validation split."""
        ...
```

### 3. `scripts/download_pusht.py`

```python
"""Download PushT dataset for Diffusion Policy.

Usage:
    python scripts/download_pusht.py --output data/

Downloads from the official Diffusion Policy data server.
"""
import os
import urllib.request
from pathlib import Path


PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht_cchi_v7_replay.zarr.zip"


def download_pusht(output_dir: str = "data"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "pusht_image.zarr.zip"
    zarr_path = output_path / "pusht_image.zarr"

    if zarr_path.exists():
        print(f"Dataset already exists at {zarr_path}")
        return str(zarr_path)

    print(f"Downloading PushT dataset to {zip_path}...")
    urllib.request.urlretrieve(PUSHT_URL, str(zip_path))

    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(output_path))

    # Cleanup
    zip_path.unlink()

    print(f"Dataset ready at {zarr_path}")
    return str(zarr_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    download_pusht(args.output)
```

---

## Data Pipeline Flow

```
zarr file on disk
  ↓ __getitem__(idx)
  ↓ Load numpy arrays for horizon-length sequence
  ↓ Pad at episode boundaries (replicate edge)
  ↓ Image: uint8 HWC → float32 CHW [0,1]
  ↓ Agent pos: extract first 2 dims of state
  ↓ Return dict of numpy arrays
  ↓
collate_batch()
  ↓ Stack numpy arrays → mx.array
  ↓ Send to policy.compute_loss()
```

**Key design:** `__getitem__` returns numpy arrays, NOT mx.arrays. The `collate_batch` function in the training loop (PRD-06) converts to mx.array. This avoids creating many small mx.arrays during data loading.

---

## Tests

### `tests/test_dataset.py`

```python
def test_pusht_dataset_shapes(tmp_path):
    """Create a small synthetic zarr and verify shapes."""
    import zarr
    # Create minimal zarr
    root = zarr.open(str(tmp_path / "test.zarr"), "w")
    N = 200
    root.create_dataset("data/img", data=np.random.randint(0, 256, (N, 96, 96, 3), dtype=np.uint8))
    root.create_dataset("data/state", data=np.random.randn(N, 5).astype(np.float32))
    root.create_dataset("data/action", data=np.random.randn(N, 2).astype(np.float32))
    root.create_dataset("meta/episode_ends", data=np.array([100, 200]))

    ds = PushTImageDataset(str(tmp_path / "test.zarr"), horizon=16, pad_before=1, pad_after=7)
    assert len(ds) > 0

    sample = ds[0]
    assert sample['obs']['image'].shape == (16, 3, 96, 96)
    assert sample['obs']['agent_pos'].shape == (16, 2)
    assert sample['action'].shape == (16, 2)

def test_pusht_image_range(tmp_path):
    """Images should be in [0, 1] float32."""
    # ... create zarr
    sample = ds[0]
    assert sample['obs']['image'].dtype == np.float32
    assert sample['obs']['image'].min() >= 0.0
    assert sample['obs']['image'].max() <= 1.0

def test_pusht_episode_boundary_padding(tmp_path):
    """Sequences at episode boundaries should be properly padded."""
    # ... create zarr with short episodes
    # Verify first sample is edge-replicate padded
    ...

def test_pusht_normalizer(tmp_path):
    """Normalizer should produce values in expected range."""
    # ... create zarr
    normalizer = ds.get_normalizer(mode='limits')
    sample = ds[0]
    normed_action = normalizer['action'].normalize(mx.array(sample['action']))
    # Should be in [-1, 1]
    assert float(mx.min(normed_action)) >= -1.5  # some tolerance
    assert float(mx.max(normed_action)) <= 1.5

def test_collate_batch(tmp_path):
    """Collating multiple samples produces batched mx.arrays."""
    # ... create zarr
    samples = [ds[i] for i in range(4)]
    batch = collate_batch(samples)
    assert isinstance(batch['obs']['image'], mx.array)
    assert batch['obs']['image'].shape == (4, 16, 3, 96, 96)
    assert batch['action'].shape == (4, 16, 2)
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | `__getitem__` returns correct shapes: image (T,3,96,96), agent_pos (T,2), action (T,2) | exact |
| 2 | Images in [0,1] float32, CHW format | range + dtype check |
| 3 | Episode boundary padding works (replicate edge) | visual + numeric |
| 4 | Normalizer round-trip preserves data | atol=1e-5 |
| 5 | `collate_batch` produces valid mx.array batches | type + shape check |
| 6 | Works with real PushT zarr when available | integration test |
| 7 | Download script retrieves and extracts dataset | manual test |

---

## Upstream Sync Notes

**Files to watch:**
- `dataset/pusht_image_dataset.py` — if zarr key names or data format changes
- `common/replay_buffer.py` — if storage format changes
- Data URL — if dataset hosting location changes

**Data format stability:** The PushT zarr format has been stable since the initial paper release. Unlikely to change.
