"""Tests for PushT dataset, replay buffer, collation, and normalizer."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

import mlx.core as mx

from diffusion_policy_mlx.dataset.pusht_image_dataset import (
    PushTImageDataset,
    SequenceSampler,
    create_indices,
    _SingleFieldNormalizer,
)
from diffusion_policy_mlx.dataset.replay_buffer import ReplayBuffer
from diffusion_policy_mlx.training.collate import collate_batch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zarr_path(tmp_path):
    """Create a small synthetic zarr dataset and return its path."""
    p = str(tmp_path / "test.zarr")
    N = 200
    rng = np.random.default_rng(42)
    root = zarr.open(p, mode="w")
    root.create_array(
        "data/img",
        data=rng.integers(0, 256, (N, 96, 96, 3), dtype=np.uint8),
    )
    root.create_array(
        "data/state",
        data=rng.standard_normal((N, 5)).astype(np.float32),
    )
    root.create_array(
        "data/action",
        data=rng.standard_normal((N, 2)).astype(np.float32),
    )
    root.create_array(
        "meta/episode_ends",
        data=np.array([100, 200], dtype=np.int64),
    )
    return p


@pytest.fixture
def short_zarr_path(tmp_path):
    """Zarr with very short episodes to test boundary padding."""
    p = str(tmp_path / "short.zarr")
    N = 20
    rng = np.random.default_rng(7)
    root = zarr.open(p, mode="w")
    root.create_array(
        "data/img",
        data=rng.integers(0, 256, (N, 96, 96, 3), dtype=np.uint8),
    )
    root.create_array(
        "data/state",
        data=rng.standard_normal((N, 5)).astype(np.float32),
    )
    root.create_array(
        "data/action",
        data=rng.standard_normal((N, 2)).astype(np.float32),
    )
    # Two short episodes of length 10 each
    root.create_array(
        "meta/episode_ends",
        data=np.array([10, 20], dtype=np.int64),
    )
    return p


# ---------------------------------------------------------------------------
# PushTImageDataset tests
# ---------------------------------------------------------------------------

class TestPushTImageDatasetShapes:
    """Verify __getitem__ output shapes and types."""

    def test_basic_shapes(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        assert len(ds) > 0

        sample = ds[0]
        assert sample["obs"]["image"].shape == (16, 3, 96, 96)
        assert sample["obs"]["agent_pos"].shape == (16, 2)
        assert sample["action"].shape == (16, 2)

    def test_dtypes(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        sample = ds[0]
        assert sample["obs"]["image"].dtype == np.float32
        assert sample["obs"]["agent_pos"].dtype == np.float32
        assert sample["action"].dtype == np.float32

    def test_all_samples_valid_shape(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=8, pad_before=0, pad_after=0)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["obs"]["image"].shape == (8, 3, 96, 96)
            assert sample["obs"]["agent_pos"].shape == (8, 2)
            assert sample["action"].shape == (8, 2)


class TestPushTImageRange:
    """Images must be float32 in [0, 1]."""

    def test_image_range(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        sample = ds[0]
        img = sample["obs"]["image"]
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_image_range_multiple(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        for i in range(min(10, len(ds))):
            img = ds[i]["obs"]["image"]
            assert img.min() >= 0.0
            assert img.max() <= 1.0


class TestEpisodeBoundaryPadding:
    """Sequences at episode boundaries should be properly padded."""

    def test_first_sample_padding(self, short_zarr_path):
        """First sample with pad_before=3 should replicate the first frame."""
        ds = PushTImageDataset(
            short_zarr_path, horizon=8, pad_before=3, pad_after=0
        )
        # The first index should start at -3 relative to episode start,
        # meaning 3 frames of padding.
        sample = ds[0]
        img = sample["obs"]["image"]
        assert img.shape == (8, 3, 96, 96)
        # First 3 frames should be identical (replicated first frame)
        # (They come from padding)
        np.testing.assert_array_equal(img[0], img[1])
        np.testing.assert_array_equal(img[0], img[2])
        # Frame 3 (index 3) should differ from frame 0 in general
        # (not guaranteed but extremely likely with random data)

    def test_last_sample_padding(self, short_zarr_path):
        """Last sample of an episode with pad_after should replicate last frame."""
        ds = PushTImageDataset(
            short_zarr_path, horizon=8, pad_before=0, pad_after=3
        )
        # Get the last sample
        sample = ds[len(ds) - 1]
        img = sample["obs"]["image"]
        assert img.shape == (8, 3, 96, 96)
        # Last 3 frames should be identical (replicated last frame)
        np.testing.assert_array_equal(img[-1], img[-2])
        np.testing.assert_array_equal(img[-1], img[-3])

    def test_sequence_stays_within_episode(self, zarr_path):
        """Verify that sampled indices never cross episode boundaries."""
        episode_ends = np.array([100, 200], dtype=np.int64)
        indices = create_indices(
            episode_ends, sequence_length=16, pad_before=1, pad_after=7
        )
        for row in indices:
            buf_start, buf_end, _, _ = row
            # Must be within one episode
            if buf_start < 100:
                assert buf_end <= 100, (
                    f"Cross-boundary: {buf_start}-{buf_end}"
                )
            else:
                assert buf_start >= 100 and buf_end <= 200


class TestCreateIndices:
    """Unit tests for the index creation function."""

    def test_no_padding(self):
        episode_ends = np.array([10, 20])
        indices = create_indices(episode_ends, sequence_length=5)
        # Each episode of length 10 yields 10-5+1 = 6 sequences
        assert len(indices) == 12

    def test_with_padding(self):
        episode_ends = np.array([10])
        indices = create_indices(
            episode_ends, sequence_length=5, pad_before=2, pad_after=2
        )
        # min_start = -2, max_start = 10 - 5 + 2 = 7 => 10 sequences
        assert len(indices) == 10

    def test_episode_mask(self):
        episode_ends = np.array([10, 20])
        mask = np.array([True, False])
        indices = create_indices(
            episode_ends, sequence_length=5, episode_mask=mask
        )
        # Only first episode: 6 sequences
        assert len(indices) == 6

    def test_empty_mask(self):
        episode_ends = np.array([10, 20])
        mask = np.array([False, False])
        indices = create_indices(
            episode_ends, sequence_length=5, episode_mask=mask
        )
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    """Test normalizer fitting and round-trip."""

    def test_normalizer_returns_dict(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")
        assert "action" in normalizer
        assert "obs" in normalizer
        assert "agent_pos" in normalizer["obs"]
        assert "image" in normalizer["obs"]

    def test_normalizer_action_range(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")
        sample = ds[0]
        normed = normalizer["action"].normalize(sample["action"])
        # In-distribution actions should be roughly in [-1, 1]
        assert normed.min() >= -1.5  # some tolerance
        assert normed.max() <= 1.5

    def test_normalizer_round_trip(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")

        sample = ds[0]
        action = sample["action"]
        normed = normalizer["action"].normalize(action)
        recovered = normalizer["action"].unnormalize(normed)
        np.testing.assert_allclose(action, recovered, atol=1e-5)

    def test_normalizer_agent_pos_round_trip(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="limits")

        sample = ds[0]
        agent_pos = sample["obs"]["agent_pos"]
        normed = normalizer["obs"]["agent_pos"].normalize(agent_pos)
        recovered = normalizer["obs"]["agent_pos"].unnormalize(normed)
        np.testing.assert_allclose(agent_pos, recovered, atol=1e-5)

    def test_identity_normalizer(self):
        norm = _SingleFieldNormalizer.create_identity((3,))
        data = np.random.rand(10, 3).astype(np.float32)
        np.testing.assert_allclose(norm.normalize(data), data, atol=1e-7)
        np.testing.assert_allclose(norm.unnormalize(data), data, atol=1e-7)

    def test_gaussian_mode(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        normalizer = ds.get_normalizer(mode="gaussian")
        sample = ds[0]
        action = sample["action"]
        normed = normalizer["action"].normalize(action)
        recovered = normalizer["action"].unnormalize(normed)
        np.testing.assert_allclose(action, recovered, atol=1e-5)


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------

class TestCollateBatch:
    """Test collate_batch produces valid mx.array batches."""

    def test_collate_shapes(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        samples = [ds[i] for i in range(4)]
        batch = collate_batch(samples)
        assert isinstance(batch["obs"]["image"], mx.array)
        assert batch["obs"]["image"].shape == (4, 16, 3, 96, 96)
        assert batch["obs"]["agent_pos"].shape == (4, 16, 2)
        assert batch["action"].shape == (4, 16, 2)

    def test_collate_dtypes(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7)
        samples = [ds[i] for i in range(2)]
        batch = collate_batch(samples)
        assert batch["obs"]["image"].dtype == mx.float32
        assert batch["action"].dtype == mx.float32

    def test_collate_single_sample(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=8, pad_before=0, pad_after=0)
        batch = collate_batch([ds[0]])
        assert batch["obs"]["image"].shape == (1, 8, 3, 96, 96)

    def test_collate_empty(self):
        result = collate_batch([])
        assert result == {}

    def test_collate_preserves_values(self, zarr_path):
        ds = PushTImageDataset(zarr_path, horizon=8, pad_before=0, pad_after=0)
        sample = ds[0]
        batch = collate_batch([sample])
        np.testing.assert_allclose(
            np.asarray(batch["action"])[0],
            sample["action"],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    """Test the lightweight replay buffer."""

    def test_create_from_path(self, zarr_path):
        buf = ReplayBuffer.create_from_path(zarr_path)
        assert buf.n_episodes == 2
        assert buf.n_steps == 200

    def test_copy_from_path(self, zarr_path):
        buf = ReplayBuffer.copy_from_path(zarr_path, keys=["img", "state", "action"])
        assert buf.n_episodes == 2
        assert "img" in buf
        assert buf["img"].shape[0] == 200

    def test_get_episode(self, zarr_path):
        buf = ReplayBuffer.create_from_path(zarr_path)
        ep0 = buf.get_episode(0)
        assert ep0["img"].shape[0] == 100
        ep1 = buf.get_episode(1)
        assert ep1["img"].shape[0] == 100

    def test_episode_slice(self, zarr_path):
        buf = ReplayBuffer.create_from_path(zarr_path)
        s = buf.get_episode_slice(0)
        assert s == slice(0, 100)
        s = buf.get_episode_slice(1)
        assert s == slice(100, 200)


# ---------------------------------------------------------------------------
# SequenceSampler tests
# ---------------------------------------------------------------------------

class TestSequenceSampler:
    """Test the sequence sampler respects episode boundaries."""

    def test_sampler_length(self):
        data = {
            "x": np.arange(20).reshape(20, 1),
        }
        episode_ends = np.array([10, 20])
        sampler = SequenceSampler(
            data=data,
            episode_ends=episode_ends,
            sequence_length=5,
            pad_before=0,
            pad_after=0,
        )
        # Each episode of length 10: 6 valid starts => 12 total
        assert len(sampler) == 12

    def test_sampler_no_cross_episode(self):
        """Sequences should never contain data from two episodes."""
        data = {
            "ep_id": np.concatenate([
                np.zeros(10, dtype=np.int32),
                np.ones(10, dtype=np.int32),
            ]).reshape(20, 1),
        }
        episode_ends = np.array([10, 20])
        sampler = SequenceSampler(
            data=data,
            episode_ends=episode_ends,
            sequence_length=5,
            pad_before=2,
            pad_after=2,
        )
        for i in range(len(sampler)):
            seq = sampler.sample_sequence(i)
            vals = np.unique(seq["ep_id"])
            assert len(vals) == 1, (
                f"Sample {i} crosses episodes: {vals}"
            )

    def test_sampler_padding_values(self):
        """Padding should replicate edge frames."""
        data = {
            "x": np.arange(10).reshape(10, 1).astype(np.float32),
        }
        episode_ends = np.array([10])
        sampler = SequenceSampler(
            data=data,
            episode_ends=episode_ends,
            sequence_length=5,
            pad_before=2,
            pad_after=0,
        )
        # First sample starts at offset -2 relative to episode start
        seq = sampler.sample_sequence(0)
        x = seq["x"].flatten()
        # First two should be padded with value 0 (first frame)
        assert x[0] == 0.0
        assert x[1] == 0.0
        assert x[2] == 0.0  # actual first frame
        assert x[3] == 1.0
        assert x[4] == 2.0


# ---------------------------------------------------------------------------
# Validation split tests
# ---------------------------------------------------------------------------

class TestValidationSplit:
    """Test train/val splitting."""

    def test_val_dataset(self, zarr_path):
        ds = PushTImageDataset(
            zarr_path, horizon=8, pad_before=0, pad_after=0,
            val_ratio=0.5, seed=42,
        )
        val_ds = ds.get_validation_dataset()
        # With 2 episodes and 50% val, we get 1 train + 1 val
        train_len = len(ds)
        val_len = len(val_ds)
        # Both should be non-zero
        assert train_len > 0
        assert val_len > 0
        # Together they should cover all possible indices
        total_no_split = 2 * (100 - 8 + 1)  # 2 episodes, 93 each
        assert train_len + val_len == total_no_split

    def test_no_val(self, zarr_path):
        ds = PushTImageDataset(
            zarr_path, horizon=8, pad_before=0, pad_after=0,
            val_ratio=0.0,
        )
        val_ds = ds.get_validation_dataset()
        assert len(val_ds) == 0
