"""Tests for PyTorch -> MLX weight conversion.

Tests cover:
  - Conv2d weight shape transposition: (64,3,7,7) -> (64,7,7,3)
  - Conv1d weight shape transposition: (256,128,5) -> (256,5,128)
  - Linear weights unchanged
  - Full state dict conversion with mixed param types
  - Shape-based matcher pairing by shape
  - Key mapping for UNet, ResNet, and obs encoder
  - Round-trip: dummy torch state dict -> convert -> verify shapes
"""

import os
import sys

import numpy as np
import pytest

# Ensure scripts/ is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "scripts"),
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


from convert_weights import (
    _is_conv1d_weight,
    _is_conv2d_weight,
    convert_state_dict,
    extract_normalizer,
    map_key_path,
    map_obs_encoder_key,
    map_resnet_key,
    map_unet_key,
    shape_based_match,
    transpose_conv1d,
    transpose_conv2d,
)

# ---------------------------------------------------------------------------
# Shape transposition tests
# ---------------------------------------------------------------------------


class TestConv2dTransposition:
    """Test Conv2d weight transposition OIHW -> OHWI."""

    def test_basic_shape(self):
        w = np.random.randn(64, 3, 7, 7).astype(np.float32)
        result = transpose_conv2d(w)
        assert result.shape == (64, 7, 7, 3)

    def test_values_preserved(self):
        w = np.random.randn(16, 8, 3, 3).astype(np.float32)
        result = transpose_conv2d(w)
        # result[o, h, w, i] == w[o, i, h, w]
        np.testing.assert_array_equal(result[0, 1, 2, 3], w[0, 3, 1, 2])

    def test_1x1_conv(self):
        w = np.random.randn(128, 64, 1, 1).astype(np.float32)
        result = transpose_conv2d(w)
        assert result.shape == (128, 1, 1, 64)

    def test_detection(self):
        assert _is_conv2d_weight("conv1.weight", np.zeros((64, 3, 7, 7)))
        assert not _is_conv2d_weight("conv1.bias", np.zeros((64,)))
        assert not _is_conv2d_weight("conv1.weight", np.zeros((64, 3, 7)))


class TestConv1dTransposition:
    """Test Conv1d weight transposition OIK -> OKI."""

    def test_basic_shape(self):
        w = np.random.randn(256, 128, 5).astype(np.float32)
        result = transpose_conv1d(w)
        assert result.shape == (256, 5, 128)

    def test_values_preserved(self):
        w = np.random.randn(32, 16, 3).astype(np.float32)
        result = transpose_conv1d(w)
        # result[o, k, i] == w[o, i, k]
        np.testing.assert_array_equal(result[0, 1, 2], w[0, 2, 1])

    def test_kernel_1(self):
        w = np.random.randn(64, 32, 1).astype(np.float32)
        result = transpose_conv1d(w)
        assert result.shape == (64, 1, 32)

    def test_detection(self):
        assert _is_conv1d_weight("conv.weight", np.zeros((256, 128, 5)))
        assert not _is_conv1d_weight("conv.bias", np.zeros((256,)))
        assert not _is_conv1d_weight("embedding.weight", np.zeros((100, 50, 3)))


class TestLinearUnchanged:
    """Test that linear weights are not transposed."""

    def test_2d_weight_not_transposed(self):
        """Linear weights (2D) should pass through unchanged."""
        state = {"linear.weight": np.random.randn(64, 32).astype(np.float32)}
        # Wrap in torch tensor if available
        if HAS_TORCH:
            state = {k: torch.tensor(v) for k, v in state.items()}

        result = convert_state_dict(state, skip_normalizer=False)
        assert "linear.weight" in result
        assert result["linear.weight"].shape == (64, 32)

    def test_1d_bias_not_transposed(self):
        """Bias (1D) should pass through unchanged."""
        state = {"linear.bias": np.random.randn(64).astype(np.float32)}
        if HAS_TORCH:
            state = {k: torch.tensor(v) for k, v in state.items()}

        result = convert_state_dict(state, skip_normalizer=False)
        assert "linear.bias" in result
        assert result["linear.bias"].shape == (64,)


# ---------------------------------------------------------------------------
# Full state dict conversion
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestConvertStateDict:
    """Test full state dict conversion."""

    def test_conv2d_weight_converted(self):
        state = {"conv.weight": torch.randn(64, 3, 7, 7)}
        result = convert_state_dict(state, skip_normalizer=False)
        assert "conv.weight" in result
        assert result["conv.weight"].shape == (64, 7, 7, 3)

    def test_conv1d_weight_converted(self):
        state = {"conv.weight": torch.randn(256, 128, 5)}
        result = convert_state_dict(state, skip_normalizer=False)
        assert "conv.weight" in result
        assert result["conv.weight"].shape == (256, 5, 128)

    def test_mixed_params(self):
        state = {
            "conv2d.weight": torch.randn(64, 3, 7, 7),
            "conv1d.weight": torch.randn(256, 128, 5),
            "linear.weight": torch.randn(512, 256),
            "bn.weight": torch.randn(64),
            "bn.bias": torch.randn(64),
        }
        result = convert_state_dict(state, skip_normalizer=False)
        assert result["conv2d.weight"].shape == (64, 7, 7, 3)
        assert result["conv1d.weight"].shape == (256, 5, 128)
        assert result["linear.weight"].shape == (512, 256)
        assert result["bn.weight"].shape == (64,)
        assert result["bn.bias"].shape == (64,)

    def test_normalizer_skipped(self):
        state = {
            "normalizer.params_dict.action.scale": torch.randn(2),
            "model.weight": torch.randn(64, 32),
        }
        result = convert_state_dict(state, skip_normalizer=True)
        assert "model.weight" in result
        assert all("normalizer" not in k for k in result)

    def test_normalizer_included(self):
        state = {
            "normalizer.params_dict.action.scale": torch.randn(2),
            "model.weight": torch.randn(64, 32),
        }
        result = convert_state_dict(state, skip_normalizer=False)
        assert "model.weight" in result
        # normalizer keys are mapped via map_key_path which returns None
        # (they're intended to be extracted separately)

    def test_num_batches_tracked_skipped(self):
        state = {
            "bn.weight": torch.randn(64),
            "bn.num_batches_tracked": torch.tensor(100),
        }
        result = convert_state_dict(state, skip_normalizer=False)
        assert "bn.weight" in result
        assert all("num_batches_tracked" not in k for k in result)


# ---------------------------------------------------------------------------
# Normalizer extraction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestExtractNormalizer:
    def test_extracts_normalizer_keys(self):
        state = {
            "normalizer.params_dict.action.scale": torch.tensor([1.0, 2.0]),
            "normalizer.params_dict.action.offset": torch.tensor([0.0, 0.5]),
            "model.weight": torch.randn(64, 32),
        }
        norm = extract_normalizer(state)
        assert len(norm) == 2
        assert "normalizer.params_dict.action.scale" in norm
        assert "normalizer.params_dict.action.offset" in norm

    def test_empty_when_no_normalizer(self):
        state = {"model.weight": torch.randn(64, 32)}
        norm = extract_normalizer(state)
        assert len(norm) == 0


# ---------------------------------------------------------------------------
# Key mapping tests
# ---------------------------------------------------------------------------


class TestUnetKeyMapping:
    """Test key mapping for ConditionalUnet1D."""

    def test_diffusion_step_encoder_linear1(self):
        result = map_unet_key("diffusion_step_encoder.1.weight")
        assert result == "diffusion_step_encoder_linear1.weight"

    def test_diffusion_step_encoder_linear2(self):
        result = map_unet_key("diffusion_step_encoder.3.bias")
        assert result == "diffusion_step_encoder_linear2.bias"

    def test_final_conv_block(self):
        result = map_unet_key("final_conv.0.block.0.weight")
        assert result == "final_block.conv._conv.weight"

    def test_final_conv_groupnorm(self):
        result = map_unet_key("final_conv.0.block.1.weight")
        assert result == "final_block.group_norm._norm.weight"

    def test_final_conv_last(self):
        result = map_unet_key("final_conv.1.weight")
        assert result == "final_conv._conv.weight"

    def test_conv1d_block_in_resblock(self):
        # down_modules.0.0.blocks.0.block.0.weight -> Conv1d in first Conv1dBlock
        result = map_unet_key("down_modules.0.0.blocks.0.block.0.weight")
        assert result == "down_modules.0.0.blocks.0.conv._conv.weight"

    def test_groupnorm_in_resblock(self):
        result = map_unet_key("down_modules.0.0.blocks.0.block.1.weight")
        assert result == "down_modules.0.0.blocks.0.group_norm._norm.weight"

    def test_cond_encoder(self):
        result = map_unet_key("down_modules.0.0.cond_encoder.1.weight")
        assert result == "down_modules.0.0.cond_linear.weight"

    def test_residual_conv(self):
        result = map_unet_key("down_modules.0.0.residual_conv.weight")
        assert result == "down_modules.0.0.residual_conv._conv.weight"

    def test_residual_conv_bias(self):
        result = map_unet_key("down_modules.0.0.residual_conv.bias")
        assert result == "down_modules.0.0.residual_conv._conv.bias"

    def test_downsample_conv(self):
        result = map_unet_key("down_modules.0.2.conv.weight")
        assert result == "down_modules.0.2.conv._conv.weight"

    def test_upsample_conv(self):
        result = map_unet_key("up_modules.1.2.conv.weight")
        assert result == "up_modules.1.2.conv._conv.weight"

    def test_mid_modules(self):
        result = map_unet_key("mid_modules.0.blocks.0.block.0.weight")
        assert result == "mid_modules.0.blocks.0.conv._conv.weight"


class TestResnetKeyMapping:
    """Test key mapping for ResNet."""

    def test_downsample_conv(self):
        assert map_resnet_key("layer1.0.downsample.0.weight") == "layer1.0.downsample.conv.weight"

    def test_downsample_bn(self):
        assert map_resnet_key("layer1.0.downsample.1.weight") == "layer1.0.downsample.bn.weight"

    def test_regular_key(self):
        assert map_resnet_key("conv1.weight") == "conv1.weight"

    def test_num_batches_tracked_empty(self):
        assert map_resnet_key("bn1.num_batches_tracked") == ""


class TestObsEncoderKeyMapping:
    """Test key mapping for observation encoder."""

    def test_robomimic_pattern(self):
        key = "obs_encoder.obs_nets.image.nets.0.nets.conv1.weight"
        result = map_obs_encoder_key(key)
        assert result == "obs_encoder.key_model_map.image.conv1.weight"

    def test_robomimic_with_nets_prefix(self):
        key = "obs_encoder.nets.obs_nets.image.nets.0.nets.conv1.weight"
        result = map_obs_encoder_key(key)
        assert result == "obs_encoder.key_model_map.image.conv1.weight"

    def test_randomizer_skipped(self):
        key = "obs_encoder.obs_randomizers.image.crop_module.weight"
        result = map_obs_encoder_key(key)
        assert result is None

    def test_resnet_downsample_mapping(self):
        key = "obs_encoder.obs_nets.image.nets.0.nets.layer1.0.downsample.0.weight"
        result = map_obs_encoder_key(key)
        assert result == "obs_encoder.key_model_map.image.layer1.0.downsample.conv.weight"


class TestFullKeyPath:
    """Test the top-level map_key_path function."""

    def test_model_prefix(self):
        result = map_key_path("model.diffusion_step_encoder.1.weight")
        assert result == "model.diffusion_step_encoder_linear1.weight"

    def test_normalizer_returns_none(self):
        result = map_key_path("normalizer.params_dict.action.scale")
        assert result is None

    def test_num_batches_tracked_returns_none(self):
        result = map_key_path("obs_encoder.bn1.num_batches_tracked")
        assert result is None


# ---------------------------------------------------------------------------
# Shape-based matcher
# ---------------------------------------------------------------------------


class TestShapeBasedMatcher:
    """Test shape-based parameter matching."""

    def test_unique_shapes(self):
        source = {
            "a": np.zeros((64, 3, 7, 7)),
            "b": np.zeros((128, 64, 3, 3)),
            "c": np.zeros((256,)),
        }
        target = {
            "x": ("x", (64, 3, 7, 7)),
            "y": ("y", (128, 64, 3, 3)),
            "z": ("z", (256,)),
        }
        mapping = shape_based_match(source, target)
        assert mapping == {"a": "x", "b": "y", "c": "z"}

    def test_duplicate_shapes_ordered(self):
        """When multiple params have the same shape, match by order."""
        source = {
            "a1": np.zeros((64,)),
            "a2": np.zeros((64,)),
        }
        target = {
            "b1": ("b1", (64,)),
            "b2": ("b2", (64,)),
        }
        mapping = shape_based_match(source, target)
        assert mapping["a1"] == "b1"
        assert mapping["a2"] == "b2"

    def test_missing_target_shape(self):
        """Source shapes not present in target should be unmapped."""
        source = {
            "a": np.zeros((64, 3, 7, 7)),
            "b": np.zeros((999,)),
        }
        target = {
            "x": ("x", (64, 3, 7, 7)),
        }
        mapping = shape_based_match(source, target)
        assert "a" in mapping
        assert "b" not in mapping

    def test_empty_inputs(self):
        mapping = shape_based_match({}, {})
        assert mapping == {}


# ---------------------------------------------------------------------------
# Round-trip test: create dummy torch state dict -> convert -> verify
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestRoundTrip:
    """Round-trip conversion test."""

    def test_dummy_unet_state_dict(self):
        """Create a dummy UNet-like state dict, convert, verify shapes."""
        state = {
            # Diffusion step encoder
            "model.diffusion_step_encoder.1.weight": torch.randn(1024, 256),
            "model.diffusion_step_encoder.1.bias": torch.randn(1024),
            "model.diffusion_step_encoder.3.weight": torch.randn(256, 1024),
            "model.diffusion_step_encoder.3.bias": torch.randn(256),
            # Down module Conv1dBlock (Conv1d weight)
            "model.down_modules.0.0.blocks.0.block.0.weight": torch.randn(256, 2, 5),
            "model.down_modules.0.0.blocks.0.block.0.bias": torch.randn(256),
            "model.down_modules.0.0.blocks.0.block.1.weight": torch.randn(256),
            "model.down_modules.0.0.blocks.0.block.1.bias": torch.randn(256),
            # Cond encoder
            "model.down_modules.0.0.cond_encoder.1.weight": torch.randn(256, 256),
            "model.down_modules.0.0.cond_encoder.1.bias": torch.randn(256),
            # Residual conv
            "model.down_modules.0.0.residual_conv.weight": torch.randn(256, 2, 1),
            "model.down_modules.0.0.residual_conv.bias": torch.randn(256),
            # Final conv
            "model.final_conv.0.block.0.weight": torch.randn(256, 256, 5),
            "model.final_conv.0.block.1.weight": torch.randn(256),
            "model.final_conv.1.weight": torch.randn(2, 256, 1),
            "model.final_conv.1.bias": torch.randn(2),
            # Normalizer (should be skipped)
            "normalizer.params_dict.action.scale": torch.tensor([1.0, 2.0]),
        }

        result = convert_state_dict(state, skip_normalizer=True)

        # Check key mapping
        assert "model.diffusion_step_encoder_linear1.weight" in result
        assert "model.diffusion_step_encoder_linear2.weight" in result
        assert "model.down_modules.0.0.blocks.0.conv._conv.weight" in result
        assert "model.down_modules.0.0.blocks.0.group_norm._norm.weight" in result
        assert "model.down_modules.0.0.cond_linear.weight" in result
        assert "model.down_modules.0.0.residual_conv._conv.weight" in result
        assert "model.final_block.conv._conv.weight" in result
        assert "model.final_block.group_norm._norm.weight" in result
        assert "model.final_conv._conv.weight" in result

        # Check shapes
        # Conv1d: (256, 2, 5) -> (256, 5, 2)
        assert result["model.down_modules.0.0.blocks.0.conv._conv.weight"].shape == (256, 5, 2)
        # Residual conv: (256, 2, 1) -> (256, 1, 2)
        assert result["model.down_modules.0.0.residual_conv._conv.weight"].shape == (256, 1, 2)
        # Final conv block: (256, 256, 5) -> (256, 5, 256)
        assert result["model.final_block.conv._conv.weight"].shape == (256, 5, 256)
        # Final conv: (2, 256, 1) -> (2, 1, 256)
        assert result["model.final_conv._conv.weight"].shape == (2, 1, 256)
        # Linear: unchanged
        assert result["model.diffusion_step_encoder_linear1.weight"].shape == (1024, 256)
        # GroupNorm: 1D, unchanged
        assert result["model.down_modules.0.0.blocks.0.group_norm._norm.weight"].shape == (256,)

        # Normalizer not in result
        assert all("normalizer" not in k for k in result)

    def test_dummy_resnet_state_dict(self):
        """Create a dummy ResNet obs encoder state dict, convert, verify."""
        state = {
            "obs_encoder.obs_nets.image.nets.0.nets.conv1.weight": torch.randn(64, 3, 7, 7),
            "obs_encoder.obs_nets.image.nets.0.nets.bn1.weight": torch.randn(64),
            "obs_encoder.obs_nets.image.nets.0.nets.bn1.bias": torch.randn(64),
            "obs_encoder.obs_nets.image.nets.0.nets.layer1.0.conv1.weight": torch.randn(
                64, 64, 3, 3
            ),
            "obs_encoder.obs_nets.image.nets.0.nets.layer1.0.downsample.0.weight": torch.randn(
                64, 64, 1, 1
            ),
            "obs_encoder.obs_nets.image.nets.0.nets.layer1.0.downsample.1.weight": torch.randn(64),
            "obs_encoder.obs_nets.image.nets.0.nets.bn1.num_batches_tracked": torch.tensor(100),
        }

        result = convert_state_dict(state, skip_normalizer=False)

        # Conv2d: (64, 3, 7, 7) -> (64, 7, 7, 3)
        k = "obs_encoder.key_model_map.image.conv1.weight"
        assert k in result
        assert result[k].shape == (64, 7, 7, 3)

        # Downsample conv: (64, 64, 1, 1) -> (64, 1, 1, 64)
        k = "obs_encoder.key_model_map.image.layer1.0.downsample.conv.weight"
        assert k in result
        assert result[k].shape == (64, 1, 1, 64)

        # Downsample bn
        k = "obs_encoder.key_model_map.image.layer1.0.downsample.bn.weight"
        assert k in result
        assert result[k].shape == (64,)

        # num_batches_tracked should be skipped
        assert all("num_batches_tracked" not in k for k in result)
