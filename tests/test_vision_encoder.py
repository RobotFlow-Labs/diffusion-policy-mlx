"""Tests for PRD-02: Vision Encoder (ResNet + MultiImageObsEncoder + CropRandomizer)."""

import mlx.core as mx
import numpy as np
import pytest

from diffusion_policy_mlx.compat.vision import (
    Identity,
    load_torchvision_weights,
    replace_submodules,
    resnet18,
    resnet34,
    resnet50,
)
from diffusion_policy_mlx.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy_mlx.model.vision.model_getter import get_resnet
from diffusion_policy_mlx.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

try:
    import torch
    import torchvision

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# -----------------------------------------------------------------------
# ResNet shape tests
# -----------------------------------------------------------------------


class TestResNet18Shape:
    def test_output_shape_224(self):
        """ResNet18 feature extractor: (B,3,224,224) -> (B,512)."""
        model = resnet18()
        model.fc = Identity()
        x = mx.random.normal((2, 3, 224, 224))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"

    def test_output_shape_small_input(self):
        """ResNet18 with PushT-sized 76x76 crops."""
        model = resnet18()
        model.fc = Identity()
        x = mx.random.normal((2, 3, 76, 76))
        out = model(x)
        mx.eval(out)
        assert out.shape[0] == 2
        assert out.ndim == 2, f"Expected 2D output, got {out.ndim}D"
        # 76x76 → after stride-2 conv + stride-2 maxpool → 19x19
        # then 4 stages with strides [1,2,2,2] → 19→19→10→5→3
        # adaptive avg pool → 512
        assert out.shape[1] == 512


class TestResNet50Shape:
    def test_output_shape_224(self):
        """ResNet50 feature extractor: (B,3,224,224) -> (B,2048)."""
        model = resnet50()
        model.fc = Identity()
        x = mx.random.normal((2, 3, 224, 224))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 2048), f"Expected (2, 2048), got {out.shape}"


class TestResNet34Shape:
    def test_output_shape_224(self):
        """ResNet34 feature extractor: (B,3,224,224) -> (B,512)."""
        model = resnet34()
        model.fc = Identity()
        x = mx.random.normal((2, 3, 224, 224))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 512), f"Expected (2, 512), got {out.shape}"


# -----------------------------------------------------------------------
# Cross-framework numerical test
# -----------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch/torchvision not installed")
class TestCrossFramework:
    def test_resnet18_matches_torchvision(self):
        """With same weights, MLX ResNet18 output should match PyTorch within tolerance."""
        # Build torch model
        torch_model = torchvision.models.resnet18(weights=None)
        torch_model.fc = torch.nn.Identity()
        torch_model.eval()

        # Build MLX model
        mlx_model = resnet18()
        mlx_model.fc = Identity()

        # Copy weights torch → MLX
        state_dict = torch_model.state_dict()
        load_torchvision_weights(mlx_model, state_dict)
        mlx_model.eval()  # Must match torch eval mode (BatchNorm uses running stats)

        # Generate same input
        rng = np.random.RandomState(42)
        x_np = rng.randn(2, 3, 224, 224).astype(np.float32)

        # Torch forward
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(x_np)).numpy()

        # MLX forward
        mlx_out = np.array(mlx_model(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_resnet34_matches_torchvision(self):
        """With same weights, MLX ResNet34 output should match PyTorch within tolerance."""
        torch_model = torchvision.models.resnet34(weights=None)
        torch_model.fc = torch.nn.Identity()
        torch_model.eval()

        mlx_model = resnet34()
        mlx_model.fc = Identity()

        state_dict = torch_model.state_dict()
        load_torchvision_weights(mlx_model, state_dict)
        mlx_model.eval()

        rng = np.random.RandomState(42)
        x_np = rng.randn(2, 3, 224, 224).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_model(torch.tensor(x_np)).numpy()

        mlx_out = np.array(mlx_model(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)

    def test_resnet50_matches_torchvision(self):
        """With same weights, MLX ResNet50 output should match PyTorch within tolerance."""
        torch_model = torchvision.models.resnet50(weights=None)
        torch_model.fc = torch.nn.Identity()
        torch_model.eval()

        mlx_model = resnet50()
        mlx_model.fc = Identity()

        state_dict = torch_model.state_dict()
        load_torchvision_weights(mlx_model, state_dict)
        mlx_model.eval()  # Must match torch eval mode

        rng = np.random.RandomState(42)
        x_np = rng.randn(2, 3, 224, 224).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_model(torch.tensor(x_np)).numpy()

        mlx_out = np.array(mlx_model(mx.array(x_np)))

        np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)


# -----------------------------------------------------------------------
# Model getter tests
# -----------------------------------------------------------------------


class TestModelGetter:
    def test_get_resnet18(self):
        """get_resnet returns feature extractor with Identity fc."""
        model = get_resnet("resnet18", weights=None)
        x = mx.random.normal((1, 3, 224, 224))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 512)

    def test_get_resnet_invalid_name(self):
        with pytest.raises(ValueError):
            get_resnet("resnet101")


# -----------------------------------------------------------------------
# CropRandomizer tests
# -----------------------------------------------------------------------


class TestCropRandomizer:
    def test_train_mode_shape(self):
        """Random crop in training mode produces correct output shape."""
        cr = CropRandomizer((3, 96, 96), crop_height=76, crop_width=76)
        cr.train()
        x = mx.random.normal((4, 3, 96, 96))
        out = cr(x)
        mx.eval(out)
        assert out.shape == (4, 3, 76, 76), f"Expected (4, 3, 76, 76), got {out.shape}"

    def test_eval_mode_shape(self):
        """Center crop in eval mode produces correct output shape."""
        cr = CropRandomizer((3, 96, 96), crop_height=76, crop_width=76)
        cr.eval()
        x = mx.random.normal((4, 3, 96, 96))
        out = cr(x)
        mx.eval(out)
        assert out.shape == (4, 3, 76, 76), f"Expected (4, 3, 76, 76), got {out.shape}"

    def test_eval_mode_deterministic(self):
        """Center crop should be deterministic."""
        cr = CropRandomizer((3, 96, 96), crop_height=76, crop_width=76)
        cr.eval()
        x = mx.random.normal((2, 3, 96, 96))
        out1 = cr(x)
        out2 = cr(x)
        mx.eval(out1, out2)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))

    def test_train_mode_varies(self):
        """Random crop in train mode should produce different results (probabilistically)."""
        cr = CropRandomizer((3, 96, 96), crop_height=76, crop_width=76)
        cr.train()
        x = mx.random.normal((4, 3, 96, 96))
        out1 = cr(x)
        out2 = cr(x)
        mx.eval(out1, out2)
        # Very unlikely to be exactly the same
        assert not np.array_equal(np.array(out1), np.array(out2))

    def test_multi_crop(self):
        """num_crops > 1 should multiply the batch dimension."""
        cr = CropRandomizer((3, 96, 96), crop_height=76, crop_width=76, num_crops=3)
        cr.train()
        x = mx.random.normal((4, 3, 96, 96))
        out = cr(x)
        mx.eval(out)
        assert out.shape == (12, 3, 76, 76), f"Expected (12, 3, 76, 76), got {out.shape}"


# -----------------------------------------------------------------------
# MultiImageObsEncoder tests
# -----------------------------------------------------------------------


class TestMultiImageObsEncoder:
    def test_single_camera_with_lowdim(self):
        """Single RGB camera + low-dim agent_pos → (B, D) output."""
        shape_meta = {
            "obs": {
                "image": {"shape": (3, 96, 96), "type": "rgb"},
                "agent_pos": {"shape": (2,), "type": "low_dim"},
            }
        }
        rgb_model = resnet18()
        rgb_model.fc = Identity()

        enc = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=(76, 76),
            random_crop=True,
        )

        obs = {
            "image": mx.random.normal((4, 3, 96, 96)),
            "agent_pos": mx.random.normal((4, 2)),
        }
        out = enc(obs)
        mx.eval(out)
        assert out.ndim == 2
        assert out.shape[0] == 4
        # Should be 512 (resnet18 features) + 2 (agent_pos) = 514
        assert out.shape[1] == 514, f"Expected 514, got {out.shape[1]}"

    def test_output_shape_method(self):
        """output_shape() should return the feature dimension."""
        shape_meta = {
            "obs": {
                "image": {"shape": (3, 96, 96), "type": "rgb"},
                "agent_pos": {"shape": (2,), "type": "low_dim"},
            }
        }
        rgb_model = resnet18()
        rgb_model.fc = Identity()

        enc = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=(76, 76),
            random_crop=False,  # use center crop for deterministic shape
        )
        out_shape = enc.output_shape()
        assert out_shape == (514,), f"Expected (514,), got {out_shape}"

    def test_shared_rgb_model(self):
        """share_rgb_model=True should work with multiple cameras."""
        shape_meta = {
            "obs": {
                "camera0": {"shape": (3, 96, 96), "type": "rgb"},
                "camera1": {"shape": (3, 96, 96), "type": "rgb"},
                "agent_pos": {"shape": (2,), "type": "low_dim"},
            }
        }
        rgb_model = resnet18()
        rgb_model.fc = Identity()

        enc = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=(76, 76),
            share_rgb_model=True,
        )

        obs = {
            "camera0": mx.random.normal((2, 3, 96, 96)),
            "camera1": mx.random.normal((2, 3, 96, 96)),
            "agent_pos": mx.random.normal((2, 2)),
        }
        out = enc(obs)
        mx.eval(out)
        assert out.shape[0] == 2
        # 2 cameras * 512 + 2 = 1026
        assert out.shape[1] == 1026, f"Expected 1026, got {out.shape[1]}"

    def test_no_crop(self):
        """Encoder without cropping should still work."""
        shape_meta = {
            "obs": {
                "image": {"shape": (3, 96, 96), "type": "rgb"},
            }
        }
        rgb_model = resnet18()
        rgb_model.fc = Identity()

        enc = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
        )

        obs = {"image": mx.random.normal((2, 3, 96, 96))}
        out = enc(obs)
        mx.eval(out)
        assert out.shape == (2, 512)


# -----------------------------------------------------------------------
# replace_submodules tests
# -----------------------------------------------------------------------


class TestReplaceSubmodules:
    def test_replace_batchnorm(self):
        """replace_submodules can swap BatchNorm for GroupNorm in ResNet."""
        import mlx.nn as nn

        model = resnet18()

        # Count BatchNorm modules before
        bn_count_before = 0
        for name, m in model.leaf_modules().items():
            if isinstance(m, nn.BatchNorm):
                bn_count_before += 1
        assert bn_count_before > 0, "ResNet18 should have BatchNorm layers"

        # Replace
        model = replace_submodules(
            root_module=model,
            predicate=lambda m: isinstance(m, nn.BatchNorm),
            func=lambda m: nn.GroupNorm(
                num_groups=max(1, m.weight.shape[0] // 16),
                dims=m.weight.shape[0],
            ),
        )

        # Count BatchNorm after — should be 0
        bn_count_after = 0
        for name, m in model.leaf_modules().items():
            if isinstance(m, nn.BatchNorm):
                bn_count_after += 1
        assert bn_count_after == 0, f"Expected 0 BatchNorm after replace, got {bn_count_after}"

        # Model should still work
        model.fc = Identity()
        x = mx.random.normal((1, 3, 224, 224))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 512)
