"""
MultiImageObsEncoder — multi-camera observation encoder.

Processes multiple RGB image streams through ResNet backbones and concatenates
the resulting features with low-dimensional observation features.

Input images are expected in NCHW format (B, C, H, W).
"""

from typing import Dict, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from diffusion_policy_mlx.compat.vision import (
    clone_module,
    replace_submodules,
)
from diffusion_policy_mlx.model.vision.crop_randomizer import CropRandomizer

# ImageNet normalization constants (NCHW layout for broadcasting)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _imagenet_normalize(x):
    """Normalize NCHW image from [0,1] to ImageNet stats."""
    mean = mx.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = mx.array(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (x - mean) / std


class _Resizer(nn.Module):
    """Nearest-neighbor resize for NCHW tensors."""

    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w

    def __call__(self, x):
        # x: (B, C, H, W) → transpose to NHWC for interpolation, then back
        B, C, H, W = x.shape
        if H == self.h and W == self.w:
            return x
        # MLX doesn't have F.interpolate; use a simple approach:
        # transpose to NHWC, use nearest neighbor via array indexing
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))  # (B, H, W, C)
        # Compute indices for nearest-neighbor resize
        h_idx = (mx.arange(self.h).astype(mx.float32) * (H / self.h)).astype(mx.int32)
        w_idx = (mx.arange(self.w).astype(mx.float32) * (W / self.w)).astype(mx.int32)
        x_nhwc = x_nhwc[:, h_idx][:, :, w_idx]  # (B, new_H, new_W, C)
        return mx.transpose(x_nhwc, (0, 3, 1, 2))  # back to NCHW


class MultiImageObsEncoder(nn.Module):
    """Encode multi-camera RGB + low-dim observations into a flat feature vector.

    Args:
        shape_meta: Dict with structure
            ``{'obs': {'key': {'shape': tuple, 'type': 'rgb'|'low_dim'}, ...}}``.
        rgb_model: A vision backbone (e.g. ResNet) or dict mapping keys to models.
        resize_shape: Optional ``(H, W)`` or dict per key.
        crop_shape: Optional ``(H, W)`` or dict per key.
        random_crop: Whether to use random (train) / center (eval) crop.
        use_group_norm: Replace BatchNorm with GroupNorm in rgb_model.
        share_rgb_model: Use a single shared model for all RGB inputs.
        imagenet_norm: Apply ImageNet normalization to RGB inputs.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
    ):
        super().__init__()

        self.rgb_keys = []
        self.low_dim_keys = []
        self.key_model_map = {}
        self.key_transform_map = {}  # key → list of callables
        self.key_shape_map = {}
        self.share_rgb_model = share_rgb_model
        self.imagenet_norm = imagenet_norm
        self.shape_meta = shape_meta

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            if use_group_norm:
                rgb_model = replace_submodules(
                    root_module=rgb_model,
                    predicate=lambda m: isinstance(m, nn.BatchNorm),
                    func=lambda m: nn.GroupNorm(
                        num_groups=(
                            max(1, m.num_features // 16) if hasattr(m, "num_features") else 4
                        ),
                        dims=m.num_features if hasattr(m, "num_features") else m.weight.shape[0],
                        pytorch_compatible=True,
                    ),
                )
            self.key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            self.key_shape_map[key] = shape

            if obs_type == "rgb":
                self.rgb_keys.append(key)

                # Configure model
                if not share_rgb_model:
                    # Note: MLX nn.Module subclasses dict, so check Module first
                    if isinstance(rgb_model, nn.Module):
                        this_model = clone_module(rgb_model)
                    elif isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        raise TypeError(
                            f"rgb_model must be nn.Module or dict, got {type(rgb_model)}"
                        )

                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda m: isinstance(m, nn.BatchNorm),
                            func=lambda m: nn.GroupNorm(
                                num_groups=(
                                    max(1, m.num_features // 16)
                                    if hasattr(m, "num_features")
                                    else 4
                                ),
                                dims=(
                                    m.num_features
                                    if hasattr(m, "num_features")
                                    else m.weight.shape[0]
                                ),
                                pytorch_compatible=True,
                            ),
                        )
                    self.key_model_map[key] = this_model

                # Build transform pipeline
                transforms = []
                input_shape = shape

                # Resize
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    transforms.append(_Resizer(h, w))
                    input_shape = (shape[0], h, w)

                # Crop
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        transforms.append(
                            CropRandomizer(
                                input_shape=input_shape,
                                crop_height=h,
                                crop_width=w,
                                num_crops=1,
                                pos_enc=False,
                            )
                        )
                    else:
                        # Static center crop via CropRandomizer in eval mode
                        cr = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                        cr.eval()
                        transforms.append(cr)

                self.key_transform_map[key] = transforms

            elif obs_type == "low_dim":
                self.low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        self.rgb_keys = sorted(self.rgb_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

    def __call__(self, obs_dict):
        """
        Args:
            obs_dict: Dict mapping observation keys to arrays.
                RGB keys: (B, C, H, W).
                Low-dim keys: (B, D).

        Returns:
            (B, total_feature_dim) concatenated feature vector.
        """
        batch_size = None
        features = []

        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                # Apply transforms
                for t in self.key_transform_map.get(key, []):
                    img = t(img)
                # ImageNet normalization
                if self.imagenet_norm:
                    img = _imagenet_normalize(img)
                imgs.append(img)

            # Stack all cameras: (N*B, C, H, W)
            imgs = mx.concatenate(imgs, axis=0)
            feature = self.key_model_map["rgb"](imgs)
            # (N*B, D) → (N, B, D) → (B, N, D) → (B, N*D)
            n_cams = len(self.rgb_keys)
            feature = feature.reshape(n_cams, batch_size, -1)
            feature = mx.transpose(feature, (1, 0, 2))
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                # Apply transforms
                for t in self.key_transform_map.get(key, []):
                    img = t(img)
                # ImageNet normalization
                if self.imagenet_norm:
                    img = _imagenet_normalize(img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)

        result = mx.concatenate(features, axis=-1)
        return result

    def output_shape(self):
        """Compute total output feature dimension by doing a dummy forward pass."""
        obs_shape_meta = self.shape_meta["obs"]
        example_obs = {}
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            example_obs[key] = mx.zeros((1,) + shape)
        out = self.__call__(example_obs)
        return tuple(out.shape[1:])
