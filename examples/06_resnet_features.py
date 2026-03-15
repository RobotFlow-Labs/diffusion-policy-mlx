"""Extract visual features from images using the MLX ResNet backbone.

Shows how the vision encoder processes observations and produces
feature vectors for conditioning the diffusion policy.
"""

import time

import mlx.core as mx
import numpy as np

from diffusion_policy_mlx.model.vision.model_getter import get_resnet
from diffusion_policy_mlx.model.vision.multi_image_obs_encoder import (
    MultiImageObsEncoder,
)


def main():
    print("=== ResNet Feature Extraction in MLX ===\n")
    mx.random.seed(42)

    # 1. Create a ResNet18 feature extractor
    print("1. Building ResNet18 backbone...")
    resnet = get_resnet("resnet18")  # fc replaced with Identity
    mx.eval(resnet.parameters())

    import mlx.utils
    flat_params = list(mlx.utils.tree_flatten(resnet.parameters()))
    param_count = sum(p.size for _, p in flat_params)
    print(f"   Parameters: {param_count:,}")

    # 2. Create a synthetic image (gradient pattern, NCHW format)
    print("\n2. Creating synthetic test images...")
    B = 4  # batch size
    H, W = 96, 96

    # Generate a gradient image (more interesting than pure noise)
    y_grid = np.linspace(0, 1, H).reshape(1, 1, H, 1).repeat(B, axis=0).repeat(W, axis=3)
    x_grid = np.linspace(0, 1, W).reshape(1, 1, 1, W).repeat(B, axis=0).repeat(H, axis=2)
    images_np = np.stack([
        y_grid[:, 0],             # R channel: vertical gradient
        x_grid[:, 0],             # G channel: horizontal gradient
        (y_grid[:, 0] + x_grid[:, 0]) / 2,  # B channel: diagonal gradient
    ], axis=1).astype(np.float32)

    images = mx.array(images_np)  # (B, 3, H, W) -- NCHW
    print(f"   Input shape: {images.shape}  (batch, channels, height, width)")

    # 3. Extract features with raw ResNet
    print("\n3. Extracting features with ResNet18...")
    start = time.perf_counter()
    features = resnet(images)
    mx.eval(features)
    elapsed = time.perf_counter() - start

    print(f"   Output shape: {features.shape}  (batch, feature_dim)")
    print(f"   Feature dim:  {features.shape[1]}  (512 for ResNet18)")
    print(f"   Time:         {elapsed*1000:.1f}ms")
    print(f"   Feature stats: mean={float(mx.mean(features)):.4f}, "
          f"std={float(mx.std(features)):.4f}, "
          f"min={float(mx.min(features)):.4f}, "
          f"max={float(mx.max(features)):.4f}")

    # 4. Use the full MultiImageObsEncoder (as used in the policy)
    print("\n4. Using MultiImageObsEncoder (full observation pipeline)...")
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
    }
    obs_encoder = MultiImageObsEncoder(
        shape_meta=shape_meta,
        rgb_model=get_resnet("resnet18"),
        crop_shape=(76, 76),
        share_rgb_model=True,
        imagenet_norm=True,
    )
    mx.eval(obs_encoder.parameters())

    # Simulate a single-timestep observation
    obs = {
        "image": images,                          # (B, 3, 96, 96)
        "agent_pos": mx.random.normal((B, 2)),    # (B, 2)
    }

    start = time.perf_counter()
    encoded = obs_encoder(obs)
    mx.eval(encoded)
    elapsed = time.perf_counter() - start

    print(f"   Output shape: {encoded.shape}  (batch, rgb_features + low_dim)")
    print(f"   Total dim:    {encoded.shape[1]}  (512 from ResNet + 2 from agent_pos = 514)")
    print(f"   Time:         {elapsed*1000:.1f}ms")

    # 5. Show the output_shape helper
    computed_shape = obs_encoder.output_shape()
    print(f"\n5. Output shape from output_shape(): {computed_shape}")

    print("\nDone! The ResNet extracts 512-dim features from 96x96 images.")


if __name__ == "__main__":
    main()
