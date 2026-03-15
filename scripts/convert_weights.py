"""Convert PyTorch Diffusion Policy checkpoint to MLX format.

Usage:
    python scripts/convert_weights.py \
        --checkpoint path/to/pytorch.ckpt \
        --output checkpoints/pusht_mlx

Handles:
  - Policy model weights (UNet + ResNet obs encoder)
  - EMA model weights
  - Normalizer state (scale/offset)
  - Conv2d weight transposition: (C_out, C_in, H, W) -> (C_out, H, W, C_in)
  - Conv1d weight transposition: (C_out, C_in, K) -> (C_out, K, C_in)
  - Key path mapping: upstream module hierarchy -> our MLX hierarchy
  - Shape-based fallback matching when name mapping fails
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key mapping rules: upstream PyTorch -> our MLX hierarchy
# ---------------------------------------------------------------------------

# Rules for the ConditionalUnet1D model subtree.
# Upstream uses nn.Sequential / nn.ModuleList, ours uses named attributes.

_UNET_KEY_RULES: List[Tuple[str, str]] = [
    # Diffusion step encoder:
    # Upstream: diffusion_step_encoder.0 = SinusoidalPosEmb (no learnable params)
    # Upstream: diffusion_step_encoder.1 = Linear  -> ours: diffusion_step_encoder_linear1
    # Upstream: diffusion_step_encoder.2 = Mish (no params)
    # Upstream: diffusion_step_encoder.3 = Linear  -> ours: diffusion_step_encoder_linear2
    (r"^diffusion_step_encoder\.1\.", "diffusion_step_encoder_linear1."),
    (r"^diffusion_step_encoder\.3\.", "diffusion_step_encoder_linear2."),
    # Final conv:
    # Upstream: final_conv.0 = Conv1dBlock -> ours: final_block
    # Upstream: final_conv.1 = nn.Conv1d   -> ours: final_conv
    (r"^final_conv\.0\.", "final_block."),
    (r"^final_conv\.1\.", "final_conv._conv."),
]

# Rules for Conv1dBlock within a matched prefix.
# Upstream Conv1dBlock stores layers in nn.Sequential named 'block':
#   block.0 = Conv1d, block.1 = GroupNorm, block.2 = Mish
# Ours uses named attributes:
#   conv._conv, group_norm._norm, mish (no params)

_CONV1D_BLOCK_RULES: List[Tuple[str, str]] = [
    (r"\.block\.0\.", ".conv._conv."),
    (r"\.block\.1\.", ".group_norm._norm."),
]

# Rules for ConditionalResidualBlock1D.
# Upstream: cond_encoder = nn.Sequential(Mish, Linear, Rearrange)
#   cond_encoder.1.weight/bias -> cond_linear.weight/bias
# Upstream: residual_conv = nn.Conv1d or nn.Identity
#   residual_conv.weight/bias -> residual_conv._conv.weight/bias (if not Identity)

_COND_RESBLOCK_RULES: List[Tuple[str, str]] = [
    (r"\.cond_encoder\.1\.", ".cond_linear."),
]

# Rules for Downsample1d / Upsample1d: upstream conv is nn.Conv1d / nn.ConvTranspose1d,
# ours wraps in _Conv1d / _ConvTranspose1d with inner ._conv.
# Upstream: down_modules.I.2.conv.weight -> ours: down_modules.I.2.conv._conv._conv.weight
# Wait, let's check: upstream Downsample1d has self.conv = nn.Conv1d
# Our Downsample1d has self.conv = _Conv1d which has self._conv = nn.Conv1d
# So upstream path: *.conv.weight -> ours: *.conv._conv.weight

_RESAMPLE_RULES: List[Tuple[str, str]] = [
    # Downsample1d / Upsample1d: conv -> conv._conv
    # But we need to be careful not to match Conv1dBlock's .conv.
    # These only appear at down_modules.I.2 and up_modules.I.2
]

# Rules for the ResNet obs encoder.
# Upstream uses robomimic's obs encoder which wraps a ResNet backbone.
# Typical upstream key pattern:
#   obs_encoder.obs_nets.{key}.nets.0.nets.{resnet_key}
# or for PushT:
#   obs_encoder.nets.obs_nets.{key}.nets.0.nets.{resnet_key}
# Our structure:
#   obs_encoder.key_model_map.{key}.{resnet_key}
#   (or obs_encoder.key_model_map.rgb.{resnet_key} if share_rgb_model)

# ResNet internal mapping (torchvision -> ours):
#   downsample.0 -> downsample.conv
#   downsample.1 -> downsample.bn

_RESNET_RULES: List[Tuple[str, str]] = [
    (r"\.downsample\.0\.", ".downsample.conv."),
    (r"\.downsample\.1\.", ".downsample.bn."),
]


def _is_conv2d_weight(key: str, value: np.ndarray) -> bool:
    """Check if a parameter is a Conv2d weight (4D, not embedding)."""
    return "weight" in key and value.ndim == 4


def _is_conv1d_weight(key: str, value: np.ndarray) -> bool:
    """Check if a parameter is a Conv1d/ConvTranspose1d weight (3D, not embedding)."""
    return "weight" in key and value.ndim == 3 and "embedding" not in key


def transpose_conv2d(value: np.ndarray) -> np.ndarray:
    """Transpose Conv2d weight: OIHW (PyTorch) -> OHWI (MLX)."""
    return np.transpose(value, (0, 2, 3, 1))


def transpose_conv1d(value: np.ndarray) -> np.ndarray:
    """Transpose Conv1d weight: OIK (PyTorch) -> OKI (MLX)."""
    return np.transpose(value, (0, 2, 1))


def _apply_rules(key: str, rules: List[Tuple[str, str]]) -> str:
    """Apply regex substitution rules to a key."""
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def map_unet_key(key: str) -> str:
    """Map an upstream UNet key to our MLX UNet key path.

    Handles:
    - diffusion_step_encoder Sequential indexing -> named attributes
    - final_conv Sequential indexing -> final_block / final_conv
    - Conv1dBlock internal: block.{0,1} -> conv._conv / group_norm._norm
    - ConditionalResidualBlock1D: cond_encoder.1 -> cond_linear
    - Residual conv: residual_conv.{weight,bias} -> residual_conv._conv.{weight,bias}
    - Downsample1d/Upsample1d: conv -> conv._conv
    """
    result = key

    # Apply UNet-level rules
    result = _apply_rules(result, _UNET_KEY_RULES)

    # Apply Conv1dBlock rules (block.0/1 -> conv._conv / group_norm._norm)
    result = _apply_rules(result, _CONV1D_BLOCK_RULES)

    # Apply ConditionalResidualBlock1D rules (cond_encoder.1 -> cond_linear)
    result = _apply_rules(result, _COND_RESBLOCK_RULES)

    # Handle residual_conv: if it's a Conv1d (not Identity), add ._conv
    # In upstream: residual_conv.weight -> ours: residual_conv._conv.weight
    # Match residual_conv.weight or residual_conv.bias (but not if already has ._conv)
    result = re.sub(
        r"\.residual_conv\.(?!_conv)(weight|bias)",
        r".residual_conv._conv.\1",
        result,
    )

    # Handle Downsample1d/Upsample1d conv paths:
    # Upstream: down_modules.I.2.conv.weight -> ours: down_modules.I.2.conv._conv.weight
    # Upstream: up_modules.I.2.conv.weight -> ours: up_modules.I.2.conv._conv.weight
    # These are the 3rd element (index 2) in down/up modules which is Downsample/Upsample.
    # Pattern: (down_modules|up_modules).N.2.conv.(weight|bias)
    result = re.sub(
        r"(down_modules|up_modules)\.(\d+)\.2\.conv\.(weight|bias)",
        r"\1.\2.2.conv._conv.\3",
        result,
    )

    return result


def map_resnet_key(key: str) -> str:
    """Map torchvision/robomimic ResNet key to our MLX ResNet key."""
    result = key
    result = _apply_rules(result, _RESNET_RULES)

    # Skip num_batches_tracked
    if "num_batches_tracked" in result:
        return ""

    return result


def map_obs_encoder_key(key: str) -> Optional[str]:
    """Map an upstream obs_encoder key to our MLX obs_encoder key.

    Upstream robomimic obs encoder structures vary. Common patterns:
    - obs_encoder.nets.obs_nets.{cam_key}.nets.0.nets.{resnet_key}
    - obs_encoder.obs_nets.{cam_key}.nets.0.nets.{resnet_key}

    Our structure:
    - obs_encoder.key_model_map.{cam_key}.{resnet_key}

    For low-dim observations, there may not be a model.
    """
    # Try different upstream patterns
    # Pattern 1: obs_encoder.nets.obs_nets.{key}.nets.0.nets.{rest}
    m = re.match(
        r"^obs_encoder\.(?:nets\.)?obs_nets\.([^.]+)\.nets\.0\.nets\.(.*)",
        key,
    )
    if m:
        cam_key = m.group(1)
        resnet_key = m.group(2)
        resnet_key = map_resnet_key(resnet_key)
        if not resnet_key:
            return None
        return f"obs_encoder.key_model_map.{cam_key}.{resnet_key}"

    # Pattern 2: obs_encoder.obs_randomizers.* (crop randomizers - skip)
    if "obs_randomizers" in key:
        return None

    # Pattern 3: obs_encoder.{rest} - pass through with resnet mapping
    if key.startswith("obs_encoder."):
        rest = key[len("obs_encoder.") :]
        rest = map_resnet_key(rest)
        if not rest:
            return None
        return f"obs_encoder.{rest}"

    return key


def map_key_path(pytorch_key: str) -> Optional[str]:
    """Map a full PyTorch state dict key to our MLX parameter path.

    The upstream state dict for DiffusionUnetHybridImagePolicy has these prefixes:
    - model.* -> UNet parameters
    - obs_encoder.* -> Vision encoder parameters
    - normalizer.* -> Normalizer parameters (handled separately)

    Returns None for keys that should be skipped.
    """
    # Skip normalizer keys (handled separately)
    if "normalizer" in pytorch_key:
        return None

    # Skip num_batches_tracked
    if "num_batches_tracked" in pytorch_key:
        return None

    # UNet model keys
    if pytorch_key.startswith("model."):
        unet_key = pytorch_key[len("model.") :]
        mapped = map_unet_key(unet_key)
        return f"model.{mapped}"

    # Obs encoder keys
    if pytorch_key.startswith("obs_encoder."):
        return map_obs_encoder_key(pytorch_key)

    # Unknown prefix - keep as-is
    return pytorch_key


def convert_state_dict(
    state_dict: dict,
    skip_normalizer: bool = True,
) -> Dict[str, mx.array]:
    """Convert a PyTorch state dict to MLX parameter dict.

    Key transformations:
      1. Conv2d weights: (C_out, C_in, H, W) -> (C_out, H, W, C_in)
      2. Conv1d weights: (C_out, C_in, K) -> (C_out, K, C_in)
      3. Key path mapping: upstream module hierarchy -> our hierarchy
      4. All tensors: torch.Tensor -> numpy -> mx.array
    """
    mlx_params: Dict[str, mx.array] = {}

    for key, value in state_dict.items():
        # Skip non-tensor entries
        if HAS_TORCH and isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_value = value
        else:
            continue

        # Skip normalizer keys if requested
        if skip_normalizer and "normalizer" in key:
            continue

        # Skip num_batches_tracked
        if "num_batches_tracked" in key:
            continue

        # Map key path
        mlx_key = map_key_path(key)
        if mlx_key is None:
            continue

        # Conv2d weight transpose: OIHW -> OHWI
        if _is_conv2d_weight(key, np_value):
            np_value = transpose_conv2d(np_value)

        # Conv1d weight transpose: OIK -> OKI
        elif _is_conv1d_weight(key, np_value):
            np_value = transpose_conv1d(np_value)

        mlx_params[mlx_key] = mx.array(np_value)

    return mlx_params


def extract_normalizer(state_dict: dict) -> Dict[str, mx.array]:
    """Extract normalizer parameters from state dict.

    Upstream stores normalizer params inline in the policy state dict:
    - normalizer.params_dict.action.input_stats.min
    - normalizer.params_dict.action.scale
    - normalizer.params_dict.action.offset
    - normalizer.params_dict.obs.agent_pos.scale
    - etc.
    """
    norm_params: Dict[str, mx.array] = {}

    for key, value in state_dict.items():
        if "normalizer" not in key:
            continue

        if HAS_TORCH and isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_value = value
        else:
            continue

        norm_params[key] = mx.array(np_value)

    return norm_params


def shape_based_match(
    source_params: Dict[str, np.ndarray],
    target_params: Dict[str, Tuple[str, tuple]],
) -> Dict[str, str]:
    """Match source parameters to target parameters by shape.

    When name-based mapping fails, this matches parameters by their shapes.
    For parameters with the same shape, it uses order of appearance.

    Args:
        source_params: Dict of {source_key: numpy_array}
        target_params: Dict of {target_key: (target_key, shape_tuple)}

    Returns:
        Dict mapping source_key -> target_key
    """
    # Group source params by shape
    source_by_shape: Dict[tuple, List[str]] = {}
    for key, arr in source_params.items():
        shape = tuple(arr.shape)
        if shape not in source_by_shape:
            source_by_shape[shape] = []
        source_by_shape[shape].append(key)

    # Group target params by shape
    target_by_shape: Dict[tuple, List[str]] = {}
    for key, (_, shape) in target_params.items():
        if shape not in target_by_shape:
            target_by_shape[shape] = []
        target_by_shape[shape].append(key)

    # Match by shape and order
    mapping: Dict[str, str] = {}
    for shape, source_keys in source_by_shape.items():
        if shape in target_by_shape:
            target_keys = target_by_shape[shape]
            for i, src_key in enumerate(source_keys):
                if i < len(target_keys):
                    mapping[src_key] = target_keys[i]

    return mapping


def convert_with_shape_matching(
    source_state_dict: dict,
    target_param_shapes: Dict[str, tuple],
) -> Dict[str, mx.array]:
    """Convert state dict using shape-based matching as fallback.

    First tries name-based mapping via map_key_path. For any unmapped
    parameters, falls back to shape-based matching.

    Args:
        source_state_dict: PyTorch state dict
        target_param_shapes: Dict of {mlx_param_path: shape_tuple}

    Returns:
        Dict of {mlx_param_path: mx.array}
    """
    # First pass: name-based mapping
    mapped = convert_state_dict(source_state_dict, skip_normalizer=True)

    # Find unmapped source params
    mapped_targets = set(mapped.keys())
    needed_targets = set(target_param_shapes.keys()) - mapped_targets

    if not needed_targets:
        return mapped

    # Collect unmapped source params
    unmapped_source: Dict[str, np.ndarray] = {}
    for key, value in source_state_dict.items():
        if "normalizer" in key or "num_batches_tracked" in key:
            continue
        if HAS_TORCH and isinstance(value, torch.Tensor):
            np_value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_value = value
        else:
            continue

        mlx_key = map_key_path(key)
        if mlx_key is not None and mlx_key in mapped_targets:
            continue

        # Apply shape transformations
        if _is_conv2d_weight(key, np_value):
            np_value = transpose_conv2d(np_value)
        elif _is_conv1d_weight(key, np_value):
            np_value = transpose_conv1d(np_value)

        unmapped_source[key] = np_value

    # Build needed target info
    needed_target_info = {k: (k, target_param_shapes[k]) for k in needed_targets}

    # Shape-based matching
    shape_mapping = shape_based_match(unmapped_source, needed_target_info)

    for src_key, tgt_key in shape_mapping.items():
        mapped[tgt_key] = mx.array(unmapped_source[src_key])

    still_unmapped = needed_targets - set(shape_mapping.values())
    if still_unmapped:
        logger.warning(
            "Could not map %d target parameters: %s",
            len(still_unmapped),
            list(still_unmapped)[:10],
        )

    return mapped


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    format: str = "npz",
) -> None:
    """Convert a full PyTorch checkpoint to MLX format.

    Args:
        checkpoint_path: Path to .ckpt file
        output_dir: Directory to save converted weights
        format: Output format ('npz' or 'safetensors')
    """
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for weight conversion. Install with: pip install torch"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load PyTorch checkpoint
    logger.info("Loading checkpoint from %s", checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning(
            "weights_only=True failed (checkpoint may contain non-tensor objects). "
            "Falling back to weights_only=False -- only use this with TRUSTED checkpoints."
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dicts
    if "state_dicts" in ckpt:
        model_state = ckpt["state_dicts"]["model"]
        ema_state = ckpt["state_dicts"].get("ema_model", None)
    elif "model" in ckpt:
        model_state = ckpt["model"]
        ema_state = ckpt.get("ema", None)
    else:
        model_state = ckpt
        ema_state = None

    # Convert model weights
    logger.info("Converting model weights (%d parameters)", len(model_state))
    mlx_weights = convert_state_dict(model_state)
    _save_weights(mlx_weights, output / f"model.{format}", format)
    logger.info(
        "Saved model weights: %d parameters, shapes: %s",
        len(mlx_weights),
        {k: tuple(v.shape) for k, v in list(mlx_weights.items())[:5]},
    )

    # Convert EMA weights
    if ema_state is not None:
        logger.info("Converting EMA weights (%d parameters)", len(ema_state))
        mlx_ema = convert_state_dict(ema_state)
        _save_weights(mlx_ema, output / f"ema.{format}", format)
        logger.info("Saved EMA weights: %d parameters", len(mlx_ema))

    # Extract and save normalizer
    normalizer_state = extract_normalizer(model_state)
    if normalizer_state:
        logger.info("Saving normalizer state (%d entries)", len(normalizer_state))
        _save_weights(normalizer_state, output / f"normalizer.{format}", format)

    # Save conversion metadata
    meta_path = output / "conversion_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"source: {checkpoint_path}\n")
        f.write(f"format: {format}\n")
        f.write(f"model_params: {len(mlx_weights)}\n")
        if ema_state:
            f.write(f"ema_params: {len(mlx_ema)}\n")
        f.write(f"normalizer_entries: {len(normalizer_state)}\n")
        f.write("\nModel parameter keys:\n")
        for k in sorted(mlx_weights.keys()):
            f.write(f"  {k}: {tuple(mlx_weights[k].shape)}\n")

    print(f"Converted checkpoint saved to {output}")
    print(f"  Model parameters: {len(mlx_weights)}")
    if ema_state:
        print(f"  EMA parameters: {len(mlx_ema)}")
    print(f"  Normalizer entries: {len(normalizer_state)}")


def _save_weights(weights: Dict[str, mx.array], path: Path, format: str) -> None:
    """Save weights in the specified format."""
    if format == "npz":
        mx.savez(str(path), **weights)
    elif format == "safetensors":
        try:
            from safetensors.numpy import save_file

            np_weights = {k: np.array(v) for k, v in weights.items()}
            save_file(np_weights, str(path))
        except ImportError:
            logger.warning("safetensors not installed, falling back to npz format")
            npz_path = path.with_suffix(".npz")
            mx.savez(str(npz_path), **weights)
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Diffusion Policy checkpoint to MLX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch .ckpt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="npz",
        choices=["npz", "safetensors"],
        help="Output format (default: npz)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    convert_checkpoint(args.checkpoint, args.output, args.format)


if __name__ == "__main__":
    main()
