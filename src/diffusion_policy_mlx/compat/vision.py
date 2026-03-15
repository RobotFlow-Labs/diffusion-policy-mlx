"""
ResNet implementation in pure MLX.

Provides ResNet18/34/50 with NCHW public interface and NHWC internal computation.
Weight loading from torchvision state_dict format is supported.
"""

from typing import Callable, List, Optional, Type, Union

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clone_module(module: "nn.Module") -> "nn.Module":
    """Deep-clone an MLX nn.Module (copy.deepcopy fails on nanobind objects).

    Creates a fresh instance of the same class and loads the weights from the
    original module.
    """
    import json
    import io

    # Save weights to a flat list
    weights = module.parameters()

    # Reconstruct from class — we need the constructor args. For our ResNet
    # this is handled by re-creating via factory functions. For a generic
    # approach, we save/load the weights into a new instance created by the
    # same factory.

    # Use MLX's built-in save/load mechanism
    flat_weights = dict(_flatten_dict(weights))

    # Create new instance by calling the class constructor
    # This requires that __init__ has been called and the module tree structure
    # can be inferred from the weights alone. For MLX this works via load_weights.
    import tempfile, os
    from pathlib import Path

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_path = f.name

    try:
        module.save_weights(tmp_path)

        # Create a fresh instance of the same architecture
        # We need to reconstruct via the same constructor.
        # For ResNet, we stored _block_type and can reconstruct.
        new_module = _reconstruct_module(module)
        new_module.load_weights(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return new_module


def _flatten_dict(d, prefix=""):
    """Flatten a nested dict with dot-separated keys."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_dict(v, new_key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            items.extend(_flatten_dict(v, new_key))
    else:
        items.append((prefix, d))
    return items


def _reconstruct_module(module: "nn.Module") -> "nn.Module":
    """Reconstruct a module of the same type/architecture.

    Supports ResNet and common nn layers. For ResNet, uses stored metadata.
    """
    if isinstance(module, ResNet):
        block_type = module._block_type
        # Infer layers from the stored layer lists
        layers_config = [
            len(module.layer1),
            len(module.layer2),
            len(module.layer3),
            len(module.layer4),
        ]
        # Infer num_classes from fc
        if isinstance(module.fc, Identity):
            num_classes = 512 * block_type.expansion  # dummy, will be replaced
        elif isinstance(module.fc, nn.Linear):
            num_classes = module.fc.weight.shape[0]
        else:
            num_classes = 1000
        new = ResNet(block_type, layers_config, num_classes=num_classes)
        # If fc was Identity, set it on the clone too
        if isinstance(module.fc, Identity):
            new.fc = Identity()
        return new
    else:
        raise TypeError(
            f"clone_module: don't know how to reconstruct {type(module).__name__}. "
            f"Please add support or avoid deepcopy."
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class Identity(nn.Module):
    """Identity module (passthrough)."""
    def __call__(self, x):
        return x


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet basic block (used in ResNet18/34)."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        # All convs use NHWC internally (MLX default)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample

    def __call__(self, x):
        # x: (B, H, W, C) in NHWC
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = nn.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck block (used in ResNet50)."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm(planes * self.expansion)
        self.downsample = downsample

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = nn.relu(out)
        return out


# ---------------------------------------------------------------------------
# Downsample helper (conv + bn as a sequential-like callable)
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    """1x1 conv + batchnorm for residual projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """Full ResNet backbone.

    Public interface uses NCHW (matching upstream / PyTorch convention).
    Internal computation uses NHWC (MLX native).
    Transposes happen at entry and exit of ``__call__``.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self._block_type = block
        self.in_planes = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head (often replaced with Identity for feature extraction)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # -- layer builder -------------------------------------------------------

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = _Downsample(self.in_planes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return layers  # list of modules — iterated in __call__

    # -- forward -------------------------------------------------------------

    def __call__(self, x):
        """
        Args:
            x: (B, C, H, W) — NCHW convention (public API).

        Returns:
            (B, feature_dim) after adaptive avg pool + flatten, **or**
            (B, num_classes) if fc is the original Linear layer.
        """
        # NCHW → NHWC
        x = mx.transpose(x, (0, 2, 3, 1))

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.maxpool(x)

        # Residual stages
        for blk in self.layer1:
            x = blk(x)
        for blk in self.layer2:
            x = blk(x)
        for blk in self.layer3:
            x = blk(x)
        for blk in self.layer4:
            x = blk(x)

        # Adaptive average pool: (B, H', W', C) → (B, C)
        x = mx.mean(x, axis=(1, 2))

        # fc head (Identity when used as feature extractor)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# ---------------------------------------------------------------------------
# Weight loading from torchvision state_dict
# ---------------------------------------------------------------------------

def _torch_bn_key_to_mlx(key: str) -> str:
    """Map torchvision BatchNorm2d param names to mlx.nn.BatchNorm names.

    torchvision: weight, bias, running_mean, running_var, num_batches_tracked
    mlx BatchNorm: weight, bias, running_mean, running_var
    """
    if "num_batches_tracked" in key:
        return ""  # skip
    return key


def _map_state_dict_key(torch_key: str) -> str:
    """Convert a torchvision ResNet state_dict key to our MLX param path.

    Examples:
        'conv1.weight' → 'conv1.weight'
        'layer1.0.conv1.weight' → 'layer1.0.conv1.weight'
        'layer1.0.downsample.0.weight' → 'layer1.0.downsample.conv.weight'
        'layer1.0.downsample.1.weight' → 'layer1.0.downsample.bn.weight'
    """
    parts = torch_key.split(".")

    # Map downsample.0 → downsample.conv, downsample.1 → downsample.bn
    new_parts = []
    i = 0
    while i < len(parts):
        if parts[i] == "downsample" and i + 1 < len(parts):
            idx = parts[i + 1]
            if idx == "0":
                new_parts.extend(["downsample", "conv"])
            elif idx == "1":
                new_parts.extend(["downsample", "bn"])
            else:
                new_parts.extend([parts[i], parts[i + 1]])
            i += 2
        else:
            new_parts.append(parts[i])
            i += 1

    return ".".join(new_parts)


def load_torchvision_weights(mlx_resnet: ResNet, torch_state_dict: dict):
    """Load weights from a torchvision ResNet state_dict into an MLX ResNet.

    Handles:
    - Conv2d weight transposition: OIHW → OHWI (PyTorch → MLX)
    - BatchNorm param mapping
    - Downsample indexing (Sequential indices → named attrs)

    Args:
        mlx_resnet: Target MLX ResNet model.
        torch_state_dict: state_dict from torchvision.models.resnetN().
    """
    import numpy as np

    weights = {}
    for torch_key, value in torch_state_dict.items():
        # Skip batch tracking counter
        if "num_batches_tracked" in torch_key:
            continue

        mlx_key = _map_state_dict_key(torch_key)

        # Convert to numpy
        if hasattr(value, "detach"):  # torch tensor
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)

        # Transpose conv weights: OIHW → OHWI
        if "conv" in mlx_key and mlx_key.endswith(".weight") and arr.ndim == 4:
            arr = np.transpose(arr, (0, 2, 3, 1))

        weights[mlx_key] = mx.array(arr)

    # Apply to model using mlx's load_weights with a flat dict
    # We need to convert flat dot-separated keys to nested structure
    mlx_resnet.load_weights(list(weights.items()))


# ---------------------------------------------------------------------------
# replace_submodules utility
# ---------------------------------------------------------------------------

def replace_submodules(
    root_module: nn.Module,
    predicate: Callable,
    func: Callable,
) -> nn.Module:
    """Replace all submodules matching predicate with func(module).

    Used to swap BatchNorm -> GroupNorm globally in a ResNet.

    MLX modules store children as dict items (nn.Module subclasses dict),
    so we iterate over dict keys rather than vars().

    Args:
        root_module: The root module to modify in-place.
        predicate: Returns True if a submodule should be replaced.
        func: Given a matching submodule, returns its replacement.

    Returns:
        The root_module (possibly replaced if it matches predicate itself).
    """
    if predicate(root_module):
        return func(root_module)

    _replace_recursive(root_module, predicate, func)
    return root_module


def _replace_recursive(parent: nn.Module, predicate: Callable, func: Callable):
    """Recursively replace submodules in-place.

    MLX nn.Module is a dict subclass, so children are stored as dict items.
    """
    # Iterate over dict keys (module children in MLX)
    for key in list(parent.keys()):
        child = parent[key]
        if isinstance(child, nn.Module):
            if predicate(child):
                parent[key] = func(child)
            else:
                _replace_recursive(child, predicate, func)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    if predicate(item):
                        child[i] = func(item)
                    else:
                        _replace_recursive(item, predicate, func)
