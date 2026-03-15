"""
Model getter for vision backbones.

Provides ``get_resnet(name, weights)`` that returns a ResNet feature extractor
(fc replaced with identity) in pure MLX.
"""

from diffusion_policy_mlx.compat.vision import (
    Identity,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
)

_MODEL_FNS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def get_resnet(name: str, weights=None, **kwargs) -> ResNet:
    """Get a ResNet backbone configured as a feature extractor.

    Args:
        name: One of ``'resnet18'``, ``'resnet34'``, ``'resnet50'``.
        weights: ``'IMAGENET1K_V1'`` to load pretrained weights, or ``None``.
        **kwargs: Extra keyword arguments forwarded to the ResNet constructor.

    Returns:
        A :class:`ResNet` with ``fc`` replaced by an :class:`Identity` module
        so that the output is the feature vector (e.g. 512-d for ResNet18).
    """
    if name not in _MODEL_FNS:
        raise ValueError(f"Unknown model name '{name}'. Choose from {list(_MODEL_FNS.keys())}")

    model = _MODEL_FNS[name](**kwargs)

    if weights == "IMAGENET1K_V1":
        _load_pretrained(model, name)

    # Replace classification head with identity → feature extractor
    model.fc = Identity()
    return model


def _load_pretrained(model: ResNet, name: str):
    """Load pretrained torchvision weights (requires torch + torchvision)."""
    try:
        import torch
        import torchvision
    except ImportError:
        raise ImportError(
            "Loading pretrained weights requires torch and torchvision. "
            "Install them or pass weights=None."
        )

    from diffusion_policy_mlx.compat.vision import load_torchvision_weights

    torch_model_fn = getattr(torchvision.models, name)
    torch_model = torch_model_fn(weights="IMAGENET1K_V1")
    state_dict = torch_model.state_dict()
    load_torchvision_weights(model, state_dict)
