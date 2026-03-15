# PRD-02: Vision Encoder (ResNet + MultiImageObsEncoder)

**Status:** Not Started
**Depends on:** PRD-01 (compat layer)
**Blocks:** PRD-05 (Policy Assembly)

---

## Objective

Port the visual observation encoder: ResNet18/34/50 backbone in pure MLX, `MultiImageObsEncoder` for multi-camera inputs, and `CropRandomizer` for augmentation.

---

## Upstream Reference Files

| File | Key Classes |
|------|------------|
| `model/vision/model_getter.py` | `get_resnet(name, weights)` |
| `model/vision/multi_image_obs_encoder.py` | `MultiImageObsEncoder` |
| `model/vision/crop_randomizer.py` | `CropRandomizer` |
| `common/pytorch_util.py` | `replace_submodules()` |

---

## Deliverables

### 1. `compat/vision.py` — ResNet in MLX

Full ResNet18/34/50 implementation from scratch:

```python
class BasicBlock(mlx.nn.Module):
    """ResNet basic block (ResNet18/34)."""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        # conv1: 3x3, stride
        # bn1: BatchNorm → GroupNorm (see below)
        # conv2: 3x3, stride=1
        # bn2
        # downsample: optional 1x1 conv for residual

class Bottleneck(mlx.nn.Module):
    """ResNet bottleneck block (ResNet50)."""
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        # conv1: 1x1
        # conv2: 3x3, stride
        # conv3: 1x1, expansion
        # bn1, bn2, bn3

class ResNet(mlx.nn.Module):
    """Full ResNet backbone."""
    def __init__(self, block, layers, num_classes=1000):
        # conv1: 7x7, stride=2
        # bn1 → maxpool
        # layer1-4: _make_layer(block, planes, blocks, stride)
        # avgpool → fc

    def __call__(self, x):
        # x: (B, C, H, W) in NCHW convention
        # Internally convert to NHWC for MLX Conv2d
        # Return: (B, feature_dim) after avgpool+flatten

    def _make_layer(self, block, planes, blocks, stride=1):
        ...

def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
```

**Feature dimensions (after avgpool, before fc):**
- ResNet18: 512
- ResNet34: 512
- ResNet50: 2048

**NCHW↔NHWC Strategy:**
- Public interface: NCHW (matches upstream callers)
- Internal conv ops: NHWC (what MLX Conv2d expects)
- Transpose at entry and exit of forward()
- All intermediate shapes are NHWC

**Weight Loading from PyTorch:**

```python
def load_torchvision_weights(mlx_resnet, torch_state_dict):
    """Convert torchvision ResNet weights to MLX format.

    Weight format changes:
    - Conv2d: PyTorch (C_out, C_in, H, W) → MLX (C_out, H, W, C_in)
    - Linear: same (no change needed)
    - BatchNorm/GroupNorm: same shape
    """
    for key, value in torch_state_dict.items():
        if 'conv' in key and 'weight' in key:
            # OIHW → OHWI
            value = value.permute(0, 2, 3, 1)
        ...
```

**BatchNorm handling:**
- Upstream uses `nn.BatchNorm2d`
- The policy often replaces it with GroupNorm via `replace_submodules()`
- We implement both but default to GroupNorm (simpler in MLX, no running stats needed for inference)

### 2. `model/vision/model_getter.py`

```python
def get_resnet(name: str, weights=None, **kwargs):
    """Get ResNet backbone with optional pretrained weights.

    Args:
        name: 'resnet18', 'resnet34', 'resnet50'
        weights: 'IMAGENET1K_V1' or None
    Returns:
        ResNet with fc replaced by identity (feature extractor)
    """
    model_fn = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}
    model = model_fn[name](**kwargs)

    if weights == 'IMAGENET1K_V1':
        # Load converted torchvision weights from safetensors
        load_pretrained_weights(model, name)

    # Replace fc with identity — we want features, not classification
    model.fc = lambda x: x  # or Identity module
    return model
```

### 3. `model/vision/multi_image_obs_encoder.py`

```python
class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
        shape_meta: dict,
        rgb_model,              # ResNet instance or dict of instances
        resize_shape=None,      # (H, W) or None
        crop_shape=None,        # (H, W) or None
        random_crop=True,
        use_group_norm=False,
        share_rgb_model=False,
        imagenet_norm=False):
        """
        shape_meta example:
        {
            'obs': {
                'image': {'shape': (3, 96, 96), 'type': 'rgb'},
                'agent_pos': {'shape': (2,), 'type': 'low_dim'},
            }
        }
        """
        ...

    def __call__(self, obs_dict):
        """
        Input: obs_dict with keys matching shape_meta
          - RGB: (B, C, H, W) per camera
          - low_dim: (B, D) per feature
        Output: (B, total_feature_dim) concatenated features
        """
        features = []
        for key in self.rgb_keys:
            img = obs_dict[key]
            # Resize → Crop → Normalize → Vision model
            feat = self.process_rgb(key, img)
            features.append(feat)

        for key in self.lowdim_keys:
            features.append(obs_dict[key])

        return mx.concatenate(features, axis=-1)

    def output_shape(self) -> int:
        """Returns total concatenated feature dimension."""
        ...
```

**Multi-camera handling:**
- `share_rgb_model=True`: Stack all camera images → single forward pass → reshape
- `share_rgb_model=False`: Separate model per camera (deep copy)
- Default PushT: single camera, so this is simple case

### 4. `model/vision/crop_randomizer.py`

```python
class CropRandomizer(mlx.nn.Module):
    def __init__(self, input_shape, crop_height, crop_width,
                 num_crops=1, pos_enc=False):
        """
        input_shape: (C, H, W)
        """
        ...

    def __call__(self, x):
        """
        x: (B, C, H, W)
        Returns: (B*num_crops, C, crop_h, crop_w)
        """
        if self.training:
            return self._random_crop(x)
        else:
            return self._center_crop(x)

    def _random_crop(self, x):
        """Random spatial crops."""
        B, C, H, W = x.shape
        # Sample random top-left corners
        h_start = mx.random.randint(0, H - self.crop_height, (B, self.num_crops))
        w_start = mx.random.randint(0, W - self.crop_width, (B, self.num_crops))
        # Gather crops (need to handle indexing carefully in MLX)
        ...

    def _center_crop(self, x):
        """Center crop for evaluation."""
        h_off = (x.shape[2] - self.crop_height) // 2
        w_off = (x.shape[3] - self.crop_width) // 2
        return x[:, :, h_off:h_off+self.crop_height, w_off:w_off+self.crop_width]
```

### 5. Utility: `replace_submodules`

```python
def replace_submodules(root_module, predicate, func):
    """Replace all submodules matching predicate with func(module).

    Used to swap BatchNorm2d → GroupNorm globally in ResNet.
    """
    # Walk module tree, replace matching modules
    ...
```

---

## ImageNet Normalization

```python
IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = mx.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

def imagenet_normalize(x):
    """x: (B, 3, H, W) in [0, 1] → normalized."""
    return (x - IMAGENET_MEAN) / IMAGENET_STD
```

---

## Tests

### `tests/test_vision_encoder.py`

```python
def test_resnet18_output_shape():
    model = resnet18()
    model.fc = lambda x: x  # feature extractor
    x = mx.random.normal((2, 3, 224, 224))
    out = model(x)
    assert out.shape == (2, 512)

def test_resnet50_output_shape():
    model = resnet50()
    model.fc = lambda x: x
    x = mx.random.normal((2, 3, 224, 224))
    out = model(x)
    assert out.shape == (2, 2048)

def test_resnet18_small_input():
    """PushT uses 76x76 crops."""
    model = resnet18()
    model.fc = lambda x: x
    x = mx.random.normal((2, 3, 76, 76))
    out = model(x)
    assert out.shape[0] == 2
    assert out.ndim == 2  # (B, feat_dim)

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_resnet18_matches_torchvision():
    """With same weights, output should match within tolerance."""
    import torchvision
    torch_model = torchvision.models.resnet18(weights=None)
    torch_model.fc = torch.nn.Identity()
    torch_model.eval()

    mlx_model = resnet18()
    mlx_model.fc = lambda x: x
    # Copy weights from torch → mlx
    copy_resnet_weights(torch_model, mlx_model)

    x_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    torch_out = torch_model(torch.tensor(x_np)).detach().numpy()
    mlx_out = np.array(mlx_model(mx.array(x_np)))

    np.testing.assert_allclose(mlx_out, torch_out, atol=1e-4, rtol=1e-4)

def test_multi_image_obs_encoder():
    shape_meta = {
        'obs': {
            'image': {'shape': (3, 96, 96), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        }
    }
    rgb_model = resnet18()
    rgb_model.fc = lambda x: x
    enc = MultiImageObsEncoder(shape_meta, rgb_model, crop_shape=(76, 76))
    obs = {
        'image': mx.random.normal((4, 3, 96, 96)),
        'agent_pos': mx.random.normal((4, 2)),
    }
    out = enc(obs)
    assert out.ndim == 2
    assert out.shape[0] == 4

def test_crop_randomizer_train():
    cr = CropRandomizer((3, 96, 96), 76, 76)
    cr.train()
    x = mx.random.normal((4, 3, 96, 96))
    out = cr(x)
    assert out.shape == (4, 3, 76, 76)

def test_crop_randomizer_eval():
    cr = CropRandomizer((3, 96, 96), 76, 76)
    cr.eval()
    x = mx.random.normal((4, 3, 96, 96))
    out = cr(x)
    assert out.shape == (4, 3, 76, 76)
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | ResNet18 forward: (B,3,224,224) → (B,512) | shape exact |
| 2 | ResNet18 forward: (B,3,76,76) → (B, feat) | shape valid |
| 3 | ResNet50 forward: (B,3,224,224) → (B,2048) | shape exact |
| 4 | ResNet18 numerics match torchvision (same weights) | atol=1e-4 |
| 5 | MultiImageObsEncoder produces (B, D) feature vector | shape correct |
| 6 | CropRandomizer train: random crops, correct output shape | shape exact |
| 7 | CropRandomizer eval: center crop, deterministic | shape exact, deterministic |
| 8 | Weight loading from torchvision state_dict works | no errors |

---

## Upstream Sync Notes

**Files to watch:**
- `model/vision/model_getter.py` — if new backbone options are added
- `model/vision/multi_image_obs_encoder.py` — if new observation types are added
- `model/vision/crop_randomizer.py` — if augmentation changes

**Weight format note:** When upstream updates torchvision version, weight key names may change. Our weight converter maps torchvision state_dict keys to our ResNet parameter paths.
