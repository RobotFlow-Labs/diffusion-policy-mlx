# PRD-05: Policy Assembly

**Status:** Not Started
**Depends on:** PRD-01 (compat), PRD-02 (vision), PRD-03 (UNet), PRD-04 (scheduler)
**Blocks:** PRD-06 (Training Loop)

---

## Objective

Assemble the full `DiffusionUnetHybridImagePolicy` — the main policy class that ties the vision encoder, UNet denoiser, and noise scheduler into a complete diffusion policy with `predict_action()` and `compute_loss()`.

---

## Upstream Reference

| File | Classes |
|------|---------|
| `policy/diffusion_unet_hybrid_image_policy.py` | `DiffusionUnetHybridImagePolicy` |
| `policy/base_image_policy.py` | `BaseImagePolicy` |
| `model/common/normalizer.py` | `LinearNormalizer`, `SingleFieldLinearNormalizer` |
| `model/diffusion/mask_generator.py` | `LowdimMaskGenerator` |
| `model/common/module_attr_mixin.py` | `ModuleAttrMixin` |

---

## Deliverables

### 1. `model/common/normalizer.py` — LinearNormalizer

```python
class SingleFieldLinearNormalizer:
    """Normalizes a single tensor field using learned scale/offset.

    Modes:
      'limits': scale to [output_min, output_max] based on data min/max
      'gaussian': standardize to mean=0, std=1

    Usage:
      norm = SingleFieldLinearNormalizer.create_fit(data, mode='limits')
      x_norm = norm.normalize(x)
      x_orig = norm.unnormalize(x_norm)
    """

    def __init__(self, scale, offset, input_stats=None):
        self.scale = scale        # mx.array
        self.offset = offset      # mx.array
        self.input_stats = input_stats

    @classmethod
    def create_fit(cls, data, last_n_dims=1, mode='limits',
                   output_max=1.0, output_min=-1.0, range_eps=1e-4,
                   fit_offset=True):
        """Fit normalizer from data.

        data: mx.array of shape (..., D) where last_n_dims are normalized
        """
        # Flatten all but last_n_dims
        data_flat = data.reshape(-1, *data.shape[-last_n_dims:])
        if mode == 'limits':
            input_min = mx.min(data_flat, axis=0)
            input_max = mx.max(data_flat, axis=0)
            input_range = mx.clip(input_max - input_min, a_min=range_eps, a_max=None)
            scale = (output_max - output_min) / input_range
            if fit_offset:
                offset = output_min - input_min * scale
            else:
                offset = mx.zeros_like(scale)
        elif mode == 'gaussian':
            mean = mx.mean(data_flat, axis=0)
            std = mx.clip(mx.std(data_flat, axis=0), a_min=range_eps, a_max=None)
            scale = 1.0 / std
            offset = -mean / std if fit_offset else mx.zeros_like(scale)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return cls(scale=scale, offset=offset)

    @classmethod
    def create_identity(cls, shape):
        """Identity normalizer (no-op)."""
        return cls(scale=mx.ones(shape), offset=mx.zeros(shape))

    @classmethod
    def create_manual(cls, scale, offset):
        return cls(scale=mx.array(scale), offset=mx.array(offset))

    def normalize(self, x):
        return x * self.scale + self.offset

    def unnormalize(self, x):
        return (x - self.offset) / self.scale

    def state_dict(self):
        return {'scale': self.scale, 'offset': self.offset}

    def load_state_dict(self, d):
        self.scale = d['scale']
        self.offset = d['offset']


class LinearNormalizer:
    """Dict-based normalizer: one SingleFieldLinearNormalizer per key.

    Usage:
      normalizer = LinearNormalizer()
      normalizer.fit({'action': action_data, 'obs': {'image': ..., 'agent_pos': ...}})
      batch_norm = normalizer.normalize(batch)
      batch_orig = normalizer.unnormalize(batch_norm)
    """

    def __init__(self):
        self.normalizers = {}

    def fit(self, data_dict, **kwargs):
        """Fit normalizer for each key in data_dict."""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                self.normalizers[key] = {}
                for subkey, subvalue in value.items():
                    self.normalizers[key][subkey] = SingleFieldLinearNormalizer.create_fit(
                        subvalue, **kwargs)
            else:
                self.normalizers[key] = SingleFieldLinearNormalizer.create_fit(
                    value, **kwargs)

    def __getitem__(self, key):
        return self.normalizers[key]

    def __setitem__(self, key, value):
        self.normalizers[key] = value

    def __contains__(self, key):
        return key in self.normalizers

    def normalize(self, x):
        """Normalize a dict or single tensor."""
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                if key in self.normalizers:
                    if isinstance(value, dict) and isinstance(self.normalizers[key], dict):
                        result[key] = {k: self.normalizers[key][k].normalize(v)
                                       for k, v in value.items()
                                       if k in self.normalizers[key]}
                    elif isinstance(self.normalizers[key], SingleFieldLinearNormalizer):
                        result[key] = self.normalizers[key].normalize(value)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return result
        return x

    def unnormalize(self, x):
        """Inverse of normalize."""
        # Mirror of normalize with unnormalize calls
        ...
```

### 2. `model/diffusion/mask_generator.py` — LowdimMaskGenerator

```python
class LowdimMaskGenerator:
    """Generate observation/action masks for diffusion inpainting.

    During training, masks indicate which dimensions contain observed data
    (obs) vs data to be generated (actions).

    Args:
        action_dim: number of action dimensions
        obs_dim: number of observation dimensions
        max_n_obs_steps: how many timesteps of obs are visible
        fix_obs_steps: if True, always use max_n_obs_steps; else random
        action_visible: if True, past actions are also visible
    """
    def __init__(self, action_dim, obs_dim, max_n_obs_steps=2,
                 fix_obs_steps=True, action_visible=False):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    def __call__(self, shape):
        """
        shape: (B, T, D) where D = action_dim + obs_dim
        returns: (B, T, D) boolean mask (True = observed/visible)
        """
        B, T, D = shape
        assert D == self.action_dim + self.obs_dim

        mask = mx.zeros(shape, dtype=mx.bool_)

        if self.fix_obs_steps:
            n_obs = self.max_n_obs_steps
        else:
            n_obs = int(mx.random.randint(1, self.max_n_obs_steps + 1, ()).item())

        # Obs dims are visible for first n_obs steps
        # obs occupies last obs_dim dimensions
        obs_mask = mx.zeros((B, T, D), dtype=mx.bool_)
        # Set obs dimensions for first n_obs timesteps
        # ... (index assignment via mx.where or construction)

        return mask
```

### 3. `policy/base_image_policy.py`

```python
class BaseImagePolicy(mlx.nn.Module):
    """Abstract base class for image-conditioned policies."""

    def predict_action(self, obs_dict):
        """
        obs_dict: {key: (B, To, *shape)} observation dict
        returns: {'action': (B, Ta, Da), 'action_pred': (B, T, Da)}
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state (for recurrent policies)."""
        pass

    def set_normalizer(self, normalizer):
        """Set the data normalizer."""
        raise NotImplementedError
```

### 4. `policy/diffusion_unet_hybrid_image_policy.py`

```python
class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    """THE main policy: vision encoder + UNet diffusion.

    Architecture:
      obs_dict → MultiImageObsEncoder → obs_features
      obs_features + noise → ConditionalUnet1D → denoised actions
      DDPMScheduler controls noise addition (train) and removal (inference)

    Key modes:
      obs_as_global_cond=True (default):
        Observations encoded as global conditioning vector
        UNet input: (B, action_dim, horizon)
        UNet cond: (B, n_obs_steps * obs_feature_dim)

      obs_as_global_cond=False:
        Observations concatenated with noisy actions
        UNet input: (B, action_dim + obs_dim, horizon)
        Mask-based inpainting for conditioning
    """

    def __init__(self,
        shape_meta: dict,
        noise_scheduler,           # DDPMScheduler instance
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = None,
        obs_as_global_cond: bool = True,
        crop_shape: tuple = (76, 76),
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        **kwargs):

        super().__init__()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        # Build observation encoder
        obs_encoder = self._build_obs_encoder(
            shape_meta, crop_shape, obs_encoder_group_norm, eval_fixed_crop)
        self.obs_encoder = obs_encoder

        # Determine obs feature dim
        obs_feature_dim = obs_encoder.output_shape()

        # Build UNet
        if obs_as_global_cond:
            global_cond_dim = obs_feature_dim * n_obs_steps
            input_dim = shape_meta['action']['shape'][0]  # action_dim
            local_cond_dim = None
        else:
            global_cond_dim = None
            input_dim = shape_meta['action']['shape'][0] + obs_feature_dim
            local_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            local_cond_dim=local_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps or noise_scheduler.num_train_timesteps
        self.normalizer = LinearNormalizer()

        # Mask generator (for non-global-cond mode)
        if not obs_as_global_cond:
            self.mask_generator = LowdimMaskGenerator(
                action_dim=shape_meta['action']['shape'][0],
                obs_dim=obs_feature_dim,
                max_n_obs_steps=n_obs_steps,
            )

    def _build_obs_encoder(self, shape_meta, crop_shape,
                            obs_encoder_group_norm, eval_fixed_crop):
        """Build MultiImageObsEncoder from shape_meta."""
        rgb_model = get_resnet('resnet18')
        return MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            crop_shape=crop_shape,
            use_group_norm=obs_encoder_group_norm,
            share_rgb_model=True,
            imagenet_norm=True,
        )

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def predict_action(self, obs_dict):
        """Inference: generate actions via iterative denoising.

        Args:
            obs_dict: {'image': (B, To, C, H, W), 'agent_pos': (B, To, D), ...}

        Returns:
            {'action': (B, n_action_steps, action_dim),
             'action_pred': (B, horizon, action_dim)}
        """
        # Normalize observations
        nobs = self.normalizer.normalize({'obs': obs_dict})['obs']

        # Encode observations
        B = list(nobs.values())[0].shape[0]
        To = self.n_obs_steps

        if self.obs_as_global_cond:
            # Flatten temporal dim into batch for encoder
            # (B, To, C, H, W) → (B*To, C, H, W)
            obs_flat = {k: v[:, :To].reshape(B * To, *v.shape[2:])
                        for k, v in nobs.items()}
            obs_features = self.obs_encoder(obs_flat)  # (B*To, feat_dim)
            obs_features = obs_features.reshape(B, -1)  # (B, To*feat_dim)
            global_cond = obs_features
            local_cond = None

            # Initialize from noise
            action_dim = self.model.input_dim  # from UNet
            trajectory = mx.random.normal((B, self.horizon, action_dim))

            # Iterative denoising
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                # UNet expects (B, input_dim, T) per our convention
                model_input = mx.transpose(trajectory, axes=(0, 2, 1))
                noise_pred = self.model(model_input, t, global_cond=global_cond)
                # noise_pred comes back as (B, T, input_dim)

                out = self.noise_scheduler.step(noise_pred, t, trajectory)
                trajectory = out.prev_sample

        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(trajectory)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end, :]

        return {'action': action, 'action_pred': action_pred}

    def compute_loss(self, batch):
        """Training loss: MSE between predicted and true noise.

        Args:
            batch: {'obs': {...}, 'action': (B, T, Da)}

        Returns:
            scalar loss
        """
        # Normalize
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']  # (B, horizon, action_dim)

        B = naction.shape[0]
        To = self.n_obs_steps

        if self.obs_as_global_cond:
            # Encode observations
            obs_flat = {k: v[:, :To].reshape(B * To, *v.shape[2:])
                        for k, v in nobs.items()}
            obs_features = self.obs_encoder(obs_flat)
            obs_features = obs_features.reshape(B, -1)
            global_cond = obs_features

            trajectory = naction  # (B, horizon, action_dim)

            # Sample noise and timesteps
            noise = mx.random.normal(trajectory.shape)
            timesteps = mx.random.randint(
                0, self.noise_scheduler.num_train_timesteps, (B,))

            # Add noise
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps)

            # Predict noise
            model_input = mx.transpose(noisy_trajectory, axes=(0, 2, 1))
            noise_pred = self.model(
                model_input, timesteps, global_cond=global_cond)

            # MSE loss
            loss = mx.mean((noise_pred - noise) ** 2)
            return loss
```

---

## PushT Default Configuration

```python
# From upstream configs/image_pusht_diffusion_policy_cnn.yaml
shape_meta = {
    'obs': {
        'image': {'shape': (3, 96, 96), 'type': 'rgb'},
        'agent_pos': {'shape': (2,), 'type': 'low_dim'},
    },
    'action': {'shape': (2,)}
}

policy = DiffusionUnetHybridImagePolicy(
    shape_meta=shape_meta,
    noise_scheduler=DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    ),
    horizon=16,
    n_action_steps=8,
    n_obs_steps=2,
    num_inference_steps=100,
    obs_as_global_cond=True,
    crop_shape=(76, 76),
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
)
```

---

## Tests

### `tests/test_normalizer.py`

```python
def test_single_field_normalizer_limits():
    data = mx.random.uniform(shape=(100, 2)) * 10 - 5  # range [-5, 5]
    norm = SingleFieldLinearNormalizer.create_fit(data, mode='limits')
    normed = norm.normalize(data)
    # Should be in [-1, 1]
    assert float(mx.min(normed)) >= -1.01
    assert float(mx.max(normed)) <= 1.01
    # Round-trip
    recovered = norm.unnormalize(normed)
    np.testing.assert_allclose(np.array(recovered), np.array(data), atol=1e-5)

def test_single_field_normalizer_gaussian():
    data = mx.random.normal((1000, 3)) * 5 + 10
    norm = SingleFieldLinearNormalizer.create_fit(data, mode='gaussian')
    normed = norm.normalize(data)
    # Should have ~mean=0, ~std=1
    assert abs(float(mx.mean(normed))) < 0.1
    assert abs(float(mx.std(normed)) - 1.0) < 0.1

def test_linear_normalizer_dict():
    norm = LinearNormalizer()
    norm.fit({
        'action': mx.random.uniform(shape=(100, 16, 2)),
        'agent_pos': mx.random.uniform(shape=(100, 16, 2)),
    })
    batch = {
        'action': mx.random.uniform(shape=(4, 16, 2)),
        'agent_pos': mx.random.uniform(shape=(4, 16, 2)),
    }
    out = norm.normalize(batch)
    assert out['action'].shape == (4, 16, 2)
```

### `tests/test_policy.py`

```python
def test_policy_predict_action_shape():
    """predict_action returns correct shapes."""
    shape_meta = {
        'obs': {
            'image': {'shape': (3, 96, 96), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)}
    }
    scheduler = DDPMScheduler(num_train_timesteps=10)
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=16, n_action_steps=8, n_obs_steps=2,
        num_inference_steps=5,
        down_dims=(32, 64),  # small for testing
    )
    # Set identity normalizer
    policy.set_normalizer(...)

    obs = {
        'image': mx.random.normal((2, 2, 3, 96, 96)),  # (B, To, C, H, W)
        'agent_pos': mx.random.normal((2, 2, 2)),       # (B, To, D)
    }
    result = policy.predict_action(obs)
    assert result['action'].shape == (2, 8, 2)  # (B, n_action_steps, Da)
    assert result['action_pred'].shape == (2, 16, 2)  # (B, horizon, Da)

def test_policy_compute_loss_scalar():
    """compute_loss returns a scalar."""
    # ... similar setup
    batch = {
        'obs': {
            'image': mx.random.normal((4, 16, 3, 96, 96)),
            'agent_pos': mx.random.normal((4, 16, 2)),
        },
        'action': mx.random.normal((4, 16, 2)),
    }
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0  # scalar
    assert float(loss) > 0

def test_policy_loss_is_differentiable():
    """Gradients flow through compute_loss."""
    # ... setup
    def loss_fn(model, batch):
        return model.compute_loss(batch)
    loss, grads = mlx.nn.value_and_grad(policy, loss_fn)(policy, batch)
    assert float(loss) > 0
    # Check some grads are non-zero
    ...
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | `predict_action` output shape: (B, n_action_steps, Da) | exact |
| 2 | `compute_loss` returns scalar > 0 | exact |
| 3 | Loss is differentiable (grads non-zero) | non-zero grads |
| 4 | Normalizer round-trip: normalize → unnormalize ≈ identity | atol=1e-5 |
| 5 | Policy with default PushT config instantiates without error | no error |
| 6 | Inference loop runs for N steps without error | no NaN/Inf |

---

## Upstream Sync Notes

**Files to watch:**
- `policy/diffusion_unet_hybrid_image_policy.py` — constructor params, predict_action logic
- `model/common/normalizer.py` — normalization modes and edge cases
- Upstream configs in `diffusion_policy/config/` — default hyperparameters

**Checkpoint compatibility:** Weight conversion from PyTorch checkpoints requires mapping upstream parameter paths to our module hierarchy. The normalizer state must be preserved separately (it's not part of the neural network).
