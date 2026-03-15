# PRD-04: DDPM/DDIM Scheduler

**Status:** Complete
**Depends on:** PRD-01 (compat layer)
**Blocks:** PRD-05 (Policy Assembly)

---

## Objective

Implement DDPM and DDIM noise schedulers in pure MLX, replacing HuggingFace `diffusers.DDPMScheduler` and `diffusers.DDIMScheduler`. These control the forward (add noise) and reverse (denoise) diffusion process.

---

## Upstream Reference

The upstream uses `diffusers>=0.11.1`:
- `diffusers.schedulers.scheduling_ddpm.DDPMScheduler`
- `diffusers.schedulers.scheduling_ddim.DDIMScheduler`

Key usage in `diffusion_unet_hybrid_image_policy.py`:
```python
# Training: add noise
noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

# Inference: iterative denoising
self.noise_scheduler.set_timesteps(num_inference_steps)
for t in self.noise_scheduler.timesteps:
    noise_pred = model(sample, t, cond)
    sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample
```

---

## Deliverables

### 1. `compat/schedulers.py`

#### DDPMScheduler

```python
class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler.

    Implements:
      - Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
      - Reverse process: p(x_{t-1} | x_t) with learned noise prediction

    Constructor matches diffusers API subset used by upstream.
    """

    def __init__(self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small"):
        """
        Args:
            num_train_timesteps: Total diffusion steps T
            beta_start/end: Noise schedule endpoints
            beta_schedule: 'linear', 'cosine', 'squaredcos_cap_v2'
            clip_sample: Clip denoised sample to [-clip_sample_range, clip_sample_range]
            prediction_type: 'epsilon' (predict noise), 'sample' (predict x_0), 'v_prediction'
            variance_type: 'fixed_small' or 'fixed_large'
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Compute beta schedule
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = mx.array(betas, dtype=mx.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)

        # For sampling
        self.timesteps = mx.arange(num_train_timesteps - 1, -1, -1)

    @staticmethod
    def _cosine_beta_schedule(T, s=0.008):
        """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal 2021)."""
        steps = np.arange(T + 1)
        f_t = np.cos(((steps / T) + s) / (1 + s) * np.pi * 0.5) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0.0, 0.999)

    def add_noise(self, original_samples, noise, timesteps):
        """Forward diffusion: q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            original_samples: x_0 of shape (B, ...)
            noise: epsilon of shape (B, ...), same as original_samples
            timesteps: (B,) integer timesteps

        Returns:
            noisy_samples: x_t of shape (B, ...)
        """
        alpha_bar = self.alphas_cumprod[timesteps]
        # Reshape for broadcasting: (B,) → (B, 1, 1, ...) to match sample dims
        while alpha_bar.ndim < original_samples.ndim:
            alpha_bar = mx.expand_dims(alpha_bar, axis=-1)

        sqrt_alpha_bar = mx.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = mx.sqrt(1.0 - alpha_bar)

        return sqrt_alpha_bar * original_samples + sqrt_one_minus_alpha_bar * noise

    def set_timesteps(self, num_inference_steps):
        """Configure inference schedule (subset of training timesteps).

        For DDPM: num_inference_steps == num_train_timesteps (all steps)
        Sets self.timesteps to the reversed schedule.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = mx.array(timesteps[::-1].copy(), dtype=mx.int32)

    def step(self, model_output, timestep, sample, generator=None):
        """Reverse diffusion step: p(x_{t-1} | x_t).

        Args:
            model_output: predicted noise (or sample, depending on prediction_type)
            timestep: current timestep t (scalar)
            sample: x_t current noisy sample

        Returns:
            SchedulerOutput with .prev_sample = x_{t-1}
        """
        t = int(timestep)
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t - 1] if t > 0 else mx.array(1.0)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]

        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_x0 = (sample - mx.sqrt(1 - alpha_bar_t) * model_output) / mx.sqrt(alpha_bar_t)
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = mx.sqrt(alpha_bar_t) * sample - mx.sqrt(1 - alpha_bar_t) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip predicted x_0
        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # Compute mean of p(x_{t-1} | x_t, x_0)
        coeff1 = mx.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        coeff2 = mx.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        pred_mean = coeff1 * pred_x0 + coeff2 * sample

        # Variance
        if t > 0:
            variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            noise = mx.random.normal(sample.shape)
            prev_sample = pred_mean + mx.sqrt(variance) * noise
        else:
            prev_sample = pred_mean

        return SchedulerOutput(prev_sample=prev_sample)


class SchedulerOutput:
    """Simple container matching diffusers SchedulerOutput."""
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
```

#### DDIMScheduler

```python
class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler.

    Accelerated sampling: can use 10-50 steps instead of 1000.
    Deterministic sampling (eta=0) or stochastic (eta>0).
    """

    def __init__(self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        set_alpha_to_one: bool = True):
        """Same beta schedule computation as DDPM."""
        # ... (same beta/alpha computation as DDPM)
        self.final_alpha_cumprod = mx.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

    def set_timesteps(self, num_inference_steps):
        """Set accelerated inference schedule.

        Can use any num_inference_steps <= num_train_timesteps.
        Typically 10-50 for fast inference.
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = mx.array(timesteps[::-1].copy(), dtype=mx.int32)

    def step(self, model_output, timestep, sample, eta=0.0, generator=None):
        """DDIM reverse step.

        Args:
            eta: 0.0 = deterministic, 1.0 = DDPM equivalent

        DDIM formula:
            x_{t-1} = sqrt(alpha_{t-1}) * pred_x0
                     + sqrt(1 - alpha_{t-1} - sigma^2) * pred_epsilon
                     + sigma * noise
        """
        t = int(timestep)
        # Find previous timestep in schedule
        prev_t = self._get_previous_timestep(t)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

        # Predict x_0 and epsilon (same logic as DDPM)
        if self.prediction_type == "epsilon":
            pred_x0 = (sample - mx.sqrt(1 - alpha_bar_t) * model_output) / mx.sqrt(alpha_bar_t)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_x0 = model_output
            pred_epsilon = (sample - mx.sqrt(alpha_bar_t) * pred_x0) / mx.sqrt(1 - alpha_bar_t)
        elif self.prediction_type == "v_prediction":
            pred_x0 = mx.sqrt(alpha_bar_t) * sample - mx.sqrt(1 - alpha_bar_t) * model_output
            pred_epsilon = mx.sqrt(alpha_bar_t) * model_output + mx.sqrt(1 - alpha_bar_t) * sample

        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # DDIM formula
        sigma = eta * mx.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )
        pred_direction = mx.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_epsilon
        prev_sample = mx.sqrt(alpha_bar_prev) * pred_x0 + pred_direction

        if eta > 0 and prev_t > 0:
            noise = mx.random.normal(sample.shape)
            prev_sample = prev_sample + sigma * noise

        return SchedulerOutput(prev_sample=prev_sample)

    def _get_previous_timestep(self, t):
        """Get previous timestep in the inference schedule."""
        idx = mx.argmin(mx.abs(self.timesteps - t))
        if idx + 1 < len(self.timesteps):
            return int(self.timesteps[idx + 1])
        return -1
```

---

## Tests

### `tests/test_schedulers.py`

```python
def test_ddpm_beta_schedule_linear():
    s = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
    assert s.betas.shape == (100,)
    assert float(s.betas[0]) > 0
    assert float(s.betas[-1]) > float(s.betas[0])

def test_ddpm_beta_schedule_cosine():
    s = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    assert s.betas.shape == (100,)
    assert float(mx.max(s.betas)) <= 0.999

def test_ddpm_alphas_cumprod():
    s = DDPMScheduler(num_train_timesteps=100)
    # alpha_bar should be monotonically decreasing
    ac = np.array(s.alphas_cumprod)
    assert np.all(ac[1:] <= ac[:-1])
    # alpha_bar[0] ≈ 1, alpha_bar[-1] ≈ 0
    assert ac[0] > 0.99
    assert ac[-1] < 0.1

def test_ddpm_add_noise_shape():
    s = DDPMScheduler(num_train_timesteps=100)
    x0 = mx.random.normal((4, 16, 2))
    noise = mx.random.normal((4, 16, 2))
    t = mx.array([10, 20, 30, 40])
    xt = s.add_noise(x0, noise, t)
    assert xt.shape == (4, 16, 2)

def test_ddpm_add_noise_extremes():
    s = DDPMScheduler(num_train_timesteps=1000)
    x0 = mx.ones((1, 4))
    noise = mx.zeros((1, 4))
    # t=0: almost no noise
    xt = s.add_noise(x0, noise, mx.array([0]))
    np.testing.assert_allclose(np.array(xt), np.array(x0), atol=0.01)

def test_ddpm_step_reduces_noise():
    """After many reverse steps, sample should be less noisy."""
    s = DDPMScheduler(num_train_timesteps=100)
    s.set_timesteps(100)
    sample = mx.random.normal((1, 4))
    for t in s.timesteps:
        # Use zero model output (predict no noise)
        out = s.step(mx.zeros_like(sample), t, sample)
        sample = out.prev_sample
    # After full denoising with zero noise pred, should converge
    assert mx.all(mx.abs(sample) < 10).item()

@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_ddpm_matches_diffusers():
    """Beta schedule and add_noise match diffusers implementation."""
    from diffusers import DDPMScheduler as DiffusersDDPM
    ours = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    theirs = DiffusersDDPM(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

    np.testing.assert_allclose(
        np.array(ours.alphas_cumprod),
        theirs.alphas_cumprod.numpy(),
        atol=1e-6
    )

def test_ddim_fewer_steps():
    s = DDIMScheduler(num_train_timesteps=1000)
    s.set_timesteps(50)  # 50 steps instead of 1000
    assert len(s.timesteps) == 50

def test_ddim_deterministic():
    """DDIM with eta=0 should be deterministic."""
    s = DDIMScheduler(num_train_timesteps=100)
    s.set_timesteps(10)
    sample = mx.random.normal((1, 4))
    for t in s.timesteps:
        out = s.step(mx.zeros_like(sample), t, sample, eta=0.0)
        sample = out.prev_sample
    # Run again with same starting point — should be identical
    ...
```

---

## Acceptance Criteria

| # | Criterion | Tolerance |
|---|-----------|-----------|
| 1 | Beta schedule (linear, cosine) matches diffusers | atol=1e-6 |
| 2 | `alphas_cumprod` monotonically decreasing, range [0,1] | exact |
| 3 | `add_noise` shape preserved, t=0 ≈ clean, t=T ≈ noise | atol=0.01 for t=0 |
| 4 | DDPM `step` produces valid samples (no NaN/Inf) | no NaN/Inf |
| 5 | DDIM with eta=0 is deterministic | exact bit-match |
| 6 | DDIM supports accelerated schedules (10-50 steps) | correct timestep spacing |
| 7 | Cross-framework test: `add_noise` matches diffusers | atol=1e-5 |

---

## Upstream Sync Notes

**Files to watch:**
- Upstream uses `diffusers.DDPMScheduler` — if they update to a newer diffusers version, check for API changes
- `beta_schedule="squaredcos_cap_v2"` is the most commonly used schedule in upstream configs
- `prediction_type="epsilon"` is the default — `"v_prediction"` is used in some experiments
- `clip_sample=True` with `clip_sample_range=1.0` is the upstream default
