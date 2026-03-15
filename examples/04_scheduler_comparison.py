"""Compare DDPM and DDIM schedulers: quality vs speed tradeoff.

Shows how DDIM generates actions faster with fewer denoising steps,
while DDPM uses the full reverse process. Both start from the same noise.
"""

import time

import mlx.core as mx
import numpy as np

from diffusion_policy_mlx import (
    DDIMScheduler,
    DDPMScheduler,
    DiffusionUnetHybridImagePolicy,
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def build_policy(scheduler, num_inference_steps):
    """Build a small policy with the given scheduler."""
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        crop_shape=(76, 76),
    )
    normalizer = LinearNormalizer()
    normalizer.params_dict["obs"] = {
        "image": SingleFieldLinearNormalizer.create_identity(shape=(3,)),
        "agent_pos": SingleFieldLinearNormalizer.create_identity(shape=(2,)),
    }
    normalizer.params_dict["action"] = SingleFieldLinearNormalizer.create_identity(shape=(2,))
    policy.set_normalizer(normalizer)
    mx.eval(policy.parameters())
    return policy


def time_inference(policy, obs, label, n_runs=3):
    """Time inference over multiple runs, return (actions, avg_time)."""
    # Warmup
    result = policy.predict_action(obs)
    mx.eval(result["action"])

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = policy.predict_action(obs)
        mx.eval(result["action"])
        times.append(time.perf_counter() - start)

    avg = np.mean(times)
    print(f"  {label}: {avg:.4f}s avg ({n_runs} runs)")
    return result["action"], avg


def main():
    print("=== Scheduler Comparison: DDPM vs DDIM ===\n")
    mx.random.seed(42)

    num_train_timesteps = 100

    # Build DDPM policy (many inference steps)
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    ddpm_steps = 100
    policy_ddpm = build_policy(ddpm_scheduler, num_inference_steps=ddpm_steps)

    # Build DDIM policy (few inference steps) -- share the same weights
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    ddim_steps = 10
    policy_ddim = build_policy(ddim_scheduler, num_inference_steps=ddim_steps)
    # Copy weights from DDPM to DDIM so they use the same UNet
    import mlx.utils
    flat_weights = mlx.utils.tree_flatten(policy_ddpm.parameters())
    policy_ddim.load_weights(flat_weights)
    mx.eval(policy_ddim.parameters())

    # Create observations
    obs = {
        "image": mx.random.normal((1, 2, 3, 96, 96)),
        "agent_pos": mx.random.normal((1, 2, 2)),
    }

    print("Timing inference:\n")
    action_ddpm, time_ddpm = time_inference(policy_ddpm, obs, f"DDPM ({ddpm_steps} steps)")
    action_ddim, time_ddim = time_inference(policy_ddim, obs, f"DDIM ({ddim_steps} steps)")

    speedup = time_ddpm / time_ddim if time_ddim > 0 else float("inf")

    print("\nResults:")
    ddpm_lo, ddpm_hi = float(mx.min(action_ddpm)), float(mx.max(action_ddpm))
    ddim_lo, ddim_hi = float(mx.min(action_ddim)), float(mx.max(action_ddim))
    print(f"  DDPM action range: [{ddpm_lo:.3f}, {ddpm_hi:.3f}]")
    print(f"  DDIM action range: [{ddim_lo:.3f}, {ddim_hi:.3f}]")
    print(f"  DDIM speedup:      {speedup:.1f}x faster")
    print(f"\n  DDPM: {ddpm_steps} denoising steps, higher quality, slower")
    print(f"  DDIM: {ddim_steps} denoising steps, deterministic, much faster")
    print("\nDone!")


if __name__ == "__main__":
    main()
