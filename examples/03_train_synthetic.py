"""Train a diffusion policy on synthetic data for 50 steps.

Demonstrates the full training pipeline -- loss computation, gradient
updates, and inference -- without needing any real dataset download.
Completes in under 30 seconds.
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np

from diffusion_policy_mlx import (
    DDPMScheduler,
    DiffusionUnetHybridImagePolicy,
    LinearNormalizer,
)


def make_synthetic_batch(batch_size, horizon, n_obs_steps):
    """Generate a synthetic training batch with circular motion actions."""
    # Synthetic images: random noise (simulating camera observations)
    images = mx.random.normal((batch_size, horizon, 3, 96, 96)) * 0.1
    # Synthetic agent positions: random 2D positions
    agent_pos = mx.random.normal((batch_size, horizon, 2)) * 0.5
    # Synthetic actions: circular motion (learnable pattern)
    t = mx.broadcast_to(
        mx.arange(horizon).reshape(1, horizon), (batch_size, horizon)
    ).astype(mx.float32) / horizon * 2.0 * float(np.pi)
    actions = mx.stack([mx.cos(t), mx.sin(t)], axis=-1) * 0.3
    return {
        "obs": {"image": images, "agent_pos": agent_pos},
        "action": actions,
    }


def main():
    print("=== Training Diffusion Policy on Synthetic Data ===\n")
    mx.random.seed(42)
    np.random.seed(42)

    # 1. Configuration
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    batch_size = 4
    num_train_steps = 50
    num_train_timesteps = 10  # small for speed

    # 2. Build policy
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=3,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        crop_shape=(76, 76),
    )

    # 3. Fit normalizer on a sample batch
    sample = make_synthetic_batch(16, horizon, n_obs_steps)
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": sample["obs"], "action": sample["action"]})
    policy.set_normalizer(normalizer)
    mx.eval(policy.parameters())

    # 4. Optimizer
    optimizer = mlx.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-6)

    # 5. Training loop
    loss_and_grad_fn = nn.value_and_grad(policy, lambda m, b: m.compute_loss(b))
    losses = []
    start = time.perf_counter()

    print(f"Training for {num_train_steps} steps (batch_size={batch_size})...\n")
    for step in range(num_train_steps):
        batch = make_synthetic_batch(batch_size, horizon, n_obs_steps)
        loss, grads = loss_and_grad_fn(policy, batch)
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state)

        loss_val = float(loss)
        losses.append(loss_val)
        if (step + 1) % 10 == 0:
            print(f"  step {step+1:3d}/{num_train_steps}  loss={loss_val:.4f}")

    elapsed = time.perf_counter() - start
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/num_train_steps:.2f}s/step)")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss decreased: {losses[0] > losses[-1]}")

    # 6. Run inference
    obs = {
        "image": mx.random.normal((1, n_obs_steps, 3, 96, 96)),
        "agent_pos": mx.random.normal((1, n_obs_steps, 2)),
    }
    result = policy.predict_action(obs)
    mx.eval(result["action"])
    print(f"\nInference output shape: {result['action'].shape}")
    print(f"  Predicted actions (first 3 steps): {result['action'][0, :3].tolist()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
