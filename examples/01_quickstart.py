"""Minimal example: create a diffusion policy and run inference on random data.

No dataset download required -- uses synthetic observations.
Demonstrates the core predict_action() API in ~20 lines of code.
"""

import time

import mlx.core as mx

from diffusion_policy_mlx import (
    DDPMScheduler,
    DiffusionUnetHybridImagePolicy,
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def main():
    print("=== Diffusion Policy MLX: Quickstart ===\n")

    # 1. Define the observation and action shapes
    shape_meta = {
        "obs": {
            "image": {"shape": (3, 96, 96), "type": "rgb"},
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }

    # 2. Create a noise scheduler (DDPM with cosine schedule)
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # 3. Build the policy with a small architecture (fast for demo)
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=3,  # few steps for speed
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
        crop_shape=(76, 76),
    )

    # 4. Set up an identity normalizer (no real data to fit)
    normalizer = LinearNormalizer()
    normalizer.params_dict["obs"] = {
        "image": SingleFieldLinearNormalizer.create_identity(shape=(3,)),
        "agent_pos": SingleFieldLinearNormalizer.create_identity(shape=(2,)),
    }
    normalizer.params_dict["action"] = SingleFieldLinearNormalizer.create_identity(shape=(2,))
    policy.set_normalizer(normalizer)

    # Materialize all model parameters
    mx.eval(policy.parameters())

    # 5. Create synthetic observations (batch=1, 2 observation steps)
    obs = {
        "image": mx.random.normal((1, 2, 3, 96, 96)),
        "agent_pos": mx.random.normal((1, 2, 2)),
    }

    # 6. Run inference and time it
    print("Running predict_action()...")
    start = time.perf_counter()
    result = policy.predict_action(obs)
    mx.eval(result["action"], result["action_pred"])
    elapsed = time.perf_counter() - start

    # 7. Print results
    act_shape = result["action"].shape
    print(f"  action shape:      {act_shape}  (batch, n_action_steps, action_dim)")
    print(f"  action_pred shape: {result['action_pred'].shape}  (batch, horizon, action_dim)")
    print(f"  action sample:     {result['action'][0, 0].tolist()}")
    print(f"  inference time:    {elapsed:.3f}s")
    print("\nDone! The policy generated an 8-step action trajectory from random observations.")


if __name__ == "__main__":
    main()
