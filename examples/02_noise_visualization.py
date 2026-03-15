"""Visualize the forward and reverse diffusion process.

Shows how noise is added (forward process) and removed (reverse process)
step by step. Saves a figure to examples/outputs/diffusion_process.png
if matplotlib is available, otherwise prints a text summary.
"""

import os

import mlx.core as mx
import numpy as np

from diffusion_policy_mlx import DDPMScheduler


def main():
    print("=== Diffusion Process Visualization ===\n")
    mx.random.seed(42)

    # 1. Create a clean 1D signal: a sine wave (simulating an action trajectory)
    T = 64  # trajectory length
    t = np.linspace(0, 4 * np.pi, T)
    clean_signal = np.sin(t).astype(np.float32)
    x0 = mx.array(clean_signal).reshape(1, T, 1)  # (B=1, T, D=1)

    # 2. Set up the scheduler
    num_train_steps = 100
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    # 3. Forward process: progressively add noise at selected timesteps
    forward_timesteps = [0, 10, 25, 50, 75, 99]
    noise = mx.random.normal(x0.shape)
    forward_samples = {}
    print("Forward process (adding noise):")
    for ts in forward_timesteps:
        t_arr = mx.array([ts], dtype=mx.int32)
        noisy = scheduler.add_noise(x0, noise, t_arr)
        mx.eval(noisy)
        snr = float(scheduler.alphas_cumprod[ts] / (1.0 - scheduler.alphas_cumprod[ts]))
        lo, hi = float(mx.min(noisy)), float(mx.max(noisy))
        print(f"  t={ts:3d}  SNR={snr:8.3f}  range=[{lo:.2f}, {hi:.2f}]")
        forward_samples[ts] = np.array(noisy[0, :, 0])

    # 4. Reverse process: denoise from pure noise using the scheduler
    #    We use a simple "oracle" model that knows the true noise for demonstration.
    print("\nReverse process (removing noise):")
    scheduler.set_timesteps(20)  # 20 inference steps
    trajectory = mx.random.normal(x0.shape)
    reverse_snapshots = {"start": np.array(trajectory[0, :, 0])}

    steps_taken = 0
    for t_step in scheduler.timesteps:
        # Oracle: compute the true noise at this timestep
        alpha_bar = scheduler.alphas_cumprod[int(t_step)]
        true_noise = (trajectory - mx.sqrt(alpha_bar) * x0) / mx.sqrt(1.0 - alpha_bar)
        out = scheduler.step(true_noise, t_step, trajectory)
        trajectory = out.prev_sample
        steps_taken += 1
        if steps_taken in (1, 5, 10, 15, 20):
            mx.eval(trajectory)
            reverse_snapshots[f"step_{steps_taken}"] = np.array(trajectory[0, :, 0])

    mx.eval(trajectory)
    mse = float(mx.mean((trajectory - x0) ** 2))
    print(f"  After {steps_taken} denoising steps, MSE to original: {mse:.6f}")
    reverse_snapshots["final"] = np.array(trajectory[0, :, 0])

    # 5. Try to save a visualization
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "diffusion_process.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Forward process
        ax = axes[0]
        ax.set_title("Forward Diffusion: Clean Signal -> Pure Noise", fontsize=13)
        for ts in forward_timesteps:
            alpha = 1.0 - ts / (num_train_steps - 1)
            ax.plot(forward_samples[ts], alpha=max(alpha, 0.15), label=f"t={ts}")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Reverse process
        ax = axes[1]
        ax.set_title("Reverse Diffusion: Noise -> Recovered Signal", fontsize=13)
        labels = list(reverse_snapshots.keys())
        for i, (label, data) in enumerate(reverse_snapshots.items()):
            alpha = 0.2 + 0.8 * (i / max(len(labels) - 1, 1))
            ax.plot(data, alpha=alpha, label=label)
        ax.plot(clean_signal, "k--", linewidth=2, alpha=0.5, label="ground truth")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Value")
        ax.set_xlabel("Trajectory Timestep")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\nFigure saved to: {output_path}")
    except ImportError:
        print("\nmatplotlib not installed -- skipping figure. Install it for visual output.")

    print("\nDone! The forward process adds noise; the reverse process recovers the signal.")


if __name__ == "__main__":
    main()
