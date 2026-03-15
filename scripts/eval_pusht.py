"""Evaluate a trained Diffusion Policy on PushT environment.

This script loads a converted MLX checkpoint, runs inference against
the PushT environment, and reports success rate, average reward,
and inference timing.

Supports two modes:
  1. Real PushT environment (``--env real``, default) using the pymunk-based
     PushT simulation with PushTImageRunner.
  2. Synthetic environment (``--env synthetic``) for testing the evaluation
     pipeline without physics dependencies.

Usage:
    python scripts/eval_pusht.py \\
        --checkpoint checkpoints/pusht_mlx \\
        --n-episodes 50

    # With real PushT environment (default):
    python scripts/eval_pusht.py --checkpoint dir/ --n-episodes 10

    # With synthetic environment (testing only):
    python scripts/eval_pusht.py --checkpoint dir/ --n-episodes 10 --env synthetic

    # Show per-episode details:
    python scripts/eval_pusht.py --checkpoint dir/ --n-episodes 10 --verbose
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np

from diffusion_policy_mlx.compat.schedulers import DDIMScheduler, DDPMScheduler
from diffusion_policy_mlx.env.pusht_env import PushTEnv
from diffusion_policy_mlx.env.pusht_image_runner import PushTImageRunner
from diffusion_policy_mlx.model.diffusion.conditional_unet1d import ConditionalUnet1D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    episode: int
    total_reward: float
    success: bool
    num_steps: int
    inference_time_ms: float


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.success for e in self.episodes]))

    @property
    def avg_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_reward for e in self.episodes]))

    @property
    def avg_inference_ms(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.inference_time_ms for e in self.episodes]))

    def summary(self) -> str:
        return (
            f"Results over {self.n_episodes} episodes:\n"
            f"  Success rate: {self.success_rate:.1%}\n"
            f"  Average reward: {self.avg_reward:.3f}\n"
            f"  Avg inference: {self.avg_inference_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class DiffusionPolicyInference:
    """Inference wrapper for a converted Diffusion Policy.

    Encapsulates the UNet model and diffusion scheduler for action prediction.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        scheduler,
        action_dim: int = 2,
        obs_dim: int = 514,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
    ):
        self.model = model
        self.scheduler = scheduler
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps

    def predict_action(self, obs_features: mx.array) -> mx.array:
        """Predict action sequence from observation features.

        Args:
            obs_features: (B, n_obs_steps * obs_dim) global conditioning

        Returns:
            (B, n_action_steps, action_dim) predicted actions
        """
        B = obs_features.shape[0]
        T = self.horizon
        Da = self.action_dim

        # Initialize noisy trajectory
        trajectory = mx.random.normal((B, T, Da))

        # Set up scheduler
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop
        for t in self.scheduler.timesteps:
            t_val = int(t.item()) if isinstance(t, mx.array) else int(t)

            model_output = self.model(
                trajectory,
                mx.array([t_val], dtype=mx.int32),
                global_cond=obs_features,
            )
            mx.eval(model_output)

            result = self.scheduler.step(model_output, t_val, trajectory)
            trajectory = result.prev_sample
            mx.eval(trajectory)

        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        actions = trajectory[:, start:end, :]

        return actions


def load_policy(
    checkpoint_dir: str,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    num_inference_steps: int = 100,
    use_ddim: bool = False,
    ddim_steps: int = 10,
) -> DiffusionPolicyInference:
    """Load a converted Diffusion Policy from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model.npz (and optionally ema.npz)
        action_dim: Action dimension (2 for PushT)
        obs_dim: Observation feature dimension
        horizon: Prediction horizon
        n_obs_steps: Number of observation steps
        n_action_steps: Number of action steps to execute
        num_inference_steps: DDPM inference steps
        use_ddim: Use DDIM scheduler for faster inference
        ddim_steps: Number of DDIM steps

    Returns:
        DiffusionPolicyInference wrapper
    """
    ckpt_path = Path(checkpoint_dir)

    # Create model
    global_cond_dim = obs_dim * n_obs_steps
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )

    # Load weights (prefer EMA if available)
    ema_path = ckpt_path / "ema.npz"
    model_path = ckpt_path / "model.npz"

    weights_path = ema_path if ema_path.exists() else model_path
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        # Filter to model.* keys and strip prefix
        model_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                model_weights[k[len("model.") :]] = v
        if model_weights:
            model.load_weights(list(model_weights.items()))
            logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning(
            "No weights found at %s, using random initialization",
            checkpoint_dir,
        )

    # Create scheduler
    if use_ddim:
        scheduler = DDIMScheduler(
            num_train_timesteps=num_inference_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        effective_steps = ddim_steps
    else:
        scheduler = DDPMScheduler(
            num_train_timesteps=num_inference_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        effective_steps = num_inference_steps

    return DiffusionPolicyInference(
        model=model,
        scheduler=scheduler,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=effective_steps,
    )


# ---------------------------------------------------------------------------
# Synthetic PushT environment (placeholder)
# ---------------------------------------------------------------------------


class SyntheticPushTEnv:
    """Synthetic PushT environment for testing the evaluation pipeline.

    Generates random observations and computes a simple reward based on
    action magnitude (placeholder for actual PushT physics).

    The real PushT environment requires pygame and shapely. This placeholder
    allows testing the full evaluation pipeline without those dependencies.
    """

    def __init__(
        self,
        obs_dim: int = 514,
        action_dim: int = 2,
        max_steps: int = 300,
        seed: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._total_reward = 0.0
        self._target = self._rng.randn(action_dim) * 0.5

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self._step_count = 0
        self._total_reward = 0.0
        self._target = self._rng.randn(self.action_dim) * 0.5
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)."""
        self._step_count += 1

        # Simple reward: negative distance to target
        dist = np.linalg.norm(action[: self.action_dim] - self._target)
        reward = max(0.0, 1.0 - dist)
        self._total_reward += reward

        done = self._step_count >= self.max_steps
        info = {
            "success": self._total_reward / max(1, self._step_count) > 0.5,
            "step": self._step_count,
        }

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Generate synthetic observation."""
        return self._rng.randn(self.obs_dim).astype(np.float32)


# ---------------------------------------------------------------------------
# MLX Policy adapter for PushTImageRunner
# ---------------------------------------------------------------------------


class MLXPolicyAdapter:
    """Wraps DiffusionPolicyInference for the PushTImageRunner interface.

    The runner provides obs_dict with ``image`` (B, T, H, W, C) and
    ``agent_pos`` (B, T, 2).  This adapter encodes images to features
    (or uses a flat representation) and calls the underlying diffusion
    policy to predict actions.
    """

    def __init__(self, policy: DiffusionPolicyInference):
        self.policy = policy

    def predict_action(self, obs_dict):
        """Predict actions from observation dict.

        For now, encode image as flattened mean features + agent_pos to
        match the policy's expected obs_dim.  A proper vision encoder
        would be used in a full pipeline.
        """
        import mlx.core as mx

        images = obs_dict["image"]  # (B, T, H, W, C) float32
        agent_pos = obs_dict["agent_pos"]  # (B, T, 2)

        B, T = images.shape[:2]

        # Simple feature extraction: downsample + flatten per timestep
        # Then concatenate with agent_pos to form global conditioning.
        # The target obs_dim per step is self.policy.obs_dim
        obs_dim = self.policy.obs_dim
        features_list = []
        for t_idx in range(T):
            img = images[:, t_idx]  # (B, H, W, C)
            pos = agent_pos[:, t_idx]  # (B, 2)
            # Flatten image to fixed-size feature via mean pooling on patches
            # Image is (B, 96, 96, 3) => pool to (B, 512) to match obs_dim - 2
            feat_dim = obs_dim - 2
            # Simple: reshape to patches and mean-pool
            img_flat = img.reshape(B, -1)  # (B, H*W*C)
            # Downsample to feat_dim via strided selection
            if img_flat.shape[1] >= feat_dim:
                stride = img_flat.shape[1] // feat_dim
                img_feat = img_flat[:, :feat_dim * stride:stride][:, :feat_dim]
            else:
                # Pad
                img_feat = np.zeros((B, feat_dim), dtype=np.float32)
                img_feat[:, :img_flat.shape[1]] = img_flat
            # Normalize
            img_feat = img_feat / 255.0
            # Concat with agent_pos (normalized to [0, 1])
            pos_norm = pos / 512.0
            step_feat = np.concatenate([img_feat, pos_norm], axis=-1)  # (B, obs_dim)
            features_list.append(step_feat)

        # Stack and flatten across time
        features = np.concatenate(features_list, axis=-1)  # (B, T * obs_dim)
        features_mx = mx.array(features, dtype=mx.float32)

        # Predict
        actions_mx = self.policy.predict_action(features_mx)
        mx.eval(actions_mx)

        # Convert back to numpy and scale to workspace
        actions = np.array(actions_mx)
        # Clamp to workspace
        actions = np.clip(actions * 512, 0, 512)

        return actions


# ---------------------------------------------------------------------------
# Evaluation with real PushT environment
# ---------------------------------------------------------------------------


def evaluate_real(
    checkpoint_dir: str,
    n_episodes: int = 50,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    num_inference_steps: int = 100,
    use_ddim: bool = False,
    ddim_steps: int = 10,
    max_episode_steps: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run policy evaluation using the real PushT environment + runner."""
    policy = load_policy(
        checkpoint_dir=checkpoint_dir,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        use_ddim=use_ddim,
        ddim_steps=ddim_steps,
    )

    adapter = MLXPolicyAdapter(policy)
    env = PushTEnv(render_size=96, max_steps=max_episode_steps)
    runner = PushTImageRunner(
        policy=adapter,
        env=env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_episode_steps,
    )

    result = runner.run(n_episodes=n_episodes, seed=seed, verbose=verbose)

    print(f"\n{'=' * 50}")
    print(f"Results over {n_episodes} episodes (real PushT env):")
    print(f"  Success rate: {result['success_rate']:.1%}")
    print(f"  Mean reward: {result['mean_reward']:.3f}")
    print(f"  Mean max reward: {result['mean_max_reward']:.3f}")
    print(f"  Mean episode length: {result['mean_episode_length']:.1f}")
    print(f"  Mean inference time: {result['mean_inference_time'] * 1000:.1f}ms")
    print(f"{'=' * 50}")

    return result


# ---------------------------------------------------------------------------
# Evaluation loop (synthetic / legacy)
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_dir: str,
    n_episodes: int = 50,
    action_dim: int = 2,
    obs_dim: int = 514,
    horizon: int = 16,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    num_inference_steps: int = 100,
    use_ddim: bool = False,
    ddim_steps: int = 10,
    max_episode_steps: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> EvalResult:
    """Run policy evaluation on PushT environment.

    Args:
        checkpoint_dir: Path to converted checkpoint directory
        n_episodes: Number of evaluation episodes
        action_dim: Action dimension
        obs_dim: Observation feature dimension
        horizon: Prediction horizon
        n_obs_steps: Number of observation steps for conditioning
        n_action_steps: Number of action steps to execute per prediction
        num_inference_steps: Diffusion denoising steps
        use_ddim: Use DDIM for faster inference
        ddim_steps: DDIM steps if use_ddim is True
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        verbose: Print per-episode results
    """
    # Load policy
    policy = load_policy(
        checkpoint_dir=checkpoint_dir,
        action_dim=action_dim,
        obs_dim=obs_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        use_ddim=use_ddim,
        ddim_steps=ddim_steps,
    )

    eval_result = EvalResult()

    for ep in range(n_episodes):
        env = SyntheticPushTEnv(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_steps=max_episode_steps,
            seed=seed + ep,
        )

        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        total_inference_time = 0.0
        obs_history: List[np.ndarray] = []
        action_queue: List[np.ndarray] = []

        while not done:
            obs_history.append(obs)

            # Use queued actions if available
            if action_queue:
                action = action_queue.pop(0)
            else:
                # Build obs features from history
                if len(obs_history) >= n_obs_steps:
                    recent = obs_history[-n_obs_steps:]
                else:
                    # Pad with first observation
                    recent = [obs_history[0]] * (n_obs_steps - len(obs_history)) + obs_history

                # Concatenate observations into global conditioning
                obs_features = np.concatenate(recent, axis=-1)
                obs_features = mx.array(obs_features.reshape(1, -1), dtype=mx.float32)

                # Predict actions
                start_t = time.perf_counter()
                actions = policy.predict_action(obs_features)
                mx.eval(actions)
                inference_time = time.perf_counter() - start_t
                total_inference_time += inference_time

                # Queue up action steps
                actions_np = np.array(actions[0])  # (n_action_steps, action_dim)
                action = actions_np[0]
                for i in range(1, min(len(actions_np), n_action_steps)):
                    action_queue.append(actions_np[i])

            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

        ep_result = EpisodeResult(
            episode=ep,
            total_reward=total_reward,
            success=info.get("success", False),
            num_steps=step_count,
            inference_time_ms=total_inference_time * 1000,
        )
        eval_result.episodes.append(ep_result)

        if verbose:
            print(
                f"Episode {ep}: reward={total_reward:.3f}, "
                f"success={ep_result.success}, "
                f"steps={step_count}, "
                f"inference={total_inference_time * 1000:.1f}ms"
            )

    print(f"\n{'=' * 50}")
    print(eval_result.summary())
    print(f"{'=' * 50}")

    return eval_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on PushT environment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to converted checkpoint directory",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["real", "synthetic"],
        default="real",
        help="Environment mode: 'real' (PushT physics) or 'synthetic' (placeholder)",
    )
    parser.add_argument(
        "--use-ddim",
        action="store_true",
        help="Use DDIM scheduler for faster inference",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.env == "real":
        evaluate_real(
            checkpoint_dir=args.checkpoint,
            n_episodes=args.n_episodes,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        evaluate(
            checkpoint_dir=args.checkpoint,
            n_episodes=args.n_episodes,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            seed=args.seed,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
