"""
L6b Experiment Runner: Deep RL Validation with DQN.

Compares:
- DQN Baseline (no ARC)
- DQN + ARC Observation Augmentation (agent perceives arousal/risk)
- DQN + ARC Plasticity + Replay Gating (memory gate affects learning)

On environment:
- NonStationaryCartPole (pole length changes every N episodes)

This validates ARC's transfer learning benefits beyond tabular Q-learning.
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch as th
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stable-baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# Local imports
from envs.cartpole_nonstationary import NonStationaryCartPole
from agents.arc_dqn_wrapper import ARCGymWrapper, ARCWrapperConfig
from agents.arc_replay_buffer import ARCGatedReplayBuffer, ARCGatedReplayConfig


def _make_arc_lr_schedule(
    base_lr: float,
    gate: dict,
    lr_min_factor: float = 0.10,
    shift_lr_boost: float = 0.50,
):
    """
    Learning-rate schedule that reads ARC gate values updated by a callback.

    - In stable periods: lr = base_lr * clamp(u_mem, lr_min_factor..1)
    - In shift mode: lr is boosted to accelerate adaptation (while still allowing u_mem to protect)
    """

    def schedule(_progress_remaining: float) -> float:
        u_mem = float(gate.get("u_mem", 1.0))
        shift_active = bool(gate.get("shift_active", False))

        lr_factor = max(lr_min_factor, min(1.0, u_mem))
        if shift_active:
            lr_factor = lr_factor * (1.0 + shift_lr_boost)

        return float(base_lr * lr_factor)

    return schedule


class _ARCExplorationSchedule:
    """Wrap SB3's exploration schedule to increase epsilon during detected shifts."""

    def __init__(self, base_schedule, gate: dict, shift_eps_floor: float = 0.20):
        self._base_schedule = base_schedule
        self._gate = gate
        self._shift_eps_floor = float(shift_eps_floor)

    def __call__(self, progress_remaining: float) -> float:
        eps = float(self._base_schedule(progress_remaining))
        if bool(self._gate.get("shift_active", False)):
            eps = max(eps, self._shift_eps_floor)
        return float(eps)


class MetricsCallback(BaseCallback):
    """Callback to track training metrics."""
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 1000,
        verbose: int = 0,
        arc_gate: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.arc_gate = arc_gate
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_steps: List[int] = []
        self.arc_metrics: List[Dict[str, float]] = []
        
        # Episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Update ARC gate signals (used by custom schedules) from infos
        if self.arc_gate is not None:
            infos = self.locals.get("infos", [{}])
            info0 = infos[0] if infos else {}
            self.arc_gate["u_mem"] = float(info0.get("arc_u_mem", 1.0))
            self.arc_gate["shift_active"] = bool(info0.get("arc_shift_active", False))

            # Feed RL-derived "prediction error" back into ARC (1-step delayed):
            # use instantaneous TD-error magnitude as a bounded PE proxy.
            try:
                env0 = self.training_env.envs[0]  # type: ignore[attr-defined]
            except Exception:
                env0 = None

            if env0 is not None and hasattr(env0, "set_external_signals"):
                new_obs = self.locals.get("new_obs", None)
                actions = self.locals.get("actions", None)
                rewards = self.locals.get("rewards", None)
                dones = self.locals.get("dones", None)

                if new_obs is not None and actions is not None and rewards is not None and dones is not None:
                    # Previous obs is still in the model at this point (before _store_transition updates it)
                    prev_obs = getattr(self.model, "_last_obs", None)
                    if prev_obs is not None and hasattr(self.model, "q_net") and hasattr(self.model, "q_net_target"):
                        device = getattr(self.model, "device", "cpu")

                        obs_t = th.as_tensor(prev_obs, device=device).float()
                        next_obs_t = th.as_tensor(new_obs, device=device).float()

                        act = np.array(actions).reshape(-1, 1)
                        rew = np.array(rewards).reshape(-1, 1)
                        done = np.array(dones).reshape(-1, 1).astype(np.float32)

                        act_t = th.as_tensor(act, device=device).long()
                        rew_t = th.as_tensor(rew, device=device).float()
                        done_t = th.as_tensor(done, device=device).float()

                        with th.no_grad():
                            q = self.model.q_net(obs_t)
                            q_sa = th.gather(q, dim=1, index=act_t)
                            next_q = self.model.q_net_target(next_obs_t).max(dim=1, keepdim=True).values
                            target = rew_t + (1.0 - done_t) * float(self.model.gamma) * next_q
                            td_err = (target - q_sa).abs().mean().item()

                        # Normalize to [0, 1] with a smooth squash (CartPole Q-values are O(1e2))
                        pe = float(np.tanh(td_err / 10.0))
                        u_exog = pe  # treat surprise as uncertainty proxy
                        env0.set_external_signals(pe=pe, u_exog=u_exog)

        # Track episode reward
        self.current_episode_reward += self.locals.get("rewards", [0])[0]
        self.current_episode_length += 1
        
        # Check if episode ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Get ARC metrics if available
            if hasattr(self.training_env.envs[0], 'get_arc_metrics'):
                metrics = self.training_env.envs[0].get_arc_metrics()
                self.arc_metrics.append(metrics)
            
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            self.eval_rewards.append(mean_reward)
            self.eval_steps.append(self.n_calls)
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: eval_reward={mean_reward:.1f} +/- {std_reward:.1f}")
        
        return True


def run_dqn_experiment(
    condition: str,
    env_config: Dict[str, Any],
    seed: int,
    total_timesteps: int = 50000,
    eval_freq: int = 5000,
) -> Dict[str, Any]:
    """
    Run a single DQN experiment.
    
    Args:
        condition: "baseline", "arc", or "arc_gating_only"
        env_config: Configuration for NonStationaryCartPole
        seed: Random seed
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        
    Returns:
        Dictionary with experiment results
    """
    np.random.seed(seed)
    
    # Create environments
    train_env = NonStationaryCartPole(**env_config)
    eval_env = NonStationaryCartPole(**env_config)

    # ARC gate values updated online by callback
    arc_gate: Optional[dict] = None
    
    # Apply ARC wrapper based on condition
    if condition == "arc_obs":
        # Conditioned policy: perceive ARC internal state (better for function approximation)
        arc_config = ARCWrapperConfig(
            use_reward_shaping=False,  # Reward shaping hurts DQN training
            use_observation_augmentation=True,  # Add ARC state to observations
            use_shift_detection=True,
            shift_boost_steps=200,  # one CartPole episode
        )
        train_env = ARCGymWrapper(train_env, config=arc_config)
        eval_env = ARCGymWrapper(eval_env, config=arc_config)

    elif condition == "arc_plasticity":
        # Keep the policy input identical to baseline; connect ARC to learning instead.
        arc_config = ARCWrapperConfig(
            use_reward_shaping=False,
            use_observation_augmentation=False,
            use_shift_detection=True,
            shift_boost_steps=200,
            mem_gate_include_uncertainty=True,
        )
        train_env = ARCGymWrapper(train_env, config=arc_config)
        eval_env = ARCGymWrapper(eval_env, config=arc_config)
        arc_gate = {"u_mem": 1.0, "shift_active": False}

    # Create callback
    callback = MetricsCallback(eval_env, eval_freq=eval_freq, verbose=1, arc_gate=arc_gate)
    
    # Create DQN agent
    base_lr = 1e-3
    learning_rate = base_lr
    replay_buffer_class = None
    replay_buffer_kwargs = None

    # In "arc_plasticity": connect ARC signals to DQN learning via schedules and replay gating
    if condition == "arc_plasticity" and arc_gate is not None:
        learning_rate = _make_arc_lr_schedule(base_lr=base_lr, gate=arc_gate)
        replay_buffer_class = ARCGatedReplayBuffer
        replay_buffer_kwargs = {
            "arc_config": ARCGatedReplayConfig(
                enable=True,
                u_mem_threshold=0.2,
                min_transitions_to_gate=1000,
                bypass_when_shift_active=True,
            )
        }

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=seed,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    # Patch exploration schedule to boost epsilon during ARC shift mode
    if condition == "arc_plasticity" and arc_gate is not None:
        model.exploration_schedule = _ARCExplorationSchedule(
            base_schedule=model.exploration_schedule,
            gate=arc_gate,
            shift_eps_floor=0.25,
        )
    
    # Train
    print(f"\nTraining {condition} (seed={seed})...")
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Final evaluation
    final_reward, final_std = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    
    # Compute success rate (CartPole success = surviving 195+ steps)
    successes = []
    for _ in range(20):
        obs, info = eval_env.reset()
        ep_len = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_len += 1
            if terminated or truncated:
                break
        successes.append(1.0 if ep_len >= 195 else 0.0)
    success_rate = np.mean(successes)
    
    # Collect results
    results = {
        "condition": condition,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "train_time_sec": train_time,
        "final_reward_mean": final_reward,
        "final_reward_std": final_std,
        "success_rate": success_rate,
        "n_episodes": len(callback.episode_rewards),
        "mean_episode_reward": np.mean(callback.episode_rewards) if callback.episode_rewards else 0,
        "mean_episode_length": np.mean(callback.episode_lengths) if callback.episode_lengths else 0,
    }
    
    # Add ARC-specific metrics if available
    if callback.arc_metrics:
        results["mean_arousal"] = np.mean([m.get("mean_arousal", 0) for m in callback.arc_metrics])
        results["mean_risk"] = np.mean([m.get("mean_risk", 0) for m in callback.arc_metrics])

    # Replay gating stats (only for arc_plasticity)
    if hasattr(model, "replay_buffer") and hasattr(model.replay_buffer, "get_gate_stats"):
        stats = model.replay_buffer.get_gate_stats()
        for k, v in stats.items():
            results[f"replay_{k}"] = v
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="L6b DQN Validation Experiments")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--change-every", type=int, default=50, help="Episodes between pole length changes")
    parser.add_argument("--outdir", type=str, default="outputs_L6_dqn", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    env_config = {
        "change_every": args.change_every,
        "pole_lengths": (0.5, 1.0, 1.5),  # Short, medium, long poles
    }
    
    conditions = ["baseline", "arc_obs", "arc_plasticity"]
    all_results = []
    
    print("=" * 70)
    print("L6b Deep RL Validation: DQN on Non-Stationary CartPole")
    print("=" * 70)
    print(f"Timesteps: {args.timesteps}")
    print(f"Seeds: {args.seeds}")
    print(f"Pole length changes every: {args.change_every} episodes")
    print(f"Conditions: {conditions}")
    print("=" * 70)
    
    for condition in conditions:
        for seed in range(args.seeds):
            results = run_dqn_experiment(
                condition=condition,
                env_config=env_config,
                seed=seed,
                total_timesteps=args.timesteps,
            )
            all_results.append(results)
            
            print(f"  {condition} seed={seed}: reward={results['final_reward_mean']:.1f}, "
                  f"success={results['success_rate']*100:.1f}%")
    
    # Save results
    results_path = os.path.join(out_dir, "dqn_results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            fieldnames = sorted({k for r in all_results for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
    print(f"\nSaved: {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for condition in conditions:
        cond_results = [r for r in all_results if r["condition"] == condition]
        rewards = [r["final_reward_mean"] for r in cond_results]
        successes = [r["success_rate"] for r in cond_results]
        
        print(f"\n{condition.upper()}:")
        print(f"  Final Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"  Success Rate: {np.mean(successes)*100:.1f}% +/- {np.std(successes)*100:.1f}%")
    
    # Compute improvement
    baseline_success = np.mean([r["success_rate"] for r in all_results if r["condition"] == "baseline"])
    arc_obs_success = np.mean([r["success_rate"] for r in all_results if r["condition"] == "arc_obs"])
    arc_plasticity_success = np.mean([r["success_rate"] for r in all_results if r["condition"] == "arc_plasticity"])
    
    print("\n" + "-" * 70)
    print("IMPROVEMENT OVER BASELINE:")
    print(f"  ARC_OBS:         {(arc_obs_success - baseline_success)*100:+.1f} pp")
    print(f"  ARC_PLASTICITY:  {(arc_plasticity_success - baseline_success)*100:+.1f} pp")
    
    # Save summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("L6b DQN Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for condition in conditions:
            cond_results = [r for r in all_results if r["condition"] == condition]
            rewards = [r["final_reward_mean"] for r in cond_results]
            successes = [r["success_rate"] for r in cond_results]
            f.write(f"{condition}:\n")
            f.write(f"  Final Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}\n")
            f.write(f"  Success Rate: {np.mean(successes)*100:.1f}%\n\n")
        f.write("\nImprovement over baseline:\n")
        f.write(f"  ARC_OBS: {(arc_obs_success - baseline_success)*100:+.1f} pp\n")
        f.write(f"  ARC_PLASTICITY: {(arc_plasticity_success - baseline_success)*100:+.1f} pp\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
