"""
SCIENTIFIC ARC-DQN VALIDATION EXPERIMENT

Goal: Determine if ARC genuinely helps DQN in non-stationary environments.

Design Principles:
1. FAIR: Same environment, hyperparameters for all conditions
2. RIGOROUS: Multiple seeds, statistical tests
3. UNBIASED: Report all results, even if ARC doesn't help
4. REPRODUCIBLE: Fixed seeds, logged parameters

Hypothesis:
H0: ARC does not improve DQN performance in non-stationary environments
H1: ARC improves DQN's ability to maintain performance across distribution shifts
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# Add parent directory
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from envs.cartpole_nonstationary import NonStationaryCartPole
from agents.arc_dqn_wrapper import ARCGymWrapper, ARCWrapperConfig

# ==============================================================================
# EXPERIMENTAL CONFIGURATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    """All experimental parameters - same for all conditions."""
    # Training
    total_timesteps: int = 100000  # Longer training
    eval_freq: int = 2000
    n_eval_episodes: int = 10
    
    # Environment
    change_every: int = 25  # More frequent changes
    pole_lengths: tuple = (0.3, 0.5, 1.0, 1.5)  # More variety
    
    # DQN hyperparameters (SAME for all conditions)
    learning_rate: float = 1e-4
    buffer_size: int = 50000
    learning_starts: int = 1000
    batch_size: int = 32
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    
    # Statistical
    seeds: tuple = (42, 123, 456, 789, 1010)  # 5 seeds
    
    # Output
    output_dir: str = "outputs_arc_dqn_scientific"

# ==============================================================================
# METRICS CALLBACK
# ==============================================================================

class ScientificMetricsCallback(BaseCallback):
    """Track detailed metrics for scientific analysis."""
    
    def __init__(self, eval_env, config: ExperimentConfig, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        
        # Metrics storage
        self.eval_rewards = []
        self.eval_steps = []
        self.phase_at_eval = []
        self.episode_count = 0
        
        # Per-episode tracking
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Phase change tracking
        self.rewards_by_phase = {i: [] for i in range(len(config.pole_lengths))}
        self.current_phase = 0
        
    def _on_step(self):
        # Track episode rewards
        self.current_ep_reward += self.locals.get("rewards", [0])[0]
        self.current_ep_length += 1
        
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.rewards_by_phase[self.current_phase].append(self.current_ep_reward)
            
            self.current_ep_reward = 0
            self.current_ep_length = 0
            self.episode_count += 1
            
            # Update phase
            self.current_phase = (self.episode_count // self.config.change_every) % len(self.config.pole_lengths)
        
        # Periodic evaluation
        if self.n_calls % self.config.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, 
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True
            )
            self.eval_rewards.append(mean_reward)
            self.eval_steps.append(self.n_calls)
            self.phase_at_eval.append(self.current_phase)
            
            if self.verbose:
                print(f"  Step {self.n_calls}: reward={mean_reward:.1f}±{std_reward:.1f}, phase={self.current_phase}")
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        # Overall
        final_rewards = self.eval_rewards[-5:] if len(self.eval_rewards) >= 5 else self.eval_rewards
        
        # Per-phase analysis
        phase_means = {}
        phase_stds = {}
        for phase, rewards in self.rewards_by_phase.items():
            if rewards:
                phase_means[f"phase_{phase}_mean"] = np.mean(rewards)
                phase_stds[f"phase_{phase}_std"] = np.std(rewards)
        
        # Recovery: How fast does performance recover after phase change?
        # (measured as variance in first 5 episodes after each phase)
        
        return {
            "final_mean_reward": np.mean(final_rewards),
            "final_std_reward": np.std(final_rewards),
            "overall_mean_reward": np.mean(self.eval_rewards) if self.eval_rewards else 0,
            "total_episodes": len(self.episode_rewards),
            **phase_means,
            **phase_stds,
        }

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_single_experiment(
    condition: str,
    seed: int,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Run a single experiment with given condition and seed."""
    
    print(f"\n{'='*60}")
    print(f"Condition: {condition}, Seed: {seed}")
    print(f"{'='*60}")
    
    # Create environment
    base_env = NonStationaryCartPole(
        change_every=config.change_every,
        pole_lengths=config.pole_lengths,
    )
    
    # Apply ARC wrapper based on condition
    if condition == "baseline":
        train_env = base_env
        eval_env = NonStationaryCartPole(
            change_every=config.change_every,
            pole_lengths=config.pole_lengths,
        )
    elif condition == "arc_full":
        arc_config = ARCWrapperConfig(
            use_reward_shaping=True,
            use_observation_augmentation=False,
            stability_bonus=0.1,
        )
        train_env = ARCGymWrapper(base_env, config=arc_config)
        eval_env = ARCGymWrapper(
            NonStationaryCartPole(
                change_every=config.change_every,
                pole_lengths=config.pole_lengths,
            ),
            config=arc_config
        )
    elif condition == "arc_gating_only":
        arc_config = ARCWrapperConfig(
            use_reward_shaping=False,
            use_observation_augmentation=False,
            arc_k_mem_block=0.8,  # Strong memory gating
        )
        train_env = ARCGymWrapper(base_env, config=arc_config)
        eval_env = ARCGymWrapper(
            NonStationaryCartPole(
                change_every=config.change_every,
                pole_lengths=config.pole_lengths,
            ),
            config=arc_config
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    # Create DQN with IDENTICAL hyperparameters
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        seed=seed,
        verbose=0,
    )
    
    # Callback for metrics
    callback = ScientificMetricsCallback(eval_env, config, verbose=1)
    
    # Train
    print(f"Training for {config.total_timesteps} timesteps...")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    
    # Get results
    summary = callback.get_summary()
    summary["condition"] = condition
    summary["seed"] = seed
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return summary

def run_full_experiment(config: ExperimentConfig):
    """Run complete experiment across all conditions and seeds."""
    
    print("="*70)
    print("SCIENTIFIC ARC-DQN VALIDATION")
    print("="*70)
    print(f"Timesteps: {config.total_timesteps}")
    print(f"Phase change every: {config.change_every} episodes")
    print(f"Pole lengths: {config.pole_lengths}")
    print(f"Seeds: {config.seeds}")
    print(f"Conditions: baseline, arc_full, arc_gating_only")
    print("="*70)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run experiments
    results = []
    conditions = ["baseline", "arc_full", "arc_gating_only"]
    
    for condition in conditions:
        for seed in config.seeds:
            try:
                result = run_single_experiment(condition, seed, config)
                results.append(result)
            except Exception as e:
                print(f"ERROR in {condition}/seed={seed}: {e}")
                continue
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.output_dir, f"results_{timestamp}.csv")
    
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    for condition in conditions:
        cond_results = [r for r in results if r["condition"] == condition]
        if cond_results:
            rewards = [r["final_mean_reward"] for r in cond_results]
            print(f"\n{condition}:")
            print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Min/Max: {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    
    # Statistical test (if we have enough data)
    baseline_rewards = [r["final_mean_reward"] for r in results if r["condition"] == "baseline"]
    arc_rewards = [r["final_mean_reward"] for r in results if r["condition"] == "arc_full"]
    
    if len(baseline_rewards) >= 3 and len(arc_rewards) >= 3:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(arc_rewards, baseline_rewards)
        print(f"\nT-test (ARC vs Baseline):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at α=0.05: {'YES' if p_value < 0.05 else 'NO'}")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    config = ExperimentConfig()
    results = run_full_experiment(config)
