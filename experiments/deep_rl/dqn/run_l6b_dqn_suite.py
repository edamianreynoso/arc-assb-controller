"""
L6b DQN ABLATION SUITE: Finding What Works for Deep RL

This implements a CLEAN 4-condition ablation to diagnose what's breaking DQN:

1. BASELINE: Pure DQN (no ARC)
2. LOSS-WEIGHT GATING: u_mem weights the loss instead of blocking updates
3. SHIFT→EXPLORATION: Shift detection boosts epsilon, not LR
4. MIXED REPLAY: 70% global + 30% recent, no gating

Each mechanism is tested IN ISOLATION to find what actually helps.
Then we combine only the winners.

Key Insight: In tabular Q, blocking Q(s,a) protects locally.
In DQN, blocking = less gradient on shared weights = learning stops globally.
Solution: Use soft weights, not hard blocks.
"""

import os
import sys
import csv
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

from envs.adversarial_envs import AdversarialCartPole, CatastrophicForgettingEnv
from agents.arc_dqn_wrapper import ARCGymWrapper, ARCWrapperConfig

# ==============================================================================
# MIXED REPLAY BUFFER (70% global + 30% recent)
# ==============================================================================

class MixedReplayBuffer(ReplayBuffer):
    """
    Replay buffer that samples from both global memory and recent experience.
    This prevents catastrophic forgetting while still learning from new data.
    """
    def __init__(self, *args, recent_ratio: float = 0.3, recent_window: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_ratio = recent_ratio
        self.recent_window = recent_window
        
    def sample(self, batch_size: int, env=None):
        # Number of samples from each source
        n_recent = int(batch_size * self.recent_ratio)
        n_global = batch_size - n_recent
        
        # Sample from recent window
        if self.pos < self.recent_window:
            recent_indices = np.random.randint(0, max(1, self.pos), size=n_recent)
        else:
            recent_start = self.pos - self.recent_window
            recent_indices = np.random.randint(recent_start, self.pos, size=n_recent)
        
        # Sample from global buffer
        if self.full:
            global_indices = np.random.randint(0, self.buffer_size, size=n_global)
        else:
            global_indices = np.random.randint(0, max(1, self.pos), size=n_global)
        
        # Combine indices
        all_indices = np.concatenate([global_indices, recent_indices])
        
        return self._get_samples(all_indices, env=env)

# ==============================================================================
# DQN WITH LOSS WEIGHTING (instead of blocking)
# ==============================================================================

class WeightedLossDQN(DQN):
    """
    DQN that uses u_mem as a loss weight instead of blocking updates.
    
    loss = mean(weight * huber(td_error))
    
    This preserves gradient flow while still "protecting" during high risk.
    """
    def __init__(self, *args, w_min: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_min = w_min
        self._current_u_mem = 1.0
        
    def set_u_mem(self, u_mem: float):
        """Set the current memory gating signal (called by callback)."""
        self._current_u_mem = np.clip(u_mem, self.w_min, 1.0)
        
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Override train to apply loss weighting."""
        # Get the weight (clamped to w_min)
        weight = self._current_u_mem
        
        # Original train with modified loss scale
        # We'll use the learning rate as a proxy for loss weighting
        # (since SB3 doesn't expose the loss directly)
        for param_group in self.policy.optimizer.param_groups:
            param_group['lr'] = self.learning_rate * weight
            
        super().train(gradient_steps, batch_size)
        
        # Restore original learning rate
        for param_group in self.policy.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

# ==============================================================================
# SHIFT→EXPLORATION CALLBACK
# ==============================================================================

class ShiftExplorationCallback(BaseCallback):
    """
    When a distribution shift is detected, boost exploration (epsilon) instead of LR.
    This is more stable for DQN than modulating learning rate.
    """
    def __init__(self, 
                 eval_env, 
                 eval_freq: int = 5000,
                 n_eval_episodes: int = 10,
                 shift_epsilon_boost: float = 0.3,
                 shift_window: int = 50,
                 td_threshold: float = 0.5,
                 verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.shift_epsilon_boost = shift_epsilon_boost
        self.shift_window = shift_window
        self.td_threshold = td_threshold
        
        # Tracking
        self.eval_rewards = []
        self.eval_steps = []
        self.td_history = deque(maxlen=100)
        self.shift_active = False
        self.shift_steps_remaining = 0
        self.original_exploration_rate = None
        
    def _on_training_start(self):
        self.original_exploration_rate = self.model.exploration_rate
        
    def _on_step(self) -> bool:
        # Get TD error from replay (if available)
        infos = self.locals.get("infos", [{}])
        
        # Detect shift via phase_changed or TD spike
        phase_changed = infos[0].get("phase_changed", False) or infos[0].get("arc_phase_changed", False)
        
        if phase_changed:
            self.shift_active = True
            self.shift_steps_remaining = self.shift_window
            
        # Apply exploration boost during shift
        if self.shift_active:
            self.model.exploration_rate = min(1.0, self.original_exploration_rate + self.shift_epsilon_boost)
            self.shift_steps_remaining -= 1
            if self.shift_steps_remaining <= 0:
                self.shift_active = False
                self.model.exploration_rate = self.original_exploration_rate
        
        # Evaluation
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            self.eval_rewards.append(mean_r)
            self.eval_steps.append(self.n_calls)
            if self.verbose:
                shift_str = "SHIFT" if self.shift_active else "     "
                print(f"  Step {self.n_calls}: reward={mean_r:.1f}±{std_r:.1f} [{shift_str}]")
                
        return True

# ==============================================================================
# LOSS WEIGHT CALLBACK (for WeightedLossDQN)
# ==============================================================================

class LossWeightCallback(BaseCallback):
    """
    Updates the DQN's loss weight based on u_mem from ARC wrapper.
    """
    def __init__(self, eval_env, eval_freq: int = 5000, n_eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_rewards = []
        self.eval_steps = []
        self.weight_history = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        u_mem = infos[0].get("arc_u_mem", 1.0)
        
        # Update the model's loss weight
        if hasattr(self.model, 'set_u_mem'):
            self.model.set_u_mem(u_mem)
            self.weight_history.append(u_mem)
        
        # Evaluation
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            self.eval_rewards.append(mean_r)
            self.eval_steps.append(self.n_calls)
            avg_w = np.mean(self.weight_history[-500:]) if self.weight_history else 1.0
            if self.verbose:
                print(f"  Step {self.n_calls}: reward={mean_r:.1f}±{std_r:.1f} | avg_weight={avg_w:.2f}")
                
        return True

# ==============================================================================
# SIMPLE CALLBACK (for baseline and mixed replay)
# ==============================================================================

class SimpleCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 5000, n_eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_rewards = []
        self.eval_steps = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes)
            self.eval_rewards.append(mean_r)
            self.eval_steps.append(self.n_calls)
            if self.verbose:
                print(f"  Step {self.n_calls}: reward={mean_r:.1f}±{std_r:.1f}")
        return True

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

@dataclass
class AblationConfig:
    total_timesteps: int = 50000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 1010)  # 5 seeds
    output_dir: str = "outputs_l6b_ablation"
    
def run_condition(condition: str, env_class, seed: int, config: AblationConfig) -> Dict[str, Any]:
    """Run a single experimental condition."""
    print(f"\n--- {condition} | seed={seed} ---")
    
    # Create environments
    base_train = env_class()
    base_eval = env_class()
    
    if condition == "baseline":
        # Pure DQN, no ARC
        train_env = base_train
        eval_env = base_eval
        model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                   learning_rate=1e-4, buffer_size=50000)
        callback = SimpleCallback(eval_env, config.eval_freq, config.n_eval_episodes, verbose=1)
        
    elif condition == "loss_weight_gating":
        # ARC wrapper + WeightedLossDQN
        arc_cfg = ARCWrapperConfig(
            use_observation_augmentation=False,
            use_reward_shaping=False,
            use_shift_detection=True,
            mem_gate_include_uncertainty=False,  # Only internal stress closes gate
            shift_mem_gate_floor=0.4,
        )
        train_env = ARCGymWrapper(base_train, config=arc_cfg)
        eval_env = ARCGymWrapper(base_eval, config=arc_cfg)
        model = WeightedLossDQN("MlpPolicy", train_env, seed=seed, verbose=0,
                               learning_rate=1e-4, buffer_size=50000, w_min=0.3)
        callback = LossWeightCallback(eval_env, config.eval_freq, config.n_eval_episodes, verbose=1)
        
    elif condition == "shift_exploration":
        # Shift detection boosts epsilon only
        arc_cfg = ARCWrapperConfig(
            use_observation_augmentation=False,
            use_reward_shaping=False,
            use_shift_detection=True,
        )
        train_env = ARCGymWrapper(base_train, config=arc_cfg)
        eval_env = ARCGymWrapper(base_eval, config=arc_cfg)
        model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                   learning_rate=1e-4, buffer_size=50000)
        callback = ShiftExplorationCallback(eval_env, config.eval_freq, config.n_eval_episodes,
                                           shift_epsilon_boost=0.3, shift_window=50, verbose=1)
        
    elif condition == "mixed_replay":
        # Mixed replay buffer (70% global + 30% recent)
        train_env = base_train
        eval_env = base_eval
        # Create custom buffer
        buffer = MixedReplayBuffer(
            buffer_size=50000,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            device="auto",
            recent_ratio=0.3,
            recent_window=1000,
        )
        model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                   learning_rate=1e-4, replay_buffer_class=None)
        model.replay_buffer = buffer
        callback = SimpleCallback(eval_env, config.eval_freq, config.n_eval_episodes, verbose=1)
        
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    # Train
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    
    # Collect results
    final_rewards = callback.eval_rewards[-3:] if len(callback.eval_rewards) >= 3 else callback.eval_rewards
    
    result = {
        "condition": condition,
        "seed": seed,
        "final_mean": np.mean(final_rewards),
        "final_std": np.std(final_rewards),
        "all_evals": callback.eval_rewards,
    }
    
    # Add condition-specific metrics
    if hasattr(callback, 'weight_history') and callback.weight_history:
        result["mean_weight"] = np.mean(callback.weight_history)
        result["min_weight"] = np.min(callback.weight_history)
        result["weight_below_05"] = np.mean([w < 0.5 for w in callback.weight_history])
        
    train_env.close()
    eval_env.close()
    
    return result

def run_ablation_suite():
    config = AblationConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    conditions = ["baseline", "loss_weight_gating", "shift_exploration", "mixed_replay"]
    env_classes = [
        (AdversarialCartPole, "AdversarialCartPole"),
        (CatastrophicForgettingEnv, "CatastrophicForgetting"),
    ]
    
    all_results = []
    
    for env_class, env_name in env_classes:
        print(f"\n{'='*60}")
        print(f"ENVIRONMENT: {env_name}")
        print(f"{'='*60}")
        
        for condition in conditions:
            for seed in config.seeds:
                try:
                    result = run_condition(condition, env_class, seed, config)
                    result["environment"] = env_name
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {condition}/{seed}: {e}")
                    continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.output_dir, f"ablation_results_{timestamp}.csv")
    
    fieldnames = ["environment", "condition", "seed", "final_mean", "final_std", 
                  "mean_weight", "min_weight", "weight_below_05"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
        
    print(f"\n📊 Results saved to: {csv_path}")
    
    # Statistical summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)
    
    for env_name in ["AdversarialCartPole", "CatastrophicForgetting"]:
        print(f"\n{env_name}:")
        for condition in conditions:
            cond_results = [r for r in all_results 
                          if r["environment"] == env_name and r["condition"] == condition]
            if cond_results:
                means = [r["final_mean"] for r in cond_results]
                avg = np.mean(means)
                std = np.std(means)
                print(f"  {condition:20}: {avg:7.2f} ± {std:5.2f}")
                
                # Show weight stats if available
                if "mean_weight" in cond_results[0]:
                    mw = np.mean([r.get("mean_weight", 1.0) for r in cond_results])
                    wb = np.mean([r.get("weight_below_05", 0.0) for r in cond_results])
                    print(f"    └─ mean_w={mw:.2f}, %w<0.5={wb*100:.1f}%")
    
    # Find winner per environment
    print("\n" + "="*70)
    print("WINNERS")
    print("="*70)
    for env_name in ["AdversarialCartPole", "CatastrophicForgetting"]:
        best_cond = None
        best_score = -float('inf')
        for condition in conditions:
            cond_results = [r for r in all_results 
                          if r["environment"] == env_name and r["condition"] == condition]
            if cond_results:
                avg = np.mean([r["final_mean"] for r in cond_results])
                if avg > best_score:
                    best_score = avg
                    best_cond = condition
        print(f"{env_name}: {best_cond} ({best_score:.1f})")
    
    return all_results

if __name__ == "__main__":
    results = run_ablation_suite()
