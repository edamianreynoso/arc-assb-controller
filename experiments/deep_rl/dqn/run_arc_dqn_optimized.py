"""
OPTIMIZED ARC-DQN EXPERIMENT: THE ULTIMATE TEST

This experiment tests the most sophisticated integration of ARC with Deep RL:
1. AFFECTIVE-CONDITIONED POLICY: The agent perceives its own arousal/risk.
2. MEMORY GATING: Learning rate is suppressed during high-risk/unstable states.

Environment: AdversarialCartPole (high noise, extreme shifts, action failures).
"""

import os
import sys
import csv
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from envs.adversarial_envs import AdversarialCartPole
from agents.arc_dqn_wrapper import ARCGymWrapper, ARCWrapperConfig

# ==============================================================================
# ARC OPTIMIZED CALLBACK
# ==============================================================================

class ARCOptimizedCallback(BaseCallback):
    """
    ARC Callback that implements:
    1. Metric tracking
    2. Step-wise Learning Rate Modulation (Memory Gating)
    """
    def __init__(self, eval_env, config, use_gating: bool = True, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.use_gating = use_gating
        
        # Metrics
        self.eval_rewards = []
        self.eval_steps = []
        self.gating_ratios = []
        
        # Internal state for LR modulation
        self.original_lr = None
        
    def _on_training_start(self):
        # Store original LR from the optimizer
        # In Adam, we look at the param_groups
        self.original_lr = self.model.learning_rate
        
    def _on_step(self):
        # 1. Get Memory Gating Signal from the last environment info
        # The info is in self.locals["infos"] (a list of dicts for each parallel env)
        infos = self.locals.get("infos", [{}])
        u_mem = infos[0].get("arc_u_mem", 1.0)  # 1.0 = normal, < 1.0 = gated
        
        # 2. Modulate Learning Rate
        if self.use_gating:
            # We bypass the standard scheduler and force the optimizer's LR
            # This mimics ARC's alpha modulation: alpha_eff = alpha * u_mem
            new_lr = self.original_lr * u_mem
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
                
            # Log gating
            if u_mem < 0.5:
                self.gating_ratios.append(1.0)
            else:
                self.gating_ratios.append(0.0)
        
        # 3. Periodic evaluation
        if self.n_calls % self.config["eval_freq"] == 0:
            mean_r, std_r = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.config["n_eval_episodes"]
            )
            self.eval_rewards.append(mean_r)
            self.eval_steps.append(self.n_calls)
            if self.verbose:
                print(f"  Step {self.n_calls}: reward={mean_r:.1f}±{std_r:.1f} | gate_active={np.mean(self.gating_ratios[-100:] if self.gating_ratios else [0]):.2f}")
                
        return True

# ==============================================================================
# RUNNER
# ==============================================================================

def run_optimized_experiment():
    # Shared Config
    config = {
        "total_timesteps": 100000,
        "eval_freq": 5000,
        "n_eval_episodes": 10,
        "seeds": [42, 123, 456],
        "output_dir": "outputs_arc_dqn_optimized"
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    conditions = ["baseline", "arc_conditioned", "arc_full_optimized"]
    all_results = []
    
    for condition in conditions:
        for seed in config["seeds"]:
            print(f"\n--- RUNNING: {condition} | seed={seed} ---")
            
            # Create Env
            # We use AdversarialCartPole because it has the most "noise" and "shocks"
            base_env = AdversarialCartPole()
            eval_base = AdversarialCartPole()
            
            if condition == "baseline":
                train_env = base_env
                eval_env = eval_base
                use_gating = False
            elif condition == "arc_conditioned":
                # Conditioned policy: perceive affective states
                arc_cfg = ARCWrapperConfig(use_observation_augmentation=True)
                train_env = ARCGymWrapper(base_env, config=arc_cfg)
                eval_env = ARCGymWrapper(eval_base, config=arc_cfg)
                use_gating = False
            elif condition == "arc_full_optimized":
                # Conditioned policy + LR/Memory Gating
                arc_cfg = ARCWrapperConfig(use_observation_augmentation=True)
                train_env = ARCGymWrapper(base_env, config=arc_cfg)
                eval_env = ARCGymWrapper(eval_base, config=arc_cfg)
                use_gating = True
            
            # DQN model
            model = DQN(
                "MlpPolicy", 
                train_env, 
                seed=seed, 
                verbose=0,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                target_update_interval=500,
                exploration_fraction=0.2
            )
            
            callback = ARCOptimizedCallback(eval_env, config, use_gating=use_gating, verbose=1)
            
            # Train
            model.learn(total_timesteps=config["total_timesteps"], callback=callback)
            
            # Log results
            final_score = np.mean(callback.eval_rewards[-3:])
            all_results.append({
                "condition": condition,
                "seed": seed,
                "final_mean": final_score,
                "all_evals": callback.eval_rewards
            })
            
            train_env.close()
            eval_env.close()
            
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config["output_dir"], f"final_scientific_results_{timestamp}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "final_mean"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({"condition": r["condition"], "seed": r["seed"], "final_mean": r["final_mean"]})
            
    print(f"\n✅ EXPERIMENT COMPLETE. Results saved to {path}")
    
    # Statistical Summary
    print("\nSUMMARY:")
    for cond in conditions:
        scores = [r["final_mean"] for r in all_results if r["condition"] == cond]
        print(f"  {cond:18}: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

if __name__ == "__main__":
    run_optimized_experiment()
