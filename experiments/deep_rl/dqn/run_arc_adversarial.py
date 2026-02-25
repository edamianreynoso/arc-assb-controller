"""
ARC-DQN ADVERSARIAL EXPERIMENT

Tests ARC's value in truly challenging conditions:
- AdversarialCartPole: extreme changes, noise, action failures
- CatastrophicForgettingEnv: complete dynamics changes
- HighStressEnv: shorter episodes with penalties
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from envs.adversarial_envs import AdversarialCartPole, CatastrophicForgettingEnv, HighStressEnv
from agents.arc_dqn_wrapper import ARCGymWrapper, ARCWrapperConfig

@dataclass
class AdversarialConfig:
    total_timesteps: int = 80000
    eval_freq: int = 4000
    n_eval_episodes: int = 10
    seeds: tuple = (42, 123, 456)  # 3 seeds for speed
    output_dir: str = "outputs_arc_adversarial"

class SimpleCallback(BaseCallback):
    def __init__(self, eval_env, config, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.config = config
        self.eval_rewards = []
        self.eval_steps = []
        
    def _on_step(self):
        if self.n_calls % self.config.eval_freq == 0:
            mean_r, std_r = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.config.n_eval_episodes
            )
            self.eval_rewards.append(mean_r)
            self.eval_steps.append(self.n_calls)
            if self.verbose:
                print(f"  Step {self.n_calls}: reward={mean_r:.1f}±{std_r:.1f}")
        return True

def run_adversarial_experiment(env_class, env_name: str, config: AdversarialConfig):
    print(f"\n{'='*60}")
    print(f"ENVIRONMENT: {env_name}")
    print(f"{'='*60}")
    
    results = []
    conditions = ["baseline", "arc_full", "arc_gating_only"]
    
    for condition in conditions:
        for seed in config.seeds:
            print(f"\n--- {condition} | seed={seed} ---")
            
            # Create environments
            base_env = env_class()
            eval_base = env_class()
            
            if condition == "baseline":
                train_env = base_env
                eval_env = eval_base
            elif condition == "arc_full":
                arc_cfg = ARCWrapperConfig(
                    use_reward_shaping=True,
                    stability_bonus=0.2,
                    instability_penalty=-0.1,
                )
                train_env = ARCGymWrapper(base_env, config=arc_cfg)
                eval_env = ARCGymWrapper(eval_base, config=arc_cfg)
            else:  # arc_gating_only
                arc_cfg = ARCWrapperConfig(
                    use_reward_shaping=False,
                    arc_k_mem_block=0.9,
                )
                train_env = ARCGymWrapper(base_env, config=arc_cfg)
                eval_env = ARCGymWrapper(eval_base, config=arc_cfg)
            
            # Train
            model = DQN("MlpPolicy", train_env, seed=seed, verbose=0,
                       learning_rate=1e-4, buffer_size=30000)
            callback = SimpleCallback(eval_env, config)
            model.learn(total_timesteps=config.total_timesteps, callback=callback)
            
            # Final evaluation
            final_rewards = callback.eval_rewards[-3:] if len(callback.eval_rewards) >= 3 else callback.eval_rewards
            
            results.append({
                "environment": env_name,
                "condition": condition,
                "seed": seed,
                "final_mean": np.mean(final_rewards),
                "final_std": np.std(final_rewards),
                "all_evals": callback.eval_rewards,
            })
            
            train_env.close()
            eval_env.close()
    
    return results

def main():
    config = AdversarialConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    
    all_results = []
    
    # Test each environment
    envs = [
        (AdversarialCartPole, "AdversarialCartPole"),
        (CatastrophicForgettingEnv, "CatastrophicForgetting"),
        (HighStressEnv, "HighStress"),
    ]
    
    for env_class, env_name in envs:
        results = run_adversarial_experiment(env_class, env_name, config)
        all_results.extend(results)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL ENVIRONMENTS")
    print("="*70)
    
    for env_name in ["AdversarialCartPole", "CatastrophicForgetting", "HighStress"]:
        print(f"\n{env_name}:")
        for condition in ["baseline", "arc_full", "arc_gating_only"]:
            cond_results = [r for r in all_results 
                          if r["environment"] == env_name and r["condition"] == condition]
            if cond_results:
                means = [r["final_mean"] for r in cond_results]
                print(f"  {condition}: {np.mean(means):.2f} ± {np.std(means):.2f}")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.output_dir, f"adversarial_results_{timestamp}.csv")
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["environment", "condition", "seed", "final_mean", "final_std"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k != "all_evals"})
    
    print(f"\n📊 Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
