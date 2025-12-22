"""
L6 Ablation Experiment Runner
Tests the contribution of Memory Gating vs Shift Detection in ChangingGoalGridWorld.
"""

import os
import sys
import csv
import argparse
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import ChangingGoalGridWorld
from agents.q_learning import QLearningAgent, ARCQLearningAgent, QLearningConfig
from experiments.run_l6 import ExperimentConfig, run_experiment, compute_final_metrics

def main():
    parser = argparse.ArgumentParser(description="L6 Ablation Study")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="outputs_L6_ablation")
    args = parser.parse_args()
    
    config = ExperimentConfig(n_episodes=args.episodes, n_seeds=args.seeds)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Environment: The non-stationary one where these mechanisms matter
    env_class = ChangingGoalGridWorld
    env_kwargs = {"change_every": 50}
    
    # Agents to compare
    agent_configs = [
        ("ql_baseline", QLearningAgent, {}),
        ("arc_full", ARCQLearningAgent, {"use_shift_detection": True, "use_mem_gating": True}),
        ("arc_no_shift", ARCQLearningAgent, {"use_shift_detection": False, "use_mem_gating": True}),
        ("arc_no_gating", ARCQLearningAgent, {"use_shift_detection": True, "use_mem_gating": False}),
    ]
    
    all_results = []
    
    print(f"Running L6 Ablation: {config.n_episodes} episodes x {config.n_seeds} seeds")
    print("=" * 60)
    
    for agent_name, agent_class, agent_kwargs in agent_configs:
        print(f"\nEvaluating {agent_name}...")
        for seed in range(config.n_seeds):
            results = run_experiment(
                agent_class, env_class, config, seed,
                agent_kwargs=agent_kwargs, env_kwargs=env_kwargs
            )
            # Override agent name in results for logging
            for r in results:
                r["agent"] = agent_name
            all_results.extend(results)
            
    # Compute and save final metrics
    final = compute_final_metrics(all_results, config)
    
    final_path = os.path.join(out_dir, "ablation_metrics.csv")
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        rows = [{"agent": a, "env": e, **m} for (a, e), m in final.items()]
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
            
    print("\n" + "=" * 60)
    print("ABLATION RESULTS (ChangingGoalGridWorld)")
    print("=" * 60)
    for (agent, env), m in sorted(final.items()):
        print(f"  {agent:15}: Success={m['final_success_rate']*100:5.1f}% | Reward={m['final_reward_mean']:7.2f}")
    
    print(f"\nResults saved to: {final_path}")

if __name__ == "__main__":
    main()
