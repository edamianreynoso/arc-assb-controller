"""
L6 Experiment Runner: Real RL Validation

Compares:
- QLearningAgent (baseline)
- ARCQLearningAgent (ARC-modulated)

On environments:
- GridWorld (basic)
- StochasticGridWorld (noisy transitions)
- ChangingGoalGridWorld (transfer learning)
"""

import os
import sys
import csv
import argparse
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.gridworld import GridWorld, StochasticGridWorld, ChangingGoalGridWorld, GridWorldConfig
from agents.q_learning import QLearningAgent, ARCQLearningAgent, QLearningConfig

@dataclass
class ExperimentConfig:
    n_episodes: int = 200
    n_seeds: int = 10
    eval_every: int = 10  # Evaluate policy every N episodes

def run_episode(agent, env, train: bool = True) -> Dict[str, Any]:
    """Run single episode, return metrics."""
    state, info = env.reset()
    if train and hasattr(agent, "on_reset") and agent.name == "ql_arc":
        agent.on_reset(info)
    total_reward = 0.0
    steps = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        if train:
            if hasattr(agent, 'update') and agent.name == "ql_arc":
                agent.update(state, action, reward, next_state, terminated or truncated, info)
            else:
                agent.update(state, action, reward, next_state, terminated or truncated)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if terminated or truncated:
            break
    
    if train:
        agent.decay_epsilon()
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "reached_goal": terminated and tuple(env.agent_pos) == env.config.goal_pos,
        "trap_hits": env.trap_hits,
    }

def evaluate_policy(agent, env, n_eval: int = 5) -> Dict[str, float]:
    """Evaluate current policy without exploration."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy evaluation
    
    rewards = []
    successes = []
    
    for _ in range(n_eval):
        result = run_episode(agent, env, train=False)
        rewards.append(result["total_reward"])
        successes.append(1.0 if result["reached_goal"] else 0.0)
    
    agent.epsilon = original_epsilon
    
    return {
        "eval_reward": np.mean(rewards),
        "eval_success_rate": np.mean(successes),
    }

def run_experiment(agent_class, env_class, config: ExperimentConfig, 
                   seed: int, agent_kwargs: Dict = None, env_kwargs: Dict = None) -> List[Dict]:
    """Run full training experiment, return episode-wise results."""
    np.random.seed(seed)
    
    env_kwargs = env_kwargs or {}
    agent_kwargs = agent_kwargs or {}
    
    env = env_class(**env_kwargs)
    
    # Configure agent for this environment
    q_config = QLearningConfig(n_states=env.n_states, n_actions=env.n_actions)
    agent = agent_class(config=q_config, **agent_kwargs)
    
    results = []
    
    for episode in range(config.n_episodes):
        agent.reset_episode_stats()
        ep_result = run_episode(agent, env, train=True)
        
        # Periodic evaluation
        if episode % config.eval_every == 0:
            eval_result = evaluate_policy(agent, env)
            ep_result.update(eval_result)
        
        # Add episode metadata
        ep_result["episode"] = episode
        ep_result["seed"] = seed
        ep_result["agent"] = agent.name
        ep_result["env"] = env_class.__name__
        ep_result["epsilon"] = agent.epsilon
        
        # ARC-specific metrics
        if hasattr(agent, 'arousal_history') and agent.arousal_history:
            ep_result["mean_arousal"] = np.mean(agent.arousal_history)
            ep_result["blocked_updates"] = agent.blocked_updates
        
        results.append(ep_result)
    
    return results


def summarize_seed_run(results: List[Dict[str, Any]], tail_frac: float = 0.2) -> float:
    """Mean episodic reward over the last `tail_frac` of episodes."""
    if not results:
        return 0.0
    n = len(results)
    start = int((1.0 - tail_frac) * n)
    tail = results[start:]
    return float(np.mean([r["total_reward"] for r in tail]))

def aggregate_results(all_results: List[Dict], config: ExperimentConfig) -> Dict[str, Any]:
    """Aggregate results across seeds for summary statistics."""
    # Group by agent, env, episode
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for r in all_results:
        key = (r["agent"], r["env"], r["episode"])
        grouped[key].append(r)
    
    summary = []
    for (agent, env, episode), group in grouped.items():
        rewards = [r["total_reward"] for r in group]
        successes = [1.0 if r["reached_goal"] else 0.0 for r in group]
        
        summary.append({
            "agent": agent,
            "env": env,
            "episode": episode,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "success_rate": np.mean(successes),
            "n_seeds": len(group),
        })
    
    return summary

def compute_final_metrics(all_results: List[Dict], config: ExperimentConfig) -> Dict[str, Dict]:
    """Compute final comparison metrics between agents."""
    from collections import defaultdict
    
    # Group by agent and env
    final_metrics = defaultdict(lambda: defaultdict(list))
    
    for r in all_results:
        agent = r["agent"]
        env = r["env"]
        
        # Use last 20% of episodes for final metrics
        if r["episode"] >= config.n_episodes * 0.8:
            final_metrics[(agent, env)]["reward"].append(r["total_reward"])
            final_metrics[(agent, env)]["success"].append(1.0 if r["reached_goal"] else 0.0)
            if "mean_arousal" in r:
                final_metrics[(agent, env)]["arousal"].append(r["mean_arousal"])
    
    results = {}
    for (agent, env), metrics in final_metrics.items():
        results[(agent, env)] = {
            "final_reward_mean": np.mean(metrics["reward"]),
            "final_reward_std": np.std(metrics["reward"]),
            "final_success_rate": np.mean(metrics["success"]),
            "mean_arousal": np.mean(metrics.get("arousal", [0])),
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="L6 RL Validation Experiments")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--outdir", type=str, default="outputs_L6", help="Output directory")
    args = parser.parse_args()
    
    config = ExperimentConfig(n_episodes=args.episodes, n_seeds=args.seeds)
    
    # Create output directory
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    # Experiment configurations
    experiments = [
        ("GridWorld", GridWorld, {}),
        ("StochasticGridWorld", StochasticGridWorld, {"slip_prob": 0.15}),
        ("ChangingGoalGridWorld", ChangingGoalGridWorld, {"change_every": 50}),
    ]
    
    agents = [
        ("baseline", QLearningAgent, {}),
        ("arc", ARCQLearningAgent, {}),
    ]
    
    print(f"Running L6 experiments: {config.n_episodes} episodes x {config.n_seeds} seeds")
    print("=" * 60)
    
    for env_name, env_class, env_kwargs in experiments:
        for agent_name, agent_class, agent_kwargs in agents:
            print(f"\n{env_name} + {agent_name}:")
            
            for seed in range(config.n_seeds):
                results = run_experiment(
                    agent_class, env_class, config, seed,
                    agent_kwargs=agent_kwargs, env_kwargs=env_kwargs
                )
                all_results.extend(results)
                tail_mean = summarize_seed_run(results, tail_frac=0.2)
                print(f"  Seed {seed}: tail_mean_reward={tail_mean:.2f}")
    
    # Collect all possible keys from all results
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    
    # Save raw results with consistent columns
    raw_path = os.path.join(out_dir, "raw_results.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            # Fill missing keys with None
            row = {k: r.get(k, None) for k in all_keys}
            w.writerow(row)
    print(f"\nWrote raw results: {raw_path}")
    
    # Save aggregated summary
    summary = aggregate_results(all_results, config)
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=summary[0].keys())
            w.writeheader()
            w.writerows(summary)
    print(f"Wrote summary: {summary_path}")
    
    # Print final comparison
    final = compute_final_metrics(all_results, config)
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (last 20% of episodes)")
    print("=" * 60)
    
    for (agent, env), metrics in sorted(final.items()):
        print(f"\n{env} + {agent}:")
        print(f"  Final Reward: {metrics['final_reward_mean']:.3f} +/- {metrics['final_reward_std']:.3f}")
        print(f"  Success Rate: {metrics['final_success_rate']*100:.1f}%")
        if metrics['mean_arousal'] > 0:
            print(f"  Mean Arousal: {metrics['mean_arousal']:.3f}")
    
    # Save final metrics
    final_path = os.path.join(out_dir, "final_metrics.csv")
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        rows = [{"agent": a, "env": e, **m} for (a, e), m in final.items()]
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote final metrics: {final_path}")

if __name__ == "__main__":
    main()
