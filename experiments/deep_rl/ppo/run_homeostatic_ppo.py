"""
Homeostasis-First PPO: The Philosophical Shift.

CORE INSIGHT:
In biology, organisms don't optimize for "maximum performance".
They optimize for HOMEOSTASIS (maintaining internal stability).
Performance is a MEANS to homeostasis, not the goal itself.

PREVIOUS APPROACH (failed):
    R = EnvReward - λ * Arousal²  ("Win, and try to be calm")

NEW APPROACH:
    R = -Arousal + λ * EnvReward  ("Be calm first, win second")

This is a FUNDAMENTAL shift in what the agent cares about.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
from envs.cartpole_nonstationary import NonStationaryCartPole
from sim.state import State
from sim.dynamics import step_dynamics
import yaml


def load_arc_config():
    config_path = REPO_ROOT / "configs" / "v2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SimplePolicy(nn.Module):
    """Standard PPO policy network."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        
    def forward(self, obs):
        features = self.net(obs)
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value


class HomeostaticPPO:
    """
    PPO where the reward prioritizes internal stability over external success.
    
    Reward = -Arousal + λ * NormalizedEnvReward
    
    This makes the agent WANT to be calm, with success as a bonus.
    """
    def __init__(
        self,
        env: gym.Env,
        homeostasis_weight: float = 1.0,  # Weight on -arousal
        success_weight: float = 0.1,       # Weight on env reward (small!)
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        seed: int = 0,
        use_homeostatic: bool = True,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.env = env
        self.homeostasis_weight = homeostasis_weight
        self.success_weight = success_weight
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_homeostatic = use_homeostatic
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.policy = SimplePolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # ARC state
        self.arc_cfg = load_arc_config()
        self.assb_state = self._init_assb_state()
        
        # Buffers
        self.obs_buffer = []
        self.act_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []
        
        # Metrics
        self.episode_rewards = []
        self.episode_arousals = []
        
    def _init_assb_state(self) -> State:
        cfg = self.arc_cfg
        return State(
            phi=cfg.get("phi0", 0.75), g=cfg.get("g0", 0.75),
            p=cfg.get("p0", 0.75), i=cfg.get("i0", 0.70),
            s=cfg.get("s0", 0.30), v=cfg.get("v0", 0.55),
            a=cfg.get("a0", 0.30), mf=cfg.get("mf0", 0.25),
            ms=cfg.get("ms0", 0.20), u=cfg.get("u_base", 0.20)
        )
    
    def _compute_homeostatic_reward(self, env_reward: float, terminated: bool) -> Tuple[float, float]:
        """
        THE KEY MECHANISM: Homeostasis-first reward.
        
        Returns:
            shaped_reward: The reward the agent actually sees
            arousal: Current arousal level (for metrics)
        """
        cfg = self.arc_cfg
        
        # Compute uncertainty from environment signals
        # Negative reward = something went wrong = high uncertainty
        pe = max(0, -env_reward / 10)  # Normalized prediction error
        u_exog = 0.7 if terminated and env_reward < 0 else 0.1
        
        # Update ASSB dynamics (simplified)
        a_excess = max(0.0, self.assb_state.a - cfg["a_safe"])
        risk = cfg["arc_w_u"] * max(self.assb_state.u, u_exog) + cfg["arc_w_a"] * a_excess
        risk = min(1.0, risk)
        
        control = {
            "u_dmg": min(1.0, cfg["arc_k_dmg"] * risk),
            "u_att": min(1.0, cfg["arc_k_att"] * self.assb_state.u),
            "u_mem": 1.0,
            "u_calm": min(1.0, cfg["arc_k_calm"] * a_excess),
            "u_reapp": 0.0,
        }
        
        self.assb_state = step_dynamics(
            self.assb_state, pe=pe, reward=env_reward/100,
            u_exog=u_exog, control=control, cfg=cfg
        )
        
        arousal = self.assb_state.a
        
        # THE HOMEOSTATIC REWARD
        # Primary: minimize arousal (be calm)
        # Secondary: get environment reward (survive/win)
        homeostatic_component = -arousal * self.homeostasis_weight
        success_component = (env_reward / 500) * self.success_weight  # Normalized to [0,1]
        
        shaped_reward = homeostatic_component + success_component
        
        return shaped_reward, arousal
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.policy(obs_t)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self):
        obs = torch.FloatTensor(np.array(self.obs_buffer))
        actions = torch.LongTensor(self.act_buffer)
        old_logprobs = torch.FloatTensor(self.logprob_buffer)
        advantages = torch.FloatTensor(self.advantage_buffer)
        returns = torch.FloatTensor(self.return_buffer)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                probs, values = self.policy(batch_obs)
                dist = Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, total_timesteps: int) -> Dict[str, Any]:
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_arousal_sum = 0
        episode_steps = 0
        step = 0
        
        while step < total_timesteps:
            self.obs_buffer = []
            self.act_buffer = []
            self.reward_buffer = []
            self.value_buffer = []
            self.logprob_buffer = []
            self.done_buffer = []
            
            for _ in range(self.n_steps):
                action, log_prob, value = self.select_action(obs)
                next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Compute shaped reward
                if self.use_homeostatic:
                    shaped_reward, arousal = self._compute_homeostatic_reward(env_reward, terminated)
                    episode_arousal_sum += arousal
                else:
                    shaped_reward = env_reward
                    arousal = 0
                
                self.obs_buffer.append(obs)
                self.act_buffer.append(action)
                self.reward_buffer.append(shaped_reward)
                self.value_buffer.append(value)
                self.logprob_buffer.append(log_prob)
                self.done_buffer.append(float(done))
                
                episode_reward += env_reward  # Track TRUE env reward
                episode_steps += 1
                step += 1
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    if self.use_homeostatic:
                        self.episode_arousals.append(episode_arousal_sum / max(1, episode_steps))
                    episode_reward = 0
                    episode_arousal_sum = 0
                    episode_steps = 0
                    obs, _ = self.env.reset()
                    self.assb_state = self._init_assb_state()
                else:
                    obs = next_obs
                
                if step >= total_timesteps:
                    break
            
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                _, next_value = self.policy(obs_t)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(
                self.reward_buffer, self.value_buffer, self.done_buffer, next_value
            )
            self.advantage_buffer = advantages
            self.return_buffer = returns
            
            self.update()
        
        return {
            "episode_rewards": self.episode_rewards,
            "mean_reward": np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            "mean_arousal": np.mean(self.episode_arousals) if self.episode_arousals else 0,
        }
    
    def evaluate(self, n_episodes: int = 20) -> Tuple[float, float, float]:
        """Evaluate on TRUE environment reward."""
        rewards = []
        lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0
            ep_len = 0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs, _ = self.policy(obs_t)
                    action = probs.argmax(1).item()
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated
            
            rewards.append(ep_reward)
            lengths.append(ep_len)
        
        success_rate = sum(1 for l in lengths if l >= 195) / len(lengths)
        return np.mean(rewards), np.std(rewards), success_rate


def run_experiment(condition: str, seed: int, timesteps: int = 100000) -> Dict[str, Any]:
    np.random.seed(seed)
    
    # Extreme environment
    env = NonStationaryCartPole(
        change_every=15,
        pole_lengths=(0.3, 0.5, 1.0, 2.0, 3.0)
    )
    
    use_homeostatic = (condition == "homeostatic")
    
    agent = HomeostaticPPO(
        env=env,
        homeostasis_weight=1.0,
        success_weight=0.1,  # Small!
        seed=seed,
        use_homeostatic=use_homeostatic,
    )
    
    print(f"Training {condition} (seed={seed})...")
    start = datetime.now()
    results = agent.train(timesteps)
    duration = (datetime.now() - start).total_seconds()
    
    mean_reward, std_reward, success_rate = agent.evaluate(n_episodes=20)
    
    env.close()
    
    return {
        "condition": condition,
        "seed": seed,
        "train_time": duration,
        "final_reward": mean_reward,
        "final_std": std_reward,
        "success_rate": success_rate,
        "mean_arousal": results.get("mean_arousal", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()
    
    conditions = ["baseline", "homeostatic"]
    all_results = []
    
    print("=" * 60)
    print("HOMEOSTASIS-FIRST PPO: R = -Arousal + λ*EnvReward")
    print("=" * 60)
    
    for cond in conditions:
        print(f"\n--- {cond.upper()} ---")
        for seed in range(args.seeds):
            res = run_experiment(cond, seed, args.timesteps)
            all_results.append(res)
            print(f"  Seed {seed}: Reward={res['final_reward']:.1f}, "
                  f"Success={res['success_rate']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        rewards = [r["final_reward"] for r in cond_results]
        successes = [r["success_rate"] for r in cond_results]
        
        print(f"\n{cond.upper()}:")
        print(f"  Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"  Success Rate: {np.mean(successes)*100:.1f}%")
        if cond == "homeostatic":
            arousals = [r["mean_arousal"] for r in cond_results]
            print(f"  Mean Arousal: {np.mean(arousals):.3f}")
    
    baseline_success = np.mean([r["success_rate"] for r in all_results if r["condition"] == "baseline"])
    homeo_success = np.mean([r["success_rate"] for r in all_results if r["condition"] == "homeostatic"])
    
    print("\n" + "-" * 60)
    print(f"SUCCESS RATE IMPROVEMENT: {(homeo_success - baseline_success)*100:+.1f} pp")


if __name__ == "__main__":
    main()
