"""
ARC-PPO with Attention Modulation (LunarLander-v2).

Key Insight: Instead of modulating external hyperparameters (LR, clip_range),
we modulate INTERNAL representations via an attention mechanism.

Hypothesis:
- High arousal → Focus on basic survival features (position, velocity)
- Low arousal → Use all features normally (fine control)

This mimics how biological stress narrows attention to essential stimuli.
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
from collections import deque
import argparse

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
from sim.state import State
from sim.dynamics import step_dynamics
import yaml


def load_arc_config():
    config_path = REPO_ROOT / "configs" / "v2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Custom PPO with Arousal-Modulated Attention
# ==============================================================================

class ARCAttentionPolicy(nn.Module):
    """
    Policy network with arousal-modulated attention.
    
    The attention layer learns which features are important, but arousal
    can override this to focus on "survival" features when stressed.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        
        # Feature attention weights (learned)
        self.attention = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Sigmoid()
        )
        
        # Survival mask: which features are "essential"
        # For CartPole: [x, x_dot, theta, theta_dot]
        # Position (x) and angle (theta) are most critical for survival
        self.register_buffer('survival_mask', torch.tensor(
            [0.8, 0.5, 1.0, 0.6]  # Higher = more essential (angle most important)
        ))
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        
        # Actor head
        self.actor = nn.Linear(hidden, act_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden, 1)
        
    def forward(self, obs: torch.Tensor, arousal: float = 0.0):
        """
        Forward pass with arousal-modulated attention.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            arousal: Current arousal level [0, 1]
        
        Returns:
            action_probs: Action probabilities
            value: State value
        """
        # Compute learned attention
        learned_attention = self.attention(obs)
        
        # Blend with survival mask based on arousal
        # High arousal → more weight on survival features
        # Low arousal → use learned attention
        survival_attention = self.survival_mask.unsqueeze(0).expand_as(learned_attention)
        
        # Interpolate: arousal=0 → learned, arousal=1 → survival
        blended_attention = (1 - arousal) * learned_attention + arousal * survival_attention
        
        # Apply attention
        attended_obs = obs * blended_attention
        
        # Forward through backbone
        features = self.backbone(attended_obs)
        
        # Actor and Critic
        logits = self.actor(features)
        action_probs = torch.softmax(logits, dim=-1)
        value = self.critic(features)
        
        return action_probs, value, blended_attention


class ARCPPO:
    """
    PPO with ARC-modulated attention.
    """
    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        seed: int = 0,
        use_arc: bool = True,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_arc = use_arc
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.policy = ARCAttentionPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # ARC state
        self.arc_cfg = load_arc_config()
        self.assb_state = self._init_assb_state()
        self.current_arousal = 0.0
        
        # Rollout buffers
        self.obs_buffer = []
        self.act_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []
        self.arousal_buffer = []
        
        # Metrics
        self.episode_rewards = []
        self.arousal_history = []
        
    def _init_assb_state(self) -> State:
        cfg = self.arc_cfg
        return State(
            phi=cfg.get("phi0", 0.75), g=cfg.get("g0", 0.75),
            p=cfg.get("p0", 0.75), i=cfg.get("i0", 0.70),
            s=cfg.get("s0", 0.30), v=cfg.get("v0", 0.55),
            a=cfg.get("a0", 0.30), mf=cfg.get("mf0", 0.25),
            ms=cfg.get("ms0", 0.20), u=cfg.get("u_base", 0.20)
        )
    
    def _compute_arc_signals(self, reward: float, crashed: bool) -> float:
        """Compute arousal from reward signal."""
        cfg = self.arc_cfg
        
        # Use reward as prediction error proxy
        # Negative reward → high PE → high arousal
        pe = max(0, -reward / 100)  # Normalize
        
        # Crash → high uncertainty
        u_exog = 0.8 if crashed else 0.2
        
        # Compute risk
        a_excess = max(0.0, self.assb_state.a - cfg["a_safe"])
        s_excess = max(0.0, self.assb_state.s - cfg["s_safe"])
        
        risk = (cfg["arc_w_u"] * max(self.assb_state.u, u_exog) +
                cfg["arc_w_a"] * a_excess +
                cfg["arc_w_s"] * s_excess)
        risk = max(0.0, min(1.0, risk))
        
        # Control signals
        control = {
            "u_dmg": min(1.0, cfg["arc_k_dmg"] * risk),
            "u_att": min(1.0, cfg["arc_k_att"] * self.assb_state.u),
            "u_mem": 1.0 - min(1.0, cfg.get("arc_k_mem_block", 2.0) * risk),
            "u_calm": min(1.0, cfg["arc_k_calm"] * a_excess),
            "u_reapp": 0.0,
        }
        
        # Update ASSB state
        self.assb_state = step_dynamics(
            self.assb_state, pe=pe, reward=reward/100,
            u_exog=u_exog, control=control, cfg=cfg
        )
        
        return self.assb_state.a
    
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Select action with current policy."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        
        arousal = self.current_arousal if self.use_arc else 0.0
        
        with torch.no_grad():
            probs, value, _ = self.policy(obs_t, arousal)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self):
        """PPO update step."""
        # Convert buffers to tensors
        obs = torch.FloatTensor(np.array(self.obs_buffer))
        actions = torch.LongTensor(self.act_buffer)
        old_logprobs = torch.FloatTensor(self.logprob_buffer)
        advantages = torch.FloatTensor(self.advantage_buffer)
        returns = torch.FloatTensor(self.return_buffer)
        arousals = self.arousal_buffer if self.use_arc else [0.0] * len(self.obs_buffer)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Use mean arousal for batch (simplification)
                batch_arousal = np.mean([arousals[i] for i in batch_idx]) if self.use_arc else 0.0
                
                # Forward pass
                probs, values, _ = self.policy(batch_obs, batch_arousal)
                dist = Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
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
        """Train the agent."""
        obs, _ = self.env.reset()
        episode_reward = 0
        step = 0
        
        while step < total_timesteps:
            # Collect rollout
            self.obs_buffer = []
            self.act_buffer = []
            self.reward_buffer = []
            self.value_buffer = []
            self.logprob_buffer = []
            self.done_buffer = []
            self.arousal_buffer = []
            
            for _ in range(self.n_steps):
                action, log_prob, value = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.obs_buffer.append(obs)
                self.act_buffer.append(action)
                self.reward_buffer.append(reward)
                self.value_buffer.append(value)
                self.logprob_buffer.append(log_prob)
                self.done_buffer.append(float(done))
                
                # Update ARC state
                if self.use_arc:
                    crashed = terminated and reward < 0
                    self.current_arousal = self._compute_arc_signals(reward, crashed)
                    self.arousal_buffer.append(self.current_arousal)
                    self.arousal_history.append(self.current_arousal)
                
                episode_reward += reward
                step += 1
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    episode_reward = 0
                    obs, _ = self.env.reset()
                    self.assb_state = self._init_assb_state()
                    self.current_arousal = 0.3
                else:
                    obs = next_obs
                
                if step >= total_timesteps:
                    break
            
            # Compute GAE
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                _, next_value, _ = self.policy(obs_t, self.current_arousal if self.use_arc else 0.0)
                next_value = next_value.item()
            
            advantages, returns = self.compute_gae(
                self.reward_buffer, self.value_buffer, self.done_buffer, next_value
            )
            self.advantage_buffer = advantages
            self.return_buffer = returns
            
            # Update policy
            self.update()
        
        return {
            "episode_rewards": self.episode_rewards,
            "mean_reward": np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0,
            "mean_arousal": np.mean(self.arousal_history) if self.arousal_history else 0,
        }
    
    def evaluate(self, n_episodes: int = 20) -> Tuple[float, float]:
        """Evaluate the agent."""
        rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0
            done = False
            arousal = 0.3
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs, _, _ = self.policy(obs_t, arousal if self.use_arc else 0.0)
                    action = probs.argmax(1).item()
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
        
        return np.mean(rewards), np.std(rewards)


def run_experiment(condition: str, seed: int, timesteps: int = 100000) -> Dict[str, Any]:
    """Run a single experiment."""
    np.random.seed(seed)
    
    # Use NonStationaryCartPole with extreme parameters
    # Much harder: rapid changes, extreme pole lengths
    from envs.cartpole_nonstationary import NonStationaryCartPole
    env = NonStationaryCartPole(
        change_every=15,  # Very rapid changes
        pole_lengths=(0.3, 0.5, 1.0, 2.0, 3.0)  # Extreme range
    )
    
    use_arc = (condition == "arc_attention")
    
    agent = ARCPPO(
        env=env,
        lr=3e-4,
        gamma=0.99,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        seed=seed,
        use_arc=use_arc,
    )
    
    print(f"Training {condition} (seed={seed})...")
    start = datetime.now()
    results = agent.train(timesteps)
    duration = (datetime.now() - start).total_seconds()
    
    # Evaluate
    mean_reward, std_reward = agent.evaluate(n_episodes=20)
    
    env.close()
    
    return {
        "condition": condition,
        "seed": seed,
        "train_time": duration,
        "final_reward": mean_reward,
        "final_std": std_reward,
        "mean_arousal": results.get("mean_arousal", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()
    
    conditions = ["baseline", "arc_attention"]
    all_results = []
    
    print("=" * 60)
    print("ARC-PPO Attention Modulation (LunarLander-v2)")
    print("=" * 60)
    
    for cond in conditions:
        print(f"\n--- {cond.upper()} ---")
        for seed in range(args.seeds):
            res = run_experiment(cond, seed, args.timesteps)
            all_results.append(res)
            print(f"  Seed {seed}: Reward={res['final_reward']:.1f} +/- {res['final_std']:.1f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        rewards = [r["final_reward"] for r in cond_results]
        
        print(f"\n{cond.upper()}:")
        print(f"  Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        if cond == "arc_attention":
            arousals = [r["mean_arousal"] for r in cond_results]
            print(f"  Mean Arousal: {np.mean(arousals):.3f}")
    
    baseline_reward = np.mean([r["final_reward"] for r in all_results if r["condition"] == "baseline"])
    arc_reward = np.mean([r["final_reward"] for r in all_results if r["condition"] == "arc_attention"])
    
    print("\n" + "-" * 60)
    print(f"IMPROVEMENT: {arc_reward - baseline_reward:+.1f} points")


if __name__ == "__main__":
    main()
