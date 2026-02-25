"""
ADVERSARIAL ENVIRONMENTS FOR ARC-DQN VALIDATION

These environments are designed to stress-test learning systems
with conditions where affective regulation should help:
1. Sudden, extreme distribution shifts
2. Observation noise (partial observability)
3. Adversarial reward signals
4. Within-episode changes
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

class AdversarialCartPole(gym.Env):
    """
    CartPole with adversarial conditions:
    - Extreme pole length changes (0.2 → 3.0)
    - Observation noise
    - Random action failures (stochastic transitions)
    - Occasional reward inversions
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        change_every: int = 15,  # More frequent
        pole_lengths: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 3.0),  # More extreme
        obs_noise_std: float = 0.1,  # Observation noise
        action_fail_prob: float = 0.1,  # Action failure probability
        reward_inversion_prob: float = 0.05,  # Adversarial rewards
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.change_every = change_every
        self.pole_lengths = pole_lengths
        self.obs_noise_std = obs_noise_std
        self.action_fail_prob = action_fail_prob
        self.reward_inversion_prob = reward_inversion_prob
        
        self.episode_count = 0
        self.current_phase = 0
        self.prev_phase = 0
        self.step_count = 0
        
        self._base_env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self._base_env.action_space
        self.observation_space = self._base_env.observation_space
        
        self._set_pole_length(self.pole_lengths[0])
        
    def _set_pole_length(self, length: float):
        if hasattr(self._base_env, 'unwrapped'):
            env = self._base_env.unwrapped
            env.length = length
            env.polemass_length = env.masspole * length
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.prev_phase = self.current_phase
        self.current_phase = (self.episode_count // self.change_every) % len(self.pole_lengths)
        phase_changed = (self.current_phase != self.prev_phase)
        
        new_length = self.pole_lengths[self.current_phase]
        self._set_pole_length(new_length)
        
        obs, base_info = self._base_env.reset(seed=seed, options=options)
        
        # Add observation noise
        obs = obs + np.random.normal(0, self.obs_noise_std, obs.shape)
        
        self.episode_count += 1
        self.step_count = 0
        
        info = {
            **base_info,
            "episode": self.episode_count,
            "phase": self.current_phase,
            "pole_length": new_length,
            "phase_changed": phase_changed,
            "u_exog": 0.5 if phase_changed else 0.1,
            "pe": 0.3 if phase_changed else 0.1,
        }
        
        return obs.astype(np.float32), info
    
    def step(self, action):
        self.step_count += 1
        
        # Action failure (stochastic transitions)
        if np.random.random() < self.action_fail_prob:
            action = 1 - action  # Flip action
        
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        
        # Add observation noise
        obs = obs + np.random.normal(0, self.obs_noise_std, obs.shape)
        
        # Adversarial reward inversion
        if np.random.random() < self.reward_inversion_prob:
            reward = -reward
        
        info["phase"] = self.current_phase
        info["pole_length"] = self.pole_lengths[self.current_phase]
        info["pe"] = 0.2 if self.step_count < 10 else 0.1
        info["u_exog"] = 0.15
        
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def render(self):
        return self._base_env.render()
    
    def close(self):
        self._base_env.close()


class CatastrophicForgettingEnv(gym.Env):
    """
    Environment designed to cause catastrophic forgetting.
    The dynamics completely change every N episodes.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        change_every: int = 20,
        gravity_levels: Tuple[float, ...] = (5.0, 9.8, 15.0, 25.0),
        force_mags: Tuple[float, ...] = (5.0, 10.0, 20.0, 30.0),
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.change_every = change_every
        self.gravity_levels = gravity_levels
        self.force_mags = force_mags
        
        self.episode_count = 0
        self.current_phase = 0
        
        self._base_env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self._base_env.action_space
        self.observation_space = self._base_env.observation_space
        
    def _apply_dynamics(self):
        if hasattr(self._base_env, 'unwrapped'):
            env = self._base_env.unwrapped
            phase = self.current_phase % len(self.gravity_levels)
            env.gravity = self.gravity_levels[phase]
            env.force_mag = self.force_mags[phase]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        prev_phase = self.current_phase
        self.current_phase = (self.episode_count // self.change_every) % len(self.gravity_levels)
        phase_changed = (self.current_phase != prev_phase)
        
        self._apply_dynamics()
        obs, base_info = self._base_env.reset(seed=seed, options=options)
        self.episode_count += 1
        
        info = {
            **base_info,
            "episode": self.episode_count,
            "phase": self.current_phase,
            "phase_changed": phase_changed,
            "gravity": self.gravity_levels[self.current_phase % len(self.gravity_levels)],
            "force_mag": self.force_mags[self.current_phase % len(self.force_mags)],
            "u_exog": 0.6 if phase_changed else 0.1,
            "pe": 0.4 if phase_changed else 0.1,
        }
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        info["phase"] = self.current_phase
        info["pe"] = 0.1
        info["u_exog"] = 0.1
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._base_env.render()
    
    def close(self):
        self._base_env.close()


class HighStressEnv(gym.Env):
    """
    Environment with high "stress" conditions:
    - Shorter episode time limits
    - Negative rewards for being near boundaries
    - Bonus for stability, penalty for oscillation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 100,  # Shorter episodes
        boundary_penalty: float = -0.5,
        oscillation_penalty: float = -0.1,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.boundary_penalty = boundary_penalty
        self.oscillation_penalty = oscillation_penalty
        
        self._base_env = gym.make("CartPole-v1", render_mode=render_mode)
        self.action_space = self._base_env.action_space
        self.observation_space = self._base_env.observation_space
        
        self.step_count = 0
        self.prev_action = None
        self.action_switches = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self._base_env.reset(seed=seed, options=options)
        self.step_count = 0
        self.prev_action = None
        self.action_switches = 0
        info["u_exog"] = 0.2
        info["pe"] = 0.1
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        # Track oscillation
        if self.prev_action is not None and action != self.prev_action:
            self.action_switches += 1
        self.prev_action = action
        
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        
        # Shorter episodes
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Boundary penalty (cart near edge)
        cart_pos = obs[0]
        if abs(cart_pos) > 1.5:
            reward += self.boundary_penalty
        
        # Oscillation penalty
        if self.action_switches > self.step_count * 0.8:
            reward += self.oscillation_penalty
        
        info["step"] = self.step_count
        info["oscillation_rate"] = self.action_switches / max(1, self.step_count)
        info["pe"] = 0.15
        info["u_exog"] = 0.1
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._base_env.render()
    
    def close(self):
        self._base_env.close()


if __name__ == "__main__":
    print("Testing Adversarial CartPole...")
    env = AdversarialCartPole()
    
    for ep in range(10):
        obs, info = env.reset()
        print(f"Ep {ep}: pole={info['pole_length']:.1f}, phase={info['phase']}, changed={info.get('phase_changed')}")
        
        total_r = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        print(f"  Reward: {total_r:.1f}")
    
    env.close()
    print("\n✅ All environments work!")
