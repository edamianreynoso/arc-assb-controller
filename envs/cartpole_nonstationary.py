"""
Non-Stationary CartPole Environment for L6b Deep RL Validation.

The pole length changes periodically to test adaptation and transfer learning.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class NonStationaryCartPole(gym.Env):
    """
    CartPole with periodically changing pole length.
    
    This creates a non-stationary environment to test ARC's ability
    to protect learned knowledge during distribution shifts while
    still allowing adaptation.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        change_every: int = 100,
        pole_lengths: Tuple[float, ...] = (0.5, 1.0, 1.5),
    ):
        """
        Args:
            render_mode: Gym render mode
            change_every: Change pole length every N episodes
            pole_lengths: Tuple of pole lengths to cycle through
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.change_every = change_every
        self.pole_lengths = pole_lengths
        
        # Episode counter
        self.episode_count = 0
        self.current_phase = 0
        self.prev_phase = 0
        
        # Create base environment
        self._base_env = gym.make("CartPole-v1", render_mode=render_mode)
        
        # Copy spaces from base env
        self.action_space = self._base_env.action_space
        self.observation_space = self._base_env.observation_space
        
        # Set initial pole length
        self._set_pole_length(self.pole_lengths[0])
        
    def _set_pole_length(self, length: float):
        """Modify the pole length in the base environment."""
        # Access the internal CartPole physics
        if hasattr(self._base_env, 'unwrapped'):
            env = self._base_env.unwrapped
            env.length = length
            # Adjust other parameters for stability
            env.polemass_length = env.masspole * length
            
    def _get_current_pole_length(self) -> float:
        """Get current pole length based on episode count."""
        phase = (self.episode_count // self.change_every) % len(self.pole_lengths)
        return self.pole_lengths[phase]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment, potentially with new pole length."""
        super().reset(seed=seed)
        
        # Track phase changes
        self.prev_phase = self.current_phase
        self.current_phase = (self.episode_count // self.change_every) % len(self.pole_lengths)
        phase_changed = (self.current_phase != self.prev_phase)
        
        # Update pole length at phase boundaries
        new_length = self._get_current_pole_length()
        self._set_pole_length(new_length)
        
        # Reset base environment
        obs, base_info = self._base_env.reset(seed=seed, options=options)
        
        # Increment episode counter for next reset
        self.episode_count += 1
        
        # Add phase info
        info = {
            **base_info,
            "episode": self.episode_count,
            "phase": self.current_phase,
            "pole_length": new_length,
            "phase_changed": phase_changed,
            "u_exog": 0.3 if phase_changed else 0.1,  # Higher uncertainty on phase change
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        
        # Add ARC-relevant info
        info["phase"] = self.current_phase
        info["pole_length"] = self._get_current_pole_length()
        info["pe"] = 0.1  # Prediction error (simplified)
        info["u_exog"] = 0.1  # Uncertainty
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self._base_env.render()
    
    def close(self):
        """Close the environment."""
        self._base_env.close()


class StepWiseNonStationaryCartPole(NonStationaryCartPole):
    """
    Variant where pole length changes within episodes (more challenging).
    
    This tests ARC's ability to handle sudden within-episode shifts.
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        change_every_steps: int = 500,
        pole_lengths: Tuple[float, ...] = (0.5, 1.0),
    ):
        super().__init__(render_mode=render_mode, pole_lengths=pole_lengths)
        self.change_every_steps = change_every_steps
        self.total_steps = 0
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step with potential within-episode pole length change."""
        self.total_steps += 1
        
        # Check for phase change
        new_phase = (self.total_steps // self.change_every_steps) % len(self.pole_lengths)
        phase_changed = (new_phase != self.current_phase)
        
        if phase_changed:
            self.current_phase = new_phase
            self._set_pole_length(self.pole_lengths[new_phase])
        
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        
        # Add ARC-relevant info
        info["phase"] = self.current_phase
        info["pole_length"] = self.pole_lengths[self.current_phase]
        info["phase_changed"] = phase_changed
        info["pe"] = 0.3 if phase_changed else 0.1
        info["u_exog"] = 0.4 if phase_changed else 0.1
        
        return obs, reward, terminated, truncated, info


def make_nonstationary_cartpole(
    variant: str = "episode",
    change_every: int = 100,
    **kwargs
) -> gym.Env:
    """Factory function to create non-stationary CartPole variants."""
    if variant == "episode":
        return NonStationaryCartPole(change_every=change_every, **kwargs)
    elif variant == "step":
        return StepWiseNonStationaryCartPole(change_every_steps=change_every, **kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test the environment
    env = NonStationaryCartPole(change_every=5)
    
    for ep in range(15):
        obs, info = env.reset()
        print(f"Episode {ep}: pole_length={info['pole_length']:.1f}, phase={info['phase']}, changed={info.get('phase_changed', False)}")
        
        total_reward = 0
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  Reward: {total_reward:.1f}, Steps: {step+1}")
    
    env.close()
