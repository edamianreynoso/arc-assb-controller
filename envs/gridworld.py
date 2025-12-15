"""
GridWorld Environment for L6 Validation.

A simple 5x5 grid with:
- Start position: (0, 0)
- Goal position: (4, 4) - reward +1
- Trap positions: (2, 2), (1, 3) - reward -0.5
- Each step: reward -0.01 (encourages efficiency)

Compatible with gymnasium-style API for future scaling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

@dataclass
class GridWorldConfig:
    size: int = 5
    goal_pos: Tuple[int, int] = (4, 4)
    trap_positions: Tuple[Tuple[int, int], ...] = ((2, 2), (1, 3))
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    trap_reward: float = -0.5
    max_steps: int = 100

class GridWorld:
    """
    Simple GridWorld environment.
    
    State: (row, col) flattened to integer
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self, config: Optional[GridWorldConfig] = None):
        self.config = config or GridWorldConfig()
        self.size = self.config.size
        self.n_states = self.size * self.size
        self.n_actions = 4
        
        self.agent_pos = [0, 0]
        self.steps = 0
        self.done = False
        
        # For ASSB integration: track environmental stress signals
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.trap_hits = 0
        
    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pos = [0, 0]
        self.steps = 0
        self.done = False
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.trap_hits = 0
        
        return self._get_state(), self._get_info()
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Returns: (state, reward, terminated, truncated, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, False, self._get_info()
        
        # Move agent
        row, col = self.agent_pos
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        self.agent_pos = [row, col]
        self.steps += 1
        
        # Calculate reward
        reward = self.config.step_penalty
        terminated = False
        
        if tuple(self.agent_pos) == self.config.goal_pos:
            reward = self.config.goal_reward
            terminated = True
        elif tuple(self.agent_pos) in self.config.trap_positions:
            reward = self.config.trap_reward
            self.trap_hits += 1
        
        truncated = self.steps >= self.config.max_steps
        self.done = terminated or truncated
        
        self.last_reward = reward
        self.cumulative_reward += reward
        
        return self._get_state(), reward, terminated, truncated, self._get_info()
    
    def _get_state(self) -> int:
        """Flatten (row, col) to single integer state."""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for ASSB integration."""
        return {
            "pos": tuple(self.agent_pos),
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward,
            "trap_hits": self.trap_hits,
            # ASSB signals
            "pe": abs(self.last_reward - 0.0) if self.last_reward != 0 else 0.1,  # Prediction error proxy
            "u_exog": 0.2 + 0.3 * (self.trap_hits / max(1, self.steps)),  # Uncertainty from traps
        }
    
    def render(self) -> str:
        """Text rendering of grid."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        for trap in self.config.trap_positions:
            grid[trap[0]][trap[1]] = 'T'
        
        grid[self.config.goal_pos[0]][self.config.goal_pos[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        
        return '\n'.join([' '.join(row) for row in grid])


class StochasticGridWorld(GridWorld):
    """
    GridWorld with stochastic transitions (slip probability).
    More challenging for RL agents.
    """
    
    def __init__(self, config: Optional[GridWorldConfig] = None, slip_prob: float = 0.1):
        super().__init__(config)
        self.slip_prob = slip_prob
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        # With slip_prob, take random action instead
        if np.random.random() < self.slip_prob:
            action = np.random.randint(0, self.n_actions)
        return super().step(action)


class ChangingGoalGridWorld(GridWorld):
    """
    GridWorld where goal position changes after N episodes.
    Tests transfer learning and memory retention.
    """
    
    def __init__(self, config: Optional[GridWorldConfig] = None, 
                 goal_positions: Tuple[Tuple[int, int], ...] = ((4, 4), (0, 4), (4, 0)),
                 change_every: int = 50):
        super().__init__(config)
        self.goal_positions = goal_positions
        self.change_every = change_every
        self.episode_count = 0
        self.current_goal_idx = 0
        self.goal_changed = False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        self.episode_count += 1
        self.goal_changed = False
        
        # Change goal periodically
        new_idx = (self.episode_count // self.change_every) % len(self.goal_positions)
        if new_idx != self.current_goal_idx:
            self.current_goal_idx = new_idx
            self.config.goal_pos = self.goal_positions[new_idx]
            self.goal_changed = True
        
        return super().reset(seed)
    
    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["goal_phase"] = self.current_goal_idx
        info["episode"] = self.episode_count
        info["goal_changed"] = self.goal_changed
        # Higher uncertainty during goal transition periods
        if self.episode_count % self.change_every < 5:
            info["u_exog"] = 0.7  # High uncertainty = goal just changed
        return info
