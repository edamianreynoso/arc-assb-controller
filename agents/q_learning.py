"""
Q-Learning Agent with ASSB/ARC Integration for L6 Validation.

Two variants:
- QLearningAgent: Vanilla Q-Learning (baseline)
- ARCQLearningAgent: Q-Learning with ARC modulating learning rate and exploration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.state import State
from sim.dynamics import step_dynamics

@dataclass
class QLearningConfig:
    alpha: float = 0.1          # Learning rate
    gamma: float = 0.99         # Discount factor
    epsilon: float = 0.1        # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    n_states: int = 25
    n_actions: int = 4

class QLearningAgent:
    """Vanilla Q-Learning agent (baseline)."""
    
    name = "ql_baseline"
    
    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config or QLearningConfig()
        self.Q = np.zeros((self.config.n_states, self.config.n_actions))
        self.epsilon = self.config.epsilon
        
        # Stats for analysis
        self.td_errors: List[float] = []
        self.rewards: List[float] = []
        
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.config.n_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> float:
        """Q-Learning update. Returns TD error."""
        target = reward + (0 if done else self.config.gamma * np.max(self.Q[next_state]))
        td_error = target - self.Q[state, action]
        
        self.Q[state, action] += self.config.alpha * td_error
        
        self.td_errors.append(abs(td_error))
        self.rewards.append(reward)
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
    
    def reset_episode_stats(self):
        """Reset per-episode statistics."""
        self.td_errors = []
        self.rewards = []


class ARCQLearningAgent(QLearningAgent):
    """
    Q-Learning agent with ARC (Affective Regulation Core) integration.
    
    ARC modulates:
    - Learning rate (alpha): Reduced under high stress to prevent overwriting
    - Exploration (epsilon): Reduced under high uncertainty to avoid danger
    - Memory consolidation: Blocks updates when arousal is too high
    """
    
    name = "ql_arc"
    
    def __init__(self, config: Optional[QLearningConfig] = None, 
                 arc_config: Optional[Dict[str, Any]] = None,
                 use_shift_detection: bool = True,
                 use_mem_gating: bool = True):
        super().__init__(config)
        self.use_shift_detection = use_shift_detection
        self.use_mem_gating = use_mem_gating
        
        # Load ARC configuration from v2.yaml if not provided
        if arc_config is None:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "configs", "v2.yaml")
            with open(config_path, "r", encoding="utf-8") as f:
                self.arc_cfg = yaml.safe_load(f)
        else:
            self.arc_cfg = arc_config
        
        # Ensure initial state values exist
        self.arc_cfg.setdefault("u0", 0.2)
        
        # Initialize ASSB state
        self.assb_state = State(
            phi=self.arc_cfg["phi0"], g=self.arc_cfg["g0"],
            p=self.arc_cfg["p0"], i=self.arc_cfg["i0"],
            s=self.arc_cfg["s0"], v=self.arc_cfg["v0"],
            a=self.arc_cfg["a0"], mf=self.arc_cfg["mf0"],
            ms=self.arc_cfg["ms0"], u=self.arc_cfg["u0"]
        )
        
        # Track ARC metrics
        self.arousal_history: List[float] = []
        self.learning_rate_history: List[float] = []
        self.blocked_updates: int = 0

        # For non-stationary environments (e.g., changing goals)
        self.last_episode: Optional[int] = None
        self.last_goal_phase: Optional[int] = None
        self.shift_steps_remaining: int = 0
        self.shift_boost_steps: int = 30
        self.shift_eps_boost: float = 0.30
        self.shift_alpha_boost: float = 0.50
        self.uncertainty_eps_gain: float = 0.50
        self.uncertainty_alpha_gain: float = 0.30
        
    def on_reset(self, env_info: Optional[Dict[str, Any]] = None):
        """Hook for environments that provide episode-level context."""
        if not env_info:
            return
        self.last_episode = env_info.get("episode", self.last_episode)
        self.last_goal_phase = env_info.get("goal_phase", self.last_goal_phase)
        
        if self.use_shift_detection and env_info.get("goal_changed", False):
            self.shift_steps_remaining = self.shift_boost_steps
            
        u_exog = float(env_info.get("u_exog", self.assb_state.u))
        self.assb_state.u = max(self.assb_state.u, u_exog)

    def _compute_arc_control(self, u_exog: Optional[float] = None) -> Dict[str, float]:
        """Compute ARC control signals based on current ASSB state (+ optional exogenous uncertainty)."""
        st = self.assb_state
        cfg = self.arc_cfg
        uncertainty = max(float(st.u), float(u_exog if u_exog is not None else st.u))
        
        # Decompose risk (useful for RL: uncertainty should not always block learning)
        a_excess = max(0.0, st.a - cfg["a_safe"])
        s_excess = max(0.0, st.s - cfg["s_safe"])
        risk = (cfg["arc_w_u"] * uncertainty +
                cfg["arc_w_a"] * a_excess +
                cfg["arc_w_s"] * s_excess)
        risk = max(0.0, min(1.0, risk))

        risk_memory = (cfg["arc_w_a"] * a_excess + cfg["arc_w_s"] * s_excess)
        risk_memory = max(0.0, min(1.0, risk_memory))
        
        # Control signals
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk_memory)
        u_calm = min(1.0, cfg["arc_k_calm"] * max(0.0, st.a - cfg["a_safe"]))
        
        return {
            "u_dmg": u_dmg,
            "u_att": u_att,
            "u_mem": u_mem,
            "u_calm": u_calm,
            "u_reapp": 0.0,
            "risk": risk,
            "uncertainty": uncertainty,
        }
    
    def _modulate_learning_rate(self, base_alpha: float, arc_signals: Dict[str, float]) -> float:
        """
        Modulate learning rate based on ARC state.
        - High arousal -> Lower alpha (protect existing knowledge)
        - High uncertainty -> Lower alpha (don't learn from noise)
        """
        mem_gate = arc_signals["u_mem"] if self.use_mem_gating else 1.0
        uncertainty = float(arc_signals.get("uncertainty", 0.0))
        shift = 1.0 if (self.use_shift_detection and self.shift_steps_remaining > 0) else 0.0
        
        # Adaptation: uncertainty/shift can raise alpha, but arousal/DMN (via mem_gate) can still protect memory.
        alpha = base_alpha * (1.0 + self.uncertainty_alpha_gain * uncertainty + self.shift_alpha_boost * shift)
        modulated_alpha = alpha * mem_gate
        
        return float(modulated_alpha)
    
    def _modulate_exploration(self, base_epsilon: float, arc_signals: Dict[str, float]) -> float:
        """
        Modulate exploration based on ARC state.
        - High risk -> Lower epsilon (exploit safe actions)
        - Low risk -> Higher epsilon (can afford to explore)
        """
        uncertainty = float(arc_signals.get("uncertainty", 0.0))
        shift = 1.0 if (self.use_shift_detection and self.shift_steps_remaining > 0) else 0.0
        
        # In RL, uncertainty / context shift should increase exploration (phasic "search mode").
        modulated_epsilon = base_epsilon * (1.0 + self.uncertainty_eps_gain * uncertainty + self.shift_eps_boost * shift)
        
        return float(min(0.6, max(self.config.epsilon_min, modulated_epsilon)))
    
    def select_action(self, state: int) -> int:
        """ARC-modulated action selection."""
        arc_signals = self._compute_arc_control()
        
        # Modulate exploration
        current_epsilon = self._modulate_exploration(self.epsilon, arc_signals)
        
        if np.random.random() < current_epsilon:
            return np.random.randint(self.config.n_actions)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool, env_info: Optional[Dict] = None) -> float:
        """
        ARC-modulated Q-Learning update.
        
        1. Update ASSB state based on environment
        2. Compute ARC control signals
        3. Modulate learning rate
        4. Possibly block update if too stressed
        """
        # Get environmental signals
        pe = float(env_info.get("pe", 0.1)) if env_info else 0.1
        u_exog = float(env_info.get("u_exog", 0.2)) if env_info else 0.2

        # Detect episode / context shifts (goal changes)
        if env_info:
            ep = env_info.get("episode")
            phase = env_info.get("goal_phase")
            goal_changed = bool(env_info.get("goal_changed", False))
            if ep is not None and ep != self.last_episode:
                self.last_episode = ep
            if phase is not None and phase != self.last_goal_phase:
                self.last_goal_phase = phase
            if goal_changed:
                self.shift_steps_remaining = self.shift_boost_steps
        if self.shift_steps_remaining > 0:
            self.shift_steps_remaining -= 1
        
        # Compute ARC signals
        arc_signals = self._compute_arc_control(u_exog=u_exog)
        arc_control = {k: arc_signals[k] for k in ["u_dmg", "u_att", "u_mem", "u_calm", "u_reapp"]}
        
        # Update ASSB state
        self.assb_state = step_dynamics(
            self.assb_state, pe=pe, reward=reward, u_exog=u_exog,
            control=arc_control, cfg=self.arc_cfg
        )
        
        # Track metrics
        self.arousal_history.append(self.assb_state.a)
        
        # Modulate learning rate
        modulated_alpha = self._modulate_learning_rate(self.config.alpha, arc_signals)
        self.learning_rate_history.append(modulated_alpha)
        
        # Shift-aware memory gating: bypass protection during active shift detection
        # Rationale: When a goal change is detected, the agent NEEDS to update Q-values
        # to learn the new optimal policy. Memory gating should not block this.
        shift_active = self.use_shift_detection and self.shift_steps_remaining > 0
        
        # Block update only if memory gate is low AND we're NOT in shift mode
        if self.use_mem_gating and arc_signals["u_mem"] < 0.2 and not shift_active:
            self.blocked_updates += 1
            return 0.0  # No update
        
        # Standard Q-Learning update with modulated alpha
        target = reward + (0 if done else self.config.gamma * np.max(self.Q[next_state]))
        td_error = target - self.Q[state, action]
        
        self.Q[state, action] += modulated_alpha * td_error
        
        self.td_errors.append(abs(td_error))
        self.rewards.append(reward)
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        # ARC agent has slower epsilon decay (ARC handles exploration modulation)
        self.epsilon = max(self.config.epsilon_min,
                          self.epsilon * (self.config.epsilon_decay ** 0.5))
    
    def reset_episode_stats(self):
        """Reset per-episode statistics."""
        super().reset_episode_stats()
        self.arousal_history = []
        self.learning_rate_history = []
    
    def reset_assb_state(self):
        """Reset ASSB state to initial values (between experiments)."""
        self.assb_state = State(
            phi=self.arc_cfg["phi0"], g=self.arc_cfg["g0"],
            p=self.arc_cfg["p0"], i=self.arc_cfg["i0"],
            s=self.arc_cfg["s0"], v=self.arc_cfg["v0"],
            a=self.arc_cfg["a0"], mf=self.arc_cfg["mf0"],
            ms=self.arc_cfg["ms0"], u=self.arc_cfg["u0"]
        )
        self.blocked_updates = 0
