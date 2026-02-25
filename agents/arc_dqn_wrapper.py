"""
ARC Wrapper for Stable-Baselines3 DQN.

This module provides a gymnasium wrapper that integrates ARC's
affective regulation with deep RL algorithms like DQN.

The wrapper:
1. Maintains an internal ASSB state
2. Computes ARC control signals
3. Optionally modulates reward signal
4. Tracks affective metrics for analysis
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.state import State
from sim.dynamics import step_dynamics


@dataclass
class ARCWrapperConfig:
    """Configuration for ARC wrapper."""
    # Control features
    use_reward_shaping: bool = False  # Whether to add ARC-based reward shaping
    use_observation_augmentation: bool = False  # Add ARC state to observations
    use_shift_detection: bool = True  # Track distribution shifts (phase changes)
    
    # Thresholds
    a_safe: float = 0.6
    s_safe: float = 0.55
    
    # ARC weights
    arc_w_u: float = 0.40
    arc_w_a: float = 0.40
    arc_w_s: float = 0.35
    
    # Control gains
    arc_k_dmg: float = 0.95
    arc_k_att: float = 0.75
    arc_k_calm: float = 0.85
    arc_k_mem_block: float = 0.50

    # Deep-RL integration helpers
    shift_boost_steps: int = 50  # Steps to treat as "shift mode" after a phase change
    shift_u_exog: float = 0.4  # Minimum uncertainty during shift mode
    shift_mem_gate_floor: float = 0.5  # Don't over-block learning during shift mode (u_mem lower bound)
    mem_gate_include_uncertainty: bool = False  # If True, let uncertainty close u_mem (more aggressive)
    
    # Stability bonus/penalty
    stability_bonus: float = 0.1  # Bonus for maintaining low arousal
    instability_penalty: float = 0.0  # Penalty for high arousal (disabled by default)


class ARCGymWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that integrates ARC affective regulation.
    
    This wrapper maintains an internal ASSB state and computes
    ARC control signals that can be used for:
    - Reward shaping (optional)
    - Observation augmentation (optional)
    - Metric tracking
    
    For DQN and other function approximation methods, we don't
    directly modulate learning rate (that's internal to the algorithm).
    Instead, we can influence learning through reward shaping or
    by adding the affective state to observations.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[ARCWrapperConfig] = None,
        arc_yaml_path: Optional[str] = None,
    ):
        super().__init__(env)
        
        self.config = config or ARCWrapperConfig()
        
        # Load ARC configuration from YAML if provided
        if arc_yaml_path:
            import yaml
            with open(arc_yaml_path, "r", encoding="utf-8") as f:
                self.arc_cfg = yaml.safe_load(f)
        else:
            # Default configuration matching v2.yaml
            self.arc_cfg = self._default_arc_config()
        
        # Initialize ASSB state
        self.assb_state = self._init_assb_state()
        
        # Tracking
        self.arousal_history: List[float] = []
        self.valence_history: List[float] = []
        self.risk_history: List[float] = []
        self.risk_memory_history: List[float] = []
        self.u_mem_history: List[float] = []
        self.uncertainty_history: List[float] = []
        self.shift_active_history: List[bool] = []
        self.episode_steps = 0
        self.phase_changed = False
        self.current_phase = 0
        self.shift_steps_remaining = 0

        # Optional external signals from the RL algorithm (e.g., TD error)
        self._external_pe: Optional[float] = None
        self._external_u_exog: Optional[float] = None
        
        # Modify observation space if augmentation is enabled
        if self.config.use_observation_augmentation:
            # Add 4 ARC states: arousal, valence, risk, memory_gating_signal
            extra_low = np.zeros(4, dtype=np.float32)
            extra_high = np.ones(4, dtype=np.float32)
            self.observation_space = spaces.Box(
                low=np.concatenate([self.observation_space.low.astype(np.float32), extra_low]),
                high=np.concatenate([self.observation_space.high.astype(np.float32), extra_high]),
                dtype=np.float32
            )

    def set_external_signals(self, pe: Optional[float] = None, u_exog: Optional[float] = None) -> None:
        """
        Optionally inject RL-derived signals (e.g., TD error as prediction error).

        Notes:
        - These are read on the next `step()` call (1-step delayed vs the transition that produced them).
        - Values are clipped to [0, 1] because ASSB dynamics expect bounded inputs.
        """
        if pe is None:
            self._external_pe = None
        else:
            self._external_pe = float(np.clip(float(pe), 0.0, 1.0))

        if u_exog is None:
            self._external_u_exog = None
        else:
            self._external_u_exog = float(np.clip(float(u_exog), 0.0, 1.0))
    
    def _default_arc_config(self) -> Dict[str, Any]:
        """Return default ARC configuration matching v2.yaml."""
        return {
            # Initial state values
            "phi0": 0.75, "g0": 0.75, "p0": 0.75, "i0": 0.70,
            "s0": 0.30, "v0": 0.55, "a0": 0.30, "mf0": 0.25,
            "ms0": 0.20, "u0": 0.20,
            
            # Thresholds
            "a_safe": self.config.a_safe,
            "s_safe": self.config.s_safe,
            
            # ARC weights
            "arc_w_u": self.config.arc_w_u,
            "arc_w_a": self.config.arc_w_a,
            "arc_w_s": self.config.arc_w_s,
            
            # Control gains
            "arc_k_dmg": self.config.arc_k_dmg,
            "arc_k_att": self.config.arc_k_att,
            "arc_k_calm": self.config.arc_k_calm,
            "arc_k_mem_block": self.config.arc_k_mem_block,
            "arc_k_reapp": 0.55,
            
            # Performance parameters
            "omega_s": 0.35,
            "perf_bias": 0.25,
            "perf_gain": 0.85,
            "w_u": 0.25,
            "w_a": 0.30,
            "w_s": 0.20,
            
            # Attention/Uncertainty dynamics
            "k_u_att": 0.30,
            
            # Integration dynamics
            "k_i_att": 0.25,
            "k_i_u": 0.06,
            "mu_i": 0.03,
            
            # Precision dynamics
            "k_p_pe": 0.15,
            "k_p_u": 0.05,
            "k_p_i": 0.12,
            "mu_p": 0.015,
            
            # Gating dynamics
            "k_g_i": 0.08,
            "k_g_p": 0.12,
            "k_g_u": 0.08,
            "k_g_a": 0.10,
            "mu_g": 0.015,
            
            # Phi dynamics
            "k_phi_gp": 0.08,
            "mu_phi": 0.015,
            
            # Narrative intensity (rumination) dynamics
            "k_s_u": 0.08,
            "k_s_pe": 0.06,
            "k_s_dmg": 0.25,
            "mu_s": 0.025,
            
            # Arousal dynamics
            "k_a_pe": 0.15,
            "k_a_u": 0.12,
            "k_a_s": 0.08,
            "k_a_calm": 0.40,
            "mu_a": 0.05,
            
            # Valence dynamics
            "k_v_r": 0.25,
            "k_v_pe": 0.08,
            "k_v_u": 0.05,
            "k_v_reapp": 0.12,
            "mu_v": 0.04,
            
            # Memory dynamics
            "eta0": 0.18,
            "k_eta_a": 0.60,
            "w_mem_pe": 0.55,
            "w_mem_a": 0.30,
            "w_mem_v": 0.25,
            "mu_mf": 0.06,
            "k_ms": 0.02,
            "mu_ms": 0.01,
        }
    
    def _init_assb_state(self) -> State:
        """Initialize ASSB state."""
        cfg = self.arc_cfg
        return State(
            phi=cfg.get("phi0", 0.5),
            g=cfg.get("g0", 0.5),
            p=cfg.get("p0", 0.6),
            i=cfg.get("i0", 0.5),
            s=cfg.get("s0", 0.4),
            v=cfg.get("v0", 0.6),
            a=cfg.get("a0", 0.3),
            mf=cfg.get("mf0", 0.5),
            ms=cfg.get("ms0", 0.5),
            u=cfg.get("u0", 0.2),
        )
    
    def _compute_arc_signals(self, u_exog: float = 0.1, shift_active: bool = False) -> Dict[str, float]:
        """Compute ARC control signals from current ASSB state."""
        st = self.assb_state
        cfg = self.arc_cfg
        
        # Compute risk components
        uncertainty = max(float(st.u), float(u_exog))
        a_excess = max(0.0, st.a - cfg["a_safe"])
        s_excess = max(0.0, st.s - cfg["s_safe"])
        
        # Total risk (uncertainty + internal stress)
        risk = (cfg["arc_w_u"] * uncertainty +
                cfg["arc_w_a"] * a_excess +
                cfg["arc_w_s"] * s_excess)
        risk = max(0.0, min(1.0, risk))

        # Memory risk: by default, gate memory primarily on internal overload (arousal + narrative),
        # not on uncertainty. For some deep-RL settings (catastrophic forgetting), enabling uncertainty
        # gating can be beneficial.
        if self.config.mem_gate_include_uncertainty:
            risk_memory = (cfg["arc_w_u"] * uncertainty + cfg["arc_w_a"] * a_excess + cfg["arc_w_s"] * s_excess)
        else:
            risk_memory = (cfg["arc_w_a"] * a_excess + cfg["arc_w_s"] * s_excess)
        risk_memory = max(0.0, min(1.0, risk_memory))
        
        # Control signals
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_calm = min(1.0, cfg["arc_k_calm"] * a_excess)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - a_excess))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk_memory)
        if shift_active:
            u_mem = max(float(self.config.shift_mem_gate_floor), float(u_mem))
        
        return {
            "risk": risk,
            "risk_memory": risk_memory,
            "uncertainty": uncertainty,
            "u_dmg": u_dmg,
            "u_calm": u_calm,
            "u_att": u_att,
            "u_mem": u_mem,
            "u_reapp": 0.0,
        }
    
    def _shape_reward(self, reward: float, arc_signals: Dict[str, float]) -> float:
        """Apply ARC-based reward shaping."""
        if not self.config.use_reward_shaping:
            return reward
        
        # Stability bonus: reward for maintaining low arousal
        stability_bonus = 0.0
        if self.assb_state.a < self.config.a_safe:
            stability_bonus = self.config.stability_bonus
        
        # Instability penalty
        instability_penalty = 0.0
        if self.assb_state.a > self.config.a_safe:
            excess = self.assb_state.a - self.config.a_safe
            instability_penalty = self.config.instability_penalty * excess
        
        return reward + stability_bonus - instability_penalty
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Add ARC state to observation."""
        if not self.config.use_observation_augmentation:
            return obs
        
        shift_active = self.shift_steps_remaining > 0
        arc_signals = self._compute_arc_signals(shift_active=shift_active)
        arc_obs = np.array([
            self.assb_state.a,  # Arousal
            self.assb_state.v,  # Valence
            arc_signals["risk"],  # Risk
            arc_signals["u_mem"], # Memory gating signal (0 = gated, 1 = normal)
        ], dtype=np.float32)
        
        return np.concatenate([obs, arc_obs])
    
    def reset(self, **kwargs):
        """Reset environment and ASSB state."""
        obs, info = self.env.reset(**kwargs)
        
        # Partially reset ASSB state (maintain some memory)
        # Full reset would lose the protective memory effect
        self.assb_state = State(
            phi=self.arc_cfg.get("phi0", 0.5),
            g=self.arc_cfg.get("g0", 0.5),
            p=self.arc_cfg.get("p0", 0.6),
            i=self.arc_cfg.get("i0", 0.5),
            s=self.arc_cfg.get("s0", 0.4),
            v=self.arc_cfg.get("v0", 0.6),
            a=min(self.assb_state.a, 0.4),  # Maintain some arousal
            mf=self.assb_state.mf,  # Keep memory
            ms=self.assb_state.ms,  # Keep memory
            u=info.get("u_exog", 0.2),
        )
        
        # Track phase changes
        self.phase_changed = info.get("phase_changed", False)
        self.current_phase = info.get("phase", 0)
        if self.config.use_shift_detection and self.phase_changed and self.config.shift_boost_steps > 0:
            self.shift_steps_remaining = int(self.config.shift_boost_steps)
        else:
            self.shift_steps_remaining = 0
        self.episode_steps = 0
        
        # Clear per-episode tracking
        self.arousal_history = []
        self.valence_history = []
        self.risk_history = []
        self.risk_memory_history = []
        self.u_mem_history = []
        self.uncertainty_history = []
        self.shift_active_history = []
        
        # Add ARC info
        info["arc_arousal"] = self.assb_state.a
        info["arc_phase_changed"] = self.phase_changed
        info["arc_shift_active"] = self.shift_steps_remaining > 0
        
        return self._augment_observation(obs), info
    
    def step(self, action):
        """Step environment with ARC integration."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Extract environmental signals
        pe = float(info.get("pe", 0.1))
        u_exog = float(info.get("u_exog", 0.1))

        # Allow RL-side injection (e.g., TD error) to drive internal dynamics
        if self._external_pe is not None:
            pe = float(self._external_pe)
        if self._external_u_exog is not None:
            u_exog = max(float(u_exog), float(self._external_u_exog))
        phase_changed = bool(info.get("phase_changed", False))
        if self.config.use_shift_detection and phase_changed and self.config.shift_boost_steps > 0:
            self.shift_steps_remaining = int(self.config.shift_boost_steps)

        shift_active = self.shift_steps_remaining > 0
        if shift_active:
            self.shift_steps_remaining -= 1

        # Increase uncertainty during shift mode
        if shift_active:
            u_exog = max(float(u_exog), float(self.config.shift_u_exog))
            self.phase_changed = True
        
        # Compute ARC signals
        arc_signals = self._compute_arc_signals(u_exog=u_exog, shift_active=shift_active)
        
        # Build control dict for dynamics
        control = {
            "u_dmg": arc_signals["u_dmg"],
            "u_att": arc_signals["u_att"],
            "u_mem": arc_signals["u_mem"],
            "u_calm": arc_signals["u_calm"],
            "u_reapp": arc_signals["u_reapp"],
        }
        
        # Update ASSB state
        self.assb_state = step_dynamics(
            self.assb_state,
            pe=pe,
            reward=reward,
            u_exog=u_exog,
            control=control,
            cfg=self.arc_cfg,
        )
        
        # Track metrics
        self.arousal_history.append(self.assb_state.a)
        self.valence_history.append(self.assb_state.v)
        self.risk_history.append(arc_signals["risk"])
        self.risk_memory_history.append(float(arc_signals.get("risk_memory", 0.0)))
        self.u_mem_history.append(float(arc_signals.get("u_mem", 1.0)))
        self.uncertainty_history.append(float(arc_signals.get("uncertainty", u_exog)))
        self.shift_active_history.append(bool(shift_active))
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, arc_signals)
        
        # Add ARC info
        info["arc_arousal"] = self.assb_state.a
        info["arc_valence"] = self.assb_state.v
        info["arc_risk"] = arc_signals["risk"]
        info["arc_uncertainty"] = arc_signals.get("uncertainty", u_exog)
        info["arc_risk_memory"] = arc_signals.get("risk_memory", 0.0)
        info["arc_u_mem"] = arc_signals["u_mem"]
        info["arc_shift_active"] = shift_active

        # Episode summary metrics (must be stored in `info` because VecEnv auto-resets).
        if terminated or truncated:
            ep_metrics = self.get_arc_metrics()
            for k, v in ep_metrics.items():
                info[f"arc_ep_{k}"] = float(v)
        
        return self._augment_observation(obs), shaped_reward, terminated, truncated, info
    
    def get_arc_metrics(self) -> Dict[str, float]:
        """Get summary ARC metrics for the episode."""
        if not self.arousal_history:
            return {"mean_arousal": 0.0, "max_arousal": 0.0, "mean_risk": 0.0}
        
        return {
            "mean_arousal": np.mean(self.arousal_history),
            "max_arousal": np.max(self.arousal_history),
            "mean_valence": np.mean(self.valence_history),
            "mean_risk": np.mean(self.risk_history),
            "high_arousal_ratio": np.mean([a > self.config.a_safe for a in self.arousal_history]),
            "mean_risk_memory": float(np.mean(self.risk_memory_history)) if self.risk_memory_history else 0.0,
            "mean_u_mem": float(np.mean(self.u_mem_history)) if self.u_mem_history else 1.0,
            "min_u_mem": float(np.min(self.u_mem_history)) if self.u_mem_history else 1.0,
            "mean_uncertainty": float(np.mean(self.uncertainty_history)) if self.uncertainty_history else 0.0,
            "shift_active_ratio": float(np.mean(self.shift_active_history)) if self.shift_active_history else 0.0,
        }


def make_arc_wrapped_env(
    env_id: str = "CartPole-v1",
    arc_config: Optional[ARCWrapperConfig] = None,
    **env_kwargs
) -> ARCGymWrapper:
    """Factory function to create ARC-wrapped environments."""
    env = gym.make(env_id, **env_kwargs)
    return ARCGymWrapper(env, config=arc_config)


if __name__ == "__main__":
    # Test the wrapper
    from envs.cartpole_nonstationary import NonStationaryCartPole
    
    base_env = NonStationaryCartPole(change_every=5)
    wrapped_env = ARCGymWrapper(base_env)
    
    print("Testing ARC Wrapper...")
    for ep in range(10):
        obs, info = wrapped_env.reset()
        total_reward = 0
        
        for step in range(200):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        metrics = wrapped_env.get_arc_metrics()
        print(f"Episode {ep}: reward={total_reward:.1f}, "
              f"mean_arousal={metrics['mean_arousal']:.3f}, "
              f"mean_risk={metrics['mean_risk']:.3f}")
    
    wrapped_env.close()
