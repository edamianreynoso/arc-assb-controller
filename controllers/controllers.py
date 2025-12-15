"""
ARC Controllers - Public Reference Implementation

This module contains the reference implementation of ARC v1 (proportional control)
as described in the paper. For production use cases, contact: edamianreynoso@gmail.com
"""

from typing import Dict, Any
import math


def clip01(x: float) -> float:
    """Clip value to [0, 1]."""
    return max(0.0, min(1.0, x))


class ARCv1Controller:
    """
    ARC v1: Proportional Controller
    
    Reference implementation for paper reproducibility.
    See Section 4.3 and Appendix C.
    """
    
    def __init__(self):
        self.name = "arc_v1"
    
    def act(self, state, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute control actions based on internal state and observations.
        
        Args:
            state: Current ASSB state (with fields .a, .s, .u, etc.)
            obs: Observations dict with 'pe', 'reward', 'u_exog'
            cfg: Configuration dict with controller parameters
        
        Returns:
            Control actions dict with u_dmg, u_att, u_mem, u_calm, u_reapp
        """
        # Extract state variables
        a = state.a  # arousal
        s = state.s  # narrative gain
        u = state.u  # uncertainty
        
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.6)
        s_safe = cfg.get("s_safe", 0.55)
        
        # Risk weights
        w_u = cfg.get("arc_w_u", 0.4)
        w_a = cfg.get("arc_w_a", 0.3)
        w_s = cfg.get("arc_w_s", 0.35)
        
        # Control gains
        k_dmg = cfg.get("arc_k_dmg", 0.95)
        k_calm = cfg.get("arc_k_calm", 0.85)
        k_att = cfg.get("arc_k_att", 0.75)
        k_reapp = cfg.get("arc_k_reapp", 0.5)
        k_mem_block = cfg.get("arc_k_mem_block", 0.8)
        
        # Compute risk signal (Eq. C.1)
        a_excess = max(0.0, a - a_safe)
        s_excess = max(0.0, s - s_safe)
        risk = clip01(w_u * u + w_a * a_excess + w_s * s_excess)
        
        # Compute control actions (Eq. C.2)
        u_dmg = min(1.0, k_dmg * risk)
        u_att = min(1.0, k_att * u * (1.0 - a_excess))
        u_mem = 1.0 - min(1.0, k_mem_block * risk)
        u_calm = min(1.0, k_calm * a_excess)
        u_reapp = min(1.0, k_reapp * u * (1.0 - risk))
        
        return {
            "u_dmg": u_dmg,
            "u_att": u_att,
            "u_mem": u_mem,
            "u_calm": u_calm,
            "u_reapp": u_reapp,
        }


class NoController:
    """Baseline: No control (all actions zero except u_mem=1)."""
    
    def __init__(self):
        self.name = "no_control"
    
    def act(self, state, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        return {
            "u_dmg": 0.0,
            "u_att": 0.0,
            "u_mem": 1.0,
            "u_calm": 0.0,
            "u_reapp": 0.0,
        }


class NaiveCalmController:
    """Baseline: Only calming, no other regulation."""
    
    def __init__(self):
        self.name = "naive_calm"
    
    def act(self, state, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        a_safe = cfg.get("a_safe", 0.6)
        a_excess = max(0.0, state.a - a_safe)
        
        return {
            "u_dmg": 0.0,
            "u_att": 0.0,
            "u_mem": 1.0,
            "u_calm": min(1.0, a_excess * 2.0),
            "u_reapp": 0.0,
        }


# Controller registry
CONTROLLERS = {
    "arc_v1": ARCv1Controller,
    "no_control": NoController,
    "naive_calm": NaiveCalmController,
}


def get_controller(name: str):
    """Get controller by name."""
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {name}. Available: {list(CONTROLLERS.keys())}")
    return CONTROLLERS[name]()
