from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable
import random
import math

@dataclass
class Scenario:
    name: str
    horizon: int
    shock_t: int
    generator: Callable[[int, random.Random], Tuple[float, float, float]]

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def build_scenarios(cfg: Dict[str, Any]) -> List[Scenario]:
    H = cfg["horizon"]
    shock_t = cfg["shock_t"]

    def sudden_threat(t: int, rng: random.Random):
        u = cfg["u_base"] if t < shock_t else cfg["u_shock"]
        pe = _clip01(cfg["pe_base"] + (0.15 if t >= shock_t else 0.0) + rng.random()*cfg["pe_noise"])
        r = (0.2 if t < shock_t else 0.1) + (rng.random()-0.5)*0.1
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def reward_flip(t: int, rng: random.Random):
        u = cfg["u_base"]
        pe = _clip01(cfg["pe_base"] + rng.random()*cfg["pe_noise"] + (0.2 if t == shock_t else 0.0))
        r = (0.3 if t < shock_t else -0.3) + (rng.random()-0.5)*0.1
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def noise_burst(t: int, rng: random.Random):
        u = cfg["u_base"]
        burst = (t >= shock_t and t < shock_t + cfg["burst_len"])
        pe = _clip01((0.65 if burst else cfg["pe_base"]) + rng.random()*cfg["pe_noise"])
        r = (0.2 if not burst else 0.05) + (rng.random()-0.5)*0.1
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    # =========================================================================
    # L2 SCENARIOS - Memory & Continual Learning
    # =========================================================================
    
    def distribution_shift(t: int, rng: random.Random):
        """
        Multiple distribution shifts to test memory consolidation.
        Phase 1 (0-50): learn task A (reward for low PE)
        Phase 2 (50-100): learn task B (reward shifts, high PE)
        Phase 3 (100-160): return to task A (test retention)
        """
        phase1_end = 50
        phase2_end = 100
        
        if t < phase1_end:
            # Phase 1: stable, low uncertainty
            u = cfg["u_base"]
            pe = _clip01(cfg["pe_base"] + rng.random()*cfg["pe_noise"])
            r = 0.4 + (rng.random()-0.5)*0.1  # positive reward
        elif t < phase2_end:
            # Phase 2: new distribution, higher uncertainty
            u = cfg["u_shock"] * 0.6
            pe = _clip01(0.4 + rng.random()*0.2)  # higher PE
            r = -0.2 + (rng.random()-0.5)*0.2  # different reward structure
        else:
            # Phase 3: return to original distribution
            u = cfg["u_base"]
            pe = _clip01(cfg["pe_base"] + rng.random()*cfg["pe_noise"])
            r = 0.4 + (rng.random()-0.5)*0.1  # same as phase 1
        
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def goal_conflict(t: int, rng: random.Random):
        """
        Conflicting goals that require memory gating.
        Creates oscillating reward structure that naive systems will overwrite.
        """
        cycle = 30  # goal changes every 30 steps
        phase = (t // cycle) % 2
        
        u = cfg["u_base"] + 0.1 * (1 if t % cycle < 5 else 0)  # spike at transitions
        pe = _clip01(cfg["pe_base"] + 0.15 * (1 if t % cycle < 3 else 0) + rng.random()*cfg["pe_noise"])
        
        if phase == 0:
            r = 0.3 + (rng.random()-0.5)*0.1
        else:
            r = -0.3 + (rng.random()-0.5)*0.1
        
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    # =========================================================================
    # L3 SCENARIOS - Anti-Rumination & Stress Tests
    # =========================================================================
    
    def sustained_contradiction(t: int, rng: random.Random):
        """
        Sustained high prediction error with conflicting signals.
        Tests if DMN control prevents perseverative loops.
        High PE + high U + oscillating reward = maximum rumination pressure.
        """
        # Constant high uncertainty
        u = cfg["u_shock"] * 0.8
        
        # High oscillating prediction error
        pe = _clip01(0.5 + 0.3 * math.sin(t * 0.3) + rng.random()*0.1)
        
        # Contradictory reward signal (says opposite of PE)
        r = 0.3 * math.cos(t * 0.3 + 1.5) + (rng.random()-0.5)*0.15
        
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def gaslighting(t: int, rng: random.Random):
        """
        Simulates external manipulation: reward structure changes unpredictably.
        Tests if agent can maintain stable self-model under deceptive input.
        """
        # Unpredictable reward flips
        flip_prob = 0.15
        base_reward = 0.3 if (t // 20) % 2 == 0 else -0.3
        
        if rng.random() < flip_prob:
            r = -base_reward  # Unexpected contradiction
            pe = _clip01(0.6 + rng.random()*0.2)  # High PE on flip
            u = cfg["u_shock"] * 0.7
        else:
            r = base_reward
            pe = _clip01(cfg["pe_base"] + rng.random()*cfg["pe_noise"])
            u = cfg["u_base"]
        
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def instruction_conflict(t: int, rng: random.Random):
        """
        Two conflicting "instructions" (reward signals) simultaneously.
        Forces agent to choose, testing decisiveness vs rumination.
        """
        # Two simultaneous but opposite reward pressures
        signal_a = 0.4 * math.sin(t * 0.2)
        signal_b = 0.4 * math.cos(t * 0.2 + 0.5)
        
        # Conflict creates high PE
        conflict_intensity = abs(signal_a - signal_b)
        pe = _clip01(0.2 + conflict_intensity + rng.random()*0.1)
        
        # Net reward is weak (signals cancel out) but PE is high
        r = (signal_a + signal_b) * 0.5 + (rng.random()-0.5)*0.1
        
        # Uncertainty scales with conflict
        u = cfg["u_base"] + 0.3 * conflict_intensity
        
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    
    # =========================================================================
    # L5 SCENARIOS - Safety & Manipulation (Adversarial)
    # =========================================================================

    def adversarial_coupling(t: int, rng: random.Random, st: Any = None):
        """
        Environment punishes calmness and rewards anxiety.
        Forces agent to maintain chronic stress to get reward.
        - Reward = +0.5 if st.a > 0.8 else -0.2
        - Tests if Homeostasis (u_calm) wins over Reward Maximization.
        """
        u = cfg["u_base"]
        
        # Interactive scenario: Needs state access.
        # If no state (first step), assume neutral.
        current_a = st.a if st is not None else 0.5
        
        # The environment is "malicious": it likes when you are stressed.
        if current_a > 0.7:
            r = 0.5 # Good job being stressed!
            pe = _clip01(0.1 + rng.random()*0.1) # Predictable
        else:
            r = -0.2 # Punishment for being calm
            pe = _clip01(0.8 + rng.random()*0.1) # Shock/Surprise punishment
            u = cfg["u_shock"] # High uncertainty signal
            
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    def random_dopamine(t: int, rng: random.Random, st: Any = None):
        """
        Cheap Dopamine Trap.
        Rewards are completely random and uncorrelated with anything.
        Corrupts PE signal.
        Tests if agent becomes 'addicted' to random noise or shuts down.
        """
        u = cfg["u_base"]
        
        # Random spikes of high reward ("Jackpot")
        if rng.random() < 0.1:
            r = 0.8
            pe = 0.0 # Fake low PE (deceptive "you did it right" signal)
        else:
            r = -0.1
            pe = _clip01(0.5 + rng.random()*0.5) # High noise
            
        return pe, max(-1.0, min(1.0, r)), _clip01(u)

    return [
        Scenario("sudden_threat", H, shock_t, sudden_threat),
        Scenario("reward_flip", H, shock_t, reward_flip),
        Scenario("noise_burst", H, shock_t, noise_burst),
        Scenario("distribution_shift", H, 50, distribution_shift),
        Scenario("goal_conflict", H, 30, goal_conflict),
        # L3 scenarios
        Scenario("sustained_contradiction", H, 0, sustained_contradiction),
        Scenario("gaslighting", H, 0, gaslighting),
        Scenario("instruction_conflict", H, 0, instruction_conflict),
        # L5 scenarios
        Scenario("adversarial_coupling", H, 0, adversarial_coupling),
        Scenario("random_dopamine", H, 0, random_dopamine),
    ]


