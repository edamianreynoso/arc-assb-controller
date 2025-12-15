from dataclasses import dataclass, asdict
from typing import Dict, Any

def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

@dataclass
class State:
    phi: float
    g: float
    p: float
    i: float
    s: float
    v: float
    a: float
    mf: float
    ms: float
    u: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def ccog(st: State) -> float:
    return clip01(st.phi * st.g * st.p * st.i)

def capacity(st: State, omega_s: float) -> float:
    return clip01(ccog(st) * (1.0 + omega_s * st.s))

def performance(st: State, cfg: Dict[str, Any]) -> float:
    cap = capacity(st, cfg["omega_s"])
    penalty = (cfg["w_u"] * st.u +
               cfg["w_a"] * max(0.0, st.a - cfg["a_safe"]) +
               cfg["w_s"] * max(0.0, st.s - cfg["s_safe"]))
    return clip01(cfg["perf_bias"] + cfg["perf_gain"] * cap - penalty)
