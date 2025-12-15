from typing import Dict, Any
from .state import State, clip01

def step_dynamics(st: State, pe: float, reward: float, u_exog: float, control: Dict[str, float], cfg: Dict[str, Any]) -> State:
    u_dmg  = control.get("u_dmg", 0.0)
    u_att  = control.get("u_att", 0.0)
    u_mem  = control.get("u_mem", 1.0)
    u_calm = control.get("u_calm", 0.0)
    u_reapp= control.get("u_reapp", 0.0)

    u_eff = clip01(u_exog * (1.0 - cfg["k_u_att"] * u_att))

    i_next = clip01(st.i + cfg["k_i_att"] * u_att - cfg["mu_i"] * (st.i - cfg["i0"]) - cfg["k_i_u"] * u_eff)
    p_next = clip01(st.p - cfg["k_p_pe"] * pe - cfg["k_p_u"] * u_eff + cfg["k_p_i"] * i_next + cfg["mu_p"] * (cfg["p0"] - st.p))
    g_next = clip01(st.g + cfg["k_g_i"] * i_next + cfg["k_g_p"] * p_next - cfg["k_g_u"] * u_eff - cfg["k_g_a"] * max(0.0, st.a - cfg["a_safe"]) + cfg["mu_g"] * (cfg["g0"] - st.g))
    phi_next = clip01(st.phi + cfg["k_phi_gp"] * (g_next * p_next) - cfg["mu_phi"] * (st.phi - cfg["phi0"]))

    s_drive = cfg["k_s_u"] * u_eff + cfg["k_s_pe"] * pe
    s_next = clip01(st.s + s_drive - cfg["mu_s"] * (st.s - cfg["s0"]) - cfg["k_s_dmg"] * u_dmg)

    a_next = clip01(st.a + cfg["k_a_pe"] * pe + cfg["k_a_u"] * u_eff + cfg["k_a_s"] * max(0.0, s_next - cfg["s_safe"])
                    - cfg["mu_a"] * (st.a - cfg["a0"]) - cfg["k_a_calm"] * u_calm)

    v_next = clip01(st.v + cfg["k_v_r"] * (0.5 * (reward + 1.0)) - cfg["k_v_pe"] * pe - cfg["k_v_u"] * u_eff
                    - cfg["mu_v"] * (st.v - cfg["v0"]) + cfg["k_v_reapp"] * u_reapp)

    priority = clip01(cfg["w_mem_pe"] * pe + cfg["w_mem_a"] * abs(a_next - cfg["a0"]) + cfg["w_mem_v"] * abs(v_next - cfg["v0"]))
    write = priority * u_mem

    eta = cfg["eta0"] * clip01(1.0 + cfg["k_eta_a"] * max(0.0, a_next - cfg["a_safe"]))
    mf_next = clip01(st.mf + eta * write - cfg["mu_mf"] * (st.mf - cfg["mf0"]))
    ms_next = clip01(st.ms + cfg["k_ms"] * mf_next - cfg["mu_ms"] * (st.ms - cfg["ms0"]))

    return State(phi=phi_next, g=g_next, p=p_next, i=i_next, s=s_next, v=v_next, a=a_next, mf=mf_next, ms=ms_next, u=u_eff)
