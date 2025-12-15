from typing import Dict, Any, List
import math

def control_effort(control: List[Dict[str, float]]) -> float:
    """
    Proxy de "costo metabólico" del control.

    Promedia por-step la magnitud de intervención del controlador.
    - `u_dmg`, `u_att`, `u_calm`, `u_reapp`: esfuerzo directo [0,1]
    - `u_mem`: 1 significa "sin bloqueo", así que esfuerzo = (1 - u_mem)
    """
    if not control:
        return 0.0
    total = 0.0
    n = 0
    for u in control:
        if not isinstance(u, dict):
            continue
        u_dmg = float(u.get("u_dmg", 0.0))
        u_att = float(u.get("u_att", 0.0))
        u_calm = float(u.get("u_calm", 0.0))
        u_reapp = float(u.get("u_reapp", 0.0))
        u_mem = float(u.get("u_mem", 1.0))
        total += abs(u_dmg) + abs(u_att) + abs(u_calm) + abs(u_reapp) + abs(1.0 - u_mem)
        n += 1
    return total / max(1, n)

def recovery_time(perf: List[float], a: List[float], shock_t: int, cfg: Dict[str, Any]) -> int:
    """
    Calcula el tiempo de recuperación post-shock.
    Retorna pasos hasta que performance vuelve a baseline y arousal baja.
    Si el sistema estaba pre-colapsado (perf < 0.2 antes del shock), retorna rt_max.
    """
    w = cfg["baseline_window"]
    rt_max = cfg.get("rt_max", len(perf) - shock_t)
    
    pre_start = max(0, shock_t - w)
    baseline = sum(perf[pre_start:shock_t]) / max(1, (shock_t - pre_start))
    
    # Si el sistema ya estaba colapsado antes del shock, marcar como no recuperable
    if baseline < 0.20:
        return rt_max
    
    target_low = max(0.0, baseline - cfg["rt_eps"])
    target_high = min(1.0, baseline + cfg["rt_eps"])
    a_safe = cfg["a_safe"]
    
    for t in range(shock_t, len(perf)):
        if target_low <= perf[t] <= target_high and a[t] <= a_safe + cfg["rt_a_eps"]:
            return t - shock_t
    return rt_max

def rt_normalized(rt: int, cfg: Dict[str, Any]) -> float:
    """RT normalizado a [0,1] donde 0=recuperación instantánea, 1=nunca recupera."""
    rt_max = cfg.get("rt_max", 100)
    return min(1.0, rt / rt_max)

def overshoot(a: List[float], cfg: Dict[str, Any]) -> float:
    return max(0.0, max(a) - cfg["a_safe"])

def rumination_index(s: List[float], cfg: Dict[str, Any]) -> float:
    tau = cfg["s_rum_tau"]
    above = [1 if x > tau else 0 for x in s]
    frac = sum(above) / max(1, len(above))
    runs, current = [], 0
    for v in above:
        if v == 1:
            current += 1
        elif current > 0:
            runs.append(current); current = 0
    if current > 0:
        runs.append(current)
    persistence = (sum(runs) / max(1, len(runs))) / max(1, len(s)) if runs else 0.0
    return frac + cfg["ri_persistence_weight"] * persistence

def stability_post_shock(perf: List[float], shock_t: int) -> float:
    """Varianza de performance después del shock."""
    post = perf[shock_t:]
    if len(post) < 2:
        return 0.0
    m = sum(post) / len(post)
    return math.sqrt(sum((x - m)**2 for x in post) / len(post))

def narrative_dominance_ratio(s: List[float], perf: List[float], shock_t: int, cfg: Dict[str, Any]) -> float:
    """
    NDR: Narrative Dominance Ratio.
    Mide cuánto domina S(t) sobre la evidencia (performance) post-shock.
    
    Fórmula: fracción de tiempo donde S está alto pero performance no mejora.
    NDR alto = narrativa dominando sin mejora funcional (alucinación/rumiación).
    """
    s_safe = cfg.get("s_safe", 0.55)
    post_s = s[shock_t:]
    post_perf = perf[shock_t:]
    
    if len(post_s) < 2:
        return 0.0
    
    # Detectar "dominancia narrativa sin mejora"
    # S alto + performance no mejorando (derivada <= 0)
    dominance_count = 0
    for i in range(1, len(post_s)):
        s_high = post_s[i] > s_safe
        perf_improving = post_perf[i] > post_perf[i-1] + 0.01  # umbral pequeño
        
        if s_high and not perf_improving:
            dominance_count += 1
    
    return dominance_count / max(1, len(post_s) - 1)

# =============================================================================
# L2 METRICS - Memory & Continual Learning
# =============================================================================

def retention_index(perf: List[float], mf: List[float], phase1_end: int = 50, phase3_start: int = 100) -> float:
    """
    Mide cuánto se retiene del aprendizaje de fase 1 después de fase 2.
    Compara performance en fase 3 vs fase 1.
    Retention = 1.0 si performance en fase 3 >= fase 1.
    """
    if len(perf) < phase3_start + 10:
        return 0.0
    
    phase1_perf = sum(perf[10:phase1_end]) / max(1, phase1_end - 10)  # skip warm-up
    phase3_perf = sum(perf[phase3_start:phase3_start+50]) / 50
    
    if phase1_perf < 0.1:
        return 0.0
    
    return min(1.0, phase3_perf / phase1_perf)

def adaptation_speed(perf: List[float], phase2_start: int = 50, window: int = 20) -> float:
    """
    Mide qué tan rápido el sistema se adapta a la nueva distribución (fase 2).
    Retorna pasos hasta alcanzar 80% del máximo en fase 2.
    """
    phase2_perf = perf[phase2_start:]
    if len(phase2_perf) < window:
        return float(len(phase2_perf))
    
    max_phase2 = max(phase2_perf[:50]) if len(phase2_perf) >= 50 else max(phase2_perf)
    target = 0.8 * max_phase2
    
    for i, p in enumerate(phase2_perf):
        if p >= target:
            return float(i)
    return float(len(phase2_perf))

def memory_stability(mf: List[float], ms: List[float]) -> float:
    """
    Mide estabilidad de memoria: baja varianza = mejor consolidación.
    Retorna 1 - normalized_variance.
    """
    if len(mf) < 2:
        return 1.0
    
    m = sum(mf) / len(mf)
    var = sum((x - m)**2 for x in mf) / len(mf)
    # Normalizar: var > 0.1 es inestable
    return max(0.0, 1.0 - var * 10)

def compute_metrics(trace: Dict[str, List[float]], shock_t: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    perf = trace["perf"]; a = trace["a"]; s = trace["s"]
    mf = trace.get("mf", [0.0]*len(perf))
    ms = trace.get("ms", [0.0]*len(perf))
    ctrl = trace.get("control", [])
    
    mean_perf = sum(perf)/max(1,len(perf))
    m = mean_perf
    v = sum((x-m)**2 for x in perf)/max(1,len(perf))
    
    rt = recovery_time(perf, a, shock_t, cfg)
    
    return {
        "RT": float(rt),
        "RT_norm": float(rt_normalized(rt, cfg)),
        "Overshoot": float(overshoot(a, cfg)),
        "RI": float(rumination_index(s, cfg)),
        "NDR": float(narrative_dominance_ratio(s, perf, shock_t, cfg)),
        "ControlEffort": float(control_effort(ctrl)),
        "PerfMean": float(mean_perf),
        "PerfStd": float(math.sqrt(v)),
        "StabilityPost": float(stability_post_shock(perf, shock_t)),
        # L2 metrics
        "Retention": float(retention_index(perf, mf)),
        "AdaptSpeed": float(adaptation_speed(perf)),
        "MemStability": float(memory_stability(mf, ms)),
    }

