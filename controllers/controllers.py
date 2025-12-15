from typing import Dict, Any
from sim.state import State

class NoControl:
    name = "no_control"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        return {"u_dmg":0.0,"u_att":0.0,"u_mem":1.0,"u_calm":0.0,"u_reapp":0.0}

class NaiveCalm:
    name = "naive_calm"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        a_safe = cfg["a_safe"]
        u_calm = min(1.0, max(0.0, (st.a - a_safe) / max(1e-6, (1.0 - a_safe))))
        return {"u_dmg":0.0,"u_att":0.0,"u_mem":1.0,"u_calm":u_calm,"u_reapp":0.0}

class ARCv1:
    name = "arc_v1"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        risk = (cfg["arc_w_u"] * st.u +
                cfg["arc_w_a"] * max(0.0, st.a - cfg["a_safe"]) +
                cfg["arc_w_s"] * max(0.0, st.s - cfg["s_safe"]))
        risk = max(0.0, min(1.0, risk))
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk)
        u_calm = min(1.0, cfg["arc_k_calm"] * max(0.0, st.a - cfg["a_safe"]))
        u_reapp = min(1.0, cfg["arc_k_reapp"] * st.u * (1.0 - risk))
        return {"u_dmg":u_dmg,"u_att":u_att,"u_mem":u_mem,"u_calm":u_calm,"u_reapp":u_reapp}

class PerfOptimized:
    """Baseline competitivo: maximiza performance sin regular afecto."""
    name = "perf_optimized"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Alta atención constante, sin regulación de DMN ni arousal
        u_att = cfg.get("perf_opt_att", 0.70)
        return {"u_dmg":0.0,"u_att":u_att,"u_mem":1.0,"u_calm":0.0,"u_reapp":0.0}

# ============================================================================
# ABLATION CONTROLLERS - Para estudiar contribución de cada componente de ARC
# ============================================================================

class ARC_NoDMG:
    """Ablation: ARC sin control de DMN (g_dmg = 0)."""
    name = "arc_no_dmg"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        risk = (cfg["arc_w_u"] * st.u +
                cfg["arc_w_a"] * max(0.0, st.a - cfg["a_safe"]) +
                cfg["arc_w_s"] * max(0.0, st.s - cfg["s_safe"]))
        risk = max(0.0, min(1.0, risk))
        u_dmg = 0.0  # ABLATED
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk)
        u_calm = min(1.0, cfg["arc_k_calm"] * max(0.0, st.a - cfg["a_safe"]))
        u_reapp = min(1.0, cfg["arc_k_reapp"] * st.u * (1.0 - risk))
        return {"u_dmg":u_dmg,"u_att":u_att,"u_mem":u_mem,"u_calm":u_calm,"u_reapp":u_reapp}

class ARC_NoCalm:
    """Ablation: ARC sin control de arousal (g_calm = 0)."""
    name = "arc_no_calm"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        risk = (cfg["arc_w_u"] * st.u +
                cfg["arc_w_a"] * max(0.0, st.a - cfg["a_safe"]) +
                cfg["arc_w_s"] * max(0.0, st.s - cfg["s_safe"]))
        risk = max(0.0, min(1.0, risk))
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk)
        u_calm = 0.0  # ABLATED
        u_reapp = min(1.0, cfg["arc_k_reapp"] * st.u * (1.0 - risk))
        return {"u_dmg":u_dmg,"u_att":u_att,"u_mem":u_mem,"u_calm":u_calm,"u_reapp":u_reapp}

class ARC_NoMem:
    """Ablation: ARC sin gating de memoria (g_mem = 1 siempre)."""
    name = "arc_no_mem"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        risk = (cfg["arc_w_u"] * st.u +
                cfg["arc_w_a"] * max(0.0, st.a - cfg["a_safe"]) +
                cfg["arc_w_s"] * max(0.0, st.s - cfg["s_safe"]))
        risk = max(0.0, min(1.0, risk))
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0  # ABLATED - no memory gating
        u_calm = min(1.0, cfg["arc_k_calm"] * max(0.0, st.a - cfg["a_safe"]))
        u_reapp = min(1.0, cfg["arc_k_reapp"] * st.u * (1.0 - risk))
        return {"u_dmg":u_dmg,"u_att":u_att,"u_mem":u_mem,"u_calm":u_calm,"u_reapp":u_reapp}

class ARC_NoReapp:
    """Ablation: ARC sin reappraisal (g_reapp = 0)."""
    name = "arc_no_reapp"
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        risk = (cfg["arc_w_u"] * st.u +
                cfg["arc_w_a"] * max(0.0, st.a - cfg["a_safe"]) +
                cfg["arc_w_s"] * max(0.0, st.s - cfg["s_safe"]))
        risk = max(0.0, min(1.0, risk))
        u_dmg = min(1.0, cfg["arc_k_dmg"] * risk)
        u_att = min(1.0, cfg["arc_k_att"] * st.u * (1.0 - max(0.0, st.a - cfg["a_safe"])))
        u_mem = 1.0 - min(1.0, cfg["arc_k_mem_block"] * risk)
        u_calm = min(1.0, cfg["arc_k_calm"] * max(0.0, st.a - cfg["a_safe"]))
        u_reapp = 0.0  # ABLATED
        return {"u_dmg":u_dmg,"u_att":u_att,"u_mem":u_mem,"u_calm":u_calm,"u_reapp":u_reapp}

# =============================================================================
# L4 - HIERARCHICAL MULTI-SCALE CONTROL
# =============================================================================

class ARCv2_Hierarchical:
    """
    ARCv2: Control jerárquico multi-escala.
    
    Tres niveles de control operando a diferentes escalas temporales:
    - FAST (τ~1): Estabiliza arousal (E) - respuesta inmediata
    - MEDIUM (τ~5): Regula narrativa (S) vs evidencia - adaptación
    - SLOW (τ~20): Ajusta setpoints y metas - meta-regulación
    
    Cada nivel puede modular los parámetros del nivel inferior.
    """
    name = "arc_v2_hier"
    
    def __init__(self):
        # Estado interno del controlador jerárquico
        self.slow_a_setpoint = 0.4      # Setpoint adaptativo de arousal
        self.slow_s_setpoint = 0.3      # Setpoint adaptativo de DMN
        self.slow_perf_baseline = 0.9   # Performance baseline esperado
        self.slow_counter = 0           # Contador para updates lentos
        self.medium_counter = 0         # Contador para updates medios
        self.last_medium_output = {"u_dmg": 0.0, "u_reapp": 0.0} # Zero-order hold state
        
        # Historial para el nivel lento
        self.perf_history = []
        self.a_history = []
        self.s_history = []
    
    def _fast_control(self, st: State, cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Nivel FAST (τ~1): Estabiliza arousal inmediatamente.
        Opera cada timestep. Respuesta reactiva pura.
        """
        a_error = max(0.0, st.a - self.slow_a_setpoint)
        
        # Control proporcional de arousal
        u_calm = min(1.0, cfg.get("arc_k_calm", 1.5) * a_error)
        
        # Atención modulada por arousal inverso
        u_att = min(1.0, cfg.get("arc_k_att", 0.8) * st.u * max(0.0, 1.0 - a_error))
        
        return {"u_calm": u_calm, "u_att": u_att}
    
    def _medium_control(self, st: State, cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Nivel MEDIUM (τ~5): Regula narrativa vs evidencia.
        Opera cada ~5 pasos. Previene rumiación.
        """
        s_safe = cfg.get("s_safe", 0.55)
        s_excess = max(0.0, st.s - s_safe)
        s_error = s_excess / max(1e-6, (1.0 - s_safe))
        
        # Control de DMN gain proporcional al exceso narrativo (normalizado)
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * 1.6 * s_error)
        
        # Reappraisal cuando hay alta incertidumbre + alta narrativa
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.55) * st.u * min(1.0, s_error))
        
        return {"u_dmg": u_dmg, "u_reapp": u_reapp}
    
    def _slow_control(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]):
        """
        Nivel SLOW (τ~20): Ajusta setpoints y metas.
        Opera cada ~20 pasos. Meta-regulación adaptativa.
        """
        # Calcular performance actual
        current_perf = obs.get("perf", 0.5)
        self.perf_history.append(current_perf)
        self.a_history.append(st.a)
        self.s_history.append(st.s)
        
        # Mantener solo últimos 20 puntos
        if len(self.perf_history) > 20:
            self.perf_history = self.perf_history[-20:]
            self.a_history = self.a_history[-20:]
            self.s_history = self.s_history[-20:]
        
        if len(self.perf_history) < 10:
            return  # No hay suficiente historia
        
        # Estadísticas recientes
        mean_perf = sum(self.perf_history) / len(self.perf_history)
        mean_a = sum(self.a_history) / len(self.a_history)
        mean_s = sum(self.s_history) / len(self.s_history)
        
        # Ajustar setpoints basándose en desempeño
        if mean_perf < self.slow_perf_baseline * 0.8:
            # Performance bajo: bajar setpoints (ser más estricto)
            self.slow_a_setpoint = max(0.2, self.slow_a_setpoint - 0.02)
            self.slow_s_setpoint = max(0.2, self.slow_s_setpoint - 0.02)
        elif mean_perf > self.slow_perf_baseline * 0.95:
            # Performance bueno: relajar setpoints (permitir más rango)
            self.slow_a_setpoint = min(0.6, self.slow_a_setpoint + 0.01)
            self.slow_s_setpoint = min(0.5, self.slow_s_setpoint + 0.01)
        
        # Actualizar baseline de performance (expectativa adaptativa)
        self.slow_perf_baseline = 0.95 * self.slow_perf_baseline + 0.05 * mean_perf
    
    def _compute_memory_gate(self, st: State, risk: float, cfg: Dict[str, Any]) -> float:
        """Memory gating basado en risk combinado de todos los niveles."""
        return 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.3) * risk)
    
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Calcular risk global
        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.3) * max(0.0, st.a - cfg.get("a_safe", 0.6)) +
                cfg.get("arc_w_s", 0.3) * max(0.0, st.s - cfg.get("s_safe", 0.55)))
        risk = max(0.0, min(1.0, risk))
        
        # Nivel FAST: cada paso
        fast = self._fast_control(st, cfg)
        
        # Nivel MEDIUM: cada 5 pasos (o siempre si presión narrativa alta)
        self.medium_counter += 1
        narrative_pressure = max(0.0, st.s - max(cfg.get("s_safe", 0.55), self.slow_s_setpoint))
        if self.medium_counter >= 5 or risk > 0.5 or narrative_pressure > 0.02:
            self.last_medium_output = self._medium_control(st, cfg)
            self.medium_counter = 0
        
        # Nivel SLOW: cada 20 pasos
        self.slow_counter += 1
        if self.slow_counter >= 20:
            self._slow_control(st, obs, cfg)
            self.slow_counter = 0
        
        # Combinar outputs de todos los niveles
        u_mem = self._compute_memory_gate(st, risk, cfg)
        
        return {
            "u_dmg": self.last_medium_output["u_dmg"],
            "u_att": fast["u_att"],
            "u_mem": u_mem,
            "u_calm": fast["u_calm"],
            "u_reapp": self.last_medium_output["u_reapp"]
        }
# =============================================================================
# L4-REV2 - META-CONTROL (NEUROMODULATION)
# =============================================================================

class ARCv3_MetaControl:
    """
    ARCv3: Meta-Control Neuronal (Eficiencia Energética).
    
    Inspirado en el sistema Locus Coeruleus - Norepinefrina (LC-NE).
    - Fast Loop (Tonic): Control reactivo base (como ARCv1).
    - Slow Loop (Phasic/Meta): Modula las GANANCIAS (K) globales.
    
    Hipótesis:
    - High Performance + Stability -> Bajar K (Relax, save energy)
    - Low Performance / Shock -> Subir K (Alert, high control)
    """
    name = "arc_v3_meta"
    
    def __init__(self):
        self.current_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0

        # Meta-parámetros (gain scheduling)
        self.target_perf = 0.90
        self.gain_decay = 0.01
        self.gain_boost = 0.05
        self.gain_min = 0.80
        self.gain_max = 1.40
    
    def _update_meta_state(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]):
        """Actualiza el nivel de 'alertness' (ganancia global) cada ~20 pasos."""
        current_perf = float(obs.get("perf", 0.5))
        self.perf_history.append(current_perf)
        if len(self.perf_history) > 20:
            self.perf_history.pop(0)
            
        mean_perf = sum(self.perf_history) / len(self.perf_history)

        s_safe = cfg.get("s_safe", 0.55)
        s_tau = cfg.get("s_rum_tau", s_safe)
        a_safe = cfg.get("a_safe", 0.60)
        a_excess = max(0.0, st.a - a_safe)
        s_excess = max(0.0, st.s - s_safe)
        s_error = s_excess / max(1e-6, (1.0 - s_safe))
        s_rum_excess = max(0.0, st.s - s_tau)
        s_rum_error = s_rum_excess / max(1e-6, (1.0 - s_tau))

        # Risk proxy (includes narrative pressure explicitly)
        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.4) * a_excess +
                cfg.get("arc_w_s", 0.35) * s_excess +
                0.40 * max(s_error, s_rum_error))
        risk = max(0.0, min(1.0, risk))

        if mean_perf >= (self.target_perf - 0.02) and risk < 0.15:
            self.current_gain = max(self.gain_min, self.current_gain - self.gain_decay)
        elif mean_perf < (self.target_perf - 0.10) or risk > 0.45:
            self.current_gain = min(self.gain_max, self.current_gain + self.gain_boost)
        else:
            self.current_gain = max(self.gain_min, self.current_gain - self.gain_decay * 0.5)
            
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # 1. Meta-Control Update (Slow Loop)
        self.slow_counter += 1
        if self.slow_counter >= 20:
            self._update_meta_state(st, obs, cfg)
            self.slow_counter = 0
            
        # 2. Modulación de Parámetros (Gain Scheduling)
        # Aplicamos la ganancia actual a las constantes base del config
        # Do not relax DMN suppression below baseline: it is safety-critical for anti-rumination.
        k_dmg = cfg.get("arc_k_dmg", 0.95) * max(1.0, self.current_gain)
        k_calm = cfg.get("arc_k_calm", 0.85) * self.current_gain
        # Atención suele subir con arousal, aquí la modulamos también
        k_att = cfg.get("arc_k_att", 0.75) * self.current_gain
        k_reapp = cfg.get("arc_k_reapp", 0.55) * self.current_gain
        k_mem = cfg.get("arc_k_mem_block", 0.90) * self.current_gain
        
        # 3. Fast Loop (Reactive Control - Zero Latency)
        # Misma lógica que ARCv1 pero con K dinámicas
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        s_tau = cfg.get("s_rum_tau", s_safe)
        a_excess = max(0.0, st.a - a_safe)
        s_excess = max(0.0, st.s - s_safe)
        s_error = s_excess / max(1e-6, (1.0 - s_safe))
        s_rum_excess = max(0.0, st.s - s_tau)
        s_rum_error = s_rum_excess / max(1e-6, (1.0 - s_tau))
        a_error = a_excess / max(1e-6, (1.0 - a_safe))

        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.3) * a_excess +
                cfg.get("arc_w_s", 0.3) * s_excess)
        risk = max(0.0, min(1.0, risk))
        
        # Acciones calculadas con ganancia modulada
        u_dmg = min(1.0, k_dmg * (risk + 1.2 * s_error + 10.0 * s_rum_error))
        u_att = min(1.0, k_att * st.u * max(0.0, 1.0 - a_excess))
        u_mem = 1.0 - min(1.0, k_mem * (risk + 0.5 * max(s_error, s_rum_error)))
        u_calm = min(1.0, k_calm * a_error)
        u_reapp = min(1.0, k_reapp * st.u * min(1.0, max(s_error, s_rum_error)))
        
        # Guardamos gain en debug (opcional, hacky way: return in dict if needed by metric)
        # Por ahora standard output
        return {"u_dmg":u_dmg, "u_att":u_att, "u_mem":u_mem, "u_calm":u_calm, "u_reapp":u_reapp}


# =============================================================================
# ARC-PID: PROPORCIONAL-INTEGRAL-DERIVATIVO CONTROLLER
# =============================================================================

class ARCv1_PID:
    """
    ARC v1 with PID (Proportional-Integral-Derivative) control.
    
    Improves over pure proportional control by:
    - Integral term: Eliminates steady-state error
    - Derivative term: Anticipates changes, reduces overshoot
    
    Includes anti-windup to prevent integral saturation.
    """
    name = "arc_v1_pid"
    
    def __init__(self):
        # PID state (per control channel)
        self.integral_risk = 0.0
        self.integral_arousal = 0.0
        self.integral_narrative = 0.0
        self.prev_risk = 0.0
        self.prev_arousal = 0.0
        self.prev_narrative = 0.0
        
    def _pid_control(self, error: float, prev_error: float, integral: float,
                    k_p: float, k_i: float, k_d: float, dt: float = 1.0) -> tuple:
        """
        Compute PID output with anti-windup.
        Returns (output, new_integral).
        """
        # Proportional
        P = k_p * error
        
        # Integral with anti-windup clamp
        new_integral = integral + k_i * error * dt
        new_integral = max(-1.0, min(1.0, new_integral))  # Anti-windup
        I = new_integral
        
        # Derivative (filtered)
        D = k_d * (error - prev_error) / dt
        
        # Total output (clamped)
        output = max(0.0, min(1.0, P + I + D))
        
        return output, new_integral
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        
        # Compute errors (deviations from safe region)
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        
        # Composite risk signal (same as v1)
        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.3) * a_error +
                cfg.get("arc_w_s", 0.35) * s_error)
        risk = max(0.0, min(1.0, risk))
        
        # PID gains from config
        k_p = cfg.get("pid_k_p", 0.80)
        k_i = cfg.get("pid_k_i", 0.15)
        k_d = cfg.get("pid_k_d", 0.25)
        
        # PID control for risk (main signal)
        pid_output, self.integral_risk = self._pid_control(
            error=risk,
            prev_error=self.prev_risk,
            integral=self.integral_risk,
            k_p=k_p, k_i=k_i, k_d=k_d
        )
        self.prev_risk = risk
        
        # PID control for arousal (separate channel)
        pid_arousal, self.integral_arousal = self._pid_control(
            error=a_error,
            prev_error=self.prev_arousal,
            integral=self.integral_arousal,
            k_p=k_p * 1.2, k_i=k_i * 0.8, k_d=k_d * 1.5  # Faster arousal response
        )
        self.prev_arousal = a_error
        
        # PID control for narrative (DMN)
        pid_narrative, self.integral_narrative = self._pid_control(
            error=s_error,
            prev_error=self.prev_narrative,
            integral=self.integral_narrative,
            k_p=k_p * 1.0, k_i=k_i * 1.2, k_d=k_d * 0.8  # Stronger integral for rumination
        )
        self.prev_narrative = s_error
        
        # Map PID outputs to control actions
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * (pid_output + 0.5 * pid_narrative))
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * st.u * (1.0 - a_error))
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * pid_output)
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * pid_arousal)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - pid_output))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        """Reset PID state between episodes."""
        self.integral_risk = 0.0
        self.integral_arousal = 0.0
        self.integral_narrative = 0.0
        self.prev_risk = 0.0
        self.prev_arousal = 0.0
        self.prev_narrative = 0.0


# =============================================================================
# ARCv3_PID_Meta: HYBRID PID + META-CONTROL (Best of Both Worlds)
# =============================================================================

class ARCv3_PID_Meta:
    """
    ARCv3 with PID: Combines PID control with adaptive gain scheduling.
    
    - PID: Provides precise error correction with integral (steady-state) and 
           derivative (anticipation) terms
    - Meta-Control: Adaptively modulates control effort based on performance
    
    Hypothesis: This combination should achieve:
    - Lower overshoot than pure P (from derivative term)
    - Zero steady-state error (from integral term)
    - Lower control effort when stable (from gain scheduling)
    """
    name = "arc_v3_pid_meta"
    
    def __init__(self):
        # PID state
        self.integral_risk = 0.0
        self.prev_risk = 0.0
        
        # Meta-control state (from ARCv3_MetaControl)
        self.current_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0
        
        # Meta-parameters
        self.target_perf = 0.90
        self.gain_decay = 0.015
        self.gain_boost = 0.06
        self.gain_min = 0.70  # Can relax more than pure meta
        self.gain_max = 1.30
    
    def _update_meta_gain(self, perf: float, risk: float, cfg: dict):
        """Adaptive gain scheduling based on performance and risk."""
        self.perf_history.append(perf)
        if len(self.perf_history) > 20:
            self.perf_history.pop(0)
        
        mean_perf = sum(self.perf_history) / len(self.perf_history)
        
        # If performing well and low risk, relax control
        if mean_perf > self.target_perf and risk < 0.2:
            self.current_gain = max(self.gain_min, self.current_gain - self.gain_decay)
        # If performing poorly or high risk, intensify
        elif mean_perf < self.target_perf - 0.10 or risk > 0.5:
            self.current_gain = min(self.gain_max, self.current_gain + self.gain_boost)
    
    def _pid_control(self, error: float, k_p: float, k_i: float, k_d: float) -> float:
        """PID with anti-windup, modulated by current_gain."""
        # Adaptive gains
        P = k_p * self.current_gain * error
        
        # Integral with anti-windup
        self.integral_risk += k_i * self.current_gain * error
        self.integral_risk = max(-0.5, min(0.5, self.integral_risk))
        I = self.integral_risk
        
        # Derivative
        D = k_d * self.current_gain * (error - self.prev_risk)
        self.prev_risk = error
        
        return max(0.0, min(1.0, P + I + D))
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        
        # Compute errors
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        
        # Composite risk signal
        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.3) * a_error +
                cfg.get("arc_w_s", 0.35) * s_error)
        risk = max(0.0, min(1.0, risk))
        
        # Meta-control update (every 20 steps)
        self.slow_counter += 1
        if self.slow_counter >= 20:
            perf = obs.get("perf", 0.5)
            self._update_meta_gain(perf, risk, cfg)
            self.slow_counter = 0
        
        # PID gains (from config)
        k_p = cfg.get("pid_k_p", 0.80)
        k_i = cfg.get("pid_k_i", 0.15)
        k_d = cfg.get("pid_k_d", 0.25)
        
        # PID-controlled risk response
        pid_output = self._pid_control(risk, k_p, k_i, k_d)
        
        # Control actions with adaptive gains
        k_dmg = cfg.get("arc_k_dmg", 0.95) * max(1.0, self.current_gain)  # Never relax DMN
        k_calm = cfg.get("arc_k_calm", 0.85) * self.current_gain
        k_att = cfg.get("arc_k_att", 0.75) * self.current_gain
        
        u_dmg = min(1.0, k_dmg * (pid_output + 0.3 * s_error))
        u_att = min(1.0, k_att * st.u * (1.0 - a_error))
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * pid_output)
        u_calm = min(1.0, k_calm * a_error * (1.0 + pid_output))
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - pid_output))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        """Reset state between episodes."""
        self.integral_risk = 0.0
        self.prev_risk = 0.0
        self.current_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0


# =============================================================================
# LQR CONTROLLERS: OPTIMAL CONTROL
# =============================================================================

import numpy as np

class ARCv1_LQR:
    """
    ARC v1 with LQR (Linear Quadratic Regulator) control.
    
    LQR minimizes the cost function:
    J = Σ(x'Qx + u'Ru)
    
    Where:
    - x = [a_error, s_error, u] (state deviations)
    - u = [u_calm, u_dmg, u_att] (control signals)
    
    Optimal gains K computed via Discrete Algebraic Riccati Equation (DARE).
    """
    name = "arc_v1_lqr"
    
    def __init__(self):
        # OPTIMAL LQR Gains from Riccati solution (lqr_optimize.py)
        # Row 0: Gains for u_calm in response to [a_error, s_error, uncertainty]
        # Row 1: Gains for u_dmg in response to [a_error, s_error, uncertainty]
        # Row 2: Gains for u_att in response to [a_error, s_error, uncertainty]
        self.K = np.array([
            [1.5071, 0.0985, 0.4586],  # K_calm (negated for control)
            [0.0036, 2.0943, 0.0879],  # K_dmg (negated for control)
            [0.0380, -0.0197, 0.9244], # K_att
        ])
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        
        # State vector (errors from safe region)
        x = np.array([
            max(0.0, st.a - a_safe),  # a_error
            max(0.0, st.s - s_safe),  # s_error
            st.u                       # uncertainty
        ])
        
        # LQR control: u = K @ x (optimal state feedback)
        u_ctrl = self.K @ x
        
        # Scale and map to control actions
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * u_ctrl[0])
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * u_ctrl[1])
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u_ctrl[2])
        
        # Memory and reappraisal derived from risk level
        risk = max(u_ctrl[0], u_ctrl[1])
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * risk)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - min(1.0, x[0] + x[1])))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        pass  # LQR is stateless


class ARCv1_LQI:
    """
    ARC v1 with LQI (Linear Quadratic Integral) control.
    
    Combines:
    - LQR: Optimal state feedback (from Riccati solution)
    - Integral term: Specifically for narrative (S) to eliminate rumination
    
    This hybrid achieves:
    - High performance (from LQR)
    - Low overshoot (from LQR)
    - Zero rumination (from integral term on S)
    """
    name = "arc_v1_lqi"
    
    def __init__(self):
        # OPTIMAL LQR Gains from Riccati solution
        self.K = np.array([
            [1.5071, 0.0985, 0.4586],  # K_calm
            [0.0036, 2.0943, 0.0879],  # K_dmg
            [0.0380, -0.0197, 0.9244], # K_att
        ])
        
        # Integral state for narrative (S) - key for anti-rumination
        self.integral_s = 0.0
        self.ki_s = 0.25  # Integral gain for narrative error
        
        # Additional integral for arousal (optional)
        self.integral_a = 0.0
        self.ki_a = 0.10  # Lower gain for arousal
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        s_rum_tau = cfg.get("s_rum_tau", s_safe)
        
        # State vector (errors from safe region)
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        s_rum_error = max(0.0, st.s - s_rum_tau)  # Rumination-specific threshold
        
        x = np.array([a_error, s_error, st.u])
        
        # LQR control: u = K @ x (optimal state feedback)
        u_lqr = self.K @ x
        
        # Integral terms for eliminating steady-state error
        # Narrative integral (critical for anti-rumination)
        self.integral_s += self.ki_s * s_rum_error
        self.integral_s = max(0.0, min(1.0, self.integral_s))  # Anti-windup
        
        # Arousal integral
        self.integral_a += self.ki_a * a_error
        self.integral_a = max(0.0, min(0.5, self.integral_a))  # Conservative windup
        
        # Combined control: LQR + Integral
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * (u_lqr[0] + self.integral_a))
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * (u_lqr[1] + 1.5 * self.integral_s))
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u_lqr[2])
        
        # Memory gating based on combined risk
        risk = max(u_lqr[0], u_lqr[1], self.integral_s)
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * risk)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - min(1.0, x[0] + x[1])))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        """Reset integral states."""
        self.integral_s = 0.0
        self.integral_a = 0.0


class ARCv3_LQR_Meta:
    """
    ARCv3 with LQR + Meta-Control: Optimal control with adaptive gains.
    
    Combines:
    - LQR: Optimal state feedback control
    - Meta-Control: Adaptive R matrix (control cost) based on performance
    
    When performance is high, R increases (control becomes expensive),
    leading to more conservative control. When performance drops,
    R decreases (control is cheap), leading to aggressive intervention.
    """
    name = "arc_v3_lqr_meta"
    
    def __init__(self):
        # LQR state-feedback gains (base values)
        self.K_base = np.array([1.2, 0.9, 0.6])  # [arousal, narrative, uncertainty]
        
        # Adaptive R value (control cost)
        self.R_current = 1.0
        self.R_min = 0.5   # Cheap control when performance is low
        self.R_max = 2.0   # Expensive control when stable
        
        # Meta-control state
        self.current_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0
        self.target_perf = 0.90
        
    def _update_meta_control(self, perf: float, risk: float):
        """Adapt R based on performance (meta-level control of LQR)."""
        self.perf_history.append(perf)
        if len(self.perf_history) > 20:
            self.perf_history.pop(0)
        
        mean_perf = sum(self.perf_history) / len(self.perf_history)
        
        # Adaptive R: high perf -> high R (less control), low perf -> low R (more control)
        if mean_perf > self.target_perf:
            self.R_current = min(self.R_max, self.R_current * 1.05)
            self.current_gain = max(0.7, self.current_gain - 0.02)
        elif mean_perf < self.target_perf - 0.10 or risk > 0.4:
            self.R_current = max(self.R_min, self.R_current * 0.92)
            self.current_gain = min(1.3, self.current_gain + 0.04)
    
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # Safety thresholds
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        
        # State vector
        x = np.array([
            max(0.0, st.a - a_safe),
            max(0.0, st.s - s_safe),
            st.u
        ])
        
        # Composite risk
        risk = cfg.get("arc_w_u", 0.4) * x[2] + cfg.get("arc_w_a", 0.3) * x[0] + cfg.get("arc_w_s", 0.35) * x[1]
        
        # Meta-control update (every 20 steps)
        self.slow_counter += 1
        if self.slow_counter >= 20:
            perf = obs.get("perf", 0.5)
            self._update_meta_control(perf, risk)
            self.slow_counter = 0
        
        # LQR with adaptive gains: K_effective = K_base * current_gain / sqrt(R)
        K_effective = self.K_base * self.current_gain / np.sqrt(self.R_current)
        
        # Control: u = K @ x
        u_lqr = K_effective @ x
        
        # Map to individual control actions
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * K_effective[1] * x[1])
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * K_effective[0] * x[0])
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * K_effective[2] * x[2])
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * min(1.0, u_lqr))
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - risk))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        """Reset meta-control state."""
        self.R_current = 1.0
        self.current_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0


# =============================================================================
# ARC ULTIMATE: MPC + LQI + META-CONTROL (State-of-the-Art)
# =============================================================================

class ARC_Ultimate:
    """
    ARC Ultimate: The most advanced controller combining:
    
    1. MPC (Model Predictive Control): Optimizes over N-step horizon
    2. LQI (Linear Quadratic Integral): Optimal gains from Riccati + integral
    3. Meta-Control: Adaptive R (control cost) based on performance
    4. Anti-Rumination Priority: Aggressive integral for narrative (S)
    
    This controller represents the theoretical best-case for affective regulation.
    
    Control Law:
        u(t) = K_lqi @ [x; integral(x)] + u_mpc(x, horizon) * meta_gain
    
    Where:
        - K_lqi: Optimal gains from DARE (Riccati)
        - u_mpc: Predictive correction from N-step lookahead
        - meta_gain: Adaptive scaling based on performance history
    """
    name = "arc_ultimate"
    
    def __init__(self):
        # ===== LQI Component (from Riccati solution) =====
        self.K_lqr = np.array([
            [1.5071, 0.0985, 0.4586],  # K_calm
            [0.0036, 2.0943, 0.0879],  # K_dmg
            [0.0380, -0.0197, 0.9244], # K_att
        ])
        
        # Integral states
        self.integral_a = 0.0
        self.integral_s = 0.0
        self.integral_u = 0.0
        
        # Integral gains (tuned for anti-rumination)
        self.Ki = np.array([0.12, 0.30, 0.08])  # [arousal, narrative(high), uncertainty]
        
        # ===== MPC Component =====
        self.horizon = 5  # Prediction horizon
        
        # Linearized system matrices (from ASSB dynamics)
        self.A = np.array([
            [0.92, 0.05, 0.15],
            [0.10, 0.95, 0.08],
            [0.03, 0.02, 0.94],
        ])
        self.B = np.array([
            [-0.25, -0.05, 0.0],
            [-0.10, -0.30, -0.05],
            [-0.03, -0.05, -0.20],
        ])
        
        # MPC cost weights
        self.Q_mpc = np.diag([3.0, 6.0, 1.5])  # State cost (narrative highest)
        self.R_mpc = np.diag([0.5, 0.3, 0.8])  # Control cost
        
        # ===== Meta-Control Component =====
        self.meta_gain = 1.0
        self.meta_gain_min = 0.6
        self.meta_gain_max = 1.4
        self.target_perf = 0.90
        self.perf_history = []
        self.slow_counter = 0
        
    def _mpc_optimize(self, x0: np.ndarray, cfg: dict) -> np.ndarray:
        """
        Simplified MPC: Predict N steps and compute corrective action.
        
        Uses iterative prediction: x_{k+1} = A @ x_k + B @ u_k
        Minimizes: J = Σ(x'Qx + u'Ru) over horizon
        
        For efficiency, uses greedy single-step optimization repeated.
        """
        x = x0.copy()
        total_cost = 0.0
        u_mpc = np.zeros(3)
        
        for k in range(self.horizon):
            # Predict next state with current optimal control
            # Greedy: u_k = argmin (x'Qx + u'Ru) for linearized system
            # Solution: u = -inv(R + B'QB) @ B'Q @ A @ x
            
            try:
                BTQ = self.B.T @ self.Q_mpc
                inv_term = np.linalg.inv(self.R_mpc + BTQ @ self.B)
                u_opt = inv_term @ BTQ @ self.A @ x
            except np.linalg.LinAlgError:
                u_opt = np.zeros(3)
            
            # Accumulate first-step control (weighted by decay)
            decay = 0.8 ** k  # Discount future corrections
            u_mpc += decay * u_opt
            
            # Predict next state
            x = self.A @ x + self.B @ np.clip(u_opt, 0, 1)
            
            # Accumulate cost for monitoring
            total_cost += float(x.T @ self.Q_mpc @ x)
        
        # Normalize
        u_mpc /= sum([0.8**k for k in range(self.horizon)])
        
        return np.clip(u_mpc, 0, 1)
    
    def _update_meta_gain(self, perf: float, risk: float, s_error: float):
        """
        Adaptive meta-control: Adjust control intensity based on performance.
        """
        self.perf_history.append(perf)
        if len(self.perf_history) > 25:
            self.perf_history.pop(0)
        
        mean_perf = sum(self.perf_history) / len(self.perf_history)
        
        # Aggressive response to rumination risk
        if s_error > 0.2:  # High rumination risk
            self.meta_gain = min(self.meta_gain_max, self.meta_gain + 0.08)
        elif mean_perf > self.target_perf and risk < 0.15:  # Stable and safe
            self.meta_gain = max(self.meta_gain_min, self.meta_gain - 0.02)
        elif mean_perf < self.target_perf - 0.15 or risk > 0.4:  # Crisis
            self.meta_gain = min(self.meta_gain_max, self.meta_gain + 0.05)
    
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        # ===== State Extraction =====
        a_safe = cfg.get("a_safe", 0.60)
        s_safe = cfg.get("s_safe", 0.55)
        s_rum_tau = cfg.get("s_rum_tau", s_safe)
        
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        s_rum_error = max(0.0, st.s - s_rum_tau)
        
        x = np.array([a_error, s_error, st.u])
        
        # Composite risk
        risk = (cfg.get("arc_w_u", 0.4) * st.u +
                cfg.get("arc_w_a", 0.3) * a_error +
                cfg.get("arc_w_s", 0.35) * s_error)
        
        # ===== LQI: Optimal Feedback + Integral =====
        # Update integrals with anti-windup
        self.integral_a = np.clip(self.integral_a + self.Ki[0] * a_error, 0, 0.6)
        self.integral_s = np.clip(self.integral_s + self.Ki[1] * s_rum_error, 0, 1.0)  # Higher limit for rumination
        self.integral_u = np.clip(self.integral_u + self.Ki[2] * st.u, 0, 0.4)
        
        integral_vec = np.array([self.integral_a, self.integral_s, self.integral_u])
        
        # LQR feedback
        u_lqr = self.K_lqr @ x
        
        # LQI = LQR + Integral
        u_lqi = u_lqr + np.array([
            cfg.get("arc_k_calm", 0.85) * self.integral_a,
            cfg.get("arc_k_dmg", 0.95) * self.integral_s * 1.5,  # Amplify for anti-rumination
            cfg.get("arc_k_att", 0.75) * self.integral_u,
        ])
        
        # ===== MPC: Predictive Correction =====
        u_mpc = self._mpc_optimize(x, cfg)
        
        # ===== Meta-Control Update =====
        self.slow_counter += 1
        if self.slow_counter >= 15:
            perf = obs.get("perf", 0.5)
            self._update_meta_gain(perf, risk, s_error)
            self.slow_counter = 0
        
        # ===== Combine All Components =====
        # Final control: weighted combination of LQI and MPC, scaled by meta_gain
        alpha = 0.65  # LQI weight
        beta = 0.35   # MPC weight
        
        u_combined = alpha * u_lqi + beta * u_mpc * self.meta_gain
        
        # Map to control actions
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * u_combined[0])
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * u_combined[1])
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u_combined[2])
        
        # Memory gating and reappraisal
        max_risk = max(u_combined[0], u_combined[1], self.integral_s)
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * max_risk)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - min(1.0, risk)))
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        """Reset all stateful components."""
        self.integral_a = 0.0
        self.integral_s = 0.0
        self.integral_u = 0.0
        self.meta_gain = 1.0
        self.perf_history = []
        self.slow_counter = 0


# =============================================================================
# ARCv2_LQI: HIERARCHICAL + LQI (Best of Both Worlds)
# =============================================================================

class ARCv2_LQI:
    """
    ARCv2 with LQI: Multi-timescale hierarchical + LQR optimal + integral.
    
    Combines:
    - ARCv2: 3-level control (fast/medium/slow timescales) for efficiency
    - LQI: Optimal gains + integral for anti-rumination
    """
    name = "arc_v2_lqi"
    
    def __init__(self):
        # LQI gains (Riccati)
        self.K = np.array([[1.5071, 0.0985, 0.4586], [0.0036, 2.0943, 0.0879], [0.0380, -0.0197, 0.9244]])
        self.integral_s = 0.0
        self.integral_a = 0.0
        self.ki_s = 0.28
        self.ki_a = 0.10
        
        # Hierarchical state
        self.medium_counter = 0
        self.slow_counter = 0
        self.last_medium = {"u_dmg": 0.0, "u_reapp": 0.0}
        self.perf_history = []
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        a_safe, s_safe = cfg.get("a_safe", 0.60), cfg.get("s_safe", 0.55)
        s_rum_tau = cfg.get("s_rum_tau", s_safe)
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        s_rum_error = max(0.0, st.s - s_rum_tau)
        x = np.array([a_error, s_error, st.u])
        
        # Update integrals
        self.integral_s = np.clip(self.integral_s + self.ki_s * s_rum_error, 0, 1.0)
        self.integral_a = np.clip(self.integral_a + self.ki_a * a_error, 0, 0.5)
        
        u_lqr = self.K @ x
        
        # Fast loop (every step): arousal + attention
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * (u_lqr[0] + self.integral_a))
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u_lqr[2])
        
        # Medium loop (every 5 steps): narrative
        self.medium_counter += 1
        if self.medium_counter >= 5 or s_rum_error > 0.1:
            self.last_medium["u_dmg"] = min(1.0, cfg.get("arc_k_dmg", 0.95) * (u_lqr[1] + 1.5 * self.integral_s))
            self.last_medium["u_reapp"] = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - u_lqr[1]))
            self.medium_counter = 0
        
        risk = max(a_error, s_error, self.integral_s)
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * risk)
        
        return {"u_dmg": self.last_medium["u_dmg"], "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": self.last_medium["u_reapp"]}
    
    def reset(self):
        self.integral_s = 0.0
        self.integral_a = 0.0
        self.medium_counter = 0
        self.last_medium = {"u_dmg": 0.0, "u_reapp": 0.0}


# =============================================================================
# ARC_Robust: H-INFINITY INSPIRED (Maximum Robustness)
# =============================================================================

class ARC_Robust:
    """
    ARC Robust: H∞ inspired controller for maximum robustness.
    Uses conservative gains with robustness margins + integral for anti-rumination.
    """
    name = "arc_robust"
    
    def __init__(self):
        self.gamma = 1.5  # Robustness parameter
        self.K_base = np.array([1.0, 1.2, 0.7])
        self.integral_s = 0.0
        self.ki_s = 0.22
        self.disturbance_est = 0.1
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        a_safe, s_safe = cfg.get("a_safe", 0.60), cfg.get("s_safe", 0.55)
        s_rum_tau = cfg.get("s_rum_tau", s_safe)
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_safe)
        s_rum_error = max(0.0, st.s - s_rum_tau)
        x = np.array([a_error, s_error, st.u])
        
        self.integral_s = np.clip(self.integral_s + self.ki_s * s_rum_error, 0, 0.8)
        
        # H∞ robust control with margins
        u_nominal = self.K_base * x
        robustness_margin = self.gamma * self.disturbance_est * 0.3
        u_robust = u_nominal + robustness_margin
        u_robust[1] += self.integral_s  # Add integral for narrative
        
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * u_robust[0])
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * u_robust[1])
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u_robust[2])
        
        risk = max(u_robust[0], u_robust[1])
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * risk)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - min(1.0, np.sum(x))))
        
        # Update disturbance estimate
        self.disturbance_est = 0.9 * self.disturbance_est + 0.1 * np.linalg.norm(x)
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        self.integral_s = 0.0
        self.disturbance_est = 0.1


# =============================================================================
# ARC_Adaptive: SELF-TUNING (Online Learning)
# =============================================================================

class ARC_Adaptive:
    """
    ARC Adaptive: Self-tuning controller with online parameter optimization.
    Automatically adjusts gains based on observed performance.
    """
    name = "arc_adaptive"
    
    def __init__(self):
        self.K = np.array([1.2, 1.5, 0.8])
        self.Ki = np.array([0.10, 0.25, 0.05])
        self.integral = np.zeros(3)
        self.lr = 0.005
        self.perf_history = []
        self.adapt_counter = 0
        self.best_K = self.K.copy()
        
    def act(self, st: State, obs: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
        a_safe, s_safe = cfg.get("a_safe", 0.60), cfg.get("s_safe", 0.55)
        s_rum_tau = cfg.get("s_rum_tau", s_safe)
        a_error = max(0.0, st.a - a_safe)
        s_error = max(0.0, st.s - s_rum_tau)
        x = np.array([a_error, s_error, st.u])
        
        self.integral = np.clip(self.integral + self.Ki * x, 0, 1.0)
        u = self.K * x + self.integral
        
        u_calm = min(1.0, cfg.get("arc_k_calm", 0.85) * u[0])
        u_dmg = min(1.0, cfg.get("arc_k_dmg", 0.95) * u[1])
        u_att = min(1.0, cfg.get("arc_k_att", 0.75) * u[2])
        
        risk = max(u[0], u[1])
        u_mem = 1.0 - min(1.0, cfg.get("arc_k_mem_block", 0.8) * risk)
        u_reapp = min(1.0, cfg.get("arc_k_reapp", 0.5) * st.u * (1.0 - min(1.0, np.sum(x))))
        
        # Online adaptation
        perf = obs.get("perf", 0.5)
        self.perf_history.append(perf)
        if len(self.perf_history) > 30:
            self.perf_history.pop(0)
        
        self.adapt_counter += 1
        if self.adapt_counter >= 20:
            mean_perf = sum(self.perf_history) / len(self.perf_history)
            if mean_perf < 0.90:
                self.K = np.clip(self.K + self.lr * np.abs(x), 0.5, 2.5)
                self.Ki = np.clip(self.Ki + self.lr * 0.5, 0.05, 0.5)
            if s_error > 0.05:
                self.Ki[1] = min(0.5, self.Ki[1] + 0.02)
            self.adapt_counter = 0
        
        return {"u_dmg": u_dmg, "u_att": u_att, "u_mem": u_mem, "u_calm": u_calm, "u_reapp": u_reapp}
    
    def reset(self):
        self.integral = np.zeros(3)
        self.perf_history = []
        self.adapt_counter = 0





