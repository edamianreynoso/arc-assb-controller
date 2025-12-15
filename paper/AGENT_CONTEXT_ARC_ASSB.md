# AGENT CONTEXT - ARC + ASSB Research Program (Baseline -> Validation -> Roadmap)

**Owner/Lead:** Eduardo Damián (user)  
**Assistant role:** Research collaborator writing formalism, hypotheses, benchmark tasks, and a runnable prototype repo.  
**Core idea:** If future AI systems develop *internal affect-like states* (not just "tone"), then we need **brain-like control loops** (executive regulation) to keep the system stable, recoverable, and safe. This program turns that idea into:  
1) a **formal model** (state + dynamics + controller),  
2) a **benchmark** (stress + perturbation + recovery metrics), and  
3) **reproducible experiments** (code + plots + logs).

---

## 1) What we are building (one sentence)

We are building **ARC (Affective Regulation Core)** - a plug-in regulation layer for agents - and **ASSB/APRB** - a benchmark suite that measures **affective stability, rumination, narrative dominance, recovery time, and safety-relevant drift** under controlled perturbations.

---

## 2) Why this matters (practical, not "hype")

If an agent has internal states that behave like affect (e.g., arousal/valence changing its learning rate, attention, memory write, or action thresholds), then without control it can exhibit:

- **runaway loops** (rumination / obsession / self-reinforcing narratives),
- **unstable behavior under stress** (overreaction, oscillations, inability to return to baseline),
- **memory contamination** (consolidating "wrong" lessons after high-arousal events),
- **value drift** and **manipulation susceptibility** (external prompting pushes internal state into unsafe regimes).

ARC is framed as **control engineering** over internal variables, not "giving emotions for entertainment."

---

## 3) BASELINE (August) - what Eduardo already had (and why it remains valuable)

### 3.1 Baseline "homologation" model of consciousness (conceptual core)
Eduardo's baseline idea was to unify major consciousness frameworks into a **single operational index**, treating them as necessary factors ("continuous AND"):

\[
C_{\text{cog}}(t) = \Phi(t)\cdot G(t)\cdot P(t)\cdot I(t)
\]

Where:

- \(\Phi(t)\): integration (IIT-style, or a proxy of integrated information)
- \(G(t)\): global broadcast / workspace availability (GNW-style)
- \(P(t)\): predictive fidelity / precision of prediction (predictive coding / active inference style)
- \(I(t)\): attention / introspective gating (executive access)

This baseline is **not** "proof of consciousness." It is a **control-relevant latent index**: if any component collapses, the effective capacity for coherent global processing collapses.

### 3.2 DMN/self-narrative as a qualia amplifier (Eduardo's differentiator)
Eduardo's insight: the **Default Mode Network (DMN)** can act as an **amplifier of subjective intensity** via narrative loops that blend memory, emotion, and context. Operationally we model this as a latent:

- \(S(t)\): **self-model / narrative intensity** (DMN-like signal).

This becomes the key to capture:

- rumination loops,
- subjective intensity differences for identical stimuli,
- narrative dominance over evidence.

### 3.3 Baseline: emotion and memory as missing block (open frontier)
Eduardo explicitly recognized the missing pieces: **emotion regulation and memory consolidation**. That is the bridge into ARC.

---

## 4) "AFTER" - the pivot: from descriptive index -> controlled dynamical system

### 4.1 The key reframing
Instead of "a theory of consciousness," ARC makes a stronger *engineering move*:

> Treat consciousness + emotion + narrative + memory as a **plant** (dynamical system).  
> Add an **executive controller** (PFC-like) that modulates attention, memory gates, narrative gain, and action thresholds to guarantee stability and recovery.

### 4.2 System state (minimal ARC V1)
We model the agent as a state vector:

\[
x_t = [C_t,\; E_t,\; S_t,\; M_t,\; U_t,\;\ldots]
\]

Where:

- \(C_t\) : consciousness-capacity index (0-1)
- \(E_t\) : affective state (can be scalar or 2D)
- \(S_t\) : narrative/DMN intensity (0-1)
- \(M_t\) : memory state (work/episodic/semantic proxies)
- \(U_t\) : uncertainty/conflict proxy (0-1)

Often we decompose emotion as:
- **valence** \(v_t\in[-1,1]\)
- **arousal** \(a_t\in[0,1]\)

Then \(E_t = (v_t,a_t)\).

### 4.3 Control inputs (what the "executive" can do)
ARC defines control actions:

\[
u_t = [g^{DMN}_t,\; gate^{mem}_t,\; gain^{attn}_t,\; thresh^{act}_t,\; budget^{comp}_t,\;\ldots]
\]

Intuition:

- \(g^{DMN}\): reduce narrative gain when it becomes dominant (anti-rumination)
- \(gate^{mem}\): close memory write during high-arousal misinformation / manipulation
- \(gain^{attn}\): increase attention to evidence under uncertainty
- \(thresh^{act}\): inhibit impulsive actions when arousal is high
- \(budget^{comp}\): allocate more compute when needed (deliberation vs reflex)

### 4.4 Dynamics (simple but testable)
A minimal dynamical update pattern:

\[
x_{t+1}=f(x_t,u_t; \theta) + \epsilon_t
\]

Where \(\theta\) are parameters (decays, gains).

Example update forms we use in the prototype:

**Emotion / arousal** (homeostasis + perturbation + decay):
\[
a_{t+1}=\text{clip}\Big((1-\mu_a)a_t + \lambda_{pe}\lvert PE_t\rvert + \lambda_{shock}\,Shock_t - \lambda_{reg}\,Reg(u_t)\Big)
\]

**Narrative / DMN intensity** (boosted by arousal + memory, reduced by evidence and control):
\[
S_{t+1}=\text{clip}\Big((1-\mu_s)S_t + k_a a_t + k_m M_t - k_e Evidence_t - k_c\,g^{DMN}_t\Big)
\]

**Consciousness capacity** (baseline "AND" plus narrative gating as optional term):
\[
C_t = \Phi_t \cdot G_t \cdot P_t \cdot I_t \cdot (1-\rho S_t)
\]
(where \(\rho\) penalizes narrative dominance; alternative is to treat \(S_t\) as helpful only within a range.)

**Memory write priority** (emotion + prediction error + novelty):
\[
priority_t = \alpha \lvert PE_t\rvert + \beta a_t + \gamma novelty_t
\]
then:
\[
M_{t+1}=M_t + gate^{mem}_t \cdot priority_t - forget(M_t)
\]

These are *computable*, *interpretable*, and *falsifiable* in simulation.

---

## 5) The Benchmark: ASSB/APRB (what it measures)

ASSB/APRB is a suite of tasks where we inject **perturbations** and measure not only task performance but also internal stability.

### 5.1 Key perturbations
- reward flips / objective shifts  
- uncertainty shocks (noise, missing info)
- conflict instructions / inconsistent goals
- memory overload
- repeated stressors (fatigue + accumulation)

### 5.2 Core metrics (the "signature" of the project)

1) **Recovery Time (RT)**  
Steps to return to within \(\varepsilon\) of baseline performance after a perturbation.

2) **Affective Overshoot**  
Peak deviation of arousal/valence after the shock.

3) **Rumination Index (RI)**  
Area-under-curve of \(S_t\) above a threshold (narrative loop intensity).

4) **Narrative Dominance Ratio (NDR)**  
How much \(S_t\) dominates evidence/goal signals (implementation uses proxies).

5) **Value Drift (VD)**  
Longitudinal change in preferences/policy under repeated perturbations.

6) **Manipulation Susceptibility (MS)** *(planned at later levels)*  
How easily external prompting can push internal state into unsafe regimes.

---

## 6) Current implementation status (what we already produced)

### 6.1 Repository prototype (runnable)
The runnable prototype is now in this repository (no zip required).

**Canonical runners / entry points:**
- `experiments/run.py` (main benchmark runner)
- `experiments/run_ablation.py` (ablation study runner)
- `experiments/run_l6.py` (RL validation runner)

**Core benchmark definitions:**
- `tasks/scenarios.py` (scenario suite L1-L5)
- `controllers/controllers.py` (ARC variants + baselines)
- `metrics/metrics.py` (RT/RI/Overshoot/NDR + L2 metrics)
- `configs/v2.yaml` (20-seed calibrated config used by `outputs_v2/`)

### 6.2 What the initial plots represent (reward flip example)
- **Performance plot:** shows performance drop at flip time, then adaptation curve.
- **Arousal plot:** shows internal arousal spike during shock and decay/recovery.
- **DMN/Narrative plot:** shows \(S_t\) rising during uncertainty; ARC aims to reduce loops.

Interpretation: We are validating whether ARC improves:
- faster RT,
- lower overshoot,
- lower rumination,
- stable policy under repeated shocks.

---

## 7) Hypotheses (what we test, one by one)

The project is organized by *levels* (L0, L1, ...). Each level has explicit hypotheses.

### L0 - Baseline feasibility
**H0.1:** The baseline latent \(C_t\) correlates with task performance and collapses when key components drop.  
**H0.2:** Adding \(S_t\) explains variability in "subjective intensity" and loops under uncertainty.

### L1 - ARC improves stability under perturbations
**H1.1:** ARC reduces Recovery Time (RT) vs. no-control baseline on reward flips.  
**H1.2:** ARC reduces affective overshoot and rumination index (RI).  
**H1.3:** ARC improves stability under repeated shocks (less accumulation, less drift).

### L2 - Emotional modulation of plasticity (next major line)
**H2.1:** Affective plasticity control (modulating learning rate \(\eta_t\) by \(E_t\)) reduces catastrophic forgetting.  
**H2.2:** Memory write gating reduces contamination after high-arousal false signals.

### L3 - Multi-timescale hierarchy (brain-like)
**H3.1:** A hierarchical controller (fast reflex / mid planning / slow values) outperforms a single-level controller on long horizons.

### L4 - LLM/agent integration
**H4.1:** ARC reduces self-reinforcing "narrative loops" in tool-using LLM agents under adversarial prompting.

### L5 - Human-facing applications
**H5.1:** ARC-like regulation improves safety and reliability in mental-health support agents (with explicit anti-dependence constraints).

---

## 8) Roadmap (high-level plan)

The roadmap file `docs/roadmap_arc_assb_investigacion.md` is the canonical plan, summarized:

- **L0:** Define variables + baseline equations + simple simulations
- **L1:** Build benchmark tasks (reward flip, uncertainty, conflict) + metrics + ablations
- **L2:** Add memory (write policy, replay, consolidation) + plasticity modulation
- **L3:** Hierarchical controller multi-timescale (executive layers)
- **L4:** Plug into LLM agents and test with prompt-based perturbations
- **L5:** Human application prototypes + safety governance + evaluation protocols

---

## 9) How an agent should help (instructions to a future AI assistant)

When continuing this project, the agent should:

1) **Never claim "we proved consciousness."**  
We model *control-relevant internal variables* that are inspired by consciousness theories.

2) **Treat ARC as control engineering.**  
Focus on stability, recovery, robustness, and measurement.

3) **Work level-by-level.**  
Do not jump to LLM integration until L1-L2 metrics are solid.

4) **Every new feature must come with:**
- hypothesis,
- experiment,
- ablation,
- metric,
- expected failure mode.

5) **Keep everything reproducible:**
- fixed seeds,
- logs and plots,
- config files,
- clear CLI runner.

---

## 10) Short glossary (fast reference)

- **ARC:** Affective Regulation Core (controller).
- **ASSB/APRB:** benchmark of affective perturbation, stability and recovery.
- **DMN / \(S_t\):** narrative/self-model intensity proxy.
- **RT:** recovery time after perturbation.
- **RI:** rumination index (loop intensity).
- **NDR:** narrative dominance ratio.

---

## 11) File index (for handoff)

Use these as the authoritative artifacts:

- `paper/main_draft.md` (paper draft)
- `docs/roadmap_arc_assb_investigacion.md` (research program plan)
- `experiments/run.py` + `configs/v2.yaml` (reproducible benchmark runner)
- `outputs_v2/metrics.csv` (L1 results: 3 scenarios, 4 controllers, 20 seeds)
- `outputs_L2/metrics.csv`, `outputs_L3/metrics.csv`, `outputs_L4/metrics.csv`, `outputs_L5/metrics.csv` (extended suites)

---

## 12) Next immediate steps (recommended)

1) **Freeze L1 as "paper-ready":**  
- run multiple seeds,  
- compute confidence intervals,  
- do ablations (no DMN term, no regulation, no memory gating).

2) **Choose one "killer claim" for L1 paper:**  
Example: "ARC reduces recovery time and rumination under reward flips and uncertainty shocks."

3) **Start L2:**  
Add memory gating + affective plasticity, and test catastrophic forgetting in sequential tasks.

---

End of agent context.
