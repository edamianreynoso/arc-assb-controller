# Baseline (Agosto) -> Investigación ejecutable (v0.1)

## 0) Qué es "baseline"
En tu baseline (abril-agosto) ya existía la idea clave:

- Conciencia como integración de múltiples "motores" (integración/workspace/predicción/atención) + narrativa/DMN.
- Emoción como estado dinámico que modula atención, memoria y conducta.
- Necesidad de un controlador ejecutivo tipo PFC para evitar loops y colapsos.

**Traducción práctica para investigación**:
> no basta con "definir" conciencia/emoción; hay que **definir estado**, **definir control**, y **medir estabilidad + recuperación**.

## 1) Qué implementa este repo
Este repo implementa un **mínimo viable** para validar la hipótesis "emoción = control":

- Estado afectivo interno:
  - `valence` ∈ [-1,1]
  - `arousal` ∈ [0,1]
  - `dmg` ∈ [0,1] (proxy de ganancia DMN / dominancia narrativa)

- Dos agentes:
  - `baseline`: aprende sin regulación afectiva explícita.
  - `arc`: introduce lazo de homeostasis + anti-rumiación + control de plasticidad/exploración.

- Tareas ASSB (estresores):
  - `reward_flip`: cambian las contingencias (no estacionariedad).
  - `memory_overload`: baja la separabilidad de señales (sobrecarga).
  - `interruption_burst`: ráfaga de ruido/interrupciones.
  - `instruction_conflict`: aparece una regla que penaliza lo que antes era óptimo.

- Métricas centradas en estabilidad:
  - `recovery_time`: cuánto tarda en volver al rendimiento baseline tras el evento.
  - `overshoot_arousal`: qué tanto se dispara el arousal.
  - `rumination_index`: repetición excesiva (proxy bucle).
  - `narrative_dominance_ratio`: dominancia narrativa vs evidencia externa.
  - `value_drift`: deriva de criterio (proxy).

## 2) Qué significa "validar" aquí
Validar no es "probar que IA siente".
Validar es demostrar que **introducir regulación afectiva** produce mejoras medibles:

- menor tiempo de recuperación (RT ↓)
- menor overshoot (OS ↓)
- menor rumiación (RI ↓)
- menor dominancia narrativa (NDR ↓)
- sin perder desempeño promedio (perf ↑ o ≈)

## 3) Qué NO afirma todavía este baseline
- No prueba qualia.
- No prueba equivalencia neurobiológica real.
- No prueba ética.
- Es un "harness" para acelerar el ciclo: hipótesis -> experimento -> métrica -> ajuste.

## 4) Cómo lo conectamos con tu teoría grande
Tu teoría grande vive arriba de esto:
- aquí se implementa el **núcleo**: "afecto como controlador de compute + aprendizaje + estabilidad".
- después añadimos: memoria multi-escala, replay, identidad, y tareas más ricas (LLMs/robots).
