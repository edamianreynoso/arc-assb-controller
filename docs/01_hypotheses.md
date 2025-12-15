# Hipótesis V0.1 (para validar una por una)

## H1 - ARC reduce el tiempo de recuperación
**Hipótesis:** Para tareas con perturbación (reward_flip, interruption_burst), el agente ARC tendrá menor `recovery_time` que baseline.

**Necesario:**
- N >= 10 seeds (recomendado 30)
- mismo steps por seed
- comparación estadística simple (media/IC95% o test no-paramétrico)

**Criterio de éxito:**
- RT_arc < RT_base en la mayoría de seeds
- diferencia consistente entre tareas

## H2 - ARC limita "overshoot" afectivo
**Hipótesis:** ARC tendrá menor `overshoot_arousal` después del evento.

**Interpretación:** el sistema evita escalada (homeostasis funcional).

## H3 - ARC reduce rumiación y dominancia narrativa
**Hipótesis:** ARC tendrá menor `rumination_index` y menor `narrative_dominance_ratio`.

**Interpretación:** la ganancia DMN no secuestra decisiones cuando hay evidencia suficiente.

## H4 - ARC reduce deriva de criterio
**Hipótesis:** ARC reducirá `value_drift` (mayor estabilidad longitudinal) sin colapsar desempeño.

## "Ablations" (para demostrar causalidad)
Repite H1-H4 apagando componentes:

- ARC sin homeostasis (k_homeo = 0)
- ARC sin anti-rumiación (k_dmg = 0)
- ARC sin bloqueo de plasticidad (quitar factor por dmg)
- ARC sin clamp de seguridad (dmg_max alto)

Si una ablation rompe la métrica, ese componente es causal (no adorno).
