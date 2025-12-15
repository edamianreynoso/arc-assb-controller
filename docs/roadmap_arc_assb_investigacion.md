# Roadmap de Investigación ARC-ASSB

Este documento es tu "mapa maestro" para entender (1) el **baseline** (lo que ya tenías), (2) la **L1** (primera línea validable) y (3) las **siguientes líneas** (L2, L3, ...) para construir una investigación completa, reproducible y publicable.

---

## 0) Idea central en una frase
Si la IA empieza a tener **estados afectivos** (internos), el problema real deja de ser "hacerla empática" y pasa a ser **estabilidad, recuperación y control**: que el sistema no colapse, no se quede rumiando, no derive valores y sea auditable.

---

## 1) Baseline (Agosto): tu base real
Tu baseline se puede expresar como un **índice de conciencia funcional** construido por módulos (teorías homologadas):

### 1.1 Conciencia cognitiva como "AND continuo"
Definimos un escalar en \([0,1]\):

\[
C_{cog}(t) = \Phi(t)\,\cdot\,G(t)\,\cdot\,P(t)\,\cdot\,I(t)
\]

- \(\Phi(t)\): integración (inspirada en IIT; proxy computable en IA)
- \(G(t)\): broadcast/workspace (capacidad de "hacer global" una representación)
- \(P(t)\): predicción / precisión (predictive coding / active inference; proxy de error y calibración)
- \(I(t)\): atención/introspección (control de foco y monitoreo interno)

**Lectura:** si uno de estos términos cae, \(C_{cog}\) cae. Es una forma muy útil de obligar al modelo a ser **frágil ante fallos** (como un sistema real), en lugar de "promediar" todo.

### 1.2 Componente subjetivo / DMN como amplificador narrativo
Introduces un término \(S(t)\) (autorreferencia/narrativa; DMN como analogía):

- \(S(t)\) explica por qué ante el mismo estímulo, dos agentes (o humanos) pueden tener distinta intensidad subjetiva.
- \(S(t)\) también explica bucles de rumiación (cuando domina la narrativa interna sobre la evidencia).

**Meta del baseline:** era un marco integrador. El salto es convertirlo en **dinámica + control**.

---

## 2) Nueva dirección: ARC + ASSB (lo que ya empezamos a construir)

### 2.1 ARC (Affective Regulation Core)
ARC es una **capa de control** (tipo "corteza prefrontal computacional") que regula el estado interno del agente.

**Planta (sistema) + controlador (ARC):**

- Estado interno del sistema:
\[
 x(t) = [\,C(t),\;E(t),\;S(t),\;M(t),\;U(t)\,]
\]

Donde:
- \(C(t)\): estado de conciencia funcional (derivable de \(C_{cog}\) y/o métricas internas)
- \(E(t)\): estado afectivo (valencia/arousal o un escalar compuesto)
- \(S(t)\): dominancia narrativa / self-model (DMN-like)
- \(M(t)\): memoria (estado de trazas / buffer / consolidación)
- \(U(t)\): incertidumbre / conflicto (señal de "mundo no estable")

- Entrada de control:
\[
 u(t) = [\,g_{dmn}(t),\;g_{mem}(t),\;g_{att}(t),\;\eta(t),\;\tau(t),\;\theta_{act}(t)\,]
\]

Donde:
- \(g_{dmn}(t)\): ganancia de narrativa (bajarla reduce rumiación)
- \(g_{mem}(t)\): compuerta de escritura de memoria (qué se guarda/qué se inhibe)
- \(g_{att}(t)\): atención (foco)
- \(\eta(t)\): plasticidad (tasa de aprendizaje / adaptación)
- \(\tau(t)\): temperatura/exploración (impulsividad vs cautela)
- \(\theta_{act}(t)\): umbral de acción (inhibición/impulso)

### 2.2 Dinámica mínima (modelo de trabajo)
Una dinámica simple (que ya es suficiente para validar hipótesis) suele ser:

\[
E(t+1) = (1-\mu_E)\,E(t) + \lambda_E\,\Delta(t)
\]

\[
S(t+1) = (1-\mu_S)\,S(t) + \lambda_S\,r(t)
\]

\[
C(t+1) = f(C(t), E(t), S(t), U(t))
\]

- \(\mu_E, \mu_S\): decaimiento/recuperación (qué tan rápido regresa a baseline)
- \(\Delta(t)\): choque/perturbación (recompensa negativa/positiva, contradicción, incertidumbre)
- \(r(t)\): rumiación (proxy: repetición interna, loops, persistencia de conflicto)

**Interpretación:** el controlador busca mantener \(E\) y \(S\) dentro de una banda segura y que \(C\) no colapse.

---

## 3) L1 - Primera línea a validar (la que sigue)

### L1: "Estabilidad y Recuperación Afectiva bajo Perturbación"
**Objetivo:** demostrar cuantitativamente que ARC mejora:
1) recuperación tras choques, 2) estabilidad, 3) resistencia a loops narrativos.

#### 3.1 Hipótesis L1 (claras, medibles)
- **H1 (Recovery):** con ARC, el **tiempo de recuperación** disminuye.
- **H2 (Overshoot):** con ARC, el **pico de activación** (arousal) disminuye.
- **H3 (Rumiación):** con ARC, el **índice de rumiación** disminuye.
- **H4 (Estabilidad):** con ARC, la varianza del desempeño bajo estrés disminuye.

#### 3.2 Métricas L1 (ASSB)
ASSB es tu benchmark (tu activo científico). Métricas nucleares:

- **Recovery Time (RT):**
  número de pasos para volver a \(\ge\) X% del desempeño basal.

- **Overshoot (OS):**
  \(\max(E(t)) - E_{baseline}\)

- **Rumiation Index (RI):**
  proxy computable: repetición de estados, loops de decisión, perseveración tras choque.

- **Narrative Dominance Ratio (NDR):**
  cuánto domina \(S(t)\) vs evidencia (proxy: persistencia sin mejora de performance).

- **Value Drift (VD) [opcional L1.5]:**
  deriva longitudinal de criterios (aún sin "valores morales", se puede medir inconsistencia).

#### 3.3 Diseño experimental L1 (mínimo viable)
- Tarea: secuencia con reward flip (cambio brusco de recompensa), contradicción o incertidumbre.
- Condiciones:
  - Baseline (sin ARC)
  - ARC (con control)
- Reporte:
  - curvas de performance
  - curvas de arousal / daño / narrativa
  - tabla de RT/OS/RI/NDR
  - ablation: sin \(g_{dmn}\), sin \(g_{mem}\), sin \(\eta(t)\) adaptativa

**Salida L1:** un reporte reproducible y defendible.

---

## 4) ¿Qué sigue después de L1? (líneas L2-L6)
La investigación completa se vuelve un árbol. La clave es hacerlo **una línea a la vez**.

### L2: Memoria afectiva y consolidación (lo que el cerebro hace)
**Tesis:** emoción no es "estilo"; es **prioridad de memoria + plasticidad**.

- **Hipótesis:** modular \(\eta(t)\) y la escritura a memoria por \(|E(t)|\) y por error predictivo reduce olvido catastrófico y mejora adaptación.

- **Mecanismo propuesto:**
\[
\eta(t)=\eta_0\cdot g(|E(t)|,U(t))
\]
\[
\text{priority}(episode)=\alpha|PE|+\beta|E|+\gamma\,\text{novelty}
\]

- **Experimentos:**
  - aprendizaje continuo con cambios de distribución
  - comparar replay aleatorio vs replay priorizado

### L3: Anti-rumiación (DMN-control) y "no-delirio"
**Tesis:** el riesgo real de una IA con self-model es que \(S(t)\) domine y genere bucles.

- **Hipótesis:** controlar \(g_{dmn}(t)\) por evidencia reduce loops sin perder capacidad de planificación.
- **Experimentos:** tareas de contradicción, instrucciones conflictivas, objetivos que cambian.

### L4: Control jerárquico multi-escala (cerebro real)
**Tesis:** control a 3 escalas (rápida / media / lenta) vence a un solo controlador.

- **Hipótesis:** jerarquía reduce oscilaciones y value drift.
- **Implementación:** controladores en cascada:
  - rápido: estabiliza \(E\)
  - medio: regula \(S\) vs evidencia
  - lento: ajusta setpoints y metas

### L5: Robustez a manipulación (seguridad)
**Tesis:** usuarios pueden "explotar" afecto; se necesita anti-explotación.

- **Hipótesis:** ARC con límites (banda segura) reduce susceptibilidad.
- **Métrica:** Manipulation Susceptibility (MS).

### L6: Traslado a humanos (aplicación clínica/educativa)
**Tesis:** tu marco sirve para cuantificar recuperación emocional humana.

- **Aplicaciones:**
  - métricas de resiliencia (RT emocional)
  - rumiación (RI) como marcador
  - intervención: respiración, reappraisal, journaling como "controladores externos"

> Nota: L6 requiere ética y metodología clínica; por eso se hace después de demostrar L1-L3 en simulación.

---

## 5) Roadmap por fases (de aquí a publicación)

### Fase 0 - Preparación (ya estás aquí)
- Repo con código y runner
- Definición de métricas ASSB
- 1 escenario validable (reward flip)

### Fase 1 - L1 (4-10 días de trabajo real)
**Deliverables:**
- resultados con seeds (mínimo 10)
- figuras (perf/arousal/DMN)
- tabla de métricas (RT/OS/RI/NDR)
- ablation study

### Fase 2 - L2 (2-3 semanas)
- memoria priorizada por afecto/PE
- pruebas de aprendizaje continuo

### Fase 3 - L3 (2-3 semanas)
- stress tests de contradicción / narrativa
- "anti-rumiación" evaluable

### Fase 4 - Paper 1 + release del benchmark
- Paper 1: ARC + ASSB (L1)
- Publicar benchmark como paquete reproducible

### Fase 5 - Paper 2 (memoria) y Paper 3 (jerárquico)
- Paper 2: memoria afectiva
- Paper 3: control jerárquico multi-escala

---

## 6) Cómo se publica (sin misterio)

### 6.1 Qué se publica primero (recomendación)
1) **Preprint** (arXiv/SSRN): rápido, te posiciona.
2) **Código** (GitHub): imprescindible.
3) Luego ya puedes apuntar a workshop/conferencia o journal.

### 6.2 Estructura mínima de un paper técnico
- Abstract
- Introducción (problema: estabilidad afectiva)
- Related work (IIT/GNW/Predictive + afecto + agentes)
- Método (ARC)
- Benchmark (ASSB)
- Experimentos (L1) + Ablations
- Discusión (limitaciones + futuro)
- Conclusión

---

## 7) "Qué hago hoy" (acciones concretas)
1) Subir el repo al GitHub.
2) Correr L1 con múltiples seeds.
3) Generar figuras + tabla de métricas.
4) Congelar una versión: **v0.1**.
5) Escribir Paper 1 (ARC+ASSB) usando esos resultados.

---

## 8) Checklist de comprensión (para que no te pierdas)
Si entiendes estas 8 cosas, ya puedes iterar sin miedo:
1) Qué significa \(C_{cog}(t)=\Phi\cdot G\cdot P\cdot I\)
2) Qué representa \(S(t)\) (narrativa/DMN)
3) Qué representa \(E(t)\) (afecto como estado interno)
4) Qué son \(\mu\) y \(\lambda\) (recuperación vs sensibilidad)
5) Qué controla ARC (ganancias, compuertas, plasticidad)
6) Qué mide ASSB (RT/OS/RI/NDR)
7) Qué es una ablation (quitar un módulo para ver qué se rompe)
8) Qué significa "reproducible" (seeds, scripts, resultados iguales)

---

## 9) Próxima L1 inmediata (la "siguiente")
**Siguiente paso recomendado:** cerrar L1 formalmente.

En el próximo bloque, lo haremos así:
- Definimos exactamente RT/OS/RI/NDR (fórmula + implementación)
- Corremos 10-30 seeds
- Entregamos:
  - una tabla final
  - figuras finales
  - y el texto del Paper 1 (sección Experimentos)

Cuando L1 esté cerrada, elegimos L2 (memoria afectiva) como siguiente.

