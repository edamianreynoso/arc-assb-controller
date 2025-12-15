# Related Work

This document summarizes prior work relevant to ARC and ASSB.

## Affective Computing

Picard (1997) established affective computing as the study of systems that recognize, interpret, and simulate human emotions. Most work focuses on external expression rather than internal regulation.

## Emotion in Reinforcement Learning

Moerland et al. (2018) surveyed emotion-like mechanisms in RL, finding applications in exploration, reward shaping, and intrinsic motivation. However, these approaches typically lack explicit stability guarantees.

## Emotion Regulation

Ochsner & Gross (2005) described cognitive emotion regulation strategies implemented by prefrontal cortex. ARC is inspired by these mechanisms, implementing computational analogs of reappraisal, attention deployment, and response modulation.

## Default Mode Network and Rumination

The DMN is associated with self-referential processing and has been linked to rumination in depression (Hamilton et al., 2015). ARC models narrative gain (S) as a DMN proxy and explicitly regulates it to prevent perseverative loops.

## AI Safety

Amodei et al. (2016) outlined concrete problems in AI safety. ARC addresses value drift and reward hacking by maintaining bounded internal states under adversarial conditions.

---

Full references are provided in the paper.
