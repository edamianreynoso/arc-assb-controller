# Metrics Documentation

## ASSB Core Metrics

### Performance Mean (PerfMean)
Average normalized performance across the episode.

### Recovery Time (RT)
Number of timesteps required to return to baseline performance (within ε) after perturbation.
Lower is better.

### Rumination Index (RI)
Proportion of timesteps where narrative gain (S) exceeds rumination threshold, plus persistence penalty.
Lower is better. RI = 0 indicates no rumination.

### Narrative Dominance Ratio (NDR)
Fraction of post-shock timesteps where high narrative gain (S > s_safe) coincides with non-improving performance.
Lower is better. NDR = 0 indicates narrative does not dominate over task-relevant processing.

### Control Effort
Average magnitude of control interventions per timestep.
Lower indicates more efficient regulation.

### Overshoot
Maximum arousal deviation above safety threshold.

## L2 Metrics

### Retention Index
Ratio of post-shift performance to pre-shift performance, measuring resistance to forgetting.

## L6 Metrics (RL)

### Success Rate
Proportion of episodes where the agent reached the goal.

### Total Reward
Cumulative reward per episode.

---

For formal definitions, see Appendix D of the paper.
