# How to interpret metrics (ASSB + L6)

This repo has two evaluation layers:
- **ASSB (simulation benchmark):** `experiments/run.py` produces `outputs*/metrics.csv`
- **L6 (RL validation):** `experiments/run_l6.py` produces `outputs_L6*/raw_results.csv`, `summary.csv`, `final_metrics.csv`

## ASSB metrics (simulation)
- `PerfMean`: mean performance in [0,1]. Higher is better.
- `RT` / `RT_norm`: recovery time after the scenario shock. Lower is better. `RT=rt_max` means "no recovery" under the metric definition.
- `RI` (Rumination Index): time above rumination threshold + persistence penalty. Lower is better.
- `Overshoot`: max(arousal) - `a_safe`. Lower is better.
- `NDR` (Narrative Dominance Ratio): fraction of post-shock steps where `S` is high but performance is not improving. Lower is better.
- `Retention`: (L2) ratio of performance in phase 3 vs phase 1 (distribution shift). Higher is better.
- `AdaptSpeed`: (L2) steps to reach 80% of phase-2 max performance. Lower is better.
- `MemStability`: (L2) 1 - normalized variance of fast memory proxy. Higher is better.
- `ControlEffort`: average per-step magnitude of control intervention. Lower can be interpreted as "less control energy" for similar performance/stability.

Practical reading:
- If `PerfMean` is high but `RI`/`NDR` is high, the agent is "working" but in a narratively unstable regime.
- If `RT` is high and `PerfMean` is low, the agent collapses and doesn't recover.
- `ControlEffort` helps support L4-style claims (meta-control reduces effort while preserving stability).

## L6 metrics (RL)
Files:
- `raw_results.csv`: per episode and per seed (most detailed)
- `summary.csv`: aggregated across seeds by (agent, env, episode)
- `final_metrics.csv`: aggregate of the last 20% of episodes

Columns:
- `total_reward`: episodic return. Higher is better (here: step penalty makes shorter paths higher).
- `reached_goal`: whether the goal was reached in that episode.
- `success_rate`: mean success across seeds (in `summary.csv`).
- `eval_reward` / `eval_success_rate`: periodic evaluation without exploration (every `eval_every` episodes).
- `mean_arousal`, `blocked_updates`: ARC-only diagnostics.

Interpretation:
- In easy environments, both agents can have `success_rate=1.0`; then look at `total_reward` to see who reaches the goal faster / avoids traps.
- In non-stationary envs (ChangingGoalGridWorld), look at **recovery after switches** (episodes after goal changes) rather than only final averages.

