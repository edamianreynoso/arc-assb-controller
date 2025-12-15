# Research status (what is validated here)

This repo currently supports:
- **ASSB simulation benchmark (L1-L5)** via `python -m experiments.run --config configs/v2.yaml --outdir <OUT>`
- **Ablations** via `python -m experiments.run_ablation --config configs/v2.yaml --outdir <OUT> --seeds 20`
- **RL validation (L6)** via `python -m experiments.run_l6 --episodes 200 --seeds 10 --outdir <OUT>`

## What to trust (artifacts)

### L1 (stability/recovery) - validated
- Canonical run: `python -m experiments.run --config configs/v2.yaml --outdir outputs_v2`
- Results: `outputs_v2/metrics.csv`

### L2-L5 (extended scenarios) - validated in simulation
- Results:
  - `outputs_L2/metrics.csv`
  - `outputs_L3/metrics.csv`
  - `outputs_L4/metrics.csv`
  - `outputs_L5/metrics.csv`

### L4 (meta-control) - now measurable
We added `ControlEffort` (proxy of intervention "energy") to ASSB metrics:
- Code: `metrics/metrics.py`
- Output column: `ControlEffort` in any new `outputs*/metrics.csv`
- Example run: `python -m experiments.run --config configs/v2.yaml --outdir outputs_rev11`

### L6 (RL) - validated on 3 envs, strongest on non-stationary goal changes
- Run: `python -m experiments.run_l6 --episodes 200 --seeds 10 --outdir outputs_L6_v2`
- Results: `outputs_L6_v2/final_metrics.csv`
- Learning curves: `python -m experiments.analyze_l6_curves --outdir outputs_L6_v2` (writes `outputs_L6_v2/plots/`)

## How to interpret metrics
See `docs/metrics_interpretation.md`.

