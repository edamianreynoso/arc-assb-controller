# ARC-ASSB

Code and data for the paper **"Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents"** (Damian Reynoso, 2026).

## Overview

AI agents with internal "emotional" states tend to get stuck in loops (like rumination in humans) and are easy to manipulate. This repo implements **ARC**, a control system that keeps those internal states stable---inspired by how the prefrontal cortex regulates emotions.

We tested **15 controller architectures** (P, PID, LQR, LQI, hierarchical, meta-control, H-infinity robust, adaptive, MPC) across **10 stress scenarios** with 20 seeds each. The best controllers achieve **96.6% task performance with zero rumination**.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce the paper

```bash
# L1-L5: Run the benchmark simulation (15 controllers x 10 scenarios x 20 seeds)
python experiments/run.py --config configs/v2.yaml --outdir outputs_final

# L6: Tabular Q-learning validation (20 seeds x 200 episodes)
python experiments/run_l6.py --episodes 200 --seeds 20 --outdir outputs_L6_robust

# L6 ablation: Memory gating vs shift detection
python experiments/run_l6_ablation.py

# L6b: Deep RL (DQN) exploratory suite
python experiments/run_l6b_dqn_suite.py

# Generate figures
python analysis/generate_controller_figures.py
python analysis/generate_controller_heatmaps.py
python analysis/generate_sensitivity_figures.py
python visualizations/paper_figures.py

# Verify all paper claims against data
python verify_paper_claims.py
```

Pre-computed results are included in `outputs_final/`, `outputs_L6_robust/`, `outputs_L6_ablation_final/`, and `outputs_L6b_dqn_suite_v1/` for direct figure generation and claim verification.

## Repository structure

```
assb/                    Core ASSB framework (state dynamics, agent interface)
controllers/             15 controller architectures (PID, LQR, H-inf, etc.)
metrics/                 Evaluation metrics (PerfMean, RI, NDR, RT, ControlEffort)
sim/                     Simulation engine (dynamics, state management)
tasks/                   10 benchmark scenarios (reward_flip, gaslighting, etc.)
agents/                  RL agent implementations (Q-learning, DQN wrapper)
envs/                    RL environments (GridWorld, CartPole non-stationary)
configs/                 Experiment configurations (v2.yaml)
experiments/             Experiment runners (L1-L6, ablation, sensitivity)
  deep_rl/               DQN and PPO implementations
analysis/                Figure and statistical analysis generation
visualizations/          Paper figure generation scripts
paper_latex/             LaTeX source for the paper
  main.tex               Paper source
  arxiv.sty              arXiv style file
  figures/               All 27 paper figures
outputs_final/           L1-L5 simulation results (metrics.csv)
outputs_L6_robust/       L6 tabular Q-learning results
outputs_L6_ablation_final/  L6 ablation study results
outputs_L6b_dqn_suite_v1/   L6b DQN exploratory results
figures_controllers/     Generated controller comparison figures
docs/                    Supporting documentation
verify_paper_claims.py   Automated claim verification script
```

## Key results

| Research Line | Key Finding |
|---|---|
| L1 (Stability) | ARC achieves 96.6% PerfMean vs 29.7% baseline |
| L2 (Memory) | Memory gating preserves retention under distribution shift |
| L3 (Anti-rumination) | Integral controllers achieve RI=0 (zero rumination) |
| L4 (Meta-control) | 20.9% control effort reduction via meta-switching |
| L5 (Safety) | Integral controllers collapse under adversarial coupling |
| L6 (RL) | +49.8% transfer learning in ChangingGoalGridWorld |
| L6 (Ablation) | Memory gating alone: 71.8% success (best individual component) |
| L6b (DQN) | Negative results: ARC-plasticity underperforms baseline DQN |

## Citation

```bibtex
@article{damianreynoso2026arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Dami{\'a}n Reynoso, J. Eduardo},
  year={2026}
}
```

## License

Apache-2.0 (see `LICENSE`)
