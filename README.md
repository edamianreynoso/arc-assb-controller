# ARC: Affective Regulation Core

> A Homeostatic Control Framework for Stable and Safe AI Agents

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

This repository provides the reference implementation and experimental data for the paper:

> **Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents**  
> J. Eduardo Damián Reynoso (2025)

**Key Results:**
- ARC achieves **97% performance** with **zero rumination** (vs. 30% for uncontrolled agents)
- Meta-control reduces control effort by **21%** while maintaining stability
- In reinforcement learning, ARC improves transfer learning by **+50%** in non-stationary environments

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/edamianreynoso/arc-assb-controller.git
cd arc-assb-controller
pip install -r requirements.txt
```

### 2. Reproduce Paper Results

**Run the full benchmark (L1-L5):**
```bash
python experiments/run.py --config configs/v2.yaml --outdir outputs_final
```

**Generate figures:**
```bash
python analysis/generate_controller_figures.py
```

**Run RL experiments (L6):**
```bash
python experiments/run_l6.py --episodes 200 --seeds 20 --outdir outputs_L6_robust
```

---

## Repository Structure

```
arc-assb-controller/
├── controllers/          # 15 ARC controller variants (PID, LQR, LQI, H∞, MPC, etc.)
├── tasks/                # 10 ASSB benchmark scenarios
├── experiments/          # Execution scripts (run.py, run_l6.py)
├── analysis/             # Figure generation and auditing scripts
├── outputs_final/        # Pre-computed experimental data (3000 traces)
├── figures_controllers/  # Generated benchmark figures
├── figures_L6/           # RL experiment figures
├── paper/                # Paper source (main_draft.md)
└── configs/              # Hyperparameter configurations
```

---

## Controller Suite

| Category | Controllers |
|----------|-------------|
| Baselines | `no_control`, `naive_calm`, `perf_optimized` |
| ARC v1 | `arc_v1`, `arc_v1_pid`, `arc_v1_lqr`, `arc_v1_lqi` |
| ARC v2 | `arc_v2_hier`, `arc_v2_lqi` |
| ARC v3 (Meta) | `arc_v3_meta`, `arc_v3_pid_meta`, `arc_v3_lqr_meta` |
| Robust/Adaptive | `arc_robust`, `arc_adaptive`, `arc_ultimate` |

---

## Benchmark Scenarios (ASSB)

| Scenario | Challenge |
|----------|-----------|
| `reward_flip` | Value shock |
| `noise_burst` | Sustained uncertainty |
| `sudden_threat` | Acute stress |
| `distribution_shift` | Continual learning |
| `goal_conflict` | Memory interference |
| `sustained_contradiction` | Rumination pressure |
| `gaslighting` | Manipulation resistance |
| `instruction_conflict` | Decisiveness vs. rumination |
| `adversarial_coupling` | Safety under reward hacking |
| `random_dopamine` | Dopamine trap resistance |

---

## Citation

If you use ARC or ASSB in your research, please cite:

```bibtex
@article{damianreynoso2025arc,
  title   = {Affective Regulation Core: A Homeostatic Control Framework 
             for Stable and Safe AI Agents},
  author  = {Damián Reynoso, J. Eduardo},
  journal = {arXiv preprint arXiv:2025.XXXXX},
  year    = {2025},
  url     = {https://github.com/edamianreynoso/arc-assb-controller}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions or collaborations, open an issue on this repository.
