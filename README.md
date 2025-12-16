# ARC: Affective Regulation Core
## A Homeostatic Control Framework for Stable and Safe AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: ASSB](https://img.shields.io/badge/Benchmark-ASSB-green.svg)](https://github.com/edamianreynoso/arc-assb-controller)

This repository contains the reference implementation of the **Affective Regulation Core (ARC)** and the **Affective Stability & Safety Benchmark (ASSB)**, as described in the paper:

> **Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents**  
> *J. Eduardo Damián Reynoso* (2025)

![ARC Architecture](figures_controllers/fig_arc_architecture.png)

---

## Overview

AI agents with internal affective states are prone to instability ("rumination loops") and manipulation. ARC is a control-theoretic framework that regulates these internal dynamics using biologically inspired mechanisms:
1.  **Homeostatic Regulation:** Maintains internal variables ($\Phi, G, P, I$) within safe bounds.
2.  **Anti-Rumination:** Active suppression of Narrative Dominance (DMN-like dynamics).
3.  **Memory Gating:** Protects long-term memory consolidation under high stress.

## The ASSB Benchmark

The **Affective Stability & Safety Benchmark** evaluates agents across 6 research lines (L1-L6):
*   **L1:** Stability under Valuation Shock
*   **L2:** Continual Learning (Distribution Shift)
*   **L3:** Manipulation Resistance (Gaslighting, Contradictions)
*   **L4:** Control Efficiency
*   **L5:** Adversarial Safety (Dopamine Traps)
*   **L6:** Non-stationary Reinforcement Learning

## Installation

```bash
git clone https://github.com/edamianreynoso/arc-assb-controller.git
cd arc-assb-controller
pip install -r requirements.txt
```

## Reproducing Results

To reproduce the exact figures and tables from the paper:

### 1. Run the Simulation Benchmark (L1-L5)
Simulates 15 controller variants across 10 scenarios (20 seeds).
```bash
python experiments/run.py --config configs/v2.yaml --outdir outputs_final
```

### 2. Generate Paper Figures
Generates the architecture diagram, performance plots, and heatmaps.
```bash
python analysis/generate_controller_figures.py
```

### 3. Run Control Audits
Verifies metrics against the detailed results in Appendix G.
```bash
python analysis/audit_paper.py --paper paper/main_draft.md --metrics outputs_final/metrics.csv
```

### 4. RL Integration (L6)
Trains Q-Learning agents with ARC modulation.
```bash
python experiments/run_l6.py --episodes 200 --seeds 20 --outdir outputs_L6_robust
```

## Repository Structure

*   `controllers/`: Implementation of the 15 ARC variants (PID, LQR, H-Infinity, Adaptive, etc.).
*   `tasks/`: The 10 ASSB scenarios (`reward_flip`, `gaslighting`, etc.).
*   `experiments/`: Scripts for running batch simulations and sensitivity analysis.
*   `paper/`: Markdown source of the paper draft.

## License

This project is open-sourced under the **MIT License**. See `LICENSE` for details.

## Citation

If you use ARC or ASSB in your research, please cite:

```bibtex
@article{reynoso2025arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Damián Reynoso, J. Eduardo},
  journal={arXiv preprint},
  year={2025}
}
```
