# ARC-ASSB: Affective Regulation Core & Stability Benchmark

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This repository contains the **reference implementation** for the paper:

> **Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents**
> 
> J. Eduardo Damián Reynoso, 14 December 2025

**Key Results:**
- ARC achieves 97% performance with zero rumination (vs 30% baseline)
- Meta-control reduces control effort by 21%
- +50% improvement in transfer learning for non-stationary environments

## Quick Start

```bash
# Clone
git clone https://github.com/edamianreynoso/arc-assb.git
cd arc-assb

# Install
pip install -r requirements.txt

# Run L1-L5 experiments
python -m experiments.run --config configs/v2.yaml --outdir outputs

# Run L6 RL validation
python -m experiments.run_l6 --episodes 200 --seeds 20 --outdir outputs_L6
```

## Repository Structure

```
arc-assb/
├── paper/              # Full paper (Markdown)
├── sim/                # State dynamics simulation
├── controllers/        # ARC v1 reference implementation
├── agents/             # Q-Learning + ARC integration
├── envs/               # GridWorld environments
├── metrics/            # ASSB metrics (RT, RI, NDR, etc.)
├── tasks/              # Perturbation scenarios
├── configs/            # Experiment configurations
├── figures_L6/         # Publication figures
└── analysis/           # Statistical analysis results
```

## Key Components

### ASSB Benchmark Metrics

| Metric | Description |
|--------|-------------|
| **PerfMean** | Average performance |
| **RT** | Recovery time post-shock |
| **RI** | Rumination index |
| **NDR** | Narrative dominance ratio |
| **ControlEffort** | Average control magnitude |

### ARC Controller

ARC v1 implements proportional control based on a risk signal:

```python
risk = w_U * U + w_A * [A - a_safe]+ + w_S * [S - s_safe]+
u_dmg = k_dmg * risk  # DMN suppression
```

See `controllers/controllers.py` for the full implementation.

## Citation

If you use this code, please cite:

```bibtex
@article{damian2025arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Damián Reynoso, J. Eduardo},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Commercial Use

This repository contains the reference implementation for research purposes.

For **production deployments**, **enterprise support**, or **clinical applications**, please contact:

📧 **edamianreynoso@gmail.com**

We offer:
- ✅ ARC Pro (optimized controllers)
- ✅ Enterprise support & SLAs
- ✅ Clinical compliance audits
- ✅ Custom integration

## License

Apache License 2.0 - See [LICENSE](LICENSE)
