# ARC-ASSB

Code and data for the paper **"Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents"** (Damián Reynoso, 2025).

## What's this?

AI agents with internal "emotional" states tend to get stuck in loops (like rumination in humans) and are easy to manipulate. This repo implements **ARC**, a control system that keeps those internal states stable—inspired by how the prefrontal cortex regulates emotions.

**TL;DR:** We tested 15 different controllers across 10 stress scenarios. The best ones achieve 97% task performance with zero rumination.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce the paper

```bash
# Run the benchmark (takes ~2 hours)
python experiments/run.py --config configs/v2.yaml --outdir outputs_final

# Generate figures
python analysis/generate_controller_figures.py

# RL experiments
python experiments/run_l6.py --episodes 200 --seeds 20 --outdir outputs_L6_robust
```

Or just use the pre-computed data in `outputs_final/` to generate figures directly.

## What's inside

- `controllers/` — 15 controller variants (PID, LQR, H∞, adaptive, etc.)
- `tasks/` — 10 benchmark scenarios (gaslighting, reward hacking, etc.)
- `outputs_final/` — Raw experimental data (3000 traces)
- `paper/` — Paper source

## Citation

```bibtex
@article{damianreynoso2025arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Damián Reynoso, J. Eduardo},
  year={2025}
}
```

## License

MIT
