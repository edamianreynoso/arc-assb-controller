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

Or just use the pre-computed metrics in `outputs_final/metrics.csv` to generate figures directly. Full raw traces (~3000 CSV, ~84MB) are provided as a `traces.zip` release asset.

## What's inside

- `controllers/` — 15 controller variants (PID, LQR, H∞, adaptive, etc.)
- `tasks/` — 10 benchmark scenarios (gaslighting, reward hacking, etc.)
- `outputs_final/` - Precomputed metrics (`metrics.csv`); full raw traces via Releases (`traces.zip`)
- `paper/` — Paper source

## Citation

```bibtex
@article{damianreynoso2025arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Reynoso, J. Eduardo Damián},
  year={2025}
}
```

## License

Apache-2.0 (see `LICENSE`)
