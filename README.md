# ARC-ASSB: Affective Regulation Core & Stability Benchmark

Reference implementation and artifacts for the paper in `paper/main_draft.md`.

- Paper (draft): `paper/main_draft.md`
- License: Apache 2.0 (`LICENSE`)
- arXiv: https://arxiv.org/abs/XXXX.XXXXX (placeholder)

## Highlights

- High performance with bounded affective dynamics (anti-rumination)
- Meta-control reduces control effort while maintaining stability
- RL integration improves transfer in non-stationary environments

## Install

```bash
pip install -r requirements.txt
```

## Reproduce (Appendix A)

```bash
# L1-L5: Simulation benchmark (15 controllers)
python experiments/run.py --config configs/v2.yaml --outdir outputs_final

# Controller-comparison figures
python analysis/generate_controller_figures.py

# Ablation study (ARC components)
python experiments/run_ablation.py --config configs/v2.yaml --outdir outputs_ablation --seeds 20

# L6: RL validation
python experiments/run_l6.py --episodes 200 --seeds 20 --outdir outputs_L6_robust

# L6 figures
python visualizations/paper_figures.py --data outputs_L6_robust --output figures_L6
```

## Controller Suite (15)

Implemented in `controllers/controllers.py`:

- Baselines: `no_control`, `naive_calm`, `perf_optimized`
- ARC v1: `arc_v1`, `arc_v1_pid`, `arc_v1_lqr`, `arc_v1_lqi`
- ARC v2: `arc_v2_hier`, `arc_v2_lqi`
- ARC v3: `arc_v3_meta`, `arc_v3_pid_meta`, `arc_v3_lqr_meta`
- Robust/adaptive: `arc_robust` (H-infinity inspired), `arc_adaptive`, `arc_ultimate` (MPC+LQI+Meta)

## Citation

```bibtex
@article{reynoso2025arc,
  title={Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents},
  author={Reynoso, J. Eduardo Damian},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Commercial / Support

For production deployments or support, contact `edamianreynoso@gmail.com`.
