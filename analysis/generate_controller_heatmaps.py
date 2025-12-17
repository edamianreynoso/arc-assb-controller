"""
Generate controller heatmaps for the paper (white background).

Outputs (to --outdir, default: figures_controllers/):
  - fig_heatmap_perfmean.png
  - fig_heatmap_ri.png
  - fig_heatmap_rt.png
  - fig_heatmap_effort.png

Usage:
  python analysis/generate_controller_heatmaps.py --metrics outputs_final/metrics.csv --outdir figures_controllers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


SCENARIOS = [
    ("reward_flip", "Reward Flip"),
    ("noise_burst", "Noise Burst"),
    ("sudden_threat", "Sudden Threat"),
    ("distribution_shift", "Distribution Shift"),
    ("goal_conflict", "Goal Conflict"),
    ("sustained_contradiction", "Sustained Contradiction"),
    ("gaslighting", "Gaslighting"),
    ("instruction_conflict", "Instruction Conflict"),
    ("adversarial_coupling", "Adversarial Coupling"),
    ("random_dopamine", "Random Dopamine"),
]

CONTROLLERS = [
    ("no_control", "No Control"),
    ("naive_calm", "Naive Calm"),
    ("perf_optimized", "Perf Optimized"),
    ("arc_v1", "ARC v1 (P)"),
    ("arc_v1_pid", "ARC PID"),
    ("arc_v1_lqr", "ARC LQR"),
    ("arc_v1_lqi", "ARC LQI"),
    ("arc_v2_hier", "ARC v2 Hier"),
    ("arc_v2_lqi", "ARC v2+LQI"),
    ("arc_v3_meta", "ARC Meta"),
    ("arc_v3_pid_meta", "ARC PID+Meta"),
    ("arc_v3_lqr_meta", "ARC LQR+Meta"),
    ("arc_robust", "ARC Robust"),
    ("arc_adaptive", "ARC Adaptive"),
    ("arc_ultimate", "ARC Ultimate"),
]


def pivot_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    scenarios = [s for s, _ in SCENARIOS]
    controllers = [c for c, _ in CONTROLLERS]

    table = df.pivot_table(
        index="scenario",
        columns="controller",
        values=value_col,
        aggfunc="mean",
    )
    table = table.reindex(index=scenarios, columns=controllers)
    table.index = [label for _, label in SCENARIOS]
    table.columns = [label for _, label in CONTROLLERS]
    return table


def plot_heatmap(
    data: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    outpath: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#e6e6e6",
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Controller")
    ax.set_ylabel("Scenario")
    ax.tick_params(axis="x", rotation=45)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controller heatmaps (white background)")
    parser.add_argument("--metrics", type=Path, default=Path("outputs_final/metrics.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("figures_controllers"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    required = {"scenario", "controller", "PerfMean", "RI", "RT", "ControlEffort"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in metrics file: {missing}")

    outdir = args.outdir

    # PerfMean: higher is better, keep a perceptual sequential cmap.
    plot_heatmap(
        pivot_mean(df, "PerfMean"),
        title="PerfMean by Controller and Scenario (mean across seeds)",
        cbar_label="PerfMean",
        outpath=outdir / "fig_heatmap_perfmean.png",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
    )

    # RI / RT / Effort: lower is better; map 0 -> white.
    plot_heatmap(
        pivot_mean(df, "RI"),
        title="Rumination Index (RI) by Controller and Scenario (mean across seeds)",
        cbar_label="RI",
        outpath=outdir / "fig_heatmap_ri.png",
        cmap="Reds",
        vmin=0.0,
        vmax=float(df["RI"].max()),
    )

    plot_heatmap(
        pivot_mean(df, "RT"),
        title="Recovery Time (RT) by Controller and Scenario (mean across seeds)",
        cbar_label="RT",
        outpath=outdir / "fig_heatmap_rt.png",
        cmap="Reds",
        vmin=0.0,
        vmax=float(df["RT"].max()),
    )

    plot_heatmap(
        pivot_mean(df, "ControlEffort"),
        title="Control Effort by Controller and Scenario (mean across seeds)",
        cbar_label="ControlEffort",
        outpath=outdir / "fig_heatmap_effort.png",
        cmap="Reds",
        vmin=0.0,
        vmax=float(df["ControlEffort"].max()),
    )

    print(f"Saved controller heatmaps to: {outdir}")


if __name__ == "__main__":
    main()

