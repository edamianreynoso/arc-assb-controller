"""
Generate sensitivity figures for the ARC paper (white background).

Outputs (written to --outdir, default: analysis/):
  - sensitivity_controller.png
  - sensitivity_scenario.png
  - sensitivity_variance.png

Usage:
  python analysis/generate_sensitivity_figures.py --metrics outputs_final/metrics.csv --outdir analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


CONTROLLERS = [
    "arc_v1",
    "arc_v2_hier",
    "arc_v3_meta",
    "naive_calm",
    "no_control",
    "perf_optimized",
]

ARC_CONTROLLERS = {"arc_v1", "arc_v2_hier", "arc_v3_meta"}

COLORS = {
    "perf": "#1f7a4f",
    "ri": "#6a3d9a",
    "rt": "#b36b00",
}


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()


def plot_controller_comparison(df: pd.DataFrame, outdir: Path) -> None:
    df_f = df[df["controller"].isin(CONTROLLERS)].copy()

    agg = (
        df_f.groupby("controller", as_index=True)
        .agg(
            PerfMean_mean=("PerfMean", "mean"),
            PerfMean_std=("PerfMean", "std"),
            RI_mean=("RI", "mean"),
            RI_std=("RI", "std"),
            RT_mean=("RT", "mean"),
            RT_std=("RT", "std"),
        )
        .reindex(CONTROLLERS)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(CONTROLLERS))
    labels = CONTROLLERS

    # Performance
    ax = axes[0]
    ax.bar(
        x,
        agg["PerfMean_mean"],
        yerr=agg["PerfMean_std"],
        capsize=3,
        color=COLORS["perf"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title("Performance by Controller")
    ax.set_ylabel("Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)

    # Rumination Index
    ax = axes[1]
    ax.bar(
        x,
        agg["RI_mean"],
        yerr=agg["RI_std"],
        capsize=3,
        color=COLORS["ri"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title("Rumination Index by Controller")
    ax.set_ylabel("Rumination Index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(-0.05, max(0.1, float(agg["RI_mean"].max()) * 1.15))

    # Recovery Time
    ax = axes[2]
    ax.bar(
        x,
        agg["RT_mean"],
        yerr=agg["RT_std"],
        capsize=3,
        color=COLORS["rt"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title("Recovery Time by Controller")
    ax.set_ylabel("Recovery Time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle("Controller Performance Comparison", fontsize=16)
    fig.tight_layout()
    _savefig(outdir / "sensitivity_controller.png")


def plot_scenario_difficulty(df: pd.DataFrame, outdir: Path, controller: str = "arc_v1") -> None:
    df_f = df[df["controller"] == controller].copy()
    if df_f.empty:
        raise ValueError(f"No rows for controller={controller!r} in metrics file.")

    agg = (
        df_f.groupby("scenario", as_index=True)
        .agg(
            PerfMean_mean=("PerfMean", "mean"),
            PerfMean_std=("PerfMean", "std"),
            RI_mean=("RI", "mean"),
            RI_std=("RI", "std"),
            RT_mean=("RT", "mean"),
            RT_std=("RT", "std"),
        )
        .sort_values("PerfMean_mean", ascending=True)
    )

    scenarios = list(agg.index)
    y = np.arange(len(scenarios))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.patch.set_facecolor("white")

    axes[0].barh(
        y,
        agg["PerfMean_mean"],
        xerr=agg["PerfMean_std"],
        capsize=3,
        color=COLORS["perf"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_title("Performance (Higher is Better)")
    axes[0].set_xlabel("Performance")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(scenarios, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.05)

    axes[1].barh(
        y,
        agg["RI_mean"],
        xerr=agg["RI_std"],
        capsize=3,
        color=COLORS["ri"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].set_title("Rumination (Lower is Better)")
    axes[1].set_xlabel("RI")
    # Auto-scale x-axis for RI but ensuring it's not empty if values are small
    max_ri = agg["RI_mean"].max()
    axes[1].set_xlim(0, max(0.2, max_ri * 1.2))

    axes[2].barh(
        y,
        agg["RT_mean"],
        xerr=agg["RT_std"],
        capsize=3,
        color=COLORS["rt"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[2].set_title("Recovery Time (Lower is Better)")
    axes[2].set_xlabel("Steps")
    
    fig.suptitle(f"Scenario Difficulty Analysis (Controller: {controller})", fontsize=16)
    fig.tight_layout()
    _savefig(outdir / "sensitivity_scenario.png")


def plot_variance(df: pd.DataFrame, outdir: Path) -> None:
    df_f = df[df["controller"].isin(CONTROLLERS)].copy()

    data = [df_f[df_f["controller"] == c]["PerfMean"].dropna().to_numpy() for c in CONTROLLERS]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        data,
        labels=CONTROLLERS,
        patch_artist=True,
        showfliers=False,
    )

    for box, controller in zip(bp["boxes"], CONTROLLERS):
        box.set_facecolor("#00bcd4" if controller in ARC_CONTROLLERS else "#b24a4a")
        box.set_edgecolor("black")
        box.set_linewidth(0.8)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("#f1c40f")
        median.set_linewidth(1.2)

    ax.set_title("Performance Distribution by Controller (across all seeds)", fontsize=14)
    ax.set_ylabel("Performance Mean")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _savefig(outdir / "sensitivity_variance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate white-background sensitivity figures")
    parser.add_argument("--metrics", type=Path, default=Path("outputs_final/metrics.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("analysis"))
    parser.add_argument("--arc-controller", type=str, default="arc_v1")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    required_cols = {"scenario", "controller", "PerfMean", "RI", "RT"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Metrics file missing required columns: {missing}")

    plot_controller_comparison(df, args.outdir)
    plot_scenario_difficulty(df, args.outdir, controller=args.arc_controller)
    plot_variance(df, args.outdir)

    print(f"Saved sensitivity figures to: {args.outdir}")


if __name__ == "__main__":
    main()

