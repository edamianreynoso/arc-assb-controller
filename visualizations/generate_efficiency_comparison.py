"""
Generate the L6 efficiency comparison figure (white background).

Outputs:
  - figures_L6/efficiency_comparison.png

Usage:
  python visualizations/generate_efficiency_comparison.py --summary outputs_L6_robust/summary.csv --out figures_L6/efficiency_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
    }
)


COLORS = {
    "ARC": "#00bcd4",
    "Baseline": "#ffa500",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate efficiency comparison plot")
    parser.add_argument("--summary", type=Path, default=Path("outputs_L6_robust/summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures_L6/efficiency_comparison.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    required_cols = {"agent", "env", "episode", "reward_mean", "reward_std"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Summary file missing required columns: {missing}")

    envs = ["GridWorld", "StochasticGridWorld"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor("white")

    agent_map = [
        ("ql_arc", "ARC", COLORS["ARC"]),
        ("ql_baseline", "Baseline", COLORS["Baseline"]),
    ]

    for ax, env in zip(axes, envs):
        env_df = df[df["env"] == env]
        if env_df.empty:
            raise ValueError(f"Missing env={env!r} in summary file.")

        for agent, label, color in agent_map:
            a_df = env_df[env_df["agent"] == agent].sort_values("episode")
            ax.plot(a_df["episode"], a_df["reward_mean"], color=color, linewidth=2.5, label=label)
            ax.fill_between(
                a_df["episode"],
                a_df["reward_mean"] - a_df["reward_std"],
                a_df["reward_mean"] + a_df["reward_std"],
                color=color,
                alpha=0.2,
                linewidth=0,
            )

        ax.set_title(f"{env}: Both 100% success, but who is FASTER?", fontsize=14)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("Reward (higher = faster to goal)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

