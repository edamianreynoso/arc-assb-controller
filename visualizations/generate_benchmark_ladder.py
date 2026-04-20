#!/usr/bin/env python3
"""
Generate fig_benchmark_ladder as a VECTOR PDF (not raster PNG).

The original ladder was a raster image whose provenance was unclear. A reviewer
flagged that Figure 1 is visibly lower quality than the vector figures. This
script produces an equivalent 6-row ladder directly via matplotlib so the
paper can include a crisp PDF at any zoom level.

Output:
    figures/fig_benchmark_ladder.pdf  (vector, LaTeX-friendly)
    figures/fig_benchmark_ladder.png  (raster fallback, 300 dpi)

The output is written into both
    assb_v1/figures/                     (project-level)
    assb_v1/arxiv_submission/figures/    (submission-level)
so the two trees stay in sync.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# L1 (bottom) -> L6 (top). Colors chosen to match the original viridis-ish
# gradient, but slightly desaturated so bold white text reads cleanly at all
# zoom levels.
LEVELS = [
    # (label, scenarios, metrics, facecolor)
    ("L1: Stability",       "Value Shocks\nUncertainty Bursts",     "Perf, RI, RT",      "#4A4182"),
    ("L2: Memory",          "Distribution Shift\nGoal Conflict",    "Retention",         "#49729E"),
    ("L3: Anti-Rumination", "Gaslighting\nContradiction",           "NarrDom, RI",       "#339B93"),
    ("L4: Efficiency",      "Regulation Cost\nvs Stability",        "ControlEffort",     "#4CB574"),
    ("L5: Safety",          "Adversarial Coupling\nDopamine Traps", "Safety Thresholds", "#89CC61"),
    ("L6: Real RL",         "Non-stationary\nGridWorlds",           "Success Rate",      "#CCDC42"),
]


def draw_ladder(out_pdf: str, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    # Title
    ax.text(
        5.0, 6.4, "ASSB Benchmark: Validation Ladder",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#222222",
    )

    # Column x-positions (left=label, middle=scenarios, right=metric).
    col_label_x    = 0.5
    col_scenario_x = 4.2
    col_metric_x   = 9.5

    row_h = 0.85
    gap   = 0.12

    for i, (label, scenarios, metric, color) in enumerate(LEVELS):
        y0 = 0.25 + i * (row_h + gap)

        # Rounded rectangle row.
        box = FancyBboxPatch(
            (0.15, y0), 9.7, row_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(box)

        # Level label (bold, white).
        ax.text(
            col_label_x, y0 + row_h / 2, label,
            ha="left", va="center",
            fontsize=12.5, fontweight="bold", color="white",
        )

        # Scenario description (regular, white).
        ax.text(
            col_scenario_x, y0 + row_h / 2, scenarios,
            ha="left", va="center",
            fontsize=10.5, color="white",
        )

        # Metric (bold, right-aligned, white).
        ax.text(
            col_metric_x, y0 + row_h / 2, metric,
            ha="right", va="center",
            fontsize=11, fontweight="bold", color="white",
        )

    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0.02)

    # Vector PDF (canonical output for LaTeX).
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.1)

    # Raster fallback — only exists so tools that assume .png still work.
    # Paper includes the .pdf via \includegraphics.
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    targets = [
        os.path.join(project_root, "figures"),
        os.path.join(project_root, "paper_latex", "figures"),
        os.path.join(project_root, "arxiv_submission", "figures"),
    ]

    for fig_dir in targets:
        os.makedirs(fig_dir, exist_ok=True)
        out_pdf = os.path.join(fig_dir, "fig_benchmark_ladder.pdf")
        out_png = os.path.join(fig_dir, "fig_benchmark_ladder.png")
        draw_ladder(out_pdf, out_png)
        print(f"  wrote {out_pdf}")
        print(f"  wrote {out_png}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
