"""
Generate the ARC v1 controller diagram used in the paper.

Outputs (default):
  - figures_controllers/fig_arc_v1_controller.png

Usage:
  python analysis/generate_arc_v1_controller_diagram.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    body_fontsize: int = 13,
    facecolor: str = "#DFF7FA",
    edgecolor: str = "#0B7285",
    title_color: str = "#0B7285",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=2.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(patch)

    pad_x = 0.018
    title_text = ax.text(
        x + pad_x,
        y + h - 0.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=title_color,
        zorder=2,
    )

    body_text = ax.text(
        x + pad_x,
        y + h - 0.12,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=body_fontsize,
        family="monospace",
        color="#2B2B2B",
        zorder=2,
        linespacing=1.25,
    )
    title_text.set_clip_path(patch)
    body_text.set_clip_path(patch)


def add_arrow(ax: plt.Axes, *, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "->",
            "lw": 2.5,
            "color": "#2B2B2B",
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=3,
    )


def generate(outpath: Path) -> None:
    fig = plt.figure(figsize=(14, 7), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("white")

    ax.text(
        0.5,
        0.93,
        "ARC v1: Proportional Risk Controller (Control Law Overview)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#2B2B2B",
    )

    inputs_body = (
        "From internal state x(t):\n"
        "  U  (uncertainty)\n"
        "  A  (arousal)\n"
        "  S  (narrative)\n"
        "\n"
        "Safety thresholds:\n"
        "  a_safe, s_safe"
    )
    add_box(ax, x=0.06, y=0.18, w=0.22, h=0.62, title="Inputs", body=inputs_body, body_fontsize=13)

    risk_body = (
        "risk = w_U * U\n"
        "     + w_A * [A - a_safe]^+\n"
        "     + w_S * [S - s_safe]^+\n"
        "\n"
        "[x]^+ = max(0, x)"
    )
    add_box(ax, x=0.33, y=0.18, w=0.27, h=0.62, title="Compute risk(t)", body=risk_body, body_fontsize=13)

    saturate_body = "risk = clip(risk, 0, 1)"
    add_box(ax, x=0.62, y=0.35, w=0.11, h=0.28, title="Saturate", body=saturate_body, body_fontsize=13)

    actions_body = (
        "u_dmg   = min(1, k_dmg * risk)\n"
        "u_att   = min(1, k_att * U *\n"
        "              (1 - [A - a_safe]^+))\n"
        "u_mem   = 1 - min(1,\n"
        "                  k_mem_block * risk)\n"
        "u_calm  = min(1,\n"
        "              k_calm * [A - a_safe]^+)\n"
        "u_reapp = min(1,\n"
        "              k_reapp * U * (1 - risk))"
    )
    add_box(
        ax,
        x=0.74,
        y=0.14,
        w=0.24,
        h=0.70,
        title="Control actions u(t)",
        body=actions_body,
        body_fontsize=11,
    )

    add_arrow(ax, x1=0.28, y1=0.49, x2=0.33, y2=0.49)
    add_arrow(ax, x1=0.60, y1=0.49, x2=0.62, y2=0.49)
    add_arrow(ax, x1=0.73, y1=0.49, x2=0.74, y2=0.49)

    ax.text(
        0.5,
        0.06,
        "ARC v1 uses proportional control on a bounded risk signal to modulate narrative suppression (u_dmg)\n"
        "and other regulation channels (attention, memory gating, calming, reappraisal).",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#4D4D4D",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ARC v1 controller diagram (paper figure)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures_controllers") / "fig_arc_v1_controller.png",
        help="Output PNG path",
    )
    args = parser.parse_args()
    generate(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
