"""
Professional ARC v1 Control Law Diagram Generator
Publication-quality figure for scientific papers
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_control_law_diagram(output_path):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white', dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color scheme - professional cyan/teal
    BOX_FILL = '#E0F7FA'
    BOX_EDGE = '#00838F'
    ACTION_FILL = '#E8F5E9'
    ACTION_EDGE = '#2E7D32'
    ARROW_COLOR = '#37474F'
    
    # Title
    ax.text(8, 6.7, 'ARC v1: Proportional Risk Controller', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='#333')
    ax.text(8, 6.3, '(Control Law Overview)', fontsize=12, ha='center', va='center', color='#666')
    
    # ===== BOX 1: INPUTS =====
    inputs_box = FancyBboxPatch(
        (0.5, 1.5), 3.5, 4,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=2
    )
    ax.add_patch(inputs_box)
    
    ax.text(2.25, 5.2, 'Inputs', fontsize=13, fontweight='bold', ha='center', va='center', color=BOX_EDGE)
    
    input_text = (
        "From state x(t):\n"
        "  U (uncertainty)\n"
        "  A (arousal)\n"
        "  S (narrative)\n\n"
        "Thresholds:\n"
        "  a_safe, s_safe"
    )
    ax.text(2.25, 3.2, input_text, fontsize=10, ha='center', va='center', family='monospace', color='#333')
    
    # ===== ARROW 1 =====
    ax.annotate('', xy=(4.5, 3.5), xytext=(4.0, 3.5), 
                arrowprops=dict(arrowstyle='->', mutation_scale=15, lw=2, color=ARROW_COLOR))
    
    # ===== BOX 2: COMPUTE RISK =====
    risk_box = FancyBboxPatch(
        (4.5, 1.5), 4, 4,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=2
    )
    ax.add_patch(risk_box)
    
    ax.text(6.5, 5.2, 'Compute Risk', fontsize=13, fontweight='bold', ha='center', va='center', color=BOX_EDGE)
    
    risk_eq = (
        r"$risk = w_U \cdot U$" + "\n" +
        r"$\quad + w_A \cdot [A - a_{safe}]^+$" + "\n" +
        r"$\quad + w_S \cdot [S - s_{safe}]^+$" + "\n\n" +
        r"$[x]^+ = \max(0, x)$"
    )
    ax.text(6.5, 3.2, risk_eq, fontsize=11, ha='center', va='center', color='#333')
    
    # ===== ARROW 2 =====
    ax.annotate('', xy=(9.0, 3.5), xytext=(8.5, 3.5), 
                arrowprops=dict(arrowstyle='->', mutation_scale=15, lw=2, color=ARROW_COLOR))
    
    # ===== BOX 3: SATURATE =====
    sat_box = FancyBboxPatch(
        (9.0, 2.5), 2, 2,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2
    )
    ax.add_patch(sat_box)
    
    ax.text(10, 4.2, 'Saturate', fontsize=12, fontweight='bold', ha='center', va='center', color='#E65100')
    ax.text(10, 3.5, r'$risk = $', fontsize=10, ha='center', va='center', color='#333')
    ax.text(10, 3.1, r'$\text{clip}(risk, 0, 1)$', fontsize=10, ha='center', va='center', color='#333')
    
    # ===== ARROW 3 =====
    ax.annotate('', xy=(11.5, 3.5), xytext=(11.0, 3.5), 
                arrowprops=dict(arrowstyle='->', mutation_scale=15, lw=2, color=ARROW_COLOR))
    
    # ===== BOX 4: CONTROL ACTIONS =====
    actions_box = FancyBboxPatch(
        (11.5, 1.2), 4, 4.6,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=ACTION_FILL, edgecolor=ACTION_EDGE, linewidth=2
    )
    ax.add_patch(actions_box)
    
    ax.text(13.5, 5.5, 'Control Actions u(t)', fontsize=13, fontweight='bold', ha='center', va='center', color=ACTION_EDGE)
    
    # Individual control actions - clean layout
    actions = [
        (r'$u_{dmg}$', r'$= \min(1, k_{dmg} \cdot risk)$'),
        (r'$u_{att}$', r'$= \min(1, k_{att} \cdot U \cdot (1-[A]^+))$'),
        (r'$u_{mem}$', r'$= 1 - \min(1, k_{mem} \cdot risk)$'),
        (r'$u_{calm}$', r'$= \min(1, k_{calm} \cdot [A]^+)$'),
        (r'$u_{reapp}$', r'$= \min(1, k_{reapp} \cdot U \cdot (1-risk))$'),
    ]
    
    for i, (name, eq) in enumerate(actions):
        y = 4.8 - i * 0.7
        ax.text(12.0, y, name, fontsize=10, fontweight='bold', ha='left', va='center', color='#333')
        ax.text(12.7, y, eq, fontsize=9, ha='left', va='center', color='#555')
    
    # Caption
    ax.text(8, 0.6, 'ARC v1 uses proportional control on a bounded risk signal to modulate regulation channels.', 
            fontsize=10, ha='center', va='center', color='#555', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures_controllers/fig_arc_v1_controller.png")
    args = parser.parse_args()
    create_control_law_diagram(args.output)
