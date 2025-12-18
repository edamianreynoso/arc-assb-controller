import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def create_control_law_diagram(output_path):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Styles
    box_style = dict(boxstyle='round,pad=1', facecolor='#e0f7fa', edgecolor='#006064', linewidth=2)
    saturate_style = dict(boxstyle='round,pad=0.5', facecolor='#b2ebf2', edgecolor='#006064', linewidth=2)
    action_style = dict(boxstyle='round,pad=1', facecolor='#e1f5fe', edgecolor='#01579b', linewidth=2)
    
    # 1. INPUTS Box (Left)
    input_box = patches.FancyBboxPatch((1, 1), 4, 8, **box_style)
    ax.add_patch(input_box)
    
    ax.text(3, 8, "Inputs", ha='center', va='center', fontsize=16, fontweight='bold', color='#006064')
    input_text = (
        "From internal state x(t):\n"
        "  U   (uncertainty)\n"
        "  A   (arousal)\n"
        "  S   (narrative)\n\n"
        "Safety thresholds:\n"
        "  a_safe, s_safe"
    )
    ax.text(3, 5, input_text, ha='center', va='center', fontsize=12, family='monospace')

    # Arrow 1: Inputs -> Risk
    ax.annotate("", xy=(6.5, 5), xytext=(5, 5), arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    # 2. COMPUTE RISK Box (Middle)
    risk_box = patches.FancyBboxPatch((6.5, 1), 5, 8, **box_style)
    ax.add_patch(risk_box)
    
    ax.text(9, 8, "Compute risk(t)", ha='center', va='center', fontsize=16, fontweight='bold', color='#006064')
    risk_eq = (
        r"risk = w_U * U" + "\n" +
        r"     + w_A * [A - a_safe]^+" + "\n" +
        r"     + w_S * [S - s_safe]^+" + "\n\n" +
        r"[x]^+ = max(0, x)"
    )
    ax.text(9, 5, risk_eq, ha='center', va='center', fontsize=12, family='monospace')

    # Arrow 2: Risk -> Saturate
    ax.annotate("", xy=(12.5, 5), xytext=(11.5, 5), arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    # 3. SATURATE Box (Middle-Right, smaller)
    # Fixed spacing: moved to x=12.5
    sat_box = patches.FancyBboxPatch((12.5, 3.5), 2.5, 3, **saturate_style)
    ax.add_patch(sat_box)
    
    ax.text(13.75, 5.5, "Saturate", ha='center', va='center', fontsize=14, fontweight='bold', color='#006064')
    ax.text(13.75, 4.5, "clip(risk, 0, 1)", ha='center', va='center', fontsize=11, family='monospace')

    # Arrow 3: Saturate -> Control Actions
    ax.annotate("", xy=(16, 5), xytext=(15, 5), arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))

    # 4. CONTROL ACTIONS Box (Right)
    # Fixed spacing: moved to x=16
    actions_box = patches.FancyBboxPatch((16, 1), 3.5, 8, **action_style)
    ax.add_patch(actions_box)
    
    ax.text(17.75, 8, "Control Actions u(t)", ha='center', va='center', fontsize=14, fontweight='bold', color='#01579b')
    
    actions_text = (
        r"$u_{dmg} = \min(1, k_{dmg} \cdot risk)$" + "\n\n" +
        r"$u_{att} = \min(1, k_{att} \cdot U \cdot (1 - [A]_{rise}))$" + "\n\n" +
        r"$u_{mem} = 1 - \min(1, k_{mem} \cdot risk)$" + "\n\n" +
        r"$u_{calm} = \min(1, k_{calm} \cdot [A]_{rise})$" + "\n\n" +
        r"$u_{reapp} = \min(1, k_{reapp} \cdot U \cdot (1-risk))$"
    )
    ax.text(17.75, 4.5, actions_text, ha='center', va='center', fontsize=10)

    # Title
    ax.text(10, 9.5, "ARC v1: Proportional Risk Controller (Control Law Overview)", 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved control law diagram to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures_controllers/fig_arc_v1_controller.png")
    args = parser.parse_args()
    
    create_control_law_diagram(args.output)
