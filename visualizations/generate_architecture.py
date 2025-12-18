"""
Professional ARC Architecture Diagram Generator
Publication-quality figure for scientific papers
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram(output_path):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white', dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color scheme - professional blues and grays
    ARC_FILL = '#E3F2FD'      # Light blue
    ARC_EDGE = '#1565C0'      # Dark blue
    PLANT_FILL = '#FFF8E1'    # Light amber
    PLANT_EDGE = '#F57C00'    # Orange
    COMPONENT_FILL = '#FFFFFF'
    ARROW_COLOR = '#37474F'   # Dark gray
    
    # ===== PLANT BOX (Right side) =====
    plant = FancyBboxPatch(
        (9.5, 1.5), 3.8, 6,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=PLANT_FILL, edgecolor=PLANT_EDGE, linewidth=2.5
    )
    ax.add_patch(plant)
    
    # Plant title
    ax.text(11.4, 7.2, 'Agent Internal State', fontsize=13, fontweight='bold', 
            ha='center', va='center', color=PLANT_EDGE)
    ax.text(11.4, 6.7, '(Plant)', fontsize=11, ha='center', va='center', color=PLANT_EDGE)
    
    # State variables - clean list
    states = [
        ('Φ', 'Integration (IIT)'),
        ('G', 'Global Workspace'),
        ('P', 'Predictive Precision'),
        ('S', 'Narrative Intensity'),
        ('A', 'Arousal'),
        ('V', 'Valence'),
        ('M', 'Memory'),
    ]
    for i, (sym, desc) in enumerate(states):
        y = 5.8 - i * 0.6
        ax.text(10.0, y, f'{sym}:', fontsize=11, fontweight='bold', ha='left', va='center', color='#333')
        ax.text(10.5, y, desc, fontsize=10, ha='left', va='center', color='#555')
    
    # ===== ARC CONTROLLER BOX (Left side) =====
    arc = FancyBboxPatch(
        (0.5, 1.5), 7.5, 6,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=ARC_FILL, edgecolor=ARC_EDGE, linewidth=2.5
    )
    ax.add_patch(arc)
    
    # ARC title
    ax.text(4.25, 7.2, 'Affective Regulation Core (ARC)', fontsize=14, fontweight='bold',
            ha='center', va='center', color=ARC_EDGE)
    
    # Component 1: State Monitor
    monitor = FancyBboxPatch(
        (1, 5.5), 6.5, 1.2,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COMPONENT_FILL, edgecolor=ARC_EDGE, linewidth=1.5
    )
    ax.add_patch(monitor)
    ax.text(4.25, 6.1, '1. State Monitor', fontsize=11, fontweight='bold', ha='center', va='center', color='#333')
    ax.text(4.25, 5.7, r'Check: $A > a_{safe}$, $S > s_{safe}$', fontsize=10, ha='center', va='center', color='#555')
    
    # Component 2: Risk Computing
    risk = FancyBboxPatch(
        (1, 3.8), 6.5, 1.2,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COMPONENT_FILL, edgecolor=ARC_EDGE, linewidth=1.5
    )
    ax.add_patch(risk)
    ax.text(4.25, 4.4, '2. Risk Computing', fontsize=11, fontweight='bold', ha='center', va='center', color='#333')
    ax.text(4.25, 4.0, r'$risk = w_u U + w_a [A-a_{safe}]^+ + w_s [S-s_{safe}]^+$', fontsize=10, ha='center', va='center', color='#555')
    
    # Component 3: Control Policy
    policy = FancyBboxPatch(
        (1, 1.8), 6.5, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COMPONENT_FILL, edgecolor=ARC_EDGE, linewidth=1.5
    )
    ax.add_patch(policy)
    ax.text(4.25, 3.05, '3. Control Policy', fontsize=11, fontweight='bold', ha='center', va='center', color='#333')
    ax.text(4.25, 2.55, r'$\mathbf{u}(t) = [u_{dmg}, u_{att}, u_{mem}, u_{calm}, u_{reapp}]$', fontsize=10, ha='center', va='center', color='#555')
    ax.text(4.25, 2.1, 'DMN suppression • Attention • Memory gate • Calming • Reappraisal', fontsize=8, ha='center', va='center', color='#777')
    
    # ===== ARROWS =====
    arrow_style = dict(arrowstyle='->', mutation_scale=15, lw=2, color=ARROW_COLOR)
    
    # Feedback loop: Plant -> ARC (top)
    # Up from plant
    ax.annotate('', xy=(11.4, 7.8), xytext=(11.4, 7.5), arrowprops=dict(arrowstyle='-', lw=2, color=ARROW_COLOR))
    ax.plot([11.4, 11.4], [7.5, 8.3], lw=2, color=ARROW_COLOR)
    ax.plot([11.4, 4.25], [8.3, 8.3], lw=2, color=ARROW_COLOR)
    ax.annotate('', xy=(4.25, 7.5), xytext=(4.25, 8.3), arrowprops=dict(arrowstyle='->', mutation_scale=15, lw=2, color=ARROW_COLOR))
    
    # Label for feedback
    ax.text(7.8, 8.5, r'State Observation $\mathbf{x}(t)$', fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    
    # Monitor -> Risk
    ax.annotate('', xy=(4.25, 5.0), xytext=(4.25, 5.5), arrowprops=arrow_style)
    
    # Risk -> Policy
    ax.annotate('', xy=(4.25, 3.3), xytext=(4.25, 3.8), arrowprops=arrow_style)
    
    # Policy -> Plant (Control actions)
    # Main arrow
    ax.annotate('', xy=(9.5, 4.5), xytext=(7.5, 2.5), arrowprops=dict(arrowstyle='->', mutation_scale=15, lw=2.5, color='#D32F2F'))
    ax.text(8.7, 3.7, r'$\mathbf{u}(t)$', fontsize=12, fontweight='bold', ha='center', va='center', color='#D32F2F',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    
    # Exogenous inputs (left side)
    ax.annotate('', xy=(1, 6.1), xytext=(0.2, 6.1), arrowprops=dict(arrowstyle='->', mutation_scale=12, lw=1.5, color=ARROW_COLOR))
    ax.text(0.35, 6.5, 'Exogenous\nInputs', fontsize=9, ha='center', va='center', color='#555')
    ax.text(0.35, 5.7, '(R, PE)', fontsize=8, ha='center', va='center', color='#777')
    
    # Caption
    ax.text(7, 0.6, 'Figure 1: ARC Control Loop Architecture', fontsize=12, fontweight='bold', ha='center', va='center', color='#333')
    ax.text(7, 0.2, 'The controller monitors the agent\'s internal state and applies homeostatic regulation actions.', 
            fontsize=10, ha='center', va='center', color='#555', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures_controllers/fig_arc_architecture_v2.png")
    args = parser.parse_args()
    create_architecture_diagram(args.output)
