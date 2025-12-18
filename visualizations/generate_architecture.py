import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os
import argparse

def create_architecture_diagram(output_path):
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Style config
    box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#333333', linewidth=2)
    arc_style = dict(boxstyle='round,pad=0.5', facecolor='#e6f7ff', edgecolor='#0077cc', linewidth=2)
    plant_style = dict(boxstyle='round,pad=0.5', facecolor='#fffbe6', edgecolor='#cc8400', linewidth=2)
    
    # --- DRAW BLOCKS ---
    
    # 1. PLANT (Agent Internal State)
    # xy is bottom-left corner
    plant_box = patches.FancyBboxPatch((10, 2), 4, 6, **plant_style)
    ax.add_patch(plant_box)
    ax.text(12, 8.5, "Agent Internal State\n(Plant)", ha='center', va='center', fontsize=14, fontweight='bold', color='#995400')
    
    # Internal Variables
    vars_text = (
        r"$\Phi$: Integration" + "\n" +
        r"$G$: Global Workspace" + "\n" +
        r"$P$: Precision" + "\n" +
        r"$S$: Narrative (DMN)" + "\n" +
        r"$A$: Arousal" + "\n" +
        r"$V$: Valence" + "\n" +
        r"$M$: Memory"
    )
    ax.text(12, 5, vars_text, ha='center', va='center', fontsize=12, linespacing=1.8)

    # 2. ARC CONTROLLER (The Wrapper)
    # Large container
    arc_box = patches.FancyBboxPatch((2, 1), 6, 8, **arc_style)
    ax.add_patch(arc_box)
    ax.text(5, 9.3, "Affective Regulation Core (ARC)", ha='center', va='center', fontsize=16, fontweight='bold', color='#005580')
    
    # 2a. Observer / Monitor
    ax.add_patch(patches.Rectangle((2.5, 6.5), 5, 1.5, fc='white', ec='#0077cc', lw=1.5))
    ax.text(5, 7.25, "1. State Monitor\n(Check $A > a_{safe}$, $S > s_{safe}$)", ha='center', va='center', fontsize=11)
    
    # 2b. Risk Computer
    ax.add_patch(patches.Rectangle((2.5, 4.5), 5, 1.5, fc='white', ec='#0077cc', lw=1.5))
    ax.text(5, 5.25, r"2. Risk Computing" + "\n" + r"$Risk = w_u U + w_a [A]_{rise} + w_s [S]_{rise}$", ha='center', va='center', fontsize=11)
    
    # 2c. Controller / Actions
    ax.add_patch(patches.Rectangle((2.5, 2.0), 5, 2.0, fc='white', ec='#0077cc', lw=1.5))
    ax.text(5, 3.0, "3. Control Policy $\mathbf{u}(t)$", ha='center', va='center', fontsize=12, fontweight='bold')
    # Actions list
    actions = [
        ("u_dmg (DMN Suppress)", "#ff4444"),
        ("u_att (Attention)", "#00cc66"),
        ("u_mem (Mem Gate)", "#aa00ff"),
        ("u_calm (Calm)", "#00bbff")
    ]
    
    # --- ARROWS & CONNECTIONS ---
    
    props = dict(arrowstyle='->,head_width=0.4,head_length=0.8', lw=2, color='#444444')
    
    # Feedback Loop: State -> Monitor
    ax.annotate("", xy=(2, 7.25), xytext=(12, 7.25), arrowprops=dict(arrowstyle='->', lw=2, color='#444444', connectionstyle="angle,angleA=180,angleB=90,rad=10"))
    # Actually manual path is better for complex routing
    # Upper feedback path
    ax.add_patch(patches.FancyArrowPatch((12, 8), (12, 9.5), arrowstyle='-', lw=2, color='#444444')) # Up from plant
    ax.add_patch(patches.FancyArrowPatch((12, 9.5), (5, 9.5), arrowstyle='-', lw=2, color='#444444')) # Left
    ax.add_patch(patches.FancyArrowPatch((5, 9.5), (5, 8.0), arrowstyle='->', lw=2, mutation_scale=20, color='#444444')) # Down to ARC title area? No, to Monitor
    
    # Fix: Direct line State -> Monitor
    # Left from Plant to Monitor
    ax.annotate("State Observation $\mathbf{x}(t)$", xy=(7.5, 7.25), xytext=(10, 7.25), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#444444'), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Monitor -> Risk
    ax.annotate("", xy=(5, 6.0), xytext=(5, 6.5), arrowprops=dict(arrowstyle='->', lw=2, color='#444444'))
    
    # Risk -> Control
    ax.annotate("Risk Signal", xy=(5, 4.0), xytext=(5, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#444444'),
                ha='right', va='center', rotation=90)

    # Control -> Plant
    # Output arrows from Control box
    y_start = 3.0
    
    # Draw individual action arrows
    # u_dmg
    ax.annotate(r"$u_{dmg}$", xy=(10, 5.5), xytext=(7.5, 3.5), 
                arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2), fontsize=10, color='#ff4444', weight='bold')
    
    # u_calm
    ax.annotate(r"$u_{calm}$", xy=(10, 4.5), xytext=(7.5, 3.0), 
                arrowprops=dict(arrowstyle='->', color='#00bbff', lw=2), fontsize=10, color='#00bbff', weight='bold')

    # u_att
    ax.annotate(r"$u_{att}$", xy=(10, 6.5), xytext=(7.5, 2.5), 
                arrowprops=dict(arrowstyle='->', color='#00cc66', lw=2), fontsize=10, color='#00cc66', weight='bold')

    # Exogenous Inputs
    ax.annotate("Exogenous Inputs\n(Reward, PE)", xy=(0.5, 7.25), xytext=(2.5, 7.25),
                arrowprops=dict(arrowstyle='<-', lw=2, color='#444444'), ha='center', fontsize=11)

    # Legend/Caption
    ax.text(8, 0.5, "Figure 1: ARC Control Loop Architecture. The controller monitors the agent's internal state\nand applies homeostatic regulation actions to maintain stability.", 
            ha='center', va='center', fontsize=12, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures_controllers/fig_arc_architecture_v2.png")
    args = parser.parse_args()
    
    create_architecture_diagram(args.output)
