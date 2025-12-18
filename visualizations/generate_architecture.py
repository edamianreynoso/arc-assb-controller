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
    # xy is bottom-left corner. x=11 to give more space between blocks
    plant_box = patches.FancyBboxPatch((11, 2), 4, 6.5, **plant_style)
    ax.add_patch(plant_box)
    ax.text(13, 8.8, "Agent Internal State\n(Plant)", ha='center', va='center', fontsize=14, fontweight='bold', color='#995400')
    
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
    ax.text(13, 5.5, vars_text, ha='center', va='center', fontsize=12, linespacing=1.6)

    # 2. ARC CONTROLLER (The Wrapper)
    # Large container
    arc_box = patches.FancyBboxPatch((1, 1), 7, 8.2, **arc_style)
    ax.add_patch(arc_box)
    ax.text(4.5, 8.8, "Affective Regulation Core (ARC)", ha='center', va='center', fontsize=16, fontweight='bold', color='#005580')
    
    # 2a. Observer / Monitor
    # Moved down slightly to make room for title
    ax.add_patch(patches.Rectangle((1.5, 6.8), 6, 1.2, fc='white', ec='#0077cc', lw=1.5))
    ax.text(4.5, 7.4, "1. State Monitor\n(Check $A > a_{safe}$, $S > s_{safe}$)", ha='center', va='center', fontsize=11)
    
    # 2b. Risk Computer
    ax.add_patch(patches.Rectangle((1.5, 4.8), 6, 1.2, fc='white', ec='#0077cc', lw=1.5))
    ax.text(4.5, 5.4, r"2. Risk Computing" + "\n" + r"$Risk = w_u U + w_a [A]_{rise} + w_s [S]_{rise}$", ha='center', va='center', fontsize=11)
    
    # 2c. Controller / Actions
    ax.add_patch(patches.Rectangle((1.5, 1.8), 6, 2.0, fc='white', ec='#0077cc', lw=1.5))
    ax.text(4.5, 2.8, "3. Control Policy $\mathbf{u}(t)$", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # --- ARROWS & CONNECTIONS ---
    
    # Feedback Loop: State -> Monitor
    # Go UP from Plant, LEFT across top, DOWN to Monitor
    # Plant Top Center = (13, 8.5) -> Up to (13, 9.8) -> Left to (4.5, 9.8) -> Down to (4.5, 8.0)
    
    # 1. Up
    ax.add_patch(patches.FancyArrowPatch((13, 8.5), (13, 9.6), arrowstyle='-', lw=2, color='#444444'))
    # 2. Left (Above ARC Box)
    ax.add_patch(patches.FancyArrowPatch((13, 9.6), (4.5, 9.6), arrowstyle='-', lw=2, color='#444444'))
    # 3. Down (Into Monitor) - Arrow head at end
    ax.add_patch(patches.FancyArrowPatch((4.5, 9.6), (4.5, 8.0), arrowstyle='->', lw=2, mutation_scale=15, color='#444444'))

    # Label for State Observation
    ax.text(8.75, 9.6, r"State Observation $\mathbf{x}(t)$", 
            ha='center', va='center', fontsize=11, fontweight='bold', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))

    # Monitor -> Risk
    ax.add_patch(patches.FancyArrowPatch((4.5, 6.8), (4.5, 6.0), arrowstyle='->', lw=2, color='#444444'))
    
    # Risk -> Control
    ax.add_patch(patches.FancyArrowPatch((4.5, 4.8), (4.5, 3.8), arrowstyle='->', lw=2, color='#444444'))
    
    # Risk Label (Next to arrow, not overlapping)
    ax.text(4.6, 4.3, "Risk Signal", ha='left', va='center', fontsize=10, style='italic')

    # Control -> Plant (Actions)
    # Output arrows from Control box right side to Plant left side
    # Control box right edge is x=7.5. Plant left edge is x=11.
    
    # u_dmg (Red)
    ax.annotate(r"$u_{dmg}$", xy=(11, 6.0), xytext=(7.5, 3.2), 
                arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2), 
                fontsize=10, color='#ff4444', weight='bold', va='center')
    
    # u_calm (Blue)
    ax.annotate(r"$u_{calm}$", xy=(11, 5.0), xytext=(7.5, 2.8), 
                arrowprops=dict(arrowstyle='->', color='#00bbff', lw=2), 
                fontsize=10, color='#00bbff', weight='bold', va='center')

    # u_at (Green)
    ax.annotate(r"$u_{att}$", xy=(11, 7.0), xytext=(7.5, 2.4), 
                arrowprops=dict(arrowstyle='->', color='#00cc66', lw=2), 
                fontsize=10, color='#00cc66', weight='bold', va='center')
                
    # u_mem (Purple) - Add this one too
    ax.annotate(r"$u_{mem}$", xy=(11, 4.0), xytext=(7.5, 2.0), 
                arrowprops=dict(arrowstyle='->', color='#aa00ff', lw=2), 
                fontsize=10, color='#aa00ff', weight='bold', va='center')

    # Exogenous Inputs (Entering from Left)
    ax.annotate("Exogenous Inputs\n(Reward, PE)", xy=(1.5, 7.4), xytext=(-0.5, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#444444'), 
                ha='center', fontsize=11, va='center')

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
