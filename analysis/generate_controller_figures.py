"""
Generate comparison figures for all 15 controllers.
Publication-ready figures with UNIFIED professional style.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ============ UNIFIED STYLE SYSTEM ============
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.95,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Professional color palette (matching architecture diagrams)
COLORS = {
    'primary_blue': '#1565C0',
    'primary_orange': '#F57C00', 
    'success_green': '#2E7D32',
    'warning_amber': '#FF8F00',
    'danger_red': '#C62828',
    'neutral_gray': '#616161',
}

# Controller-specific colors (harmonious palette)
CONTROLLER_COLORS = {
    'no_control': '#E53935',      # Red
    'naive_calm': '#EF5350',      # Light red
    'perf_optimized': '#78909C',  # Blue gray
    'arc_v1': '#1E88E5',          # Blue
    'arc_v1_pid': '#7B1FA2',      # Purple
    'arc_v1_lqr': '#00897B',      # Teal
    'arc_v1_lqi': '#43A047',      # Green
    'arc_v2_hier': '#FB8C00',     # Orange
    'arc_v2_lqi': '#F4511E',      # Deep orange
    'arc_v3_meta': '#66BB6A',     # Light green
    'arc_v3_pid_meta': '#FDD835', # Yellow
    'arc_v3_lqr_meta': '#26C6DA', # Cyan
    'arc_robust': '#00695C',      # Dark teal
    'arc_adaptive': '#6A1B9A',    # Dark purple
    'arc_ultimate': '#1A237E',    # Deep blue
}

LABELS = {
    'no_control': 'No Control',
    'naive_calm': 'Naive Calm',
    'perf_optimized': 'Perf Opt.',
    'arc_v1': 'ARC v1',
    'arc_v1_pid': 'ARC PID',
    'arc_v1_lqr': 'ARC LQR',
    'arc_v1_lqi': 'ARC LQI',
    'arc_v2_hier': 'ARC Hier',
    'arc_v2_lqi': 'ARC v2+LQI',
    'arc_v3_meta': 'ARC Meta',
    'arc_v3_pid_meta': 'PID+Meta',
    'arc_v3_lqr_meta': 'LQR+Meta',
    'arc_robust': 'Robust',
    'arc_adaptive': 'Adaptive',
    'arc_ultimate': 'Ultimate',
}

CONTROLLERS = list(LABELS.keys())

# Load data
df = pd.read_csv('outputs_final/metrics.csv')
os.makedirs('figures_controllers', exist_ok=True)

df_filt = df[df['controller'].isin(CONTROLLERS)]
agg = df_filt.groupby('controller').agg({
    'PerfMean': ['mean', 'std'],
    'RI': ['mean', 'std'],
    'Overshoot': ['mean', 'std'],
    'ControlEffort': ['mean', 'std']
}).round(4)



def styled_bar_chart(metric_col, ylabel, title, filename, ylim, ref_lines=None):
    """Create a professional bar chart with unified style."""
def styled_bar_chart(metric_col, ylabel, title, filename, ylim, ref_lines=None):
    """Create a professional bar chart with unified style."""
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    plt.subplots_adjust(bottom=0.2)
    
    x = np.arange(len(CONTROLLERS))
    width = 0.65
    
    means = [agg.loc[c, (metric_col, 'mean')] for c in CONTROLLERS]
    stds = [agg.loc[c, (metric_col, 'std')] for c in CONTROLLERS]
    bar_colors = [CONTROLLER_COLORS[c] for c in CONTROLLERS]
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=4, 
                  color=bar_colors, edgecolor='#333333', linewidth=0.8,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#555'})
    
    if ref_lines:
        for y, color, style, label in ref_lines:
            ax.axhline(y=y, color=color, linestyle=style, alpha=0.7, lw=1.5, label=label)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Controller Architecture', fontsize=12)
    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONTROLLERS], rotation=45, ha='right', fontsize=11)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelsize=11)
    
    if ref_lines:
        ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    # Add subtle background gradient effect
    ax.set_facecolor('#FAFAFA')
    
    plt.savefig(f'figures_controllers/{filename}', dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {filename}")


def styled_scatter_tradeoff():
    """Create professional trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    
    # Calculate limits for top 5 annotation
    perfs = df_filt.groupby('controller')['PerfMean'].mean()
    top_controllers = perfs.sort_values(ascending=False).head(6).index.tolist()
    
    # Annotate top controllers with repel-like offset
    texts = []
    
    for c in CONTROLLERS:
        perf = agg.loc[c, ('PerfMean', 'mean')]
        ri = agg.loc[c, ('RI', 'mean')]
        effort = agg.loc[c, ('ControlEffort', 'mean')]
        
        # Plot point
        ax.scatter(ri, perf, s=effort*180 + 100, c=CONTROLLER_COLORS[c], 
                   edgecolors='#333', linewidths=1.2, alpha=0.85, label=LABELS[c])
        
        # Add label directly on plot for top controllers and baseline
        if c in top_controllers or c in ['no_control', 'naive_calm']:
            ax.annotate(LABELS[c], (ri, perf), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, fontweight='bold', alpha=0.9)

    ax.set_xlabel('Rumination Index (lower = better)', fontsize=12)
    ax.set_ylabel('Performance (higher = better)', fontsize=12)
    ax.set_title('Trade-off: Performance vs Anti-Rumination\n(Bubble size indicates Control Effort)', pad=15, fontsize=14)
    
    # Move legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.95, title="Controllers")
    
    ax.set_xlim(-0.05, 1.6) # Increased xlim slightly
    ax.set_ylim(0.0, 1.05)
    
    # Reference lines
    ax.axvline(x=0.10, color=COLORS['warning_amber'], linestyle='--', alpha=0.6, lw=1.5, label='RI Warning')
    ax.axhline(y=0.90, color=COLORS['success_green'], linestyle='--', alpha=0.6, lw=1.5, label='Perf Target')
    
    # Optimal region annotation (ensure it doesn't overlap points)
    rect = patches.FancyBboxPatch((0.0, 0.90), 0.15, 0.15, boxstyle="round,pad=0.02", 
                                 linewidth=1, edgecolor='#43A047', facecolor='#C8E6C9', alpha=0.3)
    ax.add_patch(rect)
    ax.text(0.075, 0.975, 'Optimal\nRegion', ha='center', va='center', fontsize=10, 
            fontweight='bold', color='#2E7D32')
    
    ax.set_facecolor('#FAFAFA')
    plt.savefig('figures_controllers/fig_controller_tradeoff.png', dpi=300, facecolor='white')
    plt.close()
    print("✓ Saved: fig_controller_tradeoff.png")


def styled_radar_chart():
    """Create professional radar chart for top 5 controllers."""
    def composite_score(ctrl):
        perf = agg.loc[ctrl, ('PerfMean', 'mean')]
        ri = 1 - min(1.0, agg.loc[ctrl, ('RI', 'mean')])
        os = 1 - min(1.0, agg.loc[ctrl, ('Overshoot', 'mean')])
        eff = 1 - min(1.0, agg.loc[ctrl, ('ControlEffort', 'mean')] / 2.5)
        return float(np.mean([perf, ri, os, eff]))
    
    scores = {c: composite_score(c) for c in CONTROLLERS}
    top5 = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:5]
    
    categories = ['Performance', 'Anti-Rumination', 'Low Overshoot', 'Efficiency']
    N = len(categories)
    
    def get_values(ctrl):
        perf = agg.loc[ctrl, ('PerfMean', 'mean')]
        ri = 1 - min(1, agg.loc[ctrl, ('RI', 'mean')])
        os = 1 - min(1, agg.loc[ctrl, ('Overshoot', 'mean')])
        eff = 1 - min(1, agg.loc[ctrl, ('ControlEffort', 'mean')] / 2.5)
        return [perf, ri, os, eff]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for ctrl in top5:
        values = get_values(ctrl)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=LABELS[ctrl], 
                color=CONTROLLER_COLORS[ctrl], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=CONTROLLER_COLORS[ctrl])
    
    ax.set_xticks(angles[:-1])
    # Add padding to labels to prevent overlap with data or other elements
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', pad=30)
    
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Controllers: Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=35)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig('figures_controllers/fig_controller_radar.png', dpi=300, facecolor='white')
    plt.close()
    print("✓ Saved: fig_controller_radar.png")


# ============ GENERATE ALL FIGURES ============
if __name__ == "__main__":
    print("\n🎨 Generating unified-style figures...\n")
    
    # Figure 1: Performance
    styled_bar_chart(
        'PerfMean', 'Performance Mean', 
        'Performance Comparison: 15 Controller Architectures',
        'fig_controller_performance.png', (0, 1.1),
        ref_lines=[
            (0.90, COLORS['success_green'], '--', 'Target (0.90)'),
            (agg.loc['no_control', ('PerfMean', 'mean')], COLORS['danger_red'], ':', 'Baseline')
        ]
    )
    
    # Figure 2: Rumination Index (Neutral title)
    styled_bar_chart(
        'RI', 'Rumination Index (RI)',
        'Rumination Index across Controllers',
        'fig_controller_rumination.png', (0, 1.5),
        ref_lines=[
            (0.10, COLORS['warning_amber'], '--', 'Warning Threshold')
        ]
    )
    
    # Figure 3: Control Effort (Simplified)
    # Could be split, but for quick win, we keep structure but cleaner
    styled_bar_chart(
        'ControlEffort', 'Control Effort',
        'Control Effort by Architecture (Meta-Control is Efficient)',
        'fig_controller_effort.png', (0, 2.8)
    )
    
    # Figure 4: Trade-off scatter
    styled_scatter_tradeoff()
    
    # Figure 5: Radar chart
    styled_radar_chart()
    
    print("\n✅ All unified-style figures saved to figures_controllers/\n")
