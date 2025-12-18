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
    fig, ax = plt.subplots(figsize=(14, 6))
    
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
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Controller Architecture')
    ax.set_title(title, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONTROLLERS], rotation=45, ha='right')
    ax.set_ylim(ylim)
    
    if ref_lines:
        ax.legend(loc='upper right', framealpha=0.95)
    
    # Add subtle background gradient effect
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(f'figures_controllers/{filename}', dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {filename}")


def styled_scatter_tradeoff():
    """Create professional trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(11, 9))
    
    for c in CONTROLLERS:
        perf = agg.loc[c, ('PerfMean', 'mean')]
        ri = agg.loc[c, ('RI', 'mean')]
        effort = agg.loc[c, ('ControlEffort', 'mean')]
        
        ax.scatter(ri, perf, s=effort*250 + 80, c=CONTROLLER_COLORS[c], 
                   edgecolors='#333', linewidths=1.2, alpha=0.85, label=LABELS[c])
    
    ax.set_xlabel('Rumination Index (lower = better)')
    ax.set_ylabel('Performance (higher = better)')
    ax.set_title('Trade-off: Performance vs Anti-Rumination\n(bubble size = control effort)', pad=15)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(0.3, 1.05)
    
    # Reference lines
    ax.axvline(x=0.10, color=COLORS['warning_amber'], linestyle='--', alpha=0.6, lw=1.5)
    ax.axhline(y=0.90, color=COLORS['success_green'], linestyle='--', alpha=0.6, lw=1.5)
    
    # Optimal region annotation
    ax.annotate('Optimal\nRegion', xy=(0.02, 0.97), fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor='#43A047', alpha=0.9))
    
    ax.set_facecolor('#FAFAFA')
    plt.tight_layout()
    plt.savefig('figures_controllers/fig_controller_tradeoff.png', dpi=300, facecolor='white', bbox_inches='tight')
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
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Controllers: Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=25)
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
    
    # Figure 2: Rumination Index
    styled_bar_chart(
        'RI', 'Rumination Index (RI)',
        'Anti-Rumination: Controllers with Integral Action Achieve RI ≈ 0',
        'fig_controller_rumination.png', (0, 1.5),
        ref_lines=[
            (0.10, COLORS['warning_amber'], '--', 'Warning Threshold')
        ]
    )
    
    # Figure 3: Control Effort
    styled_bar_chart(
        'ControlEffort', 'Control Effort',
        'Efficiency: Meta-Control Achieves Lowest Effort',
        'fig_controller_effort.png', (0, 2.5)
    )
    
    # Figure 4: Trade-off scatter
    styled_scatter_tradeoff()
    
    # Figure 5: Radar chart
    styled_radar_chart()
    
    print("\n✅ All unified-style figures saved to figures_controllers/\n")
