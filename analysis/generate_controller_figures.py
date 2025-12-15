"""
Generate comparison figures for all 15 controllers.
Creates publication-ready figures for the ARC paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
df = pd.read_csv('outputs_final/metrics.csv')

# Create output directory
os.makedirs('figures_controllers', exist_ok=True)

# Define controller order and colors (all 15 controllers)
controllers_main = [
    'no_control', 'naive_calm', 'perf_optimized',
    'arc_v1', 'arc_v1_pid', 'arc_v1_lqr', 'arc_v1_lqi',
    'arc_v2_hier', 'arc_v2_lqi',
    'arc_v3_meta', 'arc_v3_pid_meta', 'arc_v3_lqr_meta',
    'arc_robust', 'arc_adaptive', 'arc_ultimate'
]

colors = {
    'no_control': '#e74c3c',  # Red (baseline)
    'naive_calm': '#ff7675',  # Light red
    'perf_optimized': '#95a5a6',  # Gray
    'arc_v1': '#3498db',       # Blue
    'arc_v1_pid': '#9b59b6',   # Purple
    'arc_v1_lqr': '#1abc9c',   # Teal
    'arc_v1_lqi': '#27ae60',   # Green
    'arc_v2_hier': '#f39c12',  # Orange
    'arc_v2_lqi': '#e67e22',   # Dark Orange
    'arc_v3_meta': '#2ecc71',  # Light Green
    'arc_v3_pid_meta': '#f1c40f',  # Yellow
    'arc_v3_lqr_meta': '#00bcd4',  # Cyan
    'arc_robust': '#16a085',   # Dark Teal
    'arc_adaptive': '#8e44ad', # Dark Purple
    'arc_ultimate': '#2c3e50', # Dark Blue
}

labels = {
    'no_control': 'No Control',
    'naive_calm': 'Naive Calm',
    'perf_optimized': 'Perf Optimized',
    'arc_v1': 'ARC v1 (P)',
    'arc_v1_pid': 'ARC PID',
    'arc_v1_lqr': 'ARC LQR',
    'arc_v1_lqi': 'ARC LQI',
    'arc_v2_hier': 'ARC v2 Hier',
    'arc_v2_lqi': 'ARC v2+LQI',
    'arc_v3_meta': 'ARC Meta',
    'arc_v3_pid_meta': 'ARC PID+Meta',
    'arc_v3_lqr_meta': 'ARC LQR+Meta',
    'arc_robust': 'ARC Robust',
    'arc_adaptive': 'ARC Adaptive',
    'arc_ultimate': 'ARC Ultimate',
}

# Filter and aggregate
df_filt = df[df['controller'].isin(controllers_main)]
agg = df_filt.groupby('controller').agg({
    'PerfMean': ['mean', 'std'],
    'RI': ['mean', 'std'],
    'Overshoot': ['mean', 'std'],
    'ControlEffort': ['mean', 'std']
}).round(4)

# ============ FIGURE 1: Performance Bar Chart ============
plt.figure(figsize=(16, 6))
x = np.arange(len(controllers_main))
width = 0.6

perf_means = [agg.loc[c, ('PerfMean', 'mean')] for c in controllers_main]
perf_stds = [agg.loc[c, ('PerfMean', 'std')] for c in controllers_main]
bar_colors = [colors[c] for c in controllers_main]

bars = plt.bar(x, perf_means, width, yerr=perf_stds, capsize=3, 
               color=bar_colors, edgecolor='black', linewidth=0.5)
plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='Target (0.90)')
plt.axhline(y=agg.loc['no_control', ('PerfMean', 'mean')], color='red', 
            linestyle=':', alpha=0.7, label='Baseline')

plt.ylabel('Performance Mean', fontsize=12)
plt.xlabel('Controller', fontsize=12)
plt.title('Performance Comparison: 15 Controller Architectures', fontsize=14, fontweight='bold')
plt.xticks(x, [labels[c] for c in controllers_main], rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures_controllers/fig_controller_performance.png', dpi=150)
plt.close()
print("Saved: fig_controller_performance.png")

# ============ FIGURE 2: Rumination Index ============
plt.figure(figsize=(16, 6))
ri_means = [agg.loc[c, ('RI', 'mean')] for c in controllers_main]
ri_stds = [agg.loc[c, ('RI', 'std')] for c in controllers_main]

bars = plt.bar(x, ri_means, width, yerr=ri_stds, capsize=3,
               color=bar_colors, edgecolor='black', linewidth=0.5)
plt.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Warning (0.10)')

plt.ylabel('Rumination Index (RI)', fontsize=12)
plt.xlabel('Controller', fontsize=12)
plt.title('Anti-Rumination: Controllers with Integral Action Achieve RI ≈ 0', fontsize=14, fontweight='bold')
plt.xticks(x, [labels[c] for c in controllers_main], rotation=45, ha='right')
plt.ylim(0, 1.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('figures_controllers/fig_controller_rumination.png', dpi=150)
plt.close()
print("Saved: fig_controller_rumination.png")

# ============ FIGURE 3: Trade-off Scatter (Perf vs RI) ============
plt.figure(figsize=(10, 8))

for c in controllers_main:
    perf = agg.loc[c, ('PerfMean', 'mean')]
    ri = agg.loc[c, ('RI', 'mean')]
    effort = agg.loc[c, ('ControlEffort', 'mean')]
    
    plt.scatter(ri, perf, s=effort*300 + 50, c=colors[c], 
                edgecolors='black', linewidths=1, alpha=0.8, label=labels[c])

plt.xlabel('Rumination Index (lower = better)', fontsize=12)
plt.ylabel('Performance (higher = better)', fontsize=12)
plt.title('Trade-off: Performance vs Anti-Rumination\n(bubble size = control effort)', fontsize=14, fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
plt.xlim(-0.1, 1.6)
plt.ylim(0, 1.1)
plt.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5)
plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.5)

# Annotate optimal region
plt.annotate('Optimal\nRegion', xy=(0.05, 0.95), fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('figures_controllers/fig_controller_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_controller_tradeoff.png")

# ============ FIGURE 4: Control Effort ============
plt.figure(figsize=(16, 6))
effort_means = [agg.loc[c, ('ControlEffort', 'mean')] for c in controllers_main]
effort_stds = [agg.loc[c, ('ControlEffort', 'std')] for c in controllers_main]

bars = plt.bar(x, effort_means, width, yerr=effort_stds, capsize=3,
               color=bar_colors, edgecolor='black', linewidth=0.5)

plt.ylabel('Control Effort', fontsize=12)
plt.xlabel('Controller', fontsize=12)
plt.title('Efficiency: Meta-Control Achieves Lowest Effort', fontsize=14, fontweight='bold')
plt.xticks(x, [labels[c] for c in controllers_main], rotation=45, ha='right')
plt.ylim(0, 2.5)
plt.tight_layout()
plt.savefig('figures_controllers/fig_controller_effort.png', dpi=150)
plt.close()
print("Saved: fig_controller_effort.png")

# ============ FIGURE 5: Radar Chart (Top 5) ============
def composite_score(ctrl):
    perf = agg.loc[ctrl, ('PerfMean', 'mean')]
    ri = 1 - min(1.0, agg.loc[ctrl, ('RI', 'mean')])  # Invert: lower RI = better
    os = 1 - min(1.0, agg.loc[ctrl, ('Overshoot', 'mean')])  # Invert
    eff = 1 - min(1.0, agg.loc[ctrl, ('ControlEffort', 'mean')] / 2.5)  # Invert
    return float(np.mean([perf, ri, os, eff]))

scores = {c: composite_score(c) for c in controllers_main}
top5 = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:5]
print("Top 5 controllers for radar:", top5)
categories = ['Performance', 'Anti-Rumination', 'Low Overshoot', 'Efficiency']
N = len(categories)

# Normalize metrics (higher = better for all)
def normalize(df, ctrl):
    perf = agg.loc[ctrl, ('PerfMean', 'mean')]
    ri = 1 - min(1, agg.loc[ctrl, ('RI', 'mean')])  # Invert: lower RI = better
    os = 1 - min(1, agg.loc[ctrl, ('Overshoot', 'mean')])  # Invert
    eff = 1 - min(1, agg.loc[ctrl, ('ControlEffort', 'mean')] / 2.5)  # Invert
    return [perf, ri, os, eff]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for ctrl in top5:
    values = normalize(agg, ctrl)
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=labels[ctrl], color=colors[ctrl])
    ax.fill(angles, values, alpha=0.1, color=colors[ctrl])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Top 5 Controllers: Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('figures_controllers/fig_controller_radar.png', dpi=150)
plt.close()
print("Saved: fig_controller_radar.png")

print("\nAll figures saved to figures_controllers/")
