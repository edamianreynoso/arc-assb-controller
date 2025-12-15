"""
Paper Figure Generator - Generates publication-ready plots from experiment results.

Usage:
    python visualizations/paper_figures.py --data outputs_L6 --output figures

Generates:
    - learning_curves.png: Episode rewards over time
    - metrics_comparison.png: Bar chart comparing ARC vs Baseline
    - state_dynamics.png: ASSB state variables over time
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Style configuration for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'arc': '#00d4ff',
    'baseline': '#ffa500',
    'arc_dark': '#0077aa',
    'baseline_dark': '#cc8400',
    'goal': '#00ff88',
    'danger': '#ff4444',
}

def load_data(data_dir: str) -> dict:
    """Load all CSV files from data directory."""
    data = {}
    data_path = Path(data_dir)
    
    if (data_path / "summary.csv").exists():
        data['summary'] = pd.read_csv(data_path / "summary.csv")
    if (data_path / "final_metrics.csv").exists():
        data['final'] = pd.read_csv(data_path / "final_metrics.csv")
    if (data_path / "raw_results.csv").exists():
        data['raw'] = pd.read_csv(data_path / "raw_results.csv")
    
    return data

def plot_learning_curves(data: dict, output_dir: str):
    """Plot learning curves comparing ARC vs Baseline across environments."""
    if 'summary' not in data:
        print("No summary data found for learning curves")
        return
    
    df = data['summary']
    envs = df['env'].unique()
    
    fig, axes = plt.subplots(1, len(envs), figsize=(5*len(envs), 5))
    if len(envs) == 1:
        axes = [axes]
    
    for ax, env in zip(axes, envs):
        env_data = df[df['env'] == env]
        
        for agent in ['ql_arc', 'ql_baseline']:
            agent_data = env_data[env_data['agent'] == agent]
            color = COLORS['arc'] if 'arc' in agent else COLORS['baseline']
            label = 'ARC' if 'arc' in agent else 'Baseline'
            
            ax.plot(agent_data['episode'], agent_data['reward_mean'], 
                   color=color, linewidth=2, label=label)
            ax.fill_between(agent_data['episode'], 
                           agent_data['reward_mean'] - agent_data['reward_std'],
                           agent_data['reward_mean'] + agent_data['reward_std'],
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title(env.replace('GridWorld', ' GridWorld'))
        ax.legend(loc='lower right')
        ax.axhline(y=0.93, color='gray', linestyle='--', alpha=0.5, label='Goal Reward')
    
    plt.suptitle('Learning Curves: ARC vs Baseline', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    print(f"Saved: {output_dir}/learning_curves.png")
    plt.close()

def plot_metrics_comparison(data: dict, output_dir: str):
    """Bar chart comparing final metrics between ARC and Baseline."""
    if 'final' not in data:
        print("No final metrics data found")
        return
    
    df = data['final']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics = [
        ('final_reward_mean', 'Final Reward', True),
        ('final_success_rate', 'Success Rate (%)', True),
        ('mean_arousal', 'Mean Arousal', False),
    ]
    
    for ax, (metric, title, higher_better) in zip(axes, metrics):
        envs = df['env'].unique()
        x = np.arange(len(envs))
        width = 0.35
        
        arc_vals = []
        base_vals = []
        
        for env in envs:
            arc_row = df[(df['env'] == env) & (df['agent'] == 'ql_arc')]
            base_row = df[(df['env'] == env) & (df['agent'] == 'ql_baseline')]
            
            arc_val = arc_row[metric].values[0] if len(arc_row) > 0 else 0
            base_val = base_row[metric].values[0] if len(base_row) > 0 else 0
            
            if 'success' in metric.lower():
                arc_val *= 100
                base_val *= 100
            
            arc_vals.append(arc_val)
            base_vals.append(base_val)
        
        bars1 = ax.bar(x - width/2, base_vals, width, label='Baseline', 
                       color=COLORS['baseline'], edgecolor=COLORS['baseline_dark'])
        bars2 = ax.bar(x + width/2, arc_vals, width, label='ARC', 
                       color=COLORS['arc'], edgecolor=COLORS['arc_dark'])
        
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace('GridWorld', '\nGridWorld') for e in envs], fontsize=9)
        ax.legend()
        ax.set_title(title)
        
        # Highlight winner
        for i, (arc, base) in enumerate(zip(arc_vals, base_vals)):
            winner = arc if (higher_better and arc > base) or (not higher_better and arc < base) else base
            winner_x = x[i] + width/2 if winner == arc else x[i] - width/2
            ax.annotate('★', xy=(winner_x, winner), ha='center', va='bottom', 
                       fontsize=14, color='gold')
    
    plt.suptitle('Final Metrics Comparison: ARC vs Baseline', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    print(f"Saved: {output_dir}/metrics_comparison.png")
    plt.close()

def plot_state_dynamics(data: dict, output_dir: str):
    """Plot ASSB state variables over time from raw results."""
    if 'raw' not in data:
        print("No raw data found for state dynamics")
        return
    
    df = data['raw']
    
    # Filter for a single seed and environment for cleaner visualization
    if 'seed' in df.columns:
        df = df[df['seed'] == df['seed'].min()]
    
    # Check if ASSB metrics are available
    if 'mean_arousal' not in df.columns:
        print("No ASSB state data in raw results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter for ChangingGoalGridWorld (most interesting)
    env_filter = 'ChangingGoalGridWorld' if 'ChangingGoalGridWorld' in df['env'].values else df['env'].values[0]
    df_env = df[df['env'] == env_filter]
    
    # Plot 1: Reward over episodes
    ax = axes[0, 0]
    for agent in df_env['agent'].unique():
        agent_data = df_env[df_env['agent'] == agent]
        color = COLORS['arc'] if 'arc' in agent else COLORS['baseline']
        label = 'ARC' if 'arc' in agent else 'Baseline'
        ax.plot(agent_data['episode'], agent_data['total_reward'], 
               color=color, linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    
    # Plot 2: Success rate (rolling)
    ax = axes[0, 1]
    for agent in df_env['agent'].unique():
        agent_data = df_env[df_env['agent'] == agent].copy()
        agent_data['success'] = agent_data['reached_goal'].astype(float)
        agent_data['success_rolling'] = agent_data['success'].rolling(10, min_periods=1).mean()
        color = COLORS['arc'] if 'arc' in agent else COLORS['baseline']
        label = 'ARC' if 'arc' in agent else 'Baseline'
        ax.plot(agent_data['episode'], agent_data['success_rolling'] * 100, 
               color=color, linewidth=2, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (10-episode rolling average)')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Plot 3: Arousal over episodes (ARC only)
    ax = axes[1, 0]
    arc_data = df_env[df_env['agent'] == 'ql_arc']
    if 'mean_arousal' in arc_data.columns and not arc_data['mean_arousal'].isna().all():
        ax.plot(arc_data['episode'], arc_data['mean_arousal'], 
               color=COLORS['arc'], linewidth=2, label='ARC Arousal')
        ax.axhline(y=0.6, color=COLORS['danger'], linestyle='--', alpha=0.7, label='Safe Threshold')
        ax.fill_between(arc_data['episode'], 0, arc_data['mean_arousal'], 
                       where=arc_data['mean_arousal'] > 0.6, color=COLORS['danger'], alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Arousal')
    ax.set_title('ARC Internal State: Arousal')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Plot 4: Episode length over time
    ax = axes[1, 1]
    for agent in df_env['agent'].unique():
        agent_data = df_env[df_env['agent'] == agent]
        color = COLORS['arc'] if 'arc' in agent else COLORS['baseline']
        label = 'ARC' if 'arc' in agent else 'Baseline'
        ax.plot(agent_data['episode'], agent_data['steps'], 
               color=color, linewidth=1.5, label=label, alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per Episode')
    ax.set_title('Episode Length (Lower = Faster to Goal)')
    ax.legend()
    
    plt.suptitle(f'State Dynamics: {env_filter}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_dynamics.png'))
    print(f"Saved: {output_dir}/state_dynamics.png")
    plt.close()

def plot_ablation_summary(output_dir: str):
    """Create ablation study visualization from L1 results."""
    # Hardcoded from task.md ablation results
    ablation_data = {
        'Controller': ['arc_v1', 'arc_no_dmg', 'arc_no_calm', 'arc_no_mem'],
        'PerfMean': [0.994, 0.928, 0.932, 0.994],
        'RT': [3.3, 100, 100, 3.3],
        'RI': [0.0, 1.41, 0.0, 0.0],
        'NDR': [0.0, 1.0, 0.0, 0.0],
    }
    df = pd.DataFrame(ablation_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = ['#00d4ff', '#ff4444', '#ffa500', '#00ff88']
    
    # Performance
    ax = axes[0]
    bars = ax.bar(df['Controller'], df['PerfMean'], color=colors)
    ax.set_ylabel('Performance')
    ax.set_title('Performance by Controller')
    ax.set_ylim(0.9, 1.0)
    ax.tick_params(axis='x', rotation=45)
    
    # Rumination Index
    ax = axes[1]
    bars = ax.bar(df['Controller'], df['RI'], color=colors)
    ax.set_ylabel('Rumination Index')
    ax.set_title('Rumination Index (Lower = Better)')
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight critical finding
    ax.annotate('CRITICAL!\nDMN control\nprevents rumination', 
               xy=(1, 1.41), xytext=(1.5, 1.2),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red', ha='center')
    
    # Recovery Time
    ax = axes[2]
    bars = ax.bar(df['Controller'], df['RT'], color=colors)
    ax.set_ylabel('Recovery Time (steps)')
    ax.set_title('Recovery Time (Lower = Better)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Ablation Study: Which ARC Components Are Critical?', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_summary.png'))
    print(f"Saved: {output_dir}/ablation_summary.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data", type=str, default="outputs_L6", help="Data directory")
    parser.add_argument("--output", type=str, default="figures", help="Output directory")
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data
    output_dir = base_dir / args.output
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    print(f"Saving figures to: {output_dir}")
    
    # Load data
    data = load_data(data_dir)
    
    if not data:
        print("No data files found!")
        return
    
    print(f"Found datasets: {list(data.keys())}")
    
    # Generate all figures
    print("\nGenerating figures...")
    plot_learning_curves(data, output_dir)
    plot_metrics_comparison(data, output_dir)
    plot_state_dynamics(data, output_dir)
    plot_ablation_summary(output_dir)
    
    print(f"\n✅ All figures saved to: {output_dir}")

if __name__ == "__main__":
    main()
