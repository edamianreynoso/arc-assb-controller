"""
Deep Analysis Part 3: Correlation Heatmap

Generates correlation matrix between all metrics to understand
relationships (e.g., does high RI correlate with low PerfMean?).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style
plt.style.use('dark_background')

def create_correlation_heatmap(df: pd.DataFrame, title: str, save_path: Path):
    """Create and save a correlation heatmap."""
    # Select numeric columns
    numeric_cols = ['PerfMean', 'RI', 'RT', 'Overshoot', 'NDR', 'ControlEffort', 
                   'PerfStd', 'StabilityPost', 'Retention', 'AdaptSpeed', 'MemStability']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print(f"  Not enough numeric columns for correlation")
        return
    
    corr_df = df[available_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Upper triangle mask
    
    sns.heatmap(corr_df, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
                vmin=-1, vmax=1,
                annot_kws={'size': 10})
    
    ax.set_title(title, fontsize=14, color='white', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Process each metrics file
    metrics_files = [
        ("L1", base_dir / "outputs_v2" / "metrics.csv"),
        ("L2", base_dir / "outputs_L2" / "metrics.csv"),
        ("L3", base_dir / "outputs_L3" / "metrics.csv"),
        ("L4", base_dir / "outputs_L4_rev2" / "metrics.csv"),
        ("L5", base_dir / "outputs_L5" / "metrics.csv"),
        ("L4_meta", base_dir / "outputs_rev11" / "metrics.csv"),
    ]
    
    # Combined analysis
    all_dfs = []
    
    for line_name, filepath in metrics_files:
        if not filepath.exists():
            continue
            
        print(f"\n📊 {line_name}: {filepath.name}")
        
        df = pd.read_csv(filepath)
        df['line'] = line_name
        all_dfs.append(df)
        
        # Individual heatmap
        create_correlation_heatmap(df, f'Metric Correlations - {line_name}',
                                  output_dir / f"correlation_{line_name}.png")
    
    # Combined heatmap
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n📊 Combined Analysis ({len(combined_df)} total runs)")
        create_correlation_heatmap(combined_df, 'Metric Correlations - All Lines Combined',
                                  output_dir / "correlation_combined.png")
        
        # Print key correlations
        numeric_cols = ['PerfMean', 'RI', 'RT', 'NDR', 'ControlEffort']
        available_cols = [col for col in numeric_cols if col in combined_df.columns]
        
        if len(available_cols) >= 2:
            corr = combined_df[available_cols].corr()
            
            print("\n" + "="*70)
            print("KEY CORRELATIONS (absolute value > 0.5)")
            print("="*70)
            
            for i in range(len(available_cols)):
                for j in range(i+1, len(available_cols)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.5:
                        direction = "positive" if r > 0 else "negative"
                        strength = "strong" if abs(r) > 0.7 else "moderate"
                        print(f"  {available_cols[i]} ↔ {available_cols[j]}: r={r:.3f} ({strength} {direction})")
    
    print("\n✅ Correlation analysis complete!")
    print(f"   Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
