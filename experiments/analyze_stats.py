"""
Deep Analysis Part 1: Statistical Significance Tests

Generates:
- T-tests comparing ARC vs baselines
- Confidence intervals for key metrics
- Effect sizes (Cohen's d)
- Summary table with p-values
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval."""
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

def analyze_metrics_file(filepath: Path) -> dict:
    """Analyze a single metrics file."""
    df = pd.read_csv(filepath)
    
    results = []
    
    # Identify ARC and baseline groups
    if 'controller' in df.columns:
        arc_mask = df['controller'].str.contains('arc', case=False, na=False)
        baseline_mask = df['controller'].str.contains('no_control', case=False, na=False)
        
        arc_data = df[arc_mask]
        baseline_data = df[baseline_mask]
        
        metrics = ['PerfMean', 'RI', 'RT', 'NDR', 'ControlEffort']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            arc_vals = arc_data[metric].dropna()
            base_vals = baseline_data[metric].dropna()
            
            if len(arc_vals) < 2 or len(base_vals) < 2:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(arc_vals, base_vals)
            
            # Effect size
            d = cohens_d(arc_vals, base_vals)
            
            # Confidence intervals
            arc_ci = confidence_interval(arc_vals)
            base_ci = confidence_interval(base_vals)
            
            # Determine significance
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            results.append({
                'metric': metric,
                'arc_mean': arc_vals.mean(),
                'arc_std': arc_vals.std(),
                'arc_ci_low': arc_ci[0],
                'arc_ci_high': arc_ci[1],
                'baseline_mean': base_vals.mean(),
                'baseline_std': base_vals.std(),
                'baseline_ci_low': base_ci[0],
                'baseline_ci_high': base_ci[1],
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': d,
                'significance': sig,
                'n_arc': len(arc_vals),
                'n_baseline': len(base_vals)
            })
    
    return results

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Find all metrics files
    metrics_files = [
        ("L1", base_dir / "outputs_v2" / "metrics.csv"),
        ("L2", base_dir / "outputs_L2" / "metrics.csv"),
        ("L3", base_dir / "outputs_L3" / "metrics.csv"),
        ("L4", base_dir / "outputs_L4_rev2" / "metrics.csv"),
        ("L5", base_dir / "outputs_L5" / "metrics.csv"),
        ("L4_meta", base_dir / "outputs_rev11" / "metrics.csv"),
    ]
    
    all_results = []
    
    print("=" * 70)
    print("STATISTICAL ANALYSIS: ARC vs Baseline")
    print("=" * 70)
    
    for line_name, filepath in metrics_files:
        if not filepath.exists():
            print(f"\n⚠️ {line_name}: File not found")
            continue
            
        print(f"\n{'='*70}")
        print(f"📊 {line_name}: {filepath.name}")
        print("="*70)
        
        results = analyze_metrics_file(filepath)
        
        if not results:
            print("  No comparable data found")
            continue
        
        for r in results:
            r['line'] = line_name
            all_results.append(r)
            
            # Print formatted result
            print(f"\n  {r['metric']}:")
            print(f"    ARC:      {r['arc_mean']:.3f} ± {r['arc_std']:.3f}  (95% CI: [{r['arc_ci_low']:.3f}, {r['arc_ci_high']:.3f}])")
            print(f"    Baseline: {r['baseline_mean']:.3f} ± {r['baseline_std']:.3f}  (95% CI: [{r['baseline_ci_low']:.3f}, {r['baseline_ci_high']:.3f}])")
            print(f"    t = {r['t_stat']:.3f}, p = {r['p_value']:.2e} {r['significance']}")
            print(f"    Cohen's d = {r['cohens_d']:.3f} ({'large' if abs(r['cohens_d']) > 0.8 else 'medium' if abs(r['cohens_d']) > 0.5 else 'small'})")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "statistical_tests.csv", index=False)
        print(f"\n✅ Results saved to: {output_dir / 'statistical_tests.csv'}")
        
        # Summary table
        print("\n" + "="*70)
        print("SUMMARY: Significant Differences (p < 0.05)")
        print("="*70)
        
        sig_results = results_df[results_df['p_value'] < 0.05]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                direction = "ARC better" if (row['metric'] == 'PerfMean' and row['arc_mean'] > row['baseline_mean']) or \
                                           (row['metric'] in ['RI', 'RT', 'NDR', 'ControlEffort'] and row['arc_mean'] < row['baseline_mean']) \
                            else "Baseline better"
                print(f"  {row['line']}/{row['metric']}: p={row['p_value']:.2e} {row['significance']} | d={row['cohens_d']:.2f} | {direction}")
        else:
            print("  No statistically significant differences found.")

if __name__ == "__main__":
    main()
