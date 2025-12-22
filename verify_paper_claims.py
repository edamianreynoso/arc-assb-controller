import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = x.size
    ny = y.size
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s_pooled

def verify_claims():
    print("Verifying Paper Claims...")
    
    # Load Main Metrics
    metrics_path = Path('outputs_final/metrics.csv')
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found.")
        return

    df = pd.read_csv(metrics_path)
    
    # 1. ARC v1 L1 Performance & Rumination
    # "ARC achieves 96.6% average performance with zero rumination... in stability scenarios"
    # L1 scenarios: reward_flip, noise_burst, sudden_threat
    l1_scenarios = ['reward_flip', 'noise_burst', 'sudden_threat']
    l1_df = df[df['scenario'].isin(l1_scenarios)]
    
    arc_v1_l1 = l1_df[l1_df['controller'] == 'arc_v1']
    no_control_l1 = l1_df[l1_df['controller'] == 'no_control']
    
    print(f"\nClaim 1: ARC L1 Stability")
    print(f"  ARC v1 PerfMean (L1): {arc_v1_l1['PerfMean'].mean():.3f} (Claim: 96.6%)")
    print(f"  ARC v1 RI (L1): {arc_v1_l1['RI'].mean():.3f} (Claim: 0.00 or near 0)")
    print(f"  No Control PerfMean (L1): {no_control_l1['PerfMean'].mean():.3f} (Claim: 30%)")

    # 2. ARC Meta-Control Efficiency
    # "ARC meta-control reduces control effort by 21%... L4"
    # This comparison usually uses ALL scenarios or specific ones? Section 6.5 says "L4: Meta-Control Efficiency"
    # hypothesis definition says "L4... cross-cutting analysis".
    # Let's check typical comparison: arc_v3_meta vs arc_v1 across all scenarios? Or subset?
    # Table 4 in text compares arc_v3_meta vs arc_v1. Let's assume global mean for now or check L1-L3.
    
    print(f"\nClaim 2: Meta-Control Efficiency")
    arc_v3_meta = df[df['controller'] == 'arc_v3_meta']
    arc_v1_all = df[df['controller'] == 'arc_v1']
    
    eff_v3 = arc_v3_meta['ControlEffort'].mean()
    eff_v1 = arc_v1_all['ControlEffort'].mean()
    reduction = (eff_v1 - eff_v3) / eff_v1 * 100
    
    print(f"  ARC v3 Effort: {eff_v3:.3f}")
    print(f"  ARC v1 Effort: {eff_v1:.3f}")
    print(f"  Reduction: {reduction:.1f}% (Claim: 21%)")
    
    # 3. Robust Controller Balance
    # "H-inf Robust controllers achieve the best overall balance"
    # Table 6 claims: arc_robust PerfMean=0.95, RI=0.00
    arc_robust = df[df['controller'] == 'arc_robust']
    print(f"\nClaim 3: Robust Controller")
    print(f"  Robust PerfMean: {arc_robust['PerfMean'].mean():.3f} (Claim: 0.95)")
    print(f"  Robust RI: {arc_robust['RI'].mean():.3f} (Claim: 0.00)")

    # 4. Significance tests (matches Table 10 in paper)
    print(f"\nClaim 4: Table 10 / Significance Tests (ARC vs no_control)")
    lines = {
        "L1": {"scenarios": ["reward_flip", "noise_burst", "sudden_threat"], "arc": "arc_v1", "metrics": ["PerfMean", "RI"]},
        "L2": {"scenarios": ["distribution_shift", "goal_conflict"], "arc": "arc_v1", "metrics": ["PerfMean"]},
        "L3": {"scenarios": ["sustained_contradiction", "gaslighting", "instruction_conflict"], "arc": "arc_v1", "metrics": ["PerfMean"]},
        "L5": {"scenarios": ["adversarial_coupling", "random_dopamine"], "arc": "arc_robust", "metrics": ["PerfMean"]},
    }
    for line, spec in lines.items():
        sub = df[df["scenario"].isin(spec["scenarios"])]
        arc = sub[sub["controller"] == spec["arc"]]
        base = sub[sub["controller"] == "no_control"]
        for metric in spec["metrics"]:
            x = arc[metric].to_numpy()
            y = base[metric].to_numpy()
            p = ttest_ind(x, y, equal_var=True).pvalue
            d = cohens_d(x, y)
            print(
                f"  {line} {spec['arc']} {metric}: "
                f"ARC={x.mean():.3f}, Baseline={y.mean():.3f}, p={p:.3e}, d={d:.2f}, n={x.size}"
            )

    # 5. Scenario difficulty (global, across controllers)
    print(f"\nClaim 5: Scenario Difficulty (mean PerfMean across all controllers)")
    scenario_mean = df.groupby("scenario")["PerfMean"].mean().sort_values()
    print(scenario_mean.head(3).to_string())

if __name__ == "__main__":
    verify_claims()
