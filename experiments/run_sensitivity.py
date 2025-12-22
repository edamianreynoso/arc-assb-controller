"""
Sensitivity Analysis: Safety Thresholds Sweep
Evaluates the impact of a_safe and s_safe on PerfMean and RI.
"""

import os
import sys
import csv
import argparse
import yaml
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.scenarios import build_scenarios
from controllers.controllers import ARCv1
from experiments.run import run_one

def main():
    parser = argparse.ArgumentParser(description="Threshold Sensitivity Analysis")
    parser.add_argument("--config", default="configs/v2.yaml")
    parser.add_argument("--outdir", default="outputs_sensitivity")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
        
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Sweep parameters
    a_values = [0.4, 0.5, 0.6, 0.7, 0.8]
    s_values = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Scenario and Controller
    scenarios = build_scenarios(base_cfg)
    target_scenario = [s for s in scenarios if s.name == "reward_flip"][0]
    controller = ARCv1()
    
    results = []
    
    print(f"Running Sensitivity Analysis on scenario: {target_scenario.name}")
    print("=" * 60)
    
    for a_safe in a_values:
        for s_safe in s_values:
            print(f"Testing a_safe={a_safe:.1f}, s_safe={s_safe:.1f}...")
            
            # Update config for this run
            cfg = base_cfg.copy()
            cfg["a_safe"] = a_safe
            cfg["s_safe"] = s_safe
            
            # Run multiple seeds for stability
            seeds = base_cfg["seeds"][:5] # Use first 5 seeds for speed
            perfs = []
            ris = []
            
            for seed in seeds:
                _, met = run_one(controller, target_scenario, seed, cfg)
                perfs.append(met["PerfMean"])
                ris.append(met["RI"])
            
            results.append({
                "a_safe": a_safe,
                "s_safe": s_safe,
                "perf_mean": np.mean(perfs),
                "ri_mean": np.mean(ris),
                "n_seeds": len(seeds)
            })
            
    # Save results
    out_path = os.path.join(out_dir, "sensitivity_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
        
    print(f"\nResults saved to: {out_path}")
    
    # Print summary
    print("\nSUMMARY (a_safe vs s_safe):")
    print("a_safe | s_safe | PerfMean | RI")
    print("-" * 35)
    for r in results:
        print(f"{r['a_safe']:.1f}    | {r['s_safe']:.1f}    | {r['perf_mean']:.3f}    | {r['ri_mean']:.3f}")

if __name__ == "__main__":
    main()
