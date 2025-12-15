"""
Ablation Study para ARC - Estudio L1 completo.
Ejecuta solo los controladores de ablation para comparar con ARC full.
"""
import argparse, os, csv, random
import yaml

from sim.state import State, performance, ccog, capacity
from sim.dynamics import step_dynamics
from tasks.scenarios import build_scenarios
from metrics.metrics import compute_metrics
from controllers.controllers import (
    ARCv1, ARC_NoDMG, ARC_NoCalm, ARC_NoMem, ARC_NoReapp
)

def init_state(cfg):
    return State(phi=cfg["phi0"], g=cfg["g0"], p=cfg["p0"], i=cfg["i0"],
                 s=cfg["s0"], v=cfg["v0"], a=cfg["a0"], mf=cfg["mf0"], ms=cfg["ms0"], u=cfg["u_base"])

def run_one(controller, scenario, seed, cfg):
    rng = random.Random(seed)
    st = init_state(cfg)
    trace = {k: [] for k in ["t","pe","reward","u_exog","phi","g","p","i","s","v","a","mf","ms","u","ccog","cap","perf"]}
    for t in range(scenario.horizon):
        pe, reward, u_exog = scenario.generator(t, rng)
        u_ctrl = controller.act(st, {"pe":pe,"reward":reward,"u_exog":u_exog}, cfg)
        st = step_dynamics(st, pe=pe, reward=reward, u_exog=u_exog, control=u_ctrl, cfg=cfg)
        trace["t"].append(t); trace["pe"].append(pe); trace["reward"].append(reward); trace["u_exog"].append(u_exog)
        trace["phi"].append(st.phi); trace["g"].append(st.g); trace["p"].append(st.p); trace["i"].append(st.i)
        trace["s"].append(st.s); trace["v"].append(st.v); trace["a"].append(st.a)
        trace["mf"].append(st.mf); trace["ms"].append(st.ms); trace["u"].append(st.u)
        trace["ccog"].append(ccog(st)); trace["cap"].append(capacity(st, cfg["omega_s"]))
        trace["perf"].append(performance(st, cfg))
    met = compute_metrics(trace, scenario.shock_t, cfg)
    return trace, met

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--outdir", default="outputs_ablation")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds for ablation")
    args = ap.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Override seeds for faster ablation
    seeds = list(range(1, args.seeds + 1))
    
    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    # Ablation controllers + full ARC for comparison
    controllers = [
        ARCv1(),        # Full ARC (reference)
        ARC_NoDMG(),    # Sin control DMN
        ARC_NoCalm(),   # Sin control arousal
        ARC_NoMem(),    # Sin gating memoria
        ARC_NoReapp(),  # Sin reappraisal
    ]
    
    scenarios = build_scenarios(cfg)
    rows = []
    
    total = len(scenarios) * len(controllers) * len(seeds)
    done = 0
    
    for sc in scenarios:
        for ctrl in controllers:
            for seed in seeds:
                trace, met = run_one(ctrl, sc, seed, cfg)
                row = {"scenario": sc.name, "controller": ctrl.name, "seed": seed}
                row.update(met)
                rows.append(row)
                done += 1
                if done % 20 == 0:
                    print(f"Progress: {done}/{total}")

    metrics_path = os.path.join(out_dir, "ablation_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote: {metrics_path}")
    print(f"Total runs: {len(rows)}")
    
    # Generar resumen rápido
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = (row['scenario'], row['controller'])
        groups[key].append(row)
    
    print(f"\n{'Scenario':<16} {'Controller':<14} {'PerfMean':>10} {'RT':>8} {'RI':>8} {'NDR':>8}")
    print("-"*70)
    
    for scenario in ['reward_flip', 'noise_burst', 'sudden_threat']:
        for ctrl in ['arc_v1', 'arc_no_dmg', 'arc_no_calm', 'arc_no_mem', 'arc_no_reapp']:
            key = (scenario, ctrl)
            if key in groups:
                g = groups[key]
                n = len(g)
                perf = sum(float(r['PerfMean']) for r in g) / n
                rt = sum(float(r['RT']) for r in g) / n
                ri = sum(float(r['RI']) for r in g) / n
                ndr = sum(float(r['NDR']) for r in g) / n
                print(f"{scenario:<16} {ctrl:<14} {perf:>10.3f} {rt:>8.1f} {ri:>8.3f} {ndr:>8.3f}")
        print()

if __name__ == "__main__":
    main()
