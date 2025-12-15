import argparse, os, csv, random
import yaml

from sim.state import State, performance, ccog, capacity
from sim.dynamics import step_dynamics
from tasks.scenarios import build_scenarios
from metrics.metrics import compute_metrics
from controllers.controllers import (
    NoControl,
    NaiveCalm,
    ARCv1,
    PerfOptimized,
    ARCv2_Hierarchical,
    ARCv3_MetaControl,
    ARCv1_PID,
    ARCv3_PID_Meta,
    ARCv1_LQR,
    ARCv3_LQR_Meta,
    ARCv1_LQI,
    ARC_Ultimate,
    ARCv2_LQI,
    ARC_Robust,
    ARC_Adaptive,
)

def init_state(cfg):
    return State(phi=cfg["phi0"], g=cfg["g0"], p=cfg["p0"], i=cfg["i0"],
                 s=cfg["s0"], v=cfg["v0"], a=cfg["a0"], mf=cfg["mf0"], ms=cfg["ms0"], u=cfg["u_base"])

def run_one(controller, scenario, seed, cfg):
    rng = random.Random(seed)
    st = init_state(cfg)
    trace = {k: [] for k in ["t","pe","reward","u_exog","phi","g","p","i","s","v","a","mf","ms","u","ccog","cap","perf"]}
    trace["control"] = [] # New: store control actions
    for t in range(scenario.horizon):
        # Handle scenarios that require state (interactive) vs static ones
        try:
            pe, reward, u_exog = scenario.generator(t, rng, st=st)
        except TypeError:
             # Fallback for old scenarios that don't accept st
            pe, reward, u_exog = scenario.generator(t, rng)
            
        # Provide additional signals for controllers that need them (e.g., hierarchical/meta control).
        obs = {
            "t": t,
            "pe": pe,
            "reward": reward,
            "u_exog": u_exog,
            "perf": performance(st, cfg),
            "ccog": ccog(st),
            "cap": capacity(st, cfg["omega_s"]),
        }
        u_ctrl = controller.act(st, obs, cfg)
        st = step_dynamics(st, pe=pe, reward=reward, u_exog=u_exog, control=u_ctrl, cfg=cfg)
        trace["t"].append(t); trace["pe"].append(pe); trace["reward"].append(reward); trace["u_exog"].append(u_exog)
        trace["phi"].append(st.phi); trace["g"].append(st.g); trace["p"].append(st.p); trace["i"].append(st.i)
        trace["s"].append(st.s); trace["v"].append(st.v); trace["a"].append(st.a)
        trace["mf"].append(st.mf); trace["ms"].append(st.ms); trace["u"].append(st.u)
        trace["ccog"].append(ccog(st)); trace["cap"].append(capacity(st, cfg["omega_s"]))
        trace["perf"].append(performance(st, cfg))
        trace["control"].append(u_ctrl)
    met = compute_metrics(trace, scenario.shock_t, cfg)
    return trace, met

def write_trace(path, trace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Filter keys that are lists of floats/ints for CSV
    keys = [k for k in trace.keys() if k != "control"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(len(trace[keys[0]])):
            w.writerow([trace[k][i] for k in keys])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None, help="Override output directory")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.outdir:
        out_dir = os.path.abspath(args.outdir)
    else:
        out_dir = os.path.join(os.path.dirname(args.config), "..", cfg.get("out_dir","outputs"))
        out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traces"), exist_ok=True)

    controllers = [
        NoControl(),
        NaiveCalm(),
        ARCv1(),
        ARCv1_PID(),
        ARCv1_LQR(),
        ARCv1_LQI(),
        ARC_Ultimate(),
        ARCv2_Hierarchical(),
        ARCv2_LQI(),
        ARCv3_MetaControl(),
        ARCv3_PID_Meta(),
        ARCv3_LQR_Meta(),
        ARC_Robust(),
        ARC_Adaptive(),
        PerfOptimized(),
    ]
    scenarios = build_scenarios(cfg)
    rows = []
    for sc in scenarios:
        for ctrl in controllers:
            for seed in cfg["seeds"]:
                trace, met = run_one(ctrl, sc, seed, cfg)
                write_trace(os.path.join(out_dir, "traces", f"{sc.name}__{ctrl.name}__seed{seed}.csv"), trace)
                row = {"scenario": sc.name, "controller": ctrl.name, "seed": seed}; row.update(met)
                rows.append(row)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("Wrote:", metrics_path)

if __name__ == "__main__":
    main()
