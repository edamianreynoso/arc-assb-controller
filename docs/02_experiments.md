# Experimentos V0.1 (procedimiento reproducible)

## E0 - Smoke test (1 seed)
```bash
python -m assb.run_all --outdir outputs --seeds 0 --steps 2000
```
Revisa:
- `outputs/metrics_all.csv`
- `outputs/plots/*`

## E1 - Validación H1-H4 (30 seeds)
```bash
python -m assb.run_all --outdir outputs_30 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
```

## E2 - Barrido de parámetros (calibración)
Estrategia:
- fija tareas (reward_flip, interruption_burst)
- barrer `k_homeo`, `k_dmg`, `arousal_set`, `dmg_max`
- objetivo: minimizar RT + RI sin sacrificar performance_mean

> En v0.2 lo automatizamos con un script de grid-search.

## E3 - Ablations
Crea variantes del agente ARC en `assb/agents/arc.py` y repite E1.

## Salida esperada para paper
- tabla por tarea: (perf_mean, RT, OS, RI, NDR, VD) con IC
- figura: curvas perf/arousal/dmg (promedio ± banda)
- discusión: trade-offs + límites del modelo
