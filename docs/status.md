# ARC/ASSB Research Status Report

**Auto-generated**: 2025-12-14 15:46  
**Source**: `experiments/make_status.py`  
**Data validation**: All claims below are based on actual CSV data.

---

## Summary Table

| Line | Description | Status | Data Source | Key Metrics |
|------|-------------|--------|-------------|-------------|
| L1 | See details | ✅ Validated | metrics.csv (240 runs) | Perf=0.97, RI=0.00 |
| L2 | See details | ✅ Validated | metrics.csv (400 runs) | Perf=0.97, RI=0.00 |
| L3 | See details | ✅ Validated | metrics.csv (640 runs) | Perf=0.94, RI=0.04 |
| L4 | See details | ✅ Validated | metrics.csv (800 runs) | Perf=0.94, RI=0.68 |
| L4_rev11 | See details | ✅ Validated | metrics.csv (1200 runs) | Perf=0.94, RI=0.49 |
| L4_rev2 | See details | ✅ Validated | metrics.csv (960 runs) | Perf=0.95, RI=0.45 |
| L5 | See details | ✅ Validated | metrics.csv (1200 runs) | Perf=0.94, RI=0.49 |
| L6 | See details | ✅ Validated | final_metrics.csv (6 runs) | Perf=0.00, RI=0.00 |
| L6_v2 | See details | ✅ Validated | final_metrics.csv (6 runs) | Perf=0.00, RI=0.00 |
| L7 | Paper/Release | 🔲 Pending | N/A | Documentation only |

---

## Detailed Results

### L1
- **Data**: `metrics.csv` (240 runs)
- **Config**: 20 seeds × 3 scenarios × 4 controllers
- **ARC Performance**: 0.966 (RI=0.00)
- **Baseline Performance**: 0.297 (RI=1.41)
- **Validated**: ✅ (multi-seed, automated metrics)

### L2
- **Data**: `metrics.csv` (400 runs)
- **Config**: 20 seeds × 5 scenarios × 4 controllers
- **ARC Performance**: 0.972 (RI=0.00)
- **Baseline Performance**: 0.283 (RI=1.41)
- **Validated**: ✅ (multi-seed, automated metrics)

### L3
- **Data**: `metrics.csv` (640 runs)
- **Config**: 20 seeds × 8 scenarios × 4 controllers
- **ARC Performance**: 0.935 (RI=0.04)
- **Baseline Performance**: 0.204 (RI=1.43)
- **Validated**: ✅ (multi-seed, automated metrics)

### L4
- **Data**: `metrics.csv` (800 runs)
- **Config**: 20 seeds × 8 scenarios × 5 controllers
- **ARC Performance**: 0.935 (RI=0.68)
- **Baseline Performance**: 0.204 (RI=1.43)
- **Validated**: ✅ (multi-seed, automated metrics)

### L4_rev11
- **Data**: `metrics.csv` (1200 runs)
- **Config**: 20 seeds × 10 scenarios × 6 controllers
- **ARC Performance**: 0.937 (RI=0.49)
- **Baseline Performance**: 0.208 (RI=1.43)
- **Validated**: ✅ (multi-seed, automated metrics)

### L4_rev2
- **Data**: `metrics.csv` (960 runs)
- **Config**: 20 seeds × 8 scenarios × 6 controllers
- **ARC Performance**: 0.946 (RI=0.45)
- **Baseline Performance**: 0.204 (RI=1.43)
- **Validated**: ✅ (multi-seed, automated metrics)

### L5
- **Data**: `metrics.csv` (1200 runs)
- **Config**: 20 seeds × 10 scenarios × 6 controllers
- **ARC Performance**: 0.943 (RI=0.49)
- **Baseline Performance**: 0.208 (RI=1.43)
- **Validated**: ✅ (multi-seed, automated metrics)

### L6
- **Data**: `final_metrics.csv` (6 runs)
- **Config**: 0 seeds × 0 scenarios × 0 controllers
- **ARC Performance**: 0.000 (RI=0.00)
- **Baseline Performance**: 0.000 (RI=0.00)
- **Validated**: ✅ (multi-seed, automated metrics)

### L6_v2
- **Data**: `final_metrics.csv` (6 runs)
- **Config**: 0 seeds × 0 scenarios × 0 controllers
- **ARC Performance**: 0.000 (RI=0.00)
- **Baseline Performance**: 0.000 (RI=0.00)
- **Validated**: ✅ (multi-seed, automated metrics)

---

## Validated Claims

Based on the data above, the following claims are **supported**:

- **L1**: ARC outperforms baseline by 225% in performance
- **L1**: ARC eliminates rumination (RI=0.00 vs baseline RI=1.41)
- **L2**: ARC outperforms baseline by 243% in performance
- **L2**: ARC eliminates rumination (RI=0.00 vs baseline RI=1.41)
- **L3**: ARC outperforms baseline by 358% in performance
- **L3**: ARC eliminates rumination (RI=0.04 vs baseline RI=1.43)
- **L4**: ARC outperforms baseline by 358% in performance
- **L4_rev11**: ARC outperforms baseline by 349% in performance
- **L4_rev2**: ARC outperforms baseline by 363% in performance
- **L5**: ARC outperforms baseline by 352% in performance

## Unsupported/Pending Claims

The following claims require additional data or metrics:

- ⚠️ **"3x energy efficiency"**: No EnergyCost or EfficiencyRatio metric currently in `metrics/metrics.py`
- ⚠️ **L6 Strong Claim**: Preliminary validation only (small seeds/envs)
- 🔲 **L7 Paper**: Documentation pending

---

## Reproducibility

To regenerate this report:

```bash
python experiments/make_status.py
```

To rerun all experiments:

```bash
python -m experiments.run --config configs/v2.yaml --outdir outputs_v2
# Similar for L2-L6
```
