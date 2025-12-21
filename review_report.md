# Scientific Review Report: Affective Regulation Core (ARC)
**Reviewer:** Area Chair Senior
**Date:** December 18, 2025
**Global Rating:** Conditional Acceptance (Minor Revision)

## 1. SCIENTIFIC EVALUATION

### 1.1 Strengths
| Aspect | Evaluation |
| :--- | :--- |
| **Novelty** | **High.** The "regulation-first" approach is novel vs. "emotion-as-reward-shaping". |
| **Methodological Rigor** | **Solid.** 15 controllers × 11 scenarios × multiple seeds. |
| **Reproducibility** | **Excellent.** Code, data, and configurations available. |
| **Statistical Validation** | **Strong.** t-tests, Cohen's d, confidence intervals. |

### 1.2 Scientific Weaknesses

#### ⚠️ PROBLEM 1: "Zero Rumination" Claim Overgeneralized
**Abstract claims:** "97% performance with zero rumination"

**Reality (Table 3 & CSV):**
| Controller | Real RI | Status |
| :--- | :--- | :--- |
| `arc_v1_pid` | 0.00 | ✓ |
| `arc_v1_lqi` | 0.00 | ✓
| `arc_robust` | 0.00 | ✓ |
| `arc_v1` (P) | 0.15 | ✗ |
| `arc_v3_meta` | 0.09 | ✗ |
| `arc_v2_hier` | 1.22 | ✗ |

**Verdict:** The claim is valid only for controllers with integral action. The abstract must be specific.

#### ⚠️ PROBLEM 2: Omission of Collapse in Adversarial Coupling
**PerfMean Heatmap shows clearly:** In `adversarial_coupling`, several ARC controllers collapse:
*   `arc_v1_pid`: ~0.14 (worse than `no_control` with 0.41)
*   `arc_adaptive`: ~0.19
*   `arc_ultimate`: ~0.13

This is not discussed in the main text. It is a serious omission.

#### ⚠️ PROBLEM 3: "50% Improvement" Claim Verification
**Claim (Abstract):** "ARC improves transfer learning success by 50%"

**Data L6 (final_metrics.csv):**
*   Baseline ChangingGoal success: 0.39875 (39.9%)
*   ARC ChangingGoal success: 0.5975 (59.75%)
*   Improvement: (0.5975 - 0.39875) / 0.39875 = 49.8% ✓

**Verdict:** Correct but borderline claim (49.8% ≠ 50%).

### 1.3 Statistical Validation
| Metric | Statistic | Evaluation |
| :--- | :--- | :--- |
| p-values | p < 0.001 (***) in all tests | ✅ Excellent |
| Cohen's d | d = 10.1 (L1 PerfMean), d = -589.7 (L1 RI) | ⚠️ RI has extreme d due to variance ~0 |
| Sample N | 60-600 per line | ✅ Adequate |

**Note:** Cohen's d = -589.7 for RI occurs because ARC has std = 0.0. This is technically correct but should be explained.

## 2. FIGURE AUDIT (Image by Image)

### 📊 `figures_controllers/`

**`fig_arc_architecture.png`** — [APPROVED]
*   ✅ Readable text (~14pt effective)
*   ✅ Clean design, high data-ink ratio
*   ✅ Clear information flow (x(t) → Controller → u(t))
*   ⚠️ Minor: The term "Damping, Gating" in u(t) could be expanded

**`fig_benchmark_ladder.png`** — [APPROVED]
*   ✅ Excellent label readability
*   ✅ Clear L1→L6 progression
*   ✅ Visible associated metrics (right)
*   ✅ Accessible color palette (green-blue-purple gradient)

**`fig_controller_performance.png`** — [NEEDS IMPROVEMENT]
*   ✅ Visible error bars
*   ✅ Clear reference lines (Target 0.90, Baseline)
*   ⚠️ Problem: X-axis labels rotated but some cut off at the bottom
*   ⚠️ Problem: 15 controllers make reading difficult; consider grouping by type
*   🔧 **Action:** Increase bottom margin or reduce label font size

**`fig_controller_radar.png`** — [NEEDS IMPROVEMENT]
*   ✅ Excellent concept for multi-metric comparison
*   ⚠️ **CRITICAL Problem:** "Performance" label partially hidden by "1.0" scale line
*   ⚠️ Problem: 5 overlapping controllers make it difficult to distinguish individual lines
*   ⚠️ Problem: "Low Overshoot" → inverted scale is not intuitive
*   🔧 **Action:** Move radial labels outside data area; add distinct point markers

**`fig_controller_rumination.png`** — [APPROVED WITH RESERVATIONS]
*   ✅ Clear main message: "Integral Action → RI ≈ 0"
*   ✅ Visible error bars
*   ⚠️ Minor: "Warning (0.10)" line could be thicker
*   ⚠️ Observation: Very large error bars for `arc_v1` (P); this is valuable info but not discussed in text

**`fig_controller_tradeoff.png`** — [APPROVED]
*   ✅ Excellent trade-off visualization (Pareto front visible)
*   ✅ "Optimal Region" correctly marked
*   ✅ Bubble size = control effort (well documented in title)
*   ✅ Complete and readable legend
*   ⚠️ Minor: Some points overlap at (0, ~0.9); add jitter or transparency

**`fig_controller_effort.png`** — [APPROVED]
*   ✅ Clear message: "Meta-Control = Lowest Effort" (precise title)
*   ⚠️ Inconsistency with title: Title says "Meta-Control Achieves Lowest Effort" but ARC Meta has effort ~0.61, while Naive Calm has ~0.25. Naive Calm has lower effort.
*   🔧 **CRITICAL Action:** Correct title or explain "lowest among ARC variants"

**`fig_heatmap_perfmean.png`** — [NEEDS IMPROVEMENT]
*   ✅ Rich and useful information
*   ⚠️ Problem: No numerical values in cells; difficult to read exact values
*   ⚠️ Problem: "Adversarial Coupling" row shows PID/LQI collapse but not interpreted in text
*   🔧 **Action:** Add value annotation in cells (at least for extremes)

**`fig_heatmap_ri.png`** — [NEEDS IMPROVEMENT]
*   ✅ Clear pattern: PID/LQI/Robust → RI ~0 (dark blue)
*   ⚠️ Problem: Same as above - no numerical values
*   ⚠️ Observation: `arc_v2_hier` has high RI in almost all scenarios → not discussed

**`fig_heatmap_rt.png`** — [APPROVED]
*   ✅ Evident pattern: ARC controllers → Low RT (blue)
*   ✅ Appropriate colormap (viridis)

**`fig_heatmap_effort.png`** — [APPROVED]
*   ✅ Clearly shows PID has max effort (~2.5)
*   ✅ Meta controllers have moderate effort

### 📊 `figures_L6/`

**`learning_curves.png`** — [NEEDS IMPROVEMENT]
*   ✅ 3 well-organized subplots
*   ✅ Visible variance shadow (not too opaque)
*   ⚠️ **CRITICAL Problem:** In ChangingGoal, ARC and Baseline curves are very hard to distinguish due to high variance and overlap
*   ⚠️ Problem: Legend is inside data area (bottom right corner)
*   🔧 **Action:** Move legend outside area; add vertical line at goal changes (episodes 50, 100, 150)

**`state_dynamics.png`** — [NEEDS IMPROVEMENT]
*   ✅ 4 informative subplots
*   ⚠️ Problem: Subplots too tight - Y-axis labels partially overlap
*   ⚠️ Problem: In "ARC Internal State: Arousal", red zone (safe threshold) dominates visually and distracts
*   ⚠️ Problem: "Episode Length" subplot has very noisy lines; consider smoothing
*   🔧 **Action:** Increase space between subplots; reduce red zone opacity

**`metrics_comparison.png`** — [NEEDS IMPROVEMENT]
*   ✅ Clear side-by-side comparison
*   ⚠️ Problem: Error bars (yellow squares) barely visible - too small
*   ⚠️ Problem: In "Mean Arousal", baseline has arousal = 0 (no ASSB state), which is misleading
*   🔧 **Action:** Explain in caption that baseline has no arousal tracking, or remove that subplot

**`ablation_summary.png`** — [APPROVED WITH RESERVATIONS]
*   ✅ Clear ablation message: "DMN control prevents rumination"
*   ✅ Effective "CRITICAL!" annotation
*   ⚠️ Problem: Left subplot Y-axis (Performance) starts at 0.90, not 0. This exaggerates differences
*   🔧 **Action:** Either start Y-axis at 0, or add note about truncated scale

**`efficiency_comparison.png`** — [APPROVED]
*   ✅ Clear message about similar convergence
*   ✅ Dark background with bright lines = good readability
*   ⚠️ Minor: Informal title "who is FASTER?" - consider more academic tone

**`correlation_combined.png` (analysis/)** — [APPROVED]
*   ✅ Clear correlation matrix
*   ✅ Numerical values in cells ✓
*   ✅ Appropriate divergent colormap (blue-red)
*   ✅ Black diagonal = correct (autocorrelation omitted)

**`sensitivity_controller.png` (analysis/)** — [NEEDS IMPROVEMENT]
*   ✅ Comparison of 6 controllers
*   ⚠️ Problem: Dark background but X-axis labels hard to read (low contrast)
*   ⚠️ Problem: Error bars very small - hard to see

## 3. DETECTED NUMERICAL OR TEXT ERRORS

### 3.1 Numerical Errors
| Location | Claim | Real Data | Discrepancy |
| :--- | :--- | :--- | :--- |
| Abstract | "97% performance" | 96.58% (L1 PerfMean) | ±0.4% - acceptable |
| Abstract | "30% baseline" | 29.71% | ±0.3% - acceptable |
| Abstract | "21% effort reduction" | (0.78-0.61)/0.78 = 21.8% | ✅ Correct |
| Abstract | "50% RL improvement" | 49.8% | Borderline |
| Fig effort title | "Meta = Lowest Effort" | Naive Calm = 0.25 < Meta = 0.61 | ⛔ **ERROR** |

### 3.2 Text Errors / Typos
| Location | Error |
| :--- | :--- |
| Section 7.3 | "dual control dilemma" → should be "persistence of excitation" |
| Appendix D.3 | `s_rum_tau` = 0.6 vs config `s_rum_tau` = 0.55 → inconsistency |
| Fig captions | Some color references ("dark teal") don't work in B&W |

### 3.3 Config vs Paper Inconsistencies
| Parameter | Paper | Config (`v2.yaml`) | Impact |
| :--- | :--- | :--- | :--- |
| `s_safe` | 0.55 (Appendix F) | 0.60 | Affects risk calculation |
| `s_rum_tau` | 0.6 (Appendix D.3) | 0.55 | Affects RI definition |

## 4. DEFECT SUMMARY BY SEVERITY

### 🔴 CRITICAL (Block acceptance)
*   Incorrect title in `fig_controller_effort.png` - says "Meta = Lowest" but Naive Calm is lower
*   Omission of PID/LQI adversarial collapse - important finding not reported
*   Inconsistency `s_rum_tau` (0.55 vs 0.6) - affects main metric (RI)

### 🟡 IMPORTANT (Must be corrected)
*   "Zero rumination" claim doesn't apply to all ARCs - only integral controllers
*   `learning_curves.png` ChangingGoal - indistinguishable curves
*   `state_dynamics.png` - tight subplots, overlapping labels
*   `metrics_comparison.png` - baseline arousal = 0 is misleading
*   Heatmaps without numerical values - difficult precise reading

### 🟢 MINOR (Recommended)
*   `fig_controller_radar.png` - "Performance" label cut off
*   `ablation_summary.png` - truncated Y axis exaggerates differences
*   Rotated labels cut off in several bar charts
*   Terminology "dual control" - imprecise

## 5. FINAL VERDICT
| Criterion | Score | Comment |
| :--- | :--- | :--- |
| Novelty | 8/10 | "Regulation-first" approach is genuinely new |
| Technical Solidity | 7/10 | Strong, but omissions in failure discussion |
| Clarity | 7/10 | Well written, but figures have problems |
| Reproducibility | 9/10 | Complete code and data |
| Figures | 6/10 | Several readability/precision issues |
| Truthfulness of Claims | 7/10 | Generally correct but overgeneralized |

**RECOMMENDATION: CONDITIONAL ACCEPTANCE (Minor Revision)**

The paper makes valuable contributions to the intersection of control theory and affective computing. However, it requires:
1.  Correction of the erroneous title in `fig_controller_effort`
2.  Explicit discussion of the collapse of integral controllers in adversarial scenarios
3.  Harmonization of parameters (`s_rum_tau`)
4.  Readability improvements in 4-5 key figures
5.  Qualification of claims regarding "zero rumination"

With these corrections, the paper would reach the level of publication in top-tier venues.
