# arXiv Submission Package — ARC Paper

**Title:** Affective Regulation Core: A Homeostatic Control Framework for Stable and Safe AI Agents
**Author:** J. Eduardo Damián Reynoso
**Citation style:** Damián 2026 (primary surname "Damián"; "Reynoso" is the secondary/maternal surname per Spanish convention)

## Contents

| File/folder | Purpose |
|---|---|
| `main.tex` | Complete paper source (40 pages after compilation) |
| `arxiv.sty` | Paper style class (ships with the submission; arXiv does not have this installed by default) |
| `figures/` | All 29 PNG figures referenced from `main.tex` |
| `README.md` | This file (informational only; NOT uploaded to arXiv) |

## How to upload to arXiv

1. **Zip the contents** (do NOT include `README.md` in the zip):
   ```bash
   cd arxiv_submission
   zip -r arc-paper.zip main.tex arxiv.sty figures/
   ```
2. Go to https://arxiv.org/submit
3. Category suggestions (primary → secondary):
   - Primary: `cs.AI` (Artificial Intelligence)
   - Secondary: `cs.LG` (Machine Learning), `eess.SY` (Systems and Control)
4. Upload the zip. arXiv will auto-compile with its internal TeX Live.
5. Download the preview PDF and verify visually before clicking "Submit."

## Compilation requirements (for local test)

- pdflatex ≥ TeX Live 2020 or MiKTeX
- Standard packages: `arxiv`, `hyperref`, `graphicx`, `booktabs`, `amsmath`, `amssymb`, `algorithm`, `algpseudocode`, `xcolor`, `float`, `placeins`, `needspace`, `etoolbox`, `adjustbox`, `microtype`, `doi`

Run twice for cross-references:
```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Verification checklist (done 2026-04-18)

- [x] Compiles cleanly in MiKTeX (40 pages, no undefined refs)
- [x] Author name renders as "J. Eduardo Damián Reynoso"
- [x] Cross-references resolved (Table/Figure/Section)
- [x] Bibliography entry `Damian2026b` visible in references (for future cross-paper linking)
- [x] Abstract reframed: 96.6% is L1-specific; global claim is PerfMean ≥ 0.93
- [x] Cohen's d column shows "saturated" (was -589.7)
- [x] +49.8% tabular RL claim is paired with "+79.9% memory gating alone" ablation
- [x] NDR metric formally defined in §5.2
- [x] DQN negative result elevated to main body §6.7 (not only appendix)
- [x] "Mental Health Tax" renamed to "Regulation-Performance Trade-off"
- [x] §7.5 Future Work expanded; references Paper #2 (`Damian2026b`, in preparation)

## arXiv categories rationale

- **cs.AI** is the natural home — AI agents with internal affective states
- **cs.LG** because of the RL component (L6, Q-learning ablations)
- **eess.SY** because the contribution is fundamentally control-theoretic (15 controllers including H∞, LQR, LQI)

Cross-listing to eess.SY is worth attempting because the paper's Theorem 1 is a genuine control-theoretic result; it helps reach the control-systems community.

## License to attach on arXiv

Recommended: **CC BY 4.0** (Creative Commons Attribution) — allows redistribution and reuse with attribution. Maximizes citations and field impact.
