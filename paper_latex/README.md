# Paper (LaTeX)

Main file: `main.tex` (figures in `figures/`).

Bibliography is inline via `\begin{thebibliography}` (no BibTeX step required).

## Build locally

You need a LaTeX distribution (TeX Live or MiKTeX).

```bash
cd paper_latex
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-references
```

## arXiv upload

Upload the following files to arXiv:
- `main.tex`
- `arxiv.sty`
- `figures/` (all 27 PNG files)

Select `main.tex` as the main TeX file.
