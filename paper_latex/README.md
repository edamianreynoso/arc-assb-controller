# Paper (LaTeX)

Main file: `main.tex` (bibliography: `references.bib`, figures: `figures/`).

## Build locally

You need a LaTeX distribution (TeX Live or MiKTeX).

```bash
cd paper_latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## arXiv upload

Create the arXiv source ZIP:

```powershell
.\make_arxiv_zip.ps1
```

Then upload `release_assets/arxiv_upload.zip` to arXiv and select `main.tex` as the main TeX file.
