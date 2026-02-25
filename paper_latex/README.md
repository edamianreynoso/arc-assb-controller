# Paper (LaTeX)

Main file: `main.tex` (figures: `figures/`).

Note on bibliography:
- `main.tex` currently uses an inline `thebibliography` section (no BibTeX step).
- `main_es.tex` uses `references.bib` with `natbib`.

## Build locally

You need a LaTeX distribution (TeX Live or MiKTeX).

```bash
cd paper_latex
pdflatex main.tex
pdflatex main.tex
```

If compiling the Spanish manuscript (`main_es.tex`), run:

```bash
cd paper_latex
pdflatex main_es.tex
bibtex main_es
pdflatex main_es.tex
pdflatex main_es.tex
```

## arXiv upload

Create the arXiv source ZIP:

```powershell
.\make_arxiv_zip.ps1
```

Then upload `release_assets/arxiv_upload.zip` to arXiv and select `main.tex` as the main TeX file.
