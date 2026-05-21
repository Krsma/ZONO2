# CoRL 2026 Draft

This directory contains an anonymous CoRL 2026 draft for the predictive
zonotope reduction project.

## Files

- `main.tex`: paper draft.
- `references.bib`: bibliography used by the draft.
- `corl_2026.sty`, `corlabbrvnat.bst`: official CoRL 2026 template files downloaded from the author instructions page.
- `figures/`: selected PDF figures copied from `results/tacas-main/figures/figures/`.
- `template_example.tex`, `template_example.bib`: untouched CoRL template examples for reference.

## Build

From this directory:

```sh
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Current Evidence Base

The draft is based on the existing `results/tacas-main` artifact only. The
current claims intentionally avoid promising real-robot evidence or learned
policy dominance. A stronger CoRL submission should add a robotics-facing
experiment and a learned-policy ablation trained against the focused expert.
