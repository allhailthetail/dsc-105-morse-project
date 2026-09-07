# Legacy Archive

This directory holds the project's original implementation, preserved for reference and history but no longer part of the active codebase.

- `main.org`, `functions.org` — the original Emacs Org-mode literate analysis (R via Org-babel). `functions.org` tangled into `R/project_functions.R`; `main.org` ran the EDA/denoising pipeline over the 200 real CW recordings. The 60 hardcoded transcription labels embedded in `main.org` were extracted into [`data/real/labels.csv`](../../data/real/labels.csv) during the Python migration.
- `docs/` — the LaTeX abstract (`abstract.tex`, `simplemargins.sty`), the Morse timing reference (`PARIS-standard.tex`), the bibliography (`refs.bib`), and the presentation slide deck PDF.
- `presentation/presentation-material.org` — the Org-mode script used to regenerate plots/audio for the slide deck. Its audio assets (`presentation/data/`) were **not** archived here — they were repurposed as the environmental noise-bed corpus at [`data/noise/`](../../data/noise/) for synthetic training-data generation.

See the root [`README.md`](../../README.md) and [`CLAUDE.md`](../../CLAUDE.md) for the current Python project.
