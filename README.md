# Net-Aware Learned Block Mapping for Binary-Image RDH

This repository contains the implementation and reproducibility material for
the article *Net-Aware Learned Block Mapping for Reversible Data Hiding in
Binary Images*.

The project evaluates adaptive block mapping (ABM), PPOCP, Huynh--Nguyen, and
Dong under common messages, payload targets, distortion computation,
reversibility checks, and serialized auxiliary-bit accounting.

## Repository contents

- `abm_rdh.py`: proposed reversible block-mapping implementation.
- `ppocp.py`, `huynh_nguyen.py`, `dong_adaptive.py`: independently
  implemented comparison methods.
- `ml_candidate_ranker.py`: CNN candidate-cost model.
- `evaluate_*.py`: capacity, placement, resource, and steganalysis protocols.
- `tests/`: unit and round-trip tests.
- `ml_ranker_results/`: trained model and held-out evaluation reports.
- `sota_multiload/`: per-payload CSV/JSON results and statistical analyses.
- `paper/`: JVCIR manuscript, cover letter, highlights, figures, and LaTeX
  dependencies.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
```

## Datasets

The eight binary document images used for document-scale validation are stored
in `paper/images/`.

BOSSbase is not redistributed. Download BOSSbase 1.01 from the
[Binghamton DDE distribution](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip),
then pass its directory explicitly:

```bash
python evaluate_sota_baselines.py --dataset /path/to/BOSSbase
```

Alternatively, set `BOSSBASE_DIR` before running the evaluation scripts. The
reported protocol uses seed `20260608`, 100 profile images, 100 disjoint test
images, threshold 128, and payload targets 64, 128, 256, and 512 bits.

## Principal checks

```bash
python run_experiments.py --policy hamming1
python evaluate_loso_placement.py
python analyze_sota_multiload.py
python evaluate_steganalysis.py
python benchmark_sota_resources.py
python plot_article_results.py
```

The committed CSV and JSON files preserve the result tables used by the
manuscript. Every accepted embedding run is checked for exact message recovery
and bitwise cover restoration.

## Building the paper

From `paper/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex cover_letter.tex
pdflatex graphical_abstract.tex
```

## Citation

Citation metadata will be updated when the article receives its final
bibliographic record.
