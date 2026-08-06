# Actions.md compliance report

Date: 2026-08-06

## Scope

The revision work was executed on the active v1.0 manuscript and its companion
submission files. The clean manuscript remains self-contained: it does not refer
to an earlier manuscript version, a previous version, reviewer text, or a
companion article. The red/blue comparison file is the only artifact that keeps
deleted text, by design.

## Actions Covered

| Action area | Status | Evidence |
|---|---:|---|
| Novelty repositioning | Done | Abstract, Introduction, Related Work, Discussion, Conclusion now center the contribution on serialization-aware active PEAK/ZERO mapping selection. Net capacity itself is treated as established accounting, not as the novelty. |
| Central ablation | Done | Table A0/A1/A2 isolates gross reporting, post-hoc wire charging, and serialized-net activation. Labels are synchronized in `main.tex`, `run_revision_ablations.py`, and the JSON/CSV results. |
| Exact wire model | Done | `main.tex` and `abm_rdh.py` agree on the byte-level ABM stream: marker byte, magic/version, `uint32` payload length, `uint16` table count, and byte-aligned joint combinatorial/base-9 state. |
| CNN ablation | Done | CNN diagnostic uses 20 epochs, same optimizer/loss/LR/batch/split/seeds, depth/window/width variants, MAE/Pearson/DRD, parameter count, training time, and inference time. |
| CNN positioning | Done | The CNN is presented as a learned distortion ranker; the core novelty is the serialization-aware mapping selection. The title was not changed. |
| Baseline serialization sensitivity | Done | Claims are restricted to the tested 104-bit fixed-header credit variation; no broad serialization-robustness claim is made. |
| Operating envelope | Done | The text states that no covert-security guarantee is claimed; 256/512 bits are interpreted as capacity-oriented reversible annotation settings. |
| BOSSbase external test | Done | Frozen document-trained CNN is evaluated on 100 thresholded BOSSbase images, with net-aware ON/OFF comparison and exact recovery checks. |
| Effect sizes | Done | Main paired comparisons at 256 bits report median differences, Holm-adjusted tests, and rank-biserial correlations. |
| Sections 3.3/3.4 explanations | Done | Embedding and extraction procedures include descriptive and justificatory text before the figures/algorithms. |
| Positive wording | Done | Abstract through Conclusion were edited to avoid unnecessarily negative framing while preserving scientific limitations. |
| Tables and captions | Done | Table 3 and Table 5 have one caption each; Tables 5 and 7 use text-size tabular formatting rather than resized font. |
| Figure 10 | Done | Figure 10 is reduced, split into two side-by-side panels, labelled, cited, and interpreted in the text. |
| Figures/tables/equations | Done | All 22 floats have one caption, a label, and at least one reference in the text. Equations are numbered where structurally used. |
| Bibliography | Done | Frazier 2018 is restored in the GP paragraph; Sedighi 2018 supports the future detectability-aware direction. Bibliography now has 28 cited keys and 28 entries, with no missing or uncited entries. |
| Declarations | Done | The manuscript retains non-AI declarations for ethics, competing interests, funding, data availability, and code availability. No AI-use declaration is included. Cover letters use `Submission statements`. |
| Response to reviewers | Done | `response_to_reviewers.tex/pdf` follows the requested template style and is aligned to the experiments actually performed. |
| Cover letter | Done | `revision_cover_letter.tex/pdf` follows the requested submission-cover-letter model. |

## Numerical Traceability

- Document benchmark: 2874 gross bits, 629 auxiliary bits, 2245 net bits, mean
  DRD 0.760; all values checked against serialized output.
- BOSSbase frozen transfer: net-aware ON gives 669.6 gross, 377.1 auxiliary,
  292.5 net bits, median net 159, 75/100 net-positive images, mean DRD 0.300,
  exact recovery for all rows.
- BOSSbase net-aware OFF gives 832.5 gross, 731.4 auxiliary, 101.1 net bits,
  median net -31.5, 47/100 net-positive images, exact recovery for all rows.
- CNN architecture table matches `revision_results/cnn_architecture_ablation.json`.
- Ranker table was corrected to match `revision_results/ranker_ablation.json`.
- Steganalysis AUC table matches `sota_multiload/steganalysis.json`.

## Verification

- `paper/main.pdf` compiles to 17 pages.
- `paper/response_to_reviewers.pdf`, `paper/revision_cover_letter.pdf`, and
  `paper/cover_letter.pdf` compile without unresolved citations/references.
- Bibliography/label audit: 28 cited keys, 28 BibTeX entries, no missing cited
  keys, no uncited entries, no duplicate labels, no missing refs, 22 labelled
  and cited floats.
- Python syntax check and test suite were run after the main code changes; the
  final test pass reported 54 passed tests with one non-fatal scikit-learn
  convergence warning.

## Deliverables

- Clean article: `paper/main.pdf` and `paper/article_v1_0_clean.pdf`.
- Blue-only changes: `paper/article_v1_0_blue_changes.tex` and
  `paper/article_v1_0_blue_changes_updated.pdf`.
- Red deletions plus blue additions:
  `paper/article_v1_0_red_blue_changes.tex/pdf`.
- Response to reviewers: `paper/response_to_reviewers.tex/pdf`.
- Revision cover letter: `paper/revision_cover_letter.tex/pdf`.
- Source package archive: `paper/latex_source_v1_0_revised.zip`.

Note: `paper/article_v1_0_blue_changes.pdf` is locked by an external Windows
process and could not be overwritten. The up-to-date compiled PDF is
`paper/article_v1_0_blue_changes_updated.pdf`.
