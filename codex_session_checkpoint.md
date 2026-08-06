# Codex session checkpoint

Date: 2026-08-06

Current objective:
- Resolve all issues identified in the manuscript/project audit and the user's priority list.
- Preserve reproducibility by regenerating or restricting claims to available evidence.

Important user instructions:
- Keep the article as version 1.0, with no framing as a previous or companion version.
- Do not restore old optimization sections.
- Reintroduce only the Frazier 2018 Bayesian optimization citation in the GP paragraph.
- Keep the title unchanged.
- Reposition the CNN as a secondary learned ranker, not the core novelty.
- Ensure exact wire-model accounting.
- Save this checkpoint after each important modification.

Files modified so far:
- `run_revision_ablations.py`
  - Added CNN architecture width variants.
  - Added validation inference timing fields.
  - Changed architecture default to 20 epochs.
  - Added `--only architecture` mode.
  - Added cached image positions/contexts in `PatchRanker`.
  - Added incremental JSON/CSV writes after each CNN architecture configuration.
  - Replaced the cache key by `(shape, binary bytes)` to avoid temporary-array id reuse.
- `analyze_sota_multiload.py`
  - Added rank-biserial effect size computation.
  - Added fixed-cohort bootstrap summary output.
- `paper/main.tex`
  - Corrected pattern-index formula to MSB-first row-major order.
  - Added Frazier 2018 citation to the GP Expected Improvement paragraph.
  - Replaced the wire-model description/table with byte-level ABM accounting including marker, magic, version, payload length, table count, and byte-aligned joint state.
  - Updated CNN architecture ablation table with 20-epoch depth/window/width/inference results.
  - Restricted serialization-sensitivity wording to the tested fixed-header variation.
  - Updated ML validation/RF/BOSSbase claims with regenerated results.
  - Added rank-biserial effect sizes at 256 bits.
  - Updated fixed-cohort CIs from regenerated `analysis.json`.
  - Removed the Declarations section.
- `paper/references.bib`
  - Began key cleanup for cited numeric/weak keys.
  - A broad replacement error was introduced and then corrected; remaining occurrences of `Yin2020SymmetricFlipping` and `Xuan2008RunLength` are intended citation keys.
  - Removed uncited entries after reintroducing `Frazier2018Bayesian` and citing `Sedighi2018Content`.
- `CITATION.cff`
  - Recreated with correct UTF-8 author accents.
- `paper/response_to_reviewers.tex`
  - Restricted serialization sensitivity response to the fixed-header variation.
  - Corrected CNN architecture response to reflect depth/window/width/inference results and the w32 MAE result.
- `paper/revision_cover_letter.tex`
  - Updated CNN diagnostic description and narrowed detectability wording.
  - Replaced `Author declarations` heading with `Submission statements`.
- `paper/cover_letter.tex`
  - Synchronized mechanically with `paper/revision_cover_letter.tex` to avoid conflicting cover-letter positions.
- `paper/figure_captions_and_alt_text.txt`
  - Added Figure 10 caption/alt-text entry.
- `paper/novelty_position_review.md` and `paper/systematic_novelty_audit.md`
  - Replaced old BOSSbase 516.1 claim and absolute `first` wording with regenerated values and cautious novelty positioning.
- `paper/source_package_v1_0/main.tex`, `references.bib`, `figure_captions_and_alt_text.txt`, `highlights.txt`
  - Synchronized with active manuscript/source files.
- `paper/article_v1_0_blue_changes.tex` and `paper/article_v1_0_red_blue_changes.tex`
  - Regenerated with `python make_revision_articles.py`.

Commands run:
- `py -3.13 run_revision_ablations.py --image-dir paper\images --architecture-epochs 20`
  - Timed out after about 5 minutes.
- `py -3.13 run_revision_ablations.py --image-dir paper\images --architecture-epochs 20 --only architecture`
  - Timed out after about 5 minutes.
- `py -3.13 run_revision_ablations.py --image-dir paper\images --architecture-epochs 20 --only architecture`
  - Timed out after about 15 minutes.
- `py -3.13 run_revision_ablations.py --image-dir paper\images --architecture-epochs 20 --only architecture`
  - Failed after cache optimization with `ValueError: Peak does not occur in image`.
  - Cause identified: cache key used `id(image)`, but `build_mapping_tables` passes temporary binary arrays whose ids can be reused.
- `py -3.13 run_revision_ablations.py --image-dir paper\images --architecture-epochs 20 --only architecture`
  - Completed successfully in about 349 seconds.
  - Wrote `revision_results/cnn_architecture_ablation.json` and `.csv`.
  - New rows include shallow/deep/window/width variants and inference timing.
  - Key result: wide 2x9 w32 has lowest MAE (0.0303), current 2x9 w24 has MAE 0.0339 and lower complexity than w32; patch 11 has best held-out DRD but worse MAE and slower inference.
- `py -3.13 evaluate_bossbase_net.py --dataset "C:\Users\User\Downloads\Compressed\cpu_stego_experiments\datasets\BOSSbase" --ranker-model ml_ranker_results\candidate_ranker.pt --sample-size 100 --threshold 128 --payload-fraction 0.75 --seed 20260608 --output bossbase_net_results.json`
  - Completed successfully in about 562 seconds.
  - Summary: 669.58 gross, 377.12 auxiliary, 292.46 net, median net 159.0, 75 net-positive images, mean DRD 0.3003, exact recovery/message for all 100 rows.
- `py -3.13 evaluate_bossbase_net.py ... --no-net-optimize --output bossbase_net_results_no_net_optimize.json`
  - Timed out after about 38 minutes, but wrote valid JSON/CSV with 100 rows.
  - Parsed summary: 832.51 gross, 731.44 auxiliary, 101.07 net, median net -31.5, 47 net-positive images, mean DRD 0.4287, exact recovery/message for all 100 rows.
- `py -3.13 analyze_sota_multiload.py sota_multiload\sota_64.json sota_multiload\sota_128.json sota_multiload\sota_256.json sota_multiload\sota_512.json --dataset "C:\Users\User\Downloads\Compressed\cpu_stego_experiments\datasets\BOSSbase" --output-dir sota_multiload`
  - Completed successfully in about 11 seconds.
  - Wrote `sota_multiload/analysis.json` with rank-biserial effect sizes and fixed-cohort bootstrap output.
- `py -3.13 run_uniform_agent.py --image-dir paper\images --output-dir ml_uniform_results --probability-threshold 0.7`
  - Completed successfully in about 25 seconds.
  - Wrote `ml_uniform_results/report.json`; metrics: AUC 0.9952, precision 0.8972, recall 0.9702, position accuracy 0.9707, 14798 samples.
- `py -3.13 evaluate_ml_pipeline.py --image-dir paper\images --uniform-model ml_uniform_results\uniform_agent.joblib --ranker-model ml_ranker_results\candidate_ranker.pt --output ml_pipeline_results.json --base-payload-fraction 0.75 --uniform-payload-fraction 0.75 --seed 20260607`
  - Completed successfully in about 36 seconds.
  - Wrote `ml_pipeline_results.json`; mean uniform payload 181.375 bits, mean uniform auxiliary 2691.0 bits, mean combined net -264.625 bits, exact recovery/message for all rows.
- `Select-String` corruption check after key replacement
  - Confirmed no remaining accidental `20Yin...`, `3Xuan...`, or digit/key hybrid strings outside intended citation keys.
- Bibliography checker
  - Confirmed 28 cited keys, 28 bibliography entries, zero missing cited keys, zero uncited entries.
- `python make_revision_articles.py`
  - Completed successfully and rewrote blue-only and red/blue article sources.

Current blocker:
- No current scientific blocker. Main manuscript, response letter, and cover letters compile; regenerated annotated PDFs are being stabilized after the A0/A1/A2 terminology change.

Next actions:
- Recompile `paper/main.tex`, `paper/response_to_reviewers.tex`, and regenerated annotated article versions after replacing V0/V1/V2 ablation labels with A0/A1/A2.
- Run reproducibility/test checks, synchronize the source package and archives, then prepare the final report.

Continuation update 2026-08-06:
- `paper/main.pdf` compiled successfully with resolved citations and references before the A0/A1/A2 terminology cleanup.
- `paper/response_to_reviewers.pdf`, `paper/revision_cover_letter.pdf`, and `paper/cover_letter.pdf` compiled successfully and were stabilized after second LaTeX passes.
- The original annotated PDFs `article_v1_0_blue_changes.pdf` and `article_v1_0_red_blue_changes.pdf` were locked by an external process; updated PDFs were compiled under `article_v1_0_blue_changes_updated.pdf` and `article_v1_0_red_blue_changes_updated.pdf`.
- Replaced ablation labels `V0/V1/V2` with `A0/A1/A2` in the active manuscript, response letter, and actions report to avoid any ambiguity with manuscript versioning.
- Recompiled `paper/main.tex` and `paper/response_to_reviewers.tex` after the A0/A1/A2 change.
- Regenerated `paper/article_v1_0_blue_changes.tex` and `paper/article_v1_0_red_blue_changes.tex` from the current manuscript sources.

Checkpoint update 2026-08-06:
- Harmonized the experiment labels in `run_revision_ablations.py`,
  `revision_results/activation_ablation.csv`,
  `revision_results/activation_ablation.json`, and
  `revision_results/revision_ablation_summary.json` from `V0/V1/V2` to
  `A0/A1/A2`.
- This was a label-only traceability change; numerical measurements were not
  altered.
- Rewrote the same four files as UTF-8 without BOM after the mechanical
  label replacement so JSON readers remain reproducible with standard
  `encoding='utf-8'`.
- Corrected the ranker-ablation table in `paper/main.tex` so its rounded mean
  net payload and DRD values match `revision_results/ranker_ablation.json`.
- Regenerated `paper/article_v1_0_blue_changes.tex` and
  `paper/article_v1_0_red_blue_changes.tex` after the ranker-table correction.
- Recompiled `paper/main.pdf`, `paper/article_v1_0_blue_changes_updated.pdf`,
  and `paper/article_v1_0_red_blue_changes.pdf`. The original blue PDF remains
  locked by another Windows process, so the current blue PDF is the `_updated`
  artifact.
- Rewrote `paper/actions_completion_report.md` with the current verification
  state and the locked-PDF caveat.
- Synchronized `paper/article_v1_0_clean.pdf` from `paper/main.pdf`, refreshed
  `paper/source_package_v1_0`, regenerated `paper/rdhi_page1.txt/png`, and
  moved the pre-actions manuscript backup into `paper/validation_logs`.
- Regenerated `paper/latex_source_v1_0_revised.zip` from
  `paper/source_package_v1_0` and rebuilt root `paper.zip` from the current
  deliverables, excluding the locked stale blue PDF and including
  `article_v1_0_blue_changes_updated.pdf`.
- Final verification pass:
  - Python compilation succeeded for the edited scripts and main codec modules.
  - `pytest -q -p no:cacheprovider` completed with 54 passed tests and one
    non-fatal scikit-learn convergence warning.
  - LaTeX log scan found no unresolved citations/references or rerun requests
    in the clean article, response letter, cover letters, blue-updated article,
    or red/blue article.
  - Bibliography/label audit found 28 cited keys, 28 BibTeX entries, 62
    labels, 22 floats, no missing citations, no uncited BibTeX entries, no
    duplicate labels, no missing refs, and no uncited float labels.
- Git status was checked with a command-local `safe.directory` exception. No
  commit or push was performed. Generated pytest caches were removed where
  accessible.

Recompile update 2026-08-06:
- Recompiled all `paper/*.tex` sources with PDFLaTeX/BibTeX as applicable:
  `main.tex`, `response_to_reviewers.tex`, `revision_cover_letter.tex`,
  `cover_letter.tex`, `graphical_abstract.tex`,
  `article_v1_0_blue_changes.tex`, and
  `article_v1_0_red_blue_changes.tex`.
- `main.pdf` was rebuilt after `bibtex main` and two PDFLaTeX passes.
- `article_v1_0_blue_changes.tex` was rebuilt with the job name
  `article_v1_0_blue_changes_updated` because the original blue PDF filename
  remains locked by another Windows process.
- `article_v1_0_red_blue_changes.pdf` was rebuilt after BibTeX and final
  reference-resolution passes.
- The LaTeX log scan found no fatal errors, unresolved citations, unresolved
  references, duplicate-label warnings, or remaining rerun requests in the
  rebuilt manuscript, response letter, cover letters, graphical abstract, and
  annotated article logs.
- Synchronized `article_v1_0_clean.pdf` from `main.pdf`, refreshed
  `source_package_v1_0`, regenerated `rdhi_page1.txt/png`, rebuilt
  `latex_source_v1_0_revised.zip`, and rebuilt root `paper.zip` from the
  current deliverables.

Declaration restoration update 2026-08-06:
- Reintegrated the manuscript `Declarations` section in `paper/main.tex` and
  regenerated `paper/article_v1_0_blue_changes.tex` and
  `paper/article_v1_0_red_blue_changes.tex`.
- Restored the declaration subsections for ethics approval, competing
  interests, funding, data availability, and code availability.
- Kept the AI-use declaration removed, in line with the user's clarification.
- Did not restore the first-version sentence referring to a companion
  publication, because the article must remain self-contained as version 1.0.
- Recompiled `paper/main.pdf`, `paper/article_v1_0_blue_changes_updated.pdf`,
  and `paper/article_v1_0_red_blue_changes_updated.pdf`; the original
  red/blue PDF filename was locked by Windows, so the current red/blue PDF is
  the `_updated` artifact.
- Refreshed `paper/article_v1_0_clean.pdf`, `paper/source_package_v1_0`,
  `paper/latex_source_v1_0_revised.zip`, root `paper.zip`, and the page-1
  text/PNG preview artifacts after the declaration restoration.

Final precision-pass update 2026-08-06:
- Applied the user's reading-report corrections surgically, without adding new
  experiments or changing the title.
- Harmonized the central terminology from net-payload-driven activation to
  serialization-aware net-capacity-driven activation:
  `C_{\mathrm{gross}}` is gross mapping capacity, `q=|M|` is the requested
  application payload, `P_{\mathrm{net}}=q-|A_{\mathrm{wire}}|_2` is delivered
  net payload, and `C_{\mathrm{net}}^{\mathrm{map}}` is the map-level selection
  criterion.
- Corrected Table 1 by replacing the invalid PSNR cell with `--` and adding a
  note that DRD is the main reported distortion metric while per-image PSNR and
  changed-pixel outputs are kept in the reproducibility artifacts.
- Updated the cover letters to point reproducibility materials to the public
  GitHub repository cited in the Code Availability statement rather than to an
  ambiguous accompanying source package.
- Strengthened the Reviewer 5 security-threshold answer: no tested payload is
  claimed as a covert-security threshold; 256/512-bit settings are presented as
  capacity-oriented reversible annotation points, and 64/128-bit settings are
  reported only as measured lower-detectability operating points.
- Replaced `optional CNN ranking` with secondary/deployed CNN cost ranking in
  the response letter and marked Table 5's w16 architecture as deployed.
- Aligned response wording with manuscript evidence for low-payload limits,
  high-payload operating scope, changed-pixel artifacts, and the diagnostic-only
  role of the adaptive-threshold parameter `alpha`.
- Added the bootstrap statement for median DRD confidence bands, renamed Table
  10 `Total` to `End-to-end total`, and explained the minor orchestration
  overhead included in that timer.
- Replaced the unsupported `median-positive at 128 bits` wording with the
  reported mean-net behavior on the payload-specific and fixed cohorts.
- Renamed Proposition 3.6 to `Non-interference of Distinct Blocks` and softened
  the statistical wording around distributional assumptions.
- Recompiled `paper/main.tex`, `paper/response_to_reviewers.tex`,
  `paper/revision_cover_letter.tex`, `paper/cover_letter.tex`,
  `paper/article_v1_0_blue_changes.tex`, and
  `paper/article_v1_0_red_blue_changes.tex`.
- The final LaTeX log scan found no fatal errors, unresolved citations,
  unresolved references, duplicate-label warnings, or remaining rerun requests
  in the clean manuscript, response letter, cover letters, or annotated article
  logs.
- Refreshed `paper/article_v1_0_clean.pdf`, `paper/source_package_v1_0`,
  `paper/latex_source_v1_0_revised.zip`, root `paper.zip`, and the page-1
  text/PNG preview artifacts after the final precision pass.
