# Bootstrap high-confidence path pilot visual contract

Artifact: `evaluation_bootstrap_path_pilot/figures/bootstrap_high_confidence_pilot.png`

Target venue / format: reproducible diagnostic figure, 300 dpi PNG.

Core claim: Does retaining only bootstrap-stable BCSC paths reduce spurious-path error without losing physical-path recall?

Reviewer question: Relative to each method's original sparse support, does a pre-specified high-confidence support improve Path F1 and channel-response error?

Evidence layer: mechanism pilot.

Source data: `evaluation_bootstrap_path_pilot/summary_metrics.csv`, generated from 12 deterministic clicktrains and truth-only event evaluation.

Source method / architecture content: single-click, independent clicktrain, and shared-support BCSC. Each event uses 50 conditional residual-block resamples with its extracted source held fixed.

Statistics / uncertainty: per-clicktrain medians over ten events, followed by a 2,000-resample clicktrain-clustered bootstrap 95% interval over the 12-pilot clicktrains.

Figure prototype: paired grouped bars with 95% bootstrap error bars.

Panel map:

- A: raw sparse support versus Wilson-qualified high-confidence support for Path F1.
- B: raw sparse support versus Wilson-qualified high-confidence support for 20--190 kHz channel magnitude RMSE.
- C: one three-row truth-versus-high-confidence-path figure per method, with deterministic best-recovery, typical, and largest-F1-loss events. Black circles/stems are true physical paths; method-coloured crosses/stems are retained high-confidence paths.

Exact label inventory: Path F1, Channel magnitude RMSE (dB), Raw support, High-confidence support, Single click, Independent, Shared support.

Palette: single click `#E69F00`, independent `#0072B2`, shared support `#009E73`; hatching distinguishes support type and preserves grayscale readability.

Caption role: State the conditional fixed-source scope, 50 residual resamples, 90% detection plus Wilson-lower-bound rule, and clicktrain-clustered bootstrap unit.

Output formats: PNG and CSV/JSON summary tables. Qualitative overlap figures are `figures/path_overlaps/*_truth_high_confidence_overlap.png`.

Traceability: `bootstrap_high_confidence_v4.py`, `evaluate_bootstrap_high_confidence_v4.py`, and the two output roots named above.
