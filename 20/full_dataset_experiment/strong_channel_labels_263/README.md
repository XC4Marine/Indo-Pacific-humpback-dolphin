# PulseTrain-level strong pseudo-channel labels

This directory contains 263 strong channel labels and 2335 complete single-click model-input waveforms.

## Label definition

- Click-level path confidence is at least 90% and its Wilson 95% lower bound is at least 80%.
- Paths from different clicks match when their relative delays differ by no more than +/-0.03 ms.
- A stable path must occur in at least 80% of the modeled clicks in its PulseTrain.
- One directory represents one PulseTrain-level channel label and includes all stable non-direct paths plus the direct path.

## Files

- `index.csv`: one row per PulseTrain-level label.
- `all_strong_channels.csv`: direct and stable multipath summaries for all labels.
- `all_path_members.csv`: click-level paths supporting each stable common path.
- Each label directory contains `channel_label.csv`, `path_members.csv`, `input_waveforms.npz`, `metadata.json`, and `preview.png`.

`input_waveforms.npz` stores the exact BCSC input convention: mean removal, fifth-order 20-190 kHz zero-phase Butterworth filtering, Tukey window with alpha 0.04, and per-click peak normalization. These arrays are not the unprocessed WAV samples.
