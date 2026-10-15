# Visual contract

- Artifact: paired predicted source and channel waveform examples.
- Format: diagnostic PNG, 250 dpi.
- Evidence layer: qualitative diagnostic only; no performance claim.
- Source: v5 fold-0 best checkpoints and locked experiment-22 waveform cache.
- Samples: first two positive and first two negative paired test samples by sample_index.
- Panels: 4 samples x 2 paired conditions; source uses a -5 to 5 ms axis, channel tail uses a 0 to 5 ms axis.
- Direct path: h_hat[0] is fixed to 1 and shown in each channel inset.
- Palette: Okabe-Ito blue (#0072B2) without Physics; orange (#D55E00) with Physics.
- Traceability: selected_samples.csv, predicted_x_h_examples.npz, and checkpoint metadata in render_metadata.json.
