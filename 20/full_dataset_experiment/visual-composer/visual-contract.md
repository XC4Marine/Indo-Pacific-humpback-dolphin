# Visual contract

- Artifact: preview images for all PulseTrain-level strong channel labels and their complete single-click model inputs.
- Target format: browsable local dataset package with one PNG preview per label.
- Core claim: every exported common channel is traceably supported by strict-high-confidence paths in at least 80% of the PulseTrain's modeled clicks.
- Reviewer question: do the stored click inputs visibly align with the paths used to form each common-channel pseudo-label?
- Evidence layer: qualitative audit view accompanying a structured pseudo-label dataset.
- Source data: `all_channels.csv`, `high_confidence_channels.csv`, `pulsetrain_summary.csv`, and the selected click WAV files.
- Statistics: click-level Wilson 95% lower bound >= 80%; cross-click matching tolerance +/-0.03 ms; support >=80% of modeled clicks.
- Figure prototype: one common-channel stem panel followed by a two-column grid of all modeled click-input waveforms.
- Panel map: common delay/amplitude label | all preprocessed click waveforms with actual matched-path markers.
- Exact labels: common channel, stable multipath, direct path, click identifier, relative delay, signed relative amplitude, and cross-click support.
- Palette: Okabe-Ito orange for direct path, blue for stable positive paths, and magenta for stable negative paths; line style and marker shape duplicate color semantics.
- Caption role: distinguish the complete saved input arrays from the path-focused preview window.
- Output formats: PNG preview plus CSV, JSON, and compressed NPZ data files.
- Traceability: `export_strong_channel_labels.py` rebuilds the 263 labels and all waveforms directly from the full experiment outputs.
