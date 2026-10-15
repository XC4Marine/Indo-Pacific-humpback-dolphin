# v7 Gaussian soft-label auxiliary architecture figure

Artifact: wide research-report / presentation architecture draft (16:9 PNG).

Core claim: the waveform-only classifier is trained with a BCSC-derived 64-bin delay-energy distribution auxiliary target, but BCSC target generation and the auxiliary head are absent at inference.

Source method / topology: `experiment_v7_gaussian_softlabel.py`; waveform `x [1×5761]` → Conv1d patch embedding (`kernel=32`, `stride=16`, `359×64`) → positional encoding + CLS (`360×64`) → one 64-dimensional Transformer (`4` heads, FFN `128`) → `z [64]` → classifier `64→64→1`. Training-only branch: `z` → LayerNorm → `64→128` → GELU → Dropout `0.2` → `128→64` → Softmax. BCSC cache `P/A/U [3×1133]` → `w=P·A·(1−U)` → 64-bin Gaussian target `H`; it supervises `H-hat` through the specified `m·q`-weighted path loss.

Reader scan path: waveform inference lane runs left-to-right at the top. The training-only auxiliary head hangs below the CLS feature. The BCSC target lane runs left-to-right below it, with dashed supervision links to the loss. The inference-only deployment strip restates waveform-only deployment.

Typed connections: solid blue arrows are inference/data flow; dashed teal or amber arrows are training-only supervision. The gray deployment region explicitly removes the auxiliary and BCSC paths.

Exact visible label inventory: Waveform x; Conv1d Patch Embedding; Sinusoidal Positional Encoding + CLS Token; Transformer Encoder (1 layer); CLS feature z; Classifier MLP (uses z only); Click probability; Auxiliary Distribution Head (Training only); Predicted distribution H-hat; Cached P/A/U structure map; Combine `w=P×A×(1−U)`; Delay range `0.0347–2.0000 ms`; Continuous 64-bin mapping; Gaussian smoothing `σ=1 bin, radius=3σ`; Normalized target H; Confidence q; Valid mask m; Weighted Path Loss; Deployment / Inference.

Visual encoding: accessible deep blue (`#0072B2`) for waveform/inference, amber (`#E69F00`) for BCSC target generation, teal (`#009E73`) for the auxiliary head and neutral gray for deployment. Flat rounded modules, high-contrast horizontal type, white background, thin unambiguous arrows.

Reference-layout provenance: the user-provided image informed only the transferable principles of a wide horizontal reading path, semantically grouped lanes, and restrained cool/warm branch colors. Its layout, icons, labels, and decorative elements were not copied or uploaded to the image model.

No-fabrication status: no experimental result, metric, or unsupported module is shown.
