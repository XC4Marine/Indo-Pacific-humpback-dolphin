# Visual contract

- Artifact: editable PowerPoint reconstruction of the supplied physics-informed click-classification architecture.
- Target format: one-slide PPTX, custom 1527 x 814 px canvas matching the reference aspect ratio.
- Core claim: the shared encoder separates source and causal channel-response representations; reconstruction, response physics, and classification supervision are optimized jointly.
- Reader question: how do waveform reconstruction, channel-response modeling, residual analysis, and classification connect?
- Evidence layer: mechanism / architecture.
- Source data: user-provided reference image only; no external assets or invented measurements.
- Source method content: every visible module, tensor label, loss, and connection is reconstructed from the reference.
- Figure prototype: multi-branch left-to-right scientific computation graph with a lower training-supervision lane.
- Reader scan path: input -> shared encoder -> source/response decoders -> convolution/reconstruction/residual -> feature encoders -> classifier; dashed loss links read downward.
- Typed connections: solid = tensor/data flow; dashed = training-only supervision.
- Exact label inventory: Input waveform y; 5761 samples, 576 kHz; Shared 1D convolutional encoder E; Latent representation z; Full-length source decoder D_x; x-hat [B, 5761]; Causal response decoder D_h; h-hat [B, 1153]; h_0 = 1, 0-2 ms; Causal FFT convolution; Reconstruction y-hat; Residual r = y - y-hat; Source feature encoder; Response feature encoder; Residual feature encoder; Concatenate 96-D; MLP 96->64->1; Classification logit s; Label c; Reconstruction loss; Physics loss; BCE loss; Total loss; total-loss equation.
- Palette: accessible blue/green/orange/purple/magenta branches plus neutral black/gray supervision.
- Reference-layout principles: dominant left-to-right flow, vertically synchronized decoder branches, evidence-bearing waveform insets, restrained semantic color, bottom supervision lane.
- Output formats: PPTX plus PNG preview for QA.
- PPTX editability target: native PowerPoint shapes, live text, custom native vector paths, and native lines; no raster content on the slide.
- Traceability: element names identify semantic modules; the supplied PNG is used only as a visual reference and is not embedded.

