# Experiment 20 v5: Deep-unfolded BCSC prototype

This prototype extends v4 independent BCSC with a six-stage, differentiable unfolding network.  It estimates a shared 116-sample source and ten 4-ms event channels from a raw clicktrain.  It is not a classifier.

## Protocol

- `train_v5.py` may read synthetic source and propagation truth, but uses a deterministic 48/10/11 source-group split over the 69 original `train_id` groups.
- `infer_v5.py` reads only `clicktrain_raw.npy`.  It is statically checked against truth asset references before inference.
- `evaluate_v5.py` reads truth only after prediction artifacts exist.
- Three learned seeds are trained on clean clicktrains and evaluated on clean, −5 dB, and 5 dB inputs.  The core comparison is v4 independent BCSC, fixed six-stage unfolding, and learned six-stage unfolding.

## Run

```powershell
cd D:\Project_Github\Indo-Pacific-humpback-dolphin\20\experiment_v5
D:\Python_env\toothwhale\python.exe -m unittest discover -s tests -v
D:\Python_env\toothwhale\python.exe train_v5.py
D:\Python_env\toothwhale\python.exe infer_v5.py
D:\Python_env\toothwhale\python.exe evaluate_v5.py
```

For a fast execution-path check, pass `--epochs 1 --seeds 20260908` to training. Full inference intentionally requires all three learned checkpoints.

Outputs are written beneath `D:\Project_Github\Indo-Pacific-humpback-dolphin\20\experiment_v5\outputs`:

- `group_split.csv` and `train_config.json`: locked protocol.
- `checkpoints`: three validation-selected learned models.
- `predictions`: raw-only extraction artifacts by condition and method.
- `evaluation`: event metrics, summaries, paired deltas, and the Chinese prototype report.
