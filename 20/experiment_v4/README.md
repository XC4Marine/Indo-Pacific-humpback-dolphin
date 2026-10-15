# Experiment 20 v4: truth-isolated BCSC

Run extraction from this directory:

```powershell
python run_bcsc_v4.py --phase all
python evaluate_bcsc_v4.py
```

`run_bcsc_v4.py` only accepts `clicktrain_raw.npy`; it statically rejects extraction code that names a truth asset. The evaluator is a separate program and reads truth only after extraction completes.

Outputs are resumable under `outputs/`; evaluation tables and reproducible representative-case figures are written under `evaluation/`.
