# Experiment v6

`consensus_v6.py` is the truth-isolated extractor. It reads v4 independent
artifacts and raw clicktrain windows only, writes `consensus.json`,
`bootstrap_samples.npz`, `path_confidence.csv`, and resumable `manifest.json`.

```powershell
python 20/experiment_v6/consensus_v6.py --bootstrap 100
python 20/experiment_v6/evaluate_v6.py
```

The second command is intentionally separate: it performs truth evaluation,
generates per-clicktrain PNGs and `summary.png`, and writes the Chinese report.
