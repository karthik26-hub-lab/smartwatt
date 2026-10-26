# SmartWatt

Adaptive machine learning for electricity consumption anomaly detection.
Compares four detectors on hourly consumption data:

| Detector | Idea |
|---|---|
| Fixed threshold | mean ± 3·std of training data (baseline) |
| Adaptive threshold | robust z-score vs. the same hour/day-type over the last 14 days |
| Isolation Forest | unsupervised ML on time-aware features |
| Hybrid adaptive | adaptive z-score very high, OR Isolation Forest flags it and z-score is moderately high |

Metrics: precision, recall, F1, false alarm rate (FP / all normal readings),
computed on a chronological test split (first 60% train, last 40% test).
Features and baselines use past data only, so there is no future leakage.

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run_experiment.py          # synthetic data, 10 seeds
.venv\Scripts\streamlit run dashboard.py        # optional dashboard
```

Outputs go to `results/` (metrics CSVs, `detections.png`, `metrics.png`).

Real data: `python run_experiment.py --csv file.csv --time-col Datetime --value-col kWh [--label-col label]`.
Without a label column, metrics are skipped (the abstract's "wherever reliable
labels are available") and only detections are plotted.

## Results (synthetic, mean of 10 seeds, test split)

| Detector | Precision | Recall | F1 | False alarm rate |
|---|---|---|---|---|
| Fixed threshold | 1.000 | 0.117 | 0.201 | 0.000 |
| Adaptive threshold | 0.832 | 0.997 | 0.905 | 0.007 |
| Isolation Forest | 0.687 | 0.475 | 0.549 | 0.009 |
| Hybrid adaptive | 0.895 | 0.958 | 0.924 | 0.004 |

Reading: the fixed threshold is precise but misses nearly everything that is
anomalous *for its time of day*; Isolation Forest alone lacks time context;
time-aware thresholds do most of the work; combining them gives the best F1
with the lowest false alarm rate among the adaptive methods, at a small recall cost.

## Caveats (say these in the report)

- The synthetic data has a clear daily/weekly routine, which favours time-aware
  methods. Confirm the finding on a real dataset (e.g. UCI Household Electric
  Power Consumption) with labels you injected or defined, and report that honestly.
- The hybrid thresholds (k_hi=5, k_lo=2.5) were set by hand, not tuned on a
  validation split. Tune them on a validation period before claiming gains.
- Adaptive threshold alone has higher recall than the hybrid; the hybrid's win
  is precision/F1. The ablation row is there to show this.
