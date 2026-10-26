"""Data loading and synthetic data generation with labelled anomalies."""
import numpy as np
import pandas as pd


def generate_synthetic(days=180, seed=0, anomaly_rate=0.03, drift=0.4):
    """Hourly household consumption (kWh) with injected, labelled anomalies.

    Normal behaviour: morning and evening peaks, higher weekend daytime use,
    a slow seasonal drift and multiplicative noise. Four anomaly types are
    injected: spikes, sustained high use (appliance left on), unusual night
    use, and outage-like drops. Returns columns: timestamp, consumption,
    is_anomaly, anomaly_type.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    hour = idx.hour.values
    weekend = (idx.day_of_week.values >= 5).astype(float)

    base = 0.35 + 0.0 * hour
    base += 0.9 * np.exp(-0.5 * ((hour - 7.5) / 1.4) ** 2)
    base += 1.3 * np.exp(-0.5 * ((hour - 19.5) / 2.0) ** 2)
    base += weekend * 0.6 * np.exp(-0.5 * ((hour - 13) / 3.0) ** 2)
    season = 1 + 0.25 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365) * 2)
    value = base * season * rng.lognormal(0, 0.12, len(idx))
    # concept drift: usage level ramps up by `drift` over days 90-120 (e.g. new appliance)
    ramp = np.clip((np.arange(len(idx)) - 24 * 90) / (24 * 30), 0, 1)
    value *= 1 + drift * ramp

    label = np.zeros(len(idx), dtype=int)
    kind = np.array([""] * len(idx), dtype=object)
    n_events = int(len(idx) * anomaly_rate / 2.5)
    starts = rng.choice(np.arange(24 * 14, len(idx) - 12), n_events, replace=False)
    for s in starts:
        t = rng.choice(["spike", "sustained", "night", "drop"])
        if t == "spike":
            sl = slice(s, s + 1)
            value[sl] *= rng.uniform(2.5, 4.5)
        elif t == "sustained":
            sl = slice(s, s + rng.integers(3, 7))
            value[sl] += rng.uniform(1.2, 2.0)
        elif t == "night":
            s = s - s % 24 + rng.integers(0, 5)  # 00:00-04:00
            sl = slice(s, s + rng.integers(2, 4))
            value[sl] += rng.uniform(1.0, 2.0)
        else:
            sl = slice(s, s + rng.integers(2, 5))
            value[sl] = rng.uniform(0.0, 0.05)
        label[sl] = 1
        kind[sl] = t
    return pd.DataFrame({"timestamp": idx, "consumption": value,
                         "is_anomaly": label, "anomaly_type": kind})


def load_csv(path, timestamp_col, value_col, label_col=None, resample="h"):
    """Load a real dataset (e.g. UCI household power consumption).

    Readings are resampled to `resample` (default hourly) using the mean. If
    no label column is given, `is_anomaly` is set to -1 (unlabelled), so only
    visual / unsupervised analysis is meaningful.
    """
    cols = [c for c in (timestamp_col, value_col, label_col) if c]
    df = pd.read_csv(path, usecols=cols)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col).sort_index()
    out = df[value_col].astype(float).resample(resample).mean().interpolate().to_frame("consumption")
    if label_col:
        out["is_anomaly"] = df[label_col].resample(resample).max().fillna(0).astype(int)
    else:
        out["is_anomaly"] = -1
    out["anomaly_type"] = ""
    out.index.name = "timestamp"
    return out.reset_index()
