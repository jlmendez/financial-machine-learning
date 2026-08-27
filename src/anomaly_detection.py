"""Anomaly detection with Isolation Forest and robust score summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def isolation_forest_scores(x: pd.DataFrame, contamination=0.02, seed: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=300)
    label = model.fit_predict(z)
    score = -model.score_samples(z)
    out = x.copy()
    out["anomaly_score"] = score
    out["is_anomaly"] = label == -1
    return out.sort_values("anomaly_score", ascending=False)


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))
    return 0.6745 * (series - median) / (mad if mad else 1.0)
