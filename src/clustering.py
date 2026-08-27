"""Clustering helpers: K-Means, DBSCAN and cluster-quality diagnostics."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def kmeans_scan(x: pd.DataFrame, k_values=range(2, 9), seed: int = 42) -> pd.DataFrame:
    z = StandardScaler().fit_transform(x)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(z)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(z, model.labels_)})
    return pd.DataFrame(rows)


def fit_dbscan(x: pd.DataFrame, eps=0.7, min_samples=8):
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(z)
    return labels, scaler
