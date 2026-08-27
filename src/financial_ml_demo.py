"""Supervised, ensemble, and unsupervised ML on synthetic financial data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 73


def make_credit_data(n: int = 5000, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "income": rng.lognormal(np.log(7000), 0.55, n),
        "debt_ratio": np.clip(rng.beta(2.2, 4.8, n), 0, 1),
        "utilization": np.clip(rng.beta(2.0, 3.5, n), 0, 1),
        "late_payments": rng.poisson(0.8, n),
        "tenure_years": np.clip(rng.gamma(2.5, 2.0, n), 0, 30),
        "requested_amount": rng.lognormal(np.log(35000), 0.60, n),
    })
    logit = (-2.4 + 3.1 * X["debt_ratio"] + 2.2 * X["utilization"]
             + 0.45 * X["late_payments"] - 0.00005 * X["income"]
             - 0.04 * X["tenure_years"] + rng.normal(0, 0.5, n))
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="default")
    return X, y


def supervised_demo(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                 random_state=SEED, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        print(f"\n{name}: ROC-AUC={roc_auc_score(y_test, prob):.3f}")
        print(classification_report(y_test, pred, digits=3))


def unsupervised_demo(X: pd.DataFrame) -> None:
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=SEED, n_init=20).fit(Z)
    print(f"KMeans silhouette: {silhouette_score(Z, km.labels_):.3f}")

    db = DBSCAN(eps=1.2, min_samples=12).fit(Z)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"DBSCAN clusters={n_clusters}, noise={(db.labels_ == -1).mean():.1%}")

    iso = IsolationForest(contamination=0.02, random_state=SEED).fit(Z)
    anomaly = iso.predict(Z) == -1
    print(f"IsolationForest anomalies={anomaly.mean():.1%}")


def main() -> None:
    X, y = make_credit_data()
    supervised_demo(X, y)
    unsupervised_demo(X)


if __name__ == "__main__":
    main()
