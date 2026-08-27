"""Classical supervised models for credit-risk style classification."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def logistic_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


def decision_tree(max_depth: int = 5, min_samples_leaf: int = 20, seed: int = 42):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
    )
