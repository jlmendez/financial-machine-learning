"""Random-forest and optional XGBoost ensemble builders."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def random_forest_classifier(seed: int = 42):
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def random_forest_regressor(seed: int = 42):
    return RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=4,
        random_state=seed,
        n_jobs=-1,
    )


def xgboost_classifier(seed: int = 42):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost to use this model") from exc
    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=seed,
    )
