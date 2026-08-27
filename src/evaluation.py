"""Classification metrics and threshold-selection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


def classification_summary(y_true, probability, threshold=0.5) -> dict:
    probability = np.asarray(probability)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def ks_statistic(y_true, probability) -> float:
    fpr, tpr, _ = roc_curve(y_true, probability)
    return float(np.max(tpr - fpr))


def threshold_table(y_true, probability) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    return pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
