"""Metrics and business-cost helpers (SRS sections 5.2, 5.3)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def metrics_at(y_true, proba, threshold=0.5):
    """All SRS metrics for one set of probabilities at one threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def total_cost(y_true, proba, amount, threshold, c_fp=config.C_FP):
    """Total Cost(t) = sum of Amount over missed frauds + C_FP * number of false alarms."""
    y_true = np.asarray(y_true)
    amount = np.asarray(amount)
    pred = np.asarray(proba) >= threshold
    fn_cost = amount[(y_true == 1) & ~pred].sum()
    n_fp = int(((y_true == 0) & pred).sum())
    return fn_cost + c_fp * n_fp


def cost_curve(y_true, proba, amount, thresholds=None, c_fp=config.C_FP):
    """Cost, precision and recall for every threshold in the sweep (0.01-0.99)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        rows.append({
            "threshold": t,
            "cost": total_cost(y_true, proba, amount, t, c_fp),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def no_model_cost(y_true, amount):
    """Cost if every transaction is approved: all fraud is missed."""
    return float(np.asarray(amount)[np.asarray(y_true) == 1].sum())


def meets_goal(precision, recall):
    return recall >= config.TARGET_RECALL and precision >= config.TARGET_PRECISION
