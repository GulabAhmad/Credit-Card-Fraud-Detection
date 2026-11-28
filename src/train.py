"""Model + imbalance-technique pipelines and cross-validated comparison (SRS sections 5.1, 6, 11.1)."""
import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from . import config
from .preprocess import make_preprocessor

TECHNIQUES = ["none", "class_weight", "undersample", "smote"]

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def make_model(name, technique, pos_weight=1.0):
    """Build one model. `pos_weight` = legit/fraud ratio, used only for class weights."""
    weighted = technique == "class_weight"
    if name == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if weighted else None,
            random_state=config.SEED,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=-1,
            random_state=config.SEED,
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight if weighted else 1.0,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=config.SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def make_sampler(technique):
    if technique == "smote":
        return SMOTE(random_state=config.SEED)
    if technique == "undersample":
        return RandomUnderSampler(random_state=config.SEED)
    return "passthrough"


def build_pipeline(name, technique, pos_weight=1.0):
    """scaler -> (resampler) -> model. The resampler only runs during fit,
    so validation folds and the test set keep their real imbalance."""
    return Pipeline([
        ("scale", make_preprocessor()),
        ("resample", make_sampler(technique)),
        ("model", make_model(name, technique, pos_weight)),
    ])


def cv_splitter():
    return StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)


def cv_compare(X_train, y_train, models=("lr", "xgb"), techniques=TECHNIQUES,
               results_path=None):
    """5-fold stratified CV for every model x technique. Uses training data only.

    Each result row is appended to `results_path` as soon as it is ready.
    """
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    rows = []
    for name in models:
        for tech in techniques:
            start = time.time()
            pipe = build_pipeline(name, tech, pos_weight)
            res = cross_validate(
                pipe, X_train, y_train, cv=cv_splitter(), scoring=SCORING,
                return_train_score=True, n_jobs=1,
            )
            row = {"model": name, "technique": tech}
            for key in SCORING:
                row[f"{key}_mean"] = res[f"test_{key}"].mean()
                row[f"{key}_std"] = res[f"test_{key}"].std()
            row["train_pr_auc"] = res["train_pr_auc"].mean()
            row["seconds"] = round(time.time() - start, 1)
            rows.append(row)
            print(f"{name:>4} | {tech:<12} | PR-AUC {row['pr_auc_mean']:.4f} "
                  f"± {row['pr_auc_std']:.4f} | recall {row['recall_mean']:.3f} "
                  f"| precision {row['precision_mean']:.3f} | {row['seconds']}s")
            if results_path is not None:
                pd.DataFrame(rows).to_csv(results_path, index=False)
    return pd.DataFrame(rows)
