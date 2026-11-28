"""Deep-learning models: supervised MLP and unsupervised Autoencoder (SRS section 6)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluate import metrics_at
from .train import cv_splitter


def _seed():
    tf.keras.utils.set_random_seed(config.SEED)


def build_mlp(n_features):
    """Dense 64 -> 32 -> 1, ReLU, dropout, sigmoid output."""
    _seed()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def fit_predict_mlp(X_tr, y_tr, X_va, epochs=30, batch_size=2048):
    """Scale all features on X_tr only, train with class weights and early stopping
    on a stratified validation split carved out of X_tr, return P(fraud) for X_va."""
    scaler = StandardScaler().fit(X_tr)
    Xs, Xv = scaler.transform(X_tr), scaler.transform(X_va)
    Xa, Xb, ya, yb = train_test_split(Xs, y_tr, test_size=0.1, stratify=y_tr,
                                      random_state=config.SEED)
    n0, n1 = (ya == 0).sum(), (ya == 1).sum()
    class_weight = {0: len(ya) / (2 * n0), 1: len(ya) / (2 * n1)}  # "balanced"
    model = build_mlp(Xs.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5,
                                            restore_best_weights=True)
    model.fit(Xa, ya, validation_data=(Xb, yb), epochs=epochs, batch_size=batch_size,
              class_weight=class_weight, callbacks=[stop], verbose=0)
    return model.predict(Xv, batch_size=8192, verbose=0).ravel()


def build_autoencoder(n_features):
    """30 -> 14 -> 7 -> 14 -> 30, MSE loss."""
    _seed()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(14, activation="relu"),
        tf.keras.layers.Dense(7, activation="relu"),
        tf.keras.layers.Dense(14, activation="relu"),
        tf.keras.layers.Dense(n_features, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_score_autoencoder(X_tr, y_tr, X_va, epochs=20, batch_size=512, legit_quantile=0.995):
    """Train on legit rows of X_tr only. Returns (reconstruction error for X_va,
    error threshold = `legit_quantile` of the legit training errors)."""
    legit = X_tr[np.asarray(y_tr) == 0]
    scaler = StandardScaler().fit(legit)
    L, V = scaler.transform(legit), scaler.transform(X_va)
    model = build_autoencoder(L.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                            restore_best_weights=True)
    model.fit(L, L, validation_split=0.1, epochs=epochs, batch_size=batch_size,
              callbacks=[stop], verbose=0)

    def err(A):
        return np.mean((A - model.predict(A, batch_size=8192, verbose=0)) ** 2, axis=1)

    threshold = float(np.quantile(err(L), legit_quantile))
    return err(V), threshold


def cv_deep(X_train, y_train, kind):
    """5-fold stratified CV for 'mlp' or 'autoencoder'. Returns (summary row, out-of-fold scores)."""
    X, y = np.asarray(X_train, dtype="float32"), np.asarray(y_train)
    oof = np.zeros(len(y))
    folds = []
    for k, (tr, va) in enumerate(cv_splitter().split(X, y), 1):
        if kind == "mlp":
            score = fit_predict_mlp(X[tr], y[tr], X[va])
            m = metrics_at(y[va], score, 0.5)
        else:
            score, thr = fit_score_autoencoder(X[tr], y[tr], X[va])
            m = metrics_at(y[va], score, thr)
        oof[va] = score
        folds.append(m)
        print(f"{kind} fold {k}: PR-AUC {m['pr_auc']:.4f} | recall {m['recall']:.3f} | precision {m['precision']:.3f}")
    f = pd.DataFrame(folds)
    row = {"model": kind, "technique": "class_weight" if kind == "mlp" else "anomaly (legit only)"}
    for key in ["pr_auc", "roc_auc", "recall", "precision", "f1"]:
        row[f"{key}_mean"], row[f"{key}_std"] = f[key].mean(), f[key].std(ddof=0)
    return row, oof
