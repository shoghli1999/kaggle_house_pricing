"""Evaluation: repeated K-fold cross-validation on all 1,460 labelled rows.

Every configuration is scored on exactly the same folds, so differences can be
compared fold by fold (paired) instead of only by their means.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.maximum(np.asarray(y_pred, float), 1.0)
    log_err = np.log1p(y_pred) - np.log1p(y_true)
    resid = y_true - y_pred
    return {
        "rmsle": float(np.sqrt(np.mean(log_err ** 2))),          # Kaggle's metric
        "mape_pct": float(np.mean(np.abs(resid) / y_true) * 100),
        "mae_usd": float(np.mean(np.abs(resid))),
        "r2": float(1 - np.sum(resid ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def cross_validate(estimator, X, y, n_splits=5, n_repeats=3, seed=42, groups=None):
    """Return (per-fold metrics, out-of-fold predictions of the first repeat)."""
    if groups is not None:
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splits = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X)
    rows, oof = [], pd.Series(np.nan, index=X.index)
    for k, (tr, va) in enumerate(splits):
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        rows.append({"fold": k, **metrics(y.iloc[va], pred)})
        if k < n_splits:
            oof.iloc[va] = pred
    return pd.DataFrame(rows), oof


def summarise(name, folds: pd.DataFrame) -> dict:
    return {"configuration": name,
            "rmsle_mean": folds.rmsle.mean(), "rmsle_std": folds.rmsle.std(),
            "mape_mean_pct": folds.mape_pct.mean(), "mae_mean_usd": folds.mae_usd.mean(),
            "r2_mean": folds.r2.mean(), "r2_worst_fold": folds.r2.min()}


def paired_wins(folds_a: pd.DataFrame, folds_b: pd.DataFrame, metric="rmsle") -> str:
    """How many folds configuration A beats configuration B on (lower is better)."""
    wins = int((folds_a[metric].to_numpy() < folds_b[metric].to_numpy()).sum())
    return f"{wins}/{len(folds_a)}"


# --------------------------------------------------------------- prediction intervals
def conformal_quantile(abs_residuals, alpha=0.10) -> float:
    """Finite-sample split-conformal quantile of absolute residuals."""
    r = np.sort(np.asarray(abs_residuals))
    n = len(r)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(r[k - 1])


def conformal_evaluation(estimator, X, y, alpha=0.10, n_splits=5, seed=42):
    """Cross-conformal check of 1-alpha prediction intervals in log space.

    For each outer fold: out-of-fold residuals from an inner 5-fold CV on the
    training part give the interval half-width q; the model refitted on the whole
    training part predicts the held-out fold, and we count how often the true
    price falls inside [pred / e^q, pred * e^q].
    """
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for k, (tr, va) in enumerate(outer.split(X)):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        inner_res = np.empty(len(tr))
        for itr, iva in KFold(n_splits=5, shuffle=True, random_state=seed + 1).split(Xtr):
            m = clone(estimator).fit(Xtr.iloc[itr], ytr.iloc[itr])
            inner_res[iva] = np.log1p(ytr.iloc[iva]) - np.log1p(np.maximum(m.predict(Xtr.iloc[iva]), 1))
        q = conformal_quantile(np.abs(inner_res), alpha)
        model = clone(estimator).fit(Xtr, ytr)
        pred = np.maximum(model.predict(X.iloc[va]), 1)
        lo, hi = np.expm1(np.log1p(pred) - q), np.expm1(np.log1p(pred) + q)
        yv = y.iloc[va].to_numpy()
        rows.append({"fold": k, "q_log": q, "coverage": float(np.mean((yv >= lo) & (yv <= hi))),
                     "median_width_usd": float(np.median(hi - lo)),
                     "median_width_pct_of_price": float(np.median((hi - lo) / pred) * 100)})
    return pd.DataFrame(rows)
