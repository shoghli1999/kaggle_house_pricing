#!/usr/bin/env python3
"""Fit the final model on all labelled rows and write a Kaggle submission.

    python train_final.py

Writes:
  models/final_model.joblib   the fitted blend (git-ignored; rebuilt in ~10 s)
  models/interval.json        half-width of the 90% prediction interval (log scale)
  submission.csv              predictions for Kaggle's test.csv (Id, SalePrice)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from house_prices import models as M
from house_prices.evaluate import conformal_quantile

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).parent
ALPHA = 0.10


def fit_final(train: pd.DataFrame):
    X, y = train.drop(columns=["Id", "SalePrice"], errors="ignore"), train["SalePrice"]
    estimator = M.without_large_houses(M.blend())
    # Interval half-width from out-of-fold residuals on all labelled rows.
    residuals = np.empty(len(X))
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=7).split(X):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        residuals[va] = np.log1p(y.iloc[va]) - np.log1p(np.maximum(m.predict(X.iloc[va]), 1))
    q = conformal_quantile(np.abs(residuals), ALPHA)
    model = clone(estimator).fit(X, y)
    return model, q


def main() -> int:
    train = pd.read_csv(HERE / "train.csv")
    model, q = fit_final(train)
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(model, HERE / "models" / "final_model.joblib")
    (HERE / "models" / "interval.json").write_text(json.dumps({"alpha": ALPHA, "q_log": q}, indent=2))

    test = pd.read_csv(HERE / "test.csv")
    pred = model.predict(test.drop(columns=["Id"]))
    pd.DataFrame({"Id": test["Id"], "SalePrice": np.round(pred, 2)}).to_csv(HERE / "submission.csv", index=False)
    print(f"Model saved. 90% interval = prediction x/÷ {np.exp(q):.3f}. "
          f"Wrote submission.csv with {len(test)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
