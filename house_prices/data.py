"""Small data helpers shared by the app and the tests."""
from __future__ import annotations

import pandas as pd


def feature_frame(train: pd.DataFrame) -> pd.DataFrame:
    return train.drop(columns=["Id", "SalePrice"], errors="ignore")


def typical_house(train: pd.DataFrame) -> pd.Series:
    """Median of each numeric column and the most common value of each other
    column. Missing counts as a value, so a typical house has no pool or alley."""
    X = feature_frame(train)
    return pd.Series({c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c])
                          else X[c].mode(dropna=False).iloc[0]) for c in X.columns})


def to_model_input(row: pd.Series, train: pd.DataFrame) -> pd.DataFrame:
    """One-row DataFrame with the training columns and dtypes, so numeric columns
    are not mistaken for categorical ones by the column selectors."""
    X = feature_frame(train)
    out = pd.DataFrame([row.reindex(X.columns)])
    for c in X.columns:
        out[c] = pd.to_numeric(out[c]) if pd.api.types.is_numeric_dtype(X[c]) else out[c].astype(object)
    return out
