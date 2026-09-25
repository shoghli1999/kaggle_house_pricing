"""Feature engineering for the Ames housing data.

Everything in `add_features` is computed row by row from the row itself, so it
cannot leak information between training and validation rows. Anything that
learns from data (imputation values, which columns are skewed, scaling) lives in
the scikit-learn transformers further down and is fitted on training folds only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

QUALITY = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
ORDINAL_MAPS = {
    **{c: QUALITY for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "PavedDrive": {"Y": 2, "P": 1, "N": 0},
}
# Numeric codes that are really categories.
AS_CATEGORY = ["MSSubClass", "MoSold"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with ordinal quality scores and a few domain features.

    A missing quality value in this dataset almost always means "the house has
    no such thing" (no basement, no garage, no pool), so it is scored 0.
    """
    X = X.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in X:
            X[col] = X[col].map(mapping).fillna(0).astype(float)
    for col in AS_CATEGORY:
        if col in X:
            X[col] = X[col].astype(str)

    zero = lambda c: X[c].fillna(0) if c in X else 0  # noqa: E731
    X["TotalSF"] = zero("TotalBsmtSF") + zero("1stFlrSF") + zero("2ndFlrSF")
    X["TotalBath"] = (zero("FullBath") + 0.5 * zero("HalfBath")
                      + zero("BsmtFullBath") + 0.5 * zero("BsmtHalfBath"))
    X["TotalPorchSF"] = (zero("OpenPorchSF") + zero("EnclosedPorch") + zero("3SsnPorch")
                         + zero("ScreenPorch") + zero("WoodDeckSF"))
    X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
    X["YearsSinceRemodel"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
    X["QualTimesLivArea"] = X["OverallQual"] * X["GrLivArea"]
    X["QualTimesTotalSF"] = X["OverallQual"] * X["TotalSF"]
    X["Has2ndFloor"] = (zero("2ndFlrSF") > 0).astype(float)
    X["HasGarage"] = (zero("GarageArea") > 0).astype(float)
    X["HasBasement"] = (zero("TotalBsmtSF") > 0).astype(float)
    X["HasFireplace"] = (zero("Fireplaces") > 0).astype(float)
    return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Pipeline step wrapping `add_features` (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X))


class SkewedLog1p(BaseEstimator, TransformerMixin):
    """log1p-transform the numeric columns whose skewness on the *training* data
    exceeds a threshold. Which columns are skewed is learned in `fit`, so the
    choice never sees validation rows."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        mins = np.nanmin(X, axis=0)
        skew = pd.DataFrame(X).skew().to_numpy()
        self.mask_ = (np.abs(skew) > self.threshold) & (mins >= 0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.mask_] = np.log1p(X[:, self.mask_])
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features
