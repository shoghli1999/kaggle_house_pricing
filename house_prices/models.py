"""Model configurations. Every one is a single scikit-learn estimator, so every
learned step (imputation, skew choice, scaling, encoding, regularisation
strength) is refitted inside each training fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureAdder, SkewedLog1p

CLIP_SIGMA = 5.0


def _clip(a):
    return np.clip(a, -CLIP_SIGMA, CLIP_SIGMA)


def _log_target(regressor):
    """Model log(1 + price). The Kaggle metric is RMSE on log price, and MAPE is a
    relative error, so the log scale matches both."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


class TrainingFilter(BaseEstimator, RegressorMixin):
    """Drop training rows above `max_living_area` square feet before fitting.

    The dataset's author recommends removing houses over 4,000 sq ft of living
    area: there are few of them, and some are unusual partial sales (De Cock,
    Journal of Statistics Education 19(3), 2011). Only *training* rows are
    dropped. Every validation row is still scored.
    """

    def __init__(self, estimator, max_living_area: float = 4000):
        self.estimator = estimator
        self.max_living_area = max_living_area

    def fit(self, X, y):
        keep = (X["GrLivArea"] <= self.max_living_area).to_numpy()
        self.estimator_ = clone(self.estimator).fit(X[keep], np.asarray(y)[keep])
        self.n_dropped_ = int((~keep).sum())
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


# ------------------------------------------------------------------ preprocessing
_numeric = make_column_selector(dtype_include=np.number)
_categorical = make_column_selector(dtype_exclude=np.number)


def linear_preprocessor(skew_log: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if skew_log:
        num_steps.append(("skewlog", SkewedLog1p(threshold=0.75)))
    num_steps += [("scale", StandardScaler()),
                  ("clip", FunctionTransformer(_clip, feature_names_out="one-to-one"))]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), _numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="None")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                   sparse_output=False))]), _categorical),
    ], verbose_feature_names_out=False)


def tree_preprocessor(impute_numeric: bool = False) -> ColumnTransformer:
    """Trees need no scaling. HistGradientBoosting handles missing values itself;
    classic GradientBoosting does not, so it gets median imputation."""
    num = SimpleImputer(strategy="median") if impute_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, _numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="None")),
                          ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                     unknown_value=-1))]), _categorical),
    ], verbose_feature_names_out=False)


# ------------------------------------------------------------------------ models
ALPHAS = np.logspace(-2, 3, 30)


def baseline_ridge():
    """Configuration C from the original comparison (no feature engineering)."""
    return _log_target(Pipeline([("pre", linear_preprocessor(skew_log=False)),
                                 ("model", RidgeCV(alphas=ALPHAS))]))


def ridge(features: bool = True):
    steps = ([("features", FeatureAdder())] if features else []) + [
        ("pre", linear_preprocessor()), ("model", RidgeCV(alphas=ALPHAS))]
    return _log_target(Pipeline(steps))


def lasso():
    return _log_target(Pipeline([
        ("features", FeatureAdder()), ("pre", linear_preprocessor()),
        ("model", LassoCV(alphas=np.logspace(-4.5, -1.5, 30), max_iter=50000, cv=5,
                          random_state=0))]))


def elastic_net():
    return _log_target(Pipeline([
        ("features", FeatureAdder()), ("pre", linear_preprocessor()),
        ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4.5, -1, 25),
                               max_iter=50000, cv=5, random_state=0))]))


def hist_gbm(features: bool = True, seed: int = 0):
    steps = ([("features", FeatureAdder())] if features else []) + [
        ("pre", tree_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            max_iter=1500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=seed))]
    return _log_target(Pipeline(steps))


def gbm(seed: int = 0):
    """Classic gradient boosting with Huber loss, which cares less about the few very
    unusual sales left in the training data."""
    return _log_target(Pipeline([
        ("features", FeatureAdder()), ("pre", tree_preprocessor(impute_numeric=True)),
        ("model", GradientBoostingRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=4, min_samples_leaf=10,
            max_features="sqrt", subsample=0.8, loss="huber", random_state=seed))]))


def blend(seed: int = 0):
    """Equal-weight average, in log space, of a linear model and a tree model.
    They fail in different places (the linear model on extrapolation, the trees
    on smooth size effects), so their errors are only partly correlated."""
    linear = Pipeline([("features", FeatureAdder()), ("pre", linear_preprocessor()),
                       ("model", LassoCV(alphas=np.logspace(-4.5, -1.5, 30), max_iter=50000,
                                         cv=5, random_state=0))])
    trees = Pipeline([("features", FeatureAdder()), ("pre", tree_preprocessor(impute_numeric=True)),
                      ("model", GradientBoostingRegressor(
                          n_estimators=1200, learning_rate=0.03, max_depth=4, min_samples_leaf=10,
                          max_features="sqrt", subsample=0.8, loss="huber", random_state=seed))])
    return _log_target(VotingRegressor([("linear", linear), ("trees", trees)]))


def without_large_houses(estimator):
    return TrainingFilter(estimator, max_living_area=4000)


def blend_three(seed: int = 0):
    """Experiment: add HistGradientBoosting as a third member of the blend."""
    base = blend(seed).regressor
    hist = hist_gbm(seed=seed).regressor
    return _log_target(VotingRegressor(base.estimators + [("hist", hist)]))
