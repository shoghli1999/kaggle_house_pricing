"""Checks that the evaluation cannot be fooled by leakage and that the model
behaves sensibly. Run with: pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_prices import models as M
from house_prices.evaluate import conformal_quantile, metrics
from house_prices.features import SkewedLog1p, add_features

DATA = Path(__file__).resolve().parents[1] / "train.csv"


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATA).drop(columns=["Id"])
    return df.drop(columns=["SalePrice"]), df["SalePrice"]


def test_feature_engineering_is_row_wise(data):
    """A row's features must not depend on which other rows are present."""
    X, _ = data
    full = add_features(X)
    part = add_features(X.iloc[100:200])
    pd.testing.assert_frame_equal(full.iloc[100:200], part)


def test_skew_choice_is_learned_from_training_rows_only():
    train = np.array([[1.0, 1.0], [2.0, 1.0], [100.0, 1.0], [3.0, 1.0]])
    tf = SkewedLog1p(threshold=0.75).fit(train)
    assert tf.mask_.tolist() == [True, False]
    other = np.array([[5.0, 50.0]])
    np.testing.assert_allclose(tf.transform(other), [[np.log1p(5.0), 50.0]])


def test_training_filter_drops_only_training_rows(data):
    X, y = data
    model = M.TrainingFilter(M.ridge(), max_living_area=4000).fit(X, y)
    assert model.n_dropped_ == int((X.GrLivArea > 4000).sum()) == 4
    assert len(model.predict(X)) == len(X)  # every row is still scored


def test_final_model_predicts_positive_finite_prices(data):
    X, y = data
    model = M.without_large_houses(M.blend()).fit(X.iloc[:600], y.iloc[:600])
    pred = model.predict(X.iloc[600:700])
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
    assert metrics(y.iloc[600:700], pred)["mape_pct"] < 20


def test_conformal_quantile_is_the_finite_sample_rank():
    r = np.arange(1, 11, dtype=float)  # n=10, alpha=0.1 -> ceil(11*0.9)=10th value
    assert conformal_quantile(r, 0.10) == 10.0
    assert conformal_quantile(r, 0.50) == 6.0  # ceil(11*0.5)=6


def test_app_input_keeps_training_dtypes(data):
    from house_prices.data import to_model_input, typical_house
    train = pd.read_csv(DATA)
    row = to_model_input(typical_house(train), train)
    X, _ = data
    assert list(row.columns) == list(X.columns)
    num = X.select_dtypes("number").columns
    assert all(pd.api.types.is_numeric_dtype(row[c]) for c in num)
    assert pd.isna(row["PoolQC"].iloc[0])  # the typical house has no pool
