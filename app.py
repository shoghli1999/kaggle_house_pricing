#!/usr/bin/env python3
"""Streamlit dashboard for the house price project.

    streamlit run app.py

Three pages: the price estimator (final model with a 90% prediction interval),
the evaluation results, and the overfitting lesson from Part 1. Everything shown
comes from the saved CSVs in results/ or from the final model, which is trained
once on start-up if models/ is empty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from house_prices.data import to_model_input, typical_house

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# ----------------------------------------------------------------------- helpers
@st.cache_data
def load_train() -> pd.DataFrame:
    return pd.read_csv(HERE / "train.csv")


@st.cache_resource
def load_model():
    import joblib
    path, info = HERE / "models" / "final_model.joblib", HERE / "models" / "interval.json"
    if not path.exists():
        from train_final import fit_final
        model, q = fit_final(load_train())
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)
        info.write_text(json.dumps({"alpha": 0.10, "q_log": q}))
    return joblib.load(path), json.loads(info.read_text())["q_log"]


def read(name: str):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------------------- pages
def page_estimator():
    st.header("Price estimate with a 90% interval")
    st.caption("Change the main drivers; every other column is set to a typical Ames house.")
    train = load_train()
    base = typical_house(train)
    c1, c2, c3 = st.columns(3)
    with c1:
        base["Neighborhood"] = st.selectbox("Neighbourhood", sorted(train.Neighborhood.unique()),
                                            index=sorted(train.Neighborhood.unique()).index(base["Neighborhood"]))
        base["OverallQual"] = st.slider("Overall quality (1-10)", 1, 10, int(base["OverallQual"]))
        base["KitchenQual"] = st.selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa"], index=2)
    with c2:
        base["GrLivArea"] = st.number_input("Living area (sq ft)", 400, 4000, int(base["GrLivArea"]), step=50)
        base["TotalBsmtSF"] = st.number_input("Basement area (sq ft)", 0, 3000, int(base["TotalBsmtSF"]), step=50)
        base["GarageCars"] = st.slider("Garage (cars)", 0, 4, int(base["GarageCars"]))
    with c3:
        base["YearBuilt"] = st.slider("Year built", 1872, 2010, int(base["YearBuilt"]))
        base["YearRemodAdd"] = max(st.slider("Year remodelled", 1950, 2010, int(base["YearRemodAdd"])),
                                   base["YearBuilt"])
        base["FullBath"] = st.slider("Full bathrooms", 0, 4, int(base["FullBath"]))

    model, q = load_model()
    pred = float(model.predict(to_model_input(base, train))[0])
    lo, hi = np.expm1(np.log1p(pred) - q), np.expm1(np.log1p(pred) + q)
    st.metric("Estimated price", f"${pred:,.0f}")
    st.write(f"90% prediction interval: **${lo:,.0f} – ${hi:,.0f}**. In cross-validation, intervals built "
             f"this way contained the true price for about 90% of held-out houses.")
    st.caption("The model was trained on sales in Ames, Iowa, 2006-2010. It is a portfolio project, not a valuation.")


def page_results():
    st.header("How good is the model?")
    exp = read("experiments.csv")
    if exp is None:
        st.info("Run `python experiments.py` to create results/experiments.csv.")
        return
    st.subheader("Improvement steps (15-fold repeated CV, all 1,460 rows)")
    st.dataframe(exp[["step", "configuration", "rmsle_mean", "mape_mean_pct", "r2_mean", "r2_worst_fold",
                      "folds_better_than_step0"]], hide_index=True, width="stretch")
    st.bar_chart(exp.set_index("step")["rmsle_mean"], y_label="RMSLE (lower is better)")
    for name, title in [("final_oof_metrics.csv", "Final model, out-of-fold"),
                        ("error_by_price_band.csv", "Error by price band"),
                        ("intervals.csv", "90% prediction intervals per fold"),
                        ("neighbourhood_holdout.csv", "Unseen neighbourhoods (GroupKFold)"),
                        ("largest_errors.csv", "The ten largest errors")]:
        t = read(name)
        if t is not None:
            st.subheader(title)
            st.dataframe(t, hide_index=True, width="stretch")


def page_lesson():
    st.header("Part 1: fixing overfitting the wrong way, then the right way")
    t = read("model_comparison_summary.csv")
    st.write("An early version closed the train/validation gap by making the model uniformly worse. "
             "The table compares the four configurations over five random splits "
             "(`python housing_model.py --seeds 42,0,7,123,2024`).")
    if t is not None:
        st.dataframe(t, hide_index=True, width="stretch")
    else:
        st.info("Run `python housing_model.py --seeds 42,0,7,123,2024` to create the table.")


PAGES = {"Price estimator": page_estimator, "Results": page_results, "Overfitting lesson": page_lesson}

if __name__ == "__main__":
    st.set_page_config(page_title="Ames house prices", layout="wide")
    choice = st.sidebar.radio("Page", list(PAGES))
    PAGES[choice]()
