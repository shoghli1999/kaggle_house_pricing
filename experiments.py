#!/usr/bin/env python3
"""Reproduce every number in the "Part 2" section of the README.

Protocol: 5-fold cross-validation repeated 3 times (15 folds) on all 1,460
labelled rows, with the same folds for every configuration. Metrics are computed
on every validation row, including the unusual sales that some tutorials delete
before evaluating.

    python experiments.py            # full run, about 10-15 minutes on 2 cores
    python experiments.py --quick    # 1 repeat instead of 3, for a smoke test

Writes results/experiments.csv, results/experiments_folds.csv,
results/neighbourhood_holdout.csv, results/intervals.csv,
results/error_by_price_band.csv and results/largest_errors.csv.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices import models as M
from house_prices.evaluate import (conformal_evaluation, cross_validate, metrics, paired_wins,
                                   summarise)

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).parent

# (step, configuration, what changed, estimator factory)
STEPS = [
    ("0", "Ridge, log target (configuration C from Part 1)", "starting point", M.baseline_ridge),
    ("1a", "+ log1p of skewed numeric columns", "skew fixed inside each fold", lambda: M.ridge(features=False)),
    ("1b", "+ ordinal quality scores and domain features", "TotalSF, bathrooms, age, quality × area", M.ridge),
    ("2", "+ drop houses > 4,000 sq ft from training only", "dataset author's recommendation", lambda: M.without_large_houses(M.ridge())),
    ("3a", "Lasso instead of Ridge", "sparser linear model", lambda: M.without_large_houses(M.lasso())),
    ("3b", "Elastic net", "", lambda: M.without_large_houses(M.elastic_net())),
    ("3c", "HistGradientBoosting", "tree model, same features", lambda: M.without_large_houses(M.hist_gbm())),
    ("3d", "GradientBoosting, Huber loss", "less sensitive to the remaining unusual sales", lambda: M.without_large_houses(M.gbm())),
    ("4", "Blend: Lasso + GradientBoosting (final)", "average in log space", lambda: M.without_large_houses(M.blend())),
    ("4x", "Blend of three (+ HistGradientBoosting)", "tried, not kept", lambda: M.without_large_houses(M.blend_three())),
]
FINAL_STEP = "4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", type=Path, default=HERE / "train.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    repeats = 1 if args.quick else 3

    df = pd.read_csv(args.data).drop(columns=["Id"])
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]

    summaries, folds, oof_final = [], {}, None
    for step, name, change, factory in STEPS:
        t = time.time()
        f, oof = cross_validate(factory(), X, y, n_repeats=repeats)
        folds[step] = f
        s = summarise(name, f)
        s.update(step=step, change=change, seconds=round(time.time() - t))
        s["folds_better_than_step0"] = "" if step == "0" else paired_wins(f, folds["0"])
        summaries.append(s)
        print(f"step {step:>3}  RMSLE {s['rmsle_mean']:.4f}  MAPE {s['mape_mean_pct']:.2f}%  "
              f"R2 {s['r2_mean']:.3f} (worst fold {s['r2_worst_fold']:.3f})  {name}", flush=True)
        if step == FINAL_STEP:
            oof_final = oof

    table = pd.DataFrame(summaries)[["step", "configuration", "change", "rmsle_mean", "rmsle_std",
                                     "mape_mean_pct", "mae_mean_usd", "r2_mean", "r2_worst_fold",
                                     "folds_better_than_step0", "seconds"]]
    table.round(4).to_csv(args.out / "experiments.csv", index=False)
    pd.concat({k: v for k, v in folds.items()}, names=["step"]).to_csv(args.out / "experiments_folds.csv")

    final = M.without_large_houses(M.blend())

    # Unseen neighbourhoods: every validation fold holds whole neighbourhoods.
    fg, _ = cross_validate(final, X, y, groups=df["Neighborhood"])
    pd.DataFrame([summarise("final model, GroupKFold by Neighborhood", fg)]).round(4).to_csv(
        args.out / "neighbourhood_holdout.csv", index=False)
    print(f"neighbourhood hold-out RMSLE {fg.rmsle.mean():.4f}  MAPE {fg.mape_pct.mean():.2f}%")

    # 90% prediction intervals (cross-conformal).
    ci = conformal_evaluation(final, X, y, alpha=0.10)
    ci.round(4).to_csv(args.out / "intervals.csv", index=False)
    print(f"90% intervals: coverage {ci.coverage.mean():.3f}, "
          f"median width {ci.median_width_pct_of_price.mean():.1f}% of the predicted price")

    # Where the final model is wrong (out-of-fold predictions of the first repeat).
    anomalous = (df.GrLivArea > 4000) & (df.SaleCondition == "Partial")
    overall = pd.DataFrame([
        {"rows": "all 1,460", **metrics(y, oof_final)},
        {"rows": "without the 2 partial sales over 4,000 sq ft", **metrics(y[~anomalous], oof_final[~anomalous])},
    ])
    overall.round(4).to_csv(args.out / "final_oof_metrics.csv", index=False)
    bands = pd.qcut(y, 5, labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"])
    rows = []
    for b in bands.cat.categories:
        m = bands == b
        rows.append({"price_band": b, "n": int(m.sum()),
                     "price_range_usd": f"{int(y[m].min()):,}-{int(y[m].max()):,}",
                     # R2 is left out: within a narrow price band it says little.
                     **{k: round(v, 4) for k, v in metrics(y[m], oof_final[m]).items() if k != "r2"}})
    pd.DataFrame(rows).to_csv(args.out / "error_by_price_band.csv", index=False)
    err = (np.log1p(oof_final) - np.log1p(y)).abs().sort_values(ascending=False)
    df.loc[err.index[:10], ["Neighborhood", "GrLivArea", "OverallQual", "SaleCondition", "SalePrice"]] \
        .assign(predicted=oof_final.loc[err.index[:10]].round(0)).to_csv(args.out / "largest_errors.csv")
    print(f"Written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
