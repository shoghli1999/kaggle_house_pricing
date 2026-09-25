[![tests](https://github.com/shoghli1999/kaggle_house_pricing/actions/workflows/ci.yml/badge.svg)](https://github.com/shoghli1999/kaggle_house_pricing/actions/workflows/ci.yml)

# House prices in Ames

This is the Kaggle house price data from Ames, Iowa: 1,460 sales and 79 features ([competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)). I use it as a case study, not as a competition entry.

The project has two parts. In the first one I overfitted, then "fixed" it in a way that only hid the problem, and I keep that comparison in the repo. In the second one I rebuilt the model step by step and measured every change on the same cross-validation folds.

## Where it ended up

| | Ridge from part 1 | Final model |
|---|---:|---:|
| RMSLE (the Kaggle metric), 15 CV folds | 0.1339 | 0.1185 |
| MAPE | 9.10% | 7.85% |
| Mean absolute error | $15,685 | $13,495 |
| R², mean (worst fold) | 0.759 (−0.165) | 0.871 (0.619) |

The final model is better on all 15 folds. It averages a Lasso model and a gradient boosting model (Huber loss), both fitted on the log price. When I train it, I leave out the four houses with more than 4,000 sq ft of living area, which is what the dataset's author recommends. I still score those houses in the evaluation.

A few more numbers:

- Two of those big houses were partial sales at very low prices. Many Kaggle notebooks delete them. Without them the out-of-fold RMSLE is 0.1095 and R² is 0.93. I report the number with them as the main result.
- The 90% prediction intervals contained the true price for 90.4% of held-out houses. They are about ±16% of the predicted price.
- When whole neighbourhoods are held out, RMSLE goes up to 0.1287, so the model does use neighbourhood information but doesn't depend on having seen every neighbourhood.
- There are pytest checks (run on GitHub Actions) and a Streamlit app that gives a price estimate with its interval.

## Part 1: fixing overfitting the wrong way first

Mean over five random splits (seeds 42, 0, 7, 123, 2024). MAPE in percent, lower is better. "Gap" is validation MAPE minus train MAPE.

| Configuration | Features | Train MAPE | Val MAPE | Gap (pp) | 5-fold CV MAPE | Mean val R² | Worst val R² |
|---|---|---|---|---|---|---|---|
| A Random Forest, unconstrained | 15 numeric | 4.15 | 10.34 | 6.19 | 11.34 | 0.88 | 0.84 |
| B Random Forest, over-regularised | 15 numeric | 15.27 | 15.30 | 0.02 | 15.64 | 0.69 | 0.65 |
| C Ridge, log target | all 79 | 7.93 | 9.04 | 1.11 | 9.23 | 0.84 | 0.54 |
| D Gradient boosting, log target | all 79 | 4.15 | 9.13 | 4.98 | 9.76 | 0.89 | 0.87 |

You can reproduce the table with `python housing_model.py --seeds 42,0,7,123,2024`.

Configuration B was my first fix. Its gap is almost zero, and at first I counted that as a success. But the training error went from 4.2% to 15.3% and the validation error went up with it. The model had simply become bad everywhere. You can always close the gap that way, so I stopped treating the gap as the goal and started looking at validation error first.

The real problem was that my feature selection only kept numeric columns correlated with the price. That threw away all 43 categorical columns, including `Neighborhood`, `KitchenQual` and `ExterQual`. Putting them back and modelling the log price (configuration C) improved both the error and the gap.

MAPE also hid a bad failure. An earlier version of C had 12.5% MAPE on one split, but its R² was −0.99: it predicted $2,133,254 for a house that sold for $160,000. One long-tailed feature put that house 13 standard deviations away from the training mean, and a linear model extrapolates without limit. Clipping the scaled features to ±5 fixed it. Since then I also report the worst fold, not only the mean.

The first version also had leakage: it joined `train.csv` and `test.csv` before imputing and encoding, and it chose features on the whole file before splitting. Now every step that learns from data sits inside a scikit-learn pipeline and is fitted on the training folds only.

## Part 2: improving the model one step at a time

I used 5-fold cross-validation repeated 3 times (15 folds) on all 1,460 rows, with the same folds for every model, so I could compare changes fold by fold. Imputation, the choice of which columns get a log transform, scaling, encoding and the regularisation strength are all refitted inside each training fold. LassoCV and ElasticNetCV pick their own alpha with an inner CV. The other hyperparameters were fixed before I looked at results.

| Step | Change | RMSLE | MAPE | R² (worst fold) | Beats step 0 | Kept? |
|---|---|---:|---:|---:|---:|---|
| 0 | Ridge, log target (configuration C) | 0.1339 | 9.10% | 0.759 (−0.165) | – | start |
| 1a | log1p of skewed numeric columns, chosen per fold | 0.1298 | 8.73% | 0.802 (0.224) | 15/15 | yes |
| 1b | + ordinal quality scores, total area, bathrooms, age, quality × area | 0.1301 | 8.80% | 0.804 (0.285) | 12/15 | yes (no gain for Ridge, but HistGradientBoosting went from 0.1293 to 0.1257 with it in a side run) |
| 2 | + leave the 4 houses over 4,000 sq ft out of training | 0.1260 | 8.58% | 0.827 (0.411) | 13/15 | yes |
| 3a | Lasso instead of Ridge | 0.1231 | 8.40% | 0.841 (0.461) | 15/15 | yes |
| 3b | Elastic net | 0.1232 | 8.41% | 0.842 (0.459) | 15/15 | no, beat Lasso on only 6/15 folds |
| 3c | HistGradientBoosting | 0.1263 | 8.62% | 0.879 (0.742) | 11/15 | no |
| 3d | GradientBoosting, Huber loss | 0.1217 | 8.12% | 0.879 (0.713) | 14/15 | yes |
| 4 | Average of Lasso and GradientBoosting | 0.1185 | 7.85% | 0.871 (0.619) | 15/15 | final model |
| 4x | Average of three (+ HistGradientBoosting) | 0.1190 | 7.91% | 0.878 (0.673) | 15/15 | no, beat step 4 on only 5/15 folds |

Some things I noticed:

- The new features did almost nothing for the linear model on their own. The real gain came once the four very large houses stopped pulling the linear fit, and once I added a tree model.
- Lasso and gradient boosting make different mistakes. Lasso struggles with houses outside the usual range, the trees with smooth effects of size. Their average beat each of them on 12 of 15 folds.

### Prediction intervals

For each outer fold, I ran an inner 5-fold CV on the training part and used its residuals to set a 90% split-conformal interval in log space. Then I refitted on the whole training part and checked the held-out fold. Coverage was between 87.7% and 92.8% per fold, 90.4% on average, and the median width was 32% of the predicted price (`results/intervals.csv`). In practice the interval is the prediction multiplied or divided by about 1.16.

### Unseen neighbourhoods

With `GroupKFold` by neighbourhood, each validation fold only has neighbourhoods the model never saw. RMSLE goes from 0.1185 to 0.1287 and MAPE from 7.85% to 9.14%.

### Where it goes wrong

Out-of-fold errors of the final model by price band (`results/error_by_price_band.csv`):

| Price band | Range | MAPE | RMSLE |
|---|---|---:|---:|
| lowest 20% | $34,900–$124,000 | 11.6% | 0.162 |
| 20–40% | $124,500–$147,000 | 6.6% | 0.099 |
| 40–60% | $147,400–$179,200 | 7.3% | 0.120 |
| 60–80% | $179,400–$230,000 | 6.5% | 0.103 |
| highest 20% | $230,500–$755,000 | 7.5% | 0.107 |

The cheapest houses are the hardest. Eight of the ten largest errors (`results/largest_errors.csv`) are not normal sales: partial sales of new houses, abnormal sales like foreclosures, one sale within a family and one allocation. The model assumes a normal market sale. Next I would like to model the sale condition directly, or flag unusual sales instead of pricing them.

## Running it

```bash
pip install -r requirements.txt
python experiments.py            # part 2 table and result files, about 15 min on 2 cores (--quick for 1 repeat)
python housing_model.py --seeds 42,0,7,123,2024   # part 1 table
python train_final.py            # final model, 90% interval and a Kaggle submission.csv
streamlit run app.py             # price estimate with interval, results, part 1
pytest -q
```

## Files

```
house_prices/
  features.py     quality scores and extra features (row by row), skew transform fitted per fold
  models.py       all model configurations; TrainingFilter leaves the large houses out of training
  evaluate.py     repeated CV, fold-by-fold comparison, conformal intervals
  data.py         helpers for the app
experiments.py    part 2
housing_model.py  part 1
train_final.py    final fit and submission file
app.py            Streamlit app
tests/            pytest checks
results/          the CSV files behind every number above
archive/          the first version
```

`train_final.py` writes `submission.csv` for the Kaggle test set. I haven't submitted it, so there is no leaderboard score here.
