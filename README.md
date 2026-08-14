# House price regression — fixing overfitting the wrong way, then the right way

Ames housing data (1,460 rows, 79 features), used as a case study rather than a
competition entry.

This repository started as a Random Forest that overfitted. My first fix made the
train/validation gap disappear and I recorded that as a success. It was not: the
model had become 4–5 percentage points *less* accurate. This version keeps both
attempts in the comparison so the mistake and its correction are visible, and adds
two configurations that fix the accuracy and the gap at the same time.

---

## Results

Mean over five random splits (seeds 42, 0, 7, 123, 2024). MAPE in percent, lower
is better. `gap` is validation MAPE minus train MAPE.

| Configuration | Features | Train MAPE | Val MAPE | Gap (pp) | 5-fold CV MAPE | Mean val R² | Worst val R² |
|---|---|---|---|---|---|---|---|
| **A** Random Forest, unconstrained | 15 numeric | 4.15 | 10.34 | 6.19 | 11.34 | 0.88 | 0.84 |
| **B** Random Forest, over-regularised | 15 numeric | 15.27 | 15.30 | **0.02** | 15.64 | 0.69 | 0.65 |
| **C** Ridge, log target | all 79 | 7.93 | **9.04** | 1.11 | **9.23** | 0.84 | 0.54 |
| **D** Gradient boosting, log target | all 79 | 4.15 | 9.13 | 4.98 | 9.76 | **0.89** | **0.87** |

Reproduce with:

```bash
python housing_model.py --seeds 42,0,7,123,2024
```

### What the table says

**Configuration B closed the gap perfectly and that is exactly the problem.** A
gap of 0.02 pp looks like a textbook result. It was achieved by making the model
uniformly bad — training error rose from 4.2% to 15.3% and validation error rose
with it. Any model can reach a zero gap by predicting the training mean. The gap
is a diagnostic, not an objective, and I had been selecting on it.

The decisive constraint was `min_weight_fraction_leaf=0.1`, which forces every
leaf to carry at least 10% of the training weight. That caps each tree at roughly
ten leaves regardless of what `max_depth` says, so the rest of the tuning grid was
largely decorative.

**The real problem was never model capacity — it was feature starvation.**
Configurations A and B see only the 15 numeric columns whose correlation with
`SalePrice` exceeds 0.25. That silently discards all 43 categorical columns,
including `Neighborhood`, `KitchenQual` and `ExterQual`, which are among the
strongest price signals in this dataset. Correlation-based selection cannot see
them because Pearson correlation is not defined for unordered categories.

Putting those columns back and log-transforming the target (`SalePrice` is right
skewed, and MAPE is a relative error measure, so modelling in log space matches
the metric) gives configuration C: **9.23% cross-validated MAPE against 15.64%
for the over-regularised attempt, with a gap of 1.11 pp rather than 6.19 pp.**
Better on both axes at once, from a plain linear model.

**Gradient boosting (D) is the more robust choice despite a slightly higher
MAPE.** C wins on average error but has the worst single-split R² in the table
(0.54). That asymmetry is the interesting part — see below.

---

## The fragility that MAPE hid

An earlier version of configuration C scored a perfectly respectable 12.5% MAPE
on seed 0 while its validation R² was **−0.99**. One row was predicted at
$2,133,254 against an actual $160,000.

The cause: Ames has long-tailed numeric columns (`LotArea`, `MiscVal`), and after
standardisation a validation row landed 13 standard deviations from the training
mean. A linear model extrapolates that without limit. Mean absolute percentage
error barely registered it — one bad row out of 292 moves a mean of ratios very
little — while squared error registered it immediately.

The fix is a clip to ±5 training standard deviations after scaling, fitted on
training statistics only (`CLIP_SIGMA` in `housing_model.py`). That lifts the
worst-case R² from −0.99 to 0.54. It does not make the linear model as robust as
the tree ensemble, which is why the table reports worst-case R² alongside the
mean: **if a single catastrophic prediction is costly in the application, D is
the right model even though C has the lower average error.**

The general lesson, and the reason this section exists: report an error measure
that is sensitive to the failure mode you actually care about, and report the
worst case, not only the mean.

---

## Leakage fixed from the first version

The original script concatenated `train.csv` and `test.csv` before imputing
missing values and label-encoding categories, so test-set statistics informed the
training features. Feature selection also ran over the whole training file before
the validation split, so validation rows influenced which features were chosen.

In this version every transformation lives inside an sklearn `Pipeline` and is
fitted on training folds only, including inside cross-validation.

---

## What I would do next

- Ordinal-aware encoding for the quality columns (`ExterQual`, `KitchenQual`,
  `BsmtQual` are ranked Ex > Gd > TA > Fa > Po, and both encoders here throw that
  ordering away).
- Stack C and D rather than choosing between them — their errors are unlikely to
  be correlated given how differently they fail.
- Quantile regression for prediction intervals, which is what a valuation use
  case would actually need.
- Group the validation split by `Neighborhood` to check the model is not simply
  memorising neighbourhood price levels.

---

## Layout

```
housing_model.py    # the whole comparison; four configurations, one entry point
app.py              # Streamlit dashboard over the analysis
train.csv           # Ames training data (labelled)
test.csv            # Kaggle test split; unlabelled, so unused for evaluation
results/            # written by the script, not committed by hand
requirements.txt
```

```bash
pip install -r requirements.txt
python housing_model.py                      # one split, ~10 s
python housing_model.py --seeds 42,0,7,123,2024   # the table above, ~40 s
python housing_model.py --quick              # smaller search space
streamlit run app.py                         # dashboard
```

Data: [House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
(Ames, Iowa). Not a competition submission.
