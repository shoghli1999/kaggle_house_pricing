# Archive: the first version (2025)

These files are the original Random Forest project, kept unchanged so the comparison in the main README can be checked.

They contain the two mistakes that Part 1 of the main README describes:
- train and test were concatenated before imputation and encoding (leakage);
- success was judged by a small train/validation gap instead of by validation error.

Don't use them for anything else. The current code is `house_prices/`, `experiments.py`, `train_final.py` and `app.py`.
