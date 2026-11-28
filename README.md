# Credit Card Fraud Detection

Flags fraudulent card transactions, explains why (SHAP), and picks the decision threshold that minimises the bank's cost.
Dataset: Kaggle ULB credit card fraud (284,807 transactions, 0.17% fraud), placed at `data/creditcard.csv`.

## Results (untouched test set, 56,746 transactions)

| Metric | Goal | Achieved |
|---|---|---|
| Recall (fraud caught) | ≥ 80% | **81.1%** (77 of 95) |
| Precision (alerts that are fraud) | ≥ 70% | **72.6%** (29 false alarms) |
| PR-AUC | above LR baseline (0.754 CV) | **0.817** (CV 0.851) |
| Prediction time | < 1 s | ~5 ms |
| Fraud loss | — | 14,766 → 3,479 (**76% saved**) |

Model: XGBoost (tuned), threshold t = 0.08, chosen by minimum total cost on out-of-fold training predictions with a false-alarm cost of 5 (assumption).

## How to run (Windows)

- **Demo app:** double-click `START_DEMO.bat` → browser opens the demo.
- **Notebooks:** double-click `START_NOTEBOOKS.bat` → open `01` … `05` in order.

First-time setup on a new machine (Python 3.11):

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Optional API: `.venv\Scripts\python -m uvicorn app.api:app` then POST JSON with `Time, V1 … V28, Amount` to `/predict`.

## Project layout

```
notebooks/01_eda.ipynb             EDA + accuracy trap (dummy: 99.83% accuracy, 0% recall)
notebooks/02_preprocessing.ipynb   split, scaling, leakage + stratification checks
notebooks/03_ml_models.ipynb       LR and XGBoost x 4 imbalance techniques (5-fold CV)
notebooks/04_deep_learning.ipynb   Random Forest, MLP, Autoencoder + full comparison
notebooks/05_threshold_shap.ipynb  tuning, cost threshold, final test, SHAP, save
src/                               config, preprocess, train, deep, evaluate
models/best_model.joblib           full pipeline (scaler + XGBoost)
models/threshold.json              threshold, params and test results
app/streamlit_app.py               demo UI;  app/api.py  optional FastAPI
reports/figures/                   charts for the slides
```

## Notes and limitations

- The scaler is saved inside the pipeline (no separate `scaler.joblib`), so the app cannot scale twice.
- Card-level features (e.g. transactions per card per hour) are impossible: the dataset has no card or customer ID.
- V1–V28 are anonymised PCA components, so explanations name components, not business terms.
- Data covers 2 days of 2013; a time-based check gave PR-AUC 0.81 vs 0.85 stratified, so some drift is expected.
- The false-alarm cost (5) is an illustrative assumption.
