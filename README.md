# Customer Churn Prediction System

An end-to-end churn prediction pipeline (XGBoost + SMOTE) with a Streamlit
app for interactive, explainable predictions.

## What this does

- Trains an XGBoost classifier on the `Churn_Modelling.csv` banking dataset
  to predict which customers are likely to churn.
- Handles class imbalance with SMOTE (applied only to the training split,
  never the test split, to avoid leakage).
- Adds engineered features (`Zero_Balance`, `Balance_Salary_Ratio`,
  `Products_Per_Tenure`) on top of the raw fields.
- Tunes hyperparameters with `RandomizedSearchCV`, optimizing PR-AUC
  rather than accuracy, since the target is ~80/20 imbalanced.
- Tunes the decision threshold on validation data (maximizing F1 on the
  churn class) instead of using a blind 0.5 cutoff.
- Saves a `feature_schema.json` alongside the model containing the exact
  training feature order, the tuned threshold, real feature averages, and
  final metrics — this is the single source of truth the Streamlit app
  reads from, so training and serving can never silently drift apart.
- Serves predictions through a Streamlit app with SHAP-based
  explainability (force plot per prediction), a risk-factor breakdown,
  and a downloadable prediction report.

## Project structure

```
feature_engineering.py   # shared preprocessing — used by training, the app, and the API
model_utils.py             # shared prediction + SHAP-explanation logic — used by the app and the API
churn_training.py          # trains the model, saves xgb_model.pkl + feature_schema.json
churn_app.py                 # Streamlit app
api.py                        # FastAPI service exposing /predict
tests/                          # pytest suite (feature engineering, model logic, API)
requirements.txt
DEPLOYMENT.md              # how to deploy the app, run the API, and run tests/CI
Dataset/Churn_Modelling.csv  
```

## Setup

```bash
pip install -r requirements.txt
```

Place `Churn_Modelling.csv` in a `Dataset/` folder next to the scripts.

## Run

```bash
# 1. Train the model (also writes xgb_model.pkl + feature_schema.json)
python churn_training.py

# 2. Launch the app
streamlit run churn_app.py
```

## Results

*(Fill this in after running `churn_training.py` — the numbers below are
placeholders until you rerun with the new feature engineering + tuning.)*

| Metric                          | Value |
|----------------------------------|-------|
| ROC-AUC                          | TBD   |
| PR-AUC                           | TBD   |
| F1 (churn class, default 0.5)     | TBD   |
| F1 (churn class, tuned threshold) | TBD   |
| Tuned decision threshold          | TBD   |

Previously (no feature engineering, no tuning, default threshold):
precision 0.57 / recall 0.65 / F1 0.61 on the churn class. The tuned
threshold + extra features should move recall up noticeably without a
big drop in precision — check your own numbers after training.

## Testing & API

- `pytest tests/ -v` — unit tests for preprocessing, prediction logic, and the API. See `DEPLOYMENT.md` for setup and CI instructions.
- `uvicorn api:app --reload` — run the model as a REST API (`/predict`, `/health`), sharing the exact same prediction code as the Streamlit app via `model_utils.py`.
- See `DEPLOYMENT.md` for deploying the Streamlit app publicly (Streamlit Community Cloud / Hugging Face Spaces).

## Notes on design decisions

- **Why PR-AUC for hyperparameter search, not accuracy?** With ~80% of
  customers not churning, a model that always predicts "stay" gets 80%
  accuracy while being useless. PR-AUC is sensitive to how well the
  model actually ranks and separates the minority (churn) class.
- **Why tune the threshold instead of using 0.5?** The default 0.5 cutoff
  is arbitrary once you've rebalanced training data with SMOTE. Tuning it
  against the precision-recall curve lets you pick the operating point
  that best matches the actual cost tradeoff (missing a churner is
  usually more costly than one extra false alarm).
- **Why a shared `feature_engineering.py`?** In the original version,
  the training script and the Streamlit app each re-implemented the same
  encoding logic by hand. Any change to one (e.g. adding a country) could
  silently desync the other. Now both import the same function.
- **A bug the test suite caught:** `feature_engineering.py` originally
  used `pd.get_dummies(..., drop_first=True)` to one-hot encode Geography.
  This works fine on the full training set, but on a single-row inference
  input (all the Streamlit app or API ever sends), only one country is
  ever present — so `drop_first` drops the *only* category seen, producing
  zero dummy columns. The fallback code that filled in missing dummy
  columns with 0 then silently encoded every Germany/Spain customer as if
  they were in France. Writing a unit test that explicitly checks dummy
  values for each country (`test_geography_dummy_values_correct`) caught
  this immediately. Fixed by encoding geography explicitly instead of
  via `get_dummies`, which is correct regardless of row count.
