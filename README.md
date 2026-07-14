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


| Metric                          | Value |
|----------------------------------|-------|
| ROC-AUC                          | TBD   |
| PR-AUC                           | TBD   |
| F1 (churn class, default 0.5)     | TBD   |
| F1 (churn class, tuned threshold) | TBD   |
| Tuned decision threshold          | TBD   |


## Testing & API

- `pytest tests/ -v` — unit tests for preprocessing, prediction logic, and the API. See `DEPLOYMENT.md` for setup and CI instructions.
- `uvicorn api:app --reload` — run the model as a REST API (`/predict`, `/health`), sharing the exact same prediction code as the Streamlit app via `model_utils.py`.
- See `DEPLOYMENT.md` for deploying the Streamlit app publicly (Streamlit Community Cloud / Hugging Face Spaces).


