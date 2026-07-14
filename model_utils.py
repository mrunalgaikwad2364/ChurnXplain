"""
model_utils.py

Shared model-loading, prediction, and SHAP-explanation logic used by
BOTH the Streamlit app (churn_app.py) and the FastAPI service (api.py).

Why this exists: previously the UI computed predictions and SHAP values
inline. If a second consumer (the API) reimplemented the same logic by
hand, the two could silently drift apart — exactly the kind of bug this
project has already hit once with feature ordering. Centralizing it here
means there is exactly one prediction code path in the whole project.
"""

import json
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import shap

from feature_engineering import engineer_features

MODEL_PATH = "xgb_model.pkl"
SCHEMA_PATH = "feature_schema.json"

FEATURE_LABELS = {
    "CreditScore": "Credit score",
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "Tenure (years with bank)",
    "Balance": "Account balance",
    "NumOfProducts": "Number of products held",
    "HasCrCard": "Has a credit card",
    "IsActiveMember": "Active membership status",
    "EstimatedSalary": "Estimated salary",
    "Geography_Germany": "Located in Germany",
    "Geography_Spain": "Located in Spain",
    "Zero_Balance": "Zero account balance",
    "Balance_Salary_Ratio": "Balance-to-salary ratio",
    "Products_Per_Tenure": "Products per tenure year",
}

# Actionable recommendations, keyed by feature name. Deliberately
# excludes Gender and Geography — those are demographic attributes,
# not behaviors a retention team should act on individually.
RECOMMENDATION_MAP = {
    "Age": "Offer senior-oriented benefits or personalized retirement planning outreach.",
    "IsActiveMember": "Re-engage with loyalty rewards, personalized offers, or a check-in call.",
    "NumOfProducts": "Customers holding 3+ products show unusually high churn in this dataset — schedule a relationship review.",
    "Products_Per_Tenure": "Rapid product accumulation relative to tenure — confirm onboarding support is adequate.",
    "Tenure": "Newer customers benefit from stronger early-relationship engagement.",
    "Balance": "High-balance customers may warrant dedicated relationship management.",
    "Zero_Balance": "Encourage account funding through onboarding incentives.",
    "CreditScore": "Consider offering financial wellness resources or credit-building products.",
    "EstimatedSalary": "Consider premium service tiers to increase stickiness for high earners.",
    "HasCrCard": "Consider a credit card cross-sell offer to increase product stickiness.",
    "Balance_Salary_Ratio": "Unusual balance-to-salary ratio — worth a manual account review.",
}

RAW_CUSTOMER_FIELDS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]


def load_model_and_schema(model_path: str = MODEL_PATH, schema_path: str = SCHEMA_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return model, schema


def build_input_row(raw_fields: dict, feature_names: list) -> pd.DataFrame:
    """Turn a dict of raw customer fields into a model-ready, correctly
    ordered single-row DataFrame."""
    raw_df = pd.DataFrame([{k: raw_fields[k] for k in RAW_CUSTOMER_FIELDS}])
    input_df = engineer_features(raw_df)
    return input_df.reindex(columns=feature_names, fill_value=0)


def describe_feature(name: str, raw_value) -> Optional[str]:
    """Turn a feature name + its input value into a short readable phrase."""
    label = FEATURE_LABELS.get(name, name)
    if name == "Gender":
        return f"{label}: {'Male' if raw_value == 1 else 'Female'}"
    if name in ("HasCrCard", "IsActiveMember", "Zero_Balance"):
        return f"{label}: {'Yes' if raw_value == 1 else 'No'}"
    if name in ("Geography_Germany", "Geography_Spain"):
        return label if raw_value == 1 else None  # only show if true
    if name in ("Balance", "EstimatedSalary"):
        return f"{label}: ${raw_value:,.0f}"
    if name in ("Balance_Salary_Ratio", "Products_Per_Tenure"):
        return f"{label}: {raw_value:.2f}"
    return f"{label}: {raw_value:g}"


def get_recommendations(risk_factors: List[Tuple[str, float, float]], top_n: int = 3) -> List[str]:
    recos = []
    for name, _val, _raw in risk_factors:
        reco = RECOMMENDATION_MAP.get(name)
        if reco and reco not in recos:
            recos.append(reco)
        if len(recos) == top_n:
            break
    return recos


@dataclass
class PredictionResult:
    probability: float  # 0-1
    prediction: int  # 0 or 1
    threshold: float  # 0-1
    risk_factors: List[Tuple[str, float, float]]  # (name, shap_value, raw_value)
    protective_factors: List[Tuple[str, float, float]]
    shap_explanation: object  # single-sample shap.Explanation, for plotting


def predict_with_explanation(raw_fields: dict, model, schema: dict, top_n: int = 5) -> PredictionResult:
    """Single source of truth for turning raw customer fields into a
    prediction + SHAP-based explanation. Used by both churn_app.py and api.py."""
    feature_names = schema["feature_names"]
    threshold = schema["best_threshold"]
    input_df = build_input_row(raw_fields, feature_names)

    prob = float(model.predict_proba(input_df)[0][1])
    prediction = int(prob >= threshold)

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    sv = shap_values[0].values
    raw_vals = input_df.iloc[0].values
    contributions = list(zip(feature_names, sv, raw_vals))

    risk_factors = sorted(
        [c for c in contributions if c[1] > 0], key=lambda c: c[1], reverse=True
    )[:top_n]
    protective_factors = sorted(
        [c for c in contributions if c[1] < 0], key=lambda c: c[1]
    )[:top_n]

    return PredictionResult(
        probability=prob,
        prediction=prediction,
        threshold=threshold,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        shap_explanation=shap_values[0],
    )