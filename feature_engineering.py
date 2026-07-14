"""
feature_engineering.py

Single source of truth for turning raw customer fields into model-ready
features. Both churn_training.py (bulk, from CSV) and churn_app.py
(single row, from the Streamlit form) call this exact same function,
so the two code paths can never silently drift apart.
"""

import pandas as pd

RAW_GEOGRAPHY_CATEGORIES = ["France", "Germany", "Spain"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a DataFrame with raw columns:
    CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

    Returns a DataFrame with encoded + engineered features.
    Safe to call on a full training set OR a single-row inference input —
    dummy columns are always guaranteed to exist either way.
    """
    df = df.copy()

    # --- Encode categoricals ---
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # One-hot encode geography explicitly (NOT via pd.get_dummies).
    #
    # pd.get_dummies(..., drop_first=True) is unsafe here: on a single-row
    # DataFrame (which is what inference sends), whatever category is
    # present is the only one seen, so drop_first drops it — producing
    # ZERO dummy columns. The old fallback that filled missing dummy
    # columns with 0 then silently encoded every Germany/Spain customer
    # as if they were in France. This explicit version is correct
    # regardless of how many rows or which categories are present.
    df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
    df = df.drop(columns=["Geography"])

    # --- Engineered features ---
    # These are all pure numeric transforms (no new categories to
    # keep in sync), so they're safe on single-row inputs too.
    df["Zero_Balance"] = (df["Balance"] == 0).astype(int)
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Products_Per_Tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    return df