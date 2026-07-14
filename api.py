"""
api.py

FastAPI service exposing the churn model as a REST endpoint.

Reuses the exact same prediction + SHAP-explanation logic as the
Streamlit app (via model_utils.py), so the API and the UI can never
give a different answer for the same customer.

Run locally:
    uvicorn api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs, or:
    curl -X POST http://127.0.0.1:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,
           "Tenure":5,"Balance":50000,"NumOfProducts":2,"HasCrCard":1,
           "IsActiveMember":1,"EstimatedSalary":60000}'
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_utils import (
    describe_feature,
    get_recommendations,
    load_model_and_schema,
    predict_with_explanation,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts churn probability for a bank customer, with SHAP-based explainability.",
    version="1.0.0",
)

model, schema = load_model_and_schema()


class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str = Field(..., pattern="^(France|Germany|Spain)$")
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 60000.0,
            }
        }
    }


class FactorOut(BaseModel):
    feature: str
    description: Optional[str]
    shap_impact: float


class PredictionOut(BaseModel):
    churn_probability: float
    prediction: str
    threshold_used: float
    top_risk_factors: List[FactorOut]
    top_protective_factors: List[FactorOut]
    recommendations: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerInput):
    try:
        result = predict_with_explanation(customer.model_dump(), model, schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def to_factor_list(factors):
        out = []
        for name, val, raw in factors:
            desc = describe_feature(name, raw)
            if desc:
                out.append(FactorOut(feature=name, description=desc, shap_impact=round(float(val), 4)))
        return out

    return PredictionOut(
        churn_probability=round(result.probability * 100, 2),
        prediction="Churn" if result.prediction else "Stay",
        threshold_used=round(result.threshold * 100, 2),
        top_risk_factors=to_factor_list(result.risk_factors),
        top_protective_factors=to_factor_list(result.protective_factors),
        recommendations=get_recommendations(result.risk_factors),
    )