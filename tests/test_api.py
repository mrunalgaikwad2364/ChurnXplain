"""
tests/test_api.py

Tests for the FastAPI service (api.py). Requires xgb_model.pkl and
feature_schema.json — skipped automatically if not found.
"""

import os

import pytest

MODEL_PATH = "xgb_model.pkl"
SCHEMA_PATH = "feature_schema.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH)),
    reason="Run `python churn_training.py` first to generate the model and schema.",
)

SAMPLE_PAYLOAD = {
    "CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 35,
    "Tenure": 5, "Balance": 50000.0, "NumOfProducts": 2, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 60000.0,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 100.0
    assert body["prediction"] in ("Churn", "Stay")
    assert isinstance(body["top_risk_factors"], list)
    assert isinstance(body["top_protective_factors"], list)
    assert isinstance(body["recommendations"], list)


def test_predict_rejects_invalid_geography(client):
    bad = dict(SAMPLE_PAYLOAD, Geography="Atlantis")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_age(client):
    bad = dict(SAMPLE_PAYLOAD, Age=150)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_products(client):
    bad = dict(SAMPLE_PAYLOAD, NumOfProducts=0)
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "Age"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_api_agrees_with_model_utils_directly(client):
    """The API should give the exact same probability as calling
    model_utils directly — this is the whole point of sharing the code."""
    from model_utils import load_model_and_schema, predict_with_explanation

    model, schema = load_model_and_schema()
    direct_result = predict_with_explanation(SAMPLE_PAYLOAD, model, schema)

    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    api_prob = resp.json()["churn_probability"]

    assert api_prob == pytest.approx(round(direct_result.probability * 100, 2))