"""
tests/test_model_utils.py

Tests for the shared prediction + explanation logic in model_utils.py.
Requires xgb_model.pkl and feature_schema.json to exist (run
`python churn_training.py` first) — tests are skipped automatically
if they're not found, so this doesn't break CI runs before training.
"""

import os

import pytest

from model_utils import load_model_and_schema, predict_with_explanation

MODEL_PATH = "xgb_model.pkl"
SCHEMA_PATH = "feature_schema.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH)),
    reason="Run `python churn_training.py` first to generate the model and schema.",
)

SAMPLE_CUSTOMER = dict(
    CreditScore=650, Geography="France", Gender="Male", Age=35, Tenure=5,
    Balance=50000.0, NumOfProducts=2, HasCrCard=1, IsActiveMember=1,
    EstimatedSalary=60000.0,
)


@pytest.fixture(scope="module")
def model_and_schema():
    return load_model_and_schema()


def test_schema_has_required_keys(model_and_schema):
    _, schema = model_and_schema
    for key in ["feature_names", "best_threshold", "feature_averages", "metrics"]:
        assert key in schema


def test_threshold_in_valid_range(model_and_schema):
    _, schema = model_and_schema
    assert 0.0 <= schema["best_threshold"] <= 1.0


def test_feature_names_match_model(model_and_schema):
    model, schema = model_and_schema
    assert list(model.get_booster().feature_names) == schema["feature_names"]


def test_prediction_probability_in_valid_range(model_and_schema):
    model, schema = model_and_schema
    result = predict_with_explanation(SAMPLE_CUSTOMER, model, schema)
    assert 0.0 <= result.probability <= 1.0


def test_prediction_label_matches_threshold(model_and_schema):
    model, schema = model_and_schema
    result = predict_with_explanation(SAMPLE_CUSTOMER, model, schema)
    assert result.prediction == int(result.probability >= result.threshold)


def test_risk_and_protective_factors_are_disjoint(model_and_schema):
    """A feature can't simultaneously push toward churn and away from it."""
    model, schema = model_and_schema
    result = predict_with_explanation(SAMPLE_CUSTOMER, model, schema)
    risk_names = {f[0] for f in result.risk_factors}
    protective_names = {f[0] for f in result.protective_factors}
    assert risk_names.isdisjoint(protective_names)


def test_risk_factors_sorted_by_descending_impact(model_and_schema):
    model, schema = model_and_schema
    result = predict_with_explanation(SAMPLE_CUSTOMER, model, schema)
    values = [f[1] for f in result.risk_factors]
    assert values == sorted(values, reverse=True)


def test_older_less_active_customer_scores_higher_risk(model_and_schema):
    """Sanity check: a profile that's clearly worse on known churn
    drivers (older, inactive, more products, shorter tenure) should
    get a higher churn probability than a clearly safer profile."""
    model, schema = model_and_schema
    lower_risk_profile = dict(SAMPLE_CUSTOMER, Age=25, IsActiveMember=1, NumOfProducts=2, Tenure=8)
    higher_risk_profile = dict(SAMPLE_CUSTOMER, Age=58, IsActiveMember=0, NumOfProducts=4, Tenure=1)

    lower = predict_with_explanation(lower_risk_profile, model, schema)
    higher = predict_with_explanation(higher_risk_profile, model, schema)

    assert higher.probability > lower.probability