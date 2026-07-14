"""
tests/test_feature_engineering.py

Unit tests for feature_engineering.py. These don't need a trained
model or schema file — they only test the preprocessing logic itself.
"""

import pandas as pd
import pytest

from feature_engineering import RAW_GEOGRAPHY_CATEGORIES, engineer_features


def make_raw_row(**overrides):
    base = dict(
        CreditScore=650, Geography="France", Gender="Male", Age=35, Tenure=5,
        Balance=50000.0, NumOfProducts=2, HasCrCard=1, IsActiveMember=1,
        EstimatedSalary=60000.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


def test_gender_encoding():
    assert engineer_features(make_raw_row(Gender="Male"))["Gender"].iloc[0] == 1
    assert engineer_features(make_raw_row(Gender="Female"))["Gender"].iloc[0] == 0


@pytest.mark.parametrize("country", RAW_GEOGRAPHY_CATEGORIES)
def test_geography_dummy_columns_always_present(country):
    df = engineer_features(make_raw_row(Geography=country))
    assert "Geography_Germany" in df.columns
    assert "Geography_Spain" in df.columns
    assert "Geography" not in df.columns  # original column should be gone


def test_geography_dummy_values_correct():
    df = engineer_features(make_raw_row(Geography="Germany"))
    assert df["Geography_Germany"].iloc[0] == 1
    assert df["Geography_Spain"].iloc[0] == 0

    df = engineer_features(make_raw_row(Geography="Spain"))
    assert df["Geography_Germany"].iloc[0] == 0
    assert df["Geography_Spain"].iloc[0] == 1

    # France is the dropped baseline category — both dummies should be 0
    df = engineer_features(make_raw_row(Geography="France"))
    assert df["Geography_Germany"].iloc[0] == 0
    assert df["Geography_Spain"].iloc[0] == 0


def test_zero_balance_flag():
    assert engineer_features(make_raw_row(Balance=0))["Zero_Balance"].iloc[0] == 1
    assert engineer_features(make_raw_row(Balance=1))["Zero_Balance"].iloc[0] == 0


def test_balance_salary_ratio():
    df = engineer_features(make_raw_row(Balance=50000, EstimatedSalary=100000))
    expected = 50000 / (100000 + 1)
    assert df["Balance_Salary_Ratio"].iloc[0] == pytest.approx(expected)


def test_products_per_tenure():
    df = engineer_features(make_raw_row(NumOfProducts=3, Tenure=2))
    expected = 3 / (2 + 1)
    assert df["Products_Per_Tenure"].iloc[0] == pytest.approx(expected)


def test_products_per_tenure_handles_zero_tenure():
    # Should not raise a divide-by-zero error for a brand-new customer
    df = engineer_features(make_raw_row(NumOfProducts=1, Tenure=0))
    assert df["Products_Per_Tenure"].iloc[0] == pytest.approx(1.0)


def test_works_on_bulk_dataframe_with_mixed_countries():
    rows = pd.concat(
        [make_raw_row(Geography=c) for c in RAW_GEOGRAPHY_CATEGORIES],
        ignore_index=True,
    )
    df = engineer_features(rows)
    assert len(df) == len(RAW_GEOGRAPHY_CATEGORIES)
    assert {"Geography_Germany", "Geography_Spain"}.issubset(df.columns)


def test_does_not_mutate_input_dataframe():
    original = make_raw_row()
    original_copy = original.copy()
    engineer_features(original)
    pd.testing.assert_frame_equal(original, original_copy)