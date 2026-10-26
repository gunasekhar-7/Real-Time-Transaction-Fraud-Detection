"""Local explanations must describe the value the operator sent, not its code."""

import numpy as np
import pandas as pd

from src.explainability.explainer import FraudShapExplainer


class _StubTreeExplainer:
    expected_value = 0.0

    def shap_values(self, _arr):
        return np.array([[0.6, -0.9, 0.2]])


def _explainer():
    explainer = FraudShapExplainer.__new__(FraudShapExplainer)
    explainer.feature_names = ["id_31", "TransactionAmt", "DeviceType"]
    explainer.categories = {
        "id_31": pd.Index(["chrome", "safari"]),
        "DeviceType": pd.Index(["desktop", "mobile"]),
    }
    explainer.base_value = 0.0
    explainer.explainer = _StubTreeExplainer()
    return explainer


def test_categorical_codes_are_decoded_in_reason_text():
    result = _explainer().explain_transaction(
        np.array([1.0, 125.0, np.nan]), fraud_prob=0.5, top_k=3
    )
    risks = {factor["feature"]: factor for factor in result.top_risk_factors}

    assert risks["id_31"]["feature_value_display"] == "safari"
    assert "Browser (safari) increased fraud risk" in risks["id_31"]["reason_code"]
    assert "(1.0)" not in risks["id_31"]["reason_code"]

    assert risks["DeviceType"]["feature_value_display"] == "missing"
    assert risks["DeviceType"]["feature_value"] is None
    assert "Device Type (missing) increased fraud risk" in risks["DeviceType"]["reason_code"]


def test_numeric_categorical_labels_lose_the_float_suffix():
    explainer = _explainer()
    explainer.categories["addr1"] = pd.Index(["204.0", "476.0"])

    assert explainer._value_text("addr1", 1.0) == "476"
    assert explainer._value_text("addr1", np.nan) == "missing"


def test_numeric_values_keep_readable_formatting():
    result = _explainer().explain_transaction(
        np.array([1.0, 125.0, np.nan]), fraud_prob=0.5, top_k=3
    )

    assert result.top_protective_factors[0]["feature_value_display"] == "125"
    assert (
        "Transaction Amount (125) reduced risk" in (result.top_protective_factors[0]["reason_code"])
    )
