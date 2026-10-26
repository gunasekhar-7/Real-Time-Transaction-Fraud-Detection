"""TreeSHAP local and global explainability for fraud scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import shap


@dataclass
class ExplanationResult:
    """Explaining an individual scored transaction."""

    base_value: float
    raw_score: float
    fraud_probability: float
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    all_attributions: Dict[str, float]


# Plain-English names for the features the model actually uses.
FEATURE_DESCRIPTIONS = {
    "TransactionAmt": "Transaction Amount",
    "ProductCD": "Product Category",
    "card1": "Card Identifier",
    "card2": "Card Attribute",
    "card3": "Card Bank Code",
    "card4": "Card Network",
    "card5": "Card Bank Code (2)",
    "card6": "Card Type (Credit/Debit)",
    "addr1": "Billing Region",
    "addr2": "Billing Country",
    "dist1": "Billing-to-Device Distance",
    "dist2": "Billing-to-Device Distance (2)",
    "P_emaildomain": "Purchaser Email Domain",
    "R_emaildomain": "Recipient Email Domain",
    "DeviceType": "Device Type",
    "DeviceInfo": "Device Fingerprint",
    "id_30": "Operating System",
    "id_31": "Browser",
    "id_33": "Screen Resolution",
    "acct_tx_count_1h": "Transactions in the Past Hour",
    "acct_tx_count_24h": "Transactions in the Past 24 Hours",
    "acct_amt_sum_24h": "Amount Spent in the Past 24 Hours",
    "acct_secs_since_prev": "Seconds Since Previous Transaction",
    "acct_amt_vs_mean": "Amount vs. Account Average",
    "acct_tx_count_prior": "Prior Transactions on Account",
    "acct_days_since_first": "Days Since First Account Transaction",
    "hour_of_day": "Local Hour",
    "day_of_week": "Day of Week",
}

_BLOCK_NAMES = {
    "V": "Anonymised Risk Signal",
    "C": "Payment Count History",
    "D": "Days Since Prior Event",
    "M": "Match Flag",
}


def describe_feature(name: str) -> str:
    """Human label for a feature, including the vendor blocks."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    prefix, rest = name[:1], name[1:]
    if prefix in _BLOCK_NAMES and rest.isdigit():
        return f"{_BLOCK_NAMES[prefix]} {name}"
    return f"Feature {name}"


class FraudShapExplainer:
    """Pre-computes and caches TreeSHAP explainer for high-speed local inference."""

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        # Categorical vocabularies, attached by the predictor once the feature
        # pipeline is loaded. Empty until then, so raw codes are never shown.
        self.categories: Dict[str, Any] = {}
        # Fast TreeExplainer initialization
        self.explainer = shap.TreeExplainer(self.model)
        # Expected base value (bias term in log-odds or margin)
        self.base_value = self._read_base_value()

    def attach_categories(self, categories: Dict[str, Any]) -> None:
        """Teach the explainer how to turn encoded codes back into labels."""
        self.categories = categories or {}

    def _value_text(self, name: str, value: float) -> str:
        """Render a feature value as the operator sent it, not as its code.

        Categorical features are stored as integer positions into the pipeline
        vocabulary. Printing those positions produced reasons like
        "Browser (52.0)", which mean nothing to a fraud analyst.
        """
        if not np.isfinite(value):
            return "missing"
        vocabulary = getattr(self, "categories", {}).get(name)
        if vocabulary is not None and float(value).is_integer():
            index = int(value)
            if 0 <= index < len(vocabulary):
                label = str(vocabulary[index])
                # Float-typed vendor columns stringify integers as "476.0".
                if label.endswith(".0") and label[:-2].isdigit():
                    label = label[:-2]
                return label
        return f"{value:,.2f}".rstrip("0").rstrip(".")

    def _read_base_value(self) -> Optional[float]:
        """Read SHAP's base value, which some models only populate after the first call."""
        value = self.explainer.expected_value
        if value is None:
            return None
        if isinstance(value, (list, np.ndarray)):
            return float(np.ravel(value)[-1])
        return float(value)

    def explain_transaction(
        self,
        feature_vector: np.ndarray,
        fraud_prob: float,
        top_k: int = 5,
    ) -> ExplanationResult:
        """Compute fast local SHAP attributions for a single transaction vector.

        feature_vector: shape (1, n_features) or (n_features,)
        """
        arr = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        raw_shap = self.explainer.shap_values(arr)
        if self.base_value is None:
            self.base_value = self._read_base_value()

        # Handle binary classification output format differences in SHAP versions
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[-1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif hasattr(raw_shap, "values"):
            shap_vals = raw_shap.values[0]
            if len(shap_vals.shape) == 2:
                shap_vals = shap_vals[:, -1]
        elif len(raw_shap.shape) == 3:
            shap_vals = raw_shap[0, :, -1]
        elif len(raw_shap.shape) == 2:
            shap_vals = raw_shap[0]
        else:
            shap_vals = raw_shap

        attributions = {name: float(val) for name, val in zip(self.feature_names, shap_vals)}

        # Sort features by contribution
        sorted_factors = sorted(
            attributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        top_risk_factors = []
        top_protective_factors = []

        for name, val in sorted_factors:
            friendly_name = describe_feature(name)
            idx = self.feature_names.index(name)
            feat_val = float(arr[0, idx])
            value_text = self._value_text(name, feat_val)

            entry = {
                "feature": name,
                "feature_name": friendly_name,
                # NaN is not valid JSON; null plus the "missing" display keeps
                # the payload serializable and still tells the analyst why.
                "feature_value": round(feat_val, 2) if np.isfinite(feat_val) else None,
                "feature_value_display": value_text,
                "shap_impact": round(val, 4),
                "reason_code": (
                    f"{friendly_name} ({value_text}) increased fraud risk by +{val:.2f}"
                    if val > 0
                    else f"{friendly_name} ({value_text}) reduced risk by {abs(val):.2f}"
                ),
            }

            if val > 0 and len(top_risk_factors) < top_k:
                top_risk_factors.append(entry)
            elif val < 0 and len(top_protective_factors) < top_k:
                top_protective_factors.append(entry)

        raw_score = float(self.base_value + sum(shap_vals))

        return ExplanationResult(
            base_value=round(self.base_value, 4),
            raw_score=round(raw_score, 4),
            fraud_probability=round(float(fraud_prob), 4),
            top_risk_factors=top_risk_factors,
            top_protective_factors=top_protective_factors,
            all_attributions={k: round(v, 4) for k, v in attributions.items()},
        )
