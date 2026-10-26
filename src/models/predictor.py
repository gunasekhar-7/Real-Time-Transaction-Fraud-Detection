"""Low-latency inference wrapper for production scoring."""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np

from src.config import get_settings
from src.data.schema import account_from_values
from src.features.pipeline import FeaturePipeline
from src.explainability.explainer import FraudShapExplainer
from src.observability import increment

APPROVE_RATIO = 0.5
FALLBACK_THRESHOLDS = (0.35, 0.70)


def serving_thresholds(decline_threshold: float) -> tuple[float, float]:
    """Split one decline threshold into (approve, decline) bands.

    Above the cost-optimal decline threshold the transaction is declined; the
    lower half of that band goes to manual review. Approve must stay strictly
    above zero because the request schema rejects 0.
    """
    decline = round(float(decline_threshold), 4)
    approve = max(0.01, round(decline * APPROVE_RATIO, 4))
    return approve, decline


def load_serving_thresholds(metadata_path: Optional[str] = None) -> tuple[float, float]:
    """Read the cost-optimal threshold from metadata, with a safe fallback.

    A missing or malformed file must not take scoring offline, so it falls back
    to the prototype defaults instead of raising at import time.
    """
    path = metadata_path or get_settings().metadata_path
    try:
        with open(path, "r") as f:
            metadata = json.load(f)
        decline = float(metadata["optimal_threshold"])
    except (OSError, KeyError, TypeError, ValueError):
        return FALLBACK_THRESHOLDS

    approve, decline = serving_thresholds(decline)
    if not 0.0 < approve < decline < 1.0:
        return FALLBACK_THRESHOLDS
    return approve, decline


class FraudPredictor:
    """Production inference engine encapsulating model, feature pipeline, and SHAP explainer."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline_path: Optional[str] = None,
        explainer_path: Optional[str] = None,
        approve_threshold: float = 0.35,
        decline_threshold: float = 0.70,
        cache_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.model_path
        self.pipeline_path = pipeline_path or settings.pipeline_path
        self.explainer_path = explainer_path or settings.explainer_path
        self.approve_threshold = approve_threshold
        self.decline_threshold = decline_threshold

        self.model = None
        self.pipeline: Optional[FeaturePipeline] = None
        self.explainer: Optional[FraudShapExplainer] = None

        self._cache_size = cache_size or settings.predictor_cache_size
        self._recent: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
        # shortcut: one process-wide lock; move dedupe to Redis when running >1 worker
        self._lock = threading.Lock()

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        if os.path.exists(self.pipeline_path):
            with open(self.pipeline_path, "rb") as f:
                self.pipeline = pickle.load(f)
        if os.path.exists(self.explainer_path):
            with open(self.explainer_path, "rb") as f:
                self.explainer = pickle.load(f)
        if self.explainer is not None and self.pipeline is not None:
            self.explainer.attach_categories(self.pipeline.categories)
        self.warm_up()

    def warm_up(self) -> None:
        """Pay first-call costs (booster + SHAP base value) at startup, not on a request."""
        if not self.is_ready or not self.pipeline.feature_columns:
            return
        vector = np.zeros((1, len(self.pipeline.feature_columns)), dtype=np.float32)
        self._probability(vector)
        if self.explainer is not None:
            self.explainer.explain_transaction(vector[0], fraud_prob=0.0)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None

    @property
    def cache_entries(self) -> int:
        """Idempotency entries currently retained (bounded by cache_size)."""
        return len(self._recent)

    def predict_single(
        self,
        transaction_dict: Dict[str, Any],
        include_shap: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Score a transaction.

        A repeat of the same (account, transaction_id) reuses the first call's
        features and probability, so a gateway retry cannot change the decision.
        """
        with self._lock:
            if not self.is_ready:
                self._load_artifacts()
                if not self.is_ready:
                    raise RuntimeError("FraudPredictor artifacts are not loaded.")

            tx_id = transaction_dict.get("transaction_id")
            account = account_from_values(
                transaction_dict.get("card1"), transaction_dict.get("addr1")
            )
            key = (account, str(tx_id)) if tx_id else None

            entry = self._recent.get(key) if key else None
            if entry is None:
                if key:
                    increment("fraud_prediction_cache_misses_total")
                vector = self.pipeline.transform_single(transaction_dict)
                prob = self._probability(vector)
                if key:
                    self._recent[key] = {"probability": prob, "vector": vector}
                    while len(self._recent) > self._cache_size:
                        self._recent.popitem(last=False)
            else:
                self._recent.move_to_end(key)
                vector, prob = entry["vector"], entry["probability"]
                increment("fraud_prediction_cache_hits_total")

            decision, risk_tier, active_threshold = self._decide(prob, threshold)
            increment("fraud_decisions_total", {"decision": decision})

            result: Dict[str, Any] = {
                "transaction_id": str(tx_id) if tx_id else f"tx_gen_{int(time.time() * 1000)}",
                "fraud_probability": round(prob, 4),
                "decision": decision,
                "risk_tier": risk_tier,
                "threshold_used": round(active_threshold, 3),
            }
            if include_shap and self.explainer is not None:
                result["explanation"] = self._explain(vector, prob)
            return result

    def _probability(self, vector: np.ndarray) -> float:
        if hasattr(self.model, "booster_"):
            return float(self.model.booster_.predict(vector)[0])
        return float(self.model.predict_proba(vector)[0, 1])

    def _decide(self, prob: float, threshold: Optional[float] = None) -> tuple[str, str, float]:
        active_threshold = threshold if threshold is not None else self.decline_threshold
        if prob < self.approve_threshold:
            return "APPROVE", "LOW", active_threshold
        if prob < active_threshold:
            return "MANUAL_REVIEW", "MEDIUM", active_threshold
        return "DECLINE", "HIGH", active_threshold

    def _explain(self, vector: np.ndarray, prob: float) -> Dict[str, Any]:
        increment("fraud_shap_explanations_total")
        explanation = self.explainer.explain_transaction(vector, fraud_prob=prob, top_k=4)
        return {
            "base_value": explanation.base_value,
            "top_risk_factors": explanation.top_risk_factors,
            "top_protective_factors": explanation.top_protective_factors,
        }
