"""Prediction routes for low-latency single and batch scoring."""

from __future__ import annotations

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    TransactionPayload,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ThresholdConfig,
    ExplanationPayload,
    RiskFactor,
)
from src.api.security import require_api_key
from src.config import get_settings
from src.models.predictor import FraudPredictor, load_serving_thresholds
from src.observability import request_logger

router = APIRouter(tags=["Fraud Scoring"])

# Global singleton predictor instance
_predictor: FraudPredictor | None = None
_approve, _decline = load_serving_thresholds(get_settings().metadata_path)
_active_thresholds = ThresholdConfig(approve_threshold=_approve, decline_threshold=_decline)


def get_predictor() -> FraudPredictor:
    """Retrieve or initialize singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor(
            approve_threshold=_active_thresholds.approve_threshold,
            decline_threshold=_active_thresholds.decline_threshold,
        )
    return _predictor


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score one transaction (SHAP adds latency)",
    dependencies=[Depends(require_api_key)],
)
def predict_transaction(
    payload: TransactionPayload,
    include_shap: bool = Query(True, description="Whether to include local SHAP attribution"),
) -> PredictionResponse:
    """Score an incoming payment transaction in real time."""
    start_time = time.perf_counter()
    predictor = get_predictor()

    try:
        data = payload.model_dump()
        result = predictor.predict_single(
            data,
            include_shap=include_shap,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        explanation_payload = None
        if include_shap and "explanation" in result:
            exp = result["explanation"]
            explanation_payload = ExplanationPayload(
                base_value=exp["base_value"],
                top_risk_factors=[RiskFactor(**rf) for rf in exp.get("top_risk_factors", [])],
                top_protective_factors=[
                    RiskFactor(**pf) for pf in exp.get("top_protective_factors", [])
                ],
            )

        return PredictionResponse(
            transaction_id=result["transaction_id"],
            fraud_probability=result["fraud_probability"],
            decision=result["decision"],
            risk_tier=result["risk_tier"],
            threshold_used=result["threshold_used"],
            latency_ms=round(elapsed_ms, 2),
            explanation=explanation_payload,
        )
    except Exception:
        # Log the traceback once; do not echo paths or model internals to callers.
        request_logger.exception("prediction_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Scoring failed"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a batch of transactions",
    dependencies=[Depends(require_api_key)],
)
def predict_batch(
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Score multiple transactions sequentially or in parallel."""
    predictor = get_predictor()
    results: List[PredictionResponse] = []
    approved = 0
    review = 0
    declined = 0
    total_latency = 0.0

    for tx in payload.transactions:
        t0 = time.perf_counter()
        res = predictor.predict_single(
            tx.model_dump(),
            include_shap=False,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        total_latency += latency

        decision = res["decision"]
        if decision == "APPROVE":
            approved += 1
        elif decision == "MANUAL_REVIEW":
            review += 1
        else:
            declined += 1

        results.append(
            PredictionResponse(
                transaction_id=res["transaction_id"],
                fraud_probability=res["fraud_probability"],
                decision=decision,
                risk_tier=res["risk_tier"],
                threshold_used=res["threshold_used"],
                latency_ms=round(latency, 2),
                explanation=None,
            )
        )

    avg_latency = total_latency / max(1, len(payload.transactions))

    return BatchPredictionResponse(
        total_processed=len(payload.transactions),
        approved_count=approved,
        review_count=review,
        declined_count=declined,
        average_latency_ms=round(avg_latency, 2),
        results=results,
    )


@router.get("/threshold", response_model=ThresholdConfig, summary="Get active decision thresholds")
def get_thresholds() -> ThresholdConfig:
    return _active_thresholds


@router.post(
    "/threshold",
    response_model=ThresholdConfig,
    summary="Update decision thresholds dynamically",
    dependencies=[Depends(require_api_key)],
)
def set_thresholds(config: ThresholdConfig) -> ThresholdConfig:
    global _active_thresholds
    _active_thresholds = config
    predictor = get_predictor()
    predictor.approve_threshold = config.approve_threshold
    predictor.decline_threshold = config.decline_threshold
    return _active_thresholds
