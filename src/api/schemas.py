"""Pydantic V2 schemas for FastAPI request and response validation."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

FeatureValue = Union[float, int, str, None]
MAX_FEATURES = 500
RESERVED_FEATURE_KEYS = frozenset(
    {"transaction_id", "TransactionDT", "TransactionAmt", "card1", "addr1"}
)


class TransactionPayload(BaseModel):
    """Incoming card transaction, shaped after the IEEE-CIS schema.

    ``card1`` plus ``addr1`` form the household key that velocity features are
    computed against. Vendor columns that a gateway would not send directly
    (``C*``, ``D*``, ``M*``, ``V*``, ``id_*``) go in ``features``; missing
    entries are passed to the model as unknown, never imputed here.
    """

    transaction_id: Optional[str] = Field(
        default=None, max_length=128, description="Unique transaction ID"
    )
    TransactionDT: float = Field(0.0, ge=0.0, description="Seconds from the dataset epoch")
    TransactionAmt: float = Field(..., gt=0.0, description="Transaction amount")

    card1: float = Field(..., description="Card identifier")
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None

    ProductCD: Optional[str] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    features: Dict[str, FeatureValue] = Field(
        default_factory=dict,
        max_length=MAX_FEATURES,
        description="Remaining vendor columns (C*, D*, M*, V*, id_*); scalar values only",
    )

    @field_validator("features")
    @classmethod
    def _reject_reserved_keys(cls, value: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        # These fields drive validation, the account key or idempotency, so a
        # nested copy must not bypass the top-level checks.
        reserved = RESERVED_FEATURE_KEYS.intersection(value)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"features must not contain reserved keys: {names}")
        return value


class RiskFactor(BaseModel):
    """Individual SHAP risk factor explaining prediction."""

    feature: str
    feature_name: str
    feature_value: Optional[float] = Field(
        None, description="Numeric value; null when the source field was missing"
    )
    feature_value_display: Optional[str] = None
    shap_impact: float
    reason_code: str


class ExplanationPayload(BaseModel):
    """SHAP explanation breakdown."""

    base_value: float
    top_risk_factors: List[RiskFactor] = []
    top_protective_factors: List[RiskFactor] = []


class PredictionResponse(BaseModel):
    """Scoring result with risk tier, decision, and SHAP explanation."""

    transaction_id: str
    fraud_probability: float
    decision: str = Field(..., description="APPROVE | MANUAL_REVIEW | DECLINE")
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH")
    threshold_used: float
    latency_ms: float
    explanation: Optional[ExplanationPayload] = None


class BatchPredictionRequest(BaseModel):
    """Batch list of transactions to score."""

    transactions: List[TransactionPayload] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Summary of batch scoring."""

    total_processed: int
    approved_count: int
    review_count: int
    declined_count: int
    average_latency_ms: float
    results: List[PredictionResponse]


class ThresholdConfig(BaseModel):
    """Dynamic decision threshold configuration."""

    approve_threshold: float = Field(0.35, ge=0.01, le=0.99)
    decline_threshold: float = Field(0.70, ge=0.01, le=0.99)

    @model_validator(mode="after")
    def _ordered(self) -> "ThresholdConfig":
        if self.approve_threshold >= self.decline_threshold:
            raise ValueError("approve_threshold must be lower than decline_threshold")
        return self


class AttackTriggerRequest(BaseModel):
    """Trigger an attack scenario in the live streaming engine."""

    scenario: str = Field(
        "card_testing", description="card_testing | account_takeover | burst_fraud | normal_traffic"
    )
    num_events: int = Field(20, ge=1, le=200)
