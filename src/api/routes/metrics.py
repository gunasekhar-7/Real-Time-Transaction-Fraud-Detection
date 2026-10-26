"""Metrics routes serving model benchmarks, PR curves, and cost-utility data."""

from __future__ import annotations

import json
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from src.config import get_settings

router = APIRouter(tags=["Metrics & Benchmarks"])


@router.get(
    "/metrics",
    summary="Get model evaluation metrics, benchmark comparisons, and threshold curve",
)
def get_model_metrics() -> Dict[str, Any]:
    """Retrieve full performance metrics and PR-AUC trade-off curves."""
    metadata_path = get_settings().metadata_path
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not found. Please train models first.",
        )
    with open(metadata_path, "r") as f:
        data = json.load(f)
    return data
