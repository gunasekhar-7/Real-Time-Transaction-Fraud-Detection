"""FinTech performance metrics: PR-AUC, Precision-Recall curves, Recall@K, calibration."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def top_k_capture(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    fraction: float = 0.01,
) -> Dict[str, float]:
    """Fraud caught by reviewing exactly the top `fraction` of rows by score.

    Rank-based, not percentile-based: a percentile cut uses ``>=`` and pulls in
    every tied score, so a model with repeated values can "review" far more than
    the promised share of traffic.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_prob, dtype=float)
    count = max(1, int(np.ceil(len(labels) * fraction)))
    reviewed = labels[np.argsort(-scores, kind="stable")[:count]]

    total_frauds = int(labels.sum())
    caught = int(reviewed.sum())
    return {
        "reviewed": count,
        "caught": caught,
        "recall": float(caught / total_frauds) if total_frauds else 0.0,
        "precision": float(caught / count),
    }


def calibration_report(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    bins: int = 10,
) -> Dict[str, Any]:
    """Brier, log loss, expected calibration error and the reliability table."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)

    table: List[Dict[str, Any]] = []
    ece = 0.0
    for bucket in range(bins):
        mask = index == bucket
        if not mask.any():
            continue
        mean_predicted = float(scores[mask].mean())
        observed = float(labels[mask].mean())
        share = float(mask.mean())
        ece += share * abs(observed - mean_predicted)
        table.append(
            {
                "bin_lower": round(float(edges[bucket]), 2),
                "bin_upper": round(float(edges[bucket + 1]), 2),
                "count": int(mask.sum()),
                "mean_predicted": round(mean_predicted, 4),
                "observed_rate": round(observed, 4),
            }
        )

    return {
        "brier": round(float(brier_score_loss(labels, scores)), 6),
        "log_loss": round(float(log_loss(labels, scores, labels=[0, 1])), 6),
        "expected_calibration_error": round(ece, 6),
        "exact_zero_share": round(float((scores == 0.0).mean()), 6),
        "exact_one_share": round(float((scores == 1.0).mean()), 6),
        "reliability": table,
    }


def calculate_evaluation_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics highlighting imbalanced credibility."""
    y_true_arr = np.array(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    # 1. Curve Integrals
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # 2. Confusion Matrix & Rates at threshold
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # False Discovery Rate (FDR) and False Alarm Ratio
    false_discovery_rate = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # 3. Recall @ Fixed Precision targets (80%, 90%)
    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    # Recall where precision >= 0.90
    high_prec_mask = precisions >= 0.90
    recall_at_90_precision = float(recalls[high_prec_mask].max()) if high_prec_mask.any() else 0.0

    # Precision where recall >= 0.80
    high_rec_mask = recalls >= 0.80
    precision_at_80_recall = float(precisions[high_rec_mask].max()) if high_rec_mask.any() else 0.0

    # 4. Top-K% capture (rank-based, so ties cannot inflate the review queue)
    top_1_pct = top_k_capture(y_true_arr, y_prob_arr, fraction=0.01)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_discovery_rate": round(false_discovery_rate, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "recall_at_90_precision": round(recall_at_90_precision, 4),
        "precision_at_80_recall": round(precision_at_80_recall, 4),
        "recall_at_top_1_pct_volume": round(top_1_pct["recall"], 4),
        "precision_at_top_1_pct_volume": round(top_1_pct["precision"], 4),
    }


def compute_pr_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute downsampled PR curve points for web visualization."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Sample 50 evenly spaced points for fast JSON transmission
    indices = np.linspace(0, len(thresholds) - 1, min(50, len(thresholds)), dtype=int)
    return {
        "precision": [round(float(precisions[i]), 4) for i in indices],
        "recall": [round(float(recalls[i]), 4) for i in indices],
        "thresholds": [round(float(thresholds[i]), 4) for i in indices],
    }
