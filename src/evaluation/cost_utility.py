"""Financial Cost-Utility matrix and decision threshold optimizer.

Calculates the net business impact of classification decisions by balancing:
1. False Negatives (Missed Fraud): Direct charge loss + chargeback penalty + brand loss
2. False Positives (False Alarms): Operational review costs + customer friction / churn
3. True Positives (Intercepted Fraud): Intercepted value minus minor verification cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class CostMatrix:
    """Financial cost assumptions for fraud decisioning."""

    chargeback_fee: float = 25.0
    brand_damage_penalty: float = 10.0
    fp_investigation_cost: float = 5.0
    fp_customer_friction_cost: float = 15.0
    tp_review_cost: float = 1.50
    tn_cost: float = 0.0

    @property
    def cost_per_false_positive(self) -> float:
        """Total loss when a good customer's card is erroneously declined."""
        return self.fp_investigation_cost + self.fp_customer_friction_cost


class CostUtilityOptimizer:
    """Optimizes classification decision thresholds for net business financial profit/loss."""

    def __init__(self, cost_matrix: CostMatrix | None = None) -> None:
        self.costs = cost_matrix or CostMatrix()

    def evaluate_threshold(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray,
        amounts: np.ndarray | pd.Series,
        threshold: float,
    ) -> Dict[str, Any]:
        """Compute financial cost at a specific decision threshold."""
        y_true_arr = np.array(y_true, dtype=int)
        amounts_arr = np.array(amounts, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

        # Fraud dollars missed (False Negatives)
        fn_mask = (y_true_arr == 1) & (y_pred == 0)
        missed_fraud_dollar_loss = float(amounts_arr[fn_mask].sum())
        total_fn_cost = missed_fraud_dollar_loss + fn * (
            self.costs.chargeback_fee + self.costs.brand_damage_penalty
        )

        # Operational & friction loss from False Positives
        total_fp_cost = fp * self.costs.cost_per_false_positive

        # Verification operational overhead for True Positives
        total_tp_cost = tp * self.costs.tp_review_cost

        # Total expected business loss
        total_loss = total_fn_cost + total_fp_cost + total_tp_cost

        # Dollars of fraud prevented
        tp_mask = (y_true_arr == 1) & (y_pred == 1)
        fraud_dollars_prevented = float(amounts_arr[tp_mask].sum())

        return {
            "threshold": round(float(threshold), 3),
            "total_financial_loss": round(total_loss, 2),
            "fraud_dollars_prevented": round(fraud_dollars_prevented, 2),
            "missed_fraud_dollar_loss": round(missed_fraud_dollar_loss, 2),
            "false_positive_friction_cost": round(total_fp_cost, 2),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    def find_optimal_threshold(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray,
        amounts: np.ndarray | pd.Series,
        thresholds: np.ndarray | None = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Search threshold grid to locate threshold minimizing total financial loss."""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 46)

        results = []
        for th in thresholds:
            metrics = self.evaluate_threshold(y_true, y_prob, amounts, th)
            results.append(metrics)

        # Optimal threshold minimizes total_financial_loss
        best_result = min(results, key=lambda x: x["total_financial_loss"])

        # Also evaluate default 0.5 threshold for comparison
        default_result = self.evaluate_threshold(y_true, y_prob, amounts, 0.50)
        dollar_savings_over_default = (
            default_result["total_financial_loss"] - best_result["total_financial_loss"]
        )
        best_result["dollar_savings_over_default_05"] = round(dollar_savings_over_default, 2)

        return best_result, results
