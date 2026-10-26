"""Training pipeline for LightGBM, XGBoost, and a baseline on IEEE-CIS."""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import asdict
from typing import Any, Dict, List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
from xgboost import XGBClassifier

from src.data.ieee import load_ieee, time_split
from src.data.schema import AMOUNT_COL, ID_COL, TARGET
from src.evaluation.cost_utility import CostMatrix, CostUtilityOptimizer
from src.evaluation.metrics import (
    calculate_evaluation_metrics,
    calibration_report,
    compute_pr_curve,
)
from src.explainability.explainer import FraudShapExplainer
from src.features.pipeline import FeaturePipeline
from src.models.baseline import IsolationForestBaseline
from src.models.predictor import serving_thresholds


def save_pipeline(pipeline: FeaturePipeline, path: str) -> None:
    """Persist the pipeline with empty account history so serving starts clean."""
    pipeline.aggregator.clear()
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def _metrics(y_true: np.ndarray, probs: np.ndarray, name: str) -> Dict[str, Any]:
    metrics = calculate_evaluation_metrics(y_true, probs, threshold=0.5)
    metrics["model_name"] = name
    return metrics


def train_and_evaluate_all(
    output_dir: str = "models",
    test_frac: float = 0.20,
) -> Dict[str, Any]:
    """Train the baseline and supervised models, then save production artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("[STEP 1] Loading IEEE-CIS training data...")
    print("=" * 70)
    df = load_ieee()
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraud Count: {int(df[TARGET].sum()):,} ({df[TARGET].mean() * 100:.3f}%)")

    print("\n" + "=" * 70)
    print("[STEP 2] Time-based split and feature construction...")
    print("=" * 70)
    train_df, test_df = time_split(df, test_frac=test_frac)
    print(f"Train set: {len(train_df):,} | Test set: {len(test_df):,}")
    del df

    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(
        f"Features: {len(pipeline.feature_columns)} selected from "
        f"{len(pipeline.numeric_columns) + len(pipeline.categorical_columns)} candidates "
        f"({len(pipeline.model_categorical_columns)} categorical)"
    )

    test_amounts = test_df.loc[X_test.index, AMOUNT_COL].to_numpy(dtype=float)
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()
    categorical = pipeline.model_categorical_columns

    neg_count = int((y_train_arr == 0).sum())
    pos_count = int((y_train_arr == 1).sum())
    scale_pos_weight = float(neg_count / max(1, pos_count))
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

    models: Dict[str, Any] = {}
    benchmark_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("[MODEL 1] Training Unsupervised Baseline (Isolation Forest)...")
    print("=" * 70)
    iso_forest = IsolationForestBaseline(contamination=float(y_train_arr.mean()), random_state=42)
    iso_forest.fit(X_train)
    iso_scores = iso_forest.predict_anomaly_score(X_test)
    iso_metrics = _metrics(y_test_arr, iso_scores, "Isolation Forest (Unsupervised Baseline)")
    benchmark_results.append(iso_metrics)
    print(f"Baseline PR-AUC: {iso_metrics['pr_auc']:.4f} | ROC-AUC: {iso_metrics['roc_auc']:.4f}")

    print("\n" + "=" * 70)
    print("[MODEL 2] Training XGBoost Classifier...")
    print("=" * 70)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train_arr)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = _metrics(y_test_arr, xgb_probs, "XGBoost (Cost-Sensitive)")
    benchmark_results.append(xgb_metrics)
    models["XGBoost"] = (xgb_model, xgb_probs, xgb_metrics)
    print(f"XGBoost PR-AUC: {xgb_metrics['pr_auc']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

    print("\n" + "=" * 70)
    print("[MODEL 3] Training LightGBM Classifier...")
    print("=" * 70)
    lgbm_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        scale_pos_weight=float(np.sqrt(scale_pos_weight)),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    lgbm_model.fit(X_train, y_train_arr, categorical_feature=categorical)
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_metrics = _metrics(y_test_arr, lgbm_probs, "LightGBM (Cost-Sensitive)")
    benchmark_results.append(lgbm_metrics)
    models["LightGBM"] = (lgbm_model, lgbm_probs, lgbm_metrics)
    print(f"LightGBM PR-AUC: {lgbm_metrics['pr_auc']:.4f} | ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")
    print(f"Recall @ 90% Precision: {lgbm_metrics['recall_at_90_precision']:.4f}")
    print(f"Top 1% Volume Fraud Capture: {lgbm_metrics['recall_at_top_1_pct_volume'] * 100:.1f}%")

    # -------------------------------------------------------------
    # Champion: highest PR-AUC. Accuracy is meaningless at 3.5% positives,
    # and the review found the prototype hardcoded LightGBM instead.
    # -------------------------------------------------------------
    benchmark_results.sort(key=lambda row: row["pr_auc"], reverse=True)
    for rank, row in enumerate(benchmark_results, start=1):
        row["rank"] = rank

    champion_name = max(models, key=lambda name: models[name][2]["pr_auc"])
    champion_model, champion_probs, champion_metrics = models[champion_name]
    runner_up = benchmark_results[1]["pr_auc"] if len(benchmark_results) > 1 else 0.0
    champion_margin = round(float(champion_metrics["pr_auc"]) - float(runner_up), 4)
    base_rate = float(y_test_arr.mean())
    assert champion_metrics["pr_auc"] > base_rate, (
        f"Champion PR-AUC {champion_metrics['pr_auc']} does not beat the "
        f"{base_rate:.3f} base rate; the scores are probably inverted."
    )
    print(
        f"\n>>> Champion: {champion_name} "
        f"(PR-AUC: {champion_metrics['pr_auc']:.4f} | ROC-AUC: {champion_metrics['roc_auc']:.4f} "
        f"| margin over runner-up: {champion_margin:+.4f})"
    )

    calibration = calibration_report(y_test_arr, champion_probs)
    print(
        f"Calibration: Brier {calibration['brier']:.4f} | "
        f"ECE {calibration['expected_calibration_error']:.4f} | "
        f"scored exactly 0.0: {calibration['exact_zero_share'] * 100:.1f}%"
    )

    print("\n" + "=" * 70)
    print("[STEP 4] Optimizing Decision Threshold via Financial Cost-Utility...")
    print("=" * 70)
    cost_opt = CostUtilityOptimizer(CostMatrix())
    best_threshold_metrics, threshold_curve = cost_opt.find_optimal_threshold(
        y_true=y_test_arr,
        y_prob=champion_probs,
        amounts=test_amounts,
    )
    optimal_th = best_threshold_metrics["threshold"]
    default_cost = cost_opt.evaluate_threshold(y_test_arr, champion_probs, test_amounts, 0.50)[
        "total_financial_loss"
    ]
    print(f"Default Threshold: 0.50 -> Loss: ${default_cost:,.2f}")
    print(
        f"Optimal Threshold: {optimal_th:.3f} -> Loss: ${best_threshold_metrics['total_financial_loss']:,.2f}"
    )
    print(
        f"Estimated Dollar Savings over Default: ${best_threshold_metrics['dollar_savings_over_default_05']:,.2f}"
    )

    reviewed = int(
        best_threshold_metrics["true_positives"] + best_threshold_metrics["false_positives"]
    )
    positives = int(
        best_threshold_metrics["true_positives"] + best_threshold_metrics["false_negatives"]
    )
    approve_th, _ = serving_thresholds(optimal_th)
    operations = {
        "serving_thresholds": {"approve": approve_th, "decline": optimal_th},
        "review_share_of_traffic": round(reviewed / len(y_test_arr), 6),
        "reviews_per_1000_transactions": round(reviewed / len(y_test_arr) * 1000, 1),
        "precision_at_threshold": round(best_threshold_metrics["true_positives"] / reviewed, 4)
        if reviewed
        else 0.0,
        "recall_at_threshold": round(best_threshold_metrics["true_positives"] / positives, 4)
        if positives
        else 0.0,
        "fraud_dollars_prevented": best_threshold_metrics["fraud_dollars_prevented"],
        "missed_fraud_dollars": best_threshold_metrics["missed_fraud_dollar_loss"],
        "false_positive_cost": best_threshold_metrics["false_positive_friction_cost"],
        "total_financial_loss": best_threshold_metrics["total_financial_loss"],
        "dollar_savings_over_default": best_threshold_metrics["dollar_savings_over_default_05"],
        "cost_assumptions": asdict(cost_opt.costs),
    }
    print(
        f"Serving thresholds: approve < {approve_th:.3f} | review < {optimal_th:.3f} | "
        f"{operations['reviews_per_1000_transactions']:.1f} reviews per 1,000 transactions"
    )

    print("\n" + "=" * 70)
    print("[STEP 5] Initializing TreeSHAP Explainer...")
    print("=" * 70)
    shap_explainer = FraudShapExplainer(
        model=champion_model, feature_names=pipeline.feature_columns
    )
    shap_explainer.attach_categories(pipeline.categories)
    sample_exp = shap_explainer.explain_transaction(
        X_test.iloc[0].to_numpy(), fraud_prob=float(champion_probs[0])
    )
    print("Sample SHAP Top Risk Factors:")
    for rf in sample_exp.top_risk_factors[:3]:
        print(f"  - {rf['reason_code']}")

    print("\n" + "=" * 70)
    print("[STEP 6] Serializing Production Artifacts...")
    print("=" * 70)
    model_path = os.path.join(output_dir, "fraud_detector_lgbm.pkl")
    pipeline_path = os.path.join(output_dir, "feature_pipeline.pkl")
    explainer_path = os.path.join(output_dir, "shap_explainer.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    sample_path = os.path.join(output_dir, "stream_sample.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(champion_model, f)
    save_pipeline(pipeline, pipeline_path)
    with open(explainer_path, "wb") as f:
        pickle.dump(shap_explainer, f)
    # Demo stream replays real feature vectors (including real frauds) instead of
    # inventing 400 columns. Only the transaction id is dropped.
    test_df.drop(columns=[c for c in (ID_COL,) if c in test_df.columns]).sample(
        n=min(2000, len(test_df)), random_state=42
    ).to_pickle(sample_path)

    metadata = {
        "model_version": "2.1.0",
        "champion_algorithm": champion_name,
        "champion_margin_over_runner_up": champion_margin,
        "dataset": "IEEE-CIS Fraud Detection (train_transaction + train_identity)",
        "split_method": "time (last 20% of TransactionDT)",
        "dataset_size": len(train_df) + len(test_df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "fraud_rate": round(float(np.concatenate([y_train_arr, y_test_arr]).mean()), 6),
        "test_fraud_rate": round(float(y_test_arr.mean()), 6),
        "pr_auc": champion_metrics["pr_auc"],
        "roc_auc": champion_metrics["roc_auc"],
        "recall_at_top_1_pct_volume": champion_metrics["recall_at_top_1_pct_volume"],
        "precision_at_top_1_pct_volume": champion_metrics["precision_at_top_1_pct_volume"],
        "recall_at_90_precision": champion_metrics["recall_at_90_precision"],
        "calibration": calibration,
        "default_threshold": 0.5,
        "optimal_threshold": optimal_th,
        "dollar_savings_over_default": best_threshold_metrics["dollar_savings_over_default_05"],
        "operations": operations,
        "benchmark_comparison": benchmark_results,
        "threshold_curve": threshold_curve,
        "pr_curve": compute_pr_curve(y_test_arr, champion_probs),
        "feature_names": pipeline.feature_columns,
        "categorical_features": pipeline.model_categorical_columns,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved pipeline to: {pipeline_path}")
    print("\n[DONE] Training and Serialization Complete!")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_all()
