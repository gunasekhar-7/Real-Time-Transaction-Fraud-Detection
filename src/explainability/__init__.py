"""Explainability module providing local TreeSHAP attribution for analyst triage."""

from .explainer import FraudShapExplainer, ExplanationResult

__all__ = ["FraudShapExplainer", "ExplanationResult"]
