# Real-Time Transaction Fraud Detection

## Project Context

- **Domain**: FinTech / payment risk, ML engineering and model serving.
- **Objective**: A production-shaped fraud scoring service on IEEE-CIS
  transactions (590,540 rows, 3.499% fraud), with a chronological holdout,
  account-level velocity features, cost-based decisions, local explanations and
  an operator dashboard.
- **Scope**: Reference implementation and portfolio project. It is not a
  certified compliance system or a live production deployment; the README lists
  the operational limits.

## Key Tenets

- **Credibility over naive metrics**: select by PR-AUC on a time split, report
  calibration and dollar-weighted cost, and never quote accuracy on a 3.4%
  positive rate.
- **No leakage**: the last 20% of `TransactionDT` is the test set; velocity
  features are causal and the pipeline is fit on train only.
- **Service-shaped delivery**: sub-25 ms p95 scoring (SHAP is opt-in and
  measured separately), strict request validation, API-key auth, rate limits and
  bounded in-memory state.
- **Explainability with honest wording**: TreeSHAP produces human-readable
  analyst triage reasons. It is not certified for adverse-action notices.
- **Streaming simulation**: replay held-out rows through a WebSocket feed, with
  card-testing and account-takeover bursts for demonstration.
- **Operator visibility**: live decisions, a SHAP inspector, the real PR and cost
  curves, and dependency-free Prometheus metrics.

## Tech Stack

- **Language**: Python 3.11, vanilla JavaScript for the dashboard.
- **ML / data**: LightGBM, XGBoost, scikit-learn Isolation Forest, SHAP, NumPy,
  pandas.
- **Backend**: FastAPI, Uvicorn, Pydantic V2, WebSockets.
- **Frontend**: HTML, CSS and vanilla JS served by FastAPI.
- **Delivery**: Docker, Docker Compose, GitHub Actions, pytest, ruff.
