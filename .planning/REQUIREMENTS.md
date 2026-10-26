# Project Requirements

## Core Functional Requirements

1. **Data ingestion and benchmark**
   - Load the IEEE-CIS `train_transaction` and `train_identity` tables, join on
     `TransactionID`, and cache the result locally.
   - Use the real class balance (3.499% fraud) and a chronological split (last
     20% of `TransactionDT`); do not resample or shuffle.
   - Key account history on the `card1|addr1` proxy because the dataset has no
     customer identifier.
2. **Causal feature state**
   - Compute per-account velocity online: 1h/24h counts, 24h spend, seconds since
     the previous event, amount relative to account history, and days since the
     first event.
   - Bound account state with LRU eviction and a per-account event cap; keep batch
     and single-row encoding on the same vocabulary.
3. **Modeling and evaluation**
   - Compare cost-sensitive LightGBM and XGBoost against an Isolation Forest
     baseline.
   - Select the champion by PR-AUC; record the margin, ROC-AUC, top-1%-volume
     capture and recall at 90% precision.
   - Report Brier score, log loss, expected calibration error and a reliability
     table. Do not add SMOTE unless it is evaluated, not assumed.
4. **Explainability**
   - Produce local TreeSHAP risk and protective factors with decoded categorical
     values and `null` for missing values.
   - Describe these as analyst triage reasons; do not claim regulatory
     certification or FCRA compliance.
5. **Low-latency serving**
   - FastAPI endpoints for single and batch scoring (batch cap 500), health,
     metrics, thresholds and Prometheus text.
   - Keep scoring p95 under 25 ms; treat SHAP as an opt-in cost with its own
     latency bound.
   - Return `503` from `/health` when the model is not loaded.
6. **Streaming simulation**
   - Replay held-out IEEE-CIS rows and generate card-testing and account-takeover
     bursts over `/ws/transactions`.
   - Label the feed as simulated and keep mutation routes authenticated.
7. **Operator dashboard**
   - Show live decisions, flagged share/value, the SHAP inspector, the real PR
     curve and the offline cost curve from `metadata.json`.
   - Support keyboard navigation and screen readers; never invent data when
     metadata is missing.
8. **Delivery and portfolio**
   - Non-root Docker image, healthcheck and production fail-fast for missing
     `API_KEY` or wildcard CORS.
   - CI with ruff lint, format check, `node --check` and pytest.
   - Load test with a p95 gate, a README generated from measured artifacts, and
     resume bullets that match those numbers.
