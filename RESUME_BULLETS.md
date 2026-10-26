# Resume Bullet Points: Real-Time Transaction Fraud Detection

Metrics below are reproduced from `models/metadata.json` on **118,108 held-out
transactions** (the last 20% by `TransactionDT`). They describe an offline
benchmark and a portfolio service, not a production deployment. Keep the wording
that matches what you actually built.

---

## For Machine Learning Engineer / Applied AI Roles

- **Built an end-to-end fraud scoring service on 590,540 IEEE-CIS transactions**
  (3.499% fraud), covering ingestion, causal account-level velocity features,
  model selection, cost-based thresholds, explainability and a live dashboard.
- **Selected the champion by PR-AUC on a chronological split**: LightGBM reached
  **0.5498 PR-AUC / 0.9030 ROC-AUC**, beating XGBoost (0.4854) by **+0.0644** and
  an Isolation Forest baseline (0.0903) by **6.1×**.
- **Cut 242 candidate features to 130 by LightGBM gain**, including 23 categorical
  fields with a vocabulary shared between batch and single-row scoring, and
  verified that training and serving use the same transformations.
- **Validated calibration instead of assuming it**: Brier **0.0223**, log loss
  **0.0931**, expected calibration error **0.0109**, with a reliability table
  exposing the top decile's overconfidence.
- **Shipped TreeSHAP explanations** with decoded categorical labels and null-safe
  JSON, so analysts see "Card Type (credit)" or "missing" rather than opaque
  feature codes.

## For FinTech / Risk Analytics Roles

- **Replaced accuracy with decision-relevant metrics** on a 3.4%-fraud holdout:
  PR-AUC, recall at a fixed 1%-of-volume review budget (**25.7% recall at 88.3%
  precision**) and dollar-weighted cost.
- **Derived approve/review/decline bands from a cost-optimal threshold of 0.150**,
  producing an estimated **6.75% review share**, **33.5% precision** and **65.6%
  recall**, with **$76.3k lower modeled loss than the 0.50 default**.
- **Built an explicit cost-utility model** for missed fraud, chargebacks, brand
  damage, investigation and customer friction, then exposed the full threshold
  curve in the dashboard instead of hiding the trade-off.
- **Kept the analysis honest**: the dashboard labels flagged volume as "flagged,"
  not "prevented fraud," because the live stream has no fraud label.
- **Documented the business assumptions** so a risk team can replace the default
  costs with its own chargeback and review economics.

## For Backend / MLOps Roles

- **Served LightGBM behind FastAPI** at **3.5 ms p95 server-side scoring latency**
  (300 requests, concurrency 20), passing a 25 ms p95 gate; explanations are
  opt-in and measured separately (~42 ms p95).
- **Hardened a payment-shaped input path**: API-key auth, per-client rate limits
  with `Retry-After`, LRU-bounded account state, a 2,000-event per-account cap,
  idempotent retries and a 500-row batch cap.
- **Added production fail-fast behavior**: the service refuses to start in
  production without `API_KEY` or with wildcard CORS, and reports `503` when the
  model is not loaded.
- **Made the service observable** with JSON request logs using route templates and
  a dependency-free Prometheus endpoint for requests, latency, decisions, cache,
  SHAP and state gauges.
- **Containerized with a non-root runtime image** (1.06 GB, UID 10001,
  runtime-only dependencies) plus healthchecks, and added CI running ruff lint,
  format check, `node --check`, **66 passing tests** and a Docker build.

---

## Interview Talking Points

### 1. "Why not optimize accuracy?"

> "With 3.4% fraud, predicting 'legitimate' every time is 96.6% accurate and
> useless. I optimize PR-AUC for ranking and then choose a threshold from an
> explicit cost model. At the 1%-of-volume review budget, the champion catches
> 25.7% of fraud and 88.3% of reviewed cases are fraud."

### 2. "How did you avoid leakage?"

> "The split is the last 20% of transactions by `TransactionDT`, not random. The
> velocity features are causal — an event only sees earlier events for the same
> account — and the pipeline is fit on train only. The feature pipeline artifact
> is shared by training and serving so the transformations cannot drift apart."

### 3. "How did you handle class imbalance?"

> "I compared cost-sensitive LightGBM and XGBoost against an unsupervised
> Isolation Forest baseline, selected by PR-AUC, and checked calibration rather
> than relying on ROC-AUC. I did not add SMOTE; with high-dimensional,
> mostly-missing IEEE-CIS columns, synthetic positives were harder to justify
> than reweighting and threshold selection."

### 4. "How fast is it?"

> "Scoring without explanations is 3.5 ms p95 server-side at concurrency 20,
> measured over 300 requests with a 25 ms gate. TreeSHAP adds roughly 25–35 ms, so
> it is optional. Throughput from the bundled client is harness-bound, so the
> latency gate is the number I stand behind. The dashboard's 'Avg Explained
> Latency' is end-to-end and includes SHAP; it is not the scoring SLA."

### 5. "Are those explanations FCRA-compliant adverse-action reasons?"

> "No. They are local TreeSHAP attributions with human-readable feature values,
> built for analyst triage. Calling them compliant adverse-action notices would
> require legal review, governed reason codes, and validation against the
> specific regulation. I deliberately avoided that claim."

### 6. "What is the biggest limitation?"

> "The account key is `card1|addr1`, which is a proxy — it can merge a household
> or split a customer who moves. Replayed rows also start with empty velocity
> state, so their features differ from what the account had at that moment. The
> next step is a real feature store keyed on customer identity, not a bigger
> model."

### 7. "What would you do before calling it production-ready?"

> "Persist account state and rate limits in Redis, add drift monitoring and a
> retraining loop fed by chargebacks, calibrate the top score band, and replace
> the in-process Prometheus registry with a fleet-wide collector. I would also
> load-test with realistic burst traffic rather than a single repeated payload."

---

## Claims To Avoid

These appeared in an earlier draft and are not supported by the current
artifacts:

- "0.17% fraud rate" — the real dataset is 3.499% fraud.
- "PR-AUC 0.68", "ROC-AUC 0.9996", "100% fraud capture" — synthetic prototype.
- "7.8 ms p95" — current scoring p95 is 3.5 ms; explained p95 is ~42 ms.
- "12/12 tests" - the suite now has 66 tests.
- "FCRA compliant" or "regulatory-grade reason codes" — not established.
- "Prevented $363k" — that is an offline estimate of fraud dollars at the chosen
  threshold, not realized savings.
- "Production deployment" — the repository is a reference implementation with
  documented single-process limits.
