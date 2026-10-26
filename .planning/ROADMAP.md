# Project Roadmap

Status: complete. Stages 0–7 are done and the published metrics come from the
IEEE-CIS artifacts (see `STATE.md`).

## Done

### Stage 0 — Guardrails
- [x] `.gitignore` for data, venv, caches and secrets.
- [x] Delete the duplicate schema (`src/data/validator.py`), unused dependencies
      (`imbalanced-learn`, `websockets`) and dead functions.
- [x] Correct this roadmap; the old phases claimed SMOTE, global SHAP plots and
      CI/CD that were never built.

### Stage 2A — IEEE-CIS ingestion
- [x] `src/data/ieee.py`: join identity, shrink dtypes, local cache.
- [x] `src/data/schema.py`: column contract and household proxy (`card1|addr1`).
- [x] Time-based split (`time_split`), tests in `tests/test_ieee.py`.
- [x] Verified on the real data: 590,540 rows x 435 cols, 3.499% fraud,
      39,974 accounts, 183 days, 24.4% identity coverage.

### Stage 1 — Decision path fixes
- [x] Pipeline: fit on train only, transform test, clear state before saving
      (`save_pipeline`); the shipped artifact was cleaned too.
- [x] Threshold ordering validator; the predictor is the single source of truth;
      the stream reads live values.
- [x] Idempotent scoring for a repeated `(account, transaction_id)`.
- [x] Batch cap (500) and `/health` 503 without a model.
- [x] Regression tests in `tests/test_regressions.py`.

### Stage 2B — Features
- [x] Causal velocity keyed on `account_id` (`card1|addr1`): 1h/24h counts,
      24h spend, seconds since previous, amount vs. account mean, prior count,
      days since first transaction.
- [x] Raw `TransactionDT` dropped; `hour_of_day`, `day_of_week` and
      `days_since_first` derived instead.
- [x] 130 of 242 candidates kept by LightGBM gain; 23 categoricals code-encoded
      with one vocabulary shared by batch and single-row scoring.
- [x] Champion chosen by PR-AUC instead of being hardcoded.
- [x] Isolation Forest direction fixed (it was scoring normal traffic as risky).
- [x] API payload reshaped to the IEEE-CIS schema; the demo stream replays
      held-out rows instead of inventing PCA columns.

### Stage 2C — Honest metrics
- [x] Rank-based top-1% capture (`top_k_capture`); a percentile cut pulled in
      every tied score, so the review queue was unbounded.
- [x] Calibration report: Brier, log loss, expected calibration error and a
      10-bin reliability table, stored in `metadata.json`.
- [x] Comparison table ranked by PR-AUC; champion margin recorded.
- [x] Assertion that the champion beats the base rate, so an inverted score
      cannot ship silently.

### Stage 2D — Cost and latency
- [x] Serving thresholds load from `metadata.json`: decline = cost-optimal
      0.150, approve = 0.075; missing or nonsense metadata falls back to
      0.35/0.70 instead of failing startup.
- [x] `operations` block in `metadata.json`: review share, precision/recall at
      the threshold, reviews per 1,000, dollars prevented, missed dollars, the
      cost assumptions and the savings over 0.50.
- [x] Dashboard fetches `/threshold` on load and posts a matching approval
      band, so the slider and the enforced decision never disagree.
- [x] SHAP latency measured in the test suite: p50 ~38 ms, p95 ~42 ms on the
      latest run (100 ms bound; 22-43 ms across runs).

### Stage 3 — Edge hardening
- [x] `src/config.py`: environment variables are the single runtime config
      source. Stale synthetic `config/*.yaml` deleted (nothing read them).
- [x] Production fails fast without `API_KEY`, and rejects wildcard CORS.
- [x] `X-API-Key` required on scoring and mutation routes when a key is set;
      read-only routes and the demo WebSocket stay public.
- [x] CORS allow-list, no credentials, `X-API-Key` allowed; verified preflight.
- [x] Per-client sliding-window rate limit (600/min default) returning 429 with
      `Retry-After`; deterministic limiter tests.
- [x] Dashboard sends the key on control requests and prompts once on 401.

### Stage 4 — State and observability
- [x] Account state bounded twice, because a payment stream is attacker-shaped
      input: least-recently-seen accounts evicted past `MAX_ACCOUNTS` (50,000)
      and each account's 24h history capped at 2,000 events (real-data maximum:
      174).
- [x] Idempotency cache stays LRU-bounded by `PREDICTOR_CACHE_SIZE` (10,000) and
      is exported as a gauge.
- [x] JSON request logs: method, route template, status, duration, client. Route
      templates, not raw URLs, keep cardinality and log volume bounded.
- [x] Dependency-free Prometheus registry at `/prometheus`: request counter and
      latency histogram, decision/cache/SHAP counters, model and state gauges.

### Stage 5 - Dashboard correctness
- [x] SHAP reasons decode the pipeline vocabulary: "Browser (safari)", not
      "Browser (52.0)". Numeric values keep readable formatting and missing
      values read as "missing".
- [x] Champion badge, benchmark subtitle and PR operating point come from
      `metadata.json`; the stale "PR-AUC 0.68" and "0.172% test set" labels are
      gone.
- [x] Cost simulator shows the real `threshold_curve` dollars and says
      "unavailable" instead of inventing a synthetic loss curve.
- [x] KPIs renamed to flagged share/value; the stream has no fraud label, so the
      dashboard no longer calls flagged volume "prevented fraud".
- [x] Accessibility: tab roles and arrow-key navigation, slider label and
      value text, live feed as an ARIA log, keyboard-selectable rows, visible
      focus styles, labelled PR chart.
- [x] Tests for categorical decoding and API display values; `node --check` on
      `frontend/app.js`.

### Stage 6 - Delivery guardrails
- [x] GitHub Actions CI on push/PR: Python 3.11, `ruff check`,
      `ruff format --check`, `node --check frontend/app.js`, then `pytest -q`.
- [x] Dev-only tooling in `requirements-dev.txt`; the runtime requirements stay
      unchanged. The existing Python sources were formatted once with ruff.
- [x] `scripts/load_test.py`: warmup, unique transactions over 50 accounts, a
      bounded async client, server-side and round-trip percentiles, and an
      optional `--max-p95` gate.
- [x] Verified 300 requests at concurrency 20: server p50 2.2 ms, p95 3.5 ms,
      p99 5.3 ms; the 25 ms p95 gate passed. Throughput is not quoted because the
      bundled client caps near 60 req/s against any endpoint on one laptop and
      would measure the harness, not the service.

### Stage 7 - Delivery and narrative
- [x] Docker image builds and runs with the production env contract
      (`APP_ENV=production`, `API_KEY`, explicit `ALLOWED_ORIGINS`). Verified:
      1.06 GB image, non-root UID 10001, healthy container, `/predict`,
      `/health` and `/prometheus` reachable, 401 without the key and 200 with it.
- [x] Production fail-fast verified in the image: missing `API_KEY` and wildcard
      CORS both exit with a clear error.
- [x] Regenerate README and `RESUME_BULLETS.md` from `models/metadata.json`;
      remove every synthetic metric and unsupported compliance claim.
- [x] Add the dashboard screenshot (`docs/dashboard.png`) and a short runbook
      (train, test, serve, load test, deploy, rollback).

### Post-review hardening
- [x] Reject reserved keys inside `features` (`transaction_id`, `TransactionDT`,
      `TransactionAmt`, `card1`, `addr1`) so a nested copy cannot bypass
      top-level validation or the account key; cap `features` at 500 scalar
      values and `transaction_id` at 128 characters.
- [x] Make explicit payload fields win over nested `features` in the pipeline as
      defense in depth.
- [x] Stop returning internal exception text from `/predict`; log the traceback
      and return a generic `Scoring failed` message.
- [x] Cap the demo attack queue at 500 events and the public WebSocket feed at
      100 subscribers.
- [x] Remove stale "regulatory" wording from the OpenAPI description and the
      explainability module.
- [x] Add a Docker build job to CI after tests.
- [x] Add regression tests for the input contract, error redaction, queue and
      socket bounds, and explicit-field precedence. Suite is now 66 tests.
- [x] Repo housekeeping: LF/binary `.gitattributes`, `data/README.md`,
      cache/empty-folder cleanup, package/API version alignment, and a
      metadata-driven champion label in `/health`.

### Review fixes (production hardening)
- [x] **Move TreeSHAP outside the global lock** — `FraudPredictor.predict_single`
      now releases the lock before calling the read-only explainer, so concurrent
      scoring requests are not blocked during the 10–20 ms SHAP computation.
- [x] **Single background producer for WebSocket feed** — replaced per-connection
      polling loops with one async producer that scores once and broadcasts to all
      subscriber queues. Opening multiple dashboard tabs no longer duplicates
      SHAP work or starves the event loop.
- [x] **Proxy-aware rate limiting** — `TRUSTED_PROXIES` env var (comma-separated
      IPs/CIDRs or `"*"`) enables `X-Forwarded-For` / `X-Real-IP` parsing so
      genuine clients are not pooled behind a load balancer IP.
- [x] **Monotonic virtual clock for stream replay** — `TransactionStreamGenerator`
      now increments a virtual `TransactionDT` (~10 s/event) instead of sampling
      raw test-set timestamps, preventing negative time deltas in the velocity
      aggregator.
- [x] **Route-labeled latency histogram** — `fraud_request_duration_seconds`
      now carries `route` labels so Prometheus scrapers can chart `/predict`
      latency independently from static assets and health checks.
- [x] Updated regression tests for the new stream connection cap and histogram
      labels. Suite remains at 66 tests.

Not required for the current scope; these are the next real-world steps.

- Replace the `card1|addr1` proxy with a customer key and a shared feature store;
  replayed rows currently start with empty velocity state.
- Move rate limits, idempotency, account state and Prometheus counters out of the
  process before running more than one worker.
- Add drift monitoring and a chargeback-fed retraining loop.
- Decide whether the top score band needs isotonic calibration after a larger
  validation window.
