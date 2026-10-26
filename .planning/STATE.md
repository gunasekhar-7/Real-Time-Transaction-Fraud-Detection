# Project State

## Status: COMPLETE (Stages 0–7)

The README, `RESUME_BULLETS.md`, dashboard labels and planning docs now report
the IEEE-CIS artifacts. The service, image and test suite were verified on
2026-10-25/26.

## Verified data (IEEE-CIS train)

- 590,540 transactions, 435 columns, 20,663 frauds (3.499%).
- 183 days, `TransactionDT` 86,400 -> 15,811,131.
- Identity joined for 24.4% of rows; keep missing values, do not fill.
- Account proxy `card1|addr1`: 39,974 groups (`card1` alone is issuer-level,
  one value covers up to 14,932 transactions).
- Time split: 472,432 train (3.514% fraud) / 118,108 test (3.441% fraud).

## Current model (Stage 2C retrain, time split)

| Rank | Model | PR-AUC | ROC-AUC | Recall @ top 1% | Precision @ top 1% |
|---|---|---|---|---|---|
| 1 | LightGBM | 0.5498 | 0.9030 | 0.257 | 0.883 |
| 2 | XGBoost | 0.4854 | 0.9016 | 0.247 | 0.851 |
| 3 | Isolation Forest | 0.0903 | 0.7223 | 0.003 | 0.010 |

- 130 features selected by gain from 242 candidates (23 categorical).
- Champion margin over runner-up: +0.0644 PR-AUC.
- Calibration: Brier 0.0223, log loss 0.0931, ECE 0.0109. The top decile
  (mean score 0.960) observes 0.893 fraud, so it is slightly overconfident;
  no score is exactly 0.0 or 1.0.
- Cost-optimal threshold 0.150; estimated $76k saved vs. 0.50 on the test set.
- Single-row scoring: 0.18 ms for features; explained requests measure p50 ~38 ms,
  p95 ~42 ms (22-43 ms across runs on this machine).

## Done

- Stage 0 guardrails: `.gitignore`, duplicate schema deleted, unused deps and
  dead functions removed, roadmap corrected.
- Stage 1 decision path: pipeline state cleared on save, threshold ordering
  enforced, stream reads live thresholds, repeat scoring is idempotent, batch
  capped at 500, `/health` returns 503 without a model.
- Stage 2A ingestion: `load_ieee` + `shrink` + `time_split`; cache at
  `data/processed/ieee.pkl` (979 MB, gitignored).
- Stage 2B features: account-keyed causal velocity, derived time features,
  gain-based feature budget, IEEE-CIS API payload, real-row demo stream.
- Stage 2C metrics: rank-based top-1% capture, calibration report (Brier/ECE/
  reliability), PR-AUC-ranked comparison table, champion-vs-base-rate assert.
- Stage 2D operations: serving thresholds from `metadata.json` (approve 0.075 /
  decline 0.150), `operations` block (6.7% review share, 67.5 reviews per 1,000,
  recall 0.656 at threshold, $363.7k prevented, $76.3k saved vs. 0.50),
  dashboard threshold sync, SHAP latency recorded in tests (p50 ~38 ms /
  p95 ~42 ms on the latest run).
- Stage 3 edge hardening: env-only settings (`src/config.py`), API-key auth on
  scoring/mutation routes, CORS allow-list with production fail-fast, per-client
  rate limit (600/min, 429 + `Retry-After`), stale YAML config deleted.
- Stage 4 state and observability: LRU-bounded account state (`MAX_ACCOUNTS`,
  `MAX_EVENTS_PER_ACCOUNT`) and dedupe cache gauge, JSON request logs, and a
  dependency-free Prometheus endpoint at `/prometheus`.
- Stage 5 dashboard: SHAP reasons decode categorical labels, champion and
  benchmark labels come from metadata, the cost simulator uses the real
  threshold curve, flagged KPIs are labelled honestly, and the dashboard has
  tab/feed/slider keyboard and screen-reader support.
- Stage 6 delivery guardrails: GitHub Actions runs ruff lint, ruff format check,
  `node --check` and pytest on Python 3.11. Dev tooling is isolated in
  `requirements-dev.txt`. `scripts/load_test.py` measured 300 requests at
  concurrency 20 on a clean process: server p50 2.2 ms, p95 3.5 ms, p99 5.3 ms;
  the 25 ms p95 gate passed. Throughput is harness-bound (a bare uvicorn route
  measures the same ~60 req/s ceiling), so only latency is quoted in the docs.
- Stage 7 delivery: hardened Dockerfile (non-root UID 10001, `libgomp1` only,
  `requirements-serving.txt`, Python healthcheck, one deliberate worker) and a
  `.dockerignore` that keeps data, tests and the licensed stream sample out of
  the image. Image `fraud-shield:stage7` is 1.06 GB; the container is healthy
  non-root, `/predict`, `/health` and `/prometheus` respond, and production
  fail-fast was confirmed for a missing key and wildcard CORS.
- Stage 7 docs: README and resume bullets regenerated from
  `models/metadata.json`; synthetic metrics, unsupported FCRA claims and the old
  test count removed. Dashboard screenshot saved at `docs/dashboard.png` with a
  runbook, security/config table, API table and honest limitations in the
  README.
- Live-dashboard bug fixed in Stage 7: SHAP emitted `NaN` for missing numeric
  features, which broke the browser's JSON parser. `feature_value` is nullable,
  non-finite values serialize as `null`, and regression tests assert strict JSON.
- Post-review hardening: `features` cannot override validated identity, amount or
  timing fields; feature count and scalar values are bounded; `/predict` no
  longer echoes internal exception text; the demo attack queue is capped at 500
  and the WebSocket feed at 100 subscribers; stale regulatory wording is gone
  from the OpenAPI description; CI builds the Docker image after tests. The suite
  grew to 66 tests.
- Rebuilt the image after hardening and ran it as non-root: container healthy,
  `/predict` returned 200, and a nested `TransactionAmt` in `features` returned
  422.
- Repo organization: `.gitattributes` normalizes LF and marks model/image files
  binary; `data/README.md` documents the local data contract; generated caches
  and the empty `notebooks/` folder were removed. `pyproject.toml` now matches the
  service version (1.2.0) and Python 3.11, and `/health` reads the champion label
  from `metadata.json` instead of hardcoding it.
- Tests: 66/66 pass; ruff check, ruff format check and `node --check` are clean.

## Next

No review stages remain. Optional backlog: shared feature store/customer key,
Redis-backed state and limits, drift monitoring, chargeback feedback, and a
decision on isotonic calibration for the top score band.

All work is currently uncommitted. CI imports the app, so a commit must include
`models/*.pkl` (except the gitignored `models/stream_sample.pkl`).

## Known issues

The demo WebSocket is unauthenticated (browsers cannot send headers on a WS
handshake); it is a simulation feed and mutations still require the key.

## Caveats

- The pickle cache is a local shortcut because pyarrow needs numpy >= 2 and the
  environment is pinned to 1.26. Revisit when the ML stack is upgraded.
- `models/stream_sample.pkl` is generated by training and gitignored: it is a
  slice of licensed data. Without it the stream falls back to a small template.
- Velocity features for a replayed row start from empty state, so they differ
  from what the account had at that moment in the dataset. A real deployment
  needs a feature store (Stage 4).
- The rate limiter and idempotency cache are in-process: with multiple workers
  each has its own budget and dedupe set. Move both to Redis when scaling out.
- Account state is bounded by LRU, not persisted: an evicted account restarts
  its velocity history. That is the deliberate trade against unbounded memory;
  a real deployment needs a shared feature store.
- Prometheus counters are per-process too. Scrape every worker (or use the
  multiprocess collector) before reading them as fleet-wide totals.
