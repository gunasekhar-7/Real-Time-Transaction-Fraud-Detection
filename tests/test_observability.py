"""Bounded account state and observability tests."""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.state_aggregator import SlidingWindowStateAggregator
from src.observability import (
    JsonFormatter,
    ObservabilityMiddleware,
    increment,
    observe,
    render_metrics,
    reset_metrics,
    set_gauge,
)


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset_metrics()
    yield
    reset_metrics()


def test_state_evicts_least_recently_seen_accounts():
    agg = SlidingWindowStateAggregator(max_accounts=2)
    agg.update_and_extract("a", 0.0, 10.0)
    agg.update_and_extract("b", 1.0, 10.0)
    agg.update_and_extract("a", 2.0, 10.0)  # refresh a
    agg.update_and_extract("c", 3.0, 10.0)  # evicts b

    assert set(agg._history) == {"a", "c"}
    assert "b" not in agg._stats
    assert "b" not in agg._recent_1h
    assert agg.state_size() == {"accounts": 2, "events": 3}


def test_per_account_history_is_capped():
    agg = SlidingWindowStateAggregator(max_events_per_account=2)
    for i in range(5):
        agg.update_and_extract("a", float(i), 1.0)

    assert agg.state_size()["events"] == 2
    features = agg.update_and_extract("a", 6.0, 1.0)
    assert features["acct_tx_count_24h"] == 2
    assert features["acct_tx_count_prior"] == 5.0


def test_state_event_count_tracks_window_pruning():
    agg = SlidingWindowStateAggregator(window_24h_sec=100.0)
    agg.update_and_extract("a", 0.0, 1.0)
    agg.update_and_extract("a", 50.0, 1.0)
    agg.update_and_extract("a", 200.0, 1.0)  # both earlier events age out

    assert agg.state_size() == {"accounts": 1, "events": 1}


def test_metrics_exposition_has_types_and_cumulative_buckets():
    increment("demo_total", {"route": "/x", "status": "200"})
    increment("demo_total", {"route": "/x", "status": "200"})
    set_gauge("demo_gauge", 3)
    observe("demo_seconds", 0.02)
    observe("demo_seconds", 0.5)

    text = render_metrics()

    assert "# TYPE demo_total counter" in text
    assert 'demo_total{route="/x",status="200"} 2' in text
    assert "# TYPE demo_gauge gauge" in text
    assert "demo_gauge 3" in text
    assert 'demo_seconds_bucket{le="0.005"} 0' in text
    assert 'demo_seconds_bucket{le="0.025"} 1' in text
    assert 'demo_seconds_bucket{le="0.5"} 2' in text
    assert 'demo_seconds_bucket{le="+Inf"} 2' in text
    assert "demo_seconds_count 2" in text


def test_middleware_records_route_template_not_raw_path():
    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)

    @app.get("/items/{item_id}")
    def read_item(item_id: int) -> dict:
        return {"item_id": item_id}

    client = TestClient(app)
    assert client.get("/items/7").status_code == 200
    assert client.get("/missing/123").status_code == 404

    text = render_metrics()
    assert 'fraud_requests_total{route="/items/{item_id}",status="200"} 1' in text
    assert 'fraud_requests_total{route="unmatched",status="404"} 1' in text
    assert "fraud_request_duration_seconds_count 2" in text
    assert "/missing/123" not in text


def test_json_formatter_emits_one_structured_line():
    record = logging.LogRecord("fraud.api", logging.INFO, __file__, 1, "request", None, None)
    record.event = {"status": 200, "duration_ms": 1.5}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "request"
    assert payload["level"] == "INFO"
    assert payload["status"] == 200


def test_prometheus_endpoint_serves_model_gauges():
    from src.api.main import app

    client = TestClient(app)
    first = client.get("/prometheus")
    second = client.get("/prometheus")

    assert first.status_code == 200
    assert "fraud_model_loaded 1" in second.text
    assert "fraud_accounts_tracked" in second.text
    # The scrape that records the first request shows up in the next body.
    assert "fraud_request_duration_seconds_bucket" in second.text
