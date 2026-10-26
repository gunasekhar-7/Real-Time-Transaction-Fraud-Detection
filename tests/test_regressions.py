"""Regression tests for the decision-path bugs found in the 2026-10 review."""

import json
import pickle
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.main import app
from src.api.schemas import ThresholdConfig
from src.features.pipeline import FeaturePipeline
from src.models.baseline import IsolationForestBaseline
from src.models.predictor import FALLBACK_THRESHOLDS, load_serving_thresholds
from src.models.train import save_pipeline

client = TestClient(app)


def test_save_pipeline_clears_account_state(ieee_frame, tmp_path):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(ieee_frame)
    assert pipeline.aggregator._history

    path = str(tmp_path / "pipeline.pkl")
    save_pipeline(pipeline, path)
    with open(path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded.aggregator._history == {}


def test_shipped_pipeline_has_no_account_state():
    with open("models/feature_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    assert pipeline.aggregator._history == {}


def test_threshold_config_rejects_inverted_order():
    with pytest.raises(ValueError):
        ThresholdConfig(approve_threshold=0.9, decline_threshold=0.1)


def test_threshold_endpoint_rejects_inverted_order():
    res = client.post("/threshold", json={"approve_threshold": 0.9, "decline_threshold": 0.1})
    assert res.status_code == 422


def test_serving_thresholds_come_from_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"optimal_threshold": 0.15}))

    assert load_serving_thresholds(str(path)) == (0.075, 0.15)


def test_serving_thresholds_fall_back_on_missing_or_nonsense_metadata(tmp_path):
    assert load_serving_thresholds(str(tmp_path / "missing.json")) == FALLBACK_THRESHOLDS

    path = tmp_path / "metadata.json"
    for bad in ({"optimal_threshold": 1.5}, {"optimal_threshold": "high"}, {}):
        path.write_text(json.dumps(bad))
        assert load_serving_thresholds(str(path)) == FALLBACK_THRESHOLDS


def test_repeat_transaction_is_scored_once():
    payload = {
        "transaction_id": f"tx_{uuid.uuid4().hex}",
        "card1": 7777.0,
        "addr1": 55.0,
        "TransactionDT": 500.0,
        "TransactionAmt": 250.0,
        "features": {"V14": -3.0},
    }
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()

    assert first["fraud_probability"] == second["fraud_probability"]
    assert first["decision"] == second["decision"]
    assert first["explanation"] == second["explanation"]


def test_batch_endpoint_rejects_oversized_request():
    txs = [{"card1": 1.0, "TransactionAmt": 1.0}] * 501
    res = client.post("/predict/batch", json={"transactions": txs})
    assert res.status_code == 422


def test_health_returns_503_without_model(monkeypatch):
    class NotReady:
        is_ready = False

    import src.api.main as main

    monkeypatch.setattr(main, "get_predictor", lambda: NotReady())
    res = TestClient(main.app).get("/health")
    assert res.status_code == 503


def test_stream_scores_with_live_thresholds():
    with client.websocket_connect("/ws/transactions") as ws:
        ws.receive_json()
        client.post("/threshold", json={"approve_threshold": 0.05, "decline_threshold": 0.99})
        event = ws.receive_json()

    assert event["threshold_used"] == 0.99
    json.dumps(event, allow_nan=False)  # the browser's JSON.parse rejects NaN


def test_stream_connection_cap(monkeypatch):
    from src.api.routes import stream as stream_routes

    stream_routes._active_connections.clear()
    monkeypatch.setattr(stream_routes, "MAX_WS_CONNECTIONS", 1)

    with client.websocket_connect("/ws/transactions") as ws:
        ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws/transactions") as second:
                second.receive_json()

    assert closed.value.code == 1013


def test_isolation_forest_scores_outliers_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(300, 3))
    model = IsolationForestBaseline(random_state=0).fit(normal)

    assert model.predict_anomaly_score(np.array([[12.0, 12.0, 12.0]]))[0] > float(
        model.predict_anomaly_score(normal).mean()
    )
