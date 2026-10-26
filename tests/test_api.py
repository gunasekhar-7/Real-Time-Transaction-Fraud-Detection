"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def _payload(**overrides):
    payload = {
        "card1": 9500,
        "addr1": 299,
        "TransactionDT": 450.0,
        "TransactionAmt": 125.0,
        "ProductCD": "W",
        "features": {"V14": -5.5, "C1": 1.0},
    }
    payload.update(overrides)
    return payload


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["champion_algorithm"] == "LightGBM"
    assert data["sla_target_p95_ms"] == 25.0


def test_predict_single_endpoint():
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["decision"] in ["APPROVE", "MANUAL_REVIEW", "DECLINE"]
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "latency_ms" in data
    assert data["explanation"] is not None
    factors = (
        data["explanation"]["top_risk_factors"] + data["explanation"]["top_protective_factors"]
    )
    assert factors
    assert all(factor["feature_value_display"] for factor in factors)


def test_predict_batch_endpoint():
    payload = {
        "transactions": [
            _payload(card1=1, TransactionAmt=15.0),
            _payload(card1=2, TransactionAmt=2500.0),
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_threshold_get_and_post():
    get_res = client.get("/threshold")
    assert get_res.status_code == 200
    curr = get_res.json()
    assert "approve_threshold" in curr

    post_res = client.post(
        "/threshold", json={"approve_threshold": 0.25, "decline_threshold": 0.65}
    )
    assert post_res.status_code == 200
    updated = post_res.json()
    assert updated["approve_threshold"] == 0.25
    assert updated["decline_threshold"] == 0.65


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "pr_auc" in data
    assert "roc_auc" in data
    assert "benchmark_comparison" in data
    assert "threshold_curve" in data
    assert "pr_curve" in data
    assert data["operations"]["serving_thresholds"]["decline"] == data["optimal_threshold"]


def test_predict_error_does_not_echo_internals(monkeypatch):
    from src.api.routes.predict import get_predictor

    predictor = get_predictor()

    def fail(*_args, **_kwargs):
        raise RuntimeError("secret /app/models path")

    monkeypatch.setattr(predictor, "predict_single", fail)
    response = client.post("/predict", json=_payload())

    assert response.status_code == 500
    assert response.json()["detail"] == "Scoring failed"
    assert "secret" not in response.text


def test_attack_queue_is_bounded():
    from src.api.routes import stream as stream_routes

    stream_routes._attack_queue.clear()
    try:
        burst = {"scenario": "card_testing", "num_events": 200}
        assert client.post("/stream/attack", json=burst).status_code == 200
        assert client.post("/stream/attack", json=burst).status_code == 200
        assert client.post("/stream/attack", json=burst).status_code == 429
    finally:
        stream_routes._attack_queue.clear()
