"""Latency benchmark verification: ensures scoring meets sub-25ms p95 SLA."""

import time
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

PAYLOAD = {
    "card1": 9500,
    "addr1": 299,
    "TransactionDT": 3600.0,
    "TransactionAmt": 149.99,
    "ProductCD": "W",
    "features": {"V14": -2.3, "C1": 1.0},
}


def test_latency_sla_under_25ms():
    """Verify that p95 scoring latency is strictly under 25ms SLA."""
    # Warmup query
    client.post("/predict?include_shap=false", json=PAYLOAD)

    server_latencies = []
    client_latencies = []
    # Execute 100 consecutive scoring calls without SHAP
    for _ in range(100):
        t0 = time.perf_counter()
        resp = client.post("/predict?include_shap=false", json=PAYLOAD)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        client_latencies.append((t1 - t0) * 1000.0)
        server_latencies.append(resp.json()["latency_ms"])

    p50_server = float(np.percentile(server_latencies, 50))
    p95_server = float(np.percentile(server_latencies, 95))
    p99_server = float(np.percentile(server_latencies, 99))

    p50_client = float(np.percentile(client_latencies, 50))
    p95_client = float(np.percentile(client_latencies, 95))

    print("\n[LATENCY SLA BENCHMARK - Server Scoring Engine]")
    print(f"  - p50: {p50_server:.2f}ms")
    print(f"  - p95: {p95_server:.2f}ms (Target: < 25.0ms SLA)")
    print(f"  - p99: {p99_server:.2f}ms")
    print(
        f"  - Client E2E (including test framework overhead): p50={p50_client:.2f}ms, p95={p95_client:.2f}ms"
    )

    # Assert sub-25ms server SLA requirement
    assert p95_server < 25.0, (
        f"Server scoring p95 latency was {p95_server:.2f}ms, which breaches the 25ms SLA!"
    )


def test_shap_latency_stays_interactive():
    """Explained requests pay TreeSHAP on top of scoring; keep p95 under 100ms."""
    client.post("/predict", json=PAYLOAD)  # warm

    latencies = []
    for _ in range(20):
        resp = client.post("/predict?include_shap=true", json=PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["explanation"] is not None
        latencies.append(resp.json()["latency_ms"])

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(f"\n[SHAP LATENCY] p50={p50:.2f}ms  p95={p95:.2f}ms")
    assert p95 < 100.0, f"SHAP p95 latency was {p95:.2f}ms"
