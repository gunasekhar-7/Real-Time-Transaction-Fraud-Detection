"""WebSocket streaming feed and attack simulation triggers."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Set, List
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from src.api.schemas import AttackTriggerRequest
from src.api.security import require_api_key
from src.streaming.producer import TransactionStreamGenerator
from src.api.routes.predict import get_predictor

router = APIRouter(tags=["Streaming Simulation"])

# The demo queue is authenticated, but repeated bursts must not grow it forever.
MAX_ATTACK_QUEUE = 500
# The dashboard socket is public, so cap subscribers to bound work per process.
MAX_WS_CONNECTIONS = 100

# Active WebSocket subscribers
_active_connections: Set[WebSocket] = set()
_stream_generator = TransactionStreamGenerator(random_seed=42)
_attack_queue: List[dict] = []


@router.websocket("/ws/transactions")
async def websocket_transaction_stream(websocket: WebSocket) -> None:
    """Real-time transaction stream WebSocket for the FinTech Operator Dashboard."""
    await websocket.accept()
    if len(_active_connections) >= MAX_WS_CONNECTIONS:
        await websocket.close(code=1013)
        return
    _active_connections.add(websocket)
    predictor = get_predictor()

    try:
        while True:
            # Check if an attack burst has queued items; otherwise generate standard traffic
            if _attack_queue:
                tx_event = _attack_queue.pop(0)
            else:
                # 1.5% random fraud rate in regular stream
                is_fraud = _stream_generator.rng.random() < 0.015
                tx_event = _stream_generator.generate_single_event(is_fraud=is_fraud)

            # Score transaction in real time with SHAP explainability
            t0 = time.perf_counter()
            score_result = predictor.predict_single(
                tx_event,
                include_shap=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            scored_event = {
                "transaction_id": tx_event["transaction_id"],
                "user_id": f"card_{tx_event.get('card1')}",
                "Time": tx_event["TransactionDT"],
                "Amount": tx_event["TransactionAmt"],
                "fraud_probability": score_result["fraud_probability"],
                "decision": score_result["decision"],
                "risk_tier": score_result["risk_tier"],
                "threshold_used": score_result["threshold_used"],
                "latency_ms": round(elapsed_ms, 2),
                "explanation": score_result.get("explanation"),
                "raw_payload": tx_event,
            }

            await websocket.send_text(json.dumps(scored_event))

            # Pace the live stream to 1.5 seconds per event for human dashboard readability
            await asyncio.sleep(1.2)

    except WebSocketDisconnect:
        _active_connections.discard(websocket)
    except Exception:
        _active_connections.discard(websocket)


@router.post(
    "/stream/attack",
    summary="Trigger a coordinated fraud attack burst",
    dependencies=[Depends(require_api_key)],
)
async def trigger_attack(payload: AttackTriggerRequest) -> dict:
    """Inject a burst of attack transactions into the live streaming pipeline."""
    if len(_attack_queue) + payload.num_events > MAX_ATTACK_QUEUE:
        raise HTTPException(
            status_code=429,
            detail="Attack queue is full; let the live feed drain before queuing more events.",
        )
    events = _stream_generator.generate_scenario_burst(
        scenario=payload.scenario,
        num_events=payload.num_events,
    )
    _attack_queue.extend(events)
    return {
        "status": "ATTACK_TRIGGERED",
        "scenario": payload.scenario,
        "events_queued": len(events),
        "message": f"Successfully injected {len(events)} transactions simulating '{payload.scenario}'.",
    }
