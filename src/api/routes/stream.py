"""WebSocket streaming feed and attack simulation triggers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Dict
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

# Active WebSocket subscriber queues (one per connection)
_subscriber_queues: Dict[WebSocket, asyncio.Queue] = {}
_stream_generator = TransactionStreamGenerator(random_seed=42)
_attack_queue: List[dict] = []

# Background producer task
_producer_task: asyncio.Task | None = None
_producer_lock = asyncio.Lock()

_logger = logging.getLogger("fraud.stream")


async def _broadcast_event(scored_event: dict) -> None:
    """Fan out a scored event to all connected clients."""
    disconnected = []
    # Iterate over a snapshot to avoid "dictionary changed size during iteration"
    # if a client connects/disconnects concurrently.
    for ws, queue in list(_subscriber_queues.items()):
        try:
            queue.put_nowait(scored_event)
        except asyncio.QueueFull:
            # Slow consumer: drop the oldest event to make room, then retry once.
            # If still full, mark for disconnect.
            try:
                queue.get_nowait()
                queue.put_nowait(scored_event)
            except asyncio.QueueFull:
                disconnected.append(ws)
    for ws in disconnected:
        _subscriber_queues.pop(ws, None)
        # Close the WebSocket explicitly so the client handler doesn't hang.
        asyncio.create_task(ws.close(code=1008, reason="Slow consumer"))


async def _producer_loop() -> None:
    """Single background loop that generates, scores, and broadcasts events."""
    predictor = get_predictor()
    try:
        while _subscriber_queues:
            # Check if an attack burst has queued items; otherwise generate standard traffic
            if _attack_queue:
                tx_event = _attack_queue.pop(0)
            else:
                # 1.5% random fraud rate in regular stream
                is_fraud = _stream_generator.rng.random() < 0.015
                tx_event = _stream_generator.generate_single_event(is_fraud=is_fraud)

            # Score transaction in real time with SHAP explainability.
            # Offload synchronous CPU-bound inference to a worker thread so we don't
            # block the async event loop (feature calc + TreeSHAP ~15-20ms).
            t0 = time.perf_counter()
            score_result = await asyncio.to_thread(
                predictor.predict_single,
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

            await _broadcast_event(scored_event)

            # Pace the live stream to ~1.5 seconds per event for human dashboard readability
            await asyncio.sleep(1.2)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        _logger.exception("producer_loop crashed: %s", exc)
    finally:
        global _producer_task
        _producer_task = None


async def _ensure_producer_started() -> None:
    """Start the background producer if there are subscribers and it's not running."""
    global _producer_task
    async with _producer_lock:
        if _subscriber_queues and (_producer_task is None or _producer_task.done()):
            _producer_task = asyncio.create_task(_producer_loop())


async def _maybe_stop_producer() -> None:
    """Stop the background producer if no subscribers remain."""
    global _producer_task
    async with _producer_lock:
        if not _subscriber_queues and _producer_task is not None:
            _producer_task.cancel()
            try:
                await _producer_task
            except asyncio.CancelledError:
                pass
            _producer_task = None
            # Clear any residual attack events so a future reconnection doesn't
            # replay the remainder of an old burst.
            _attack_queue.clear()


@router.websocket("/ws/transactions")
async def websocket_transaction_stream(websocket: WebSocket) -> None:
    """Real-time transaction stream WebSocket for the FinTech Operator Dashboard."""
    await websocket.accept()
    if len(_subscriber_queues) >= MAX_WS_CONNECTIONS:
        await websocket.close(code=1013)
        return

    # Create a per-connection queue with backpressure
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subscriber_queues[websocket] = queue

    await _ensure_producer_started()

    try:
        while True:
            scored_event = await queue.get()
            await websocket.send_text(json.dumps(scored_event))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _subscriber_queues.pop(websocket, None)
        await _maybe_stop_producer()


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
