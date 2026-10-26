"""HTTP load test for /predict.

Run the API first, then:
    python scripts/load_test.py --requests 300 --concurrency 20

The server-side ``latency_ms`` is reported separately from the client round trip
so network and JSON overhead do not hide the scoring cost. Exits non-zero when a
request fails or ``--max-p95`` is breached.
"""

import argparse
import asyncio
import os
import sys
import time

import httpx
import numpy as np

PAYLOAD = {
    "card1": 9500,
    "addr1": 299,
    "TransactionDT": 3600.0,
    "TransactionAmt": 149.99,
    "ProductCD": "W",
    "features": {"V14": -2.3, "C1": 1.0},
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


async def run(args: argparse.Namespace) -> int:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    semaphore = asyncio.Semaphore(args.concurrency)
    server_ms: list[float] = []
    client_ms: list[float] = []
    failures: list[str] = []

    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:

        async def score(index: int) -> None:
            async with semaphore:
                payload = {
                    **PAYLOAD,
                    "transaction_id": f"load-{index}",
                    "TransactionDT": 3600.0 + index,
                    "TransactionAmt": round(10.0 + (index % 490), 2),
                    # A spread of accounts exercises the velocity state without
                    # growing it without bound.
                    "card1": 9000 + (index % 50),
                }
                started = time.perf_counter()
                try:
                    response = await client.post(
                        args.url, params={"include_shap": "false"}, json=payload, headers=headers
                    )
                    elapsed = (time.perf_counter() - started) * 1000.0
                except httpx.HTTPError as exc:
                    failures.append(f"request {index}: {exc}")
                    return
                if response.status_code != 200:
                    failures.append(f"request {index}: HTTP {response.status_code}")
                    return
                server_ms.append(float(response.json()["latency_ms"]))
                client_ms.append(elapsed)

        for i in range(args.warmup):
            await score(i)
        server_ms.clear()
        client_ms.clear()

        started = time.perf_counter()
        await asyncio.gather(*(score(i) for i in range(args.requests)))
        wall_seconds = time.perf_counter() - started

    if failures:
        print(f"FAIL: {len(failures)}/{args.requests} requests failed")
        for message in failures[:5]:
            print(f"  {message}")
        if any("429" in message for message in failures):
            print("  Hint: raise RATE_LIMIT_PER_MINUTE or lower --requests.")
        return 1

    p50 = percentile(server_ms, 50)
    p95 = percentile(server_ms, 95)
    p99 = percentile(server_ms, 99)
    client_p95 = percentile(client_ms, 95)
    throughput = len(server_ms) / wall_seconds if wall_seconds else 0.0

    print(f"requests={len(server_ms)} concurrency={args.concurrency} url={args.url}")
    print(f"server  p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")
    print(f"client  p95={client_p95:.2f}ms (round trip)")
    print(f"throughput={throughput:.1f} req/s")

    if args.max_p95 is not None and p95 > args.max_p95:
        print(f"FAIL: server p95 {p95:.2f}ms exceeds --max-p95 {args.max_p95:.2f}ms")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument(
        "--max-p95",
        type=float,
        default=None,
        help="Fail when server-side p95 exceeds this many milliseconds.",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
