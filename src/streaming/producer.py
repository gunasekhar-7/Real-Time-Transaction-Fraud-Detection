"""Replay real IEEE-CIS transactions, including real frauds, as a live stream.

The demo used to invent 28 PCA columns that no longer exist in the model. It now
replays feature vectors sampled from the held-out test set, so what the dashboard
shows is what the shipped model actually scores. Attack bursts replay real fraud
rows rather than fabricated amounts; the model still decides, and some are missed.

TransactionDT in generated events uses a monotonic virtual clock so that the
sliding-window aggregator never sees time go backwards, even when rows are
sampled out of chronological order from the test set.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SAMPLE_PATH = "models/stream_sample.pkl"

# Typed fields of TransactionPayload; everything else is a vendor column.
CORE_KEYS = (
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
)
SKIP_KEYS = {"account_id", "isFraud", "TransactionID", "TransactionDT"}

TEMPLATE = {
    "TransactionAmt": 45.0,
    "ProductCD": "W",
    "card1": 9500.0,
    "card2": 321.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 299.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop",
}


def _clean(value: Any) -> Any:
    """Make a pandas cell JSON and pydantic safe."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, str):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


class TransactionStreamGenerator:
    """Serves legitimate and fraudulent test-set rows for the live dashboard."""

    def __init__(self, random_seed: int = 42, sample_path: str = SAMPLE_PATH) -> None:
        self.rng = np.random.default_rng(random_seed)
        self.tx_counter = 1
        # Virtual clock: starts at 1 day in seconds, increments ~10s per event
        self._virtual_dt = 86400.0
        self._legit: Optional[pd.DataFrame] = None
        self._fraud: Optional[pd.DataFrame] = None

        if os.path.exists(sample_path):
            sample = pd.read_pickle(sample_path)
            if "isFraud" in sample.columns:
                self._legit = sample[sample["isFraud"] == 0]
                self._fraud = sample[sample["isFraud"] == 1]
            else:
                self._legit = sample

    def _row(self, fraud: bool) -> Dict[str, Any]:
        pool = (
            self._fraud if fraud and self._fraud is not None and len(self._fraud) else self._legit
        )
        if pool is None or not len(pool):
            return dict(TEMPLATE)
        row = pool.iloc[int(self.rng.integers(len(pool)))].to_dict()
        return {key: _clean(value) for key, value in row.items()}

    def _payload(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self.tx_counter += 1
        self._virtual_dt += self.rng.uniform(5.0, 15.0)  # ~10s between events
        payload = {key: row.get(key) for key in CORE_KEYS}
        payload["TransactionDT"] = self._virtual_dt
        payload["transaction_id"] = f"tx_{self.tx_counter:07d}"
        payload["features"] = {
            key: value
            for key, value in row.items()
            if key not in CORE_KEYS and key not in SKIP_KEYS
        }
        return payload

    def generate_single_event(
        self,
        is_fraud: bool = False,
        scenario: str = "normal",
    ) -> Dict[str, Any]:
        """Return one transaction event payload."""
        return self._payload(self._row(is_fraud))

    def generate_scenario_burst(
        self,
        scenario: str = "card_testing",
        num_events: int = 15,
    ) -> List[Dict[str, Any]]:
        """Return a burst of real fraudulent transactions."""
        return [self._payload(self._row(True)) for _ in range(num_events)]
