"""Causal per-account velocity features for IEEE-CIS transactions.

Velocity is keyed on the household proxy ``card1|addr1``. Keying on ``card1``
alone would measure issuer behaviour (one value covers ~15k transactions), not
account behaviour.

Every feature describes the past only: the snapshots are taken before the
current transaction is appended to state, so batch training and online serving
see the same point-in-time view.

State is bounded twice, because a payment stream is attacker-influenced input:
least-recently-seen accounts are evicted past ``max_accounts``, and each
account's 24h history is capped. On the real data the busiest account peaks at
174 transactions in 24h, so the cap is a safety net rather than a feature change.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd

from src.data.schema import ACCOUNT_COL, AMOUNT_COL, TIME_COL

# Order matters: persisted pipelines and tests index by name.
FEATURE_NAMES = (
    "acct_tx_count_1h",
    "acct_tx_count_24h",
    "acct_amt_sum_24h",
    "acct_secs_since_prev",
    "acct_amt_vs_mean",
    "acct_tx_count_prior",
    "acct_days_since_first",
    "hour_of_day",
    "day_of_week",
)


class SlidingWindowStateAggregator:
    """Tracks per-account activity to compute causal velocity features."""

    def __init__(
        self,
        window_1h_sec: float = 3600.0,
        window_24h_sec: float = 86400.0,
        max_accounts: int = 50_000,
        max_events_per_account: int = 2_000,
    ) -> None:
        if max_accounts < 1 or max_events_per_account < 1:
            raise ValueError("state bounds must be at least 1")
        self.window_1h_sec = window_1h_sec
        self.window_24h_sec = window_24h_sec
        self.max_accounts = max_accounts
        self.max_events_per_account = max_events_per_account
        self._history: "OrderedDict[str, Deque[Tuple[float, float]]]" = OrderedDict()
        self._recent_1h: Dict[str, Deque[float]] = {}
        self._stats: Dict[str, Dict[str, float]] = {}
        self._events = 0

    def clear(self) -> None:
        """Reset internal memory state."""
        self._history.clear()
        self._recent_1h.clear()
        self._stats.clear()
        self._events = 0

    def state_size(self) -> Dict[str, int]:
        """Current tracked accounts and retained events, for monitoring."""
        return {"accounts": len(self._history), "events": self._events}

    def update_and_extract(
        self,
        account_id: str,
        timestamp: float,
        amount: float,
    ) -> Dict[str, float]:
        """Return features for the current transaction, then record it."""
        history = self._history.get(account_id)
        if history is None:
            history = self._history[account_id] = deque()
            self._recent_1h[account_id] = deque()
            self._stats[account_id] = {
                "count": 0.0,
                "total": 0.0,
                "first_ts": timestamp,
                "last_ts": None,
            }
            self._evict_oldest_account()
        else:
            self._history.move_to_end(account_id)

        stats = self._stats[account_id]
        recent = self._recent_1h[account_id]

        # Prune to the windows, then snapshot the past.
        cutoff_24h = timestamp - self.window_24h_sec
        while history and history[0][0] < cutoff_24h:
            history.popleft()
            self._events -= 1
        cutoff_1h = timestamp - self.window_1h_sec
        while recent and recent[0] < cutoff_1h:
            recent.popleft()

        prior_count = stats["count"]
        if prior_count == 0:
            secs_since_prev = np.nan
            amt_vs_mean = np.nan
        else:
            secs_since_prev = max(0.0, timestamp - stats["last_ts"])
            mean = stats["total"] / prior_count
            amt_vs_mean = float(np.clip(amount / max(1.0, mean), 0.01, 50.0))

        features = {
            "acct_tx_count_1h": float(len(recent)),
            "acct_tx_count_24h": float(len(history)),
            "acct_amt_sum_24h": float(sum(a for _, a in history)),
            "acct_secs_since_prev": secs_since_prev,
            "acct_amt_vs_mean": amt_vs_mean,
            "acct_tx_count_prior": float(prior_count),
            "acct_days_since_first": max(0.0, (timestamp - stats["first_ts"]) / 86400.0),
            "hour_of_day": (timestamp % 86400.0) / 3600.0,
            "day_of_week": (timestamp / 86400.0) % 7.0,
        }

        history.append((timestamp, amount))
        recent.append(timestamp)
        self._events += 1
        # shortcut: a burst past the cap undercounts the 24h window; raise the cap or
        # move history to a feature store before that becomes reachable in production.
        while len(history) > self.max_events_per_account:
            history.popleft()
            self._events -= 1
        while len(recent) > self.max_events_per_account:
            recent.popleft()

        stats["count"] = prior_count + 1.0
        stats["total"] += amount
        stats["last_ts"] = timestamp
        return features

    def _evict_oldest_account(self) -> None:
        """Drop the least recently seen account once the bound is exceeded."""
        while len(self._history) > self.max_accounts:
            oldest, evicted = self._history.popitem(last=False)
            self._events -= len(evicted)
            self._recent_1h.pop(oldest, None)
            self._stats.pop(oldest, None)

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity features to a frame, in time order.

        Continues from the aggregator's current history; call ``clear()`` to
        start over. Rows keep their index labels so targets stay aligned.
        """
        frame = df.sort_values(TIME_COL, kind="mergesort")
        accounts = frame[ACCOUNT_COL].astype(str).to_numpy()
        times = frame[TIME_COL].to_numpy(dtype=float)
        amounts = frame[AMOUNT_COL].to_numpy(dtype=float)

        columns = {name: np.empty(len(frame), dtype=np.float32) for name in FEATURE_NAMES}
        for i, (account, t, amount) in enumerate(zip(accounts, times, amounts)):
            features = self.update_and_extract(account, t, amount)
            for name in FEATURE_NAMES:
                columns[name][i] = features[name]

        return pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)
