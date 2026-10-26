"""Tests for per-account velocity features and the IEEE-CIS feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import ACCOUNT_COL, AMOUNT_COL, TIME_COL
from src.features.pipeline import FeaturePipeline
from src.features.state_aggregator import FEATURE_NAMES, SlidingWindowStateAggregator


def _history_frame():
    return pd.DataFrame(
        {
            ACCOUNT_COL: ["a", "a", "b", "a", "b"],
            TIME_COL: [0.0, 600.0, 100.0, 7200.0, 90000.0],
            AMOUNT_COL: [50.0, 150.0, 5.0, 20.0, 7.0],
        }
    )


def test_velocity_is_causal_and_per_account():
    agg = SlidingWindowStateAggregator()
    out = agg.transform_batch(_history_frame()).sort_index()

    assert list(out["acct_tx_count_1h"]) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert list(out["acct_tx_count_prior"]) == [0.0, 1.0, 0.0, 2.0, 1.0]
    assert list(out["acct_amt_sum_24h"]) == [0.0, 50.0, 0.0, 200.0, 0.0]
    assert np.isnan(out["acct_secs_since_prev"].iloc[0])
    assert np.isnan(out["acct_amt_vs_mean"].iloc[2])


def test_batch_order_does_not_change_features():
    frame = _history_frame()
    shuffled = frame.sample(frac=1.0, random_state=0)
    ordered = SlidingWindowStateAggregator().transform_batch(frame)
    reordered = SlidingWindowStateAggregator().transform_batch(shuffled)

    pd.testing.assert_frame_equal(
        ordered.sort_index()[list(FEATURE_NAMES)],
        reordered.sort_index()[list(FEATURE_NAMES)],
    )


def test_pipeline_fit_transform_returns_targets(ieee_frame):
    pipeline = FeaturePipeline()
    X, y = pipeline.fit_transform(ieee_frame)

    assert list(X.columns) == pipeline.feature_columns
    assert len(y) == len(ieee_frame)
    assert y.dtype == "int8"
    assert "acct_tx_count_prior" in X.columns
    assert "TransactionDT" not in X.columns


def test_pipeline_selection_respects_budget(ieee_frame):
    pipeline = FeaturePipeline()
    X, _ = pipeline.fit_transform(ieee_frame, max_features=5)
    assert len(pipeline.feature_columns) == 5
    assert list(X.columns) == pipeline.feature_columns


def test_transform_continues_from_previous_history(ieee_frame):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(ieee_frame)

    later = ieee_frame.iloc[[0]].copy()
    later["TransactionDT"] = 1_000_000.0
    X, _ = pipeline.transform(later)

    assert X["acct_tx_count_prior"].iloc[0] == 3.0


def test_transform_single_matches_batch_encoding(ieee_frame):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(ieee_frame)

    payload = {
        "card1": 999.0,
        "addr1": 100.0,
        "TransactionDT": 1_000_000.0,
        "TransactionAmt": 10.0,
        "features": {"ProductCD": "W", "C1": 1.0, "V1": 0.1},
    }
    first = pipeline.transform_single(payload)[0]
    second = pipeline.transform_single({**payload, "TransactionDT": 1_000_600.0})[0]
    position = {column: i for i, column in enumerate(pipeline.feature_columns)}

    assert first[position["acct_tx_count_prior"]] == 0.0
    assert second[position["acct_tx_count_prior"]] == 1.0
    assert second[position["acct_tx_count_1h"]] == 1.0
    assert first[position["ProductCD"]] == pipeline.categories["ProductCD"].get_loc("W")
    assert np.isclose(first[position["C1"]], 1.0)


def test_explicit_fields_win_over_nested_features(ieee_frame):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(ieee_frame)

    payload = {
        "card1": 777.0,
        "addr1": 1.0,
        "TransactionDT": 1_000_000.0,
        "TransactionAmt": 10.0,
        "features": {"TransactionAmt": 9999.0},
    }
    pipeline.transform_single(payload)
    second = pipeline.transform_single({**payload, "TransactionDT": 1_000_060.0})[0]
    position = {column: i for i, column in enumerate(pipeline.feature_columns)}

    assert second[position["acct_amt_sum_24h"]] == 10.0


def test_single_and_batch_encoding_agree(ieee_frame):
    pipeline = FeaturePipeline()
    pipeline.fit_transform(ieee_frame)

    row = ieee_frame.iloc[1]
    payload = {key: (None if pd.isna(value) else value) for key, value in row.items()}
    batch, _ = pipeline.transform(ieee_frame.iloc[[1]])
    single = pipeline.transform_single(payload)[0]

    for i, column in enumerate(pipeline.feature_columns):
        if column.startswith("acct_") or column in ("hour_of_day", "day_of_week"):
            continue
        expected, actual = batch[column].iloc[0], single[i]
        assert (pd.isna(expected) and pd.isna(actual)) or np.isclose(expected, actual), column


def test_unfitted_pipeline_refuses_to_score():
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform_single({"card1": 1.0, "TransactionAmt": 5.0})
