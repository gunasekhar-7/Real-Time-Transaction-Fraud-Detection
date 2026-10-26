"""IEEE-CIS feature construction, shared by training and serving."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.schema import (
    ACCOUNT_COL,
    AMOUNT_COL,
    TARGET,
    TIME_COL,
    account_from_values,
    build_account_id,
)

from .ieee_features import (
    MAX_FEATURES,
    candidate_columns,
    encode_frame,
    fit_categories,
    select_by_gain,
)
from .state_aggregator import FEATURE_NAMES, SlidingWindowStateAggregator


class FeaturePipeline:
    """Turn raw IEEE-CIS rows into the model matrix.

    ``fit_transform`` learns categorical codes and the feature budget from the
    training frame; ``transform`` reuses both. The aggregator keeps its history
    between calls, so a later time window sees earlier transactions.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.aggregator = SlidingWindowStateAggregator(
            max_accounts=settings.max_accounts,
            max_events_per_account=settings.max_events_per_account,
        )
        self.categories: Dict[str, pd.Index] = {}
        self.category_kinds: Dict[str, str] = {}
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.feature_columns: List[str] = []

    def reset(self) -> None:
        self.aggregator.clear()

    def fit_transform(
        self,
        df: pd.DataFrame,
        max_features: int = MAX_FEATURES,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Reset history, learn the encoding and return (X, y) for training."""
        self.reset()
        self.numeric_columns, self.categorical_columns = candidate_columns(df)
        frame, y = self._build(df, fit=True)
        self.feature_columns = select_by_gain(frame, y, self.categories, max_features)
        return frame[self.feature_columns].astype("float32"), y

    @property
    def model_categorical_columns(self) -> List[str]:
        """Categorical columns that survived feature selection."""
        return [c for c in self.feature_columns if c in self.categories]

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) continuing from the aggregator's existing history."""
        if not self.feature_columns:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")
        frame, y = self._build(df, fit=False)
        return frame.reindex(columns=self.feature_columns).astype("float32"), y

    def transform_single(self, payload: Dict[str, Any]) -> np.ndarray:
        """Transform one online transaction into a (1, n_features) array."""
        if not self.feature_columns:
            raise RuntimeError("FeaturePipeline must be fitted before scoring.")

        flat = dict(payload)
        nested = flat.pop("features", None) or {}
        # Explicit payload fields win so a nested duplicate cannot override
        # validation-critical values; features only fill omitted vendor columns.
        explicit = {key: value for key, value in flat.items() if value is not None}
        flat = {**nested, **explicit}
        timestamp = flat.get(TIME_COL)
        amount = float(flat[AMOUNT_COL])
        velocity = self.aggregator.update_and_extract(
            account_from_values(flat.get("card1"), flat.get("addr1")),
            float(timestamp) if timestamp is not None else 0.0,
            amount,
        )

        vector = []
        for column in self.feature_columns:
            if column in self.categories:
                vector.append(self._code(column, flat.get(column)))
            elif column in velocity:
                vector.append(velocity[column])
            else:
                value = flat.get(column)
                vector.append(float(value) if value is not None else np.nan)
        return np.array([vector], dtype=np.float32)

    def _code(self, column: str, value: Any) -> float:
        """Encode one categorical value exactly like the batch path does."""
        if value is None:
            return np.nan
        kind = self.category_kinds.get(column, "O")
        if kind in ("i", "u"):
            key = str(int(value))
        elif kind == "f":
            key = str(float(value))
        else:
            key = str(value)
        position = self.categories[column].get_indexer([key])[0]
        return float(position) if position >= 0 else np.nan

    def _build(
        self,
        df: pd.DataFrame,
        fit: bool,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        source = list(
            dict.fromkeys(
                self.numeric_columns
                + self.categorical_columns
                + [ACCOUNT_COL, TIME_COL, AMOUNT_COL, TARGET]
            )
        )
        if ACCOUNT_COL not in df.columns:
            df = df.assign(**{ACCOUNT_COL: build_account_id(df)})
        enriched = self.aggregator.transform_batch(df[[c for c in source if c in df.columns]])

        if fit:
            self.categories = fit_categories(enriched, self.categorical_columns)
            self.category_kinds = {
                column: enriched[column].dtype.kind for column in self.categorical_columns
            }

        y = (
            enriched[TARGET].astype("int8")
            if TARGET in enriched.columns
            else pd.Series(0, index=enriched.index, dtype="int8")
        )
        frame = encode_frame(
            enriched, self.numeric_columns, self.categorical_columns, self.categories
        )
        return (
            pd.concat([frame, enriched[list(FEATURE_NAMES)].astype("float32")], axis=1),
            y,
        )
