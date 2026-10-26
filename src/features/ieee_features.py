"""IEEE-CIS feature contract: candidate columns, categorical codes, gain selection.

The raw table has 435 columns, most of them sparse. Keeping everything makes the
training frame needlessly wide, so candidates are filtered by missingness and
cardinality and then cut to a fixed budget by LightGBM gain.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from src.data.schema import ACCOUNT_COL, ID_COL, TARGET, TIME_COL

MAX_MISSING_FRACTION = 0.60
MAX_CARDINALITY = 2000
MAX_FEATURES = 130

CORE_NUMERIC = ("TransactionAmt", "dist1", "dist2")
CORE_CATEGORICAL = (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
)
_DROP = {ID_COL, TARGET, TIME_COL, ACCOUNT_COL}


def _is_block(column: str, prefix: str) -> bool:
    rest = column[len(prefix) :]
    return column.startswith(prefix) and rest.isdigit() and len(rest) > 0


def candidate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split usable columns into (numeric, categorical)."""
    missing = df.isna().mean()
    numeric: List[str] = []
    categorical: List[str] = []

    for column in df.columns:
        if column in _DROP:
            continue
        series = df[column]

        if column in CORE_CATEGORICAL or _is_block(column, "M"):
            categorical.append(column)
        elif column in CORE_NUMERIC or any(_is_block(column, prefix) for prefix in ("C", "D", "V")):
            if column in CORE_NUMERIC or missing[column] <= MAX_MISSING_FRACTION:
                numeric.append(column)
        elif is_numeric_dtype(series):
            if missing[column] <= MAX_MISSING_FRACTION:
                numeric.append(column)
        elif is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            if series.nunique(dropna=True) <= MAX_CARDINALITY:
                categorical.append(column)

    return numeric, categorical


def fit_categories(df: pd.DataFrame, columns: List[str]) -> Dict[str, pd.Index]:
    """Learn the code vocabulary for every categorical column."""
    return {
        column: pd.Index(sorted(df[column].dropna().astype(str).unique())) for column in columns
    }


def encode_frame(
    df: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
    categories: Dict[str, pd.Index],
) -> pd.DataFrame:
    """Numeric passthrough plus integer codes (-1 = missing or unseen)."""
    data: Dict[str, pd.Series] = {}
    for column in numeric:
        if column in df.columns:
            data[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
        else:
            data[column] = pd.Series(np.float32(np.nan), index=df.index)
    for column in categorical:
        if column in df.columns:
            values = df[column].astype(str).where(df[column].notna())
        else:
            values = pd.Series(np.nan, index=df.index)
        codes = categories.get(column, pd.Index([])).get_indexer(values)
        encoded = pd.Series(codes, index=df.index, dtype="float32")
        encoded[encoded < 0] = np.nan  # unseen or missing -> LightGBM missing
        data[column] = encoded
    return pd.DataFrame(data, index=df.index)


def select_by_gain(
    frame: pd.DataFrame,
    y: pd.Series,
    categories: Dict[str, pd.Index],
    max_features: int = MAX_FEATURES,
) -> List[str]:
    """Keep the top `max_features` columns by LightGBM gain, in frame order."""
    if len(frame.columns) <= max_features:
        return list(frame.columns)

    import lightgbm as lgb

    categorical = [c for c in frame.columns if c in categories]
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(frame, y, categorical_feature=categorical)
    gain = model.booster_.feature_importance(importance_type="gain")
    keep = np.argsort(-gain, kind="stable")[:max_features]
    return [frame.columns[i] for i in sorted(keep)]
