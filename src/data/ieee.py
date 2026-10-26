"""Load the IEEE-CIS training tables, join identity, shrink dtypes, cache locally."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from .schema import ACCOUNT_COL, ID_COL, TARGET, TIME_COL, build_account_id

DATA_DIR = "data/raw/ieee"
CACHE_PATH = "data/processed/ieee.pkl"
CATEGORICAL_MAX = 5000


def load_ieee(
    data_dir: str = DATA_DIR,
    cache_path: str | None = CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load and cache the labelled IEEE-CIS training data.

    The cache is a local pickle: pyarrow needs numpy>=2 and this env is pinned to
    numpy 1.26. Switch to parquet when the ML stack is upgraded.
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    tx_path = os.path.join(data_dir, "train_transaction.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(
            f"{tx_path} not found. Download it once with:\n"
            "  kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee\n"
            "(accept the competition rules on kaggle.com first)"
        )

    df = pd.read_csv(tx_path, low_memory=False)
    identity_path = os.path.join(data_dir, "train_identity.csv")
    if os.path.exists(identity_path):
        df = df.merge(pd.read_csv(identity_path), on=ID_COL, how="left")

    df[ACCOUNT_COL] = build_account_id(df)
    df = shrink(df)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_pickle(cache_path)
    return df


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant columns and downcast numeric/string types to cut memory ~4x."""
    nunique = df.nunique(dropna=False)
    keep = [c for c in df.columns if nunique[c] > 1 or c in (ID_COL, TARGET, TIME_COL)]
    df = df[keep]

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            if df[col].dtype == "float64":
                df[col] = df[col].astype("float32")
            elif df[col].dtype == "int64" and df[col].abs().max() < 2**31:
                df[col] = df[col].astype("int32")
        elif is_string_dtype(df[col]) and df[col].nunique(dropna=True) <= CATEGORICAL_MAX:
            df[col] = df[col].astype("category")
    return df


def time_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    time_col: str = TIME_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the last `test_frac` of rows becomes the test set.

    Random splitting leaks here because card-level features let the model see the
    same household in both halves.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    cutoff = df[time_col].quantile(1.0 - test_frac)
    train = df[df[time_col] <= cutoff].reset_index(drop=True)
    test = df[df[time_col] > cutoff].reset_index(drop=True)
    return train, test
