"""Tests for IEEE-CIS ingestion, account proxy and time split."""

import numpy as np
import pandas as pd
import pytest

from src.data.ieee import load_ieee, time_split
from src.data.schema import ACCOUNT_COL, TARGET


def _write_fixture(tmp_path):
    tx = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionDT": [100, 200, 300, 400, 500],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
            "card1": [11, 11, 22, 22, 33],
            "addr1": [100.0, 100.0, np.nan, np.nan, 300.0],
            "ProductCD": ["W", "W", "C", "C", "W"],
            "const_col": [1, 1, 1, 1, 1],
        }
    )
    identity = pd.DataFrame({"TransactionID": [1, 2], "DeviceType": ["desktop", "mobile"]})
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    tx.to_csv(data_dir / "train_transaction.csv", index=False)
    identity.to_csv(data_dir / "train_identity.csv", index=False)
    return data_dir


def test_load_joins_identity_and_builds_account_proxy(tmp_path):
    data_dir = _write_fixture(tmp_path)
    df = load_ieee(data_dir=str(data_dir), cache_path=str(tmp_path / "cache.pkl"))

    assert len(df) == 5
    assert TARGET in df.columns
    assert df.loc[0, ACCOUNT_COL] == "11|100"
    assert df.loc[2, ACCOUNT_COL] == "22|-1"
    assert df.loc[0, "DeviceType"] == "desktop"
    assert pd.isna(df.loc[2, "DeviceType"])
    assert "const_col" not in df.columns
    assert df["TransactionAmt"].dtype == "float32"
    assert str(df["ProductCD"].dtype) == "category"


def test_cache_is_reused_without_source_files(tmp_path):
    data_dir = _write_fixture(tmp_path)
    cache = tmp_path / "cache.pkl"
    load_ieee(data_dir=str(data_dir), cache_path=str(cache))
    assert cache.exists()

    for name in ("train_transaction.csv", "train_identity.csv"):
        (data_dir / name).unlink()
    df = load_ieee(data_dir=str(data_dir), cache_path=str(cache))
    assert len(df) == 5


def test_missing_dataset_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle competitions download"):
        load_ieee(data_dir=str(tmp_path / "nope"), cache_path=None)


def test_time_split_has_no_overlap_and_later_test(tmp_path):
    data_dir = _write_fixture(tmp_path)
    df = load_ieee(data_dir=str(data_dir), cache_path=None)
    train, test = time_split(df, test_frac=0.2)

    assert len(test) == 1
    assert train["TransactionDT"].max() < test["TransactionDT"].min()
    assert set(train["TransactionID"]).isdisjoint(set(test["TransactionID"]))


def test_time_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        time_split(pd.DataFrame({"TransactionDT": [1, 2]}), test_frac=1.5)
