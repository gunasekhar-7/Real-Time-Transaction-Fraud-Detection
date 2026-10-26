"""Schema and account-proxy tests."""

import numpy as np
import pandas as pd
import pytest

from src.api.schemas import TransactionPayload
from src.data.schema import account_from_values, build_account_id


def test_transaction_payload_accepts_ieee_fields():
    payload = TransactionPayload(
        card1=9500,
        addr1=299,
        TransactionAmt=149.99,
        ProductCD="W",
        features={"V14": -3.0, "C1": 1.0},
    )
    assert payload.card1 == 9500.0
    assert payload.features["V14"] == -3.0


def test_transaction_payload_requires_card_and_amount():
    with pytest.raises(Exception):
        TransactionPayload(TransactionAmt=10.0)
    with pytest.raises(Exception):
        TransactionPayload(card1=1.0, TransactionAmt=-10.0)


@pytest.mark.parametrize(
    "reserved", ["transaction_id", "TransactionDT", "TransactionAmt", "card1", "addr1"]
)
def test_features_cannot_override_validated_fields(reserved):
    with pytest.raises(Exception, match="reserved keys"):
        TransactionPayload(card1=1.0, TransactionAmt=10.0, features={reserved: 1})


def test_features_are_bounded_and_scalar():
    too_many = {f"V{i}": 1.0 for i in range(501)}
    with pytest.raises(Exception):
        TransactionPayload(card1=1.0, TransactionAmt=10.0, features=too_many)
    with pytest.raises(Exception):
        TransactionPayload(card1=1.0, TransactionAmt=10.0, features={"V14": {"nested": 1}})

    with pytest.raises(Exception):
        TransactionPayload(transaction_id="x" * 129, card1=1.0, TransactionAmt=10.0)


def test_account_proxy_matches_batch_and_single_paths():
    assert account_from_values(9500.0, 299.0) == "9500|299"
    assert account_from_values(9500.0, np.nan) == "9500|-1"
    assert account_from_values(None, None) == "-1|-1"

    df = pd.DataFrame({"card1": [9500.0, 42.0], "addr1": [299.0, np.nan]})
    assert list(build_account_id(df)) == ["9500|299", "42|-1"]
