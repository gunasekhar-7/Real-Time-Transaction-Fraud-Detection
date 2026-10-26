"""Column contracts for the IEEE-CIS training data."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "TransactionID"
TARGET = "isFraud"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
ACCOUNT_COL = "account_id"


def account_from_values(card, addr) -> str:
    """Build the household proxy key for a single transaction."""

    def _int(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-1"
        return str(int(value))

    return f"{_int(card)}|{_int(addr)}"


def build_account_id(
    df: pd.DataFrame,
    card_col: str = "card1",
    addr_col: str = "addr1",
) -> pd.Series:
    """Join card and billing address into a household proxy.

    IEEE-CIS has no customer ID. card1 alone is an issuer-level group (one value
    covers up to 15k transactions), which would turn account velocity into issuer
    velocity. card1 + addr1 gives ~40k groups.
    """
    card = df[card_col].fillna(-1).astype("int64").astype(str)
    if addr_col in df.columns:
        addr = df[addr_col].fillna(-1).astype("int64").astype(str)
    else:
        addr = pd.Series("-1", index=df.index)
    return card + "|" + addr
