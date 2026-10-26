"""Shared test setup: threshold singleton and a small IEEE-CIS frame."""

import os

# The suite always runs in development mode, whatever the caller's shell says.
os.environ["APP_ENV"] = "development"
for _name in (
    "API_KEY",
    "ALLOWED_ORIGINS",
    "RATE_LIMIT_PER_MINUTE",
    "MAX_ACCOUNTS",
    "MAX_EVENTS_PER_ACCOUNT",
    "PREDICTOR_CACHE_SIZE",
    "LOG_LEVEL",
):
    os.environ.pop(_name, None)

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import get_settings

DEFAULTS = {"approve_threshold": 0.35, "decline_threshold": 0.70}


@pytest.fixture(autouse=True)
def reset_thresholds():
    client = TestClient(app)
    client.post("/threshold", json=DEFAULTS)
    yield
    client.post("/threshold", json=DEFAULTS)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Env changes in one test must not leak into the next."""
    yield
    get_settings.cache_clear()


@pytest.fixture
def ieee_frame() -> pd.DataFrame:
    """Six transactions across two accounts, shaped like IEEE-CIS columns."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "isFraud": [0, 0, 1, 0, 0, 1],
            "TransactionDT": [1000.0, 1600.0, 2200.0, 1100.0, 4000.0, 90000.0],
            "TransactionAmt": [10.0, 20.0, 300.0, 15.0, 25.0, 500.0],
            "ProductCD": ["W", "W", "C", "C", "W", "C"],
            "card1": [111, 111, 111, 222, 222, 222],
            "card2": [321.0, 321.0, 321.0, 555.0, 555.0, None],
            "addr1": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
            "DeviceType": ["desktop", "mobile", "desktop", "mobile", "mobile", None],
            "M1": ["T", "F", "T", "F", "T", "F"],
            "C1": [1.0, 2.0, 3.0, 1.0, 0.0, 5.0],
            "D1": [14.0, 15.0, 14.0, 2.0, 3.0, 9.0],
            "V1": [0.1, 0.2, -0.3, 0.4, 0.5, -0.6],
            "V2": [-1.0, 0.0, 1.0, 0.5, -0.5, 2.0],
            "id_31": ["chrome", "safari", "chrome", "edge", "chrome", None],
        }
    )
