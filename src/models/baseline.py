"""Unsupervised anomaly detection baseline using Isolation Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    """Unsupervised anomaly detection baseline.

    Serves as the cold-start and novel-fraud detection benchmark against
    supervised gradient boosted tree models.
    """

    def __init__(
        self,
        contamination: float = 0.035,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self._fill: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> IsolationForestBaseline:
        """Fit Isolation Forest on features, imputing missing values."""
        values = np.asarray(X, dtype=float)
        if self._fill is None:
            self._fill = np.nan_to_num(np.nanmedian(values, axis=0), nan=0.0)
        self.model.fit(np.where(np.isnan(values), self._fill, values))
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute continuous anomaly risk score in [0, 1].

        ``decision_function`` is higher for normal points, so it is mapped
        through a decreasing sigmoid: higher output means higher fraud risk.
        """
        if self._fill is None:
            raise RuntimeError("IsolationForestBaseline must be fitted before scoring.")
        values = np.asarray(X, dtype=float)
        raw_scores = self.model.decision_function(np.where(np.isnan(values), self._fill, values))
        return 1.0 / (1.0 + np.exp(raw_scores * 5.0))
