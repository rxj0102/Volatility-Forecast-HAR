"""HAR-RV model (Corsi 2009) for realized volatility forecasting.

The Heterogeneous AutoRegressive (HAR) model captures long-memory in volatility
using a parsimonious combination of daily, weekly, and monthly RV averages:

    RV_t = α + β_D · RV_{t-1}
             + β_W · mean(RV_{t-5:t-1})
             + β_M · mean(RV_{t-22:t-1})
             + ε_t
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """Construct HAR regressors from a daily realized volatility series.

    All regressors are lagged by 1 day (no look-ahead bias).

    Args:
        rv_daily: Annualized daily realized volatility Series.

    Returns:
        DataFrame with columns HAR_D, HAR_W, HAR_M, NaN rows dropped.
    """
    har = pd.DataFrame(index=rv_daily.index)
    har["HAR_D"] = rv_daily.shift(1)
    har["HAR_W"] = rv_daily.rolling(5).mean().shift(1)
    har["HAR_M"] = rv_daily.rolling(22).mean().shift(1)
    return har.dropna()


class HARModel:
    """OLS implementation of the HAR-RV model.

    Wraps sklearn LinearRegression to provide a finance-friendly interface
    with named coefficients and in-sample prediction (shifted for use as
    a lagged ML feature).
    """

    def __init__(self) -> None:
        self._ols = LinearRegression()
        self.coef_: dict[str, float] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        """Fit OLS HAR model on training data.

        Args:
            X: HAR feature matrix (HAR_D, HAR_W, HAR_M).
            y: Target realized volatility Series.

        Returns:
            self (for method chaining).
        """
        self._ols.fit(X, y)
        self.coef_ = dict(zip(X.columns, self._ols.coef_))
        self.coef_["Intercept"] = float(self._ols.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return OOS predictions as a named Series.

        Args:
            X: HAR feature matrix aligned to the prediction period.

        Returns:
            Forecast Series with same index as X.
        """
        return pd.Series(
            self._ols.predict(X), index=X.index, name="HAR_Pred"
        )

    def predict_insample(self, X: pd.DataFrame) -> pd.Series:
        """Return in-sample predictions shifted 1 day for use as an ML feature.

        Shifting prevents the HAR IS prediction from introducing look-ahead
        bias when used as a predictor in downstream ML models.

        Args:
            X: Full (train) HAR feature matrix.

        Returns:
            Lagged IS prediction Series.
        """
        preds = self._ols.predict(X)
        return pd.Series(preds, index=X.index, name="HAR_Pred_IS").shift(1)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Return OLS R² on the provided data split."""
        return float(self._ols.score(X, y))

    def print_summary(self) -> None:
        """Print coefficient table."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        print("HAR-RV Coefficients:")
        for name, val in self.coef_.items():
            print(f"  {name:>12s}: {val:.6f}")
