"""Data acquisition and feature engineering for SPY volatility forecasting."""

import numpy as np
import pandas as pd
import yfinance as yf

ANNUALIZE = np.sqrt(252)

# Date constants
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
COVID_START = "2020-02-01"
COVID_END = "2020-06-30"
CALM_START = "2024-01-01"
CALM_END = "2025-12-31"


def download_spy_vix(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Download SPY OHLC and VIX data from Yahoo Finance.

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with SPY OHLC columns and rescaled VIX, indexed by date.
    """
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy.columns = spy.columns.get_level_values(0)

    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix.columns = vix.columns.get_level_values(0)

    df = spy.copy()
    df["VIX"] = vix["Close"] / 100  # rescale to match realized vol units
    return df.dropna(subset=["Close", "VIX"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct realized volatility proxies and lagged predictors.

    Volatility measures (all annualized):
    - RV_Daily: |log-return| × √252  (instantaneous proxy)
    - RV_5:     5-day rolling std × √252  (forecast target)
    - GK_Vol:   Garman-Klass OHLC estimator (8× more efficient than close-to-close)

    HAR components:
    - HAR_D:  yesterday's RV_Daily
    - HAR_W:  5-day rolling average of RV_Daily, lagged 1 day
    - HAR_M:  22-day rolling average of RV_Daily, lagged 1 day

    All features are lagged by at least 1 day — no look-ahead bias.

    Args:
        df: Raw OHLC + VIX DataFrame from download_spy_vix().

    Returns:
        DataFrame with features; rows with NaN in core columns dropped.
    """
    out = df.copy()

    # Log return
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))

    # Realized volatility proxies (annualized)
    out["RV_Daily"] = out["LogReturn"].abs() * ANNUALIZE
    out["RV_5"] = out["LogReturn"].rolling(5).std() * ANNUALIZE  # target variable

    # Garman-Klass OHLC estimator
    log_hl = np.log(out["High"] / out["Low"])
    log_co = np.log(out["Close"] / out["Open"])
    out["GK_Vol"] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2) * ANNUALIZE

    # Variance Risk Premium (lagged below — VIX is forward-looking)
    out["VRP"] = out["VIX"] - out["RV_5"]

    # HAR regressors (lagged 1 day to prevent look-ahead)
    out["HAR_D"] = out["RV_Daily"].shift(1)
    out["HAR_W"] = out["RV_Daily"].rolling(5).mean().shift(1)
    out["HAR_M"] = out["RV_Daily"].rolling(22).mean().shift(1)

    # Lagged feature set (≥1 day lag — strict no look-ahead)
    lag_map = {
        "RV_lag1": ("RV_5", 1),
        "RV_lag5": ("RV_5", 5),
        "RV_lag10": ("RV_5", 10),
        "VRP_lag1": ("VRP", 1),
        "VRP_lag5": ("VRP", 5),
        "GK_Vol_lag1": ("GK_Vol", 1),
        "AbsReturn_lag1": ("RV_Daily", 1),
        "Return_lag1_sq": ("LogReturn", 1),
    }
    for new_col, (src_col, lag) in lag_map.items():
        out[new_col] = out[src_col].shift(lag)
    out["Return_lag1_sq"] = out["Return_lag1_sq"] ** 2

    return out.dropna(subset=["LogReturn", "RV_5", "VIX"])
