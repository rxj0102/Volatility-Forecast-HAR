"""Data acquisition and feature engineering for SPY volatility forecasting.

Data is open and free: the SPY stand in is the Ken French US market total
return (Mkt-RF + RF) and VIX is FRED VIXCLS. No Yahoo Finance.
"""

import numpy as np
import pandas as pd

from src.data.opendata import fred, french

ANNUALIZE = np.sqrt(252)

# Date constants
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
COVID_START = "2020-02-01"
COVID_END = "2020-06-30"
CALM_START = "2024-01-01"
CALM_END = "2025-12-31"


def download_spy_vix(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Load the SPY stand in and VIX from open data.

    SPY: Ken French US market total return (Mkt-RF + RF) turned into an index.
    VIX: FRED VIXCLS. Only closes are available (no open, high, low).

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with a Close column and rescaled VIX, indexed by date.
    """
    ff = french("F-F_Research_Data_Factors_daily", start=start, end=end)
    mkt = ff["Mkt-RF"] + ff["RF"]
    df = pd.DataFrame({"Close": 100 * (1 + mkt).cumprod()})
    vix = fred("VIXCLS", start, end)["VIXCLS"].dropna()
    df["VIX"] = (vix / 100).reindex(df.index)  # rescale to match realized vol units
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

    # Garman-Klass OHLC estimator (only when high/low/open are supplied;
    # the open data loader provides closes only)
    if {"High", "Low", "Open"}.issubset(out.columns):
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
        if src_col in out.columns:
            out[new_col] = out[src_col].shift(lag)
    out["Return_lag1_sq"] = out["Return_lag1_sq"] ** 2

    return out.dropna(subset=["LogReturn", "RV_5", "VIX"])
