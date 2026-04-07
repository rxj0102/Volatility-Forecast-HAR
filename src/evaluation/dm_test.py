"""Diebold-Mariano test with HAC correction (Harvey, Leybourne & Newbold 1997).

Usage
-----
>>> from src.evaluation.dm_test import dm_test_all
>>> results = dm_test_all(y_true, har_pred, challenger_preds, loss="mse")

Interpretation
--------------
- H₀: Equal predictive accuracy between benchmark and challenger.
- Negative DM_HAC: benchmark is better.
- Positive DM_HAC: challenger is better.
- Use p_hac when iid_ok=False (autocorrelated loss differentials).
- HLN small-sample correction: t-distribution with T-1 degrees of freedom.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """Bartlett-kernel HAC variance estimate for loss differential series d.

    Args:
        d: Loss differential series (length T).
        h: Bandwidth (typically T^(1/4)).

    Returns:
        HAC variance estimate (scalar).
    """
    T = len(d)
    gamma0 = np.var(d, ddof=1)
    hac_var = gamma0
    for k in range(1, h + 1):
        gamma_k = float(np.cov(d[k:], d[:-k])[0, 1])
        hac_var += 2 * (1 - k / (h + 1)) * gamma_k
    return hac_var / T


def _loss_differential(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str,
) -> np.ndarray:
    """Compute element-wise loss differential d_t = L(e1_t) - L(e2_t)."""
    if loss == "mse":
        return (y_true - y_pred1) ** 2 - (y_true - y_pred2) ** 2
    elif loss == "qlike":
        h1 = np.clip(y_pred1, 1e-8, None)
        h2 = np.clip(y_pred2, 1e-8, None)
        qlike1 = y_true / h1 - np.log(y_true / h1) - 1
        qlike2 = y_true / h2 - np.log(y_true / h2) - 1
        return qlike1 - qlike2
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Choose 'mse' or 'qlike'.")


def dm_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss: str = "mse",
) -> dict[str, object]:
    """HAC-corrected Diebold-Mariano test for equal predictive accuracy.

    Tests H₀: E[d_t] = 0, where d_t = L(ê1_t) − L(ê2_t).

    Args:
        y_true: Realized values.
        y_pred1: Forecasts from model 1 (benchmark).
        y_pred2: Forecasts from model 2 (challenger).
        loss: Loss function — "mse" or "qlike".

    Returns:
        Dict with keys:
            DM_HAC  — HAC-corrected test statistic
            p_hac   — two-sided p-value (HLN t-dist correction)
            DM_std  — standard DM statistic (assumes IID)
            p_std   — standard p-value
            iid_ok  — True if Ljung-Box fails to reject IID (use p_std)
            h_bw    — Newey-West bandwidth (T^(1/4))
            T       — effective sample size
            better  — "model_1" if DM_HAC < 0, else "model_2"
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred1) | np.isnan(y_pred2))
    yt, yp1, yp2 = y_true[mask], y_pred1[mask], y_pred2[mask]
    T = int(mask.sum())

    d = _loss_differential(yt, yp1, yp2, loss)
    d_bar = d.mean()
    h_bw = max(1, int(T ** 0.25))

    # HAC-corrected statistic (HLN 1997)
    var_hac = _newey_west_variance(d, h_bw)
    se_hac = np.sqrt(max(var_hac, 1e-12))
    dm_hac = d_bar / se_hac
    p_hac = float(2 * stats.t.sf(abs(dm_hac), df=T - 1))

    # Standard DM (IID assumption)
    dm_std = d_bar / (d.std(ddof=1) / np.sqrt(T))
    p_std = float(2 * stats.norm.sf(abs(dm_std)))

    # Ljung-Box test for autocorrelation in loss differentials
    lb = acorr_ljungbox(d, lags=[h_bw], return_df=True)
    iid_ok = bool(lb["lb_pvalue"].iloc[0] > 0.05)

    return {
        "DM_HAC": round(float(dm_hac), 4),
        "p_hac": round(p_hac, 4),
        "DM_std": round(float(dm_std), 4),
        "p_std": round(p_std, 4),
        "iid_ok": iid_ok,
        "h_bw": h_bw,
        "T": T,
        "better": "model_1" if d_bar < 0 else "model_2",
    }


def dm_test_all(
    y_true: pd.Series,
    benchmark_pred: pd.Series,
    challenger_preds: dict[str, pd.Series],
    loss: str = "mse",
) -> pd.DataFrame:
    """Run DM tests comparing multiple challengers against a single benchmark.

    Args:
        y_true: Realized values.
        benchmark_pred: Benchmark forecasts (treated as model 1).
        challenger_preds: Dict of {name: forecast_series} for challengers.
        loss: "mse" or "qlike".

    Returns:
        DataFrame of DM test results indexed by challenger name.
    """
    yt = y_true.to_numpy()
    yp_bench = benchmark_pred.reindex(y_true.index).to_numpy()

    rows = []
    for name, pred in challenger_preds.items():
        yp = pred.reindex(y_true.index).to_numpy()
        result = dm_test(yt, yp_bench, yp, loss=loss)
        result["Challenger"] = name
        result["Loss"] = loss.upper()
        rows.append(result)

    cols = ["Challenger", "Loss", "DM_HAC", "p_hac", "iid_ok", "h_bw", "T", "better"]
    return pd.DataFrame(rows)[cols].set_index("Challenger")
