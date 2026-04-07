"""Loss functions and evaluation utilities for volatility forecasting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss for volatility evaluation (Patton 2011).

    QLIKE = E[σ²/h - log(σ²/h) - 1], where h is the volatility forecast.

    Properties:
    - Strictly consistent for the conditional variance.
    - Asymmetric: penalizes underestimation more than overestimation.
    - Robust to the choice of realized-volatility proxy (up to scaling).

    Args:
        y_true: Realized volatility (annualized).
        y_pred: Volatility forecast (annualized).

    Returns:
        Scalar QLIKE loss (lower is better).
    """
    h = np.clip(y_pred, 1e-8, None)
    ratio = y_true / h
    return float(np.mean(ratio - np.log(ratio) - 1))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of time-steps where the forecast matches the realized direction.

    Expected value is ~0.50 for well-calibrated level forecasts of a
    near-martingale series. Not a useful primary metric on its own.

    Args:
        y_true: True realized volatility array.
        y_pred: Forecast array.

    Returns:
        Scalar in [0, 1].
    """
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def evaluate(
    y_true: pd.Series, y_pred: pd.Series, name: str = ""
) -> dict[str, float | str | int]:
    """Compute all evaluation metrics for a single model.

    NaN values in either series are excluded from all metrics.

    Args:
        y_true: True realized volatility Series.
        y_pred: Forecast Series (must share index with y_true).
        name: Model identifier (appears in returned dict).

    Returns:
        Dict with keys: Model, MSE, QLIKE, R2, DirAcc, N.
    """
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask].to_numpy()
    yp = y_pred[mask].to_numpy()

    return {
        "Model": name,
        "MSE": float(mean_squared_error(yt, yp)),
        "QLIKE": qlike(yt, yp),
        "R2": float(r2_score(yt, yp)),
        "DirAcc": directional_accuracy(yt, yp),
        "N": int(mask.sum()),
    }


def evaluate_all(
    y_true: pd.Series, predictions: dict[str, pd.Series]
) -> pd.DataFrame:
    """Evaluate all models and return a comparison table sorted by MSE.

    Args:
        y_true: True realized volatility Series.
        predictions: Dict of {model_name: forecast_series}.

    Returns:
        DataFrame indexed by model name, sorted by MSE ascending.
    """
    rows = [evaluate(y_true, pred, name) for name, pred in predictions.items()]
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("MSE")
