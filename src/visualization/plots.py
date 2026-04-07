"""Visualization utilities for the volatility forecasting dashboard."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COVID_START = "2020-02-01"
COVID_END = "2020-06-30"

_PALETTE: dict[str, str] = {
    "GARCH": "#4e79a7",
    "EGARCH": "#f28e2b",
    "GJR": "#e15759",
    "HAR": "#76b7b2",
    "RF": "#59a14f",
    "XGB": "#edc948",
    "Lasso": "#b07aa1",
    "EN": "#ff9da7",
    "Stacked": "#9c755f",
    "Regime": "#bab0ac",
    "Actual": "#333333",
    "VIX": "#d4a0a0",
}

_FAMILY_COLORS: dict[str, str] = {
    "VIX": "#d62728",
    "VRP": "#d62728",
    "GARCH": "#1f77b4",
    "EGARCH": "#1f77b4",
    "GJR": "#1f77b4",
    "HAR": "#2ca02c",
    "RV": "#ff7f0e",
    "GK": "#ff7f0e",
    "Return": "#ff7f0e",
    "Abs": "#ff7f0e",
}


def _model_color(name: str) -> str:
    """Look up palette color by model name prefix."""
    prefix = name.split("_")[0]
    return _PALETTE.get(prefix, "#aec7e8")


def _feature_color(feat: str) -> str:
    """Look up color for a feature by family keyword."""
    for key, color in _FAMILY_COLORS.items():
        if key in feat:
            return color
    return "#7f7f7f"


def plot_vol_timeseries(
    df: pd.DataFrame, train_cutoff: str, ax: plt.Axes
) -> None:
    """Plot full realized vol time series with GARCH overlay and event markers.

    Args:
        df: Full feature DataFrame (must contain RV_5; optionally GARCH_Vol, VIX).
        train_cutoff: Date string for the train/test split vertical line.
        ax: Axes to draw on.
    """
    ax.plot(df.index, df["RV_5"], color=_PALETTE["Actual"], lw=0.9, label="RV_5 (target)")
    for col, key in [("GARCH_Vol", "GARCH"), ("VIX", "VIX")]:
        if col in df:
            ax.plot(df.index, df[col], lw=0.6, alpha=0.7,
                    color=_PALETTE[key], label=key)
    ax.axvspan(COVID_START, COVID_END, alpha=0.12, color="red", label="COVID")
    ax.axvline(pd.Timestamp(train_cutoff), color="k", ls="--", lw=0.9, label="Train/test split")
    ax.set_title("Realized Volatility — SPY 2010–2025")
    ax.set_ylabel("Annualized Vol")
    ax.legend(fontsize=7, ncol=2)


def plot_metric_bar(
    scores: pd.Series,
    metric: str,
    ax: plt.Axes,
    log_scale: bool = False,
    ascending: bool = True,
) -> None:
    """Horizontal bar chart for a single evaluation metric.

    Args:
        scores: Series indexed by model name.
        metric: Display label for x-axis.
        ax: Axes to draw on.
        log_scale: If True, use log₁₀ x-axis.
        ascending: Sort order (ascending = best first for MSE/QLIKE).
    """
    sorted_scores = scores.sort_values(ascending=ascending)
    colors = [_model_color(n) for n in sorted_scores.index]
    sorted_scores.plot.barh(ax=ax, color=colors)
    ax.set_title(f"OOS {metric}")
    ax.set_xlabel(metric)
    if log_scale:
        ax.set_xscale("log")


def plot_forecast_vs_actual(
    y_true: pd.Series,
    preds: dict[str, pd.Series],
    ax: plt.Axes,
    top_n: int = 4,
) -> None:
    """Overlay actual realized vol with the top-N model forecasts.

    Args:
        y_true: Test-set realized volatility.
        preds: Dict of model forecasts (first top_n are plotted).
        ax: Axes to draw on.
        top_n: Maximum number of forecast lines to draw.
    """
    ax.plot(y_true.index, y_true.values, color=_PALETTE["Actual"],
            lw=1.2, label="Actual RV_5")
    for name, pred in list(preds.items())[:top_n]:
        aligned = pred.reindex(y_true.index)
        ax.plot(aligned.index, aligned.values, lw=0.7, alpha=0.8,
                color=_model_color(name), label=name)
    ax.set_title("OOS Forecast vs Actual")
    ax.set_ylabel("Annualized Vol")
    ax.legend(fontsize=7)


def plot_feature_importance(importances: pd.Series, ax: plt.Axes) -> None:
    """Horizontal bar chart of RF feature importances, color-coded by feature family.

    Args:
        importances: Series of importance scores indexed by feature name.
        ax: Axes to draw on.
    """
    sorted_imp = importances.sort_values()
    colors = [_feature_color(f) for f in sorted_imp.index]
    sorted_imp.plot.barh(ax=ax, color=colors)
    ax.set_title("RF Feature Importances (by family)")
    ax.set_xlabel("Importance")


def build_dashboard(
    df: pd.DataFrame,
    y_test: pd.Series,
    all_preds: dict[str, pd.Series],
    scores: pd.DataFrame,
    train_cutoff: str,
    rf_importances: pd.Series,
) -> plt.Figure:
    """Build the full 3×2 evaluation dashboard.

    Panel layout:
        [0,0] Full vol time series       [0,1] OOS MSE bar chart
        [1,0] OOS R² bar chart           [1,1] Forecast vs actual overlay
        [2,0] OOS QLIKE bar chart        [2,1] RF feature importances

    Args:
        df: Full feature DataFrame (for the time-series panel).
        y_test: Test-set realized volatility.
        all_preds: Dict of all model test-set predictions.
        scores: evaluate_all() output DataFrame.
        train_cutoff: Train/test split date string.
        rf_importances: RF feature importance Series.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle(
        "HAR-RV vs ML — SPY Volatility Forecast Dashboard",
        fontsize=14, y=0.995,
    )

    plot_vol_timeseries(df, train_cutoff, axes[0, 0])
    plot_metric_bar(scores["MSE"], "MSE", axes[0, 1], log_scale=True, ascending=True)
    plot_metric_bar(scores["R2"], "R²", axes[1, 0], ascending=False)
    plot_forecast_vs_actual(y_test, all_preds, axes[1, 1])
    plot_metric_bar(scores["QLIKE"], "QLIKE", axes[2, 0], ascending=True)
    plot_feature_importance(rf_importances, axes[2, 1])

    plt.tight_layout()
    return fig
