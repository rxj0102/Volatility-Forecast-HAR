"""Unit tests for loss functions and evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import directional_accuracy, evaluate, evaluate_all, qlike


def test_qlike_perfect_forecast():
    """QLIKE should be 0 when forecast equals truth."""
    y = np.array([0.1, 0.2, 0.15])
    assert qlike(y, y) == pytest.approx(0.0, abs=1e-8)


def test_qlike_underestimation_penalized_more():
    """QLIKE penalizes underestimation more than equal-magnitude overestimation."""
    y = np.full(10, 0.2)
    under = np.full(10, 0.1)   # underestimate by 0.1
    over = np.full(10, 0.3)    # overestimate by 0.1
    assert qlike(y, under) > qlike(y, over)


def test_qlike_positive():
    """QLIKE is non-negative for any finite inputs."""
    rng = np.random.default_rng(0)
    y = rng.uniform(0.05, 0.5, 100)
    yhat = rng.uniform(0.05, 0.5, 100)
    assert qlike(y, yhat) >= 0.0


def test_directional_accuracy_perfect():
    """Monotone increasing series should give 100% directional accuracy."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert directional_accuracy(y, y) == pytest.approx(1.0)


def test_directional_accuracy_opposite():
    """Monotone decreasing forecast for increasing truth gives 0% accuracy."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_rev = np.array([4.0, 3.0, 2.0, 1.0])
    assert directional_accuracy(y, y_rev) == pytest.approx(0.0)


def test_directional_accuracy_range():
    """Directional accuracy must be in [0, 1]."""
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 50)
    yhat = rng.uniform(0, 1, 50)
    acc = directional_accuracy(y, yhat)
    assert 0.0 <= acc <= 1.0


def test_evaluate_returns_expected_keys():
    """evaluate() must return a dict with all required metric keys."""
    y = pd.Series([0.1, 0.2, 0.15, 0.18, 0.12])
    result = evaluate(y, y * 1.05, name="test_model")
    assert set(result.keys()) == {"Model", "MSE", "QLIKE", "R2", "DirAcc", "N"}


def test_evaluate_n_excludes_nan():
    """evaluate() N should count only non-NaN pairs."""
    y = pd.Series([0.1, np.nan, 0.15, 0.18, 0.12])
    yhat = pd.Series([0.1, 0.2, np.nan, 0.18, 0.12])
    result = evaluate(y, yhat, name="test")
    assert result["N"] == 3  # only indices 0, 3, 4 are non-NaN in both


def test_evaluate_all_sorted_by_mse():
    """evaluate_all() output must be sorted by MSE ascending."""
    y = pd.Series(np.linspace(0.1, 0.4, 50))
    preds = {
        "good": y * 1.01,
        "bad": y * 2.0,
        "medium": y * 1.2,
    }
    result = evaluate_all(y, preds)
    mse_values = result["MSE"].tolist()
    assert mse_values == sorted(mse_values)
