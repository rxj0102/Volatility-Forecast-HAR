"""Unit tests for HAR-RV model construction and fitting."""

import numpy as np
import pandas as pd
import pytest

from src.models.har import HARModel, build_har_features


def _make_rv_series(n: int = 200) -> pd.Series:
    """Synthetic daily RV series with realistic autocorrelation."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(rng.uniform(0.08, 0.45, n), index=idx, name="RV_Daily")


def test_build_har_features_columns():
    rv = _make_rv_series()
    har = build_har_features(rv)
    assert set(har.columns) == {"HAR_D", "HAR_W", "HAR_M"}


def test_build_har_features_no_lookahead():
    """HAR_D at time t must equal RV_{t-1} (strictly lagged)."""
    rv = _make_rv_series(80)
    har = build_har_features(rv)
    # Pick a row safely past the 22-day warmup
    safe_idx = har.index[5]
    rv_pos = rv.index.get_loc(safe_idx)
    assert har.loc[safe_idx, "HAR_D"] == pytest.approx(rv.iloc[rv_pos - 1])


def test_build_har_features_length():
    """After dropping NaN from 22-day monthly average, length should be n - 22."""
    n = 100
    rv = _make_rv_series(n)
    har = build_har_features(rv)
    assert len(har) == n - 22


def test_har_model_fit_predict_no_nan():
    """Predictions on held-out data should contain no NaN values."""
    rv = _make_rv_series(200)
    har_feats = build_har_features(rv)
    target = rv.reindex(har_feats.index).dropna()
    har_feats = har_feats.loc[target.index]

    split = int(len(har_feats) * 0.8)
    model = HARModel()
    model.fit(har_feats.iloc[:split], target.iloc[:split])
    preds = model.predict(har_feats.iloc[split:])

    assert isinstance(preds, pd.Series)
    assert not preds.isna().any()
    assert len(preds) == len(har_feats) - split


def test_har_model_coef_keys():
    """Fitted coef_ dict must contain all HAR components plus Intercept."""
    rv = _make_rv_series(100)
    har_feats = build_har_features(rv)
    target = rv.reindex(har_feats.index).dropna()
    model = HARModel().fit(har_feats.loc[target.index], target)
    assert set(model.coef_.keys()) == {"HAR_D", "HAR_W", "HAR_M", "Intercept"}


def test_har_model_score_positive():
    """Train R² on a sufficient dataset should be positive."""
    rv = _make_rv_series(300)
    har_feats = build_har_features(rv)
    target = rv.reindex(har_feats.index).dropna()
    model = HARModel().fit(har_feats.loc[target.index], target)
    assert model.score(har_feats.loc[target.index], target) > 0.0


def test_har_insample_prediction_shifted():
    """In-sample prediction should be shifted 1 day (first value NaN)."""
    rv = _make_rv_series(100)
    har_feats = build_har_features(rv)
    target = rv.reindex(har_feats.index).dropna()
    model = HARModel().fit(har_feats.loc[target.index], target)
    is_pred = model.predict_insample(har_feats.loc[target.index])
    assert np.isnan(is_pred.iloc[0])
    assert not np.isnan(is_pred.iloc[1])
