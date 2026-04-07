"""ML base learners and meta-learner (stacking + regime-switching) for volatility forecasting."""

import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
N_SPLITS = 5
ALPHA_GRID = np.logspace(-9, 0, 100)

# ML feature columns (all lagged ≥1 day; set after GARCH/HAR fits)
ML_FEATURES = [
    "RV_lag1", "RV_lag5", "RV_lag10",
    "GARCH_lag1", "EGARCH_lag1", "GJR_lag1",
    "HAR_D_lag1", "HAR_W_lag1", "HAR_M_lag1",
    "VRP_lag1", "VRP_lag5",
    "AbsReturn_lag1", "Return_lag1_sq",
    "GK_Vol_lag1",
    "HAR_Pred_IS_lag1",
]


def build_base_learners() -> dict[str, object]:
    """Return fresh (unfitted) base learner instances.

    All hyperparameters are fixed via prior grid-search across similar
    equity volatility forecasting tasks.

    Returns:
        Dict of {model_name: estimator}.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    return {
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=SEED,
        ),
        "XGB": XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            verbosity=0,
        ),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(alphas=ALPHA_GRID, cv=tscv, max_iter=20_000)),
        ]),
        "EN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=ALPHA_GRID,
                cv=tscv,
                max_iter=20_000,
            )),
        ]),
    }


def fit_predict_base(
    learners: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Fit base learners on training data and predict on the test set.

    Args:
        learners: Dict of unfitted estimators (modified in-place — cloned internally).
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.

    Returns:
        Dict of {model_name: test-set prediction Series}.
    """
    predictions: dict[str, pd.Series] = {}
    for name, model in learners.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = pd.Series(preds, index=X_test.index, name=f"{name}_Pred")
    return predictions


def compute_oof_predictions(
    learners: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Generate out-of-fold (OOF) predictions for meta-learner training.

    Uses TimeSeriesSplit so that each validation fold only sees past data.
    OOF predictions prevent leakage: the meta-learner never sees a base
    model's prediction on its own training data.

    Args:
        learners: Dict of unfitted estimators (deep-copied per fold).
        X_train: Training feature matrix.
        y_train: Training target Series.

    Returns:
        DataFrame of OOF predictions, shape (n_train_rows, n_models).
        Rows that were never in a validation fold remain NaN.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = pd.DataFrame(np.nan, index=X_train.index, columns=list(learners))

    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        for name, model in learners.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            oof.loc[X_val.index, name] = m.predict(X_val)

    return oof.astype(float)


def fit_stacking_weights(
    oof_preds: pd.DataFrame,
    y_oof: pd.Series,
    n_starts: int = 10,
) -> np.ndarray:
    """Constrained SLSQP optimization for convex stacking weights.

    Minimizes OOF MSE subject to w ≥ 0 and Σw = 1 (forecast is a
    non-negative convex combination of base learners).

    Multi-start Dirichlet initialization avoids local minima.

    Args:
        oof_preds: OOF prediction matrix (n_samples, n_models).
        y_oof: True target values aligned with OOF rows.
        n_starts: Number of random restarts.

    Returns:
        Optimal weight vector (n_models,).
    """
    mask = oof_preds.notna().all(axis=1) & y_oof.notna()
    P = oof_preds[mask].values
    y = y_oof[mask].values
    n = P.shape[1]

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0.0, 1.0)] * n
    rng = np.random.default_rng(SEED)

    best_val, best_w = np.inf, np.ones(n) / n
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n))
        res = minimize(
            lambda w: float(np.mean((P @ w - y) ** 2)),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success and res.fun < best_val:
            best_val, best_w = res.fun, res.x

    return best_w


def blend_predictions(preds: dict[str, pd.Series], weights: np.ndarray) -> pd.Series:
    """Compute a weighted blend of model predictions.

    Args:
        preds: Ordered dict of prediction Series (order must match weights).
        weights: Weight vector aligned with preds.values().

    Returns:
        Blended prediction Series.
    """
    pred_matrix = pd.DataFrame(preds)
    blended = pred_matrix.values @ weights
    return pd.Series(blended, index=pred_matrix.index, name="Stacked_Pred")


def regime_switch_ensemble(
    rv_lagged: pd.Series,
    econometric_preds: dict[str, pd.Series],
    ml_preds: dict[str, pd.Series],
    lookback: int = 64,
) -> pd.Series:
    """Regime-switching ensemble using lagged volatility level.

    Regime assignment (no look-ahead — uses yesterday's RV):
    - High-vol: rv_lagged > rolling lookback-day median → use ML blend
    - Low-vol:  rv_lagged ≤ median                      → use econometric blend

    The intuition is that nonlinear models may add value in high-stress
    regimes where the linear HAR structure breaks down.

    Args:
        rv_lagged: Realized volatility Series already shifted 1 day.
        econometric_preds: Dict of econometric model forecasts.
        ml_preds: Dict of ML model forecasts.
        lookback: Rolling window (days) for the median threshold.

    Returns:
        Regime-switching prediction Series.
    """
    rolling_median = rv_lagged.rolling(lookback).median()
    high_vol = rv_lagged > rolling_median

    eco_blend = pd.DataFrame(econometric_preds).mean(axis=1)
    ml_blend = pd.DataFrame(ml_preds).mean(axis=1)

    regime_pred = np.where(high_vol, ml_blend, eco_blend)
    return pd.Series(regime_pred, index=rv_lagged.index, name="Regime_Pred")
