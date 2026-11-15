"""Does machine learning beat HAR at forecasting equity variance?

Design, fixed before the run:

  target      average daily variance over the next h trading days, h = 1, 5, 22, measured by
              squared daily log returns of the US market. The target never overlaps the data
              a forecast is built from.
  features    known at the close of day t: squared return of day t, its 5 and 22 day means
              (the HAR cascade of Corsi 2009), squared VIX in daily variance units, and the
              squared return of day t if it was negative.
  protocol    expanding window, every model refitted each January on all earlier days whose
              target is fully realised, first forecast year 2000.
  loss        QLIKE (Patton 2011), which ranks variance forecasts correctly even when the
              target is a noisy proxy such as a squared return; MSE is reported beside it.
  test        Diebold-Mariano against HAR with a Newey-West variance.

Squared daily returns are a noisy measure of variance. That lowers every R squared but does
not bias the ranking under QLIKE or MSE, which is why those two losses are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "snapshot" / "market_daily.csv"
HAR = ["rv_d", "rv_w", "rv_m"]
HARX = [*HAR, "vix2", "neg"]
MODELS = ["No change", "EWMA", "GARCH", "GJR-GARCH", "VIX", "HAR", "HAR-X", "Lasso",
          "Random forest", "Gradient boosting", "HAR-X + boosting"]
ML_MODELS = ["Lasso", "Random forest", "Gradient boosting"]
SEED = 42


@dataclass
class Config:
    horizons: tuple[int, ...] = (1, 5, 22)
    first_year: int = 2000
    ewma_lambda: float = 0.94
    rf: dict = field(default_factory=lambda: dict(n_estimators=300, min_samples_leaf=20,
                                                  max_features=0.5, n_jobs=-1, random_state=SEED))
    gbm: dict = field(default_factory=lambda: dict(max_depth=3, learning_rate=0.05, max_iter=300,
                                                   min_samples_leaf=50, random_state=SEED))


def load() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, index_col="date", parse_dates=True)
    df["r"] = np.log1p(df["ret"])
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything here uses data up to and including day t."""
    r2 = df["r"] ** 2
    f = pd.DataFrame(index=df.index)
    f["rv_d"] = r2
    f["rv_w"] = r2.rolling(5).mean()
    f["rv_m"] = r2.rolling(22).mean()
    f["vix2"] = (df["vix"] / 100.0) ** 2 / 252.0
    f["neg"] = r2.where(df["r"] < 0, 0.0)
    return f


def target(df: pd.DataFrame, h: int) -> pd.Series:
    """Mean squared return over days t+1 .. t+h, indexed by t."""
    r2 = df["r"] ** 2
    return r2.rolling(h).mean().shift(-h)


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE: log f + y / f. Defined for y = 0, minimised in expectation at f = E[y]."""
    return np.log(f) + y / f


def floor_forecasts(pred: np.ndarray, floor: float) -> np.ndarray:
    """A variance forecast must be positive. Values below the floor are set to the floor.

    The floor is the lowest EWMA variance in the training sample. Linear models in levels can
    forecast a variance near or below zero on calm days, and QLIKE punishes that without bound.
    """
    return np.maximum(pred, floor)


def garch_path(r: np.ndarray, params: dict, h: int) -> np.ndarray:
    """h-day average variance forecast made at each day, for fixed GJR-GARCH parameters.

    r is demeaned returns. sigma2[t+1] is known at t; the k-step forecast reverts to the long-run
    variance at rate (alpha + gamma / 2 + beta).
    """
    w, a, g, b = params["omega"], params["alpha"], params["gamma"], params["beta"]
    pers = a + 0.5 * g + b
    lr = w / max(1.0 - pers, 1e-6)
    n = len(r)
    nxt = np.empty(n)
    s2 = np.var(r)
    for t in range(n):
        s2 = w + (a + g * (r[t] < 0)) * r[t] ** 2 + b * s2
        nxt[t] = s2
    k = np.arange(h)
    weights = (pers ** k).mean()
    return lr + (nxt - lr) * weights


def fit_garch(r: np.ndarray, asymmetric: bool) -> dict:
    """Gaussian QMLE through the arch package; returns parameters in return units."""
    scale = 100.0
    res = arch_model(r * scale, mean="Zero", vol="GARCH", p=1, o=1 if asymmetric else 0, q=1).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]) / scale**2, "alpha": float(p["alpha[1]"]),
            "gamma": float(p["gamma[1]"]) if asymmetric else 0.0, "beta": float(p["beta[1]"])}


def walk_forward(df: pd.DataFrame, h: int, cfg: Config) -> pd.DataFrame:
    """Out-of-sample forecasts of every model, one row per forecast day, with the target."""
    X = features(df)
    y = target(df, h)
    r = df["r"].to_numpy()
    idx = df.index
    ewma = (df["r"] ** 2).ewm(alpha=1 - cfg.ewma_lambda, adjust=False).mean()
    out = []
    for year in range(cfg.first_year, idx.year.max() + 1):
        test = (idx.year == year) & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        if not test.any():
            continue
        first_test = int(np.argmax(idx.year == year))
        # a training target must be fully realised before the first forecast day of the year
        train = np.zeros(len(idx), dtype=bool)
        train[: max(first_test - h, 0)] = True
        train &= X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        Xtr, ytr, Xte = X[train], y[train].to_numpy(), X[test]
        # no forecast may be below the calmest smoothed variance seen in training
        floor = float(ewma.iloc[22:first_test].min())
        p = pd.DataFrame(index=idx[test])
        p["y"] = y[test].to_numpy()
        p["No change"] = (df["r"] ** 2).rolling(h).mean()[test].to_numpy()
        p["EWMA"] = ewma[test].to_numpy()
        mean_r = r[:first_test].mean()
        for name, asym in (("GARCH", False), ("GJR-GARCH", True)):
            params = fit_garch(r[:first_test] - mean_r, asym)
            p[name] = garch_path(r - mean_r, params, h)[test]
        p["VIX"] = LinearRegression().fit(Xtr[["vix2"]], ytr).predict(Xte[["vix2"]])
        p["HAR"] = LinearRegression().fit(Xtr[HAR], ytr).predict(Xte[HAR])
        p["HAR-X"] = LinearRegression().fit(Xtr[HARX], ytr).predict(Xte[HARX])
        lasso = make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(5), random_state=SEED))
        p["Lasso"] = lasso.fit(Xtr[HARX], ytr).predict(Xte[HARX])
        p["Random forest"] = RandomForestRegressor(**cfg.rf).fit(Xtr[HARX], ytr).predict(Xte[HARX])
        p["Gradient boosting"] = HistGradientBoostingRegressor(**cfg.gbm).fit(Xtr[HARX], ytr).predict(Xte[HARX])
        raw = p[MODELS[:-1]].to_numpy()
        p[MODELS[:-1]] = floor_forecasts(raw, floor)
        for j, m in enumerate(MODELS[:-1]):
            p[f"floored:{m}"] = raw[:, j] < floor
        p["HAR-X + boosting"] = 0.5 * (p["HAR-X"] + p["Gradient boosting"])
        out.append(p)
    return pd.concat(out)


def newey_west_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    e = d - d.mean()
    v = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(d.mean() / np.sqrt(v / n))


def dm_test(loss_model: np.ndarray, loss_bench: np.ndarray, h: int) -> dict:
    """Diebold-Mariano: negative t means the model has lower loss than the benchmark."""
    d = loss_model - loss_bench
    lags = max(h - 1, int(np.floor(4 * (len(d) / 100.0) ** (2.0 / 9.0))))
    t = newey_west_t(d, lags)
    return {"t": t, "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(t)))), "lags": lags}


def evaluate(p: pd.DataFrame, h: int, bench: str = "HAR") -> pd.DataFrame:
    y = p["y"].to_numpy()
    rows = []
    for m in MODELS:
        f = p[m].to_numpy()
        ql, se = qlike(y, f), (y - f) ** 2
        row = {"model": m, "qlike": float(ql.mean()), "mse": float(se.mean()),
               "r2": float(1 - se.sum() / ((y - y.mean()) ** 2).sum())}
        if m != bench:
            b = p[bench].to_numpy()
            dq, dm = dm_test(ql, qlike(y, b), h), dm_test(se, (y - b) ** 2, h)
            row.update({"dm_t_qlike": dq["t"], "dm_p_qlike": dq["p_two_sided"],
                        "dm_t_mse": dm["t"], "dm_p_mse": dm["p_two_sided"]})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    out["qlike_vs_har"] = out["qlike"] - out.loc[bench, "qlike"]
    return out
