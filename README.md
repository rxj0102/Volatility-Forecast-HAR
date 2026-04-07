# HAR-RV vs ML: SPY Volatility Forecasting

Can machine learning models statistically outperform HAR-RV (Corsi 2009) for next-day SPY realized volatility?

---

## Problem & Methodology

Daily realized volatility (5-day rolling annualized std of log-returns) is forecast using three model families evaluated on a strict temporal 80/20 train/test split:

| Family | Models |
|--------|--------|
| Parametric | GARCH(1,1)-t, EGARCH(1,1,1)-t, GJR-GARCH(1,1,1)-t |
| Econometric benchmark | HAR-RV — daily / weekly / monthly RV lags (Corsi 2009) |
| Machine learning | Lasso, ElasticNet, Random Forest, XGBoost |
| Meta-learners | Constrained stacking (SLSQP), regime-switching ensemble |

**Feature engineering** follows a strict lagging discipline (no look-ahead bias):
- AR lags of RV (1, 5, 10 days)
- GARCH conditional volatility lags
- Variance Risk Premium (`VIX − RV_5`, lagged)
- Garman-Klass OHLC estimator (8× more efficient than close-to-close)

**Statistical evaluation** goes beyond point metrics:
- **MSE** (primary) and **QLIKE** loss (Patton 2011 — penalizes vol underestimation)
- **HAC-corrected Diebold-Mariano tests** (Harvey, Leybourne & Newbold 1997)
- **Stress tests**: genuine OOS retrain on pre-COVID data, evaluated on 2020 crisis
- **Calm-period test**: low-vol regime 2024–2025

---

## Key Results

| Finding | Detail |
|---------|--------|
| HAR-RV is statistically indistinguishable from the best ML model | DM test p > 0.45 (HAC-corrected) on both MSE and QLIKE |
| Meta-learner degenerates to a single model | SLSQP assigns near-unity weight to ElasticNet — nonlinear models add zero incremental value |
| Regime-switching adds no measurable improvement | Equal-weight ensemble performs comparably |
| HAR generalizes better under COVID crisis | Long-memory structure is more robust to vol-spike regime shifts than tree-based models |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data | `yfinance` |
| GARCH models | `arch` |
| Statistical tests | `statsmodels`, `scipy` |
| ML models | `scikit-learn`, `xgboost` |
| Meta-learner optimization | `scipy.optimize.minimize` (SLSQP) |
| Visualization | `matplotlib`, `seaborn` |

---

## Project Structure

```
Volatility-Forecast-HAR/
├── src/
│   ├── data/
│   │   └── loader.py           # yfinance download + realized vol feature engineering
│   ├── models/
│   │   ├── garch.py            # GARCH / EGARCH / GJR-GARCH (Student-t innovations)
│   │   ├── har.py              # HAR-RV model (Corsi 2009)
│   │   └── ml.py               # Base learners, OOF stacking, regime-switching
│   ├── evaluation/
│   │   ├── metrics.py          # MSE, QLIKE, DirAcc, evaluate_all()
│   │   └── dm_test.py          # HAC-corrected Diebold-Mariano test
│   └── visualization/
│       └── plots.py            # Dashboard + individual plot utilities
├── notebooks/
│   └── 01_vol_forecast_har.ipynb   # End-to-end analysis
├── data/                       # Local data cache (gitignored)
├── tests/
│   ├── test_metrics.py         # Unit tests: loss functions
│   └── test_har.py             # Unit tests: HAR feature construction + fitting
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Run the analysis notebook**

```bash
jupyter lab notebooks/01_vol_forecast_har.ipynb
```

**3. Run the test suite**

```bash
pytest tests/ -v
```

**4. Use modules directly**

```python
from src.data.loader import download_spy_vix, build_features
from src.models.har import build_har_features, HARModel
from src.evaluation.metrics import evaluate_all
from src.evaluation.dm_test import dm_test_all

df = download_spy_vix("2010-01-01", "2025-12-31")
df = build_features(df)

har_feats = build_har_features(df["RV_Daily"])
# ... fit HAR and ML models, then:
scores = evaluate_all(y_test, all_predictions)
dm_results = dm_test_all(y_test, har_pred, ml_preds, loss="mse")
```

---

## Research-Level Improvements

### 1. Intraday Realized Variance via Tick Data
Replace daily close-to-close log-returns with **5-minute intraday realized variance** (Barndorff-Nielsen & Shephard 2002). Intraday RV is a near-perfect proxy for latent quadratic variation and dramatically reduces measurement error in the target variable and all HAR features — likely widening any detectable performance gap between models.

### 2. HAR-RV-J: Decompose Continuous vs Jump Variation
Extend HAR-RV to include a **jump component** via bipower variation (Andersen, Bollerslev & Diebold 2007): `RV = BV + J+`, where `J+ = max(RV - BV, 0)`. The jump and continuous components have distinct autocorrelation structures; HAR-RV-J has shown consistent OOS gains over plain HAR, especially around macro announcements and earnings releases.

### 3. Cross-Asset Features + Long-Range Attention Model
Augment the feature set with cross-asset volatility spillovers (VIX term structure slope, realized vol of sector ETFs, IG/HY credit spreads) and benchmark a **temporal self-attention model** (PatchTST or Informer) against HAR. This directly tests whether long-range dependencies beyond HAR's 22-day memory exist and are learnable from market data — the most likely path to beating HAR on equity volatility.
