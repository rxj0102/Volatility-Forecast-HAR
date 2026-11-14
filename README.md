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

**Statistical evaluation** goes beyond point metrics:
- **MSE** (primary) and **QLIKE** loss (Patton 2011 — penalizes vol underestimation)
- **HAC-corrected Diebold-Mariano tests** (Harvey, Leybourne & Newbold 1997)
- **Stress tests**: genuine OOS retrain on pre-COVID data, evaluated on 2020 crisis
- **Calm-period test**: low-vol regime 2024–2025

---

## Data

Open and free, no API key:

| Series | Source |
|--------|--------|
| SPY stand in | Ken French Data Library, US market total return (Mkt-RF + RF), daily |
| VIX | FRED VIXCLS |

Sample: 2010-01-04 to 2025-12-31 (4,024 days). Train 2010-02-25 to 2022-10-21 (3,188 rows), test 2022-10-24 to 2025-12-29 (798 rows). Only closes are available from these sources, so the Garman-Klass feature from the earlier Yahoo version is not used.

---

## Key Results

All numbers are copied from the executed notebook `notebooks/01_vol_forecast_har.ipynb` (identical to `05_vol_forecast_har.ipynb`).

**Out-of-sample test (798 days)**

| Model | MSE | QLIKE | R2 |
|-------|-----|-------|----|
| HAR-RV | 2.167e-03 | 0.0498 | 0.727 |
| Lasso / ElasticNet / Stacked | 2.443e-03 | 0.0547 | 0.692 |
| Equal weight ensemble | 2.514e-03 | 0.0550 | 0.683 |
| Regime switching | 2.637e-03 | 0.0627 | 0.668 |
| XGBoost | 2.685e-03 | 0.0575 | 0.662 |
| Random Forest | 2.856e-03 | 0.0588 | 0.640 |
| GJR-GARCH-t | 3.596e-03 | 0.0747 | 0.547 |
| EGARCH-t | 3.868e-03 | 0.0788 | 0.513 |
| GARCH-t | 4.493e-03 | 0.0836 | 0.434 |

| Finding | Detail |
|---------|--------|
| HAR-RV has the lowest test MSE and QLIKE | Best ML model (Lasso) has 12.8% higher MSE, but the HAC Diebold-Mariano test cannot separate them (p = 0.200 MSE, 0.102 QLIKE) |
| HAR significantly beats GARCH family and tree models | HAC DM p < 0.05 against GARCH, EGARCH, GJR, Random Forest and XGBoost on both losses |
| Meta-learner degenerates to the linear models | SLSQP weights: ElasticNet 0.68, Lasso 0.32, RF and XGB 0.00 |
| Regime switching does not help | 4.9% higher MSE than the equal weight ensemble |
| HAR generalizes best under COVID | Retrained on pre-2020 data and tested Feb to Jun 2020: HAR MSE 9.78e-03 vs ElasticNet 1.41e-02 (44% higher) and Random Forest 6.93e-02 |
| Calm period 2024 to 2025 | Same ranking: HAR R2 0.751 vs Lasso 0.721 |

HAR-RV coefficients (train): beta_D 0.068, beta_W 0.993, beta_M 0.067; train R2 0.852, test R2 0.727.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data | FRED and Ken French Data Library (`src/data/opendata.py`) |
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
│   │   ├── loader.py           # open data load + realized vol feature engineering
│   │   └── opendata.py         # FRED and Ken French downloaders (cached)
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
├── data/                       # Local data cache, data/open_cache (gitignored)
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

## Data snapshot

`data/open_data_manifest.json` records every open data file behind the results above (SHA-256, rows, first and last date). Ken French and FRED occasionally revise history; after a fresh download, check whether the inputs still match with:

```bash
python -c "from src.data.opendata import verify_manifest; print(verify_manifest('data/open_data_manifest.json'))"
```
