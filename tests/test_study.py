"""Checks that the study cannot look ahead and that its statistics are what they claim to be."""
import numpy as np
import pandas as pd
import pytest

from src import study


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return study.load()


def test_snapshot(df):
    assert df.index[0] == pd.Timestamp("1990-01-02") and df.index.is_monotonic_increasing
    assert df[["ret", "vix"]].notna().all().all() and len(df) > 9000


def test_features_use_only_the_past(df):
    full = study.features(df)
    cut = 5000
    part = study.features(df.iloc[: cut + 1])
    pd.testing.assert_series_equal(full.iloc[cut], part.iloc[cut])
    assert full.loc[df["r"] >= 0, "neg"].eq(0).all()
    np.testing.assert_allclose(full["rv_w"].iloc[100], (df["r"] ** 2).iloc[96:101].mean())


@pytest.mark.parametrize("h", [1, 5, 22])
def test_target_starts_the_day_after_and_does_not_overlap_features(df, h):
    y = study.target(df, h)
    t = 3000
    np.testing.assert_allclose(y.iloc[t], (df["r"] ** 2).iloc[t + 1: t + 1 + h].mean())
    assert y.iloc[-h:].isna().all() and y.iloc[:-h].notna().all()
    bumped = df.copy()
    bumped.iloc[t, bumped.columns.get_loc("r")] = 0.5      # a shock on day t
    # the shock does not change the target dated t (rolling sums leave float dust, hence approx)
    assert study.target(bumped, h).iloc[t] == pytest.approx(y.iloc[t], rel=1e-6)
    assert study.target(bumped, h).iloc[t - 1] > 100 * y.iloc[t - 1]
    assert study.features(bumped)["rv_d"].iloc[t] != study.features(df)["rv_d"].iloc[t]


def test_walk_forward_ignores_later_data(df):
    """Forecasts for a year must not change when everything after that year is altered."""
    cfg = study.Config(first_year=2001, rf=dict(n_estimators=5, random_state=0),
                       gbm=dict(max_iter=5, random_state=0))
    a = df.loc[:"2002-12-31"]
    b = a.copy()
    later = b.index.year == 2002
    b.loc[later, "r"] = b.loc[later, "r"] * 3.0
    pa, pb = study.walk_forward(a, 5, cfg), study.walk_forward(b, 5, cfg)
    cols = study.MODELS
    first = pa.index[pa.index.year == 2001][:-6]           # targets near year end reach into 2002
    pd.testing.assert_frame_equal(pa.loc[first, cols], pb.loc[first, cols])
    assert not np.allclose(pa.loc[pa.index.year == 2002, "HAR"], pb.loc[pb.index.year == 2002, "HAR"])


def test_garch_path_matches_a_direct_recursion():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    p = {"omega": 2e-6, "alpha": 0.05, "gamma": 0.08, "beta": 0.88}
    one = study.garch_path(r, p, 1)
    s2 = np.var(r)
    for t in range(500):
        s2 = p["omega"] + (p["alpha"] + p["gamma"] * (r[t] < 0)) * r[t] ** 2 + p["beta"] * s2
    assert one[-1] == pytest.approx(s2)
    pers = p["alpha"] + 0.5 * p["gamma"] + p["beta"]
    lr = p["omega"] / (1 - pers)
    k = [lr + pers ** j * (s2 - lr) for j in range(22)]
    assert study.garch_path(r, p, 22)[-1] == pytest.approx(np.mean(k))


def test_qlike_is_minimised_at_the_true_variance():
    rng = np.random.default_rng(1)
    y = (rng.normal(0, 0.01, 200000)) ** 2
    grid = np.array([0.5, 0.8, 1.0, 1.25, 2.0]) * 1e-4
    losses = [study.qlike(y, np.full_like(y, f)).mean() for f in grid]
    assert int(np.argmin(losses)) == 2
    assert np.isfinite(study.qlike(np.array([0.0]), np.array([1e-4]))).all()


def test_floor_and_diebold_mariano():
    np.testing.assert_allclose(study.floor_forecasts(np.array([-1.0, 1e-9, 2e-4]), 1e-5), [1e-5, 1e-5, 2e-4])
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 4000) ** 2
    # under the null the p-value is uniform: about 5 percent of draws fall below 0.05
    ps = [study.dm_test(base, base + rng.normal(0, 1, 4000), 1)["p_two_sided"] for _ in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12
    worse = study.dm_test(base + 0.2, base, 5)
    assert worse["t"] > 5 and worse["p_two_sided"] < 0.001 and worse["lags"] >= 4
    d = rng.normal(0.1, 1, 500)
    assert study.newey_west_t(d, 0) == pytest.approx(d.mean() / (d.std() / np.sqrt(500)))


def test_shar_features_split_yesterday_by_sign():
    import numpy as np
    import pandas as pd
    from src import study
    df = pd.DataFrame({"r": [0.01, -0.02, 0.03, -0.01] * 10, "vix": 20.0},
                      index=pd.bdate_range("2020-01-01", periods=40))
    f = study.features(df)
    assert np.allclose(f["neg"] + f["pos"], f["rv_d"])
    assert (f.loc[df["r"] > 0, "neg"] == 0).all() and (f.loc[df["r"] < 0, "pos"] == 0).all()
