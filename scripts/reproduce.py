"""Run the study for every horizon and write results/.

  results/metrics.json          everything the README reports
  results/losses_h<h>.csv       per-model losses and Diebold-Mariano tests against HAR
  results/figures/qlike.png     QLIKE of each model relative to HAR, by horizon
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import study  # noqa: E402

PERIODS = [("2000-2007", "2000", "2007"), ("2008-2009", "2008", "2009"), ("2010-2019", "2010", "2019"),
           ("2020", "2020", "2020"), ("2021-2026", "2021", "2026")]


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def figure(tables: dict[int, pd.DataFrame], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface, blue, grey = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb", "#2a78d6", "#8a8980"
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": ink,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    models = [m for m in study.MODELS if m != "HAR"][::-1]
    fig, axes = plt.subplots(1, len(tables), figsize=(11, 4.2), sharey=True)
    for ax, (h, t) in zip(axes, tables.items()):
        vals = np.clip(t.loc[models, "qlike_vs_har"].to_numpy(), -0.15, 0.6)
        ax.barh(models, vals, color=[blue if m in study.ML_MODELS else grey for m in models], height=0.62)
        ax.axvline(0, color=muted, linewidth=0.8)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{h} day horizon", loc="left", fontsize=10.5)
        ax.set_xlim(-0.15, 0.6)
    fig.suptitle("QLIKE loss relative to HAR (left of zero is better than HAR; machine learning in blue; "
                 "bars are cut at 0.6)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "qlike.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = study.Config()
    df = study.load()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    payload = {"sample": {"first_day": str(df.index[0].date()), "last_day": str(df.index[-1].date()),
                          "n_days": int(len(df)), "first_forecast_year": cfg.first_year},
               "horizons": {}}
    tables = {}
    for h in cfg.horizons:
        p = study.walk_forward(df, h, cfg)
        t = study.evaluate(p, h)
        tables[h] = t
        g = study.evaluate(p, h, bench="GJR-GARCH")
        t.to_csv(out / f"losses_h{h}.csv", float_format="%.6g")
        y = p["y"].to_numpy()
        periods = {}
        for lab, a, b in PERIODS:
            sub = p.loc[a:b]
            ql = {m: float(study.qlike(sub["y"].to_numpy(), sub[m].to_numpy()).mean()) for m in study.MODELS}
            periods[lab] = {m: ql[m] - ql["HAR"] for m in study.MODELS if m != "HAR"}
        payload["horizons"][str(h)] = {
            "n_forecasts": int(len(p)), "first": str(p.index[0].date()), "last": str(p.index[-1].date()),
            "models": t.to_dict(orient="index"),
            "floored_share": {m: float(p[f"floored:{m}"].mean()) for m in study.MODELS[:-1]},
            "periods": periods,
            # added after the first run showed GJR-GARCH, not HAR, was the model to beat
            "ml_vs_gjr_qlike": {m: {"t": float(g.loc[m, "dm_t_qlike"]), "p": float(g.loc[m, "dm_p_qlike"])}
                                for m in study.ML_MODELS},
            # added with SHAR after a literature review: does HAR with the sign split catch GJR-GARCH?
            "shar_vs_gjr_qlike": {"t": float(g.loc["SHAR", "dm_t_qlike"]), "p": float(g.loc["SHAR", "dm_p_qlike"])},
            "best_qlike": str(t["qlike"].idxmin()), "best_mse": str(t["mse"].idxmin()),
            "target_mean_annual_vol": float(np.sqrt(y.mean() * 252)),
        }
        print(f"h={h}\n{t.round(4).to_string()}")
    (out / "metrics.json").write_text(json.dumps(rounded(payload), indent=2) + "\n")
    figure(tables, out / "figures")


if __name__ == "__main__":
    main()
