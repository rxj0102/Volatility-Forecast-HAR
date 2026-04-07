"""GARCH family models for conditional volatility estimation."""

import numpy as np
import pandas as pd
from arch import arch_model

ANNUALIZE = np.sqrt(252)

# All models use Student-t innovations to accommodate fat tails in equity returns.
_SPECS = {
    "GARCH": dict(vol="Garch", p=1, q=1),
    "EGARCH": dict(vol="EGARCH", p=1, q=1, o=1),
    "GJR": dict(vol="GARCH", p=1, q=1, o=1),
}


def fit_garch_family(
    returns: pd.Series, verbose: bool = True
) -> dict[str, pd.Series]:
    """Fit GARCH(1,1), EGARCH(1,1,1), and GJR-GARCH(1,1,1) with Student-t innovations.

    All three models capture volatility clustering; EGARCH and GJR additionally
    model the leverage effect (negative returns raise volatility more than
    positive returns of equal magnitude).

    Args:
        returns: Daily log-return series (not pre-annualized).
        verbose: If True, print persistence and leverage diagnostics.

    Returns:
        Dictionary mapping model name → annualized conditional volatility Series.
    """
    scaled = returns * 100  # arch library expects returns in percent
    results: dict[str, pd.Series] = {}

    for name, kwargs in _SPECS.items():
        model = arch_model(scaled, mean="Constant", dist="t", **kwargs)
        fit = model.fit(disp="off", show_warning=False)
        cond_vol = fit.conditional_volatility / 100 * ANNUALIZE
        results[name] = pd.Series(cond_vol, index=returns.index, name=f"{name}_Vol")

        if verbose:
            _print_diagnostics(name, fit)

    return results


def _print_diagnostics(name: str, fit) -> None:
    """Print persistence and leverage diagnostics for a fitted GARCH model."""
    p = fit.params
    if name == "GARCH":
        persistence = p.get("alpha[1]", 0) + p.get("beta[1]", 0)
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        print(f"  [{name}] persistence={persistence:.4f}, half-life={half_life:.1f}d")
    else:
        gamma = p.get("gamma[1]", p.get("o[1]", None))
        if gamma is not None:
            direction = "confirmed (leverage)" if gamma < 0 else "positive (anti-leverage)"
            print(f"  [{name}] leverage γ={gamma:.4f} — {direction}")
