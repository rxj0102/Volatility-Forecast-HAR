# make reproduce   rerun the full analysis, write results/ and re-render the README tables
# make check       verify README Key Results match results/ (fast, used in CI)
# make test        run the unit tests
.PHONY: setup reproduce render check test

setup:
	uv sync

reproduce:
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/01_vol_forecast_har.ipynb
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
