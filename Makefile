# make reproduce   rerun every horizon (about 2 minutes), write results/ and re-render the README
# make check       verify the README matches results/metrics.json (fast, used in CI)
# make test        run the tests
.PHONY: setup reproduce render check test

setup:
	uv sync

reproduce:
	uv run python scripts/reproduce.py
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
