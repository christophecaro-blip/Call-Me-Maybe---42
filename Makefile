PYTHON = python
.DEFAULT_GOAL := run

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Error: uv is not installed."; \
		echo "Install it without admin rights with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "Then restart your shell (or 'source \$$HOME/.local/bin/env')."; \
		exit 1; \
	}

install: check-uv
	uv sync

run: install
	uv run $(PYTHON) -m src

debug: check-uv
	uv run $(PYTHON) -m pdb -m src

clean:
	find . -path ./.venv -prune -o -type d \
		\( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) \
		-exec rm -rf {} +

lint: install
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict: install
	uv run flake8 .
	uv run mypy . --strict

.PHONY: check-uv install run debug clean lint lint-strict
