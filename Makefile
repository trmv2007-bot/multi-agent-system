.PHONY: install test browser-test run lint sast typecheck verify eval clean

# Default target
all: install verify run

# Install dependencies
install:
	pip install -e ".[dev]"

# Run the multi-agent system
run:
	python -m src.main "Analyze the current state of AI agents in 2026"

# Run with custom query
run-custom:
	python -m src.main "$(QUERY)"

# Team mode: plan, fan out to parallel subagents, synthesize
run-team:
	python -m src.main "$(QUERY)" --team

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Browser test: the dashboard's real page in Chromium, on a scripted model, so
# no key, gateway or network is needed. Install the browser once with
# `python -m playwright install chromium`.
browser-test:
	python -m pytest e2e -q

# Score the answers against the eval cases. No flag runs the credential-free
# stub suite (what CI runs); `make eval ARGS=--live` runs the real cases
# against the KeyPool gateway.
eval:
	python -m evals.run $(ARGS)

# Lint
lint:
	ruff check src/ tests/ evals/ e2e/
	ruff format --check src/ tests/ evals/ e2e/

# Type check
typecheck:
	mypy src/ evals/ e2e/

# SAST scan
sast:
	semgrep scan --config=p/security-audit src/
	bandit -c pyproject.toml -r src/

# Full verification pipeline
verify: sast typecheck lint test
	@echo "✅ All verification gates passed!"

# Clean
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Start MCP server (for tools)
mcp-server:
	python -m src.mcp.server

# Start A2A server (for external agents)
a2a-server:
	python -m src.a2a.server

# Development: run with auto-reload
dev:
	watchgod -p "src/**/*.py" -c "python -m src.main 'test query'"
