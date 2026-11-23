.PHONY: install test run lint sast typecheck verify clean

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

# Lint
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# Type check
typecheck:
	mypy src/

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

# Build docker image
docker-build:
	docker build -t multi-agent-system .

# Run in docker
docker-run:
	docker run --rm -it multi-agent-system "Analyze AI agents in 2026"