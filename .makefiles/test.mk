# test.mk - Testing and coverage targets
.PHONY: test test-unit test-integration test-coverage test-fast test-hypothesis test-watch test-verbose test-clean

# Run all tests with coverage
test:
	@echo "🧪 Running all tests with coverage..."
	python -m pytest tests/ --cov=chatbot --cov-report=term-missing --cov-report=html:htmlcov --cov-fail-under=80
	@echo "✅ Tests complete"

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v
	@echo "✅ Unit tests complete"

# Run integration tests only
test-integration:
	@echo "🧪 Running integration tests..."
	python -m pytest tests/integration/ -v
	@echo "✅ Integration tests complete"

# Run tests with detailed coverage report
test-coverage:
	@echo "🧪 Running tests with detailed coverage..."
	python -m pytest tests/ --cov=chatbot --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml --cov-fail-under=70 -v
	@echo "📊 Coverage report generated in htmlcov/index.html"
	@echo "✅ Tests with coverage complete"

# Run tests quickly (parallel execution, no coverage)
test-fast:
	@echo "⚡ Running tests quickly..."
	python -m pytest tests/ -x --tb=short -q
	@echo "✅ Fast tests complete"

# Run only property-based tests (hypothesis)
test-hypothesis:
	@echo "🎲 Running property-based tests..."
	python -m pytest tests/ -m hypothesis -v
	@echo "✅ Property-based tests complete"

# Watch tests (requires pytest-watch)
test-watch:
	@echo "👀 Watching tests..."
	@if command -v ptw >/dev/null 2>&1; then \
		ptw tests/ --runner "python -m pytest --tb=short"; \
	else \
		echo "❌ pytest-watch not installed. Install with: pip install pytest-watch"; \
		exit 1; \
	fi

# Run tests with verbose output and no capture
test-verbose:
	@echo "🔍 Running tests with verbose output..."
	python -m pytest tests/ -v -s --tb=long
	@echo "✅ Verbose tests complete"

# Clean test artifacts
test-clean:
	@echo "🧹 Cleaning test artifacts..."
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf .pytest_cache/
	@find tests/ -name "*.pyc" -delete
	@find tests/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Test artifacts cleaned"