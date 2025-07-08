# testing.mk - Testing and coverage targets
.PHONY: test test-unit test-coverage

# Run all tests
test:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests complete"

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/unit/ -v
	@echo "✅ Unit tests complete"

# Run tests with coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "✅ Tests with coverage complete"