# Makefile for Python project automation
# Include separate make files for organization
include .makefiles/requirements.mk
include .makefiles/lint.mk
include .makefiles/test.mk

.PHONY: help clean ci dev

# Default target
help:
	@echo "📋 Available targets:"
	@echo ""
	@echo "📦 Requirements:"
	@echo "  install-all      - Install all dependencies (default)"
	@echo "  install-base     - Install base/production dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  install-profiling- Install profiling dependencies"
	@echo "  install-uv       - Install uv package manager"
	@echo ""
	@echo "🔧 Setup:"
	@echo "  dev             - Setup complete development environment"
	@echo ""
	@echo "🎨 Linting & Code Quality:"
	@echo "  format          - Format code with ruff"
	@echo "  lint            - Run ruff linter"
	@echo "  lint-fix        - Fix linting issues automatically"
	@echo "  type-check      - Run mypy type checker"
	@echo "  security        - Run bandit security scanner"
	@echo "  lint-all        - Run all linters and checks"
	@echo "  check           - Validate code (no formatting)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-coverage   - Run tests with coverage"
	@echo ""
	@echo "🚀 Workflows:"
	@echo "  ci              - Run CI pipeline (check + test)"
	@echo "  clean           - Clean cache and build artifacts"
	@echo ""


# Clean cache and build artifacts
clean:
	@echo "🧹 Cleaning cache and build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -name "bandit-report.json" -delete
	find . -name ".coverage" -delete
	@echo "✅ Cleanup complete"

# CI target (for GitHub Actions)
ci: check test
	@echo ""
	@echo "🎉 CI pipeline completed successfully!"
	@echo ""

# Development workflow target
dev: install-all pre-commit-install
	@echo ""
	@echo "🎉 Development environment setup complete!"
	@echo "Run 'make lint-all' to verify everything is working"
	@echo ""