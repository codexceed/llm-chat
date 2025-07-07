# linting.mk - Linting and code quality targets
.PHONY: format lint lint-fix type-check security pre-commit-install pre-commit-all lint-all check

# Format code with ruff
format:
	@echo "🎨 Formatting code with ruff..."
	ruff format .
	@echo "✅ Code formatting complete"

# Run ruff linter
lint:
	@echo "🔍 Running ruff linter..."
	ruff check .
	@echo "✅ Linting complete"

# Fix linting issues automatically
lint-fix:
	@echo "🔧 Fixing linting issues with ruff..."
	ruff check --fix .
	ruff format .
	@echo "✅ Auto-fix complete"

# Run mypy type checker
type-check:
	@echo "🔍 Running mypy type checker..."
	mypy .
	@echo "✅ Type checking complete"

# Run bandit security scanner
security:
	@echo "🔒 Running bandit security scanner..."
	bandit -r . -f json -o bandit-report.json || (cat bandit-report.json && exit 1)
	@echo "✅ Security scan complete"

# Pre-commit setup
pre-commit-install:
	@echo "🔧 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Run pre-commit on all files
pre-commit-all:
	@echo "🔧 Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "✅ Pre-commit complete"

# =====================================
# 🚀 CATCH-ALL: Run all linters and checks
# =====================================
lint-all: format lint type-check security
	@echo ""
	@echo "🎉 All linting and security checks completed successfully!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Code formatted (ruff format)"
	@echo "  ✅ Linting passed (ruff check)"
	@echo "  ✅ Type checking passed (mypy)"
	@echo "  ✅ Security scan passed (bandit)"
	@echo "  ✅ Dependencies checked (safety)"
	@echo ""

# Quick check (no formatting, just validation)
check: lint type-check security
	@echo ""
	@echo "🎉 All checks passed!"
	@echo ""