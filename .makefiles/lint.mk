# linting.mk - Linting and code quality targets
.PHONY: format lint lint-fix type-check pylint security pre-commit-install pre-commit-all lint-all lint-yaml lint-toml-sort

format:
	@echo "🎨 Formatting code with ruff..."
	ruff format
	@echo "✅ Code formatting complete"

lint:
	@echo "🔍 Running ruff linter..."
	ruff check
	@echo "✅ Linting complete"

lint-fix:
	@echo "🔧 Fixing linting issues with ruff..."
	ruff check --fix
	ruff format
	@echo "✅ Auto-fix complete"

type-check:
	@echo "🔍 Running pyright type checker..."
	pyright
	@echo "✅ Type checking complete"

pylint:
	@echo "🔍 Running pylint..."
	pylint --rcfile=.pylintrc chatbot/ tests/ scripts/
	@echo "✅ Pylint complete"

security:
	@echo "🔒 Running bandit security scanner..."
	@bandit -c pyproject.toml -r . -f json -o bandit-report.json || (cat bandit-report.json && rm -f bandit-report.json && exit 1)
	@rm -f bandit-report.json
	@echo "✅ Security scan complete"

lint-toml-sort:
	@echo "📋 Sorting TOML files..."
	toml-sort pyproject.toml -i
	@echo "✅ TOML sorting complete"

lint-yaml:
	@echo "📄 Linting YAML files..."
	yamllint . --strict
	@echo "✅ YAML linting complete"

pre-commit-install:
	@echo "🔧 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

pre-commit-all:
	@echo "🔧 Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "✅ Pre-commit complete"

# =====================================
# 🚀 CATCH-ALL: Run all linters and checks
# =====================================
lint-all: format lint-fix lint type-check pylint lint-yaml lint-toml-sort security
	@echo ""
	@echo "🎉 All linting checks completed successfully!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Code formatted (ruff format)"
	@echo "  ✅ Linting passed (ruff check)"
	@echo "  ✅ Type checking passed (pyright)"
	@echo "  ✅ Additional linting passed (pylint)"
	@echo ""

lint-fast: format lint-fix lint lint-yaml lint-toml-sort
	@echo ""
	@echo "🎉 Fast linting checks completed successfully!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Code formatted (ruff format)"
	@echo "  ✅ Linting passed (ruff check)"
	@echo "  ✅ YAML linting passed"
	@echo "  ✅ TOML sorting passed"
	@echo ""
