# Requirements Management Makefile
# This file provides targets for installing various dependency groups

# Default target - install all dependencies
.DEFAULT_GOAL := install-all

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Package manager detection
UV_AVAILABLE := $(shell command -v uv 2> /dev/null)

ifdef UV_AVAILABLE
    PIP_CMD = uv pip install
else
    PIP_CMD = pip install
endif

.PHONY: help install-all install-base install-dev install-profiling install-uv check-uv

help: ## Show this help message
	@echo "Requirements Installation Targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Package manager: $(if $(UV_AVAILABLE),$(GREEN)uv$(NC) (fast),$(YELLOW)pip$(NC) (standard))"

install-all: check-uv ## Install all dependencies (default target)
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(PIP_CMD) -e ".[all]"
	@echo "$(GREEN)✓ All dependencies installed successfully$(NC)"

install-base: check-uv ## Install base/production dependencies only
	@echo "$(GREEN)Installing base dependencies...$(NC)"
	$(PIP_CMD) -e .
	@echo "$(GREEN)✓ Base dependencies installed successfully$(NC)"

install-dev: check-uv ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP_CMD) -e ".[dev]"
	@echo "$(GREEN)✓ Development dependencies installed successfully$(NC)"

install-profiling: check-uv ## Install profiling dependencies
	@echo "$(GREEN)Installing profiling dependencies...$(NC)"
	$(PIP_CMD) -e ".[profiling]"
	@echo "$(GREEN)✓ Profiling dependencies installed successfully$(NC)"

install-uv: ## Install uv package manager
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(GREEN)uv is already installed$(NC)"; \
	else \
		echo "$(YELLOW)Installing uv package manager...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "$(GREEN)✓ uv installed successfully$(NC)"; \
		echo "$(YELLOW)Note: You may need to restart your shell or run 'source ~/.bashrc'$(NC)"; \
	fi

check-uv: ## Check if uv is available and suggest installing it
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "$(YELLOW)Note: uv is not installed. Using pip instead.$(NC)"; \
		echo "$(YELLOW)For faster installs, run 'make install-uv' first.$(NC)"; \
	fi

# Alias targets for convenience
requirements: install-all ## Alias for install-all
deps: install-all ## Alias for install-all
setup: install-all ## Alias for install-all