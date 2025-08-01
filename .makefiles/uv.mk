# uv.mk - UV package manager utilities and installation helpers

.PHONY: check-uv install-uv

# Check if uv is installed, install if missing
check-uv:
	@echo "🔍 Checking for uv installation..."
	@command -v uv >/dev/null 2>&1 || { \
		echo "⚠️  uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed successfully"; \
	}
	@echo "✅ uv is available"

# Install uv explicitly
install-uv:
	@echo "📦 Installing uv package manager..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installation complete"