# FX Trading System Makefile
# ==========================

.PHONY: install install-dev test lint format typecheck clean sample backtest help

# Default target
help:
	@echo "FX Trading System - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (black)"
	@echo "  make typecheck     Type check (mypy)"
	@echo ""
	@echo "Trading:"
	@echo "  make sample        Generate sample data"
	@echo "  make backtest      Run example backtest"
	@echo "  make walkforward   Run walk-forward analysis"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove generated files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=fx_trading --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	pytest tests/ -v -x --tb=short

# Code Quality
lint:
	ruff check src tests

lint-fix:
	ruff check src tests --fix

format:
	black src tests

format-check:
	black src tests --check

typecheck:
	mypy src

# All quality checks
check: lint format-check typecheck test

# Data Generation
sample:
	@echo "Generating sample data..."
	fx-trading generate-sample --symbol EURUSD --bars 5000 --output data/sample
	@echo "Sample data generated in data/sample/"

# Backtesting
backtest: sample
	@echo "Running example backtest..."
	fx-trading backtest configs/backtest_example.yaml
	@echo "Backtest complete. Check runs/ for results."

walkforward: sample
	@echo "Running walk-forward analysis..."
	fx-trading walkforward configs/walkforward_example.yaml
	@echo "Walk-forward complete. Check runs/ for results."

# Paper Trading
paper: sample
	@echo "Starting paper trading..."
	fx-trading paper-trade configs/paper_example.yaml

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-runs:
	rm -rf runs/*

clean-data:
	rm -rf data/sample/*
	rm -rf data/processed/*

clean-all: clean clean-runs clean-data

# Docker (optional)
docker-build:
	docker build -t fx-trading .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/runs:/app/runs fx-trading

# Development workflow
dev-setup: install-dev sample
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation."

# Quick demo
demo: sample
	@echo "Running quick demo backtest..."
	fx-trading generate-sample --symbol EURUSD --bars 1000 --output data/demo
	fx-trading backtest configs/backtest_example.yaml --verbose
	@echo ""
	@echo "Demo complete! Check runs/ for results."
