# DVAS Development Makefile
# Usage: make [target]

.PHONY: help install install-dev test test-fast test-unit test-integration test-e2e coverage lint format type-check clean docs validate

# Default target
help:
	@echo "DVAS Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install with all dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run fast tests (excludes slow/integration)"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e     Run end-to-end tests"
	@echo "  make coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Run ruff formatter (check mode)"
	@echo "  make format-fix   Run ruff formatter (fix mode)"
	@echo "  make type-check   Run mypy type checker"
	@echo "  make validate     Run all validations (lint + type-check + test-fast)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Validate documentation"
	@echo "  make doc-anchors  Check documentation anchors"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Clean cache files and temp directories"
	@echo "  make clean-all    Clean + remove virtual environment"

# Installation
install:
	pip install -e "."

install-dev:
	pip install -e ".[all]"
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -q --tb=short \
		--ignore=tests/test_cli.py \
		--ignore=tests/test_video_loader.py \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_pipeline_retry.py \
		--ignore=tests/test_student_sft_dryrun.py \
		--ignore=tests/test_utils.py \
		--ignore=tests/test_snapshots.py \
		--ignore=tests/test_train_dryrun.py

test-unit:
	pytest tests/unit/ -v --tb=short 2>/dev/null || \
	pytest tests/ -v --tb=short -m "not integration and not e2e and not slow" \
		--ignore=tests/test_cli.py \
		--ignore=tests/test_video_loader.py \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_pipeline_retry.py \
		--ignore=tests/test_student_sft_dryrun.py \
		--ignore=tests/test_utils.py \
		--ignore=tests/test_snapshots.py \
		--ignore=tests/test_train_dryrun.py

test-integration:
	pytest tests/integration/ -v --tb=short 2>/dev/null || \
	pytest tests/ -v --tb=short -m "integration"

test-e2e:
	pytest tests/e2e/ -v --tb=short 2>/dev/null || \
	pytest tests/ -v --tb=short -m "e2e"

test-ci:
	pytest tests/ -q --tb=short --strict-markers \
		--ignore=tests/test_cli.py \
		--ignore=tests/test_video_loader.py \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_pipeline_retry.py \
		--ignore=tests/test_student_sft_dryrun.py \
		--ignore=tests/test_utils.py \
		--ignore=tests/test_snapshots.py \
		--ignore=tests/test_train_dryrun.py

coverage:
	pytest tests/ \
		--ignore=tests/test_cli.py \
		--ignore=tests/test_video_loader.py \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_pipeline.py \
		--ignore=tests/test_pipeline_retry.py \
		--ignore=tests/test_student_sft_dryrun.py \
		--ignore=tests/test_utils.py \
		--ignore=tests/test_snapshots.py \
		--ignore=tests/test_train_dryrun.py \
		--cov=src/dvas --cov-report=term-missing --cov-report=html:htmlcov

# Code Quality
lint:
	ruff check src/dvas

format:
	ruff format --check src/dvas

format-fix:
	ruff format src/dvas

type-check:
	mypy src/dvas --ignore-missing-imports

security:
	bandit -r src/dvas -f json -o bandit-report.json || true

validate: lint type-check test-fast

# Documentation
docs:
	python scripts/check_doc_anchors.py --quiet

doc-anchors:
	python scripts/check_doc_anchors.py

# Maintenance
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf tmp/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage.*" -delete 2>/dev/null || true

clean-all: clean
	rm -rf venv/
	rm -rf .venv/
