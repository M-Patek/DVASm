# DVAS Development Justfile
# Usage: just [recipe]
# Install just: https://github.com/casey/just

# Default recipe
_default:
    @just --list

# Setup: Install production dependencies
install:
    pip install -e "."

# Setup: Install with all dev dependencies
install-dev:
    pip install -e ".[all]"
    pip install -e ".[dev]"

# Run all tests
test:
    pytest tests/ -v --tb=short

# Run fast tests (excludes slow/integration)
test-fast:
    pytest tests/ -q --tb=short `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py

# Run unit tests only (no integration/e2e)
test-unit:
    pytest tests/ -v --tb=short -m "not integration and not e2e and not slow" `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py

# Run integration tests
test-integration:
    pytest tests/ -v --tb=short -m "integration"

# Run end-to-end tests
test-e2e:
    pytest tests/ -v --tb=short -m "e2e"

# Run CI test suite
test-ci:
    pytest tests/ -q --tb=short --strict-markers `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py

# Run tests with coverage report
coverage:
    pytest tests/ `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py `
        --cov=src/dvas --cov-report=term-missing --cov-report=html:htmlcov

# Run ruff linter
lint:
    ruff check src/dvas

# Run ruff formatter (check mode)
format:
    ruff format --check src/dvas

# Run ruff formatter (fix mode)
format-fix:
    ruff format src/dvas

# Run mypy type checker
type-check:
    mypy src/dvas --ignore-missing-imports

# Run security scan
security:
    bandit -r src/dvas -f json -o bandit-report.json || true

# Run all validations (lint + type-check + test-fast)
validate: lint type-check test-fast

# Validate documentation
docs:
    python scripts/check_doc_anchors.py --quiet

# Check documentation anchors
doc-anchors:
    python scripts/check_doc_anchors.py

# Clean cache files and temp directories
clean:
    rm -rf .pytest_cache
    rm -rf .mypy_cache
    rm -rf htmlcov
    rm -rf .coverage
    rm -rf tmp/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>$null || true
    find . -type f -name "*.pyc" -delete 2>$null || true
    find . -type f -name ".coverage.*" -delete 2>$null || true
