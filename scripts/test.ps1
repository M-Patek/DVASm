# DVAS Local Validation Commands
# Usage: .\scripts\test.ps1 [command]
# Or import and use individual functions

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "unit", "integration", "e2e", "fast", "coverage", "ci")]
    [string]$Command = "fast"
)

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Test-All {
    Write-Header "Running ALL tests"
    python -m pytest tests/ -v --tb=short
}

function Test-Unit {
    Write-Header "Running UNIT tests (fast, no external deps)"
    python -m pytest tests/ `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py `
        -v --tb=short
}

function Test-Integration {
    Write-Header "Running INTEGRATION tests"
    python -m pytest tests/test_integration.py tests/test_pipeline.py tests/test_pipeline_retry.py tests/test_utils.py -v --tb=short
}

function Test-E2E {
    Write-Header "Running E2E tests"
    python -m pytest tests/e2e/ -v --tb=short
}

function Test-Fast {
    Write-Header "Running FAST tests (excludes slow/known-broken)"
    python -m pytest tests/ `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py `
        -q --tb=short
}

function Test-Coverage {
    Write-Header "Running tests with COVERAGE report"
    python -m pytest tests/ `
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
}

function Test-CI {
    Write-Header "Running CI test suite"
    # Same as fast but with stricter settings
    python -m pytest tests/ `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py `
        --tb=short --strict-markers -q
}

# Main dispatch
switch ($Command) {
    "all"         { Test-All }
    "unit"        { Test-Unit }
    "integration" { Test-Integration }
    "e2e"         { Test-E2E }
    "fast"        { Test-Fast }
    "coverage"    { Test-Coverage }
    "ci"          { Test-CI }
    default       { Test-Fast }
}
