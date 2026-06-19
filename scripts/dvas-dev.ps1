<#
.SYNOPSIS
    DVAS Development Task Runner

.DESCRIPTION
    PowerShell-based task runner for DVAS development workflow.
    Import with: Import-Module .\scripts\dvas-dev.ps1
    Or run individual functions after dot-sourcing.

.EXAMPLE
    . .\scripts\dvas-dev.ps1
    Test-Fast

.EXAMPLE
    Invoke-DevTask -Task test-fast
#>

$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
}

function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor $Colors.Header
    Write-Host $text -ForegroundColor $Colors.Header
    Write-Host "========================================`n" -ForegroundColor $Colors.Header
}

function Write-Success($text) {
    Write-Host "✓ $text" -ForegroundColor $Colors.Success
}

function Write-Warning($text) {
    Write-Host "⚠ $text" -ForegroundColor $Colors.Warning
}

function Write-Error($text) {
    Write-Host "✗ $text" -ForegroundColor $Colors.Error
}

# -----------------------------------------------------------------------------
# Setup Tasks
# -----------------------------------------------------------------------------

function Install-DevEnvironment {
    <#
    .SYNOPSIS
        Install development environment with all dependencies
    #>
    Write-Header "Installing DVAS Development Environment"

    # Check Python version
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion"

    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Error "Python 3.10+ required"
            return
        }
    }

    # Install in editable mode with all dependencies
    Write-Host "Installing DVAS with all dependencies..."
    pip install -e ".[all]"

    Write-Success "Development environment installed"
}

function Install-Minimal {
    <#
    .SYNOPSIS
        Install minimal dependencies only
    #>
    Write-Header "Installing Minimal Dependencies"
    pip install -e "."
    Write-Success "Minimal dependencies installed"
}

# -----------------------------------------------------------------------------
# Test Tasks
# -----------------------------------------------------------------------------

function Test-All {
    <#
    .SYNOPSIS
        Run all tests
    #>
    Write-Header "Running ALL tests"
    python -m pytest tests/ -v --tb=short
}

function Test-Fast {
    <#
    .SYNOPSIS
        Run fast tests (excludes slow/known-broken)
    #>
    Write-Header "Running FAST tests"
    python -m pytest tests/ -q --tb:short `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py
}

function Test-Unit {
    <#
    .SYNOPSIS
        Run unit tests only
    #>
    Write-Header "Running UNIT tests"
    python -m pytest tests/ -v --tb=short `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py `
        -m "not integration and not e2e and not slow"
}

function Test-Integration {
    <#
    .SYNOPSIS
        Run integration tests
    #>
    Write-Header "Running INTEGRATION tests"
    python -m pytest tests/ -v --tb=short -m "integration"
}

function Test-E2E {
    <#
    .SYNOPSIS
        Run end-to-end tests
    #>
    Write-Header "Running E2E tests"
    python -m pytest tests/ -v --tb:short -m "e2e"
}

function Test-CI {
    <#
    .SYNOPSIS
        Run CI test suite
    #>
    Write-Header "Running CI test suite"
    python -m pytest tests/ -q --tb:short --strict-markers `
        --ignore=tests/test_cli.py `
        --ignore=tests/test_video_loader.py `
        --ignore=tests/test_integration.py `
        --ignore=tests/test_pipeline.py `
        --ignore=tests/test_pipeline_retry.py `
        --ignore=tests/test_student_sft_dryrun.py `
        --ignore=tests/test_utils.py `
        --ignore=tests/test_snapshots.py `
        --ignore=tests/test_train_dryrun.py
}

function Test-Coverage {
    <#
    .SYNOPSIS
        Run tests with coverage report
    #>
    Write-Header "Running tests with COVERAGE"
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

# -----------------------------------------------------------------------------
# Code Quality Tasks
# -----------------------------------------------------------------------------

function Invoke-Linter {
    <#
    .SYNOPSIS
        Run ruff linter
    #>
    Write-Header "Running Ruff Linter"
    ruff check src/dvas
}

function Invoke-FormatCheck {
    <#
    .SYNOPSIS
        Check code formatting with ruff
    #>
    Write-Header "Checking Code Formatting"
    ruff format --check src/dvas
}

function Invoke-FormatFix {
    <#
    .SYNOPSIS
        Fix code formatting with ruff
    #>
    Write-Header "Fixing Code Formatting"
    ruff format src/dvas
}

function Invoke-TypeCheck {
    <#
    .SYNOPSIS
        Run mypy type checker
    #>
    Write-Header "Running Type Checker"
    mypy src/dvas --ignore-missing-imports
}

function Invoke-SecurityScan {
    <#
    .SYNOPSIS
        Run bandit security scanner
    #>
    Write-Header "Running Security Scan"
    bandit -r src/dvas -f json -o bandit-report.json || Write-Warning "Security issues found, see bandit-report.json"
}

# -----------------------------------------------------------------------------
# Validation Tasks
# -----------------------------------------------------------------------------

function Invoke-Validate {
    <#
    .SYNOPSIS
        Run all validations (lint + type-check + test-fast)
    #>
    Write-Header "Running FULL VALIDATION"

    $failed = $false

    try {
        Invoke-Linter
        Write-Success "Linting passed"
    } catch {
        Write-Error "Linting failed"
        $failed = $true
    }

    try {
        Invoke-TypeCheck
        Write-Success "Type checking passed"
    } catch {
        Write-Error "Type checking failed"
        $failed = $true
    }

    try {
        Test-Fast
        Write-Success "Fast tests passed"
    } catch {
        Write-Error "Fast tests failed"
        $failed = $true
    }

    if ($failed) {
        throw "Validation failed"
    }

    Write-Success "All validations passed!"
}

# -----------------------------------------------------------------------------
# Documentation Tasks
# -----------------------------------------------------------------------------

function Invoke-DocValidation {
    <#
    .SYNOPSIS
        Validate documentation anchors
    #>
    Write-Header "Validating Documentation"
    python scripts/check_doc_anchors.py
}

function Invoke-DocAnchors {
    <#
    .SYNOPSIS
        Check documentation anchors (verbose)
    #>
    Write-Header "Checking Documentation Anchors"
    python scripts/check_doc_anchors.py --verbose
}

# -----------------------------------------------------------------------------
# Maintenance Tasks
# -----------------------------------------------------------------------------

function Invoke-Clean {
    <#
    .SYNOPSIS
        Clean cache files and temp directories
    #>
    Write-Header "Cleaning Cache Files"

    $dirs = @(".pytest_cache", ".mypy_cache", "htmlcov", ".coverage", "tmp", "build", "dist")
    foreach ($dir in $dirs) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir
            Write-Host "Removed $dir"
        }
    }

    # Clean __pycache__ directories
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
    }

    # Clean .pyc files
    Get-ChildItem -Recurse -File -Filter "*.pyc" | ForEach-Object {
        Remove-Item -Force $_.FullName -ErrorAction SilentlyContinue
    }

    Write-Success "Cleanup complete"
}

# -----------------------------------------------------------------------------
# Main Task Dispatcher
# -----------------------------------------------------------------------------

function Invoke-DevTask {
    <#
    .SYNOPSIS
        Main task dispatcher

    .PARAMETER Task
        Task name to run

    .EXAMPLE
        Invoke-DevTask -Task test-fast
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("install", "install-dev", "test", "test-fast", "test-unit",
                    "test-integration", "test-e2e", "test-ci", "coverage",
                    "lint", "format", "format-fix", "type-check", "security",
                    "validate", "docs", "doc-anchors", "clean")]
        [string]$Task
    )

    switch ($Task) {
        "install" { Install-Minimal }
        "install-dev" { Install-DevEnvironment }
        "test" { Test-All }
        "test-fast" { Test-Fast }
        "test-unit" { Test-Unit }
        "test-integration" { Test-Integration }
        "test-e2e" { Test-E2E }
        "test-ci" { Test-CI }
        "coverage" { Test-Coverage }
        "lint" { Invoke-Linter }
        "format" { Invoke-FormatCheck }
        "format-fix" { Invoke-FormatFix }
        "type-check" { Invoke-TypeCheck }
        "security" { Invoke-SecurityScan }
        "validate" { Invoke-Validate }
        "docs" { Invoke-DocValidation }
        "doc-anchors" { Invoke-DocAnchors }
        "clean" { Invoke-Clean }
    }
}

# Export functions
Export-ModuleMember -Function @(
    "Install-DevEnvironment", "Install-Minimal",
    "Test-All", "Test-Fast", "Test-Unit", "Test-Integration", "Test-E2E", "Test-CI", "Test-Coverage",
    "Invoke-Linter", "Invoke-FormatCheck", "Invoke-FormatFix", "Invoke-TypeCheck", "Invoke-SecurityScan",
    "Invoke-Validate", "Invoke-DocValidation", "Invoke-DocAnchors", "Invoke-Clean",
    "Invoke-DevTask"
)
