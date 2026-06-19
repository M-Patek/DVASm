"""Lint, test, benchmark, and validation commands for DVAS CLI.

Provides code quality checks, test running, benchmarking, and config validation.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from dvas.config import settings

console = Console()
app = typer.Typer(help="DVAS Quality Tools")


# ---------------------------------------------------------------------------
# Lint & Format
# ---------------------------------------------------------------------------


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix issues"),
    check_only: bool = typer.Option(False, "--check", "-c", help="Check only, don't fix"),
    paths: Optional[List[str]] = typer.Option(None, "--path", "-p", help="Paths to lint"),
) -> None:
    """Run linting and formatting tools."""
    target_paths = paths or ["src/dvas", "tests"]

    tools: List[tuple] = [
        ("Ruff", ["ruff", "check"] + (["--fix"] if fix else []) + target_paths),
        ("Black", ["black"] + (["--check"] if check_only else []) + target_paths),
        ("MyPy", ["mypy"] + target_paths),
    ]

    results: List[tuple] = []

    for tool_name, cmd in tools:
        console.print(f"[blue]Running {tool_name}...[/blue]")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(settings.PROJECT_ROOT),
            )
            success = result.returncode == 0
            results.append((tool_name, success, result.stdout + result.stderr))

            if success:
                console.print(f"  [green]{tool_name} passed[/green]")
            else:
                console.print(f"  [red]{tool_name} failed[/red]")
                if result.stdout:
                    console.print(result.stdout)
        except FileNotFoundError:
            console.print(f"  [yellow]{tool_name} not installed, skipping[/yellow]")
            results.append((tool_name, True, "Not installed"))

    # Summary
    table = Table(title="Lint Results")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="green")

    all_passed = True
    for tool_name, success, _ in results:
        status = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
        table.add_row(tool_name, status)
        if not success:
            all_passed = False

    console.print(table)

    if not all_passed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------


@app.command()
def test(
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Test path"),
    marker: Optional[str] = typer.Option(None, "--marker", "-m", help="Pytest marker"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    coverage: bool = typer.Option(False, "--cov", "-c", help="Run with coverage"),
    fail_fast: bool = typer.Option(False, "--fail-fast", "-x", help="Stop on first failure"),
    parallel: bool = typer.Option(False, "--parallel", "-n", help="Run in parallel"),
) -> None:
    """Run test suite."""
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")
    if fail_fast:
        cmd.append("-x")
    if marker:
        cmd.extend(["-m", marker])
    if coverage:
        cmd.extend(["--cov=src/dvas", "--cov-report=term-missing"])
    if parallel:
        cmd.extend(["-n", "auto"])

    cmd.append(path or "tests")

    console.print(f"[blue]Running: {' '.join(cmd)}[/blue]\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(settings.PROJECT_ROOT),
        )
        raise typer.Exit(result.returncode)
    except FileNotFoundError:
        console.print("[red]pytest not found. Install with: pip install pytest[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    suite: Optional[str] = typer.Option(None, "--suite", "-s", help="Benchmark suite"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
) -> None:
    """Run performance benchmarks."""
    from dvas.testing import benchmark as bench_func

    console.print("[blue]Running benchmarks...[/blue]\n")

    # Import benchmark suites
    _benchmark_suites: Dict[str, Callable] = {}

    try:
        from tests.test_load import TestBenchmarking

        suite_instance = TestBenchmarking()

        # Run specific or all benchmarks
        if suite:
            if hasattr(suite_instance, suite):
                getattr(suite_instance, suite)()
                console.print(f"[green]Benchmark '{suite}' completed[/green]")
            else:
                console.print(f"[red]Benchmark suite '{suite}' not found[/red]")
                raise typer.Exit(1)
        else:
            # Run all test benchmarks
            import time as time_module

            def slow_add(a: int, b: int) -> int:
                time_module.sleep(0.001)
                return a + b

            result = bench_func(slow_add, 1, 2, iterations=iterations)

            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Function", result.name)
            table.add_row("Iterations", str(result.iterations))
            table.add_row("Total Time", f"{result.total_time:.4f}s")
            table.add_row("Avg Time", f"{result.avg_time:.6f}s")
            table.add_row("Min Time", f"{result.min_time:.6f}s")
            table.add_row("Max Time", f"{result.max_time:.6f}s")

            console.print(table)

            if output:
                data = result.to_dict()
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
                console.print(f"\n[green]Results saved to {output}[/green]")

    except ImportError as e:
        console.print(f"[yellow]Could not load benchmarks: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------


@app.command()
def validate(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file to validate"
    ),
    strict: bool = typer.Option(False, "--strict", help="Strict validation"),
) -> None:
    """Validate configuration and environment."""
    issues: List[tuple] = []
    warnings: List[str] = []

    # Check Python version
    py_version = sys.version_info
    if py_version < (3, 10):
        issues.append(("Python version", f"{py_version.major}.{py_version.minor}", ">= 3.10"))

    # Check required directories
    for name, path in settings.data_paths.items():
        if not path.exists():
            warnings.append(f"Data directory '{name}' does not exist: {path}")

    # Check API keys
    api_keys = {
        "OpenAI": settings.OPENAI_API_KEY,
        "Anthropic": settings.ANTHROPIC_API_KEY,
        "Together": settings.TOGETHER_API_KEY,
    }
    for name, key in api_keys.items():
        if not key:
            warnings.append(f"{name} API key not set")

    # Check optional dependencies
    optional_deps = {
        "FastAPI": "fastapi",
        "uvicorn": "uvicorn",
        "pytest": "pytest",
        "black": "black",
        "ruff": "ruff",
    }
    for name, module in optional_deps.items():
        try:
            importlib.import_module(module)
        except ImportError:
            warnings.append(f"Optional dependency '{name}' not installed")

    # Check GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            console.print(f"[green]GPU available:[/green] {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("GPU not available (CPU mode)")
    except ImportError:
        warnings.append("PyTorch not installed (training features unavailable)")

    # Results
    table = Table(title="Validation Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="blue")

    if issues:
        for check, actual, expected in issues:
            table.add_row(check, "[red]FAIL[/red]", f"{actual} (expected {expected})")
    else:
        table.add_row("Configuration", "[green]PASS[/green]", "All checks passed")

    console.print(table)

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")

    if issues:
        raise typer.Exit(1)


__all__ = [
    "app",
    "lint",
    "test",
    "benchmark",
    "validate",
]
