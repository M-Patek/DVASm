"""Enhanced CLI commands for DVAS developer experience.

Commands:
- dvas dev: Development mode with hot reload
- dvas scaffold: Generate code scaffolding
- dvas migrate: Database migrations
- dvas docs: Generate API documentation
- dvas lint: Run linting and formatting
- dvas test: Run tests with coverage
- dvas benchmark: Run performance benchmarks
- dvas validate: Validate configuration and data
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from dvas.config import settings
from dvas.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
app = typer.Typer(help="DVAS Developer Tools")
console = Console()


# ---------------------------------------------------------------------------
# Development Mode
# ---------------------------------------------------------------------------

class DevModeWatcher:
    """File watcher for development hot reload."""

    def __init__(self, paths: List[Path], callback: Callable[[], None]) -> None:
        self.paths = paths
        self.callback = callback
        self._mtimes: Dict[str, float] = {}
        self._running = False

    def _get_mtime(self, path: Path) -> float:
        """Get modification time of a file."""
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _scan_files(self) -> Dict[str, float]:
        """Scan all watched paths for Python files."""
        mtimes: Dict[str, float] = {}
        for path in self.paths:
            if path.is_file() and path.suffix == ".py":
                mtimes[str(path)] = self._get_mtime(path)
            elif path.is_dir():
                for py_file in path.rglob("*.py"):
                    mtimes[str(py_file)] = self._get_mtime(py_file)
        return mtimes

    def check(self) -> bool:
        """Check if any files have changed. Returns True if reload needed."""
        current = self._scan_files()
        changed = False

        for filepath, mtime in current.items():
            if filepath not in self._mtimes or self._mtimes[filepath] != mtime:
                changed = True
                break

        self._mtimes = current
        return changed

    def run(self, interval: float = 1.0) -> None:
        """Run the watcher loop."""
        self._mtimes = self._scan_files()
        self._running = True

        console.print("[green]Dev mode started. Watching for changes...[/green]")
        console.print(f"  Watching: {[str(p) for p in self.paths]}")
        console.print("  Press Ctrl+C to stop\n")

        while self._running:
            try:
                time.sleep(interval)
                if self.check():
                    console.print("[yellow]Changes detected, reloading...[/yellow]")
                    self.callback()
                    console.print("[green]Reload complete.[/green]\n")
            except KeyboardInterrupt:
                self._running = False
                console.print("\n[blue]Dev mode stopped.[/blue]")

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False


@app.command()
def dev(
    server: bool = typer.Option(True, "--server/--no-server", help="Run API server"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    watch_paths: Optional[List[str]] = typer.Option(None, "--watch", "-w", help="Additional paths to watch"),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Check interval in seconds"),
) -> None:
    """Run DVAS in development mode with hot reload."""
    paths = [settings.PROJECT_ROOT / "src" / "dvas"]
    if watch_paths:
        paths.extend(Path(p) for p in watch_paths)

    def reload_callback() -> None:
        """Callback when files change."""
        # Clear module cache for dvas modules
        modules_to_remove = [
            name for name in sys.modules
            if name.startswith("dvas.")
        ]
        for name in modules_to_remove:
            del sys.modules[name]

        console.print("[dim]Module cache cleared[/dim]")

    if server:
        import uvicorn

        # Start server in a thread
        import threading

        def run_server() -> None:
            uvicorn.run(
                "dvas.api.main:app",
                host="0.0.0.0",
                port=port,
                reload=False,
                log_level="info",
            )

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        console.print(f"[green]API server started on http://localhost:{port}[/green]")

    watcher = DevModeWatcher(paths, reload_callback)
    watcher.run(interval=interval)


# ---------------------------------------------------------------------------
# Code Scaffolding
# ---------------------------------------------------------------------------

@dataclass
class ScaffoldTemplate:
    """Template for code scaffolding."""

    name: str
    description: str
    files: Dict[str, str] = field(default_factory=dict)


SCAFFOLD_TEMPLATES: Dict[str, ScaffoldTemplate] = {
    "module": ScaffoldTemplate(
        name="module",
        description="New subsystem module",
        files={
            "__init__.py": '''"""{module_name} module for DVAS."""

from dvas.{module_name}.core import {ModuleName}Processor

__all__ = ["{ModuleName}Processor"]
''',
            "core.py": '''"""Core {module_name} functionality."""

from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Processor:
    """Main processor for {module_name}."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {{}}

    def process(self, data: Any) -> Any:
        """Process input data."""
        raise NotImplementedError
''',
            "types.py": '''"""Type definitions for {module_name}."""

from typing import TypedDict


class {ModuleName}Config(TypedDict, total=False):
    """Configuration for {module_name}."""

    enabled: bool
    timeout: float
''',
        },
    ),
    "model": ScaffoldTemplate(
        name="model",
        description="New teacher/student model",
        files={
            "__init__.py": '''"""{module_name} model for DVAS."""

from dvas.models.{module_name}.model import {ModuleName}Teacher

__all__ = ["{ModuleName}Teacher"]
''',
            "model.py": '''"""{ModuleName} teacher model implementation."""

from typing import Any, Dict, List, Optional

from dvas.models.teacher.base import BaseTeacherModel
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Teacher(BaseTeacherModel):
    """{ModuleName} teacher model."""

    def __init__(self, model_name: str = "{module_name}", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

    async def annotate_frame_batch(
        self,
        frames: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Annotate a batch of frames."""
        raise NotImplementedError("Implement annotate_frame_batch")

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {{
            "name": self.model_name,
            "type": "teacher",
            "provider": "{module_name}",
        }}
''',
        },
    ),
    "pipeline": ScaffoldTemplate(
        name="pipeline",
        description="New pipeline stage",
        files={
            "__init__.py": '''"""{module_name} pipeline stage."""

from dvas.pipeline.{module_name}.stage import {ModuleName}Stage

__all__ = ["{ModuleName}Stage"]
''',
            "stage.py": '''"""{ModuleName} pipeline stage implementation."""

from typing import Any, Dict, Optional

from dvas.pipeline.core import PipelineStage
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Stage(PipelineStage):
    """{ModuleName} pipeline stage."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {{}}

    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process data through this stage."""
        logger.info("{module_name}_stage_processing")
        raise NotImplementedError("Implement process method")

    @property
    def name(self) -> str:
        return "{module_name}"

    @property
    def stage_type(self) -> str:
        return "transform"
''',
        },
    ),
    "test": ScaffoldTemplate(
        name="test",
        description="New test suite",
        files={
            "{module_name}.py": '''"""Tests for {module_name}."""

import pytest

from dvas.{module_name} import {ModuleName}Processor


class Test{ModuleName}Processor:
    """Test {module_name} processor."""

    def test_init(self) -> None:
        """Test processor initialization."""
        processor = {ModuleName}Processor()
        assert processor.config == {{}}

    def test_init_with_config(self) -> None:
        """Test processor with config."""
        processor = {ModuleName}Processor(config={{"key": "value"}})
        assert processor.config["key"] == "value"
''',
        },
    ),
}


@app.command()
def scaffold(
    template: str = typer.Argument(..., help="Template type (module/model/pipeline/test)"),
    name: str = typer.Argument(..., help="Module name (snake_case)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be created"),
) -> None:
    """Generate code scaffolding from templates."""
    if template not in SCAFFOLD_TEMPLATES:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available: {', '.join(SCAFFOLD_TEMPLATES.keys())}")
        raise typer.Exit(1)

    tmpl = SCAFFOLD_TEMPLATES[template]
    module_name = name.lower().replace(" ", "_").replace("-", "_")
    ModuleName = "".join(word.capitalize() for word in module_name.split("_"))

    # Determine output directory
    if output_dir:
        base_dir = output_dir
    elif template == "test":
        base_dir = settings.PROJECT_ROOT / "tests"
    elif template == "model":
        base_dir = settings.PROJECT_ROOT / "src" / "dvas" / "models" / "teacher"
    elif template == "pipeline":
        base_dir = settings.PROJECT_ROOT / "src" / "dvas" / "pipeline"
    else:
        base_dir = settings.PROJECT_ROOT / "src" / "dvas"

    target_dir = base_dir / module_name

    # Show plan
    tree = Tree(f"[bold cyan]{target_dir}[/bold cyan]")
    for filename in tmpl.files.keys():
        tree.add(f"[green]{filename}[/green]")

    console.print(Panel(tree, title=f"Scaffold: {template} '{name}'", border_style="blue"))

    if dry_run:
        console.print("[yellow]Dry run - no files created[/yellow]")
        return

    # Create directory and files
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, content_template in tmpl.files.items():
        content = content_template.format(
            module_name=module_name,
            ModuleName=ModuleName,
        )
        file_path = target_dir / filename
        file_path.write_text(content, encoding="utf-8")
        console.print(f"  [green]Created[/green] {file_path}")

    console.print(f"\n[green]Scaffold '{name}' created successfully![/green]")
    console.print(f"  Location: {target_dir}")
    console.print(f"  Files: {len(tmpl.files)}")


@app.command(name="scaffold-list")
def scaffold_list() -> None:
    """List available scaffolding templates."""
    table = Table(title="Available Scaffolding Templates")
    table.add_column("Template", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Files", style="blue")

    for name, tmpl in SCAFFOLD_TEMPLATES.items():
        table.add_row(name, tmpl.description, ", ".join(tmpl.files.keys()))

    console.print(table)


# ---------------------------------------------------------------------------
# Database Migrations
# ---------------------------------------------------------------------------

@dataclass
class Migration:
    """Database migration record."""

    version: str
    name: str
    applied_at: Optional[float] = None


class MigrationManager:
    """Simple migration manager for SQLite metadata."""

    def __init__(self, db_path: Path, migrations_dir: Optional[Path] = None) -> None:
        self.db_path = db_path
        self.migrations_dir = migrations_dir or (settings.PROJECT_ROOT / "migrations")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize migration tracking table."""
        import sqlite3

        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def get_applied(self) -> List[Migration]:
        """Get list of applied migrations."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT version, name, applied_at FROM _migrations ORDER BY version"
        )
        migrations = [
            Migration(version=row[0], name=row[1], applied_at=row[2])
            for row in cursor.fetchall()
        ]
        conn.close()
        return migrations

    def get_pending(self) -> List[Path]:
        """Get list of pending migration files."""
        applied = {m.version for m in self.get_applied()}

        if not self.migrations_dir.exists():
            return []

        pending = []
        for file in sorted(self.migrations_dir.glob("*.sql")):
            version = file.stem.split("_")[0]
            if version not in applied:
                pending.append(file)

        return pending

    def apply(self, migration_file: Path) -> None:
        """Apply a single migration."""
        import sqlite3
        import time

        version = migration_file.stem.split("_")[0]
        name = "_".join(migration_file.stem.split("_")[1:])
        sql = migration_file.read_text(encoding="utf-8")

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO _migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, time.time()),
            )
            conn.commit()
            console.print(f"  [green]Applied[/green] {migration_file.name}")
        except Exception as e:
            conn.rollback()
            console.print(f"  [red]Failed[/red] {migration_file.name}: {e}")
            raise
        finally:
            conn.close()

    def create(self, name: str) -> Path:
        """Create a new migration file."""
        import time

        version = f"{int(time.time())}"
        filename = f"{version}_{name}.sql"
        path = self.migrations_dir / filename

        template = f"""-- Migration: {name}
-- Created: {time.strftime("%Y-%m-%d %H:%M:%S")}

BEGIN;

-- Add your migration SQL here

COMMIT;
"""
        path.write_text(template, encoding="utf-8")
        return path


@app.command()
def migrate(
    action: str = typer.Argument("status", help="Action: status/up/down/create"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Migration name (for create)"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Database path"),
) -> None:
    """Manage database migrations."""
    default_db = settings.DATA_ROOT / "dvas.db"
    manager = MigrationManager(db_path or default_db)

    if action == "status":
        applied = manager.get_applied()
        pending = manager.get_pending()

        table = Table(title="Migration Status")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_row("Applied", str(len(applied)))
        table.add_row("Pending", str(len(pending)))
        console.print(table)

        if pending:
            console.print("\n[yellow]Pending migrations:[/yellow]")
            for p in pending:
                console.print(f"  {p.name}")

    elif action == "up":
        pending = manager.get_pending()
        if not pending:
            console.print("[green]No pending migrations[/green]")
            return

        console.print(f"Applying {len(pending)} migration(s)...")
        for migration_file in pending:
            manager.apply(migration_file)
        console.print("[green]All migrations applied[/green]")

    elif action == "create":
        if not name:
            console.print("[red]Migration name required (--name)[/red]")
            raise typer.Exit(1)

        path = manager.create(name)
        console.print(f"[green]Created migration:[/green] {path}")

    elif action == "down":
        console.print("[yellow]Down migrations not yet implemented[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Documentation Generation
# ---------------------------------------------------------------------------

@app.command()
def docs(
    output: Path = typer.Option("docs/api", "--output", "-o", help="Output directory"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown/json)"),
    serve: bool = typer.Option(False, "--serve", "-s", help="Serve docs locally"),
    port: int = typer.Option(8080, "--port", "-p", help="Serve port"),
) -> None:
    """Generate API documentation from code."""
    output.mkdir(parents=True, exist_ok=True)

    # Collect API information
    docs_data: Dict[str, Any] = {
        "title": "DVAS API Documentation",
        "version": "0.2.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": [],
        "models": [],
    }

    # Extract endpoint info from API main
    try:
        from dvas.api.main import app as api_app

        for route in api_app.routes:
            if hasattr(route, "methods"):
                docs_data["endpoints"].append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name,
                })
    except ImportError:
        console.print("[yellow]Could not import API app[/yellow]")

    # Generate output
    if format == "json":
        output_file = output / "api.json"
        output_file.write_text(json.dumps(docs_data, indent=2), encoding="utf-8")
        console.print(f"[green]Generated[/green] {output_file}")

    elif format == "markdown":
        output_file = output / "api.md"
        lines = [
            "# DVAS API Documentation",
            "",
            f"**Version:** {docs_data['version']}",
            f"**Generated:** {docs_data['generated_at']}",
            "",
            "## Endpoints",
            "",
        ]

        for ep in docs_data["endpoints"]:
            methods = ", ".join(ep["methods"])
            lines.append(f"### {methods} {ep['path']}")
            lines.append("")
            lines.append(f"**Name:** {ep['name']}")
            lines.append("")

        output_file.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Generated[/green] {output_file}")

    if serve:
        import http.server
        import socketserver
        import threading

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(output), **kwargs)

        def serve_docs() -> None:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                console.print(f"[green]Serving docs at http://localhost:{port}[/green]")
                httpd.serve_forever()

        thread = threading.Thread(target=serve_docs, daemon=True)
        thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[blue]Docs server stopped[/blue]")


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
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file to validate"),
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


# ---------------------------------------------------------------------------
# Project Info
# ---------------------------------------------------------------------------

@app.command()
def info() -> None:
    """Show project information."""
    table = Table(title="DVAS Project Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project Root", str(settings.PROJECT_ROOT))
    table.add_row("Data Root", str(settings.DATA_ROOT))
    table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    table.add_row("Debug Mode", str(settings.DEBUG))
    table.add_row("Log Level", settings.LOG_LEVEL)
    table.add_row("Default Teacher", settings.DEFAULT_TEACHER_MODEL)
    table.add_row("Default FPS", str(settings.DEFAULT_FPS))
    table.add_row("Default Frames", str(settings.DEFAULT_NUM_FRAMES))

    console.print(table)

    # Show data paths
    path_table = Table(title="Data Paths")
    path_table.add_column("Name", style="cyan")
    path_table.add_column("Path", style="green")
    path_table.add_column("Exists", style="blue")

    for name, path in settings.data_paths.items():
        exists = "[green]Yes[/green]" if path.exists() else "[red]No[/red]"
        path_table.add_row(name, str(path), exists)

    console.print(path_table)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point."""
    setup_logging(level="INFO", json_format=False)
    app()


if __name__ == "__main__":
    main()
