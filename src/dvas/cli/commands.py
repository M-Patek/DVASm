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

import sys
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console

from dvas.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
app = typer.Typer(help="DVAS Developer Tools")
console = Console()


# Import subcommands from specialized modules
from dvas.cli.dev import DevModeWatcher
from dvas.cli.docs_info import app as docs_app, info as info_cmd
from dvas.cli.migrate import app as migrate_app
from dvas.cli.quality import app as quality_app
from dvas.cli.scaffold import app as scaffold_app

# Register subcommands
app.add_typer(scaffold_app, name="scaffold")
app.add_typer(migrate_app, name="migrate")
app.add_typer(docs_app, name="docs")
app.add_typer(quality_app, name="quality")

# Note: dev mode is registered below as it uses a different pattern


# ---------------------------------------------------------------------------
# Development Mode
# ---------------------------------------------------------------------------


@app.command()
def dev(
    server: bool = typer.Option(True, "--server/--no-server", help="Run API server"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    watch_paths: Optional[List[str]] = typer.Option(
        None, "--watch", "-w", help="Additional paths to watch"
    ),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Check interval in seconds"),
) -> None:
    """Run DVAS in development mode with hot reload."""
    from pathlib import Path

    from dvas.config import settings

    paths = [settings.PROJECT_ROOT / "src" / "dvas"]
    if watch_paths:
        paths.extend(Path(p) for p in watch_paths)

    def reload_callback() -> None:
        """Callback when files change."""
        # Clear module cache for dvas modules
        modules_to_remove = [name for name in sys.modules if name.startswith("dvas.")]
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
# Direct command aliases (for backward compatibility)
# ---------------------------------------------------------------------------


# Import command functions for direct registration
from dvas.cli.docs_info import docs as docs_cmd  # noqa: E402
from dvas.cli.migrate import migrate as migrate_cmd  # noqa: E402
from dvas.cli.quality import benchmark as benchmark_cmd  # noqa: E402
from dvas.cli.quality import lint as lint_cmd  # noqa: E402
from dvas.cli.quality import test as test_cmd  # noqa: E402
from dvas.cli.quality import validate as validate_cmd  # noqa: E402
from dvas.cli.scaffold import scaffold as scaffold_cmd  # noqa: E402
from dvas.cli.scaffold import scaffold_list as scaffold_list_cmd  # noqa: E402

# Register as top-level commands for backward compatibility
app.command()(docs_cmd)
app.command(name="scaffold-list")(scaffold_list_cmd)
app.command()(lint_cmd)
app.command()(test_cmd)
app.command()(benchmark_cmd)
app.command()(validate_cmd)
app.command()(info_cmd)
app.command()(migrate_cmd)
app.command()(scaffold_cmd)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point."""
    setup_logging(level="INFO", json_format=False)
    app()


if __name__ == "__main__":
    main()


__all__ = [
    "app",
    "main",
    "dev",
    "DevModeWatcher",
]
