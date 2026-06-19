"""Documentation generation and project info commands for DVAS CLI.

Provides API docs generation and project information display.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import typer
from rich.console import Console
from rich.table import Table

from dvas.config import settings

console = Console()
app = typer.Typer(help="DVAS Documentation")


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
                docs_data["endpoints"].append(
                    {
                        "path": route.path,
                        "methods": list(route.methods),
                        "name": route.name,
                    }
                )
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
    table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
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


__all__ = ["app", "docs", "info"]
