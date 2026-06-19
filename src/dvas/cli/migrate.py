"""Database migration commands for DVAS CLI.

Provides SQLite migration management with create, apply, and status commands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from dvas.config import settings

console = Console()
app = typer.Typer(help="DVAS Migrations")


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
        cursor = conn.execute("SELECT version, name, applied_at FROM _migrations ORDER BY version")
        migrations = [
            Migration(version=row[0], name=row[1], applied_at=row[2]) for row in cursor.fetchall()
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


__all__ = ["app", "migrate", "Migration", "MigrationManager"]
