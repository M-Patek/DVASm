"""Schema migration framework for DVAS persistence layer.

Provides versioned schema migrations for SQLite and PostgreSQL backends.
"""

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Migration:
    """A single schema migration."""

    version: str
    description: str
    up_sql: str
    down_sql: Optional[str] = None
    check_func: Optional[Callable[[Any], bool]] = None


class MigrationBackend(ABC):
    """Abstract base for migration-capable backends."""

    @abstractmethod
    def get_schema_version(self) -> str:
        """Get current schema version."""
        pass

    @abstractmethod
    def set_schema_version(self, version: str) -> None:
        """Set schema version."""
        pass

    @abstractmethod
    def execute_migration(self, migration: Migration) -> None:
        """Execute a migration."""
        pass

    @abstractmethod
    def list_applied_migrations(self) -> List[str]:
        """List already applied migration versions."""
        pass


class SQLiteMigrationBackend(MigrationBackend):
    """SQLite migration backend."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_migrations_table()

    def _ensure_migrations_table(self) -> None:
        """Create migrations table if not exists."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_schema_version(self) -> str:
        """Get current schema version."""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row[0] if row else "0.0.0"

    def set_schema_version(self, version: str) -> None:
        """Set schema version."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    def execute_migration(self, migration: Migration) -> None:
        """Execute a migration."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(migration.up_sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                (migration.version, migration.description),
            )
            conn.commit()
            logger.info("migration_applied", version=migration.version)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_applied_migrations(self) -> List[str]:
        """List applied migrations."""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        conn.close()
        return [row[0] for row in rows]


class MigrationManager:
    """Manages schema migrations.

    Usage::

        manager = MigrationManager(backend)
        manager.register(Migration(
            version="1.1.0",
            description="Add frame hash index",
            up_sql="CREATE TABLE frame_hashes (...)",
        ))
        manager.migrate("1.1.0")
    """

    def __init__(self, backend: MigrationBackend):
        self.backend = backend
        self.migrations: Dict[str, Migration] = {}

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self.migrations[migration.version] = migration
        logger.debug("migration_registered", version=migration.version)

    def register_many(self, migrations: List[Migration]) -> None:
        """Register multiple migrations."""
        for migration in migrations:
            self.register(migration)

    def get_current_version(self) -> str:
        """Get current schema version."""
        return self.backend.get_schema_version()

    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = set(self.backend.list_applied_migrations())
        pending = []

        for version in sorted(self.migrations.keys()):
            if version not in applied:
                pending.append(self.migrations[version])

        return pending

    def migrate(self, target_version: Optional[str] = None) -> bool:
        """Migrate to target version.

        Args:
            target_version: Target version, or None for latest

        Returns:
            True if migrations were applied
        """
        pending = self.get_pending_migrations()

        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            logger.info("no_pending_migrations")
            return False

        for migration in pending:
            logger.info(
                "applying_migration",
                version=migration.version,
                description=migration.description,
            )

            if migration.check_func and not migration.check_func(self.backend):
                logger.warning("migration_check_failed", version=migration.version)
                continue

            self.backend.execute_migration(migration)

        return True

    def rollback(self, version: str) -> bool:
        """Rollback to a specific version.

        Args:
            version: Version to rollback to

        Returns:
            True if rollback was performed
        """
        current = self.backend.get_schema_version()
        if version >= current:
            logger.info("no_rollback_needed", current=current, target=version)
            return False

        applied = self.backend.list_applied_migrations()
        to_rollback = [v for v in applied if v > version]

        for v in reversed(to_rollback):
            migration = self.migrations.get(v)
            if migration and migration.down_sql:
                logger.info("rolling_back", version=v)
                # Execute rollback SQL
                conn = sqlite3.connect(str(self.backend.db_path))
                conn.executescript(migration.down_sql)
                conn.execute("DELETE FROM schema_migrations WHERE version = ?", (v,))
                conn.commit()
                conn.close()

        return True

    def status(self) -> Dict[str, Any]:
        """Get migration status."""
        current = self.backend.get_schema_version()
        applied = self.backend.list_applied_migrations()
        pending = self.get_pending_migrations()

        return {
            "current_version": current,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "latest_available": max(self.migrations.keys()) if self.migrations else None,
        }


# Predefined migrations for DVAS
DVAS_MIGRATIONS = [
    Migration(
        version="1.0.0",
        description="Initial schema",
        up_sql="""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                video_path TEXT NOT NULL,
                source TEXT NOT NULL,
                model_version TEXT,
                quality_score REAL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                num_segments INTEGER DEFAULT 0,
                total_duration REAL DEFAULT 0.0,
                tags TEXT DEFAULT '[]',
                parent_id TEXT,
                json_data TEXT NOT NULL
            );
            CREATE INDEX idx_video_id ON annotations(video_id);
            CREATE INDEX idx_source ON annotations(source);
        """,
    ),
    Migration(
        version="1.1.0",
        description="Add content hash and storage path",
        up_sql="""
            ALTER TABLE annotations ADD COLUMN content_hash TEXT;
            ALTER TABLE annotations ADD COLUMN storage_path TEXT;
            CREATE INDEX idx_content_hash ON annotations(content_hash);
        """,
        down_sql="""
            DROP INDEX idx_content_hash;
            ALTER TABLE annotations DROP COLUMN content_hash;
            ALTER TABLE annotations DROP COLUMN storage_path;
        """,
    ),
    Migration(
        version="1.2.0",
        description="Add prompt and dataset version columns",
        up_sql="""
            ALTER TABLE annotations ADD COLUMN prompt_version TEXT;
            ALTER TABLE annotations ADD COLUMN dataset_version TEXT;
            CREATE INDEX idx_prompt_version ON annotations(prompt_version);
            CREATE INDEX idx_dataset_version ON annotations(dataset_version);
        """,
    ),
    Migration(
        version="1.3.0",
        description="Add video hash index table",
        up_sql="""
            CREATE TABLE video_hashes (
                video_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                frame_count INTEGER,
                duration REAL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        down_sql="DROP TABLE video_hashes;",
    ),
    Migration(
        version="1.4.0",
        description="Add frame hash index table",
        up_sql="""
            CREATE TABLE frame_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, frame_index)
            );
            CREATE INDEX idx_frame_video ON frame_hashes(video_id);
        """,
        down_sql="""
            DROP INDEX idx_frame_video;
            DROP TABLE frame_hashes;
        """,
    ),
]


def create_migration_manager(db_path: Path) -> MigrationManager:
    """Create a migration manager with default migrations.

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured MigrationManager
    """
    backend = SQLiteMigrationBackend(db_path)
    manager = MigrationManager(backend)
    manager.register_many(DVAS_MIGRATIONS)
    return manager
