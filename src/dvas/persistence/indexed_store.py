"""Indexed annotation store with SQLite backend.

Provides fast querying, full-text search, and versioning
on top of the file-based annotation storage.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import orjson

from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IndexStoreConfig:
    """Configuration for the indexed store."""

    db_path: Path = Path("data/annotations_index.db")
    wal_mode: bool = True
    enable_fts: bool = True
    enable_versioning: bool = True
    auto_sync: bool = True  # Auto-sync with file store on startup
    sync_interval: int = 300  # Seconds between auto-syncs
    max_search_results: int = 1000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnnotationIndex:
    """Indexed fields for an annotation."""

    id: str
    video_id: str
    video_path: str
    source: str
    model_version: Optional[str]
    quality_score: Optional[float]
    created_at: str
    updated_at: Optional[str]
    num_segments: int
    total_duration: float
    tags: str  # JSON list
    parent_id: Optional[str] = None


@dataclass
class SearchResult:
    """Result from a search query."""

    annotation: Annotation
    score: float = 0.0
    highlights: List[str] = field(default_factory=list)


@dataclass
class VersionInfo:
    """Information about a stored version."""

    id: str
    name: str
    description: str
    created_at: datetime
    annotation_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationQuery:
    """Query parameters for annotation search."""

    video_id: Optional[str] = None
    source: Optional[str] = None
    model_version: Optional[str] = None
    min_quality: Optional[float] = None
    max_quality: Optional[float] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    has_parent: Optional[bool] = None
    text_search: Optional[str] = None
    order_by: str = "created_at"
    order_desc: bool = True
    limit: int = 100
    offset: int = 0


# ---------------------------------------------------------------------------
# Index Store
# ---------------------------------------------------------------------------

class IndexStore:
    """SQLite-backed annotation index with full-text search.

    Provides fast querying and search capabilities on top of the
    file-based annotation storage.

    Usage::

        store = IndexStore(config)
        store.create_index()
        store.index_annotation(annotation)

        # Search by text
        results = store.search("robotic manipulation")

        # Query by filters
        query = AnnotationQuery(video_id="vid_001", min_quality=0.8)
        results = store.query(query)
    """

    def __init__(self, config: Optional[IndexStoreConfig] = None) -> None:
        self.config = config or IndexStoreConfig()
        self._db_path = Path(self.config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # Use thread-local storage for connections
        self._local = threading.local()
        # Keep main thread connection for backward compatibility
        self._connection: Optional[sqlite3.Connection] = None
        self._main_thread_id = threading.current_thread().ident

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection for the current thread."""
        current_thread_id = threading.current_thread().ident

        # Fast path: main thread uses instance connection
        if current_thread_id == self._main_thread_id and self._connection is not None:
            return self._connection

        # Thread-local storage for non-main threads
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=True,  # Now we enforce same thread
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._local.connection.row_factory = sqlite3.Row

            # Enable WAL mode for concurrent reads/writes
            if self.config.wal_mode:
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                self._local.connection.execute("PRAGMA synchronous=NORMAL")

            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys=ON")

            logger.debug("sqlite_connection_created", thread_id=current_thread_id)

        return self._local.connection

    def _get_or_create_main_connection(self) -> None:
        """Create main thread connection if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._connection.row_factory = sqlite3.Row

            if self.config.wal_mode:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=NORMAL")

            self._connection.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        """Close all database connections."""
        # Close main thread connection
        if self._connection:
            self._connection.close()
            self._connection = None

        # Note: Cannot close thread-local connections from other threads
        # They will be closed when threads exit

    def create_index(self) -> None:
        """Create all database tables and indexes."""
        conn = self._get_connection()

        # Main annotations table
        conn.execute("""
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
                json_data TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES annotations(id)
            )
        """)

        # Indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON annotations(video_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON annotations(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_version ON annotations(model_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON annotations(quality_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON annotations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON annotations(parent_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_video_source ON annotations(video_id, source)"
        )

        # Full-text search virtual table (FTS5)
        if self.config.enable_fts:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS annotations_fts USING fts5(
                    id,
                    content,
                    tokenize='porter'
                )
            """)

        # Versions table
        if self.config.enable_versioning:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    annotation_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_annotations (
                    version_id TEXT NOT NULL,
                    annotation_id TEXT NOT NULL,
                    json_snapshot TEXT NOT NULL,
                    PRIMARY KEY (version_id, annotation_id),
                    FOREIGN KEY (version_id) REFERENCES versions(id),
                    FOREIGN KEY (annotation_id) REFERENCES annotations(id)
                )
            """)

        # Sync log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                annotations_synced INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)

        conn.commit()
        logger.info("index_created", db_path=str(self._db_path))

    def index_annotation(self, annotation: Annotation) -> None:
        """Index a single annotation."""
        with self._lock:
            conn = self._get_connection()

            # Extract searchable content
            content = self._extract_searchable_content(annotation)
            tags_json = json.dumps(annotation.tags)

            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (
                    id, video_id, video_path, source, model_version,
                    quality_score, created_at, updated_at, num_segments,
                    total_duration, tags, parent_id, json_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.id,
                    annotation.video_id,
                    annotation.video_path,
                    annotation.source,
                    annotation.model_version,
                    annotation.quality_score,
                    annotation.created_at.isoformat(),
                    annotation.updated_at.isoformat() if annotation.updated_at else None,
                    len(annotation.segments),
                    annotation.get_total_duration(),
                    tags_json,
                    annotation.parent_id,
                    orjson.dumps(annotation.model_dump()).decode("utf-8"),
                ),
            )

            # Update FTS index
            if self.config.enable_fts:
                conn.execute(
                    "INSERT OR REPLACE INTO annotations_fts (id, content) VALUES (?, ?)",
                    (annotation.id, content),
                )

            conn.commit()

    def _extract_searchable_content(self, annotation: Annotation) -> str:
        """Extract searchable text content from an annotation."""
        parts = [
            annotation.video_id,
            annotation.video_path,
        ]

        for segment in annotation.segments:
            parts.append(segment.caption)
            if segment.caption_dense:
                parts.append(segment.caption_dense)
            for qa in segment.qa_pairs:
                parts.append(qa.question)
                parts.append(qa.answer)
            for obj in segment.objects:
                parts.append(obj.name)
            for action in segment.actions:
                parts.append(action.verb)
                parts.append(action.noun)

        return " ".join(parts)

    def get(self, annotation_id: str) -> Optional[Annotation]:
        """Get an annotation by ID."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT json_data FROM annotations WHERE id = ?",
                (annotation_id,),
            ).fetchone()

            if row is None:
                return None

            data = orjson.loads(row["json_data"])
            return Annotation.model_validate(data)

    def query(self, query: AnnotationQuery) -> Tuple[List[Annotation], int]:
        """Query annotations with filters.

        Returns:
            Tuple of (annotations, total_count)
        """
        with self._lock:
            conn = self._get_connection()

            conditions = []
            params: List[Any] = []

            if query.video_id:
                conditions.append("video_id = ?")
                params.append(query.video_id)

            if query.source:
                conditions.append("source = ?")
                params.append(query.source)

            if query.model_version:
                conditions.append("model_version = ?")
                params.append(query.model_version)

            if query.min_quality is not None:
                conditions.append("quality_score >= ?")
                params.append(query.min_quality)

            if query.max_quality is not None:
                conditions.append("quality_score <= ?")
                params.append(query.max_quality)

            if query.tags:
                for tag in query.tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")

            if query.created_after:
                conditions.append("created_at >= ?")
                params.append(query.created_after.isoformat())

            if query.created_before:
                conditions.append("created_at <= ?")
                params.append(query.created_before.isoformat())

            if query.has_parent is not None:
                if query.has_parent:
                    conditions.append("parent_id IS NOT NULL")
                else:
                    conditions.append("parent_id IS NULL")

            # Build WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get total count
            count_sql = f"SELECT COUNT(*) FROM annotations WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            # Get results
            order_direction = "DESC" if query.order_desc else "ASC"
            sql = f"""
                SELECT json_data FROM annotations
                WHERE {where_clause}
                ORDER BY {query.order_by} {order_direction}
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit, query.offset])

            rows = conn.execute(sql, params).fetchall()
            annotations = [
                Annotation.model_validate(orjson.loads(row["json_data"]))
                for row in rows
            ]

            return annotations, total

    def search(self, query_text: str, limit: int = 100) -> List[SearchResult]:
        """Full-text search annotations.

        Uses FTS5 for fast text search with ranking.
        """
        if not self.config.enable_fts:
            logger.warning("fts_disabled")
            return []

        with self._lock:
            conn = self._get_connection()

            # FTS5 search with ranking
            sql = """
                SELECT
                    a.id,
                    a.json_data,
                    rank
                FROM annotations_fts fts
                JOIN annotations a ON fts.id = a.id
                WHERE annotations_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """

            rows = conn.execute(sql, (query_text, limit)).fetchall()
            results = []

            for row in rows:
                annotation = Annotation.model_validate(orjson.loads(row["json_data"]))
                # Calculate a normalized score (lower rank is better)
                score = max(0.0, 1.0 - abs(row["rank"]) / 1000.0)
                results.append(SearchResult(annotation=annotation, score=score))

            return results

    def delete(self, annotation_id: str) -> bool:
        """Delete an annotation from the index."""
        with self._lock:
            conn = self._get_connection()

            # Delete from FTS
            if self.config.enable_fts:
                conn.execute(
                    "DELETE FROM annotations_fts WHERE id = ?",
                    (annotation_id,),
                )

            # Delete from main table
            cursor = conn.execute(
                "DELETE FROM annotations WHERE id = ?",
                (annotation_id,),
            )
            conn.commit()

            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            conn = self._get_connection()

            stats = {}

            # Total annotations
            row = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()
            stats["total_annotations"] = row[0] if row else 0

            # By source
            rows = conn.execute(
                "SELECT source, COUNT(*) FROM annotations GROUP BY source"
            ).fetchall()
            stats["by_source"] = {row[0]: row[1] for row in rows}

            # Quality distribution
            row = conn.execute(
                """
                SELECT
                    AVG(quality_score) as avg_quality,
                    MIN(quality_score) as min_quality,
                    MAX(quality_score) as max_quality
                FROM annotations
                WHERE quality_score IS NOT NULL
                """
            ).fetchone()
            if row:
                stats["quality_stats"] = {
                    "average": row["avg_quality"],
                    "min": row["min_quality"],
                    "max": row["max_quality"],
                }

            # Model versions
            rows = conn.execute(
                "SELECT model_version, COUNT(*) FROM annotations GROUP BY model_version"
            ).fetchall()
            stats["by_model"] = {row[0] or "unknown": row[1] for row in rows}

            # Database size
            row = conn.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            ).fetchone()
            stats["db_size_bytes"] = row[0] if row else 0

            return stats

    # -----------------------------------------------------------------------
    # Versioning
    # -----------------------------------------------------------------------

    def create_version(
        self,
        name: str,
        description: str = "",
        annotation_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a versioned snapshot of annotations."""
        if not self.config.enable_versioning:
            raise RuntimeError("Versioning is disabled")

        with self._lock:
            conn = self._get_connection()
            version_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            # Get annotations to version
            if annotation_ids:
                rows = conn.execute(
                    "SELECT id, json_data FROM annotations WHERE id IN ({})".format(
                        ",".join("?" * len(annotation_ids))
                    ),
                    annotation_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, json_data FROM annotations"
                ).fetchall()

            # Insert version
            conn.execute(
                """
                INSERT INTO versions (id, name, description, created_at, annotation_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    name,
                    description,
                    now,
                    len(rows),
                    json.dumps(metadata or {}),
                ),
            )

            # Insert version snapshots
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO version_annotations (version_id, annotation_id, json_snapshot)
                    VALUES (?, ?, ?)
                    """,
                    (version_id, row["id"], row["json_data"]),
                )

            conn.commit()

            return VersionInfo(
                id=version_id,
                name=name,
                description=description,
                created_at=now,
                annotation_count=len(rows),
                metadata=metadata or {},
            )

    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Get version information."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM versions WHERE id = ?",
                (version_id,),
            ).fetchone()

            if row is None:
                return None

            return VersionInfo(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=datetime.fromisoformat(row["created_at"]),
                annotation_count=row["annotation_count"],
                metadata=json.loads(row["metadata"]),
            )

    def restore_version(self, version_id: str) -> List[Annotation]:
        """Restore annotations from a version."""
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT json_snapshot FROM version_annotations WHERE version_id = ?",
                (version_id,),
            ).fetchall()

            annotations = []
            for row in rows:
                data = orjson.loads(row["json_snapshot"])
                annotations.append(Annotation.model_validate(data))

            return annotations

    def list_versions(self) -> List[VersionInfo]:
        """List all versions."""
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY created_at DESC"
            ).fetchall()

            return [
                VersionInfo(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    annotation_count=row["annotation_count"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in rows
            ]

    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        with self._lock:
            conn = self._get_connection()

            # Delete version annotations first
            conn.execute(
                "DELETE FROM version_annotations WHERE version_id = ?",
                (version_id,),
            )

            # Delete version
            cursor = conn.execute(
                "DELETE FROM versions WHERE id = ?",
                (version_id,),
            )
            conn.commit()

            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Sync with file store
    # -----------------------------------------------------------------------

    def sync_from_store(self, store) -> Dict[str, int]:
        """Sync annotations from file store to index.

        Args:
            store: AnnotationStore instance

        Returns:
            Dict with sync statistics
        """
        with self._lock:
            conn = self._get_connection()

            # Start sync log
            cursor = conn.execute(
                "INSERT INTO sync_log (status) VALUES ('running')"
            )
            sync_id = cursor.lastrowid

            synced = 0
            try:
                for source in ["gold", "model", "reviewed"]:
                    for annotation in store.load_all(source=source):
                        self.index_annotation(annotation)
                        synced += 1

                conn.execute(
                    """
                    UPDATE sync_log
                    SET completed_at = ?, annotations_synced = ?, status = 'completed'
                    WHERE id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), synced, sync_id),
                )
                conn.commit()

            except Exception as e:
                conn.execute(
                    "UPDATE sync_log SET status = 'failed' WHERE id = ?",
                    (sync_id,),
                )
                conn.commit()
                raise

            return {"synced": synced, "sync_id": sync_id}

    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sync history."""
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT * FROM sync_log
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "id": row["id"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "annotations_synced": row["annotations_synced"],
                    "status": row["status"],
                }
                for row in rows
            ]

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    def vacuum(self) -> None:
        """Run VACUUM to reclaim space."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("VACUUM")
            logger.info("index_vacuumed")

    def optimize(self) -> None:
        """Optimize the database (run ANALYZE and optimize FTS)."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("ANALYZE")

            if self.config.enable_fts:
                conn.execute("INSERT INTO annotations_fts(annotations_fts) VALUES('optimize')")

            conn.commit()
            logger.info("index_optimized")

    def reindex(self) -> None:
        """Rebuild all indexes."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("REINDEX")
            logger.info("index_rebuilt")
