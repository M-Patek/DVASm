"""SQLite backend for annotation metadata indexing."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import (
    BackendConfig,
    BackendStats,
    DiffResult,
    IndexEntry,
    MetadataBackend,
    QueryFilter,
    SearchResult,
    VersionInfo,
)
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteConfig(BackendConfig):
    """Configuration for SQLite backend."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        name: str = "sqlite",
        read_only: bool = False,
        wal_mode: bool = True,
        enable_fts: bool = True,
        enable_versioning: bool = True,
    ):
        from dvas.config import settings
        from dvas.persistence.backends.base import BackendType

        super().__init__(
            backend_type=BackendType.SQLITE,
            name=name,
            read_only=read_only,
        )
        self.db_path = Path(db_path or settings.DATA_ROOT / "annotations_index.db")
        self.wal_mode = wal_mode
        self.enable_fts = enable_fts
        self.enable_versioning = enable_versioning


class SQLiteBackend(MetadataBackend):
    """SQLite-backed annotation index with full-text search.

    Provides fast querying, full-text search, and versioning capabilities
    for annotation metadata.
    """

    def __init__(self, config: Optional[SQLiteConfig] = None):
        config = config or SQLiteConfig()
        super().__init__(config)
        self.config: SQLiteConfig = config
        self._db_path = Path(self.config.db_path)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connection: Optional[sqlite3.Connection] = None
        self._main_thread_id = threading.current_thread().ident

    def open(self) -> None:
        """Initialize the database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        self._closed = False
        logger.info("sqlite_backend_opened", db_path=str(self._db_path))

    def close(self) -> None:
        """Close database connections."""
        if self._connection:
            self._connection.close()
            self._connection = None
        self._closed = True
        logger.info("sqlite_backend_closed")

    def health_check(self) -> Tuple[bool, str]:
        """Check database health."""
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1")
            return True, "healthy"
        except Exception as e:
            return False, str(e)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection for the current thread."""
        current_thread_id = threading.current_thread().ident

        # Fast path: main thread uses instance connection
        if current_thread_id == self._main_thread_id and self._connection is not None:
            return self._connection

        # Thread-local storage for non-main threads
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=True,
            )
            self._local.connection.row_factory = sqlite3.Row

            if self.config.wal_mode:
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                self._local.connection.execute("PRAGMA synchronous=NORMAL")

            self._local.connection.execute("PRAGMA foreign_keys=ON")

        return self._local.connection

    def _create_tables(self) -> None:
        """Create database tables and indexes."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        # Main annotations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                video_path TEXT NOT NULL,
                source TEXT NOT NULL,
                model_version TEXT,
                prompt_version TEXT,
                dataset_version TEXT,
                quality_score REAL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                num_segments INTEGER DEFAULT 0,
                total_duration REAL DEFAULT 0.0,
                tags TEXT DEFAULT '[]',
                parent_id TEXT,
                content_hash TEXT,
                storage_path TEXT,
                json_data TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_id) REFERENCES annotations(id)
            )
        """)

        # Indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON annotations(video_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON annotations(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_version ON annotations(model_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_version ON annotations(prompt_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_version ON annotations(dataset_version)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON annotations(quality_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON annotations(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_parent ON annotations(parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_source ON annotations(video_id, source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON annotations(content_hash)")

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

        # Video hash index table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_hashes (
                video_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                frame_count INTEGER,
                duration REAL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Frame hash index table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS frame_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, frame_index)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_video ON frame_hashes(video_id)")

        # Schema migrations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
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

        # Backup log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backup_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                annotation_count INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                status TEXT DEFAULT 'completed'
            )
        """)

        conn.commit()
        conn.close()
        logger.debug("sqlite_tables_created")

    def index_annotation(self, annotation: Annotation, storage_path: Optional[str] = None) -> None:
        """Index a single annotation."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot index to read-only backend")

        with self._lock:
            conn = self._get_connection()

            content = self._extract_searchable_content(annotation)
            tags_json = json.dumps(annotation.tags)

            # Compute content hash for integrity
            content_hash = self._compute_hash(annotation)

            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (
                    id, video_id, video_path, source, model_version,
                    quality_score, created_at, updated_at, num_segments,
                    total_duration, tags, parent_id, content_hash, storage_path, json_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    annotation.get_total_duration() if hasattr(annotation, 'get_total_duration') else 0.0,
                    tags_json,
                    annotation.parent_id,
                    content_hash,
                    storage_path,
                    orjson.dumps(annotation.model_dump()).decode("utf-8"),
                ),
            )

            if self.config.enable_fts:
                conn.execute(
                    "INSERT OR REPLACE INTO annotations_fts (id, content) VALUES (?, ?)",
                    (annotation.id, content),
                )

            conn.commit()
            logger.debug("annotation_indexed", id=annotation.id)

    def _extract_searchable_content(self, annotation: Annotation) -> str:
        """Extract searchable text content from an annotation."""
        parts = [annotation.video_id, annotation.video_path]

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

    def _compute_hash(self, annotation: Annotation) -> str:
        """Compute a hash for the annotation content."""
        from dvas.utils.hash import compute_annotation_hash

        return compute_annotation_hash(annotation)

    def get(self, annotation_id: str) -> Optional[IndexEntry]:
        """Get index entry for an annotation."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                """
                SELECT id, video_id, video_path, source, model_version,
                       quality_score, created_at, updated_at, num_segments,
                       total_duration, tags, parent_id, content_hash, storage_path
                FROM annotations WHERE id = ?
                """,
                (annotation_id,),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_entry(row)

    def _row_to_entry(self, row: sqlite3.Row) -> IndexEntry:
        """Convert database row to IndexEntry."""
        return IndexEntry(
            id=row["id"],
            video_id=row["video_id"],
            video_path=row["video_path"],
            source=row["source"],
            model_version=row["model_version"],
            quality_score=row["quality_score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            num_segments=row["num_segments"],
            total_duration=row["total_duration"],
            tags=json.loads(row["tags"]),
            parent_id=row["parent_id"],
            content_hash=row["content_hash"],
            storage_path=row["storage_path"],
        )

    def query(self, query_filter: QueryFilter) -> Tuple[List[IndexEntry], int]:
        """Query annotations using filters."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()

            conditions = []
            params: List[Any] = []

            if query_filter.video_id:
                conditions.append("video_id = ?")
                params.append(query_filter.video_id)

            if query_filter.source:
                conditions.append("source = ?")
                params.append(query_filter.source)

            if query_filter.model_version:
                conditions.append("model_version = ?")
                params.append(query_filter.model_version)

            if query_filter.prompt_version:
                conditions.append("prompt_version = ?")
                params.append(query_filter.prompt_version)

            if query_filter.dataset_version:
                conditions.append("dataset_version = ?")
                params.append(query_filter.dataset_version)

            if query_filter.min_quality is not None:
                conditions.append("quality_score >= ?")
                params.append(query_filter.min_quality)

            if query_filter.max_quality is not None:
                conditions.append("quality_score <= ?")
                params.append(query_filter.max_quality)

            if query_filter.tags:
                for tag in query_filter.tags:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")

            if query_filter.created_after:
                conditions.append("created_at >= ?")
                params.append(query_filter.created_after.isoformat())

            if query_filter.created_before:
                conditions.append("created_at <= ?")
                params.append(query_filter.created_before.isoformat())

            if query_filter.has_parent is not None:
                if query_filter.has_parent:
                    conditions.append("parent_id IS NOT NULL")
                else:
                    conditions.append("parent_id IS NULL")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get total count
            count_sql = f"SELECT COUNT(*) FROM annotations WHERE {where_clause}"
            total = conn.execute(count_sql, params).fetchone()[0]

            # Get results
            order_direction = "DESC" if query_filter.order_desc else "ASC"
            sql = f"""
                SELECT id, video_id, video_path, source, model_version,
                       quality_score, created_at, updated_at, num_segments,
                       total_duration, tags, parent_id, content_hash, storage_path
                FROM annotations
                WHERE {where_clause}
                ORDER BY {query_filter.order_by} {order_direction}
                LIMIT ? OFFSET ?
            """
            params.extend([query_filter.limit, query_filter.offset])

            rows = conn.execute(sql, params).fetchall()
            entries = [self._row_to_entry(row) for row in rows]

            return entries, total

    def search(self, query_text: str, limit: int = 100) -> List[SearchResult]:
        """Full-text search annotations."""
        self.ensure_open()

        if not self.config.enable_fts:
            logger.warning("fts_disabled")
            return []

        with self._lock:
            conn = self._get_connection()

            sql = """
                SELECT a.id, a.json_data, rank
                FROM annotations_fts fts
                JOIN annotations a ON fts.id = a.id
                WHERE annotations_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """

            # Escape special FTS characters and quote the query
            escaped_query = query_text.replace('"', '""')
            rows = conn.execute(sql, (f'"{escaped_query}"', limit)).fetchall()
            results = []

            for row in rows:
                annotation = Annotation.model_validate(orjson.loads(row["json_data"]))
                score = max(0.0, 1.0 - abs(row["rank"]) / 1000.0)
                results.append(SearchResult(annotation=annotation, score=score))

            return results

    def delete_index(self, annotation_id: str) -> bool:
        """Delete an annotation from the index."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete from read-only backend")

        with self._lock:
            conn = self._get_connection()

            if self.config.enable_fts:
                conn.execute("DELETE FROM annotations_fts WHERE id = ?", (annotation_id,))

            cursor = conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
            conn.commit()

            return cursor.rowcount > 0

    def create_version(
        self,
        name: str,
        description: str = "",
        annotation_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create a versioned snapshot."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot create version in read-only backend")

        if not self.config.enable_versioning:
            raise RuntimeError("Versioning is disabled")

        with self._lock:
            conn = self._get_connection()
            version_id = self._generate_version_id()
            now = datetime.now(timezone.utc)

            if annotation_ids:
                placeholders = ",".join("?" * len(annotation_ids))
                rows = conn.execute(
                    f"SELECT id, json_data FROM annotations WHERE id IN ({placeholders})",
                    annotation_ids,
                ).fetchall()
            else:
                rows = conn.execute("SELECT id, json_data FROM annotations").fetchall()

            conn.execute(
                """
                INSERT INTO versions (id, name, description, created_at, annotation_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, name, description, now, len(rows), json.dumps(metadata or {})),
            )

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

    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        import uuid

        return str(uuid.uuid4())

    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Get version information."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            row = conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()

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

    def list_versions(self) -> List[VersionInfo]:
        """List all versions."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            rows = conn.execute("SELECT * FROM versions ORDER BY created_at DESC").fetchall()

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

    def restore_version(self, version_id: str) -> List[Dict[str, Any]]:
        """Restore annotations from a version."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT json_snapshot FROM version_annotations WHERE version_id = ?",
                (version_id,),
            ).fetchall()

            return [orjson.loads(row["json_snapshot"]) for row in rows]

    def delete_version(self, version_id: str) -> bool:
        """Delete a version."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete version from read-only backend")

        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM version_annotations WHERE version_id = ?", (version_id,))
            cursor = conn.execute("DELETE FROM versions WHERE id = ?", (version_id,))
            conn.commit()

            return cursor.rowcount > 0

    def diff_versions(
        self,
        version_id1: str,
        version_id2: str,
        annotation_id: Optional[str] = None,
    ) -> List[DiffResult]:
        """Compare two versions."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()

            # Get annotations from both versions
            sql = """
                SELECT va1.annotation_id, va1.json_snapshot as snapshot1, va2.json_snapshot as snapshot2
                FROM version_annotations va1
                JOIN version_annotations va2 ON va1.annotation_id = va2.annotation_id
                WHERE va1.version_id = ? AND va2.version_id = ?
            """
            params = [version_id1, version_id2]

            if annotation_id:
                sql += " AND va1.annotation_id = ?"
                params.append(annotation_id)

            rows = conn.execute(sql, params).fetchall()

            results = []
            for row in rows:
                data1 = orjson.loads(row["snapshot1"])
                data2 = orjson.loads(row["snapshot2"])
                diff = self._compute_diff(row["annotation_id"], data1, data2)
                results.append(diff)

            return results

    def _compute_diff(self, annotation_id: str, data1: Dict, data2: Dict) -> DiffResult:
        """Compute diff between two annotation snapshots."""
        field_changes = {}

        # Compare top-level fields
        all_fields = set(data1.keys()) | set(data2.keys())
        for field in all_fields:
            val1 = data1.get(field)
            val2 = data2.get(field)
            if val1 != val2:
                field_changes[field] = (val1, val2)

        # Compare segments
        segments_added = []
        segments_removed = []
        segments_modified = []

        segs1 = data1.get("segments", [])
        segs2 = data2.get("segments", [])

        if len(segs2) > len(segs1):
            segments_added = list(range(len(segs1), len(segs2)))
        elif len(segs1) > len(segs2):
            segments_removed = list(range(len(segs2), len(segs1)))

        for i in range(min(len(segs1), len(segs2))):
            if segs1[i] != segs2[i]:
                segments_modified.append(i)

        return DiffResult(
            annotation_id=annotation_id,
            field_changes=field_changes,
            segments_added=segments_added,
            segments_removed=segments_removed,
            segments_modified=segments_modified,
            unchanged=len(field_changes) == 0 and not segments_added and not segments_removed and not segments_modified,
        )

    def get_statistics(self) -> BackendStats:
        """Get index statistics."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()

            stats = BackendStats()

            row = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()
            stats.total_annotations = row[0] if row else 0

            rows = conn.execute("SELECT source, COUNT(*) FROM annotations GROUP BY source").fetchall()
            stats.by_source = {row[0]: row[1] for row in rows}

            rows = conn.execute(
                "SELECT model_version, COUNT(*) FROM annotations GROUP BY model_version"
            ).fetchall()
            stats.by_model = {row[0] or "unknown": row[1] for row in rows}

            row = conn.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            ).fetchone()
            stats.index_size_bytes = row[0] if row else 0

            return stats

    def vacuum(self) -> None:
        """Run VACUUM to reclaim space."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            conn.execute("VACUUM")
            logger.info("sqlite_vacuumed")

    def optimize(self) -> None:
        """Optimize the database."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            conn.execute("ANALYZE")

            if self.config.enable_fts:
                conn.execute("INSERT INTO annotations_fts(annotations_fts) VALUES('optimize')")

            conn.commit()
            logger.info("sqlite_optimized")

    def reindex(self) -> None:
        """Rebuild all indexes."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            conn.execute("REINDEX")
            logger.info("sqlite_reindexed")

    def backup(self, destination: Path) -> Path:
        """Create a backup of the database."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = destination / f"annotations_index_{timestamp}.db"
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()

            # Log backup
            count = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            conn.execute(
                "INSERT INTO backup_log (backup_path, annotation_count) VALUES (?, ?)",
                (str(backup_path), count),
            )
            conn.commit()

            logger.info("sqlite_backup_created", path=str(backup_path), count=count)
            return backup_path

    def restore(self, source: Path) -> None:
        """Restore from a backup."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot restore to read-only backend")

        if not source.exists():
            raise FileNotFoundError(f"Backup not found: {source}")

        # Close current connection
        self.close()

        # Copy backup over current database
        import shutil

        shutil.copy2(source, self._db_path)

        # Reopen
        self._closed = False
        logger.info("sqlite_restored", source=str(source))

    # -------------------------------------------------------------------------
    # Hash index methods
    # -------------------------------------------------------------------------

    def index_video_hash(
        self,
        video_id: str,
        video_path: str,
        content_hash: str,
        frame_count: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Index video content hash."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot index to read-only backend")

        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO video_hashes
                (video_id, video_path, content_hash, frame_count, duration)
                VALUES (?, ?, ?, ?, ?)
                """,
                (video_id, video_path, content_hash, frame_count, duration),
            )
            conn.commit()

    def get_video_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Find video by content hash."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM video_hashes WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()

            if row is None:
                return None

            return dict(row)

    def index_frame_hash(
        self,
        video_id: str,
        frame_index: int,
        content_hash: str,
    ) -> None:
        """Index frame content hash."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot index to read-only backend")

        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO frame_hashes
                (video_id, frame_index, content_hash)
                VALUES (?, ?, ?)
                """,
                (video_id, frame_index, content_hash),
            )
            conn.commit()

    def get_frames_by_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """Find frames by content hash."""
        self.ensure_open()

        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM frame_hashes WHERE content_hash = ?",
                (content_hash,),
            ).fetchall()

            return [dict(row) for row in rows]
