"""PostgreSQL backend for annotation metadata indexing.

Requires: pip install asyncpg psycopg2-binary
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Optional import
try:
    import asyncpg  # noqa: F401
    import psycopg2
    from psycopg2.extras import RealDictCursor

    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False


class PostgreSQLConfig(BackendConfig):
    """Configuration for PostgreSQL backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "dvas",
        user: str = "dvas",
        password: str = "",
        name: str = "postgresql",
        read_only: bool = False,
        enable_fts: bool = True,
        enable_versioning: bool = True,
        ssl_mode: str = "prefer",
    ):
        from dvas.persistence.backends.base import BackendType

        super().__init__(
            backend_type=BackendType.POSTGRESQL,
            name=name,
            read_only=read_only,
        )
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.enable_fts = enable_fts
        self.enable_versioning = enable_versioning
        self.ssl_mode = ssl_mode
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"


class PostgreSQLBackend(MetadataBackend):
    """PostgreSQL-backed annotation index for production deployments.

    Provides scalable metadata indexing with full-text search,
    versioning, and advanced querying capabilities.
    """

    def __init__(self, config: Optional[PostgreSQLConfig] = None):
        if not HAS_POSTGRES:
            raise ImportError(
                "PostgreSQL backend requires asyncpg and psycopg2-binary. "
                "Install with: pip install asyncpg psycopg2-binary"
            )

        config = config or PostgreSQLConfig()
        super().__init__(config)
        self.config: PostgreSQLConfig = config
        self._pool = None
        self._sync_conn = None

    def open(self) -> None:
        """Initialize PostgreSQL connection."""
        self._sync_conn = psycopg2.connect(
            self.config.connection_string,
            cursor_factory=RealDictCursor,
        )
        self._create_tables()
        self._closed = False
        logger.info(
            "postgresql_backend_opened", host=self.config.host, database=self.config.database
        )

    def close(self) -> None:
        """Close PostgreSQL connection."""
        if self._sync_conn:
            self._sync_conn.close()
            self._sync_conn = None

        if self._pool:
            # Async pool cleanup
            pass

        self._closed = True
        logger.info("postgresql_backend_closed")

    def health_check(self) -> Tuple[bool, str]:
        """Check PostgreSQL health."""
        try:
            if self._sync_conn is None:
                return False, "Connection not initialized"

            with self._sync_conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True, "healthy"
        except Exception as e:
            return False, str(e)

    def _create_tables(self) -> None:
        """Create PostgreSQL tables and indexes."""
        cur = self._sync_conn.cursor()

        # Main annotations table
        cur.execute("""
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
                tags JSONB DEFAULT '[]',
                parent_id TEXT REFERENCES annotations(id),
                content_hash TEXT,
                storage_path TEXT,
                json_data JSONB NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON annotations(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON annotations(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_version ON annotations(model_version)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prompt_version ON annotations(prompt_version)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_version ON annotations(dataset_version)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_quality ON annotations(quality_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON annotations(content_hash)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_gin ON annotations USING GIN(json_data)"
        )

        # Full-text search
        if self.config.enable_fts:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_annotations_fts
                ON annotations USING GIN(to_tsvector('english', json_data->>'content'))
            """)

        # Versions table
        if self.config.enable_versioning:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    annotation_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS version_annotations (
                    version_id TEXT NOT NULL REFERENCES versions(id),
                    annotation_id TEXT NOT NULL REFERENCES annotations(id),
                    json_snapshot JSONB NOT NULL,
                    PRIMARY KEY (version_id, annotation_id)
                )
            """)

        # Hash tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_hashes (
                video_id TEXT PRIMARY KEY,
                video_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                frame_count INTEGER,
                duration REAL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS frame_hashes (
                id SERIAL PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, frame_index)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_video ON frame_hashes(video_id)")

        # Migrations table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)

        self._sync_conn.commit()
        logger.debug("postgresql_tables_created")

    def _extract_searchable_content(self, annotation: Annotation) -> str:
        """Extract searchable text content."""
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
        """Compute content hash."""
        from dvas.utils.hash import compute_annotation_hash

        return compute_annotation_hash(annotation)

    def index_annotation(self, annotation: Annotation, storage_path: Optional[str] = None) -> None:
        """Index an annotation."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot index to read-only backend")

        _ = self._extract_searchable_content(annotation)  # content extracted for side effects
        content_hash = self._compute_hash(annotation)

        cur = self._sync_conn.cursor()
        cur.execute(
            """
            INSERT INTO annotations (
                id, video_id, video_path, source, model_version,
                quality_score, created_at, updated_at, num_segments,
                total_duration, tags, parent_id, content_hash, storage_path, json_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                source = EXCLUDED.source,
                model_version = EXCLUDED.model_version,
                quality_score = EXCLUDED.quality_score,
                updated_at = EXCLUDED.updated_at,
                num_segments = EXCLUDED.num_segments,
                tags = EXCLUDED.tags,
                content_hash = EXCLUDED.content_hash,
                storage_path = EXCLUDED.storage_path,
                json_data = EXCLUDED.json_data,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (
                annotation.id,
                annotation.video_id,
                annotation.video_path,
                annotation.source,
                annotation.model_version,
                annotation.quality_score,
                annotation.created_at,
                annotation.updated_at,
                len(annotation.segments),
                annotation.get_total_duration()
                if hasattr(annotation, "get_total_duration")
                else 0.0,
                json.dumps(annotation.tags),
                annotation.parent_id,
                content_hash,
                storage_path,
                json.dumps(annotation.model_dump()),
            ),
        )
        self._sync_conn.commit()
        logger.debug("annotation_indexed", id=annotation.id)

    def get(self, annotation_id: str) -> Optional[IndexEntry]:
        """Get index entry."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute(
            """
            SELECT id, video_id, video_path, source, model_version,
                   quality_score, created_at, updated_at, num_segments,
                   total_duration, tags, parent_id, content_hash, storage_path
            FROM annotations WHERE id = %s
        """,
            (annotation_id,),
        )

        row = cur.fetchone()
        if row is None:
            return None

        return IndexEntry(
            id=row["id"],
            video_id=row["video_id"],
            video_path=row["video_path"],
            source=row["source"],
            model_version=row["model_version"],
            quality_score=row["quality_score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            num_segments=row["num_segments"],
            total_duration=row["total_duration"],
            tags=row["tags"],
            parent_id=row["parent_id"],
            content_hash=row["content_hash"],
            storage_path=row["storage_path"],
        )

    def query(self, query_filter: QueryFilter) -> Tuple[List[IndexEntry], int]:
        """Query annotations."""
        self.ensure_open()

        cur = self._sync_conn.cursor()

        conditions = []
        params = []

        if query_filter.video_id:
            conditions.append("video_id = %s")
            params.append(query_filter.video_id)

        if query_filter.source:
            conditions.append("source = %s")
            params.append(query_filter.source)

        if query_filter.model_version:
            conditions.append("model_version = %s")
            params.append(query_filter.model_version)

        if query_filter.min_quality is not None:
            conditions.append("quality_score >= %s")
            params.append(query_filter.min_quality)

        if query_filter.max_quality is not None:
            conditions.append("quality_score <= %s")
            params.append(query_filter.max_quality)

        if query_filter.tags:
            for tag in query_filter.tags:
                conditions.append("tags @> %s")
                params.append(json.dumps([tag]))

        if query_filter.has_parent is not None:
            if query_filter.has_parent:
                conditions.append("parent_id IS NOT NULL")
            else:
                conditions.append("parent_id IS NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get count
        count_sql = f"SELECT COUNT(*) FROM annotations WHERE {where_clause}"
        cur.execute(count_sql, params)
        total = cur.fetchone()[0]

        # Get results
        order_direction = "DESC" if query_filter.order_desc else "ASC"
        sql = f"""
            SELECT id, video_id, video_path, source, model_version,
                   quality_score, created_at, updated_at, num_segments,
                   total_duration, tags, parent_id, content_hash, storage_path
            FROM annotations
            WHERE {where_clause}
            ORDER BY {query_filter.order_by} {order_direction}
            LIMIT %s OFFSET %s
        """
        params.extend([query_filter.limit, query_filter.offset])
        cur.execute(sql, params)

        rows = cur.fetchall()
        entries = [
            IndexEntry(
                id=row["id"],
                video_id=row["video_id"],
                video_path=row["video_path"],
                source=row["source"],
                model_version=row["model_version"],
                quality_score=row["quality_score"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                num_segments=row["num_segments"],
                total_duration=row["total_duration"],
                tags=row["tags"],
                parent_id=row["parent_id"],
                content_hash=row["content_hash"],
                storage_path=row["storage_path"],
            )
            for row in rows
        ]

        return entries, total

    def search(self, query_text: str, limit: int = 100) -> List[SearchResult]:
        """Full-text search."""
        self.ensure_open()

        if not self.config.enable_fts:
            logger.warning("fts_disabled")
            return []

        cur = self._sync_conn.cursor()
        cur.execute(
            """
            SELECT id, json_data, ts_rank(to_tsvector('english', json_data::text), plainto_tsquery('english', %s)) as score
            FROM annotations
            WHERE to_tsvector('english', json_data::text) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """,
            (query_text, query_text, limit),
        )

        rows = cur.fetchall()
        results = []

        for row in rows:
            annotation = Annotation.model_validate(row["json_data"])
            results.append(SearchResult(annotation=annotation, score=row["score"]))

        return results

    def delete_index(self, annotation_id: str) -> bool:
        """Delete from index."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete from read-only backend")

        cur = self._sync_conn.cursor()
        cur.execute("DELETE FROM annotations WHERE id = %s", (annotation_id,))
        self._sync_conn.commit()

        return cur.rowcount > 0

    def create_version(
        self,
        name: str,
        description: str = "",
        annotation_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionInfo:
        """Create version."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot create version in read-only backend")

        import uuid

        version_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        cur = self._sync_conn.cursor()

        if annotation_ids:
            cur.execute(
                "SELECT id, json_data FROM annotations WHERE id = ANY(%s)",
                (annotation_ids,),
            )
        else:
            cur.execute("SELECT id, json_data FROM annotations")

        rows = cur.fetchall()

        cur.execute(
            """
            INSERT INTO versions (id, name, description, created_at, annotation_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (version_id, name, description, now, len(rows), json.dumps(metadata or {})),
        )

        for row in rows:
            cur.execute(
                """
                INSERT INTO version_annotations (version_id, annotation_id, json_snapshot)
                VALUES (%s, %s, %s)
                """,
                (version_id, row["id"], json.dumps(row["json_data"])),
            )

        self._sync_conn.commit()

        return VersionInfo(
            id=version_id,
            name=name,
            description=description,
            created_at=now,
            annotation_count=len(rows),
            metadata=metadata or {},
        )

    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Get version info."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("SELECT * FROM versions WHERE id = %s", (version_id,))
        row = cur.fetchone()

        if row is None:
            return None

        return VersionInfo(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            annotation_count=row["annotation_count"],
            metadata=row["metadata"],
        )

    def list_versions(self) -> List[VersionInfo]:
        """List versions."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("SELECT * FROM versions ORDER BY created_at DESC")
        rows = cur.fetchall()

        return [
            VersionInfo(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=row["created_at"],
                annotation_count=row["annotation_count"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def restore_version(self, version_id: str) -> List[Dict[str, Any]]:
        """Restore version."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute(
            "SELECT json_snapshot FROM version_annotations WHERE version_id = %s",
            (version_id,),
        )
        rows = cur.fetchall()

        return [row["json_snapshot"] for row in rows]

    def delete_version(self, version_id: str) -> bool:
        """Delete version."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete version from read-only backend")

        cur = self._sync_conn.cursor()
        cur.execute("DELETE FROM version_annotations WHERE version_id = %s", (version_id,))
        cur.execute("DELETE FROM versions WHERE id = %s", (version_id,))
        self._sync_conn.commit()

        return cur.rowcount > 0

    def diff_versions(
        self,
        version_id1: str,
        version_id2: str,
        annotation_id: Optional[str] = None,
    ) -> List[DiffResult]:
        """Compare versions."""
        self.ensure_open()

        cur = self._sync_conn.cursor()

        sql = """
            SELECT va1.annotation_id, va1.json_snapshot as snapshot1, va2.json_snapshot as snapshot2
            FROM version_annotations va1
            JOIN version_annotations va2 ON va1.annotation_id = va2.annotation_id
            WHERE va1.version_id = %s AND va2.version_id = %s
        """
        params = [version_id1, version_id2]

        if annotation_id:
            sql += " AND va1.annotation_id = %s"
            params.append(annotation_id)

        cur.execute(sql, params)
        rows = cur.fetchall()

        results = []
        for row in rows:
            data1 = row["snapshot1"]
            data2 = row["snapshot2"]
            diff = self._compute_diff(row["annotation_id"], data1, data2)
            results.append(diff)

        return results

    def _compute_diff(self, annotation_id: str, data1: Dict, data2: Dict) -> DiffResult:
        """Compute diff."""
        field_changes = {}
        all_fields = set(data1.keys()) | set(data2.keys())

        for field in all_fields:
            val1 = data1.get(field)
            val2 = data2.get(field)
            if val1 != val2:
                field_changes[field] = (val1, val2)

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
            unchanged=len(field_changes) == 0
            and not segments_added
            and not segments_removed
            and not segments_modified,
        )

    def get_statistics(self) -> BackendStats:
        """Get statistics."""
        self.ensure_open()

        cur = self._sync_conn.cursor()

        cur.execute("SELECT COUNT(*) FROM annotations")
        total = cur.fetchone()[0]

        cur.execute("SELECT source, COUNT(*) FROM annotations GROUP BY source")
        by_source = {row["source"]: row["count"] for row in cur.fetchall()}

        cur.execute("SELECT model_version, COUNT(*) FROM annotations GROUP BY model_version")
        by_model = {row["model_version"] or "unknown": row["count"] for row in cur.fetchall()}

        cur.execute("SELECT pg_total_relation_size('annotations')")
        size = cur.fetchone()[0]

        return BackendStats(
            total_annotations=total,
            by_source=by_source,
            by_model=by_model,
            index_size_bytes=size,
            last_modified=datetime.now(timezone.utc),
        )

    def vacuum(self) -> None:
        """Vacuum database."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("VACUUM")
        logger.info("postgresql_vacuumed")

    def optimize(self) -> None:
        """Optimize."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("ANALYZE annotations")
        self._sync_conn.commit()
        logger.info("postgresql_optimized")

    def reindex(self) -> None:
        """Reindex."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("REINDEX TABLE annotations")
        logger.info("postgresql_reindexed")

    def backup(self, destination: Path) -> Path:
        """Backup using pg_dump."""
        self.ensure_open()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = destination / f"annotations_{timestamp}.sql"
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        import subprocess

        cmd = [
            "pg_dump",
            "-h",
            self.config.host,
            "-p",
            str(self.config.port),
            "-U",
            self.config.user,
            "-d",
            self.config.database,
            "-f",
            str(backup_path),
        ]

        env = {"PGPASSWORD": self.config.password}
        subprocess.run(cmd, env=env, check=True)

        logger.info("postgresql_backup_created", path=str(backup_path))
        return backup_path

    def restore(self, source: Path) -> None:
        """Restore using psql."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot restore to read-only backend")

        import subprocess

        cmd = [
            "psql",
            "-h",
            self.config.host,
            "-p",
            str(self.config.port),
            "-U",
            self.config.user,
            "-d",
            self.config.database,
            "-f",
            str(source),
        ]

        env = {"PGPASSWORD": self.config.password}
        subprocess.run(cmd, env=env, check=True)

        logger.info("postgresql_restored", source=str(source))

    # Hash index methods
    def index_video_hash(
        self,
        video_id: str,
        video_path: str,
        content_hash: str,
        frame_count: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Index video hash."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute(
            """
            INSERT INTO video_hashes (video_id, video_path, content_hash, frame_count, duration)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                video_path = EXCLUDED.video_path,
                content_hash = EXCLUDED.content_hash,
                frame_count = EXCLUDED.frame_count,
                duration = EXCLUDED.duration
            """,
            (video_id, video_path, content_hash, frame_count, duration),
        )
        self._sync_conn.commit()

    def get_video_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get video by hash."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("SELECT * FROM video_hashes WHERE content_hash = %s", (content_hash,))
        row = cur.fetchone()

        return dict(row) if row else None

    def index_frame_hash(self, video_id: str, frame_index: int, content_hash: str) -> None:
        """Index frame hash."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute(
            """
            INSERT INTO frame_hashes (video_id, frame_index, content_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (video_id, frame_index) DO UPDATE SET
                content_hash = EXCLUDED.content_hash
            """,
            (video_id, frame_index, content_hash),
        )
        self._sync_conn.commit()

    def get_frames_by_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """Get frames by hash."""
        self.ensure_open()

        cur = self._sync_conn.cursor()
        cur.execute("SELECT * FROM frame_hashes WHERE content_hash = %s", (content_hash,))
        return [dict(row) for row in cur.fetchall()]
