"""Parquet-backed annotation store with pgvector semantic search.

Provides ParquetAnnotationStore for high-performance columnar storage
and pgvector support for semantic similarity search.

Usage::

    from dvas.data.storage_parquet import ParquetAnnotationStore

    store = ParquetAnnotationStore("data/annotations_parquet")
    store.save(annotation)

    # Semantic search with pgvector
    results = store.semantic_search("robotic manipulation", top_k=10)

    # Columnar analytics
    stats = store.get_columnar_stats()
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple


from dvas.data.schemas import Annotation
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass
class ParquetStoreConfig:
    """Configuration for ParquetAnnotationStore.

    Attributes:
        root_path: Root directory for Parquet files
        partition_by: Column to partition data by (e.g., "source", "date")
        row_group_size: Target rows per row group in Parquet
        compression: Parquet compression codec (zstd, snappy, gzip, etc.)
        enable_duckdb: Whether to use DuckDB for SQL queries
        enable_semantic_search: Whether to enable pgvector semantic search
        embedding_dim: Dimension of embedding vectors for semantic search
        pgvector_table: PostgreSQL table name for pgvector
    """

    root_path: Path = Path("data/annotations_parquet")
    partition_by: str = "source"  # source, date, model_version
    row_group_size: int = 10000
    compression: str = "zstd"
    enable_duckdb: bool = True
    enable_semantic_search: bool = False
    embedding_dim: int = 768
    pgvector_table: str = "annotation_embeddings"


@dataclass
class SemanticSearchResult:
    """Result from semantic search."""

    annotation_id: str
    video_id: str
    score: float
    embedding: Optional[List[float]] = None


class ParquetAnnotationStore:
    """High-performance columnar annotation storage using Parquet.

    Uses Apache Parquet for efficient storage and DuckDB for fast SQL queries.
    Optionally integrates with pgvector for semantic similarity search.

    Attributes:
        config: ParquetStoreConfig
        root_path: Root directory for Parquet files
    """

    def __init__(self, config: Optional[ParquetStoreConfig] = None) -> None:
        self.config = config or ParquetStoreConfig()
        self.root_path = Path(self.config.root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

        # DuckDB connection for SQL queries
        self._duckdb_conn: Optional[Any] = None

        if not PYARROW_AVAILABLE:
            logger.warning("pyarrow not available, Parquet store will use fallback")

        if self.config.enable_duckdb and not DUCKDB_AVAILABLE:
            logger.warning("duckdb not available, SQL queries disabled")
            self.config.enable_duckdb = False

    @property
    def _db_conn(self) -> Any:
        """Get or create DuckDB connection."""
        if self._duckdb_conn is None and DUCKDB_AVAILABLE:
            self._duckdb_conn = duckdb.connect(":memory:")
            # Install and load extensions
            self._duckdb_conn.execute("INSTALL parquet")
            self._duckdb_conn.execute("LOAD parquet")
            logger.debug("duckdb_connection_created")
        return self._duckdb_conn

    def _annotation_to_dict(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert annotation to flat dict for Parquet storage."""
        return {
            "id": annotation.id,
            "video_id": annotation.video_id,
            "video_path": str(annotation.video_path),
            "source": annotation.source,
            "model_version": annotation.model_version or "",
            "quality_score": annotation.quality_score or 0.0,
            "created_at": annotation.created_at.isoformat(),
            "updated_at": annotation.updated_at.isoformat() if annotation.updated_at else None,
            "num_segments": len(annotation.segments),
            "total_duration": annotation.get_total_duration(),
            "tags": json.dumps(annotation.tags),
            "parent_id": annotation.parent_id or "",
            "json_data": annotation.model_dump_json(),
        }

    def _get_partition_path(self, annotation: Annotation) -> Path:
        """Get partition path for an annotation."""
        if self.config.partition_by == "source":
            return self.root_path / annotation.source
        elif self.config.partition_by == "date":
            date_str = annotation.created_at.strftime("%Y-%m-%d")
            return self.root_path / date_str
        elif self.config.partition_by == "model_version":
            version = annotation.model_version or "unknown"
            return self.root_path / version
        else:
            return self.root_path

    def save(self, annotation: Annotation) -> Path:
        """Save an annotation to Parquet storage.

        Args:
            annotation: Annotation to save

        Returns:
            Path to the saved Parquet file
        """
        if not PYARROW_AVAILABLE:
            raise RuntimeError("pyarrow is required for Parquet storage")

        data = self._annotation_to_dict(annotation)
        partition_path = self._get_partition_path(annotation)
        partition_path.mkdir(parents=True, exist_ok=True)

        # Use date-based filename for partitioning
        date_str = annotation.created_at.strftime("%Y%m%d")
        parquet_path = partition_path / f"annotations_{date_str}.parquet"

        # Create or append to Parquet file
        table = pa.Table.from_pydict({k: [v] for k, v in data.items()})

        if parquet_path.exists():
            # Append to existing file
            existing = pq.read_table(str(parquet_path))
            combined = pa.concat_tables([existing, table])
            pq.write_table(
                combined,
                str(parquet_path),
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
            )
        else:
            pq.write_table(
                table,
                str(parquet_path),
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
            )

        logger.debug("annotation_saved_to_parquet", path=str(parquet_path), id=annotation.id)
        return parquet_path

    def save_batch(self, annotations: List[Annotation]) -> List[Path]:
        """Save multiple annotations in batch.

        Args:
            annotations: List of annotations to save

        Returns:
            List of paths to saved Parquet files
        """
        if not annotations:
            return []

        paths = set()
        for annotation in annotations:
            path = self.save(annotation)
            paths.add(path)

        return list(paths)

    def load(self, annotation_id: str) -> Optional[Annotation]:
        """Load an annotation by ID.

        Args:
            annotation_id: Annotation ID

        Returns:
            Annotation or None if not found
        """
        if self.config.enable_duckdb and DUCKDB_AVAILABLE:
            return self._load_with_duckdb(annotation_id)

        # Fallback: scan all Parquet files
        for parquet_file in self.root_path.rglob("*.parquet"):
            try:
                table = pq.read_table(str(parquet_file), columns=["id", "json_data"])
                for i in range(table.num_rows):
                    if table.column("id")[i].as_py() == annotation_id:
                        json_data = table.column("json_data")[i].as_py()
                        return Annotation.model_validate_json(json_data)
            except Exception:
                continue

        return None

    def _load_with_duckdb(self, annotation_id: str) -> Optional[Annotation]:
        """Load annotation using DuckDB query."""
        conn = self._db_conn

        # Find all Parquet files
        parquet_files = list(self.root_path.rglob("*.parquet"))
        if not parquet_files:
            return None

        # Register Parquet files as view
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        conn.execute(
            f"CREATE OR REPLACE VIEW annotations AS SELECT * FROM read_parquet([{file_list}])"
        )

        result = conn.execute(
            "SELECT json_data FROM annotations WHERE id = ?",
            (annotation_id,),
        ).fetchone()

        if result:
            return Annotation.model_validate_json(result[0])
        return None

    def query(
        self,
        source: Optional[str] = None,
        video_id: Optional[str] = None,
        min_quality: Optional[float] = None,
        limit: int = 100,
    ) -> Tuple[List[Annotation], int]:
        """Query annotations with filters.

        Args:
            source: Filter by source
            video_id: Filter by video ID
            min_quality: Minimum quality score
            limit: Maximum results

        Returns:
            Tuple of (annotations, total_count)
        """
        if self.config.enable_duckdb and DUCKDB_AVAILABLE:
            return self._query_with_duckdb(source, video_id, min_quality, limit)

        # Fallback: scan all files
        annotations = []
        for parquet_file in self.root_path.rglob("*.parquet"):
            try:
                table = pq.read_table(str(parquet_file))
                for i in range(table.num_rows):
                    json_data = table.column("json_data")[i].as_py()
                    annotation = Annotation.model_validate_json(json_data)

                    if source and annotation.source != source:
                        continue
                    if video_id and annotation.video_id != video_id:
                        continue
                    if min_quality is not None and (annotation.quality_score or 0) < min_quality:
                        continue

                    annotations.append(annotation)
                    if len(annotations) >= limit:
                        break

                if len(annotations) >= limit:
                    break
            except Exception:
                continue

        return annotations, len(annotations)

    def _query_with_duckdb(
        self,
        source: Optional[str] = None,
        video_id: Optional[str] = None,
        min_quality: Optional[float] = None,
        limit: int = 100,
    ) -> Tuple[List[Annotation], int]:
        """Query using DuckDB SQL."""
        conn = self._db_conn

        parquet_files = list(self.root_path.rglob("*.parquet"))
        if not parquet_files:
            return [], 0

        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        conn.execute(
            f"CREATE OR REPLACE VIEW annotations AS SELECT * FROM read_parquet([{file_list}])"
        )

        # Build WHERE clause
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if video_id:
            conditions.append("video_id = ?")
            params.append(video_id)
        if min_quality is not None:
            conditions.append("quality_score >= ?")
            params.append(min_quality)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        count_result = conn.execute(
            f"SELECT COUNT(*) FROM annotations WHERE {where_clause}",
            params,
        ).fetchone()
        total = count_result[0] if count_result else 0

        # Get results
        results = conn.execute(
            f"""
            SELECT json_data FROM annotations
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()

        annotations = [Annotation.model_validate_json(row[0]) for row in results]
        return annotations, total

    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        metric: str = "cosine",
    ) -> List[SemanticSearchResult]:
        """Semantic search using pgvector.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            metric: Distance metric (cosine, l2, inner)

        Returns:
            List of semantic search results
        """
        if not self.config.enable_semantic_search:
            logger.warning("semantic_search_disabled")
            return []

        # This would connect to a PostgreSQL with pgvector extension
        # For now, return empty (requires actual pgvector setup)
        logger.info("semantic_search_not_implemented", requires="pgvector PostgreSQL extension")
        return []

    def get_columnar_stats(self) -> Dict[str, Any]:
        """Get statistics from Parquet files.

        Returns:
            Dict with columnar statistics
        """
        stats = {
            "total_files": 0,
            "total_rows": 0,
            "total_size_mb": 0.0,
            "by_source": {},
        }

        for parquet_file in self.root_path.rglob("*.parquet"):
            try:
                metadata = pq.read_metadata(str(parquet_file))
                file_size = parquet_file.stat().st_size / (1024 * 1024)  # MB

                stats["total_files"] += 1
                stats["total_rows"] += metadata.num_rows
                stats["total_size_mb"] += file_size

                # Track by source (from directory name)
                source = parquet_file.parent.name
                if source not in stats["by_source"]:
                    stats["by_source"][source] = {"files": 0, "rows": 0, "size_mb": 0.0}

                stats["by_source"][source]["files"] += 1
                stats["by_source"][source]["rows"] += metadata.num_rows
                stats["by_source"][source]["size_mb"] += file_size

            except Exception as e:
                logger.warning(
                    "failed_to_read_parquet_metadata", file=str(parquet_file), error=str(e)
                )

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = self.get_columnar_stats()
        stats["store_type"] = "parquet"
        stats["compression"] = self.config.compression
        stats["partition_by"] = self.config.partition_by
        stats["duckdb_enabled"] = self.config.enable_duckdb and DUCKDB_AVAILABLE
        stats["pyarrow_available"] = PYARROW_AVAILABLE
        return stats

    def compact(self) -> None:
        """Compact small Parquet files into larger ones."""
        if not PYARROW_AVAILABLE:
            return

        for source_dir in self.root_path.iterdir():
            if not source_dir.is_dir():
                continue

            parquet_files = list(source_dir.glob("*.parquet"))
            if len(parquet_files) <= 1:
                continue

            # Read all files and rewrite as one
            try:
                tables = [pq.read_table(str(f)) for f in parquet_files]
                combined = pa.concat_tables(tables)

                # Write combined file
                combined_path = source_dir / "combined.parquet"
                pq.write_table(
                    combined,
                    str(combined_path),
                    compression=self.config.compression,
                    row_group_size=self.config.row_group_size,
                )

                # Remove old files
                for f in parquet_files:
                    f.unlink()

                logger.info("parquet_compacted", source=source_dir.name, files=len(parquet_files))
            except Exception as e:
                logger.warning("parquet_compact_failed", source=source_dir.name, error=str(e))

    def close(self) -> None:
        """Close the store and release resources."""
        if self._duckdb_conn is not None:
            self._duckdb_conn.close()
            self._duckdb_conn = None


# ---------------------------------------------------------------------------
# pgvector integration
# ---------------------------------------------------------------------------


class PGVectorStore:
    """PostgreSQL + pgvector semantic search store.

    Provides vector storage and similarity search using PostgreSQL
    with the pgvector extension.

    Usage::

        store = PGVectorStore(
            dsn="postgresql://user:pass@localhost/db",
            table_name="annotations",
            embedding_dim=768,
        )
        store.create_table()
        store.insert(annotation_id, embedding)

        results = store.similarity_search(query_embedding, top_k=10)
    """

    def __init__(
        self,
        dsn: str,
        table_name: str = "annotation_embeddings",
        embedding_dim: int = 768,
    ):
        self.dsn = dsn
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._connection: Optional[Any] = None

    @property
    def _conn(self) -> Any:
        """Get or create database connection."""
        if self._connection is None:
            import psycopg2

            self._connection = psycopg2.connect(self.dsn)
            self._connection.autocommit = True
        return self._connection

    def create_table(self) -> None:
        """Create the pgvector table and extension."""
        cursor = self._conn.cursor()

        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create table with vector column
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                annotation_id TEXT NOT NULL,
                embedding vector({self.embedding_dim}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create vector index for fast similarity search
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
            ON {self.table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        cursor.close()
        logger.info("pgvector_table_created", table=self.table_name)

    def insert(
        self,
        annotation_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert an embedding into the vector store.

        Args:
            annotation_id: Annotation ID
            embedding: Embedding vector
            metadata: Optional metadata dict
        """

        cursor = self._conn.cursor()
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cursor.execute(
            f"""
            INSERT INTO {self.table_name} (id, annotation_id, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                created_at = CURRENT_TIMESTAMP
        """,
            (str(uuid.uuid4()), annotation_id, embedding_str, json.dumps(metadata or {})),
        )

        cursor.close()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        metric: str = "cosine",
    ) -> List[SemanticSearchResult]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            metric: Distance metric (cosine, l2, inner)

        Returns:
            List of search results
        """
        cursor = self._conn.cursor()
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Map metric to operator
        metric_ops = {
            "cosine": "<=>",
            "l2": "<->",
            "inner": "<#>",
        }
        op = metric_ops.get(metric, "<=>")

        cursor.execute(
            f"""
            SELECT annotation_id, embedding, embedding {op} %s::vector as distance
            FROM {self.table_name}
            ORDER BY embedding {op} %s::vector
            LIMIT %s
        """,
            (embedding_str, embedding_str, top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                SemanticSearchResult(
                    annotation_id=row[0],
                    video_id="",
                    score=1.0 - row[2] if metric == "cosine" else row[2],
                    embedding=row[1],
                )
            )

        cursor.close()
        return results

    def delete(self, annotation_id: str) -> bool:
        """Delete an embedding by annotation ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            f"DELETE FROM {self.table_name} WHERE annotation_id = %s",
            (annotation_id,),
        )
        rowcount = cursor.rowcount
        cursor.close()
        return rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
