"""S3/MinIO backend for annotation object storage.

Requires: pip install boto3
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import orjson

from dvas.data.schemas import Annotation
from dvas.persistence.backends.base import BackendStats, BackendConfig, StorageBackend
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional import
try:
    import boto3
    from botocore.exceptions import ClientError

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


class S3Config(BackendConfig):
    """Configuration for S3/MinIO backend."""

    def __init__(
        self,
        bucket: str = "dvas-annotations",
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        name: str = "s3",
        read_only: bool = False,
        compression: Optional[str] = None,
    ):
        from dvas.persistence.backends.base import BackendType

        super().__init__(
            backend_type=BackendType.S3,
            name=name,
            read_only=read_only,
            compression=compression,
        )
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url  # For MinIO or custom S3 endpoints
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key


class S3Backend(StorageBackend):
    """S3/MinIO object storage backend for annotations.

    Stores annotations as JSON objects in S3-compatible storage:
        s3://{bucket}/{prefix}/{source}/{annotation_id[:2]}/{annotation_id}.json

    Compatible with AWS S3, MinIO, and other S3-compatible services.
    """

    def __init__(self, config: Optional[S3Config] = None):
        if not HAS_BOTO3:
            raise ImportError("S3 backend requires boto3. Install with: pip install boto3")

        config = config or S3Config()
        super().__init__(config)
        self.config: S3Config = config
        self._client = None
        self._resource = None

    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            session_kwargs = {"region_name": self.config.region}
            if self.config.access_key and self.config.secret_key:
                session_kwargs["aws_access_key_id"] = self.config.access_key
                session_kwargs["aws_secret_access_key"] = self.config.secret_key

            session = boto3.session.Session(**session_kwargs)
            client_kwargs = {}
            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            self._client = session.client("s3", **client_kwargs)
            self._resource = boto3.resource("s3", **client_kwargs)

        return self._client

    def open(self) -> None:
        """Initialize S3 connection and ensure bucket exists."""
        client = self._get_client()

        try:
            client.head_bucket(Bucket=self.config.bucket)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                if not self.config.read_only:
                    client.create_bucket(
                        Bucket=self.config.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.config.region}
                        if self.config.region != "us-east-1"
                        else {},
                    )
                    logger.info("s3_bucket_created", bucket=self.config.bucket)
                else:
                    raise RuntimeError(f"Bucket does not exist: {self.config.bucket}")

        self._closed = False
        logger.info(
            "s3_backend_opened", bucket=self.config.bucket, endpoint=self.config.endpoint_url
        )

    def close(self) -> None:
        """Close S3 connection."""
        self._client = None
        self._resource = None
        self._closed = True
        logger.info("s3_backend_closed")

    def health_check(self) -> Tuple[bool, str]:
        """Check S3 health."""
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.config.bucket)
            return True, "healthy"
        except Exception as e:
            return False, str(e)

    def _get_key(self, annotation_id: str, source: str) -> str:
        """Get S3 key for an annotation."""
        prefix = f"{self.config.prefix}/" if self.config.prefix else ""
        return f"{prefix}{source}/{annotation_id[:2]}/{annotation_id}.json"

    def _get_source_path(self, source: str) -> str:
        """Get prefix path for a source category."""
        prefix = f"{self.config.prefix}/" if self.config.prefix else ""
        return f"{prefix}{source}"

    def save(
        self,
        annotation: Annotation,
        source: str = "model",
        overwrite: bool = False,
    ) -> str:
        """Save an annotation to S3."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot save to read-only backend")

        key = self._get_key(annotation.id, source)

        if not overwrite:
            try:
                self._get_client().head_object(Bucket=self.config.bucket, Key=key)
                raise FileExistsError(f"Annotation already exists: s3://{self.config.bucket}/{key}")
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise

        # Update timestamp
        annotation.updated_at = datetime.now(timezone.utc)

        # Serialize
        data = annotation.model_dump()
        body = orjson.dumps(data, option=orjson.OPT_INDENT_2)

        self._get_client().put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

        logger.debug("annotation_saved_s3", id=annotation.id, key=key, bucket=self.config.bucket)
        return f"s3://{self.config.bucket}/{key}"

    def load(self, annotation_id: str, source: str = "model") -> Optional[Annotation]:
        """Load an annotation from S3."""
        self.ensure_open()

        key = self._get_key(annotation_id, source)

        try:
            response = self._get_client().get_object(Bucket=self.config.bucket, Key=key)
            data = orjson.loads(response["Body"].read())
            return Annotation.model_validate(data)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                # Try other sources
                for src in ["model", "gold", "reviewed"]:
                    if src != source:
                        try:
                            key = self._get_key(annotation_id, src)
                            response = self._get_client().get_object(
                                Bucket=self.config.bucket, Key=key
                            )
                            data = orjson.loads(response["Body"].read())
                            return Annotation.model_validate(data)
                        except ClientError:
                            continue
                return None
            raise

    def load_all(
        self,
        source: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> Iterator[Annotation]:
        """Load all annotations from S3."""
        self.ensure_open()

        sources = [source] if source else ["gold", "model", "reviewed"]

        for src in sources:
            prefix = self._get_source_path(src)
            paginator = self._get_client().get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue

                    try:
                        response = self._get_client().get_object(Bucket=self.config.bucket, Key=key)
                        data = orjson.loads(response["Body"].read())
                        annotation = Annotation.model_validate(data)

                        if video_id is None or annotation.video_id == video_id:
                            yield annotation

                    except Exception as e:
                        logger.warning("failed_to_load_annotation_s3", key=key, error=str(e))

    def delete(self, annotation_id: str, source: str = "model") -> bool:
        """Delete an annotation from S3."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot delete from read-only backend")

        key = self._get_key(annotation_id, source)

        try:
            self._get_client().delete_object(Bucket=self.config.bucket, Key=key)
            logger.debug("annotation_deleted_s3", id=annotation_id, key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            raise

    def exists(self, annotation_id: str, source: str = "model") -> bool:
        """Check if annotation exists."""
        self.ensure_open()

        key = self._get_key(annotation_id, source)

        try:
            self._get_client().head_object(Bucket=self.config.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_storage_path(self, annotation_id: str, source: str = "model") -> str:
        """Get storage path for an annotation."""
        return f"s3://{self.config.bucket}/{self._get_key(annotation_id, source)}"

    def get_statistics(self) -> BackendStats:
        """Get storage statistics."""
        self.ensure_open()

        stats = BackendStats()
        by_source: Dict[str, int] = {}
        total_objects = 0
        total_size = 0

        for source in ["gold", "model", "reviewed"]:
            prefix = self._get_source_path(source)
            paginator = self._get_client().get_paginator("list_objects_v2")

            count = 0
            size = 0
            for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".json"):
                        count += 1
                        size += obj["Size"]

            by_source[source] = count
            total_objects += count
            total_size += size

        stats.total_annotations = total_objects
        stats.by_source = by_source
        stats.storage_size_bytes = total_size
        stats.last_modified = datetime.now(timezone.utc)

        return stats

    def create_version(self, name: str) -> str:
        """Create a versioned snapshot by copying objects."""
        self.ensure_open()

        if self.config.read_only:
            raise RuntimeError("Cannot create version in read-only backend")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_prefix = f"{self.config.prefix}/versions/{name}_{timestamp}"

        reviewed_prefix = self._get_source_path("reviewed")
        paginator = self._get_client().get_paginator("list_objects_v2")

        count = 0
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=reviewed_prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue

                # Copy to version prefix
                copy_source = {"Bucket": self.config.bucket, "Key": obj["Key"]}
                relative_path = obj["Key"][len(reviewed_prefix) :]
                new_key = f"{version_prefix}{relative_path}"

                self._get_client().copy_object(
                    CopySource=copy_source,
                    Bucket=self.config.bucket,
                    Key=new_key,
                )
                count += 1

        # Write manifest
        manifest = {
            "name": name,
            "timestamp": timestamp,
            "count": count,
        }
        manifest_key = f"{version_prefix}/manifest.json"
        self._get_client().put_object(
            Bucket=self.config.bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType="application/json",
        )

        logger.info("s3_version_created", name=name, prefix=version_prefix, count=count)
        return f"s3://{self.config.bucket}/{version_prefix}"

    def backup(self, destination: Path) -> None:
        """Not applicable for S3 backend - use S3 versioning or cross-region replication."""
        raise NotImplementedError(
            "S3 backend does not support local backup. "
            "Use S3 versioning or cross-region replication instead."
        )

    def restore(self, source: Path) -> None:
        """Not applicable for S3 backend."""
        raise NotImplementedError(
            "S3 backend does not support local restore. Use S3 restore operations instead."
        )
