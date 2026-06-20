"""API audit logging for DVAS.

Provides request/response logging and audit log storage.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Request

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AuditLogEntry:
    """Single audit log entry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = ""
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    method: str = ""
    path: str = ""
    query_params: str = ""
    status_code: int = 0
    response_time_ms: float = 0.0
    client_ip: str = ""
    user_agent: str = ""
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "method": self.method,
            "path": self.path,
            "query_params": self.query_params,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "request_body": self.request_body,
            "response_body": self.response_body,
            "error": self.error,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditLogEntry:
        return cls(
            id=data.get("id", str(uuid.uuid4())[:12]),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            request_id=data.get("request_id", ""),
            tenant_id=data.get("tenant_id"),
            user_id=data.get("user_id"),
            method=data.get("method", ""),
            path=data.get("path", ""),
            query_params=data.get("query_params", ""),
            status_code=data.get("status_code", 0),
            response_time_ms=data.get("response_time_ms", 0.0),
            client_ip=data.get("client_ip", ""),
            user_agent=data.get("user_agent", ""),
            request_body=data.get("request_body"),
            response_body=data.get("response_body"),
            error=data.get("error"),
            action=data.get("action", ""),
        )


class AuditLogStore:
    """In-memory audit log storage."""

    def __init__(self, max_entries: int = 100000) -> None:
        self._entries: List[AuditLogEntry] = []
        self._max_entries = max_entries

    def add(self, entry: AuditLogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def query(
        self,
        tenant_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        path_prefix: Optional[str] = None,
        status_code: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        results = self._entries
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if path_prefix:
            results = [e for e in results if e.path.startswith(path_prefix)]
        if status_code:
            results = [e for e in results if e.status_code == status_code]
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        return results[offset : offset + limit]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._entries)
        if total == 0:
            return {"total_entries": 0}
        status_codes: Dict[int, int] = {}
        for e in self._entries:
            status_codes[e.status_code] = status_codes.get(e.status_code, 0) + 1
        return {
            "total_entries": total,
            "status_code_distribution": status_codes,
            "oldest_entry": self._entries[0].timestamp if self._entries else None,
            "newest_entry": self._entries[-1].timestamp if self._entries else None,
        }


class AuditLogger:
    """Audit logger for API requests."""

    def __init__(self, store: Optional[AuditLogStore] = None) -> None:
        self._store = store or AuditLogStore()

    def log_request(
        self,
        request: Request,
        status_code: int = 0,
        response_time_ms: float = 0.0,
        request_body: Optional[str] = None,
        response_body: Optional[str] = None,
        error: Optional[str] = None,
        action: str = "",
    ) -> AuditLogEntry:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        tenant_id = getattr(request.state, "tenant_id", None)
        user_id = request.headers.get("X-User-ID")
        client_ip = request.headers.get(
            "X-Forwarded-For",
            request.client.host if request.client else "unknown",
        )
        user_agent = request.headers.get("User-Agent", "")
        entry = AuditLogEntry(
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            status_code=status_code,
            response_time_ms=response_time_ms,
            client_ip=client_ip,
            user_agent=user_agent,
            request_body=request_body,
            response_body=response_body,
            error=error,
            action=action,
        )
        self._store.add(entry)
        logger.info(
            "api_request",
            request_id=request_id,
            tenant_id=tenant_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            response_time_ms=round(response_time_ms, 2),
            action=action,
        )
        return entry

    def log_action(
        self,
        action: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            request_id="",
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            path="",
            method="",
        )
        if details:
            entry.request_body = json.dumps(details)
        self._store.add(entry)
        logger.info("audit_action", action=action, tenant_id=tenant_id, user_id=user_id)
        return entry

    def get_store(self) -> AuditLogStore:
        return self._store


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    global _audit_logger
    _audit_logger = logger
