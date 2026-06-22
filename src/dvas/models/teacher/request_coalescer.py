"""Request coalescing for identical/similar API calls.

Merges duplicate or similar requests to avoid redundant API calls.
Useful when multiple pipeline stages request the same video annotation.

Usage::

    from dvas.models.teacher.request_coalescer import RequestCoalescer

    coalescer = RequestCoalescer(similarity_threshold=0.95)

    # Multiple identical calls will be merged
    result = await coalescer.execute(
        video_hash="abc123",
        prompt="Describe the action",
        params={"temperature": 0.2},
        api_caller=call_api
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PendingRequest:
    """Represents a pending coalesced request."""

    key: str
    future: asyncio.Future
    timestamp: float
    request_count: int = field(default=1)


class RequestCoalescer:
    """合并相同/相似请求，避免重复API调用.

    当多个并发请求具有相同的视频hash和提示词时，
    只发送一个API调用，所有请求者共享结果。

    Attributes:
        similarity_threshold: 相似度阈值(未使用，保留扩展)
        _pending: 当前正在处理的请求字典
        _lock: 协程安全锁
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """初始化请求合并器.

        Args:
            similarity_threshold: 相似度阈值(当前仅支持精确匹配)
        """
        self._similarity_threshold = similarity_threshold
        self._pending: Dict[str, PendingRequest] = {}
        self._lock = asyncio.Lock()
        self._stats = {"coalesced": 0, "total": 0}

    def _compute_key(
        self,
        video_hash: str,
        prompt: str,
        model: str,
        params: Optional[dict] = None,
    ) -> str:
        """计算请求指纹.

        Args:
            video_hash: 视频内容的哈希值
            prompt: 提示词
            model: 模型名称
            params: 额外参数

        Returns:
            16字符的请求唯一标识
        """
        # 排序参数以确保一致性
        params_str = ""
        if params:
            params_str = str(sorted(params.items()))

        content = f"{video_hash}:{model}:{prompt}:{params_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def execute(
        self,
        video_hash: str,
        prompt: str,
        model: str,
        params: Optional[dict] = None,
        api_caller: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """执行请求，如果相同请求正在进行则复用结果.

        Args:
            video_hash: 视频内容哈希
            prompt: 提示词
            model: 模型名称
            params: API参数
            api_caller: 实际的API调用函数

        Returns:
            API调用结果

        Raises:
            Exception: API调用失败时抛出
        """
        if api_caller is None:
            raise ValueError("api_caller is required")

        key = self._compute_key(video_hash, prompt, model, params)

        async with self._lock:
            self._stats["total"] += 1

            if key in self._pending:
                # 等待已有请求完成
                pending = self._pending[key]
                pending.request_count += 1
                self._stats["coalesced"] += 1

                logger.debug(
                    "Coalescing request",
                    key=key,
                    total_requests=pending.request_count,
                    coalesced_count=self._stats["coalesced"],
                )

                # 释放锁后等待结果
                future = pending.future
            else:
                # 创建新请求
                future = asyncio.get_event_loop().create_future()
                self._pending[key] = PendingRequest(
                    key=key,
                    future=future,
                    timestamp=time.time(),
                )
                future = None  # 标记为新请求

        if future is not None:
            # 这是被合并的请求，等待结果
            try:
                result = await self._pending[key].future
                return result
            except Exception as e:
                logger.debug("Coalesced request failed", key=key, error=str(e))
                raise

        # 执行实际API调用
        try:
            logger.debug("Executing API call", key=key)
            result = await api_caller()

            # 设置结果，唤醒所有等待的请求
            async with self._lock:
                if key in self._pending:
                    self._pending[key].future.set_result(result)
                    del self._pending[key]

            return result

        except Exception as e:
            # 设置异常，传播给所有等待的请求
            async with self._lock:
                if key in self._pending:
                    self._pending[key].future.set_exception(e)
                    del self._pending[key]
            raise

    def get_stats(self) -> Dict[str, Any]:
        """获取合并统计信息.

        Returns:
            包含合并统计的字典
        """
        total = self._stats["total"]
        coalesced = self._stats["coalesced"]
        return {
            "total_requests": total,
            "coalesced_requests": coalesced,
            "savings_rate": coalesced / total if total > 0 else 0.0,
            "current_pending": len(self._pending),
        }

    def reset_stats(self) -> None:
        """重置统计数据."""
        self._stats = {"coalesced": 0, "total": 0}


class PerceptualHashCoalescer(RequestCoalescer):
    """使用感知哈希的相似请求合并器.

    可以合并视觉上相似但不完全相同的视频请求。
    适用于有轻微差异(压缩、裁剪)的相同视频。
    """

    def __init__(self, similarity_threshold: float = 0.9):
        """初始化感知哈希合并器.

        Args:
            similarity_threshold: 哈希相似度阈值(0-1)
        """
        super().__init__(similarity_threshold)
        self._similarity_threshold = similarity_threshold
        self._hash_cache: Dict[str, str] = {}

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希的汉明距离."""
        if len(hash1) != len(hash2):
            return float("inf")  # type: ignore

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _find_similar_key(self, video_hash: str) -> Optional[str]:
        """在pending中查找相似的key."""
        if not self._pending:
            return None

        # 简化实现：直接哈希匹配
        # 实际生产环境可以实现pHash或dHash
        return None  # 暂时禁用模糊匹配

    async def execute(
        self,
        video_hash: str,
        prompt: str,
        model: str,
        params: Optional[dict] = None,
        api_caller: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """执行请求，支持模糊匹配."""
        # 暂时使用精确匹配
        return await super().execute(video_hash, prompt, model, params, api_caller)
