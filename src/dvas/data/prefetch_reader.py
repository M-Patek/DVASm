"""Prefetch video reader with background frame loading.

Provides PrefetchVideoReader that wraps any VideoReader with a
background prefetch queue for improved I/O performance.

Usage::

    from dvas.data.prefetch_reader import PrefetchVideoReader
    from dvas.data.video_reader import VideoReader

    # Wrap any reader with prefetching
    base_reader = VideoReader("video.mp4")
    reader = PrefetchVideoReader(base_reader, prefetch_size=32)

    # Read frames with background prefetching
    for frame in reader.read_frames():
        process(frame)
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator, List, Optional

from dvas.data.video_reader import Frame, VideoReader
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class PrefetchVideoReader:
    """装饰器模式包装任意VideoReader，添加预取队列。

    使用后台工作线程提前加载帧到内存队列，减少I/O等待时间。
    特别适用于慢速存储(网络存储、HDD)上的视频处理。

    Attributes:
        _reader: 被包装的VideoReader实例
        _prefetch_size: 预取队列大小
        _num_workers: 后台预取工作线程数
    """

    def __init__(
        self,
        reader: VideoReader,
        prefetch_size: int = 32,
        num_workers: int = 2,
    ):
        """初始化预取阅读器。

        Args:
            reader: 要包装的VideoReader实例
            prefetch_size: 预取队列大小(默认32)
            num_workers: 后台预取工作线程数(默认2)
        """
        self._reader = reader
        self._prefetch_size = prefetch_size
        self._num_workers = num_workers
        self._queue: queue.Queue[Optional[Frame]] = queue.Queue(maxsize=prefetch_size)
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._iterator: Optional[Iterator[Frame]] = None
        self._video_path = getattr(reader, "video_path", "unknown")

    def __enter__(self) -> "PrefetchVideoReader":
        self._reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cleanup()
        self._reader.__exit__(exc_type, exc_val, exc_tb)

    def _cleanup(self) -> None:
        """清理后台工作线程和队列。"""
        self._stop_event.set()

        # 清空队列，解除工作线程阻塞
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # 等待工作线程结束
        for t in self._workers:
            t.join(timeout=1.0)

        self._workers.clear()
        logger.debug(
            "Prefetch reader cleaned up",
            video_path=str(self._video_path),
        )

    def _worker_loop(self) -> None:
        """后台工作线程持续填充队列。"""
        try:
            while not self._stop_event.is_set():
                try:
                    frame = next(self._iterator)
                    # 使用超时以便定期检查停止事件
                    while not self._stop_event.is_set():
                        try:
                            self._queue.put(frame, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                except StopIteration:
                    # 发送sentinel表示此线程结束
                    self._queue.put(None)
                    break
        except Exception as e:
            logger.debug(
                "Prefetch worker encountered error",
                error=str(e),
                video_path=str(self._video_path),
            )
            self._queue.put(None)

    def read_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> Iterator[Frame]:
        """生成帧，从预取队列消费。

        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引(不包含)
            step: 帧步长

        Yields:
            Frame对象
        """
        # 重置状态
        self._stop_event.clear()
        self._workers = []

        # 创建同步迭代器
        self._iterator = self._reader.read_frames(start_frame, end_frame, step)

        logger.debug(
            "Starting prefetch workers",
            num_workers=self._num_workers,
            queue_size=self._prefetch_size,
            video_path=str(self._video_path),
        )

        # 启动后台工作线程
        for _ in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

        # 从队列消费帧
        finished_workers = 0
        try:
            while finished_workers < self._num_workers:
                frame = self._queue.get()
                if frame is None:
                    finished_workers += 1
                else:
                    yield frame
        finally:
            self._cleanup()

    def get_batch(self, indices: List[int]) -> List[Frame]:
        """批量读取帧(直接委托给底层reader)。

        对于随机访问批量读取，预取没有优势，直接委托。

        Args:
            indices: 帧索引列表

        Returns:
            Frame对象列表
        """
        return self._reader.get_batch(indices)

    def get_frame(self, index: int) -> Optional[Frame]:
        """读取单帧(直接委托给底层reader)。

        Args:
            index: 帧索引

        Returns:
            Frame对象或None
        """
        return self._reader.get_frame(index)

    @property
    def metadata(self):
        """获取视频元数据(委托给底层reader)。"""
        return self._reader.metadata


class AsyncPrefetchVideoReader(PrefetchVideoReader):
    """异步版本的预取阅读器，支持async for迭代。"""

    import asyncio

    async def read_frames_async(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ):
        """异步生成帧。

        在后台线程执行同步read_frames，通过asyncio异步 yielding。

        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引(不包含)
            step: 帧步长

        Yields:
            Frame对象
        """
        import asyncio

        loop = asyncio.get_event_loop()
        iterator = self.read_frames(start_frame, end_frame, step)

        def _next_frame():
            try:
                return next(iterator)
            except StopIteration:
                return None

        while True:
            frame = await loop.run_in_executor(None, _next_frame)
            if frame is None:
                break
            yield frame
