"""Concurrency primitives and utilities for DVAS.

Provides thread pool management, work-stealing, async-to-sync bridges,
and CPU-bound task offloading with proper backpressure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

import numpy as np

from dvas.core.object_pool import PoolRegistry
from dvas.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Work-Stealing Thread Pool
# ---------------------------------------------------------------------------


class WorkStealingPool:
    """Thread pool with work-stealing for CPU-bound tasks.

    Distributes tasks across multiple worker threads, allowing idle
    workers to steal tasks from busy workers' queues.

    Usage::

        pool = WorkStealingPool(max_workers=8)
        future = pool.submit(expensive_function, arg1, arg2)
        result = future.result()
    """

    def __init__(self, max_workers: int = 8, thread_name_prefix: str = "worker") -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._task_count = 0
        self._completed_count = 0

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazy-initialize the thread pool executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix=self.thread_name_prefix,
                    )
        return self._executor

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future:
        """Submit a function to be executed in the thread pool."""
        executor = self._ensure_executor()
        self._task_count += 1

        def _wrapper(*a: Any, **kw: Any) -> T:
            try:
                return fn(*a, **kw)
            finally:
                self._completed_count += 1

        return executor.submit(_wrapper, *args, **kwargs)

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Any,
        timeout: Optional[float] = None,
    ) -> Iterator[T]:
        """Map a function over iterables using the thread pool."""
        executor = self._ensure_executor()
        return executor.map(fn, *iterables, timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> WorkStealingPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "max_workers": self.max_workers,
            "submitted": self._task_count,
            "completed": self._completed_count,
            "pending": self._task_count - self._completed_count,
        }


# ---------------------------------------------------------------------------
# Async-to-Sync Bridge for Iterators
# ---------------------------------------------------------------------------


class AsyncIteratorBridge(Generic[T]):
    """Bridge that wraps a sync iterator for async consumption.

    Uses a background thread to read from the sync iterator and
    an asyncio queue to communicate with the async consumer.
    Properly handles cleanup and avoids the asyncio.run_coroutine_threadsafe
    anti-pattern by using a simple queue with blocking put/get.

    Usage::

        bridge = AsyncIteratorBridge(sync_iter)
        async for item in bridge:
            await process(item)
    """

    def __init__(
        self,
        iterator: Iterator[T],
        *,
        queue_size: int = 16,
        sentinel: Any = None,
    ) -> None:
        self._iterator = iterator
        self._queue_size = queue_size
        self._sentinel = sentinel if sentinel is not None else object()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._started = False
        self._stopped = False
        self._error: Optional[Exception] = None

    def _run_producer(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        """Producer thread: reads from sync iterator, puts to async queue."""
        sentinel_sent = False
        try:
            for item in self._iterator:
                # Use asyncio.run_coroutine_threadsafe for thread-safe queue put
                future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                try:
                    future.result(timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("async_iterator_timeout", queue_size=queue.qsize())
                    self._error = TimeoutError("Producer timeout waiting for queue space")
                    break

            # Signal completion
            try:
                asyncio.run_coroutine_threadsafe(queue.put(self._sentinel), loop).result(
                    timeout=5.0
                )
                sentinel_sent = True
            except Exception as e:
                logger.error("failed_to_send_sentinel", error=str(e))
        except Exception as e:
            self._error = e
            logger.error("async_iterator_producer_error", error=str(e))
        finally:
            if not sentinel_sent:
                # Last resort: try to send sentinel even after error
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(self._sentinel), loop).result(
                        timeout=1.0
                    )
                except Exception:
                    pass  # Nothing more we can do

    async def _ensure_started(self) -> None:
        """Start the producer thread if not already started."""
        if self._started:
            return

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        self._started = True

        self._thread = threading.Thread(
            target=self._run_producer,
            args=(self._queue, self._loop),
            daemon=True,
            name="AsyncIteratorBridge-producer",
        )
        self._thread.start()

    def __aiter__(self) -> AsyncIteratorBridge[T]:
        return self

    async def __anext__(self) -> T:
        await self._ensure_started()
        assert self._queue is not None

        # Use timeout to prevent indefinite waiting
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            self._stopped = True
            logger.error("async_iterator_consumer_timeout", queue_size=self._queue.qsize())
            raise TimeoutError("AsyncIteratorBridge consumer timeout - producer may be dead")

        if item is self._sentinel:
            self._stopped = True
            if self._error:
                raise self._error
            raise StopAsyncIteration

        return item

    async def close(self) -> None:
        """Close the bridge and clean up resources."""
        self._stopped = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("async_iterator_thread_did_not_exit")

    async def __aenter__(self) -> AsyncIteratorBridge[T]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Process Pool for CPU-Bound Tasks
# ---------------------------------------------------------------------------


class ProcessPoolWrapper:
    """Wrapper around ProcessPoolExecutor with proper lifecycle management.

    Usage::

        pool = ProcessPoolWrapper(max_workers=4)
        result = await pool.run(cpu_intensive_function, data)
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        self._executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._task_count = 0

    def _ensure_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        """Lazy-initialize the process pool executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=self.max_workers,
                    )
        return self._executor

    async def run(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a function in the process pool."""

        executor = self._ensure_executor()
        self._task_count += 1

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: fn(*args, **kwargs),
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self) -> ProcessPoolWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# Semaphore-based concurrency limiter with queue
# ---------------------------------------------------------------------------


class ConcurrencyLimiter:
    """Limit concurrent operations with a queue for backpressure.

    Usage::

        limiter = ConcurrencyLimiter(max_concurrent=10)
        async with limiter:
            await process_item(item)
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0
        self._total_acquired = 0
        self._total_released = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a slot. Returns True when acquired."""
        await self._semaphore.acquire()
        async with self._lock:
            self._active += 1
            self._total_acquired += 1
        return True

    def release(self) -> None:
        """Release a slot."""
        self._semaphore.release()
        self._active -= 1
        self._total_released += 1

    async def __aenter__(self) -> ConcurrencyLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.release()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
        }


# ---------------------------------------------------------------------------
# Batch processor with async support
# ---------------------------------------------------------------------------


class AsyncBatchProcessor(Generic[T]):
    """Process items in batches with controlled concurrency.

    Usage::

        processor = AsyncBatchProcessor(max_concurrent=5)
        results = await processor.process(items, process_fn)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: int = 100,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processed = 0
        self._failed = 0

    async def process(
        self,
        items: List[T],
        process_fn: Callable[[T], Coroutine[Any, Any, Any]],
        *,
        on_error: Optional[Callable[[T, Exception], None]] = None,
    ) -> List[Any]:
        """Process items with controlled concurrency.

        Args:
            items: Items to process
            process_fn: Async function to process each item
            on_error: Optional callback for errors

        Returns:
            List of successful results (None for failed items)
        """
        results: List[Any] = [None] * len(items)

        async def _process_one(index: int, item: T) -> None:
            async with self._semaphore:
                try:
                    results[index] = await process_fn(item)
                    self._processed += 1
                except Exception as e:
                    self._failed += 1
                    if on_error:
                        on_error(item, e)
                    logger.error("batch_process_error", index=index, error=str(e))

        await asyncio.gather(*(_process_one(i, item) for i, item in enumerate(items)))
        return results

    async def process_with_results(
        self,
        items: List[T],
        process_fn: Callable[[T], Coroutine[Any, Any, Any]],
        *,
        on_error: Optional[Callable[[T, Exception], None]] = None,
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        """Process items and return both successes and failures.

        Returns:
            Tuple of (successful_results, failed_items)
        """
        successes: List[Any] = []
        failures: List[Dict[str, Any]] = []

        async def _process_one(item: T) -> None:
            async with self._semaphore:
                try:
                    result = await process_fn(item)
                    successes.append(result)
                    self._processed += 1
                except Exception as e:
                    self._failed += 1
                    failures.append({"item": item, "error": str(e)})
                    if on_error:
                        on_error(item, e)

        await asyncio.gather(*(_process_one(item) for item in items))
        return successes, failures


# ---------------------------------------------------------------------------
# Frame encoding pool (specialized for video frame processing)
# ---------------------------------------------------------------------------


class FrameEncoderPool:
    """Pool for parallel frame encoding operations.

    Offloads CPU-intensive frame encoding (BGR->RGB optional, PIL conversion,
    base64 encoding) to a thread pool.

    Usage::

        encoder = FrameEncoderPool(max_workers=4)
        # If frames are BGR (OpenCV default):
        encoded = await encoder.encode_frames(frames, convert_bgr_to_rgb=True)
        # If frames are already RGB:
        encoded = await encoder.encode_frames(frames, convert_bgr_to_rgb=False)
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._pool = WorkStealingPool(max_workers=max_workers, thread_name_prefix="encoder")
        self._encoded_count = 0
        self._total_bytes = 0

    def _encode_single(
        self, frame: np.ndarray, format: str = "JPEG", convert_bgr_to_rgb: bool = True
    ) -> str:
        """Encode a single frame to base64."""
        import base64
        import io

        from PIL import Image

        # Conditionally convert BGR to RGB
        if convert_bgr_to_rgb and len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = frame[:, :, ::-1]
        else:
            frame_rgb = frame

        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    async def encode_frames(
        self,
        frames: List[np.ndarray],
        format: str = "JPEG",
        convert_bgr_to_rgb: bool = True,
    ) -> List[str]:
        """Encode multiple frames in parallel.

        Args:
            frames: List of numpy arrays (BGR or RGB images)
            format: Image format (JPEG, PNG)
            convert_bgr_to_rgb: Whether to convert BGR to RGB (default True for OpenCV frames)

        Returns:
            List of base64-encoded strings
        """
        loop = asyncio.get_running_loop()

        # For small batches, encode synchronously
        if len(frames) <= 2:
            results = [self._encode_single(f, format, convert_bgr_to_rgb) for f in frames]
            self._encoded_count += len(results)
            return results

        # For larger batches, use thread pool
        tasks = [
            loop.run_in_executor(
                None,  # Uses default executor
                self._encode_single,
                frame,
                format,
                convert_bgr_to_rgb,
            )
            for frame in frames
        ]

        results = await asyncio.gather(*tasks)
        self._encoded_count += len(results)
        return list(results)

    def shutdown(self) -> None:
        """Shutdown the encoder pool."""
        self._pool.shutdown()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "encoded_count": self._encoded_count,
            "max_workers": self.max_workers,
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop, creating one if necessary."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


async def run_in_thread(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a synchronous function in a thread pool.

    Modern replacement for direct threading.Thread usage.
    Uses asyncio.to_thread when available (Python 3.9+).
    """
    try:
        # Python 3.9+
        return await asyncio.to_thread(fn, *args, **kwargs)
    except AttributeError:
        # Fallback for older Python
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def offload_to_thread(
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future:
    """Offload a function to a background thread and return a Future."""
    pool = PoolRegistry.get_or_create_thread_pool("default", max_workers=8)
    return pool.submit(fn, *args, **kwargs)
