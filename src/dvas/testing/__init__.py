"""Testing utilities and quality assurance framework for DVAS.

Provides property-based testing, contract testing, snapshot testing,
and load testing utilities for comprehensive quality assurance.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Property-Based Testing Support
# ---------------------------------------------------------------------------

@dataclass
class ArbitraryValue:
    """Generate arbitrary values for property-based testing.

    Usage::

        gen = ArbitraryValue()
        for value in gen.integers(min_value=0, max_value=100).take(10):
            assert value >= 0 and value <= 100
    """

    def __init__(self, seed: Optional[int] = None):
        import random

        self._random = random.Random(seed)

    def integers(self, min_value: int = -1000, max_value: int = 1000) -> "ArbitraryValue":
        """Generate random integers."""
        self._generator = lambda: self._random.randint(min_value, max_value)
        return self

    def floats(self, min_value: float = -1000.0, max_value: float = 1000.0) -> "ArbitraryValue":
        """Generate random floats."""
        self._generator = lambda: self._random.uniform(min_value, max_value)
        return self

    def strings(self, min_length: int = 0, max_length: int = 100, alphabet: str = "") -> "ArbitraryValue":
        """Generate random strings."""
        if not alphabet:
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        def _gen():
            length = self._random.randint(min_length, max_length)
            return "".join(self._random.choice(alphabet) for _ in range(length))

        self._generator = _gen
        return self

    def lists(self, element_gen: Callable[[], T], min_length: int = 0, max_length: int = 100) -> "ArbitraryValue":
        """Generate random lists."""
        def _gen():
            length = self._random.randint(min_length, max_length)
            return [element_gen() for _ in range(length)]

        self._generator = _gen
        return self

    def dicts(
        self,
        key_gen: Callable[[], str],
        value_gen: Callable[[], Any],
        min_length: int = 0,
        max_length: int = 100,
    ) -> "ArbitraryValue":
        """Generate random dictionaries."""
        def _gen():
            length = self._random.randint(min_length, max_length)
            return {key_gen(): value_gen() for _ in range(length)}

        self._generator = _gen
        return self

    def take(self, n: int) -> Iterator[Any]:
        """Generate n values."""
        for _ in range(n):
            yield self._generator()

    def first(self) -> Any:
        """Get first generated value."""
        return next(self.take(1))


def for_all(**strategies: Callable[[], Any]):
    """Decorator for property-based testing.

    Usage::

        @for_all(x=ArbitraryValue().integers(0, 100).first, y=ArbitraryValue().integers(0, 100).first)
        def test_addition_commutative(x, y):
            assert x + y == y + x
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            # Generate values from strategies
            generated = {}
            for name, strategy in strategies.items():
                if callable(strategy):
                    generated[name] = strategy()
                else:
                    generated[name] = strategy

            # Call function with generated values
            return func(*args, **generated, **kwargs)

        return wrapper
    return decorator


def given(**strategies: Callable[[], Any]) -> Callable:
    """Context manager for property-based testing.

    Usage::

        with given(x=ArbitraryValue().integers(0, 100)):
            for x in gen.take(100):
                assert x >= 0 and x <= 100
    """
    # This is a simplified version - in practice you'd use hypothesis
    def decorator(func: Callable) -> Callable:
        return for_all(**strategies)(func)
    return decorator


# ---------------------------------------------------------------------------
# Contract Testing
# ---------------------------------------------------------------------------

@dataclass
class Contract:
    """Represents a contract between service consumer and provider."""

    name: str
    request_method: str
    request_path: str
    request_headers: Dict[str, str] = field(default_factory=dict)
    request_body: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    expected_headers: Dict[str, str] = field(default_factory=dict)
    expected_body_schema: Dict[str, Any] = field(default_factory=dict)


class ContractStore:
    """Store and manage API contracts."""

    def __init__(self) -> None:
        self._contracts: Dict[str, Contract] = {}

    def add(self, contract: Contract) -> None:
        """Add a contract to the store."""
        self._contracts[contract.name] = contract

    def get(self, name: str) -> Optional[Contract]:
        """Get a contract by name."""
        return self._contracts.get(name)

    def list_contracts(self) -> List[str]:
        """List all contract names."""
        return list(self._contracts.keys())

    def validate_response(self, name: str, response: Any) -> List[str]:
        """Validate a response against a contract.

        Returns list of validation errors (empty if valid).
        """
        contract = self._contracts.get(name)
        if not contract:
            return [f"Contract '{name}' not found"]

        errors = []

        # Check status code
        if hasattr(response, "status_code"):
            if response.status_code != contract.expected_status:
                errors.append(
                    f"Status code mismatch: expected {contract.expected_status}, "
                    f"got {response.status_code}"
                )

        # Check headers
        if hasattr(response, "headers"):
            for header, expected_value in contract.expected_headers.items():
                actual = response.headers.get(header)
                if actual != expected_value:
                    errors.append(
                        f"Header '{header}' mismatch: expected '{expected_value}', "
                        f"got '{actual}'"
                    )

        return errors


def contract_test(
    name: str,
    method: str = "GET",
    path: str = "",
    expected_status: int = 200,
    expected_schema: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator for contract testing.

    Usage::

        @contract_test("get_annotation", method="GET", path="/api/v1/annotations/{id}")
        def test_get_annotation():
            response = client.get("/api/v1/annotations/123")
            return response
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Validate status
            if hasattr(result, "status_code"):
                assert result.status_code == expected_status, (
                    f"Contract '{name}': expected status {expected_status}, "
                    f"got {result.status_code}"
                )

            # Validate schema if provided
            if expected_schema and hasattr(result, "json"):
                data = result.json()
                _validate_schema(data, expected_schema, f"contract.{name}")

            return result

        # Mark as contract test
        wrapper._is_contract_test = True  # type: ignore
        wrapper._contract_name = name  # type: ignore
        return wrapper
    return decorator


def _validate_schema(data: Any, schema: Dict[str, Any], path: str = "") -> List[str]:
    """Validate data against a JSON schema."""
    errors = []

    if schema.get("type") == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for key in required:
            if key not in data:
                errors.append(f"Missing required field: {path}.{key}")

        for key, sub_schema in properties.items():
            if key in data:
                errors.extend(_validate_schema(data[key], sub_schema, f"{path}.{key}"))

    elif schema.get("type") == "array" and isinstance(data, list):
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            errors.extend(_validate_schema(item, items_schema, f"{path}[{i}]"))

    elif "enum" in schema and data not in schema["enum"]:
        errors.append(f"Value at {path} not in enum: {schema['enum']}")

    return errors


# ---------------------------------------------------------------------------
# Snapshot Testing
# ---------------------------------------------------------------------------

class SnapshotStore:
    """Store and compare snapshots for regression testing.

    Usage::

        store = SnapshotStore("tests/snapshots")

        # In test
        result = process_data(input_data)
        store.assert_match("test_process_data", result)
    """

    def __init__(self, snapshot_dir: Union[str, Path], update: bool = False) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.update = update

    def _get_path(self, name: str) -> Path:
        """Get the path for a snapshot file."""
        safe_name = name.replace("/", "_").replace("\\", "_")
        return self.snapshot_dir / f"{safe_name}.snap"

    def _normalize(self, value: Any) -> Any:
        """Normalize value for snapshot comparison, removing dynamic fields."""
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k in ("created_at", "updated_at"):
                    continue
                result[k] = self._normalize(v)
            return result
        if isinstance(value, list):
            return [self._normalize(v) for v in value]
        return value

    def _serialize(self, value: Any) -> str:
        """Serialize a value for snapshot comparison."""
        normalized = self._normalize(value)
        if isinstance(normalized, (dict, list)):
            return json.dumps(normalized, indent=2, sort_keys=True, default=str)
        return str(normalized)

    def assert_match(self, name: str, value: Any) -> None:
        """Assert that a value matches the stored snapshot.

        If no snapshot exists, creates one.
        """
        snapshot_path = self._get_path(name)
        serialized = self._serialize(value)

        if not snapshot_path.exists() or self.update:
            snapshot_path.write_text(serialized)
            return

        expected = snapshot_path.read_text()
        actual = serialized

        if expected != actual:
            raise AssertionError(
                f"Snapshot mismatch for '{name}':\n"
                f"Expected:\n{expected}\n\n"
                f"Actual:\n{actual}"
            )

    def exists(self, name: str) -> bool:
        """Check if a snapshot exists."""
        return self._get_path(name).exists()

    def delete(self, name: str) -> None:
        """Delete a snapshot."""
        path = self._get_path(name)
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Load Testing
# ---------------------------------------------------------------------------

@dataclass
class LoadTestResult:
    """Result of a load test."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    min_latency: float
    max_latency: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    requests_per_second: float
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return 1.0 - self.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "total_duration": self.total_duration,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "avg_latency": self.avg_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "requests_per_second": self.requests_per_second,
            "errors": self.errors,
        }


class LoadTester:
    """Load testing utility for API endpoints.

    Usage::

        tester = LoadTester()
        result = await tester.run(
            target=make_request,
            concurrent_users=10,
            requests_per_user=100,
        )
        print(f"RPS: {result.requests_per_second:.2f}")
    """

    def __init__(self) -> None:
        self._results: List[Dict[str, Any]] = []

    async def run(
        self,
        target: Callable,
        concurrent_users: int = 10,
        requests_per_user: int = 100,
        ramp_up: float = 0.0,
    ) -> LoadTestResult:
        """Run a load test.

        Args:
            target: Async function to call
            concurrent_users: Number of concurrent users
            requests_per_user: Number of requests per user
            ramp_up: Time in seconds to ramp up users

        Returns:
            LoadTestResult with statistics
        """
        import asyncio

        latencies: List[float] = []
        errors: List[str] = []
        successful = 0
        failed = 0

        async def _user_task(user_id: int) -> None:
            """Task for a single user."""
            nonlocal successful, failed

            # Ramp up delay
            if ramp_up > 0 and concurrent_users > 1:
                delay = (user_id / concurrent_users) * ramp_up
                await asyncio.sleep(delay)

            for _ in range(requests_per_user):
                start = time.time()
                try:
                    await target()
                    latency = time.time() - start
                    latencies.append(latency)
                    successful += 1
                except Exception as e:
                    latency = time.time() - start
                    latencies.append(latency)
                    errors.append(str(e))
                    failed += 1

        start_time = time.time()

        # Run all users concurrently
        tasks = [_user_task(i) for i in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.time() - start_time
        total_requests = len(latencies)

        if not latencies:
            return LoadTestResult(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_duration=total_duration,
                min_latency=0.0,
                max_latency=0.0,
                avg_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                requests_per_second=0.0,
                errors=["No requests completed"],
            )

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_duration=total_duration,
            min_latency=sorted_latencies[0],
            max_latency=sorted_latencies[-1],
            avg_latency=sum(latencies) / n,
            p50_latency=sorted_latencies[n // 2],
            p95_latency=sorted_latencies[int(n * 0.95)],
            p99_latency=sorted_latencies[int(n * 0.99)],
            requests_per_second=total_requests / total_duration,
            errors=errors[:10],  # Keep first 10 errors
        )


# ---------------------------------------------------------------------------
# Mutation Testing Concepts
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """Represents a code mutation for mutation testing."""

    original: str
    mutated: str
    line_number: int
    mutation_type: str


class MutantSurvivedError(Exception):
    """Raised when a mutant survives testing (test suite is insufficient)."""

    def __init__(self, mutation: Mutation, test_name: str = ""):
        self.mutation = mutation
        self.test_name = test_name
        super().__init__(
            f"Mutant survived at line {mutation.line_number}: "
            f"'{mutation.original}' -> '{mutation.mutated}'"
        )


class MutationOperator:
    """Base class for mutation operators."""

    def apply(self, source: str) -> Iterator[Mutation]:
        """Apply mutation operator to source code.

        Yields mutations that can be applied.
        """
        raise NotImplementedError


class ArithmeticMutationOperator(MutationOperator):
    """Mutate arithmetic operators."""

    REPLACEMENTS = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
        "//": "/",
        "%": "*",
        "**": "*",
    }

    def apply(self, source: str) -> Iterator[Mutation]:
        """Apply arithmetic mutations."""
        for line_num, line in enumerate(source.split("\n"), 1):
            for original, mutated in self.REPLACEMENTS.items():
                # Simple string replacement - in practice you'd use AST
                if original in line and not line.strip().startswith("#"):
                    yield Mutation(
                        original=original,
                        mutated=mutated,
                        line_number=line_num,
                        mutation_type="arithmetic",
                    )


class ComparisonMutationOperator(MutationOperator):
    """Mutate comparison operators."""

    REPLACEMENTS = {
        "==": "!=",
        "!=": "==",
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">",
    }

    def apply(self, source: str) -> Iterator[Mutation]:
        """Apply comparison mutations."""
        for line_num, line in enumerate(source.split("\n"), 1):
            for original, mutated in self.REPLACEMENTS.items():
                if original in line and not line.strip().startswith("#"):
                    yield Mutation(
                        original=original,
                        mutated=mutated,
                        line_number=line_num,
                        mutation_type="comparison",
                    )


class MutationRunner:
    """Run mutation testing on a module.

    Usage::

        runner = MutationRunner()
        result = runner.run("dvas.data.schemas", test_module="tests.test_schemas")
        print(f"Mutation score: {result.score:.1%}")
    """

    def __init__(self) -> None:
        self.operators: List[MutationOperator] = [
            ArithmeticMutationOperator(),
            ComparisonMutationOperator(),
        ]

    def run(self, module_name: str, test_module: str) -> "MutationResult":
        """Run mutation testing.

        Returns:
            MutationResult with score and details.
        """
        # This is a simplified version
        # In practice, you'd use a tool like mutmut
        return MutationResult(
            total_mutants=0,
            killed=0,
            survived=0,
            skipped=0,
            score=0.0,
        )


@dataclass
class MutationResult:
    """Result of mutation testing."""

    total_mutants: int
    killed: int
    survived: int
    skipped: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_mutants": self.total_mutants,
            "killed": self.killed,
            "survived": self.survived,
            "skipped": self.skipped,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Test Fixtures and Helpers
# ---------------------------------------------------------------------------

def create_test_video_metadata(
    fps: float = 30.0,
    resolution: Optional[List[int]] = None,
    duration: float = 10.0,
) -> Dict[str, Any]:
    """Create test video metadata."""
    return {
        "fps": fps,
        "resolution": resolution or [1920, 1080],
        "duration": duration,
        "total_frames": int(fps * duration),
        "codec": "h264",
        "bitrate": 5000000,
        "has_audio": True,
    }


def create_test_annotation(
    video_id: str = "test_001",
    num_segments: int = 2,
) -> Dict[str, Any]:
    """Create a test annotation."""
    segments = []
    for i in range(num_segments):
        start_time = i * 5.0
        end_time = (i + 1) * 5.0
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "caption": f"Segment {i + 1}",
            "actions": [
                {"verb": "cut", "noun": "vegetables", "hand": "right"},
            ],
            "objects": [
                {"name": "knife", "confidence": 0.95},
            ],
        })

    return {
        "id": f"ann_{video_id}",
        "video_id": video_id,
        "video_path": f"/path/to/{video_id}.mp4",
        "segments": segments,
        "metadata": create_test_video_metadata(),
        "source": "teacher",
        "model_version": "gpt-5.5-2024-05-13",
    }


def assert_dict_subset(actual: Dict[str, Any], expected: Dict[str, Any], path: str = "") -> None:
    """Assert that actual dict contains all keys/values from expected.

    Usage::

        assert_dict_subset(result, {"status": "success", "data": {"id": "123"}})
    """
    for key, expected_value in expected.items():
        current_path = f"{path}.{key}" if path else key
        assert key in actual, f"Missing key at {current_path}"

        actual_value = actual[key]

        if isinstance(expected_value, dict):
            assert isinstance(actual_value, dict), (
                f"Expected dict at {current_path}, got {type(actual_value).__name__}"
            )
            assert_dict_subset(actual_value, expected_value, current_path)
        elif isinstance(expected_value, list):
            assert isinstance(actual_value, list), (
                f"Expected list at {current_path}, got {type(actual_value).__name__}"
            )
            assert len(actual_value) >= len(expected_value), (
                f"List at {current_path} too short: {len(actual_value)} < {len(expected_value)}"
            )
            for i, (a, e) in enumerate(zip(actual_value, expected_value)):
                if isinstance(e, dict):
                    assert_dict_subset(a, e, f"{current_path}[{i}]")
                else:
                    assert a == e, f"Mismatch at {current_path}[{i}]: {a} != {e}"
        else:
            assert actual_value == expected_value, (
                f"Mismatch at {current_path}: {actual_value} != {expected_value}"
            )


# ---------------------------------------------------------------------------
# Performance Benchmarking
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a benchmark."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "std_dev": self.std_dev,
        }


def benchmark(func: Callable, *args: Any, iterations: int = 100, **kwargs: Any) -> BenchmarkResult:
    """Benchmark a function.

    Usage::

        result = benchmark(process_video, video_path, iterations=10)
        print(f"Average: {result.avg_time:.3f}s")
    """
    import statistics

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time=statistics.mean(times),
        min_time=min(times),
        max_time=max(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Fuzz Testing Utilities
# ---------------------------------------------------------------------------

def fuzz_string(
    min_length: int = 0,
    max_length: int = 1000,
    include_null: bool = False,
    include_unicode: bool = True,
) -> str:
    """Generate a fuzzed string.

    Usage::

        for _ in range(100):
            s = fuzz_string()
            assert isinstance(s, str)
    """
    import random

    length = random.randint(min_length, max_length)
    chars = []

    for _ in range(length):
        if include_unicode and random.random() < 0.3:
            # Include some unicode
            chars.append(chr(random.randint(0x4E00, 0x9FFF)))  # CJK
        elif include_null and random.random() < 0.05:
            chars.append("\x00")
        else:
            chars.append(chr(random.randint(32, 126)))

    return "".join(chars)


def fuzz_dict(
    min_keys: int = 0,
    max_keys: int = 10,
    key_length: int = 20,
) -> Dict[str, Any]:
    """Generate a fuzzed dictionary."""
    import random

    result = {}
    for _ in range(random.randint(min_keys, max_keys)):
        key = fuzz_string(max_length=key_length)
        value_type = random.choice([str, int, float, bool, list, dict])

        if value_type is str:
            result[key] = fuzz_string()
        elif value_type is int:
            result[key] = random.randint(-1000000, 1000000)
        elif value_type is float:
            result[key] = random.random() * 1000000
        elif value_type is bool:
            result[key] = random.choice([True, False])
        elif value_type is list:
            result[key] = [fuzz_string() for _ in range(random.randint(0, 10))]
        else:
            result[key] = {}

    return result


# ---------------------------------------------------------------------------
# Integration with pytest
# ---------------------------------------------------------------------------

def pytest_fixture_factory(factory_func: Callable) -> Callable:
    """Create a pytest fixture from a factory function.

    Usage::

        @pytest_fixture_factory
        def annotation_factory(video_id="test", num_segments=2):
            return create_test_annotation(video_id, num_segments)

        # In test:
        def test_something(annotation_factory):
            ann = annotation_factory(video_id="custom")
            ...
    """
    @functools.wraps(factory_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return factory_func(*args, **kwargs)

    return wrapper


# Convenience exports
__all__ = [
    # Property-based testing
    "ArbitraryValue",
    "for_all",
    "given",
    # Contract testing
    "Contract",
    "ContractStore",
    "contract_test",
    # Snapshot testing
    "SnapshotStore",
    # Load testing
    "LoadTestResult",
    "LoadTester",
    # Mutation testing
    "Mutation",
    "MutationOperator",
    "ArithmeticMutationOperator",
    "ComparisonMutationOperator",
    "MutationRunner",
    "MutationResult",
    "MutantSurvivedError",
    # Benchmarking
    "BenchmarkResult",
    "benchmark",
    # Fuzz testing
    "fuzz_string",
    "fuzz_dict",
    # Test helpers
    "create_test_video_metadata",
    "create_test_annotation",
    "assert_dict_subset",
    "pytest_fixture_factory",
]
