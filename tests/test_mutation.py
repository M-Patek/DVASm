"""Mutation testing concepts for DVAS.

Tests that verify mutation testing utilities work correctly.
"""

import pytest
from dvas.testing import (
    ArithmeticMutationOperator,
    ComparisonMutationOperator,
    Mutation,
    MutationResult,
    MutationRunner,
    MutantSurvivedError,
)


class TestArithmeticMutationOperator:
    """Test arithmetic mutation operator."""

    def test_plus_to_minus(self):
        """Test + is mutated to -."""
        source = "def add(a, b):\n    return a + b"
        operator = ArithmeticMutationOperator()
        mutations = list(operator.apply(source))

        assert len(mutations) > 0
        plus_mutation = [m for m in mutations if m.original == "+"]
        assert len(plus_mutation) == 1
        assert plus_mutation[0].mutated == "-"
        assert plus_mutation[0].mutation_type == "arithmetic"

    def test_minus_to_plus(self):
        """Test - is mutated to +."""
        source = "def subtract(a, b):\n    return a - b"
        operator = ArithmeticMutationOperator()
        mutations = list(operator.apply(source))

        minus_mutation = [m for m in mutations if m.original == "-"]
        assert len(minus_mutation) == 1
        assert minus_mutation[0].mutated == "+"

    def test_multiply_to_divide(self):
        """Test * is mutated to /."""
        source = "def multiply(a, b):\n    return a * b"
        operator = ArithmeticMutationOperator()
        mutations = list(operator.apply(source))

        mult_mutation = [m for m in mutations if m.original == "*"]
        assert len(mult_mutation) == 1
        assert mult_mutation[0].mutated == "/"

    def test_no_mutations_in_comments(self):
        """Test that comments are not mutated."""
        source = "# This is a + b\ndef calc(a, b):\n    return a + b"
        operator = ArithmeticMutationOperator()
        mutations = list(operator.apply(source))

        # Should only find mutation in code, not comment
        plus_mutations = [m for m in mutations if m.original == "+"]
        assert len(plus_mutations) == 1  # Only in the return statement

    def test_line_numbers(self):
        """Test that line numbers are correct."""
        source = "def calc(a, b):\n    return a + b\n\ndef calc2(a, b):\n    return a - b"
        operator = ArithmeticMutationOperator()
        mutations = list(operator.apply(source))

        assert len(mutations) == 2
        line_numbers = [m.line_number for m in mutations]
        assert 2 in line_numbers  # First function
        assert 5 in line_numbers  # Second function


class TestComparisonMutationOperator:
    """Test comparison mutation operator."""

    def test_equal_to_not_equal(self):
        """Test == is mutated to !=."""
        source = "def check(a, b):\n    return a == b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        eq_mutation = [m for m in mutations if m.original == "=="]
        assert len(eq_mutation) == 1
        assert eq_mutation[0].mutated == "!="

    def test_not_equal_to_equal(self):
        """Test != is mutated to ==."""
        source = "def check(a, b):\n    return a != b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        ne_mutation = [m for m in mutations if m.original == "!="]
        assert len(ne_mutation) == 1
        assert ne_mutation[0].mutated == "=="

    def test_greater_than_to_less_equal(self):
        """Test > is mutated to <=."""
        source = "def check(a, b):\n    return a > b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        gt_mutation = [m for m in mutations if m.original == ">"]
        assert len(gt_mutation) == 1
        assert gt_mutation[0].mutated == "<="

    def test_less_than_to_greater_equal(self):
        """Test < is mutated to >=."""
        source = "def check(a, b):\n    return a < b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        lt_mutation = [m for m in mutations if m.original == "<"]
        assert len(lt_mutation) == 1
        assert lt_mutation[0].mutated == ">="

    def test_greater_equal_to_less(self):
        """Test >= is mutated to <."""
        source = "def check(a, b):\n    return a >= b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        ge_mutation = [m for m in mutations if m.original == ">="]
        assert len(ge_mutation) == 1
        assert ge_mutation[0].mutated == "<"

    def test_less_equal_to_greater(self):
        """Test <= is mutated to >."""
        source = "def check(a, b):\n    return a <= b"
        operator = ComparisonMutationOperator()
        mutations = list(operator.apply(source))

        le_mutation = [m for m in mutations if m.original == "<="]
        assert len(le_mutation) == 1
        assert le_mutation[0].mutated == ">"


class TestMutationRunner:
    """Test mutation runner."""

    def test_runner_initialization(self):
        """Test runner initializes with operators."""
        runner = MutationRunner()
        assert len(runner.operators) == 2
        assert any(isinstance(op, ArithmeticMutationOperator) for op in runner.operators)
        assert any(isinstance(op, ComparisonMutationOperator) for op in runner.operators)

    def test_runner_returns_result(self):
        """Test runner returns a mutation result."""
        runner = MutationRunner()
        result = runner.run("dvas.data.schemas", "tests.test_schemas")

        assert isinstance(result, MutationResult)
        assert hasattr(result, "total_mutants")
        assert hasattr(result, "killed")
        assert hasattr(result, "survived")
        assert hasattr(result, "score")

    def test_mutation_result_to_dict(self):
        """Test mutation result serialization."""
        result = MutationResult(
            total_mutants=100,
            killed=85,
            survived=15,
            skipped=0,
            score=0.85,
        )

        d = result.to_dict()
        assert d["total_mutants"] == 100
        assert d["killed"] == 85
        assert d["survived"] == 15
        assert d["score"] == 0.85


class TestMutantSurvivedError:
    """Test mutant survived error."""

    def test_error_message(self):
        """Test error message contains mutation details."""
        mutation = Mutation(
            original="+",
            mutated="-",
            line_number=42,
            mutation_type="arithmetic",
        )

        error = MutantSurvivedError(mutation, "test_addition")
        assert "Mutant survived" in str(error)
        assert "line 42" in str(error)
        assert "'+'" in str(error)
        assert "'-'" in str(error)
        assert error.mutation == mutation
        assert error.test_name == "test_addition"

    def test_error_without_test_name(self):
        """Test error without test name."""
        mutation = Mutation(
            original="==",
            mutated="!=",
            line_number=10,
            mutation_type="comparison",
        )

        error = MutantSurvivedError(mutation)
        assert "line 10" in str(error)
        assert error.test_name == ""


class TestMutationDataclass:
    """Test Mutation dataclass."""

    def test_mutation_creation(self):
        """Test mutation creation."""
        mutation = Mutation(
            original="+",
            mutated="-",
            line_number=42,
            mutation_type="arithmetic",
        )

        assert mutation.original == "+"
        assert mutation.mutated == "-"
        assert mutation.line_number == 42
        assert mutation.mutation_type == "arithmetic"
