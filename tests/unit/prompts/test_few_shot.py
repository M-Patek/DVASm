"""Tests for few-shot example retrieval."""

import pytest

from dvas.prompts.few_shot import (
    Example,
    ExamplePack,
    SemanticExampleIndex,
    _cosine_similarity,
    _simple_embedding,
    create_domain_example_packs,
)
from dvas.prompts.registry import PromptDomain


class TestCosineSimilarity:
    """Test suite for cosine similarity computation."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert _cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert _cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_different_dimensions(self):
        """Test similarity with different dimensions."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v1, v2) == 0.0

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 0.0]
        assert _cosine_similarity(v1, v2) == 0.0


class TestSimpleEmbedding:
    """Test suite for simple embedding generation."""

    def test_embedding_generation(self):
        """Test that embeddings are generated."""
        embedding = _simple_embedding("test text", dim=64)
        assert len(embedding) == 64
        # Should be normalized
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_deterministic(self):
        """Test that embeddings are deterministic."""
        e1 = _simple_embedding("same text", dim=64)
        e2 = _simple_embedding("same text", dim=64)
        assert e1 == e2

    def test_different_texts(self):
        """Test that different texts produce different embeddings."""
        e1 = _simple_embedding("hello world", dim=64)
        e2 = _simple_embedding("goodbye world", dim=64)
        assert e1 != e2


class TestSemanticExampleIndex:
    """Test suite for SemanticExampleIndex."""

    def test_add_example(self):
        """Test adding an example to the index."""
        index = SemanticExampleIndex()
        example = Example(
            id="ex1",
            input_text="A person chopping vegetables",
            output_text="The person uses a knife to chop vegetables.",
            domain=PromptDomain.KITCHEN,
        )
        index.add_example(example)
        assert index.size() == 1

    def test_search_by_domain(self):
        """Test searching examples by domain."""
        index = SemanticExampleIndex()
        kitchen_ex = Example(
            id="k1",
            input_text="Cooking in the kitchen",
            output_text="A person is cooking.",
            domain=PromptDomain.KITCHEN,
        )
        robot_ex = Example(
            id="r1",
            input_text="Robot grasping",
            output_text="A robot grasps an object.",
            domain=PromptDomain.ROBOT,
        )
        index.add_examples([kitchen_ex, robot_ex])

        results = index.search("cooking", domain=PromptDomain.KITCHEN, top_k=3)
        assert len(results) > 0
        # Results should be from kitchen domain
        assert results[0][0].domain == PromptDomain.KITCHEN

    def test_search_quality_filter(self):
        """Test searching with quality filter."""
        index = SemanticExampleIndex()
        low_quality = Example(
            id="lq",
            input_text="Bad example",
            output_text="Poor output",
            domain=PromptDomain.GENERAL,
            quality_score=0.3,
        )
        high_quality = Example(
            id="hq",
            input_text="Good example",
            output_text="Excellent output",
            domain=PromptDomain.GENERAL,
            quality_score=0.9,
        )
        index.add_examples([low_quality, high_quality])

        results = index.search("example", min_quality=0.5, top_k=3)
        assert len(results) > 0
        assert all(r[0].quality_score >= 0.5 for r in results)

    def test_get_by_domain(self):
        """Test getting all examples for a domain."""
        index = SemanticExampleIndex()
        ex1 = Example(id="k1", input_text="Kitchen", output_text="Cooking", domain=PromptDomain.KITCHEN)
        ex2 = Example(id="r1", input_text="Robot", output_text="Grasping", domain=PromptDomain.ROBOT)
        index.add_examples([ex1, ex2])

        kitchen_examples = index.get_by_domain(PromptDomain.KITCHEN)
        assert len(kitchen_examples) == 1
        assert kitchen_examples[0].id == "k1"

    def test_get_by_tag(self):
        """Test getting examples by tag."""
        index = SemanticExampleIndex()
        ex1 = Example(
            id="t1",
            input_text="Tagged",
            output_text="Output",
            tags=["important", "review"],
        )
        ex2 = Example(
            id="t2",
            input_text="Untagged",
            output_text="Output",
            tags=["other"],
        )
        index.add_examples([ex1, ex2])

        results = index.get_by_tag("important")
        assert len(results) == 1
        assert results[0].id == "t1"


class TestExamplePack:
    """Test suite for ExamplePack."""

    def test_add_and_get_examples(self):
        """Test adding and retrieving examples."""
        pack = ExamplePack("test", PromptDomain.KITCHEN)
        pack.add(Example(id="e1", input_text="Input", output_text="Output"))
        pack.add(Example(id="e2", input_text="Input2", output_text="Output2"))

        examples = pack.get_examples(top_k=2)
        assert len(examples) == 2

    def test_get_examples_with_query(self):
        """Test getting examples with query filtering."""
        pack = ExamplePack("test", PromptDomain.KITCHEN)
        pack.add(Example(
            id="e1",
            input_text="Chopping vegetables",
            output_text="A person chops vegetables.",
            tags=["chopping"],
        ))
        pack.add(Example(
            id="e2",
            input_text="Stirring soup",
            output_text="A person stirs soup.",
            tags=["stirring"],
        ))

        results = pack.get_examples(query="chopping", top_k=2)
        assert len(results) > 0

    def test_to_prompt_text(self):
        """Test converting examples to prompt text."""
        pack = ExamplePack("test", PromptDomain.KITCHEN)
        pack.add(Example(id="e1", input_text="Input", output_text="Output"))

        text = pack.to_prompt_text()
        assert "Example 1:" in text
        assert "Input: Input" in text
        assert "Output: Output" in text

    def test_to_prompt_text_with_subset(self):
        """Test converting subset of examples."""
        pack = ExamplePack("test", PromptDomain.KITCHEN)
        ex1 = Example(id="e1", input_text="In1", output_text="Out1")
        ex2 = Example(id="e2", input_text="In2", output_text="Out2")
        pack.add(ex1)
        pack.add(ex2)

        text = pack.to_prompt_text(examples=[ex1])
        assert "Example 1:" in text
        assert "In1" in text
        assert "In2" not in text


class TestCreateDomainExamplePacks:
    """Test suite for default domain example packs."""

    def test_creates_packs(self):
        """Test that default packs are created."""
        packs = create_domain_example_packs()
        assert PromptDomain.KITCHEN in packs
        assert PromptDomain.ROBOT in packs
        assert PromptDomain.GENERAL in packs

    def test_kitchen_pack_has_examples(self):
        """Test that kitchen pack has examples."""
        packs = create_domain_example_packs()
        kitchen = packs[PromptDomain.KITCHEN]
        assert len(kitchen.examples) > 0

    def test_robot_pack_has_examples(self):
        """Test that robot pack has examples."""
        packs = create_domain_example_packs()
        robot = packs[PromptDomain.ROBOT]
        assert len(robot.examples) > 0
