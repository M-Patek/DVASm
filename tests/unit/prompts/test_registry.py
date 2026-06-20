"""Tests for prompt registry CRUD operations."""

import pytest

from dvas.prompts.registry import (
    PromptDomain,
    PromptMetadata,
    PromptRegistry,
    PromptTemplate,
)


class TestPromptRegistry:
    """Test suite for PromptRegistry CRUD operations."""

    def test_create_prompt(self):
        """Test creating a new prompt template."""
        registry = PromptRegistry()
        prompt = registry.create(
            name="test_prompt",
            template="Describe this video: {video_path}",
            domain=PromptDomain.KITCHEN,
            version="1.0.0",
            description="A test prompt",
            tags=["test", "kitchen"],
            variables=["video_path"],
        )

        assert prompt is not None
        assert prompt.metadata.name == "test_prompt"
        assert prompt.metadata.domain == PromptDomain.KITCHEN
        assert "test" in prompt.metadata.tags
        assert prompt.id is not None

    def test_get_prompt(self):
        """Test retrieving a prompt by ID."""
        registry = PromptRegistry()
        created = registry.create(name="get_test", template="Test template")
        retrieved = registry.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.metadata.name == "get_test"

    def test_get_nonexistent(self):
        """Test retrieving a non-existent prompt."""
        registry = PromptRegistry()
        assert registry.get("nonexistent") is None

    def test_get_by_name(self):
        """Test retrieving prompts by name."""
        registry = PromptRegistry()
        registry.create(name="duplicate", template="First")
        registry.create(name="duplicate", template="Second")

        results = registry.get_by_name("duplicate")
        assert len(results) == 2

    def test_update_prompt(self):
        """Test updating a prompt template."""
        registry = PromptRegistry()
        prompt = registry.create(name="update_test", template="Old template")

        updated = registry.update(prompt.id, template="New template", description="Updated")
        assert updated is not None
        assert updated.template == "New template"
        assert updated.metadata.description == "Updated"

    def test_delete_prompt(self):
        """Test deleting a prompt template."""
        registry = PromptRegistry()
        prompt = registry.create(name="delete_test", template="To be deleted")

        assert registry.delete(prompt.id) is True
        assert registry.get(prompt.id) is None

    def test_delete_nonexistent(self):
        """Test deleting a non-existent prompt."""
        registry = PromptRegistry()
        assert registry.delete("nonexistent") is False

    def test_list_all(self):
        """Test listing all prompts."""
        registry = PromptRegistry()
        registry.create(name="a", template="A")
        registry.create(name="b", template="B")

        all_prompts = registry.list_all()
        assert len(all_prompts) == 2

    def test_list_by_domain(self):
        """Test listing prompts by domain."""
        registry = PromptRegistry()
        registry.create(name="kitchen", template="K", domain=PromptDomain.KITCHEN)
        registry.create(name="robot", template="R", domain=PromptDomain.ROBOT)
        registry.create(name="general", template="G", domain=PromptDomain.GENERAL)

        kitchen_prompts = registry.list_by_domain(PromptDomain.KITCHEN)
        assert len(kitchen_prompts) == 1
        assert kitchen_prompts[0].metadata.name == "kitchen"

    def test_list_by_tag(self):
        """Test listing prompts by tag."""
        registry = PromptRegistry()
        registry.create(name="tagged", template="T", tags=["important", "review"])
        registry.create(name="untagged", template="U", tags=[])

        tagged = registry.list_by_tag("important")
        assert len(tagged) == 1
        assert tagged[0].metadata.name == "tagged"

    def test_fork_prompt(self):
        """Test forking a prompt to create a new version."""
        registry = PromptRegistry()
        parent = registry.create(name="parent", template="Parent template")
        child = registry.fork(parent.id, "child", "2.0.0")

        assert child is not None
        assert child.metadata.parent_id == parent.id
        assert child.metadata.name == "child"
        assert child.metadata.version == "2.0.0"
        assert parent.id in child.lineage

    def test_lineage_tracking(self):
        """Test lineage tracking for prompts templates."""
        registry = PromptRegistry()
        grandparent = registry.create(name="gp", template="GP")
        parent = registry.fork(grandparent.id, "parent", "1.1.0")
        child = registry.fork(parent.id, "child", "1.2.0")

        lineage = registry.get_lineage(child.id)
        assert len(lineage) == 2
        assert lineage[0].id == grandparent.id
        assert lineage[1].id == parent.id

    def test_get_children(self):
        """Test getting child prompts."""
        registry = PromptRegistry()
        parent = registry.create(name="parent", template="P")
        registry.fork(parent.id, "child1", "1.1.0")
        registry.fork(parent.id, "child2", "1.2.0")

        children = registry.get_children(parent.id)
        assert len(children) == 2


class TestPromptMetadata:
    """Test suite for PromptMetadata."""

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        meta = PromptMetadata(
            name="test",
            version="1.0.0",
            domain=PromptDomain.GENERAL,
            description="Test desc",
            tags=["a", "b"],
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["domain"] == "general"
        assert d["tags"] == ["a", "b"]

    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "domain": "kitchen",
            "description": "Test",
            "tags": ["a"],
            "author": "tester",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": None,
            "parent_id": None,
        }
        meta = PromptMetadata.from_dict(data)
        assert meta.name == "test"
        assert meta.domain == PromptDomain.KITCHEN


class TestPromptTemplate:
    """Test suite for PromptTemplate."""

    def test_hash_computation(self):
        """Test template hash computation."""
        meta = PromptMetadata(name="test", version="1.0.0", domain=PromptDomain.GENERAL)
        template = PromptTemplate(
            id="test-id",
            metadata=meta,
            template="Test template content",
        )
        assert len(template.hash) == 16
        assert template.hash != ""

    def test_update_quality(self):
        """Test quality score update."""
        meta = PromptMetadata(name="test", version="1.0.0", domain=PromptDomain.GENERAL)
        template = PromptTemplate(
            id="test-id",
            metadata=meta,
            template="Test",
        )
        template.update_quality(0.8)
        assert template.avg_quality_score == 0.8
        assert template.usage_count == 1

        template.update_quality(0.9)
        # Moving average: (0.8 * 1 + 0.9) / 2 = 0.85
        assert abs(template.avg_quality_score - 0.85) < 0.01

    def test_quality_clamping(self):
        """Test that quality scores are clamped to [0, 1]."""
        meta = PromptMetadata(name="test", version="1.0.0", domain=PromptDomain.GENERAL)
        template = PromptTemplate(
            id="test-id",
            metadata=meta,
            template="Test",
            avg_quality_score=1.5,  # type: ignore
        )
        assert template.avg_quality_score <= 1.0
