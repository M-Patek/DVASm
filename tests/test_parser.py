"""Tests for the structured response parser."""

import pytest
from dvas.data.schemas import QAPair, Object, Action
from dvas.pipeline.parser import (
    JsonBlockStrategy,
    PlainTextStrategy,
    StructuredParser,
    StructuredTextStrategy,
)


class TestJsonBlockStrategy:
    """Test JSON block parsing strategy."""

    def test_parse_json_with_code_block(self):
        """Test parsing JSON inside markdown code block."""
        strategy = JsonBlockStrategy()
        text = """```json
{"scene_description": "A person cooking", "steps": [{"action": "cut", "details": "vegetables"}]}
```"""
        result = strategy.parse(text)
        assert result is not None
        assert result.scene_description == "A person cooking"
        assert len(result.qa_pairs) == 1
        assert result.parse_method == "json_block"

    def test_parse_json_without_code_block(self):
        """Test parsing raw JSON."""
        strategy = JsonBlockStrategy()
        text = '{"scene_description": "Test", "objects": [{"name": "knife"}]}'
        result = strategy.parse(text)
        assert result is not None
        assert result.scene_description == "Test"
        assert len(result.objects) == 1
        assert result.objects[0].name == "knife"

    def test_parse_no_json(self):
        """Test that non-JSON text returns None."""
        strategy = JsonBlockStrategy()
        result = strategy.parse("This is just plain text without JSON.")
        assert result is None

    def test_parse_objects_with_state(self):
        """Test extracting objects with state attributes."""
        strategy = JsonBlockStrategy()
        text = '{"objects": [{"name": "knife", "state": "sharp"}, {"name": "bowl"}]}'
        result = strategy.parse(text)
        assert result is not None
        assert len(result.objects) == 2
        assert result.objects[0].name == "knife"
        assert result.objects[0].attributes["state"] == "sharp"

    def test_parse_hand_actions(self):
        """Test extracting hand actions."""
        strategy = JsonBlockStrategy()
        text = '{"hand_actions": [{"hand": "right", "action": "cutting", "target": "vegetables"}]}'
        result = strategy.parse(text)
        assert result is not None
        assert len(result.actions) == 1
        assert result.actions[0].verb == "cutting"
        assert result.actions[0].noun == "vegetables"

    def test_parse_actions_alternative_format(self):
        """Test extracting actions from alternative format."""
        strategy = JsonBlockStrategy()
        text = '{"actions": [{"verb": "stir", "noun": "soup", "hand": "left"}]}'
        result = strategy.parse(text)
        assert result is not None
        assert len(result.actions) == 1
        assert result.actions[0].verb == "stir"
        assert result.actions[0].noun == "soup"

    def test_parse_qa_pairs_explicit(self):
        """Test extracting explicit QA pairs."""
        strategy = JsonBlockStrategy()
        text = '{"qa_pairs": [{"question": "What is happening?", "answer": "Cooking"}]}'
        result = strategy.parse(text)
        assert result is not None
        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].question == "What is happening?"
        assert result.qa_pairs[0].answer == "Cooking"

    def test_parse_confidence_calculation(self):
        """Test confidence is calculated based on structure richness."""
        strategy = JsonBlockStrategy()
        text = '{"scene_description": "Test", "steps": [], "objects": [], "hand_actions": []}'
        result = strategy.parse(text)
        assert result is not None
        assert result.confidence > 0
        assert result.confidence <= 1.0


class TestStructuredTextStrategy:
    """Test structured text parsing strategy."""

    def test_parse_with_scene_marker(self):
        """Test parsing text with scene marker."""
        strategy = StructuredTextStrategy()
        text = "Scene: A person cooking in the kitchen.\n\nActions:\n- cut vegetables\n- stir soup"
        result = strategy.parse(text)
        assert result is not None
        assert "cooking" in result.scene_description
        assert len(result.actions) == 2

    def test_parse_with_description_marker(self):
        """Test parsing text with description marker."""
        strategy = StructuredTextStrategy()
        text = "Description: A detailed scene description.\n\nObjects: knife, bowl"
        result = strategy.parse(text)
        assert result is not None
        assert "detailed" in result.scene_description

    def test_parse_no_structure(self):
        """Test that plain text without markers returns None."""
        strategy = StructuredTextStrategy()
        result = strategy.parse("Just plain text without any markers.")
        assert result is None

    def test_parse_objects_section(self):
        """Test extracting objects from objects section."""
        strategy = StructuredTextStrategy()
        text = "Scene: Test.\n\nObjects: knife, cutting board, bowl"
        result = strategy.parse(text)
        assert result is not None
        assert len(result.objects) == 3
        assert any(obj.name == "knife" for obj in result.objects)


class TestPlainTextStrategy:
    """Test plain text fallback strategy."""

    def test_always_succeeds(self):
        """Test that plain text strategy always succeeds."""
        strategy = PlainTextStrategy()
        result = strategy.parse("Any text at all.")
        assert result is not None
        assert result.scene_description == "Any text at all."
        assert result.parse_method == "plain_text"
        assert result.confidence == 0.1

    def test_empty_text(self):
        """Test handling of empty text."""
        strategy = PlainTextStrategy()
        result = strategy.parse("")
        assert result is not None
        assert result.scene_description == ""


class TestStructuredParser:
    """Test the main StructuredParser class."""

    def test_parse_json_first(self):
        """Test that JSON is preferred over plain text."""
        parser = StructuredParser()
        text = '{"scene_description": "JSON content"}'
        result = parser.parse(text)
        assert result.parse_method == "json_block"
        assert result.scene_description == "JSON content"

    def test_fallback_to_plain_text(self):
        """Test fallback to plain text when no JSON."""
        parser = StructuredParser()
        text = "This is just plain text without any structure."
        result = parser.parse(text)
        assert result.parse_method == "plain_text"
        assert result.scene_description == text

    def test_failure_tracking(self):
        """Test that failures are tracked."""
        parser = StructuredParser()
        # Empty text should not count as failure (handled specially)
        parser.parse("")
        # All strategies succeed on empty, so no failures
        assert len(parser.failures) == 0

    def test_custom_strategies(self):
        """Test parser with custom strategy list."""
        parser = StructuredParser(strategies=[PlainTextStrategy()])
        result = parser.parse("Any text")
        assert result.parse_method == "plain_text"

    def test_get_failure_stats_empty(self):
        """Test failure stats with no failures."""
        parser = StructuredParser()
        stats = parser.get_failure_stats()
        assert stats["total_failures"] == 0

    def test_to_legacy_dict(self):
        """Test conversion to legacy dict format."""
        from dvas.pipeline.parser import ParsedSegment

        parser = StructuredParser()
        segment = ParsedSegment(
            scene_description="Test",
            qa_pairs=[QAPair(question="Q?", answer="A")],
            parse_method="test",
            confidence=0.8,
        )
        legacy = parser.to_legacy_dict(segment)
        assert legacy["scene_description"] == "Test"
        assert len(legacy["qa_pairs"]) == 1
        assert legacy["_parse_metadata"]["method"] == "test"
        assert legacy["_parse_metadata"]["confidence"] == 0.8
