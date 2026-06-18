"""Tests for Jinja2 prompt template system.

Validates the PromptManager and template rendering functionality
introduced to resolve TD-002 (hardcoded prompts).
"""

from __future__ import annotations

import pytest


class TestPromptManager:
    """Test PromptManager task registry and rendering."""

    def test_get_prompt_caption(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("caption")
        assert "Describe what is happening" in prompt

    def test_get_prompt_dense_caption(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("dense_caption")
        assert "detailed description" in prompt.lower()
        assert "scene and context" in prompt.lower()

    def test_get_prompt_qa(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("qa")
        assert "question-answer" in prompt.lower()
        assert "Q:" in prompt
        assert "A:" in prompt

    def test_get_prompt_temporal(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("temporal")
        assert "action segments" in prompt.lower()
        assert "Start and end time" in prompt

    def test_get_prompt_fine_grained(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("fine_grained")
        assert "robotic manipulation" in prompt.lower()
        assert "JSON" in prompt

    def test_get_prompt_unknown_fallback(self):
        from dvas.config.prompts import PromptManager

        # Unknown task should fallback to caption
        prompt = PromptManager.get_prompt("unknown_task")
        assert "Describe what is happening" in prompt

    def test_get_prompt_with_override(self):
        from dvas.config.prompts import PromptManager

        prompt = PromptManager.get_prompt("qa", num_questions=3)
        assert "3 question-answer" in prompt

    def test_task_registry_contents(self):
        from dvas.config.prompts import PromptManager

        expected_tasks = ["caption", "dense_caption", "qa", "temporal", "fine_grained"]
        for task in expected_tasks:
            assert task in PromptManager.TASK_REGISTRY


class TestRenderPrompt:
    """Test low-level template rendering."""

    def test_render_base_tasks_template(self):
        from dvas.config.prompts import render_prompt

        result = render_prompt("base_tasks.j2", task="caption")
        assert "Describe what is happening" in result

    def test_render_nonexistent_template_fallback(self):
        from dvas.config.prompts import render_prompt

        # Should fallback gracefully
        result = render_prompt("nonexistent.j2", task="caption")
        assert isinstance(result, str)
        assert len(result) > 0


class TestListTemplates:
    """Test template discovery."""

    def test_list_templates_returns_list(self):
        from dvas.config.prompts import list_templates

        templates = list_templates()
        assert isinstance(templates, list)
        # Should include our base template
        assert "base_tasks.j2" in templates


class TestTeacherModelIntegration:
    """Test that TeacherModel uses new prompt system."""

    def test_teacher_model_uses_prompt_manager(self):
        from dvas.models.base import ModelType
        from dvas.models.teacher.base import TeacherModel

        # Mock concrete implementation with all abstract methods
        class MockTeacher(TeacherModel):
            @property
            def model_type(self) -> ModelType:
                return ModelType.TEACHER

            @property
            def model_version(self) -> str:
                return "test-v1"

            @property
            def _capabilities(self) -> list[str]:
                return ["caption", "qa"]

            async def generate(self, prompt, **kwargs):
                return None

            async def annotate(self, video_path, **kwargs):
                return None

            async def generate_batch(self, prompts, **kwargs):
                return [None] * len(prompts)

        teacher = MockTeacher(model_name="test")

        # Should use PromptManager internally
        caption = teacher._get_default_prompt("caption")
        assert isinstance(caption, str)
        assert len(caption) > 0

        fine_grained = teacher._get_default_prompt("fine_grained")
        assert "robotic" in fine_grained.lower() or "expert" in fine_grained.lower()
