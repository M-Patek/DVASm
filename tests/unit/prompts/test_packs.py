"""Tests for prompt pack modules."""

import pytest

from dvas.prompts.packs.human_review_pack import HumanReviewPromptPack
from dvas.prompts.packs.vla_pack import VLAPromptPack
from dvas.prompts.packs.world_model_pack import WorldModelPromptPack
from dvas.prompts.registry import PromptDomain


class TestVLAPromptPack:
    """Test suite for VLAPromptPack."""

    def test_get_template(self):
        """Test getting a VLA template."""
        pack = VLAPromptPack()
        template = pack.get_template("vla_grasp_analysis")
        assert template is not None
        assert "grasp" in template.lower()

    def test_get_nonexistent_template(self):
        """Test getting non-existent template."""
        pack = VLAPromptPack()
        assert pack.get_template("nonexistent") is None

    def test_list_templates(self):
        """Test listing all templates."""
        pack = VLAPromptPack()
        templates = pack.list_templates()
        assert len(templates) > 0
        assert "vla_grasp_analysis" in templates

    def test_get_grasp_prompt(self):
        """Test getting grasp analysis prompt."""
        pack = VLAPromptPack()
        prompt = pack.get_grasp_prompt(object_name="cup")
        assert "cup" in prompt

    def test_get_trajectory_prompt(self):
        """Test getting trajectory prompt."""
        pack = VLAPromptPack()
        prompt = pack.get_trajectory_prompt()
        assert "trajectory" in prompt.lower()

    def test_create_prompt_template(self):
        """Test creating a PromptTemplate from pack."""
        pack = VLAPromptPack()
        template = pack.create_prompt_template("vla_grasp_analysis")
        assert template is not None
        assert template.metadata.domain == PromptDomain.VLA

    def test_create_nonexistent_template(self):
        """Test creating from non-existent template name."""
        pack = VLAPromptPack()
        assert pack.create_prompt_template("nonexistent") is None


class TestWorldModelPromptPack:
    """Test suite for WorldModelPromptPack."""

    def test_get_template(self):
        """Test getting a world model template."""
        pack = WorldModelPromptPack()
        template = pack.get_template("wm_state_prediction")
        assert template is not None
        assert "predict" in template.lower()

    def test_list_templates(self):
        """Test listing all templates."""
        pack = WorldModelPromptPack()
        templates = pack.list_templates()
        assert len(templates) > 0
        assert "wm_state_prediction" in templates

    def test_get_state_prediction_prompt(self):
        """Test getting state prediction prompt."""
        pack = WorldModelPromptPack()
        prompt = pack.get_state_prediction_prompt(objects_hint=["cup", "table"])
        assert "cup" in prompt
        assert "table" in prompt

    def test_get_dynamics_prompt(self):
        """Test getting dynamics prompt."""
        pack = WorldModelPromptPack()
        prompt = pack.get_dynamics_prompt(physics_type="collision")
        assert "collision" in prompt

    def test_create_prompt_template(self):
        """Test creating a PromptTemplate from pack."""
        pack = WorldModelPromptPack()
        template = pack.create_prompt_template("wm_state_prediction")
        assert template is not None
        assert template.metadata.domain == PromptDomain.WORLD_MODEL


class TestHumanReviewPromptPack:
    """Test suite for HumanReviewPromptPack."""

    def test_get_template(self):
        """Test getting a human review template."""
        pack = HumanReviewPromptPack()
        template = pack.get_template("review_overall_quality")
        assert template is not None
        assert "quality" in template.lower()

    def test_list_templates(self):
        """Test listing all templates."""
        pack = HumanReviewPromptPack()
        templates = pack.list_templates()
        assert len(templates) > 0
        assert "review_overall_quality" in templates

    def test_get_quality_review_prompt(self):
        """Test getting quality review prompt."""
        pack = HumanReviewPromptPack()
        prompt = pack.get_quality_review_prompt(
            annotation_text="The person picks up a cup.",
            video_description="Kitchen scene",
        )
        assert "person picks up a cup" in prompt
        assert "Kitchen scene" in prompt

    def test_get_comparison_prompt(self):
        """Test getting comparison prompt."""
        pack = HumanReviewPromptPack()
        prompt = pack.get_comparison_prompt(
            annotation_a="Person walks left.",
            annotation_b="Person walks right.",
        )
        assert "Person walks left" in prompt
        assert "Person walks right" in prompt

    def test_create_prompt_template(self):
        """Test creating a PromptTemplate from pack."""
        pack = HumanReviewPromptPack()
        template = pack.create_prompt_template("review_overall_quality")
        assert template is not None
        assert template.metadata.domain == PromptDomain.HUMAN_REVIEW
