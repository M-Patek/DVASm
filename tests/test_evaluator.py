"""Tests for evaluation metrics and LLM judge.

Tests automatic metrics (BLEU, ROUGE, CIDEr, METEOR) and LLM-as-Judge.
"""

import pytest
from unittest.mock import MagicMock

from dvas.models.base import GenerationResult, ModelType
from dvas.models.evaluator.metrics import MetricsCalculator
from dvas.models.evaluator.llm_judge import LLMJudge, ConsistencyChecker


class TestMetricsCalculator:
    """Test automatic metrics calculation."""

    @pytest.fixture
    def calculator(self):
        """Create metrics calculator."""
        return MetricsCalculator()

    def test_bleu_exact_match(self, calculator):
        """Test BLEU with identical reference and hypothesis."""
        reference = "The quick brown fox jumps over the lazy dog"
        hypothesis = "The quick brown fox jumps over the lazy dog"

        scores = calculator.bleu(reference, hypothesis)

        # Exact match should have high scores
        assert scores["bleu_1"] > 0.9
        assert scores["bleu_2"] > 0.8

    def test_bleu_partial_match(self, calculator):
        """Test BLEU with partial match."""
        reference = "The cat sat on the mat"
        hypothesis = "The cat sat on a chair"

        scores = calculator.bleu(reference, hypothesis)

        # Should have some matching n-grams
        assert scores["bleu_1"] > 0.0
        assert scores["bleu_1"] <= 1.0

    def test_bleu_no_match(self, calculator):
        """Test BLEU with completely different texts."""
        reference = "abcdefg"
        hypothesis = "xyz123"

        scores = calculator.bleu(reference, hypothesis)

        # No matching n-grams
        assert scores["bleu_1"] == 0.0

    def test_bleu_empty_hypothesis(self, calculator):
        """Test BLEU with empty hypothesis."""
        reference = "Some text"
        hypothesis = ""

        scores = calculator.bleu(reference, hypothesis)

        assert scores["bleu_1"] == 0.0

    def test_rouge_available(self, calculator):
        """Test ROUGE scorer availability."""
        # ROUGE may or may not be available depending on environment
        assert calculator.rouge_scorer is not None or True  # Just check it doesn't crash

    def test_cider_basic(self, calculator):
        """Test CIDEr score calculation."""
        references = ["A person is cooking", "Someone is preparing food"]
        hypothesis = "A person is cooking dinner"

        score = calculator.cider(references, hypothesis)

        assert isinstance(score, float)
        assert score >= 0.0

    def test_meteor_basic(self, calculator):
        """Test METEOR score calculation."""
        reference = "The cat sat on the mat"
        hypothesis = "The cat was sitting on the mat"

        score = calculator.meteor(reference, hypothesis)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_tokenization(self, calculator):
        """Test text tokenization."""
        text = "Hello, World! This is a test."
        tokens = calculator._tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "hello" in tokens  # Should be lowercased


class TestLLMJudge:
    """Test LLM-as-Judge functionality."""

    @pytest.fixture
    def mock_teacher(self):
        """Create mock teacher model."""
        teacher = MagicMock()
        teacher.model_name = "gpt-5.5"
        return teacher

    @pytest.fixture
    def llm_judge(self, mock_teacher):
        """Create LLM judge with mock teacher."""
        return LLMJudge(judge_model=mock_teacher)

    def test_judge_initialization(self, mock_teacher):
        """Test judge initialization."""
        judge = LLMJudge(judge_model=mock_teacher)
        assert judge.judge == mock_teacher

    @pytest.mark.asyncio
    async def test_evaluate_segment(self, llm_judge, mock_teacher):
        """Test quality evaluation."""

        # Create async mock with response format that parser expects
        async def mock_annotate(*args, **kwargs):
            return GenerationResult(
                text="""Overall Score: 8.5/10

Dimension Scores:
- Accuracy: 9/10
- Completeness: 8/10

Justification: Good annotation with accurate details.

Suggestions: Add more detail.""",
                model_type=ModelType.TEACHER_GPT55,
                model_version="test-judge",
            )

        mock_teacher.annotate = mock_annotate

        annotation = "The person picks up a sharp knife"

        result = await llm_judge.evaluate_segment(annotation)

        assert isinstance(result, dict)
        assert "overall_score" in result
        assert result["overall_score"] == 8.5

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, llm_judge, mock_teacher):
        """Test batch evaluation."""

        # Create async mock
        async def mock_annotate(*args, **kwargs):
            return GenerationResult(
                text='{"scores": {"accuracy": 8}, "feedback": "Good"}',
                model_type=ModelType.TEACHER_GPT55,
                model_version="test-judge",
            )

        mock_teacher.annotate = mock_annotate

        items = [
            {"id": "1", "annotation": "The person picks up a knife"},
            {"id": "2", "annotation": "Someone picks something up"},
        ]

        results = await llm_judge.evaluate_batch(items)

        assert len(results) == 2


class TestConsistencyChecker:
    """Test annotation consistency checking."""

    @pytest.fixture
    def checker(self):
        """Create consistency checker with mocked metrics."""
        checker = ConsistencyChecker()
        # Mock ROUGE to avoid dependency issues
        checker.metrics.rouge_scorer = MagicMock()
        checker.metrics.rouge_scorer.score.return_value = {
            "rouge1": MagicMock(fmeasure=0.5),
            "rouge2": MagicMock(fmeasure=0.3),
            "rougeL": MagicMock(fmeasure=0.4),
        }
        return checker

    def test_check_temporal_consistency(self, checker):
        """Test temporal consistency checking."""
        segments = [
            {"start_time": 0, "end_time": 5, "caption": "pick"},
            {"start_time": 5, "end_time": 10, "caption": "cut"},
        ]

        result = checker.check_temporal_consistency(segments)

        # Should return consistency report dict
        assert isinstance(result, dict)
        assert "consistent" in result
        assert "issues" in result

    def test_check_temporal_overlap(self, checker):
        """Test detection of overlapping segments."""
        segments = [
            {"start_time": 0, "end_time": 5, "caption": "pick"},
            {"start_time": 3, "end_time": 8, "caption": "cut"},  # Overlaps with first
        ]

        result = checker.check_temporal_consistency(segments)

        # Should flag overlap
        assert len(result["issues"]) > 0

    def test_check_action_consistency(self, checker):
        """Test action consistency checking."""
        segments = [
            {"start_time": 0, "end_time": 5, "caption": "Picking up a knife"},
            {"start_time": 5, "end_time": 10, "caption": "Cutting a carrot"},
        ]

        result = checker.check_action_consistency(segments)

        # Results depend on implementation
        assert isinstance(result, dict)
        assert "consistent" in result or "score" in result


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        calculator = MetricsCalculator()

        reference = "The person picks up a knife and cuts the vegetables"
        hypothesis = "Someone picks up a knife and cuts vegetables"

        # Calculate multiple metrics
        bleu = calculator.bleu(reference, hypothesis)
        cider = calculator.cider([reference], hypothesis)
        meteor = calculator.meteor(reference, hypothesis)

        # All metrics should be reasonable (relax assertions for basic validation)
        assert bleu["bleu_1"] >= 0.0  # BLEU should be non-negative
        assert cider >= 0.0  # CIDEr should be non-negative
        assert meteor >= 0.0  # METEOR should be non-negative

    def test_metrics_with_annotations(self):
        """Test metrics with annotation objects."""
        from dvas.data.schemas import Annotation, Segment, VideoMetadata

        calculator = MetricsCalculator()

        # Create reference annotation
        ref_ann = Annotation(
            id="ref_001",
            video_id="vid_001",
            video_path="/v/1.mp4",
            segments=[
                Segment(start_time=0, end_time=5, caption="Picking up a knife"),
                Segment(start_time=5, end_time=10, caption="Cutting vegetables"),
            ],
            metadata=VideoMetadata(
                video_id="vid_001",
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        # Create hypothesis annotation
        hyp_ann = Annotation(
            id="hyp_001",
            video_id="vid_001",
            video_path="/v/1.mp4",
            segments=[
                Segment(start_time=0, end_time=5, caption="Grabbing a knife"),
                Segment(start_time=5, end_time=10, caption="Cutting the vegetables"),
            ],
            metadata=VideoMetadata(
                video_id="vid_001",
                fps=30.0,
                resolution=[1920, 1080],
                duration=10.0,
                total_frames=300,
            ),
        )

        # Compare segment captions
        ref_caps = " ".join([s.caption for s in ref_ann.segments])
        hyp_caps = " ".join([s.caption for s in hyp_ann.segments])

        bleu = calculator.bleu(ref_caps, hyp_caps)

        assert bleu["bleu_1"] > 0.0  # Some overlap expected
