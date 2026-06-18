"""Tests for smart router."""



class TestVideoComplexityAnalyzer:
    """Test complexity analyzer."""

    def test_complexity_metrics(self):
        """Test complexity profile calculation."""
        from dvas.routing.smart_router import VideoComplexityProfile

        profile = VideoComplexityProfile(
            motion_score=0.8,
            scene_complexity=5,
            object_density=0.6,
            temporal_consistency=0.7,
            hand_interaction_density=0.9,
            duration_seconds=15.0,
        )

        complexity = profile.overall_complexity
        assert 0 <= complexity <= 1


class TestSmartRouter:
    """Test smart router."""

    def test_routing_strategies(self):
        """Test different routing strategies."""
        from dvas.routing.smart_router import RoutingStrategy

        strategies = [
            RoutingStrategy.COST_OPTIMIZED,
            RoutingStrategy.QUALITY_OPTIMIZED,
            RoutingStrategy.BALANCED,
            RoutingStrategy.ADAPTIVE,
        ]

        for strategy in strategies:
            assert isinstance(strategy.value, str)


class TestEnsemble:
    """Test ensemble voting."""

    def test_teacher_vote(self):
        """Test teacher vote structure."""
        from dvas.routing.ensemble import TeacherVote

        vote = TeacherVote(
            teacher_id="gpt55",
            response="Test response",
            confidence=0.9,
            cost=0.05,
            latency_ms=100.0,
        )

        assert vote.teacher_id == "gpt55"
        assert vote.confidence == 0.9
