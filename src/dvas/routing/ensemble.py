"""Multi-teacher ensemble voting with confidence aggregation."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dvas.models.base import GenerationResult
from dvas.models.teacher.base import TeacherModel
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TeacherVote:
    """Single teacher's vote."""

    teacher_id: str
    response: str
    confidence: float
    cost: float
    latency_ms: float
    reasoning: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result of ensemble voting."""

    final_response: str
    consensus_method: str
    agreement_score: float  # 0-1, how much teachers agree
    teacher_votes: List[TeacherVote]
    confidence_distribution: Dict[str, float]
    dissenting_opinions: List[Tuple[str, str]]  # teacher_id, response
    estimated_quality: float
    total_cost: float


class ConsensusMethods:
    """Available consensus methods for ensemble voting."""

    @staticmethod
    def majority_vote(votes: List[TeacherVote]) -> str:
        """Simple majority vote based on response similarity."""
        # Group by semantic similarity (simplified: exact match)
        from collections import Counter

        responses = [v.response for v in votes]
        counts = Counter(responses)
        return counts.most_common(1)[0][0]

    @staticmethod
    def weighted_by_confidence(votes: List[TeacherVote]) -> str:
        """Weighted vote by teacher confidence."""
        # Return highest confidence response
        return max(votes, key=lambda v: v.confidence).response

    @staticmethod
    def confidence_aware_merging(votes: List[TeacherVote]) -> str:
        """Merge responses considering confidence scores."""
        # Sort by confidence
        sorted_votes = sorted(votes, key=lambda v: v.confidence, reverse=True)

        if sorted_votes[0].confidence > 0.9:
            # High confidence single vote
            return sorted_votes[0].response

        # Merge top 2 responses
        top1 = sorted_votes[0]
        top2 = sorted_votes[1] if len(sorted_votes) > 1 else None

        if top2 and top1.confidence - top2.confidence < 0.2:
            # Similar confidence - do a simple merge
            return f"{top1.response}\n\n[Additional perspective]: {top2.response}"

        return top1.response

    @staticmethod
    def best_of_n_with_threshold(votes: List[TeacherVote], threshold: float = 0.7) -> Optional[str]:
        """Return best if it exceeds threshold, otherwise None."""
        best = max(votes, key=lambda v: v.confidence)
        if best.confidence >= threshold:
            return best.response
        return None


class MultiTeacherEnsemble:
    """Ensemble of multiple teachers with voting mechanisms."""

    def __init__(
        self,
        teachers: Optional[List[TeacherModel]] = None,
        consensus_method: str = "confidence_aware",
        min_agreement_threshold: float = 0.6,
        budget_limit: Optional[float] = None,
    ):
        self.teachers = teachers or self._default_teachers()
        self.consensus_method_func = self._get_consensus_method(consensus_method)
        self.min_agreement_threshold = min_agreement_threshold
        self.budget_limit = budget_limit
        self.vote_history: List[EnsembleResult] = []

    def _default_teachers(self) -> List[TeacherModel]:
        """Get default set of teachers."""
        return [
            TeacherModel(model_name="gpt-5.5"),
            TeacherModel(model_name="claude-opus-4-8"),
            TeacherModel(model_name="meta-llama/Llama-3.2-90B-Vision-Instruct"),
        ]

    def _get_consensus_method(self, name: str):
        """Get consensus method by name."""
        methods = {
            "majority": ConsensusMethods.majority_vote,
            "weighted": ConsensusMethods.weighted_by_confidence,
            "confidence_aware": ConsensusMethods.confidence_aware_merging,
            "best_of_n": ConsensusMethods.best_of_n_with_threshold,
        }
        return methods.get(name, ConsensusMethods.confidence_aware_merging)

    @staticmethod
    def _result_text(result: GenerationResult) -> str:
        """Return text from the standardized teacher response."""
        return result.text

    async def annotate_with_ensemble(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        max_teachers: Optional[int] = None,
    ) -> EnsembleResult:
        """Get annotation from ensemble of teachers."""
        teachers_to_query = self.teachers[:max_teachers] if max_teachers else self.teachers

        # Query all teachers in parallel
        async def query_teacher(teacher: TeacherModel) -> TeacherVote:
            import time

            start_time = time.time()
            teacher_id = teacher.__class__.__name__

            try:
                result = await teacher.annotate(frames=frames, prompt=prompt, task=task)

                latency = (time.time() - start_time) * 1000
                cost = self._estimate_cost(teacher_id, frames)

                # Estimate confidence from response characteristics.
                # TeacherModel.annotate() returns GenerationResult, not a dict.
                response_text = self._result_text(result)
                confidence = self._estimate_confidence(response_text)

                return TeacherVote(
                    teacher_id=teacher_id,
                    response=response_text,
                    confidence=confidence,
                    cost=cost,
                    latency_ms=latency,
                )

            except Exception as e:
                logger.error("teacher_query_failed", teacher=teacher_id, error=str(e))
                return TeacherVote(
                    teacher_id=teacher_id,
                    response="",
                    confidence=0.0,
                    cost=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # Execute all queries
        votes = await asyncio.gather(*[query_teacher(t) for t in teachers_to_query])

        # Filter out failed queries
        valid_votes = [v for v in votes if v.confidence > 0]

        if not valid_votes:
            raise RuntimeError("All teachers failed to respond")

        # Calculate agreement and consensus
        agreement = self._calculate_agreement(valid_votes)
        final_response = self.consensus_method_func(valid_votes)

        # Check if disagreement needs escalation
        dissenting = []
        if agreement < self.min_agreement_threshold:
            logger.warning(
                "low_teacher_agreement",
                agreement=agreement,
                min_threshold=self.min_agreement_threshold,
            )
            # Find dissenting opinions
            avg_conf = np.mean([v.confidence for v in valid_votes])
            dissenting = [
                (v.teacher_id, v.response) for v in valid_votes if v.confidence < avg_conf * 0.8
            ]

        result = EnsembleResult(
            final_response=final_response,
            consensus_method=self.consensus_method_func.__name__,
            agreement_score=agreement,
            teacher_votes=valid_votes,
            confidence_distribution={v.teacher_id: v.confidence for v in valid_votes},
            dissenting_opinions=dissenting,
            estimated_quality=agreement * np.mean([v.confidence for v in valid_votes]),
            total_cost=sum(v.cost for v in valid_votes),
        )

        self.vote_history.append(result)

        logger.info(
            "ensemble_vote_complete",
            num_teachers=len(valid_votes),
            agreement=agreement,
            total_cost=result.total_cost,
        )

        return result

    def _estimate_cost(self, teacher_id: str, frames: List[np.ndarray]) -> float:
        """Estimate API cost for teacher."""
        frame_count = len(frames)
        base_costs = {
            "GPTTeacher": 0.005,  # per frame
            "ClaudeTeacher": 0.004,
            "TogetherTeacher": 0.001,
        }
        return base_costs.get(teacher_id, 0.01) * frame_count

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence from response quality."""
        if not response:
            return 0.0

        score = 0.5  # Base score

        # Longer responses tend to be more detailed
        if len(response) > 200:
            score += 0.1
        if len(response) > 500:
            score += 0.1

        # Check for structure indicators
        structural_indicators = ["1.", "2.", "first", "then", "next", "finally"]
        score += sum(1 for ind in structural_indicators if ind in response.lower()) * 0.02

        # Check for action-related keywords
        action_words = [
            "hand",
            "pick",
            "place",
            "hold",
            "move",
            "grasp",
            "release",
        ]
        score += sum(1 for word in action_words if word in response.lower()) * 0.03

        return min(1.0, score)

    def _calculate_agreement(self, votes: List[TeacherVote]) -> float:
        """Calculate agreement score between teachers."""
        if len(votes) < 2:
            return 1.0

        # Use simple embedding similarity (placeholder)
        # In production, use sentence-transformers or similar
        responses = [v.response for v in votes]

        # Calculate pairwise Jaccard similarities
        similarities = []
        for i, r1 in enumerate(responses):
            for r2 in responses[i + 1 :]:
                set1 = set(r1.lower().split())
                set2 = set(r2.lower().split())
                if set1 and set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    similarities.append(jaccard)

        return float(np.mean(similarities)) if similarities else 0.5

    def get_teacher_performance_stats(self) -> Dict:
        """Get performance statistics for each teacher."""
        if not self.vote_history:
            return {}

        stats = {}
        for teacher in self.teachers:
            tid = teacher.__class__.__name__

            teacher_votes = []
            for result in self.vote_history:
                for vote in result.teacher_votes:
                    if vote.teacher_id == tid:
                        teacher_votes.append(vote)

            if teacher_votes:
                stats[tid] = {
                    "total_votes": len(teacher_votes),
                    "avg_confidence": np.mean([v.confidence for v in teacher_votes]),
                    "avg_latency_ms": np.mean([v.latency_ms for v in teacher_votes]),
                    "total_cost": sum(v.cost for v in teacher_votes),
                    "success_rate": len([v for v in teacher_votes if v.confidence > 0])
                    / len(teacher_votes),
                }

        return stats


class DisagreementResolver:
    """Handle cases where teachers disagree significantly."""

    def __init__(self):
        self.resolution_strategies = {
            "human_review": self._request_human_review,
            "tie_breaker": self._tie_breaker_llm,
            "conservative": self._conservative_merge,
        }

    def resolve(
        self,
        votes: List[TeacherVote],
        strategy: str = "tie_breaker",
    ) -> str:
        """Resolve disagreement using specified strategy."""
        resolver = self.resolution_strategies.get(strategy, self._tie_breaker_llm)
        return resolver(votes)

    def _request_human_review(self, votes: List[TeacherVote]) -> str:
        """Flag for human review."""
        logger.info("flagged_for_human_review", num_votes=len(votes))
        # Return highest confidence but mark as uncertain
        best = max(votes, key=lambda v: v.confidence)
        return f"[NEEDS REVIEW] {best.response}"

    def _tie_breaker_llm(self, votes: List[TeacherVote]) -> str:
        """Use another LLM call to break ties."""
        # Simplified: just return highest confidence
        # In production, send all responses to a meta-model
        return max(votes, key=lambda v: v.confidence).response

    def _conservative_merge(self, votes: List[TeacherVote]) -> str:
        """Merge responses conservatively."""
        # Include common elements only
        common_words = set.intersection(*[set(v.response.lower().split()) for v in votes])
        return " ".join(sorted(common_words)) if common_words else votes[0].response


class IncrementalConsensus:
    """Incremental voting - stop early if consensus reached."""

    def __init__(
        self,
        teachers: List[TeacherModel],
        target_confidence: float = 0.85,
        max_teachers: int = 3,
    ):
        self.teachers = teachers
        self.target_confidence = target_confidence
        self.max_teachers = max_teachers

    @staticmethod
    def _result_text(result: GenerationResult) -> str:
        """Return text from the standardized teacher response."""
        return result.text

    async def annotate_incremental(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        task: str = "fine_grained",
    ) -> EnsembleResult:
        """Query teachers incrementally until confidence target reached."""
        votes = []

        for i, teacher in enumerate(self.teachers[: self.max_teachers]):
            result = await teacher.annotate(frames=frames, prompt=prompt, task=task)
            response_text = self._result_text(result)

            vote = TeacherVote(
                teacher_id=teacher.__class__.__name__,
                response=response_text,
                confidence=self._estimate_confidence(response_text),
                cost=0.01,  # Placeholder
                latency_ms=0,
            )
            votes.append(vote)

            # Check if we've reached target
            if len(votes) >= 2:
                current_agreement = self._calculate_agreement(votes)
                if current_agreement >= self.target_confidence:
                    logger.info(
                        "early_stop_reached",
                        teachers_used=i + 1,
                        agreement=current_agreement,
                    )
                    break

        # Final consensus
        final_response = max(votes, key=lambda v: v.confidence).response

        return EnsembleResult(
            final_response=final_response,
            consensus_method="incremental",
            agreement_score=self._calculate_agreement(votes),
            teacher_votes=votes,
            confidence_distribution={v.teacher_id: v.confidence for v in votes},
            dissenting_opinions=[],
            estimated_quality=0.0,
            total_cost=sum(v.cost for v in votes),
        )

    def _estimate_confidence(self, response: str) -> float:
        return 0.5 if response else 0.0

    def _calculate_agreement(self, votes: List[TeacherVote]) -> float:
        return 1.0 if len(votes) < 2 else 0.7  # Placeholder
