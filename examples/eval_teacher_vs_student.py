#!/usr/bin/env python3
"""Example: Evaluate teacher vs student model quality, cost, and latency.

This script demonstrates:
1. Running inference with both teacher and student models
2. Comparing quality metrics (BLEU, ROUGE, LLM-as-Judge)
3. Comparing inference costs
4. Comparing latency
5. Generating comparison reports

Usage:
    # With pre-computed predictions
    python examples/eval_teacher_vs_student.py \
        --teacher-preds outputs/teacher_preds.jsonl \
        --student-preds outputs/student_preds.jsonl \
        --references data/gold_annotations.jsonl

    # Live evaluation (requires API keys)
    python examples/eval_teacher_vs_student.py \
        --live \
        --student-model outputs/student_sft/final \
        --test-videos data/test/videos/*.mp4 \
        --references data/test/refs.jsonl

Requirements:
    - For live eval: ANTHROPIC_API_KEY or OPENAI_API_KEY
    - Student model checkpoint (for live eval)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_predictions(path: Path) -> List[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                predictions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return predictions


def save_predictions(predictions: List[dict], path: Path) -> None:
    """Save predictions to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def predictions_to_results(predictions: List[dict]) -> List:
    """Convert prediction dicts to GenerationResult objects."""
    from dvas.models.base import GenerationResult, GenerationStatus, ModelType

    results = []
    for pred in predictions:
        result = GenerationResult(
            text=pred.get("text", ""),
            model_type=ModelType(pred.get("model_type", "teacher_claude")),
            model_version=pred.get("model_version", "unknown"),
            status=GenerationStatus(pred.get("status", "success")),
            confidence=pred.get("confidence"),
            latency_ms=pred.get("latency_ms"),
            cost_usd=pred.get("cost_usd", 0.0),
            metadata=pred.get("metadata"),
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher vs student models"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--teacher-preds",
        type=Path,
        help="Path to teacher predictions JSONL",
    )
    input_group.add_argument(
        "--live",
        action="store_true",
        help="Run live evaluation",
    )

    # Pre-computed predictions
    parser.add_argument(
        "--student-preds",
        type=Path,
        help="Path to student predictions JSONL (for pre-computed mode)",
    )

    # Live evaluation options
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="claude",
        help="Teacher model to use (claude, gpt4v, together)",
    )
    parser.add_argument(
        "--student-model",
        type=Path,
        help="Path to student model checkpoint",
    )
    parser.add_argument(
        "--test-videos",
        nargs="+",
        type=Path,
        help="Test video paths (for live eval)",
    )

    # Common options
    parser.add_argument(
        "--references",
        type=Path,
        help="Path to ground truth references (optional)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the video in detail, including hand actions and object interactions.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/teacher_vs_student_report.json"),
        help="Output path for report",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save predictions for later analysis",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM-as-Judge for quality evaluation",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.live:
        if not args.student_preds:
            print("Error: --student-preds required when using pre-computed predictions")
            sys.exit(1)
        if not args.teacher_preds.exists():
            print(f"Error: Teacher predictions not found: {args.teacher_preds}")
            sys.exit(1)
        if not args.student_preds.exists():
            print(f"Error: Student predictions not found: {args.student_preds}")
            sys.exit(1)
    else:
        if not args.student_model:
            print("Error: --student-model required for live evaluation")
            sys.exit(1)
        if not args.test_videos:
            print("Error: --test-videos required for live evaluation")
            sys.exit(1)

    print("=" * 60)
    print("TEACHER vs STUDENT EVALUATION")
    print("=" * 60)

    # Load or generate predictions
    if args.live:
        print("\nRunning live evaluation...")
        teacher_results, student_results = run_live_evaluation(
            args.teacher_model,
            args.student_model,
            args.test_videos,
            args.prompt,
            args.save_preds,
        )
    else:
        print("\nLoading pre-computed predictions...")
        teacher_preds = load_predictions(args.teacher_preds)
        student_preds = load_predictions(args.student_preds)
        teacher_results = predictions_to_results(teacher_preds)
        student_results = predictions_to_results(student_preds)
        print(f"  Loaded {len(teacher_results)} teacher predictions")
        print(f"  Loaded {len(student_results)} student predictions")

    # Load references if provided
    references = None
    if args.references:
        refs_data = load_predictions(args.references)
        references = [r.get("text", r.get("caption", "")) for r in refs_data]
        print(f"  Loaded {len(references)} references")

    # Run evaluation
    print("\nRunning comparison...")
    from dvas.models.student.evaluation import TeacherStudentEvaluator

    evaluator = TeacherStudentEvaluator()
    report = evaluator.compare_on_predictions(
        teacher_results,
        student_results,
        references,
    )

    # Print summary
    report.print_summary()

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to: {args.output}")

    # Generate detailed analysis if LLM judge requested
    if args.use_llm_judge and references:
        print("\nRunning LLM-as-Judge evaluation...")
        run_llm_judge_evaluation(teacher_results, student_results, references)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


def run_live_evaluation(
    teacher_model: str,
    student_model: Path,
    test_videos: List[Path],
    prompt: str,
    save_preds: bool,
) -> tuple:
    """Run live evaluation with both models."""
    import asyncio

    from dvas.models.base import GenerationResult, GenerationStatus, ModelType
    from dvas.models.student.inference import StudentInferenceEngine
    from dvas.models.teacher import TeacherModel

    # Load models
    print(f"  Loading teacher model: {teacher_model}")
    teacher = TeacherModel(model_name=teacher_model)

    print(f"  Loading student model: {student_model}")
    student = StudentInferenceEngine(student_model)

    # Run inference
    teacher_results = []
    student_results = []

    async def run_inference():
        for video_path in test_videos:
            if not video_path.exists():
                print(f"    Warning: Video not found: {video_path}")
                continue

            print(f"    Processing: {video_path.name}")

            # Teacher
            t_result = await teacher.annotate(
                video_path=video_path,
                prompt=prompt,
            )
            teacher_results.append(t_result)

            # Student
            s_result = await student.generate(
                video_path=video_path,
                prompt=prompt,
            )
            student_results.append(s_result)

    asyncio.run(run_inference())

    # Save predictions if requested
    if save_preds:
        teacher_preds_path = Path("outputs/teacher_preds_live.jsonl")
        student_preds_path = Path("outputs/student_preds_live.jsonl")

        teacher_preds = [
            {
                "video": str(test_videos[i]),
                "text": r.text,
                "model_type": r.model_type.value,
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
            }
            for i, r in enumerate(teacher_results)
        ]
        student_preds = [
            {
                "video": str(test_videos[i]),
                "text": r.text,
                "model_type": r.model_type.value,
                "confidence": r.confidence,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
            }
            for i, r in enumerate(student_results)
        ]

        save_predictions(teacher_preds, teacher_preds_path)
        save_predictions(student_preds, student_preds_path)
        print(f"  Saved predictions to {teacher_preds_path} and {student_preds_path}")

    return teacher_results, student_results


def run_llm_judge_evaluation(
    teacher_results: List,
    student_results: List,
    references: List[str],
) -> None:
    """Run LLM-as-Judge evaluation."""
    from dvas.models.evaluator.llm_judge import LLMJudge

    judge = LLMJudge()

    print("\n  Evaluating teacher predictions...")
    teacher_scores = []
    for result, ref in zip(teacher_results, references):
        score = judge.evaluate_prediction(result.text, ref)
        teacher_scores.append(score)

    print("  Evaluating student predictions...")
    student_scores = []
    for result, ref in zip(student_results, references):
        score = judge.evaluate_prediction(result.text, ref)
        student_scores.append(score)

    import numpy as np
    print("\n  LLM Judge Results:")
    print(f"    Teacher average score: {np.mean(teacher_scores):.3f}")
    print(f"    Student average score: {np.mean(student_scores):.3f}")
    print(f"    Score difference: {np.mean(student_scores) - np.mean(teacher_scores):+.3f}")


if __name__ == "__main__":
    main()
