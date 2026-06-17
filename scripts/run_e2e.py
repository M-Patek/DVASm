"""End-to-end pipeline test using MockTeacher.

This script runs the full annotation pipeline on a synthetic video
without requiring any API keys or external services.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from dvas.data.video_loader import VideoLoader
from dvas.models.teacher.mock import MockTeacher
from dvas.pipeline.core import AnnotationPipeline


async def run_e2e_test(video_path: Path, output_dir: Path) -> None:
    """Run end-to-end annotation pipeline on a video."""
    print("=" * 60)
    print("DVAS End-to-End Pipeline Test")
    print("=" * 60)

    # Step 1: Initialize pipeline with mock teacher
    print("\n[1/5] Initializing pipeline with MockTeacher...")
    teacher = MockTeacher(model_name="mock-teacher-v1")
    pipeline = AnnotationPipeline(
        teacher_model=teacher,
        num_frames=8,
        checkpoint_path=output_dir / "checkpoint.json",
    )
    print(f"      Teacher: {teacher.model_name}")
    print(f"      Frames per segment: {pipeline.num_frames}")

    # Step 2: Run annotation
    print("\n[2/5] Running annotation pipeline...")
    annotation = await pipeline.annotate_video(
        video_path=video_path,
        video_id="test_video_001",
    )
    print(f"      Annotation ID: {annotation.id}")
    print(f"      Video ID: {annotation.video_id}")
    print(f"      Segments: {len(annotation.segments)}")
    print(f"      Total duration: {annotation.get_total_duration():.1f}s")

    # Step 3: Inspect segments
    print("\n[3/5] Inspecting segments...")
    for i, segment in enumerate(annotation.segments):
        print(f"\n      Segment {i+1}:")
        print(f"        Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
        print(f"        Caption: {segment.caption[:80]}...")
        print(f"        Objects: {len(segment.objects)}")
        print(f"        Actions: {len(segment.actions)}")
        print(f"        QA pairs: {len(segment.qa_pairs)}")

    # Step 4: Save annotation
    print("\n[4/5] Saving annotation...")
    output_path = output_dir / "annotation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(annotation.model_dump_json(indent=2))
    print(f"      Saved to: {output_path}")

    # Step 5: Export to training format
    print("\n[5/5] Exporting to training format...")
    llava_path = output_dir / "training_llava.jsonl"
    with open(llava_path, "w", encoding="utf-8") as f:
        data = annotation.to_llava_format()
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"      LLaVA format: {llava_path}")

    # Summary
    print("\n" + "=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)
    print(f"Teacher API calls: {teacher.call_count}")
    print(f"Total frames processed: {teacher.total_frames_processed}")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        print(f"  - {f.name} ({size:,} bytes)")


def main():
    video_path = Path("tmp/test_video.mp4")
    if not video_path.exists():
        print(f"Error: Test video not found at {video_path}")
        print("Run: python scripts/generate_test_video.py")
        return 1

    output_dir = Path("tmp/e2e_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_e2e_test(video_path, output_dir))
    return 0


if __name__ == "__main__":
    exit(main())
