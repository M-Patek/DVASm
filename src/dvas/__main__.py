"""CLI entry point for DVAS."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dvas.config import settings
from dvas.data.storage import AnnotationStore
from dvas.export.adapters import ADAPTERS
from dvas.models.teacher.gpt4v import GPT4VTeacher
from dvas.pipeline.core import AnnotationPipeline
from dvas.utils.logging import setup_logging

app = typer.Typer(help="DVAS - Distilled Video Annotation Specialist")
console = Console()


@app.command()
def annotate(
    video_path: Path = typer.Argument(..., help="Path to video file"),
    video_id: str = typer.Option(None, "--video-id", "-v", help="Video ID (auto-generated if not provided)"),
    teacher_model: str = typer.Option("gpt-4o", "--model", "-m", help="Teacher model to use"),
    num_frames: int = typer.Option(16, "--frames", "-f", help="Number of frames to sample"),
    output: Path = typer.Option(None, "--output", "-o", help="Output path for annotation"),
) -> None:
    """Annotate a single video."""
    if not video_path.exists():
        console.print(f"[red]Error: Video not found: {video_path}[/red]")
        raise typer.Exit(1)

    vid = video_id or f"vid_{video_path.stem}"

    console.print(f"[blue]Annotating {video_path.name}...[/blue]")

    async def _annotate() -> None:
        teacher = GPT4VTeacher(model_name=teacher_model)
        pipeline = AnnotationPipeline(teacher_model=teacher, num_frames=num_frames)
        annotation = await pipeline.annotate_video(video_path=video_path, video_id=vid)
        return annotation

    try:
        annotation = asyncio.run(_annotate())
        console.print(f"[green]Annotation complete: {annotation.id}[/green]")
        console.print(f"  Segments: {len(annotation.segments)}")
        console.print(f"  Duration: {annotation.get_total_duration():.1f}s")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                import json
                f.write(json.dumps(annotation.model_dump(), ensure_ascii=False, indent=2))
            console.print(f"[green]Saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Annotation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def annotate_epic(
    split: str = typer.Option("train", "--split", "-s", help="EPIC split to annotate"),
    num_videos: int = typer.Option(10, "--num", "-n", help="Number of videos to annotate"),
    epic_root: Path = typer.Option(None, "--epic-root", "-e", help="EPIC-KITCHENS root directory"),
) -> None:
    """Annotate EPIC-KITCHENS dataset."""
    from dvas.pipeline.core import EPICAnnotationPipeline

    root = epic_root or settings.EPIC_KITCHENS_ROOT
    if not root:
        console.print("[red]Error: EPIC_KITCHENS_ROOT not set[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Annotating {num_videos} videos from EPIC {split}...[/blue]")

    async def _process() -> None:
        pipeline = EPICAnnotationPipeline(epic_root=root)
        annotations, failed = await pipeline.annotate_split(split=split, max_videos=num_videos)
        return annotations, failed

    try:
        annotations, failed = asyncio.run(_process())
        console.print(f"[green]Completed: {len(annotations)} annotations[/green]")
        if failed:
            console.print(f"[yellow]Failed: {len(failed)} videos[/yellow]")
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("llava", "--format", "-f", help="Export format"),
    source: str = typer.Option("gold", "--source", "-s", help="Source: gold/model/reviewed"),
) -> None:
    """Export annotations to training format."""
    store = AnnotationStore()
    annotations = store.load_all(source=source)

    if not annotations:
        console.print("[yellow]No annotations found[/yellow]")
        return

    if format not in ADAPTERS:
        console.print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for ann in annotations:
            if format == "llava":
                data = ann.to_llava_format()
            elif format == "openai":
                data = ann.to_openai_format()
            else:
                data = ann.model_dump()
            import json
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    console.print(f"[green]Exported {count} annotations to {output}[/green]")


@app.command()
def stats(
    source: str = typer.Option("all", "--source", "-s", help="Filter by source"),
) -> None:
    """Show annotation statistics."""
    store = AnnotationStore()
    store_stats = store.get_statistics()

    table = Table(title="Annotation Statistics")
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Size (MB)", style="blue")

    if source == "all":
        sources = ["gold", "model", "reviewed"]
    else:
        sources = [source]

    for src in sources:
        data = store_stats.get(src, {"count": 0, "size_mb": 0})
        table.add_row(src, str(data["count"]), f"{data['size_mb']:.2f}")

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    setup_logging(level="INFO", json_format=False)
    app()


if __name__ == "__main__":
    main()
