"""CLI export tool for DVAS."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from dvas.data.storage import AnnotationStore
from dvas.export.adapters import ADAPTERS, export_annotations

app = typer.Typer(help="Export annotations to various formats")
console = Console()


@app.command()
def list_formats():
    """List available export formats."""
    table = Table(title="Available Export Formats")
    table.add_column("Format", style="cyan")
    table.add_column("Description", style="green")

    descriptions = {
        "llava": "LLaVA training format (conversations)",
        "openai": "OpenAI fine-tuning format (messages)",
        "sharegpt": "ShareGPT/Vicuna format",
    }

    for fmt in ADAPTERS.keys():
        table.add_row(fmt, descriptions.get(fmt, "Custom format"))

    console.print(table)


@app.command()
def export(
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    format: str = typer.Option("llava", "--format", "-f", help="Export format"),
    source: str = typer.Option("gold", "--source", "-s", help="Source: gold/model/reviewed"),
    video_ids: Optional[List[str]] = typer.Option(None, "--video-id", help="Specific video IDs"),
):
    """Export annotations to specified format."""
    store = AnnotationStore()

    # Load annotations
    if video_ids:
        annotations = []
        seen_annotation_ids = set()
        for vid in video_ids:
            candidates = [
                ann
                for ann in (
                    store.load(vid, source=source),
                    store.load(f"{vid}_annotated", source=source),
                )
                if ann is not None
            ]
            candidates.extend(store.load_all(source=source, video_id=vid))

            if not candidates:
                console.print(f"[yellow]Warning: Annotation not found for {vid}[/yellow]")

            for ann in candidates:
                if ann.id not in seen_annotation_ids:
                    annotations.append(ann)
                    seen_annotation_ids.add(ann.id)
    else:
        annotations = list(store.load_all(source=source))

    if not annotations:
        console.print("[red]No annotations found for export[/red]")
        raise typer.Exit(1)

    # Export
    count = export_annotations(annotations, output, format)

    # Show statistics
    table = Table(title="Export Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Format", format)
    table.add_row("Source", source)
    table.add_row("Annotations", str(count))
    table.add_row("Output", str(output))

    # Calculate additional stats
    if annotations:
        total_duration = sum(ann.get_total_duration() for ann in annotations)
        total_segments = sum(len(ann.segments) for ann in annotations)

        table.add_row("Total Duration (s)", f"{total_duration:.1f}")
        table.add_row("Total Segments", str(total_segments))

    console.print(table)
    console.print(f"[green]Successfully exported {count} annotations[/green]")


@app.command()
def stats(
    source: str = typer.Option("all", "--source", "-s", help="Filter by source"),
):
    """Show annotation statistics."""
    store = AnnotationStore()

    if source == "all":
        sources = ["gold", "model", "reviewed"]
    else:
        sources = [source]

    table = Table(title="Annotation Statistics")
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Size (MB)", style="blue")

    stats_data = store.get_statistics()

    for src in sources:
        data = stats_data.get(src, {"count": 0, "size_mb": 0})
        table.add_row(src, str(data["count"]), f"{data['size_mb']:.2f}")

    console.print(table)


@app.command()
def inspect(
    video_id: str = typer.Argument(..., help="Video ID to inspect"),
    source: str = typer.Option("gold", "--source", "-s"),
):
    """Inspect a specific annotation."""
    store = AnnotationStore()
    annotation = store.load(video_id, source=source)

    if not annotation:
        console.print(f"[red]Annotation not found: {video_id}[/red]")
        raise typer.Exit(1)

    # Display annotation details
    table = Table(title=f"Annotation: {annotation.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Video ID", annotation.video_id)
    table.add_row("Source", annotation.source)
    table.add_row("Model", annotation.model_version or "N/A")
    table.add_row(
        "Quality Score", f"{annotation.quality_score:.2f}" if annotation.quality_score else "N/A"
    )
    table.add_row("Segments", str(len(annotation.segments)))
    table.add_row("Created", annotation.created_at.isoformat())

    console.print(table)

    # Show segments
    for i, seg in enumerate(annotation.segments):
        console.print(
            f"\n[bold]Segment {i + 1}:[/bold] {seg.start_time:.1f}s - {seg.end_time:.1f}s"
        )
        console.print(f"  Caption: {seg.caption[:100]}...")
        console.print(f"  Actions: {len(seg.actions)}, Objects: {len(seg.objects)}")


def main():
    app()


if __name__ == "__main__":
    main()
