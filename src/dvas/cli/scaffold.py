"""Code scaffolding commands for DVAS CLI.

Provides project scaffolding for modules, models, pipelines, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from dvas.config import settings

console = Console()
app = typer.Typer(help="DVAS Scaffolding")


@dataclass
class ScaffoldTemplate:
    """Template for code scaffolding."""

    name: str
    description: str
    files: Dict[str, str] = field(default_factory=dict)


SCAFFOLD_TEMPLATES: Dict[str, ScaffoldTemplate] = {
    "module": ScaffoldTemplate(
        name="module",
        description="New subsystem module",
        files={
            "__init__.py": '''"""{module_name} module for DVAS."""

from dvas.{module_name}.core import {ModuleName}Processor

__all__ = ["{ModuleName}Processor"]
''',
            "core.py": '''"""Core {module_name} functionality."""

from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Processor:
    """Main processor for {module_name}."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {{}}

    def process(self, data: Any) -> Any:
        """Process input data."""
        raise NotImplementedError
''',
            "types.py": '''"""Type definitions for {module_name}."""

from typing import TypedDict


class {ModuleName}Config(TypedDict, total=False):
    """Configuration for {module_name}."""

    enabled: bool
    timeout: float
''',
        },
    ),
    "model": ScaffoldTemplate(
        name="model",
        description="New teacher/student model",
        files={
            "__init__.py": '''"""{module_name} model for DVAS."""

from dvas.models.{module_name}.model import {ModuleName}Teacher

__all__ = ["{ModuleName}Teacher"]
''',
            "model.py": '''"""{ModuleName} teacher model implementation."""

from typing import Any, Dict, List, Optional

from dvas.models.teacher.base import BaseTeacherModel
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Teacher(BaseTeacherModel):
    """{ModuleName} teacher model."""

    def __init__(self, model_name: str = "{module_name}", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

    async def annotate_frame_batch(
        self,
        frames: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Annotate a batch of frames."""
        raise NotImplementedError("Implement annotate_frame_batch")

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {{
            "name": self.model_name,
            "type": "teacher",
            "provider": "{module_name}",
        }}
''',
        },
    ),
    "pipeline": ScaffoldTemplate(
        name="pipeline",
        description="New pipeline stage",
        files={
            "__init__.py": '''"""{module_name} pipeline stage."""

from dvas.pipeline.{module_name}.stage import {ModuleName}Stage

__all__ = ["{ModuleName}Stage"]
''',
            "stage.py": '''"""{ModuleName} pipeline stage implementation."""

from typing import Any, Dict, Optional

from dvas.pipeline.core import PipelineStage
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class {ModuleName}Stage(PipelineStage):
    """{ModuleName} pipeline stage."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {{}}

    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process data through this stage."""
        logger.info("{module_name}_stage_processing")
        raise NotImplementedError("Implement process method")

    @property
    def name(self) -> str:
        return "{module_name}"

    @property
    def stage_type(self) -> str:
        return "transform"
''',
        },
    ),
    "test": ScaffoldTemplate(
        name="test",
        description="New test suite",
        files={
            "{module_name}.py": '''"""Tests for {module_name}."""

import pytest

from dvas.{module_name} import {ModuleName}Processor


class Test{ModuleName}Processor:
    """Test {module_name} processor."""

    def test_init(self) -> None:
        """Test processor initialization."""
        processor = {ModuleName}Processor()
        assert processor.config == {{}}

    def test_init_with_config(self) -> None:
        """Test processor with config."""
        processor = {ModuleName}Processor(config={{"key": "value"}})
        assert processor.config["key"] == "value"
''',
        },
    ),
}


@app.command()
def scaffold(
    template: str = typer.Argument(..., help="Template type (module/model/pipeline/test)"),
    name: str = typer.Argument(..., help="Module name (snake_case)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be created"),
) -> None:
    """Generate code scaffolding from templates."""
    if template not in SCAFFOLD_TEMPLATES:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available: {', '.join(SCAFFOLD_TEMPLATES.keys())}")
        raise typer.Exit(1)

    tmpl = SCAFFOLD_TEMPLATES[template]
    module_name = name.lower().replace(" ", "_").replace("-", "_")
    ModuleName = "".join(word.capitalize() for word in module_name.split("_"))

    # Determine output directory
    if output_dir:
        base_dir = output_dir
    elif template == "test":
        base_dir = settings.PROJECT_ROOT / "tests"
    elif template == "model":
        base_dir = settings.PROJECT_ROOT / "src" / "dvas" / "models" / "teacher"
    elif template == "pipeline":
        base_dir = settings.PROJECT_ROOT / "src" / "dvas" / "pipeline"
    else:
        base_dir = settings.PROJECT_ROOT / "src" / "dvas"

    target_dir = base_dir / module_name

    # Show plan
    tree = Tree(f"[bold cyan]{target_dir}[/bold cyan]")
    for filename in tmpl.files.keys():
        tree.add(f"[green]{filename}[/green]")

    console.print(Panel(tree, title=f"Scaffold: {template} '{name}'", border_style="blue"))

    if dry_run:
        console.print("[yellow]Dry run - no files created[/yellow]")
        return

    # Create directory and files
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename, content_template in tmpl.files.items():
        content = content_template.format(
            module_name=module_name,
            ModuleName=ModuleName,
        )
        file_path = target_dir / filename
        file_path.write_text(content, encoding="utf-8")
        console.print(f"  [green]Created[/green] {file_path}")

    console.print(f"\n[green]Scaffold '{name}' created successfully![/green]")
    console.print(f"  Location: {target_dir}")
    console.print(f"  Files: {len(tmpl.files)}")


@app.command(name="scaffold-list")
def scaffold_list() -> None:
    """List available scaffolding templates."""
    table = Table(title="Available Scaffolding Templates")
    table.add_column("Template", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Files", style="blue")

    for name, tmpl in SCAFFOLD_TEMPLATES.items():
        table.add_row(name, tmpl.description, ", ".join(tmpl.files.keys()))

    console.print(table)


__all__ = ["app", "scaffold", "scaffold_list", "ScaffoldTemplate", "SCAFFOLD_TEMPLATES"]
