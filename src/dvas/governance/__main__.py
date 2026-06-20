"""CLI entry point for governance operations.

Usage:
    python -m dvas.governance standards list
    python -m dvas.governance standards validate EPIC-KITCHENS data.json
    python -m dvas.governance policy evaluate quality_threshold data.json
    python -m dvas.governance workflow stats
    python -m dvas.governance gates run annotation_quality data.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from dvas.governance.approval_workflow import ApprovalWorkflow
from dvas.governance.policy_engine import PolicyEngine
from dvas.governance.quality_gates import (
    GateStatus,
    QualityDimension,
    QualityGate,
    QualityGateRunner,
    QualityThreshold,
)
from dvas.governance.standards import StandardRegistry

app = typer.Typer(help="DVAS Governance CLI")

standards_app = typer.Typer(help="Annotation standards")
policy_app = typer.Typer(help="Policy engine")
workflow_app = typer.Typer(help="Approval workflows")
gates_app = typer.Typer(help="Quality gates")

app.add_typer(standards_app, name="standards")
app.add_typer(policy_app, name="policy")
app.add_typer(workflow_app, name="workflow")
app.add_typer(gates_app, name="gates")


@standards_app.command("list")
def list_standards_cmd() -> None:
    """List all registered annotation standards."""
    registry = StandardRegistry()
    names = registry.list_standards()
    for name in names:
        versions = registry.get_versions(name)
        typer.echo(f"{name}: {', '.join(versions)}")


@standards_app.command("validate")
def validate_standard(
    name: str = typer.Argument(..., help="Standard name"),
    data_file: Path = typer.Argument(..., help="JSON data file"),
    version: Optional[str] = typer.Option(None, help="Standard version"),
) -> None:
    """Validate data against an annotation standard."""
    registry = StandardRegistry()

    if not data_file.exists():
        typer.echo(f"Error: File not found: {data_file}", err=True)
        raise typer.Exit(1)

    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON: {e}", err=True)
        raise typer.Exit(1)

    errors = registry.validate_data(name, data, version)
    if errors:
        typer.echo(f"Validation failed for {name}:")
        for error in errors:
            typer.echo(f"  - {error}")
        raise typer.Exit(1)
    else:
        typer.echo(f"Data is valid against {name}")


@standards_app.command("compliance")
def check_compliance(
    name: str = typer.Argument(..., help="Standard name"),
    data_file: Path = typer.Argument(..., help="JSON data file"),
    version: Optional[str] = typer.Option(None, help="Standard version"),
) -> None:
    """Check compliance of data against a standard."""
    registry = StandardRegistry()

    if not data_file.exists():
        typer.echo(f"Error: File not found: {data_file}", err=True)
        raise typer.Exit(1)

    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON: {e}", err=True)
        raise typer.Exit(1)

    report = registry.check_compliance(name, data, version)
    typer.echo(json.dumps(report, indent=2))


@policy_app.command("evaluate")
def evaluate_policy(
    policy_id: str = typer.Argument(..., help="Policy ID"),
    data_file: Path = typer.Argument(..., help="JSON data file"),
) -> None:
    """Evaluate a policy against data."""
    engine = PolicyEngine()

    if not data_file.exists():
        typer.echo(f"Error: File not found: {data_file}", err=True)
        raise typer.Exit(1)

    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON: {e}", err=True)
        raise typer.Exit(1)

    try:
        result = engine.evaluate(policy_id, data)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(json.dumps(result.to_dict(), indent=2))


@policy_app.command("report")
def policy_report() -> None:
    """Generate a policy compliance report."""
    engine = PolicyEngine()
    report = engine.get_compliance_report()
    typer.echo(json.dumps(report, indent=2))


@workflow_app.command("stats")
def workflow_stats() -> None:
    """Show workflow statistics."""
    workflow = ApprovalWorkflow()
    stats = workflow.get_stats()
    typer.echo(json.dumps(stats, indent=2))


@gates_app.command("run")
def run_gate(
    gate_id: str = typer.Argument(..., help="Gate ID"),
    data_file: Path = typer.Argument(..., help="JSON scores file"),
) -> None:
    """Run a quality gate against scores."""
    runner = QualityGateRunner()

    if not data_file.exists():
        typer.echo(f"Error: File not found: {data_file}", err=True)
        raise typer.Exit(1)

    try:
        with open(data_file, "r") as f:
            raw_scores = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON: {e}", err=True)
        raise typer.Exit(1)

    # Convert string keys to QualityDimension
    scores: dict = {}
    for key, value in raw_scores.items():
        try:
            dim = QualityDimension(key)
            scores[dim] = float(value)
        except (ValueError, KeyError):
            pass

    # Create a default gate if not registered
    gate = QualityGate(
        gate_id=gate_id,
        thresholds=[
            QualityThreshold(QualityDimension.COMPLETENESS, min_value=0.0),
            QualityThreshold(QualityDimension.ACCURACY, min_value=0.0),
        ],
    )
    runner.register_gate(gate)

    result = runner.run(gate_id, "item_001", scores)
    typer.echo(json.dumps(result.to_dict(), indent=2))

    if result.status == GateStatus.FAIL:
        raise typer.Exit(1)


@gates_app.command("list")
def list_gates() -> None:
    """List registered quality gates."""
    runner = QualityGateRunner()
    gates = runner.list_gates()
    if gates:
        for gate_id in gates:
            typer.echo(gate_id)
    else:
        typer.echo("No gates registered")


if __name__ == "__main__":
    app()
