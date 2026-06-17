#!/usr/bin/env python3
"""Check and report on known gaps from status.yaml."""

import sys
from collections import Counter
from pathlib import Path

import yaml


def load_status(project_root: Path) -> dict:
    """Load status.yaml."""
    status_path = project_root / 'docs' / '_machine' / 'status.yaml'
    return yaml.safe_load(status_path.read_text(encoding='utf-8'))


def load_bugs(project_root: Path) -> dict:
    """Load bugs.yaml."""
    bugs_path = project_root / 'docs' / '_machine' / 'bugs.yaml'
    return yaml.safe_load(bugs_path.read_text(encoding='utf-8'))


def print_gaps(status: dict, severity_filter: str = None):
    """Print known gaps by subsystem."""
    print("\n=== Known Gaps ===\n")

    for subsystem in status.get('subsystems', []):
        gaps = subsystem.get('known_gaps', [])
        if severity_filter:
            gaps = [g for g in gaps if g.get('severity') == severity_filter]

        if gaps:
            print(f"[{subsystem['id']}] {subsystem['title']}")
            for gap in gaps:
                print(f"  • [{gap['severity'].upper()}] {gap['title']}")
                print(f"    Opened: {gap['opened']}, Next review: {gap.get('next_review', 'N/A')}")
                if 'notes' in gap:
                    print(f"    Notes: {gap['notes']}")
            print()


def print_stats(status: dict, bugs: dict):
    """Print statistics."""
    print("\n=== Statistics ===\n")

    # Gaps by severity
    all_gaps = []
    for sub in status.get('subsystems', []):
        all_gaps.extend(sub.get('known_gaps', []))

    severity_counts = Counter(g.get('severity') for g in all_gaps)
    print("Gaps by severity:")
    for sev in ['critical', 'high', 'medium', 'low']:
        if severity_counts[sev]:
            print(f"  {sev}: {severity_counts[sev]}")

    # Bugs
    bug_count = len(bugs.get('bugs', []))
    bonus_count = len(bugs.get('bonus_bugs', []))
    print(f"\nTotal bugs: {bug_count} (bonus: {bonus_count})")

    # Deliverables
    deliverables = status.get('deliverables', [])
    runnable = sum(1 for d in deliverables if d.get('status') == 'runnable')
    print(f"\nDeliverables: {runnable}/{len(deliverables)} runnable")


def main():
    project_root = Path(__file__).parent.parent
    status = load_status(project_root)
    bugs = load_bugs(project_root)

    if '--stats' in sys.argv:
        print_stats(status, bugs)
    else:
        severity = None
        if '--severity' in sys.argv:
            idx = sys.argv.index('--severity')
            severity = sys.argv[idx + 1]
        print_gaps(status, severity)


if __name__ == '__main__':
    main()
