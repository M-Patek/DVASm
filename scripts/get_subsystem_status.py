#!/usr/bin/env python3
"""Quick subsystem status overview."""

from pathlib import Path

import yaml


def main():
    project_root = Path(__file__).parent.parent
    status_path = project_root / 'docs' / '_machine' / 'status.yaml'

    if not status_path.exists():
        print("No status.yaml found")
        return

    status = yaml.safe_load(status_path.read_text(encoding='utf-8'))

    print("\n=== Subsystem Status ===\n")
    print(f"{'ID':<10} {'Status':<12} {'Health':<8} {'Gaps':<6} Title")
    print("-" * 70)

    for sub in status.get('subsystems', []):
        gap_count = len(sub.get('known_gaps', []))
        print(f"{sub['id']:<10} {sub['status']:<12} {sub['health']:<8} {gap_count:<6} {sub['title'][:40]}")

    print("\n=== Deliverables ===\n")
    for deliv in status.get('deliverables', []):
        status_icon = "✓" if deliv.get('status') == 'runnable' else "○"
        print(f"{status_icon} {deliv['id']}: {deliv['title']}")


if __name__ == '__main__':
    main()
