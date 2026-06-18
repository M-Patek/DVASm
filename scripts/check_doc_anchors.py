#!/usr/bin/env python3
"""Check that code_anchors in subsystem docs point to valid code locations."""

import os
import re
import sys
from pathlib import Path

import yaml


def extract_code_anchors(md_path: Path) -> list[str]:
    """Extract code_anchors from markdown frontmatter."""
    content = md_path.read_text(encoding="utf-8")

    # Find YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return []

    try:
        frontmatter = yaml.safe_load(match.group(1))
        return frontmatter.get('code_anchors', [])
    except yaml.YAMLError:
        return []


def check_anchor(project_root: Path, anchor: str) -> tuple[bool, str]:
    """Check if a code anchor is valid."""
    # Format: "path/to/file.py:ClassName" or "path/to/file.py:function_name"
    if ':' not in anchor:
        return False, f"Invalid format: {anchor}"

    file_path, symbol = anchor.split(':', 1)
    full_path = project_root / file_path

    if not full_path.exists():
        return False, f"File not found: {file_path}"

    content = full_path.read_text(encoding='utf-8')

    # Check for class definition
    if symbol[0].isupper():
        pattern = rf'^class\s+{re.escape(symbol)}\b'
    else:
        # Function or method (including async)
        pattern = rf'^(?:async\s+)?def\s+{re.escape(symbol)}\b'

    if re.search(pattern, content, re.MULTILINE):
        return True, "OK"

    # Also check for method in class (ClassName.method_name)
    if '.' in symbol:
        cls, method = symbol.split('.', 1)
        pattern = rf'^\s+def\s+{re.escape(method)}\b'
        if re.search(pattern, content, re.MULTILINE):
            return True, "OK"

    return False, f"Symbol '{symbol}' not found in {file_path}"


def main():
    # Use GitHub Actions workspace if available, otherwise use script location
    if 'GITHUB_WORKSPACE' in os.environ:
        project_root = Path(os.environ['GITHUB_WORKSPACE'])
    else:
        project_root = Path(__file__).parent.parent

    # Debug: print project root and list files
    print(f"Project root: {project_root}", file=sys.stderr)
    print(f"Project root exists: {project_root.exists()}", file=sys.stderr)

    subsystems_dir = project_root / 'docs' / 'subsystems'

    if not subsystems_dir.exists():
        print("No subsystems directory found")
        sys.exit(0)

    all_ok = True

    for md_file in subsystems_dir.glob('*.md'):
        anchors = extract_code_anchors(md_file)
        if not anchors:
            continue

        print(f"\n{md_file.name}:")
        for anchor in anchors:
            ok, msg = check_anchor(project_root, anchor)
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {anchor}: {msg}")
            if not ok:
                all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
