"""Development mode watcher for DVAS CLI.

Provides file watching and hot reload for development mode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.console import Console

from dvas.config import settings

console = Console()


class DevModeWatcher:
    """File watcher for development hot reload."""

    def __init__(self, paths: List[Path], callback: Callable[[], None]) -> None:
        self.paths = paths
        self.callback = callback
        self._mtimes: Dict[str, float] = {}
        self._running = False

    def _get_mtime(self, path: Path) -> float:
        """Get modification time of a file."""
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _scan_files(self) -> Dict[str, float]:
        """Scan all watched paths for Python files."""
        mtimes: Dict[str, float] = {}
        for path in self.paths:
            if path.is_file() and path.suffix == ".py":
                mtimes[str(path)] = self._get_mtime(path)
            elif path.is_dir():
                for py_file in path.rglob("*.py"):
                    mtimes[str(py_file)] = self._get_mtime(py_file)
        return mtimes

    def check(self) -> bool:
        """Check if any files have changed. Returns True if reload needed."""
        current = self._scan_files()
        changed = False

        for filepath, mtime in current.items():
            if filepath not in self._mtimes or self._mtimes[filepath] != mtime:
                changed = True
                break

        self._mtimes = current
        return changed

    def run(self, interval: float = 1.0) -> None:
        """Run the watcher loop."""
        self._mtimes = self._scan_files()
        self._running = True

        console.print("[green]Dev mode started. Watching for changes...[/green]")
        console.print(f"  Watching: {[str(p) for p in self.paths]}")
        console.print("  Press Ctrl+C to stop\n")

        while self._running:
            try:
                time.sleep(interval)
                if self.check():
                    console.print("[yellow]Changes detected, reloading...[/yellow]")
                    self.callback()
                    console.print("[green]Reload complete.[/green]\n")
            except KeyboardInterrupt:
                self._running = False
                console.print("\n[blue]Dev mode stopped.[/blue]")

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
