"""DVAS CLI module.

Provides developer experience tools including:
- Development mode with hot reload
- Code scaffolding
- Database migrations
- Documentation generation
- Linting and formatting
- Test running
- Benchmarking
"""

from dvas.cli.commands import app, main

__all__ = ["app", "main"]
