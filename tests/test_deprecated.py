"""Tests for deprecated and legacy code.

Provides minimal coverage for deprecated modules.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExportCLI:
    """Test export CLI module."""

    def test_export_cli_import(self):
        """Test that export CLI can be imported."""
        from dvas.export import cli
        assert cli is not None
