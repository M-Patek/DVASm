"""Tests for deprecated and legacy code.

Provides minimal coverage for deprecated modules.
"""



class TestExportCLI:
    """Test export CLI module."""

    def test_export_cli_import(self):
        """Test that export CLI can be imported."""
        from dvas.export import cli
        assert cli is not None
