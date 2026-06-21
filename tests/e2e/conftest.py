"""Shared fixtures for e2e tests."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide temporary directory with Windows-safe cleanup."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Windows-safe cleanup: ignore permission errors from locked files
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except (PermissionError, OSError):
        pass


@pytest.fixture
def temp_storage_dir(temp_dir):
    """Alias for temp_dir for backward compatibility."""
    return temp_dir
