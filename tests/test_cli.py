"""Tests for DVAS CLI developer tools and main CLI."""

from pathlib import Path

from typer.testing import CliRunner

from dvas.cli.commands import (
    DevModeWatcher,
    MigrationManager,
    SCAFFOLD_TEMPLATES,
    ScaffoldTemplate,
)

# Import main CLI app
from dvas.__main__ import app as main_app


runner = CliRunner()


class TestMainCLI:
    """Test main CLI entry points from __main__.py."""

    def test_main_help(self):
        """Test main CLI help."""
        result = runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "DVAS" in result.output

    def test_annotate_help(self):
        """Test annotate command help."""
        result = runner.invoke(main_app, ["annotate", "--help"])
        assert result.exit_code == 0
        assert "annotate" in result.output.lower()

    def test_annotate_missing_video(self):
        """Test annotate with non-existent video."""
        result = runner.invoke(main_app, ["annotate", "/nonexistent/video.mp4"])
        assert result.exit_code == 1

    def test_export_help(self):
        """Test export command help."""
        result = runner.invoke(main_app, ["export", "--help"])
        assert result.exit_code == 0
        assert "export" in result.output.lower()

    def test_stats_help(self):
        """Test stats command help."""
        result = runner.invoke(main_app, ["stats", "--help"])
        assert result.exit_code == 0


class TestDevModeWatcher:
    """Test development mode file watcher."""

    def test_watcher_init(self, tmp_path: Path) -> None:
        """Test watcher initialization."""
        callback_called = [False]

        def callback() -> None:
            callback_called[0] = True

        watcher = DevModeWatcher([tmp_path], callback)
        assert watcher.paths == [tmp_path]
        assert watcher.callback == callback
        assert not watcher._running

    def test_scan_files(self, tmp_path: Path) -> None:
        """Test file scanning."""
        # Create a Python file
        (tmp_path / "test.py").write_text("pass")

        watcher = DevModeWatcher([tmp_path], lambda: None)
        mtimes = watcher._scan_files()

        assert len(mtimes) == 1
        assert any("test.py" in k for k in mtimes)

    def test_check_no_changes(self, tmp_path: Path) -> None:
        """Test check with no changes."""
        (tmp_path / "test.py").write_text("pass")

        watcher = DevModeWatcher([tmp_path], lambda: None)
        # First check establishes baseline
        assert watcher.check()
        # Second check should find no changes
        assert not watcher.check()

    def test_check_with_changes(self, tmp_path: Path) -> None:
        """Test check detects changes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")

        watcher = DevModeWatcher([tmp_path], lambda: None)
        watcher.check()  # Establish baseline

        # Modify file
        import time

        time.sleep(0.1)
        test_file.write_text("changed")

        assert watcher.check()


class TestScaffoldTemplates:
    """Test code scaffolding templates."""

    def test_templates_exist(self) -> None:
        """Test that templates are defined."""
        assert "module" in SCAFFOLD_TEMPLATES
        assert "model" in SCAFFOLD_TEMPLATES
        assert "pipeline" in SCAFFOLD_TEMPLATES
        assert "test" in SCAFFOLD_TEMPLATES

    def test_template_structure(self) -> None:
        """Test template structure."""
        for name, tmpl in SCAFFOLD_TEMPLATES.items():
            assert tmpl.name == name
            assert tmpl.description
            assert len(tmpl.files) > 0

    def test_module_template_content(self) -> None:
        """Test module template content."""
        tmpl = SCAFFOLD_TEMPLATES["module"]
        assert "__init__.py" in tmpl.files
        assert "core.py" in tmpl.files
        assert "types.py" in tmpl.files

        # Check template variables are present
        content = tmpl.files["core.py"]
        assert "{module_name}" in content
        assert "{ModuleName}" in content


class TestMigrationManager:
    """Test database migration manager."""

    def test_init(self, tmp_path: Path) -> None:
        """Test migration manager initialization."""
        db_path = tmp_path / "test.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        assert manager.db_path == db_path
        assert manager.migrations_dir.exists()

    def test_get_applied_empty(self, tmp_path: Path) -> None:
        """Test getting applied migrations when empty."""
        db_path = tmp_path / "test.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        applied = manager.get_applied()
        assert applied == []

    def test_create_migration(self, tmp_path: Path) -> None:
        """Test creating a new migration."""
        db_path = tmp_path / "test.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        path = manager.create("test_migration")
        assert path.exists()
        assert path.suffix == ".sql"
        assert "test_migration" in path.name

        content = path.read_text()
        assert "BEGIN" in content
        assert "COMMIT" in content

    def test_apply_migration(self, tmp_path: Path) -> None:
        """Test applying a migration."""
        db_path = tmp_path / "test.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        # Create a migration that creates a table
        migration_file = tmp_path / "001_test.sql"
        migration_file.write_text("""
            CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);
        """)

        manager.apply(migration_file)

        # Verify table was created
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_get_pending(self, tmp_path: Path) -> None:
        """Test getting pending migrations."""
        db_path = tmp_path / "test.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        # Create a migration file
        migration_file = manager.migrations_dir / "001_test.sql"
        migration_file.write_text("SELECT 1;")

        pending = manager.get_pending()
        assert len(pending) == 1
        assert pending[0].name == "001_test.sql"


class TestCLICommands:
    """Test CLI command functions."""

    def test_scaffold_template_names(self) -> None:
        """Test that all scaffold templates have required fields."""
        for name, tmpl in SCAFFOLD_TEMPLATES.items():
            assert isinstance(tmpl, ScaffoldTemplate)
            assert tmpl.name == name
            assert isinstance(tmpl.files, dict)

    def test_dev_mode_watcher_callback(self, tmp_path: Path) -> None:
        """Test dev mode watcher callback."""
        callback_results = []

        def test_callback() -> None:
            callback_results.append(True)

        watcher = DevModeWatcher([tmp_path], test_callback)
        assert watcher.callback == test_callback

    def test_migration_manager_db_creation(self, tmp_path: Path) -> None:
        """Test that migration manager creates database."""
        db_path = tmp_path / "migrations" / "test.db"
        migrations_dir = tmp_path / "migrations"
        _manager = MigrationManager(db_path, migrations_dir=migrations_dir)
        assert db_path.exists()

    def test_scaffold_module_content(self) -> None:
        """Test module scaffold content formatting."""
        tmpl = SCAFFOLD_TEMPLATES["module"]
        core_content = tmpl.files["core.py"]

        # Format with test values
        formatted = core_content.format(
            module_name="test_module",
            ModuleName="TestModule",
        )

        assert "class TestModuleProcessor" in formatted
        assert "test_module" in formatted

    def test_scaffold_model_content(self) -> None:
        """Test model scaffold content formatting."""
        tmpl = SCAFFOLD_TEMPLATES["model"]
        model_content = tmpl.files["model.py"]

        formatted = model_content.format(
            module_name="my_model",
            ModuleName="MyModel",
        )

        assert "class MyModelTeacher" in formatted
        assert "my_model" in formatted

    def test_scaffold_pipeline_content(self) -> None:
        """Test pipeline scaffold content formatting."""
        tmpl = SCAFFOLD_TEMPLATES["pipeline"]
        stage_content = tmpl.files["stage.py"]

        formatted = stage_content.format(
            module_name="my_stage",
            ModuleName="MyStage",
        )

        assert "class MyStageStage" in formatted
        assert "my_stage" in formatted


class TestScaffoldIntegration:
    """Integration tests for scaffolding."""

    def test_scaffold_module_creation(self, tmp_path: Path) -> None:
        """Test creating a module scaffold."""
        tmpl = SCAFFOLD_TEMPLATES["module"]
        module_name = "test_module"
        ModuleName = "TestModule"

        target_dir = tmp_path / module_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename, content_template in tmpl.files.items():
            content = content_template.format(
                module_name=module_name,
                ModuleName=ModuleName,
            )
            file_path = target_dir / filename
            file_path.write_text(content, encoding="utf-8")

        # Verify files were created
        assert (target_dir / "__init__.py").exists()
        assert (target_dir / "core.py").exists()
        assert (target_dir / "types.py").exists()

        # Verify content
        core_content = (target_dir / "core.py").read_text()
        assert "class TestModuleProcessor" in core_content

    def test_scaffold_test_creation(self, tmp_path: Path) -> None:
        """Test creating a test scaffold."""
        tmpl = SCAFFOLD_TEMPLATES["test"]
        module_name = "my_feature"
        ModuleName = "MyFeature"

        target_dir = tmp_path / "tests"
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename, content_template in tmpl.files.items():
            content = content_template.format(
                module_name=module_name,
                ModuleName=ModuleName,
            )
            file_path = target_dir / filename.format(
                module_name=module_name,
                ModuleName=ModuleName,
            )
            file_path.write_text(content, encoding="utf-8")

        # Verify test file
        test_file = target_dir / "my_feature.py"
        assert test_file.exists()
        content = test_file.read_text()
        assert "class TestMyFeatureProcessor" in content


class TestMigrationIntegration:
    """Integration tests for migrations."""

    def test_full_migration_workflow(self, tmp_path: Path) -> None:
        """Test complete migration workflow."""
        db_path = tmp_path / "app.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        # Create migration
        migration_path = manager.create("add_users")
        assert migration_path.exists()

        # Modify migration to create a table
        migration_path.write_text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            INSERT INTO users (name) VALUES ('test');
        """)

        # Apply migration
        manager.apply(migration_path)

        # Verify
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM users WHERE id = 1")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test"
        conn.close()

        # Check applied migrations
        applied = manager.get_applied()
        assert len(applied) == 1
        assert "add_users" in applied[0].name

    def test_multiple_migrations(self, tmp_path: Path) -> None:
        """Test multiple sequential migrations."""
        db_path = tmp_path / "app.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        # Create two migrations
        m1 = manager.migrations_dir / "001_create_table.sql"
        m1.write_text("CREATE TABLE t1 (id INTEGER);")

        m2 = manager.migrations_dir / "002_add_column.sql"
        m2.write_text("ALTER TABLE t1 ADD COLUMN name TEXT;")

        # Apply both
        manager.apply(m1)
        manager.apply(m2)

        # Verify table structure
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(t1)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        assert "id" in columns
        assert "name" in columns

    def test_migration_idempotency(self, tmp_path: Path) -> None:
        """Test that applying same migration twice is handled."""
        db_path = tmp_path / "app.db"
        migrations_dir = tmp_path / "migrations"
        manager = MigrationManager(db_path, migrations_dir=migrations_dir)

        migration = manager.migrations_dir / "001_init.sql"
        migration.write_text("CREATE TABLE test (id INTEGER);")

        # Apply once
        manager.apply(migration)

        # Should not appear in pending
        pending = manager.get_pending()
        assert len(pending) == 0
