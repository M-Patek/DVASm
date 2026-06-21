"""Tests for Terraform infrastructure management.

Tests for TerraformManager, TerraformVariable, TerraformOutput,
TerraformModule, TerraformState, and TerraformWorkspace.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dvas.infrastructure.terraform_manager import (
    TerraformManager,
    TerraformModule,
    TerraformOutput,
    TerraformProvider,
    TerraformState,
    TerraformVariable,
    TerraformWorkspace,
)


class TestTerraformVariable:
    """Test TerraformVariable dataclass."""

    def test_basic_variable(self):
        """Test basic variable HCL generation."""
        var = TerraformVariable(name="region", type="string", description="AWS region")
        hcl = var.to_hcl()
        assert 'variable "region" {' in hcl
        assert "type        = string" in hcl
        assert 'description = "AWS region"' in hcl

    def test_variable_with_default(self):
        """Test variable with default value."""
        var = TerraformVariable(name="count", type="number", default=2)
        hcl = var.to_hcl()
        assert "default     = 2" in hcl

    def test_variable_sensitive(self):
        """Test sensitive variable."""
        var = TerraformVariable(name="api_key", type="string", sensitive=True)
        hcl = var.to_hcl()
        assert "sensitive   = true" in hcl

    def test_variable_no_description(self):
        """Test variable without description."""
        var = TerraformVariable(name="foo")
        hcl = var.to_hcl()
        assert "description" not in hcl


class TestTerraformOutput:
    """Test TerraformOutput dataclass."""

    def test_basic_output(self):
        """Test basic output HCL generation."""
        out = TerraformOutput(name="endpoint", value="aws_lb.main.dns_name")
        hcl = out.to_hcl()
        assert 'output "endpoint" {' in hcl
        assert "value = aws_lb.main.dns_name" in hcl

    def test_output_with_description(self):
        """Test output with description."""
        out = TerraformOutput(
            name="endpoint",
            value="aws_lb.main.dns_name",
            description="Load balancer endpoint",
        )
        hcl = out.to_hcl()
        assert 'description = "Load balancer endpoint"' in hcl

    def test_output_sensitive(self):
        """Test sensitive output."""
        out = TerraformOutput(name="password", value="var.password", sensitive=True)
        hcl = out.to_hcl()
        assert "sensitive = true" in hcl


class TestTerraformModule:
    """Test TerraformModule dataclass."""

    def test_basic_module(self):
        """Test basic module HCL generation."""
        mod = TerraformModule(name="vpc", source="terraform-aws-modules/vpc/aws")
        hcl = mod.to_hcl()
        assert 'module "vpc" {' in hcl
        assert 'source = "terraform-aws-modules/vpc/aws"' in hcl

    def test_module_with_version(self):
        """Test module with version."""
        mod = TerraformModule(name="vpc", source="terraform-aws-modules/vpc/aws", version="3.0")
        hcl = mod.to_hcl()
        assert 'version = "3.0"' in hcl

    def test_module_with_variables(self):
        """Test module with variables."""
        mod = TerraformModule(
            name="vpc",
            source="terraform-aws-modules/vpc/aws",
            variables={"cidr": "10.0.0.0/16"},
        )
        hcl = mod.to_hcl()
        assert '"10.0.0.0/16"' in hcl


class TestTerraformState:
    """Test TerraformState dataclass."""

    def test_local_state(self):
        """Test local state backend."""
        state = TerraformState(backend="local")
        hcl = state.to_hcl()
        assert 'backend "local"' in hcl

    def test_s3_state(self):
        """Test S3 state backend."""
        state = TerraformState(
            backend="s3",
            config={"bucket": "my-bucket", "key": "terraform.tfstate"},
        )
        hcl = state.to_hcl()
        assert 'backend "s3"' in hcl
        assert '"my-bucket"' in hcl


class TestTerraformWorkspace:
    """Test TerraformWorkspace dataclass."""

    def test_workspace_creation(self):
        """Test workspace creation."""
        ws = TerraformWorkspace(name="dev", current=True)
        assert ws.name == "dev"
        assert ws.current is True

    def test_workspace_not_current(self):
        """Test non-current workspace."""
        ws = TerraformWorkspace(name="prod")
        assert ws.current is False


class TestTerraformManager:
    """Test TerraformManager class."""

    @pytest.fixture
    def manager(self):
        """Create a TerraformManager instance with a temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TerraformManager(working_dir=Path(tmpdir))

    def test_manager_init(self, manager: TerraformManager):
        """Test manager initialization."""
        assert manager.working_dir.exists()
        assert manager.current_workspace == "default"

    def test_add_variable(self, manager: TerraformManager):
        """Test adding a variable."""
        var = TerraformVariable(name="region", type="string")
        result = manager.add_variable(var)
        assert result is manager  # chaining
        assert "region" in manager._variables

    def test_add_output(self, manager: TerraformManager):
        """Test adding an output."""
        out = TerraformOutput(name="endpoint", value="lb.dns_name")
        result = manager.add_output(out)
        assert result is manager  # chaining
        assert "endpoint" in manager._outputs

    def test_add_module(self, manager: TerraformManager):
        """Test adding a module."""
        mod = TerraformModule(name="vpc", source="./modules/vpc")
        result = manager.add_module(mod)
        assert result is manager  # chaining
        assert len(manager._modules) == 1

    def test_set_state(self, manager: TerraformManager):
        """Test setting state configuration."""
        state = TerraformState(backend="s3", config={"bucket": "my-bucket"})
        result = manager.set_state(state)
        assert result is manager  # chaining
        assert manager._state is not None

    def test_write_variables_file(self, manager: TerraformManager):
        """Test writing variables file."""
        vars_path = manager.working_dir / "terraform.tfvars"
        manager.write_variables_file({"region": "us-east-1", "count": 2}, vars_path)
        assert vars_path.exists()
        content = vars_path.read_text()
        assert "region" in content
        assert "count" in content

    def test_generate_provider_config_aws(self, manager: TerraformManager):
        """Test AWS provider config generation."""
        hcl = manager.generate_provider_config(TerraformProvider.AWS, region="us-west-2")
        assert 'provider "aws"' in hcl
        assert 'region = "us-west-2"' in hcl
        assert "hashicorp/aws" in hcl

    def test_generate_provider_config_gcp(self, manager: TerraformManager):
        """Test GCP provider config generation."""
        hcl = manager.generate_provider_config(TerraformProvider.GCP, region="us-central1")
        assert 'provider "google"' in hcl
        assert "hashicorp/google" in hcl

    def test_generate_provider_config_azure(self, manager: TerraformManager):
        """Test Azure provider config generation."""
        hcl = manager.generate_provider_config(TerraformProvider.AZURE)
        assert 'provider "azurerm"' in hcl
        assert "features" in hcl

    def test_generate_aws_module(self, manager: TerraformManager):
        """Test AWS module generation."""
        hcl = manager.generate_aws_module()
        assert 'module "dvas-aws"' in hcl
        assert "g4dn.xlarge" in hcl

    def test_generate_gcp_module(self, manager: TerraformManager):
        """Test GCP module generation."""
        hcl = manager.generate_gcp_module()
        assert 'module "dvas-gcp"' in hcl
        assert "nvidia-tesla-t4" in hcl

    def test_generate_azure_module(self, manager: TerraformManager):
        """Test Azure module generation."""
        hcl = manager.generate_azure_module()
        assert 'module "dvas-azure"' in hcl
        assert "Standard_NC4as_T4_v3" in hcl

    def test_write_main_tf(self, manager: TerraformManager):
        """Test writing main.tf."""
        manager.add_variable(TerraformVariable(name="region"))
        manager.add_output(TerraformOutput(name="endpoint", value="test"))
        manager.write_main_tf()
        main_tf = manager.working_dir / "main.tf"
        assert main_tf.exists()
        content = main_tf.read_text()
        assert "variable" in content
        assert "output" in content

    @patch("subprocess.run")
    def test_terraform_init(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform init."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.init()
        assert result.returncode == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "init" in args

    @patch("subprocess.run")
    def test_plan(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform plan."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.plan()
        assert result.returncode == 0
        args = mock_run.call_args[0][0]
        assert "plan" in args

    @patch("subprocess.run")
    def test_plan_with_targets(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform plan with targets."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.plan(targets=["aws_instance.main"])
        args = mock_run.call_args[0][0]
        assert "-target" in args
        assert "aws_instance.main" in args

    @patch("subprocess.run")
    def test_apply(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform apply."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.apply(auto_approve=True)
        assert result.returncode == 0
        args = mock_run.call_args[0][0]
        assert "apply" in args
        assert "-auto-approve" in args

    @patch("subprocess.run")
    def test_destroy(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform destroy."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.destroy(auto_approve=True)
        assert result.returncode == 0
        args = mock_run.call_args[0][0]
        assert "destroy" in args

    @patch("subprocess.run")
    def test_validate(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform validate."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.validate()
        assert result.returncode == 0
        args = mock_run.call_args[0][0]
        assert "validate" in args

    @patch("subprocess.run")
    def test_fmt(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform fmt."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.fmt()
        assert result.returncode == 0
        args = mock_run.call_args[0][0]
        assert "fmt" in args
        assert "-recursive" in args

    @patch("subprocess.run")
    def test_fmt_check(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform fmt check."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.fmt(check=True)
        args = mock_run.call_args[0][0]
        assert "-check" in args

    @patch("subprocess.run")
    def test_get_outputs(self, mock_run: MagicMock, manager: TerraformManager):
        """Test getting outputs."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"endpoint": {"value": "http://example.com"}}',
        )
        outputs = manager.get_outputs()
        assert "endpoint" in outputs

    @patch("subprocess.run")
    def test_get_outputs_empty(self, mock_run: MagicMock, manager: TerraformManager):
        """Test getting outputs with empty result."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        outputs = manager.get_outputs()
        assert outputs == {}

    @patch("subprocess.run")
    def test_get_output(self, mock_run: MagicMock, manager: TerraformManager):
        """Test getting a specific output."""
        mock_run.return_value = MagicMock(returncode=0, stdout='"http://example.com"')
        value = manager.get_output("endpoint")
        assert value == "http://example.com"

    @patch("subprocess.run")
    def test_show(self, mock_run: MagicMock, manager: TerraformManager):
        """Test terraform show."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"version": 4}')
        state = manager.show()
        assert state == {"version": 4}

    @patch("subprocess.run")
    def test_list_workspaces(self, mock_run: MagicMock, manager: TerraformManager):
        """Test listing workspaces."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="  default\n* dev\n  prod\n",
        )
        workspaces = manager.list_workspaces()
        assert len(workspaces) == 3
        names = [ws.name for ws in workspaces]
        assert "default" in names
        assert "dev" in names
        assert "prod" in names
        # Check current workspace
        current = [ws for ws in workspaces if ws.current]
        assert len(current) == 1
        assert current[0].name == "dev"

    @patch("subprocess.run")
    def test_select_workspace(self, mock_run: MagicMock, manager: TerraformManager):
        """Test selecting workspace."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.select_workspace("dev")
        assert result is True
        assert manager.current_workspace == "dev"

    @patch("subprocess.run")
    def test_create_workspace(self, mock_run: MagicMock, manager: TerraformManager):
        """Test creating workspace."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.create_workspace("staging")
        assert result is True
        assert manager.current_workspace == "staging"

    @patch("subprocess.run")
    def test_delete_workspace(self, mock_run: MagicMock, manager: TerraformManager):
        """Test deleting workspace."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = manager.delete_workspace("old")
        assert result is True

    @patch("subprocess.run")
    def test_init_failure(self, mock_run: MagicMock, manager: TerraformManager):
        """Test failed terraform init."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = manager.init()
        assert result.returncode == 1

    def test_last_plan_success_no_plan(self, manager: TerraformManager):
        """Test last_plan_success with no plan."""
        assert manager.last_plan_success is None

    @patch("subprocess.run")
    def test_plan_success_tracking(self, mock_run: MagicMock, manager: TerraformManager):
        """Test plan success tracking."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.plan()
        assert manager.last_plan_success is True

    @patch("subprocess.run")
    def test_plan_failure_tracking(self, mock_run: MagicMock, manager: TerraformManager):
        """Test plan failure tracking."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        manager.plan()
        assert manager.last_plan_success is False

    def test_chaining(self, manager: TerraformManager):
        """Test method chaining."""
        result = (
            manager.add_variable(TerraformVariable(name="region"))
            .add_output(TerraformOutput(name="endpoint", value="test"))
            .set_state(TerraformState(backend="local"))
        )
        assert result is manager
        assert len(manager._variables) == 1
        assert len(manager._outputs) == 1
        assert manager._state is not None
