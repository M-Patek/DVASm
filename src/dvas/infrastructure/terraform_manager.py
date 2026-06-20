"""Terraform infrastructure-as-code management for DVAS.

Provides operations for Terraform plan, apply, state management,
module generation, and workspace switching across AWS, GCP, and Azure.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class TerraformProvider(str, Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class TerraformVariable:
    """Terraform variable definition."""

    name: str
    type: str = "string"
    description: str = ""
    default: Optional[Any] = None
    sensitive: bool = False

    def to_hcl(self) -> str:
        """Generate HCL variable block."""
        lines = [f'variable "{self.name}" {{']
        lines.append(f"  type        = {self.type}")
        if self.description:
            lines.append(f'  description = "{self.description}"')
        if self.default is not None:
            lines.append(f"  default     = {json.dumps(self.default)}")
        if self.sensitive:
            lines.append("  sensitive   = true")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TerraformOutput:
    """Terraform output definition."""

    name: str
    value: str
    description: str = ""
    sensitive: bool = False

    def to_hcl(self) -> str:
        """Generate HCL output block."""
        lines = [f'output "{self.name}" {{']
        lines.append(f"  value = {self.value}")
        if self.description:
            lines.append(f'  description = "{self.description}"')
        if self.sensitive:
            lines.append("  sensitive = true")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TerraformModule:
    """Terraform module configuration."""

    name: str
    source: str
    version: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    providers: Dict[str, str] = field(default_factory=dict)

    def to_hcl(self) -> str:
        """Generate HCL module block."""
        lines = [f'module "{self.name}" {{']
        lines.append(f'  source = "{self.source}"')
        if self.version:
            lines.append(f'  version = "{self.version}"')
        for key, value in self.variables.items():
            lines.append(f"  {key} = {json.dumps(value)}")
        for key, value in self.providers.items():
            lines.append(f"  providers = {{ {key} = {value} }}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TerraformState:
    """Terraform state configuration."""

    backend: str = "local"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_hcl(self) -> str:
        """Generate HCL backend block."""
        lines = ["terraform {"]
        lines.append('  backend "{}" {{'.format(self.backend))
        for key, value in self.config.items():
            lines.append(f"    {key} = {json.dumps(value)}")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TerraformWorkspace:
    """Terraform workspace information."""

    name: str
    current: bool = False
    description: str = ""


class TerraformManager:
    """Manage Terraform infrastructure for DVAS.

    Usage::

        tf = TerraformManager(Path("./infra"))
        tf.init()
        tf.plan()
        if tf.plan_succeeds():
            tf.apply()
    """

    def __init__(
        self,
        working_dir: Path,
        terraform_cmd: str = "terraform",
    ) -> None:
        self.working_dir = working_dir
        self.terraform_cmd = terraform_cmd
        self._variables: Dict[str, TerraformVariable] = {}
        self._outputs: Dict[str, TerraformOutput] = {}
        self._modules: List[TerraformModule] = []
        self._state: Optional[TerraformState] = None
        self._current_workspace: str = "default"
        self._last_plan: Optional[Dict[str, Any]] = None

    def _run_cmd(
        self, args: List[str], input_text: Optional[str] = None, capture: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute a Terraform command."""
        cmd = [self.terraform_cmd] + args
        logger.debug("terraform_cmd", command=" ".join(cmd), cwd=str(self.working_dir))
        return subprocess.run(
            cmd,
            cwd=self.working_dir,
            input=input_text,
            capture_output=capture,
            text=capture,
            check=False,
        )

    def init(
        self, backend_config: Optional[Dict[str, str]] = None, upgrade: bool = False
    ) -> subprocess.CompletedProcess:
        """Run terraform init.

        Args:
            backend_config: Backend configuration overrides.
            upgrade: Upgrade modules.

        Returns:
            CompletedProcess result.
        """
        args = ["init"]
        if upgrade:
            args.append("-upgrade")
        if backend_config:
            for key, value in backend_config.items():
                args.extend(["-backend-config", f"{key}={value}"])
        result = self._run_cmd(args)
        if result.returncode == 0:
            logger.info("terraform_initialized", dir=str(self.working_dir))
        else:
            logger.error("terraform_init_failed", error=result.stderr)
        return result

    def plan(
        self,
        vars_file: Optional[Path] = None,
        targets: Optional[List[str]] = None,
        destroy: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run terraform plan.

        Args:
            vars_file: Path to variables file.
            targets: Resource targets.
            destroy: Plan for destruction.

        Returns:
            CompletedProcess result.
        """
        args = ["plan", "-out=plan.tfplan"]
        if vars_file:
            args.extend(["-var-file", str(vars_file)])
        if targets:
            for target in targets:
                args.extend(["-target", target])
        if destroy:
            args.append("-destroy")
        result = self._run_cmd(args)
        if result.returncode == 0:
            self._last_plan = {"success": True, "destroy": destroy}
            logger.info("terraform_plan_succeeded")
        else:
            self._last_plan = {"success": False, "error": result.stderr}
            logger.error("terraform_plan_failed", error=result.stderr)
        return result

    def apply(
        self, auto_approve: bool = False, plan_file: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run terraform apply.

        Args:
            auto_approve: Auto-approve without prompt.
            plan_file: Path to saved plan file.

        Returns:
            CompletedProcess result.
        """
        args = ["apply"]
        if auto_approve:
            args.append("-auto-approve")
        if plan_file:
            args.append(str(plan_file))
        else:
            args.append("plan.tfplan")
        result = self._run_cmd(args)
        if result.returncode == 0:
            logger.info("terraform_apply_succeeded")
        else:
            logger.error("terraform_apply_failed", error=result.stderr)
        return result

    def destroy(self, auto_approve: bool = False) -> subprocess.CompletedProcess:
        """Run terraform destroy.

        Args:
            auto_approve: Auto-approve without prompt.

        Returns:
            CompletedProcess result.
        """
        args = ["destroy"]
        if auto_approve:
            args.append("-auto-approve")
        result = self._run_cmd(args)
        if result.returncode == 0:
            logger.info("terraform_destroy_succeeded")
        else:
            logger.error("terraform_destroy_failed", error=result.stderr)
        return result

    def validate(self) -> subprocess.CompletedProcess:
        """Run terraform validate.

        Returns:
            CompletedProcess result.
        """
        result = self._run_cmd(["validate"])
        if result.returncode == 0:
            logger.info("terraform_validation_succeeded")
        else:
            logger.error("terraform_validation_failed", error=result.stderr)
        return result

    def fmt(self, check: bool = False) -> subprocess.CompletedProcess:
        """Run terraform fmt.

        Args:
            check: Check formatting without writing.

        Returns:
            CompletedProcess result.
        """
        args = ["fmt"]
        if check:
            args.append("-check")
        else:
            args.append("-recursive")
        return self._run_cmd(args)

    def get_outputs(self) -> Dict[str, Any]:
        """Get Terraform outputs.

        Returns:
            Dictionary of outputs.
        """
        result = self._run_cmd(["output", "-json"])
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {}

    def get_output(self, name: str) -> Optional[Any]:
        """Get a specific Terraform output.

        Args:
            name: Output name.

        Returns:
            Output value or None.
        """
        result = self._run_cmd(["output", "-json", name])
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return None

    def show(self) -> Dict[str, Any]:
        """Show current state.

        Returns:
            State dictionary.
        """
        result = self._run_cmd(["show", "-json"])
        if result.returncode == 0 and result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return {}

    def list_workspaces(self) -> List[TerraformWorkspace]:
        """List all workspaces.

        Returns:
            List of workspace objects.
        """
        result = self._run_cmd(["workspace", "list"])
        workspaces: List[TerraformWorkspace] = []
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                current = line.startswith("* ")
                name = line.lstrip("* ").strip()
                workspaces.append(
                    TerraformWorkspace(
                        name=name,
                        current=current,
                    )
                )
        return workspaces

    def select_workspace(self, name: str) -> bool:
        """Select a workspace.

        Args:
            name: Workspace name.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["workspace", "select", name])
        if result.returncode == 0:
            self._current_workspace = name
            logger.info("workspace_selected", name=name)
            return True
        return False

    def create_workspace(self, name: str) -> bool:
        """Create a new workspace.

        Args:
            name: Workspace name.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["workspace", "new", name])
        if result.returncode == 0:
            self._current_workspace = name
            logger.info("workspace_created", name=name)
            return True
        return False

    def delete_workspace(self, name: str) -> bool:
        """Delete a workspace.

        Args:
            name: Workspace name.

        Returns:
            True if successful.
        """
        result = self._run_cmd(["workspace", "delete", name])
        if result.returncode == 0:
            logger.info("workspace_deleted", name=name)
            return True
        return False

    def add_variable(self, variable: TerraformVariable) -> "TerraformManager":
        """Add a variable definition.

        Args:
            variable: Variable to add.

        Returns:
            Self for chaining.
        """
        self._variables[variable.name] = variable
        return self

    def add_output(self, output: TerraformOutput) -> "TerraformManager":
        """Add an output definition.

        Args:
            output: Output to add.

        Returns:
            Self for chaining.
        """
        self._outputs[output.name] = output
        return self

    def add_module(self, module: TerraformModule) -> "TerraformManager":
        """Add a module.

        Args:
            module: Module to add.

        Returns:
            Self for chaining.
        """
        self._modules.append(module)
        return self

    def set_state(self, state: TerraformState) -> "TerraformManager":
        """Set state configuration.

        Args:
            state: State configuration.

        Returns:
            Self for chaining.
        """
        self._state = state
        return self

    def write_variables_file(self, variables: Dict[str, Any], path: Path) -> None:
        """Write a terraform.tfvars file.

        Args:
            variables: Variable values.
            path: Output path.
        """
        lines: List[str] = []
        for key, value in variables.items():
            lines.append(f"{key} = {json.dumps(value)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("variables_file_written", path=str(path))

    def generate_provider_config(
        self,
        provider: TerraformProvider,
        region: str = "us-east-1",
        version_constraint: str = "~> 5.0",
    ) -> str:
        """Generate provider configuration.

        Args:
            provider: Cloud provider.
            region: Default region.
            version_constraint: Provider version constraint.

        Returns:
            HCL string.
        """
        lines = ["terraform {"]
        lines.append("  required_providers {")
        if provider == TerraformProvider.AWS:
            lines.append(
                f'    aws = {{\n      source  = "hashicorp/aws"\n      version = "{version_constraint}"\n    }}'
            )
        elif provider == TerraformProvider.GCP:
            lines.append(
                f'    google = {{\n      source  = "hashicorp/google"\n      version = "{version_constraint}"\n    }}'
            )
        elif provider == TerraformProvider.AZURE:
            lines.append(
                f'    azurerm = {{\n      source  = "hashicorp/azurerm"\n      version = "{version_constraint}"\n    }}'
            )
        lines.append("  }")
        lines.append("}")
        lines.append("")
        if provider == TerraformProvider.AWS:
            lines.append(f'provider "aws" {{\n  region = "{region}"\n}}')
        elif provider == TerraformProvider.GCP:
            lines.append(f'provider "google" {{\n  region = "{region}"\n}}')
        elif provider == TerraformProvider.AZURE:
            lines.append('provider "azurerm" {\n  features {}\n}')
        return "\n".join(lines)

    def generate_aws_module(self, module_name: str = "dvas-aws") -> str:
        """Generate a standard AWS module for DVAS.

        Args:
            module_name: Module name.

        Returns:
            HCL string.
        """
        lines = [f'module "{module_name}" {{']
        lines.append('  source = "./modules/aws"')
        lines.append('  vpc_cidr = "10.0.0.0/16"')
        lines.append("  az_count = 2")
        lines.append("  enable_gpu = true")
        lines.append('  instance_type = "g4dn.xlarge"')
        lines.append("}")
        return "\n".join(lines)

    def generate_gcp_module(self, module_name: str = "dvas-gcp") -> str:
        """Generate a standard GCP module for DVAS.

        Args:
            module_name: Module name.

        Returns:
            HCL string.
        """
        lines = [f'module "{module_name}" {{']
        lines.append('  source = "./modules/gcp"')
        lines.append("  project_id = var.gcp_project_id")
        lines.append("  region = var.gcp_region")
        lines.append("  enable_gpu = true")
        lines.append('  machine_type = "n1-standard-4"')
        lines.append('  accelerator_type = "nvidia-tesla-t4"')
        lines.append("}")
        return "\n".join(lines)

    def generate_azure_module(self, module_name: str = "dvas-azure") -> str:
        """Generate a standard Azure module for DVAS.

        Args:
            module_name: Module name.

        Returns:
            HCL string.
        """
        lines = [f'module "{module_name}" {{']
        lines.append('  source = "./modules/azure"')
        lines.append("  resource_group_name = var.azurerm_resource_group")
        lines.append("  location = var.azurerm_location")
        lines.append("  enable_gpu = true")
        lines.append('  vm_size = "Standard_NC4as_T4_v3"')
        lines.append("}")
        return "\n".join(lines)

    def write_main_tf(self, path: Optional[Path] = None) -> None:
        """Write a main.tf file from current configuration.

        Args:
            path: Output path. Defaults to working_dir/main.tf.
        """
        if path is None:
            path = self.working_dir / "main.tf"

        lines: List[str] = []
        if self._state:
            lines.append(self._state.to_hcl())
            lines.append("")

        for var in self._variables.values():
            lines.append(var.to_hcl())
            lines.append("")

        for mod in self._modules:
            lines.append(mod.to_hcl())
            lines.append("")

        for out in self._outputs.values():
            lines.append(out.to_hcl())
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("main_tf_written", path=str(path))

    @property
    def current_workspace(self) -> str:
        """Get the current workspace name."""
        return self._current_workspace

    @property
    def last_plan_success(self) -> Optional[bool]:
        """Get whether the last plan succeeded."""
        if self._last_plan:
            return self._last_plan.get("success")
        return None
