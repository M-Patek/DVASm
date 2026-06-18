from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from jinja2 import Environment, FileSystemLoader, Template

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    Template = None  # type: ignore

from dvas.utils.logging import get_logger

logger = get_logger(__name__)

# Package-level cache for templates
_template_cache: Dict[str, Any] = {}


def get_prompts_dir() -> Path:
    """Return directory containing prompt templates."""
    return Path(__file__).parent


@functools.lru_cache(maxsize=32)
def get_jinja_env() -> Optional[Any]:
    """Get cached Jinja2 environment.

    Returns None if jinja2 is not installed.
    """
    if not HAS_JINJA2:
        return None

    prompts_dir = get_prompts_dir()
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """Render a prompt template with given context.

    Args:
        template_name: Name of the template file (e.g., 'base_tasks.j2')
        **kwargs: Template variables

    Returns:
        Rendered prompt string

    Raises:
        FileNotFoundError: If template doesn't exist and no fallback provided
    """
    env = get_jinja_env()

    if env is None:
        # Fallback: use hardcoded simple prompts
        return _fallback_prompt(kwargs.get("task", "caption"), **kwargs)

    try:
        template = env.get_template(template_name)
        return template.render(**kwargs)
    except Exception as e:
        logger.warning("template_render_failed", template=template_name, error=str(e))
        task = kwargs.pop("task", "caption")
        return _fallback_prompt(task, **kwargs)


def _fallback_prompt(task: str, **kwargs: Any) -> str:
    """Fallback prompts when Jinja2 is not available."""
    # Remove 'task' from kwargs to avoid duplicate argument
    kwargs.pop("task", None)
    prompts: Dict[str, str] = {
        "caption": "Describe what is happening in this video. Be concise but accurate.",
        "dense_caption": """Provide a detailed description of the video, including:
1. Overall scene and context
2. Sequential actions performed
3. Tools or objects being used

Format your response as a coherent paragraph.""",
        "qa": f"""Based on this video, generate {kwargs.get("num_questions", 5)} question-answer pairs:

Format:
Q: [Question about actions, objects, or sequence]
A: [Concise answer]

Cover different aspects: what is happening, how it's done, what tools are used.""",
        "temporal": """Analyze this video and identify distinct action segments.

For each segment, provide:
- Start and end time (in seconds)
- Action label (verb + noun)
- Brief description

Format as JSON-like structure.""",
        "fine_grained": """You are an expert in robotic manipulation and egocentric video understanding.

Provide a detailed analysis of this first-person video, focusing on:

1. **Scene Understanding**: Describe the environment and context
2. **Hand Actions**: Identify left and right hand movements
3. **Object Interactions**: List all objects touched/manipulated
4. **Temporal Sequence**: Break down the action into chronological steps

Output as JSON structure.""",
    }
    return prompts.get(task, prompts["caption"])


def list_templates() -> list[str]:
    """List available prompt templates."""
    prompts_dir = get_prompts_dir()
    if not prompts_dir.exists():
        return []

    return [f.name for f in prompts_dir.iterdir() if f.suffix in (".j2", ".jinja", ".txt")]


class PromptManager:
    """High-level interface for managing prompts."""

    # Mapping of task names to (template_name, default_vars)
    TASK_REGISTRY: Dict[str, tuple[str, Dict[str, Any]]] = {
        "caption": ("base_tasks.j2", {"task": "caption"}),
        "dense_caption": ("base_tasks.j2", {"task": "dense_caption"}),
        "qa": ("base_tasks.j2", {"task": "qa", "num_questions": 5}),
        "temporal": ("base_tasks.j2", {"task": "temporal"}),
        "fine_grained": ("base_tasks.j2", {"task": "fine_grained"}),
    }

    @classmethod
    def get_prompt(cls, task: str, **overrides: Any) -> str:
        """Get prompt for a registered task.

        Args:
            task: Task name (must be in TASK_REGISTRY)
            **overrides: Override default template variables

        Returns:
            Rendered prompt string
        """
        if task not in cls.TASK_REGISTRY:
            logger.warning("unknown_task", task=task, fallback="caption")
            task = "caption"

        template_name, defaults = cls.TASK_REGISTRY[task]
        context = {**defaults, **overrides}

        return render_prompt(template_name, **context)

    @classmethod
    def register_task(
        cls,
        task: str,
        template_name: str,
        default_vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new task or override existing.

        Args:
            task: Task identifier
            template_name: Template file to use
            default_vars: Default variables for this task
        """
        cls.TASK_REGISTRY[task] = (template_name, default_vars or {})
        logger.info("task_registered", task=task, template=template_name)
