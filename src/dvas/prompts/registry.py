"""Prompt registry with CRUD operations for prompt templates.

Provides versioned storage, lineage tracking, and metadata management
for prompt templates used across the DVAS annotation pipeline.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from dvas.utils.logging import get_logger

logger = get_logger(__name__)


class PromptDomain(str, Enum):
    """Domain categories for prompt templates."""

    KITCHEN = "kitchen"
    ROBOT = "robot"
    GENERAL = "general"
    MEDICAL = "medical"
    SPORTS = "sports"
    ASSEMBLY = "assembly"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    HUMAN_REVIEW = "human_review"


@dataclass
class PromptMetadata:
    """Metadata for a prompt template."""

    name: str
    version: str
    domain: PromptDomain
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain.value,
            "description": self.description,
            "tags": self.tags.copy(),
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            domain=PromptDomain(data.get("domain", "general")),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            author=data.get("author", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            parent_id=data.get("parent_id"),
        )


@dataclass
class PromptTemplate:
    """A versioned prompt template with metadata and lineage."""

    id: str
    metadata: PromptMetadata
    template: str
    variables: List[str] = field(default_factory=list)
    avg_quality_score: float = 0.0
    usage_count: int = 0
    lineage: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fields."""
        self.avg_quality_score = max(0.0, min(1.0, self.avg_quality_score))

    @property
    def hash(self) -> str:
        """Compute hash of template content."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    def update_quality(self, score: float) -> None:
        """Update moving average quality score."""
        score = max(0.0, min(1.0, score))
        n = self.usage_count
        self.avg_quality_score = (self.avg_quality_score * n + score) / (n + 1)
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "template": self.template,
            "variables": self.variables.copy(),
            "avg_quality_score": self.avg_quality_score,
            "usage_count": self.usage_count,
            "lineage": self.lineage.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            metadata=PromptMetadata.from_dict(data["metadata"]),
            template=data["template"],
            variables=data.get("variables", []),
            avg_quality_score=data.get("avg_quality_score", 0.0),
            usage_count=data.get("usage_count", 0),
            lineage=data.get("lineage", []),
        )


class PromptRegistry:
    """Registry for prompt templates with CRUD and lineage tracking."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        self._lineage: Dict[str, List[str]] = {}
        self._domain_index: Dict[PromptDomain, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

    def create(
        self,
        name: str,
        template: str,
        domain: PromptDomain = PromptDomain.GENERAL,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
    ) -> PromptTemplate:
        """Create a new prompt template."""
        prompt_id = str(uuid.uuid4())
        metadata = PromptMetadata(
            name=name,
            version=version,
            domain=domain,
            description=description,
            tags=tags or [],
            parent_id=parent_id,
        )

        lineage: List[str] = []
        if parent_id and parent_id in self._templates:
            parent = self._templates[parent_id]
            lineage = parent.lineage + [parent_id]

        prompt = PromptTemplate(
            id=prompt_id,
            metadata=metadata,
            template=template,
            variables=variables or [],
            lineage=lineage,
        )

        self._templates[prompt_id] = prompt
        self._lineage[prompt_id] = lineage.copy()

        if domain not in self._domain_index:
            self._domain_index[domain] = set()
        self._domain_index[domain].add(prompt_id)

        if tags:
            for tag in tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(prompt_id)

        logger.info(
            "prompt_created",
            prompt_id=prompt_id,
            name=name,
            version=version,
            domain=domain.value,
        )

        return prompt

    def get(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get a prompt template by ID."""
        return self._templates.get(prompt_id)

    def get_by_name(self, name: str) -> List[PromptTemplate]:
        """Get all prompt templates with a given name."""
        return [p for p in self._templates.values() if p.metadata.name == name]

    def update(self, prompt_id: str, **kwargs: Any) -> Optional[PromptTemplate]:
        """Update a prompt template."""
        if prompt_id not in self._templates:
            return None

        prompt = self._templates[prompt_id]

        if "template" in kwargs:
            prompt.template = kwargs["template"]
        if "description" in kwargs:
            prompt.metadata.description = kwargs["description"]
        if "tags" in kwargs:
            old_tags = set(prompt.metadata.tags)
            new_tags = set(kwargs["tags"])
            for tag in old_tags - new_tags:
                if tag in self._tag_index and prompt_id in self._tag_index[tag]:
                    self._tag_index[tag].remove(prompt_id)
            for tag in new_tags - old_tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(prompt_id)
            prompt.metadata.tags = list(new_tags)
        if "variables" in kwargs:
            prompt.variables = kwargs["variables"]

        prompt.metadata.updated_at = datetime.now(timezone.utc)

        logger.info("prompt_updated", prompt_id=prompt_id)
        return prompt

    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt template."""
        if prompt_id not in self._templates:
            return False

        prompt = self._templates[prompt_id]

        if prompt.metadata.domain in self._domain_index:
            self._domain_index[prompt.metadata.domain].discard(prompt_id)

        for tag in prompt.metadata.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(prompt_id)

        del self._templates[prompt_id]
        self._lineage.pop(prompt_id, None)

        logger.info("prompt_deleted", prompt_id=prompt_id)
        return True

    def list_all(self) -> List[PromptTemplate]:
        """List all prompt templates."""
        return list(self._templates.values())

    def list_by_domain(self, domain: PromptDomain) -> List[PromptTemplate]:
        """List prompt templates by domain."""
        ids = self._domain_index.get(domain, set())
        return [self._templates[i] for i in ids if i in self._templates]

    def list_by_tag(self, tag: str) -> List[PromptTemplate]:
        """List prompt templates by tag."""
        ids = self._tag_index.get(tag, set())
        return [self._templates[i] for i in ids if i in self._templates]

    def get_lineage(self, prompt_id: str) -> List[PromptTemplate]:
        """Get lineage (ancestors) of a prompt template."""
        if prompt_id not in self._templates:
            return []

        lineage: List[PromptTemplate] = []
        for ancestor_id in self._templates[prompt_id].lineage:
            if ancestor_id in self._templates:
                lineage.append(self._templates[ancestor_id])
        return lineage

    def get_children(self, prompt_id: str) -> List[PromptTemplate]:
        """Get child prompts (descendants) of a prompt template."""
        children: List[PromptTemplate] = []
        for prompt in self._templates.values():
            if prompt_id in prompt.lineage:
                children.append(prompt)
        return children

    def fork(
        self,
        prompt_id: str,
        new_name: str,
        new_version: str,
        template_override: Optional[str] = None,
    ) -> Optional[PromptTemplate]:
        """Fork an existing prompt to create a new version."""
        if prompt_id not in self._templates:
            return None

        parent = self._templates[prompt_id]
        return self.create(
            name=new_name,
            template=template_override or parent.template,
            domain=parent.metadata.domain,
            version=new_version,
            description=parent.metadata.description,
            tags=parent.metadata.tags.copy(),
            variables=parent.variables.copy(),
            parent_id=prompt_id,
        )
