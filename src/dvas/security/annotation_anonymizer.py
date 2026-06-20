"""Annotation anonymization for DVAS.

Provides utilities to remove or mask PII from annotation outputs,
including text fields, captions, and metadata.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from dvas.data.schemas import Annotation, Segment, Object, Action
from dvas.security.pii import PIIDetector, PIIFinding, PIIType
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnonymizationConfig:
    """Configuration for annotation anonymization."""

    # Which fields to anonymize
    anonymize_captions: bool = True
    anonymize_qa_pairs: bool = True
    anonymize_object_names: bool = False
    anonymize_action_verbs: bool = False
    anonymize_metadata: bool = True

    # Anonymization strategy
    redaction_token: str = "[REDACTED]"
    use_type_hints: bool = False
    hash_ids: bool = True
    hash_salt: str = "dvas_annotation_anon"

    # PII detection settings
    enabled_pii_types: Optional[Set[PIIType]] = None

    # Fields to preserve (not anonymize)
    preserve_fields: Set[str] = field(default_factory=set)


class AnnotationAnonymizer:
    """Anonymize annotation data by removing or masking PII.

    Usage::

        anonymizer = AnnotationAnonymizer()
        anon_annotation = anonymizer.anonymize(annotation)

        # With custom config
        config = AnonymizationConfig(anonymize_captions=True, hash_ids=True)
        anonymizer = AnnotationAnonymizer(config)
        anon_annotation = anonymizer.anonymize(annotation)
    """

    # Metadata fields that may contain PII
    SENSITIVE_METADATA_FIELDS: Set[str] = {
        "annotator_name",
        "reviewer_name",
        "creator",
        "author",
        "uploader",
        "submitter",
        "owner",
        "contact",
        "email",
        "phone",
        "location",
        "address",
    }

    def __init__(self, config: Optional[AnonymizationConfig] = None) -> None:
        """Initialize the annotation anonymizer.

        Args:
            config: Anonymization configuration. Uses defaults if not provided.
        """
        self.config = config or AnonymizationConfig()
        self.pii_detector = PIIDetector(
            redaction_token=self.config.redaction_token,
            enabled_types=self.config.enabled_pii_types,
        )

    def anonymize(self, annotation: Annotation) -> Annotation:
        """Create an anonymized copy of an annotation.

        Args:
            annotation: The annotation to anonymize.

        Returns:
            A new Annotation with PII removed or masked.
        """
        # Deep copy to avoid modifying the original
        anon_data = copy.deepcopy(annotation)

        # Hash IDs if configured
        if self.config.hash_ids:
            anon_data = self._hash_ids(anon_data)

        # Anonymize text fields
        if self.config.anonymize_captions:
            anon_data = self._anonymize_captions(anon_data)

        if self.config.anonymize_qa_pairs:
            anon_data = self._anonymize_qa_pairs(anon_data)

        if self.config.anonymize_object_names:
            anon_data = self._anonymize_object_names(anon_data)

        if self.config.anonymize_metadata:
            anon_data = self._anonymize_metadata(anon_data)

        return anon_data

    def anonymize_text(self, text: str) -> str:
        """Anonymize a single text string.

        Args:
            text: The text to anonymize.

        Returns:
            Anonymized text.
        """
        if not text:
            return text

        if self.config.use_type_hints:
            return self.pii_detector.redact_with_type_hint(text)
        return self.pii_detector.redact_text(text)

    def scan_annotation(self, annotation: Annotation) -> List[PIIFinding]:
        """Scan an annotation for PII without modifying it.

        Args:
            annotation: The annotation to scan.

        Returns:
            List of PII findings.
        """
        findings: List[PIIFinding] = []

        # Scan captions
        if annotation.segments is None:
            return findings

        for segment in annotation.segments:
            if segment.caption:
                findings.extend(self.pii_detector.scan_text(segment.caption))
            if segment.caption_dense:
                findings.extend(self.pii_detector.scan_text(segment.caption_dense))

            # Scan Q&A pairs
            for qa in segment.qa_pairs:
                findings.extend(self.pii_detector.scan_text(qa.question))
                findings.extend(self.pii_detector.scan_text(qa.answer))

        return findings

    def get_pii_report(self, annotation: Annotation) -> Dict[str, Any]:
        """Generate a PII report for an annotation.

        Args:
            annotation: The annotation to analyze.

        Returns:
            Dictionary with PII statistics and findings.
        """
        findings = self.scan_annotation(annotation)

        stats: Dict[str, int] = {}
        for finding in findings:
            stats[finding.pii_type.value] = stats.get(finding.pii_type.value, 0) + 1

        return {
            "total_findings": len(findings),
            "findings_by_type": stats,
            "has_pii": len(findings) > 0,
            "findings": [f.to_dict() for f in findings],
        }

    def _hash_ids(self, annotation: Annotation) -> Annotation:
        """Hash sensitive ID fields in an annotation."""
        if hasattr(annotation, "id") and annotation.id:
            annotation.id = self._hash_value(annotation.id)

        if hasattr(annotation, "video_id") and annotation.video_id:
            annotation.video_id = self._hash_value(annotation.video_id)

        if hasattr(annotation, "parent_id") and annotation.parent_id:
            annotation.parent_id = self._hash_value(annotation.parent_id)

        # Hash segment IDs if present
        if hasattr(annotation, "segments"):
            for segment in annotation.segments:
                if hasattr(segment, "id") and segment.id:
                    segment.id = self._hash_value(segment.id)

        return annotation

    def _anonymize_captions(self, annotation: Annotation) -> Annotation:
        """Anonymize caption fields in an annotation."""
        if hasattr(annotation, "segments"):
            for segment in annotation.segments:
                if hasattr(segment, "caption") and segment.caption:
                    segment.caption = self.anonymize_text(segment.caption)
                if hasattr(segment, "caption_dense") and segment.caption_dense:
                    segment.caption_dense = self.anonymize_text(segment.caption_dense)

        return annotation

    def _anonymize_qa_pairs(self, annotation: Annotation) -> Annotation:
        """Anonymize Q&A pair fields in an annotation."""
        if hasattr(annotation, "segments"):
            for segment in annotation.segments:
                if hasattr(segment, "qa_pairs"):
                    for qa in segment.qa_pairs:
                        if hasattr(qa, "question"):
                            qa.question = self.anonymize_text(qa.question)
                        if hasattr(qa, "answer"):
                            qa.answer = self.anonymize_text(qa.answer)

        return annotation

    def _anonymize_object_names(self, annotation: Annotation) -> Annotation:
        """Anonymize object names in an annotation."""
        if hasattr(annotation, "segments"):
            for segment in annotation.segments:
                if hasattr(segment, "objects"):
                    for obj in segment.objects:
                        if hasattr(obj, "name") and obj.name:
                            obj.name = self.anonymize_text(obj.name)

        return annotation

    def _anonymize_metadata(self, annotation: Annotation) -> Annotation:
        """Anonymize metadata fields in an annotation."""
        if not hasattr(annotation, "metadata") or annotation.metadata is None:
            return annotation

        metadata = annotation.metadata
        if hasattr(metadata, "model_dump"):
            metadata_dict = metadata.model_dump()
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            return annotation

        # Remove or anonymize sensitive fields
        for field_name in list(metadata_dict.keys()):
            if field_name.lower() in self.SENSITIVE_METADATA_FIELDS:
                if field_name not in self.config.preserve_fields:
                    if isinstance(metadata_dict[field_name], str):
                        metadata_dict[field_name] = self.config.redaction_token

        return annotation

    def _hash_value(self, value: str) -> str:
        """Create a consistent hash of a value."""
        data = f"{self.config.hash_salt}:{value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def anonymize_annotation_dict(
    annotation_dict: Dict[str, Any],
    config: Optional[AnonymizationConfig] = None,
) -> Dict[str, Any]:
    """Anonymize an annotation represented as a dictionary.

    Args:
        annotation_dict: The annotation dictionary to anonymize.
        config: Optional anonymization configuration.

    Returns:
        Anonymized annotation dictionary.
    """
    config = config or AnonymizationConfig()
    detector = PIIDetector(redaction_token=config.redaction_token)

    result = copy.deepcopy(annotation_dict)

    # Recursively anonymize string fields
    def _anonymize_value(value: Any, depth: int = 0) -> Any:
        if depth > 10:
            return value

        if isinstance(value, str):
            return detector.redact_text(value)
        elif isinstance(value, dict):
            return {k: _anonymize_value(v, depth + 1) for k, v in value.items()}
        elif isinstance(value, list):
            return [_anonymize_value(item, depth + 1) for item in value]
        return value

    return _anonymize_value(result)


def mask_annotation_fields(
    annotation_dict: Dict[str, Any],
    fields_to_mask: List[str],
    mask: str = "[REDACTED]",
) -> Dict[str, Any]:
    """Mask specific fields in an annotation dictionary.

    Args:
        annotation_dict: The annotation dictionary.
        fields_to_mask: List of field paths to mask (dot notation).
        mask: The mask string to use.

    Returns:
        Annotation dictionary with specified fields masked.
    """
    result = copy.deepcopy(annotation_dict)

    for field_path in fields_to_mask:
        parts = field_path.split(".")
        current = result

        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                break
        else:
            if isinstance(current, dict) and parts[-1] in current:
                current[parts[-1]] = mask

    return result


__all__ = [
    "AnnotationAnonymizer",
    "AnonymizationConfig",
    "anonymize_annotation_dict",
    "mask_annotation_fields",
]
