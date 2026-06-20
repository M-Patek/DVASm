"""Hash utilities for content integrity checking."""

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dvas.data.schemas import Annotation


def compute_annotation_hash(annotation: "Annotation") -> str:
    """Compute a content hash for an annotation.

    Uses SHA-256 of the normalized annotation data to detect
    content changes and enable deduplication.

    Args:
        annotation: The annotation to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    # Create a normalized representation for hashing
    data = annotation.model_dump()

    # Remove fields that shouldn't affect hash (timestamps, etc.)
    data.pop("created_at", None)
    data.pop("updated_at", None)
    data.pop("indexed_at", None)
    data.pop("lineage", None)

    # Normalize to consistent string representation
    import json

    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_video_hash(video_path: str, sample_frames: int = 10) -> str:
    """Compute a content hash for a video file.

    Samples frames at regular intervals to create a content
    fingerprint that can detect duplicate videos even if
    filenames differ.

    Args:
        video_path: Path to the video file
        sample_frames: Number of frames to sample

    Returns:
        Hex-encoded SHA-256 hash
    """
    import os

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get file stats for basic hash
    stat = os.stat(video_path)
    size = stat.st_size
    mtime = stat.st_mtime

    # Try to sample frames if possible
    frame_hashes = []
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            step = max(1, total_frames // sample_frames)
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Hash frame data
                    frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
                    frame_hashes.append(frame_hash)

        cap.release()
    except Exception:
        # Fallback to file hash if video processing fails
        pass

    # Combine file metadata and frame hashes
    hasher = hashlib.sha256()
    hasher.update(f"{size}:{mtime}".encode())
    for fh in frame_hashes:
        hasher.update(fh.encode())

    return hasher.hexdigest()


def compute_frame_hash(frame_data: bytes) -> str:
    """Compute a hash for frame data.

    Args:
        frame_data: Raw frame bytes

    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(frame_data).hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute a hash for a string.

    Args:
        text: Input string

    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_hashes(hash1: str, hash2: str, threshold: float = 1.0) -> bool:
    """Compare two hashes with optional similarity threshold.

    For exact matching, threshold should be 1.0.
    For approximate matching, threshold can be lower.

    Args:
        hash1: First hash
        hash2: Second hash
        threshold: Similarity threshold (0.0 - 1.0)

    Returns:
        True if hashes match within threshold
    """
    if threshold == 1.0:
        return hash1 == hash2

    # For approximate matching, use Hamming distance
    # This works for perceptual hashes (not SHA-256)
    if len(hash1) != len(hash2):
        return False

    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    max_distance = len(hash1)
    similarity = 1.0 - (distance / max_distance)

    return similarity >= threshold
