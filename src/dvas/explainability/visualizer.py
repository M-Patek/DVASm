"""Explainability and visualization for annotations."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dvas.data.schemas import Annotation, Segment
from dvas.data.video_loader import VideoLoader
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KeyFrameInfo:
    """Information about a key frame."""

    timestamp: float
    frame_idx: int
    importance_score: float
    description: str
    objects: List[str]
    actions: List[str]


@dataclass
class AttentionHeatmap:
    """Attention heatmap for a frame."""

    frame_idx: int
    heatmap: np.ndarray  # 2D array
    max_attention_point: Tuple[int, int]
    attention_regions: List[Dict]  # bbox + score


class KeyFrameExtractor:
    """Extract key frames that best represent the video."""

    def __init__(self, num_keyframes: int = 5):
        self.num_keyframes = num_keyframes

    def extract(
        self, video_path: Path, annotation: Annotation
    ) -> List[KeyFrameInfo]:
        """Extract key frames based on annotation and visual importance."""
        keyframes = []

        with VideoLoader(video_path) as loader:
            # Use segment boundaries as key frame candidates
            for i, segment in enumerate(annotation.segments[: self.num_keyframes]):
                # Get middle of segment
                mid_time = (segment.start_time + segment.end_time) / 2

                # Read frame
                frames = list(loader.read_frames(start_time=mid_time, num_frames=1))
                if not frames:
                    continue

                # Calculate importance score
                importance = self._calculate_importance(segment)

                keyframes.append(
                    KeyFrameInfo(
                        timestamp=mid_time,
                        frame_idx=frames[0].idx,
                        importance_score=importance,
                        description=segment.caption[:100],
                        objects=[obj.name for obj in segment.objects],
                        actions=[f"{a.verb} {a.noun}" for a in segment.actions],
                    )
                )

        # Sort by importance
        keyframes.sort(key=lambda k: k.importance_score, reverse=True)
        return keyframes

    def _calculate_importance(self, segment: Segment) -> float:
        """Calculate importance score for a segment."""
        score = 0.0

        # Longer segments might be more important
        score += min(segment.duration / 10, 1.0) * 0.2

        # Segments with actions are important
        score += min(len(segment.actions) / 3, 1.0) * 0.3

        # Segments with objects are important
        score += min(len(segment.objects) / 5, 1.0) * 0.2

        # Segments with QA pairs are important
        score += min(len(segment.qa_pairs) / 2, 1.0) * 0.15

        # Caption length as proxy for detail
        score += min(len(segment.caption) / 200, 1.0) * 0.15

        return score


class AttentionVisualizer:
    """Visualize attention patterns in video understanding."""

    def __init__(self):
        self.colormap = cv2.COLORMAP_JET

    def generate_heatmap(
        self, frame: np.ndarray, attention_weights: np.ndarray
    ) -> np.ndarray:
        """Generate attention heatmap overlay on frame."""
        # Resize attention to match frame
        h, w = frame.shape[:2]
        attention_resized = cv2.resize(attention_weights, (w, h))

        # Normalize to 0-255
        attention_norm = (
            (attention_resized - attention_resized.min())
            / (attention_resized.max() - attention_resized.min() + 1e-8)
            * 255
        ).astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(attention_norm, self.colormap)

        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        return overlay

    def create_temporal_attention_map(
        self, annotation: Annotation, video_path: Path
    ) -> np.ndarray:
        """Create temporal attention visualization."""
        with VideoLoader(video_path) as loader:
            metadata = loader.metadata

            # Create timeline visualization
            timeline_width = 1200
            timeline_height = 200
            timeline = np.ones((timeline_height, timeline_width, 3), dtype=np.uint8) * 255

            # Draw timeline bar
            bar_y = timeline_height // 2
            cv2.line(
                timeline,
                (50, bar_y),
                (timeline_width - 50, bar_y),
                (200, 200, 200),
                4,
            )

            # Draw segments as colored regions
            duration = metadata.duration
            for i, segment in enumerate(annotation.segments):
                start_x = int((segment.start_time / duration) * (timeline_width - 100)) + 50
                end_x = int((segment.end_time / duration) * (timeline_width - 100)) + 50

                # Color based on importance
                importance = len(segment.actions) + len(segment.objects)
                color = self._importance_to_color(importance)

                # Draw segment region
                cv2.rectangle(
                    timeline,
                    (start_x, bar_y - 20),
                    (end_x, bar_y + 20),
                    color,
                    -1,
                )

                # Add label if space permits
                if end_x - start_x > 50:
                    label = f"S{i+1}"
                    cv2.putText(
                        timeline,
                        label,
                        (start_x + 5, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                    )

            return timeline

    def _importance_to_color(self, importance: int) -> Tuple[int, int, int]:
        """Convert importance score to BGR color."""
        colors = [
            (200, 200, 200),  # Low - gray
            (0, 255, 255),  # Medium-low - yellow
            (0, 255, 0),  # Medium - green
            (255, 0, 0),  # Medium-high - blue
            (0, 0, 255),  # High - red
        ]
        idx = min(importance, len(colors) - 1)
        return colors[idx]


class AnnotationVisualizer:
    """Visualize annotation results on video frames."""

    def __init__(self, font_size: int = 14):
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            self.font = ImageFont.load_default()

    def visualize_segment(
        self,
        frame: np.ndarray,
        segment: Segment,
        draw_objects: bool = True,
        draw_actions: bool = True,
        draw_caption: bool = True,
    ) -> np.ndarray:
        """Visualize annotation on a frame."""
        # Convert to PIL for text rendering
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        y_offset = 10

        # Draw caption
        if draw_caption and segment.caption:
            caption = segment.caption[:100] + "..." if len(segment.caption) > 100 else segment.caption
            self._draw_text_with_bg(draw, 10, y_offset, f"Caption: {caption}", (255, 255, 255))
            y_offset += 25

        # Draw actions
        if draw_actions and segment.actions:
            for action in segment.actions[:3]:  # Max 3 actions
                text = f"{action.hand.value}: {action.verb} {action.noun}"
                self._draw_text_with_bg(draw, 10, y_offset, text, (0, 255, 255))
                y_offset += 25

        # Draw objects
        if draw_objects and segment.objects:
            obj_text = "Objects: " + ", ".join(obj.name for obj in segment.objects[:5])
            self._draw_text_with_bg(draw, 10, y_offset, obj_text, (255, 255, 0))

        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_text_with_bg(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw text with semi-transparent background."""
        bbox = draw.textbbox((x, y), text, font=self.font)
        draw.rectangle(bbox, fill=(0, 0, 0, 128))
        draw.text((x, y), text, font=self.font, fill=color)

    def create_annotation_summary_image(
        self, annotation: Annotation, video_path: Path
    ) -> np.ndarray:
        """Create a summary image showing key frames and annotations."""
        extractor = KeyFrameExtractor(num_keyframes=4)
        keyframes = extractor.extract(video_path, annotation)

        # Create grid layout
        grid_width = 2
        grid_height = 2
        frame_width = 400
        frame_height = 300

        summary = np.ones(
            (frame_height * grid_height, frame_width * grid_width, 3),
            dtype=np.uint8,
        )

        with VideoLoader(video_path) as loader:
            for i, kf in enumerate(keyframes[:4]):
                row = i // grid_width
                col = i % grid_width

                # Read frame
                frames = list(loader.read_frames(start_time=kf.timestamp, num_frames=1))
                if not frames:
                    continue

                frame = frames[0].data
                frame = cv2.resize(frame, (frame_width, frame_height))

                # Add keyframe info
                cv2.putText(
                    frame,
                    f"T={kf.timestamp:.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Place in grid
                y_start = row * frame_height
                x_start = col * frame_width
                summary[y_start : y_start + frame_height, x_start : x_start + frame_width] = frame

        return summary


class ExplainabilityReport:
    """Generate explainability reports for annotations."""

    def __init__(self):
        self.visualizer = AnnotationVisualizer()
        self.attention_viz = AttentionVisualizer()
        self.keyframe_extractor = KeyFrameExtractor()

    def generate_report(
        self,
        annotation: Annotation,
        video_path: Path,
        output_dir: Path,
    ) -> Dict:
        """Generate comprehensive explainability report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "annotation_id": annotation.id,
            "video_id": annotation.video_path,
            "segments_explanation": [],
            "keyframes": [],
        }

        # Extract keyframes
        keyframes = self.keyframe_extractor.extract(video_path, annotation)
        report["keyframes"] = [
            {
                "timestamp": kf.timestamp,
                "importance": kf.importance_score,
                "description": kf.description,
            }
            for kf in keyframes
        ]

        # Explain each segment
        for i, segment in enumerate(annotation.segments):
            explanation = self._explain_segment(segment)
            report["segments_explanation"].append({
                "segment_idx": i,
                "explanation": explanation,
            })

        # Generate visualizations
        summary_img = self.visualizer.create_annotation_summary_image(
            annotation, video_path
        )
        summary_path = output_dir / f"{annotation.id}_summary.jpg"
        cv2.imwrite(str(summary_path), summary_img)
        report["summary_image"] = str(summary_path)

        logger.info("explainability_report_generated", path=str(output_dir))

        return report

    def _explain_segment(self, segment: Segment) -> Dict:
        """Generate explanation for a segment."""
        return {
            "what_happens": segment.caption,
            "key_actions": [
                {"verb": a.verb, "noun": a.noun, "hand": a.hand.value}
                for a in segment.actions
            ],
            "objects_involved": [obj.name for obj in segment.objects],
            "temporal_context": {
                "start": segment.start_time,
                "end": segment.end_time,
                "duration": segment.duration,
            },
            "reasoning": self._generate_reasoning(segment),
        }

    def _generate_reasoning(self, segment: Segment) -> str:
        """Generate natural language reasoning for the annotation."""
        parts = []

        if segment.actions:
            action_desc = ", ".join(
                f"{a.verb}ing {a.noun}" for a in segment.actions[:2]
            )
            parts.append(f"The video shows {action_desc}.")

        if segment.objects:
            obj_list = ", ".join(obj.name for obj in segment.objects[:3])
            parts.append(f"Key objects visible include {obj_list}.")

        if segment.qa_pairs:
            parts.append(f"{len(segment.qa_pairs)} question-answer pairs extracted.")

        return " ".join(parts) if parts else "No specific details extracted."


def create_attention_animation(
    video_path: Path,
    attention_sequence: List[np.ndarray],
    output_path: Path,
    fps: int = 10,
) -> None:
    """Create video with attention overlay."""
    visualizer = AttentionVisualizer()

    with VideoLoader(video_path) as loader:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        for i, attention in enumerate(attention_sequence):
            # Read corresponding frame
            timestamp = i / fps
            frames = list(loader.read_frames(start_time=timestamp, num_frames=1))

            if not frames:
                break

            frame = frames[0].data

            # Generate heatmap overlay
            overlay = visualizer.generate_heatmap(frame, attention)

            if writer is None:
                h, w = overlay.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            writer.write(overlay)

        if writer:
            writer.release()

    logger.info("attention_animation_created", path=str(output_path))
