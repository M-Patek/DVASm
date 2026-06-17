"""Inference engine for student model with vLLM support."""

from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from dvas.models.teacher.base import TeacherModel
from dvas.utils.logging import get_logger

logger = get_logger(__name__)


def load_frames_from_video(
    video_path: Path,
    num_frames: int = 16,
) -> List[Image.Image]:
    """Load frames from video for inference."""
    from dvas.data.video_loader import VideoLoader

    frames = []
    with VideoLoader(video_path) as loader:
        for frame in loader.read_frames(num_frames=num_frames):
            # Convert BGR to RGB
            rgb = frame.data[:, :, ::-1]
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

    return frames


class StudentInferenceEngine:
    """Inference engine for the fine-tuned student model."""

    def __init__(
        self,
        model_path: Union[str, Path],
        use_vllm: bool = False,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.use_vllm = use_vllm
        self.device = device

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model and processor."""
        logger.info("Loading student model", path=str(self.model_path))

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_hf()

    def _load_hf(self) -> None:
        """Load with HuggingFace transformers."""
        import torch

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        logger.info("Model loaded with HF transformers")

    def _load_vllm(self) -> None:
        """Load with vLLM for optimized inference."""
        try:
            from vllm import LLM

            self.model = LLM(
                model=str(self.model_path),
                trust_remote_code=True,
                tensor_parallel_size=1,
            )

            # vLLM uses its own tokenizer
            self.processor = self.model.get_tokenizer()

            logger.info("Model loaded with vLLM")

        except ImportError:
            logger.warning("vLLM not available, falling back to HF")
            self.use_vllm = False
            self._load_hf()

    def generate(
        self,
        video_path: Path,
        prompt: str = "Describe the video in detail, including hand actions and object interactions.",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        num_beams: int = 1,
    ) -> str:
        """Generate description for a video."""
        # Load frames
        frames = load_frames_from_video(video_path)

        if not frames:
            return ""

        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if self.use_vllm:
            return self._generate_vllm(messages, frames, max_new_tokens, temperature)
        else:
            return self._generate_hf(messages, frames, max_new_tokens, temperature, num_beams)

    def _generate_hf(
        self,
        messages: List[Dict],
        frames: List[Image.Image],
        max_new_tokens: int,
        temperature: float,
        num_beams: int,
    ) -> str:
        """Generate with HF transformers."""
        import torch

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[frames],
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=temperature > 0,
            )

        # Decode
        output_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        # Extract only the assistant response
        if "assistant" in output_text:
            output_text = output_text.split("assistant")[-1].strip()

        return output_text

    def _generate_vllm(
        self,
        messages: List[Dict],
        frames: List[Image.Image],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate with vLLM."""
        # vLLM handles prompt formatting differently
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        # Note: vLLM multi-modal support varies by version
        # This is a placeholder for the actual implementation
        output = self.model.generate(
            prompts=[messages[0]["content"][1]["text"]],
            sampling_params=sampling_params,
        )

        return output[0].outputs[0].text

    async def generate_stream(
        self,
        video_path: Path,
        prompt: str = "Describe the video in detail.",
        max_new_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Stream generation tokens."""
        # For streaming, we need to use text-generation-inference or similar
        # This is a placeholder implementation
        full_text = self.generate(video_path, prompt, max_new_tokens)

        # Yield word by word for demonstration
        words = full_text.split()
        for word in words:
            yield word + " "


class StudentTeacherBridge(TeacherModel):
    """Adapter to use student model as teacher (for cost reduction)."""

    def __init__(
        self,
        student_engine: StudentInferenceEngine,
        fallback_to_teacher: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.student = student_engine
        self.fallback_to_teacher = fallback_to_teacher
        self.confidence_threshold = confidence_threshold
        self._teacher_fallback = None

    def _load_fallback_teacher(self):
        """Load teacher model for fallback."""
        if self._teacher_fallback is None:
            from dvas.models.teacher.gpt4v import GPT4VTeacher

            self._teacher_fallback = GPT4VTeacher()

    async def annotate(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        task: str = "fine_grained",
        **kwargs,
    ) -> Dict[str, any]:
        """Annotate using student model with fallback option."""
        # Convert frames to PIL images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                rgb = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
                pil_frames.append(Image.fromarray(rgb))
            else:
                pil_frames.append(frame)

        # Try student model
        try:
            result = self.student.generate(
                video_path=kwargs.get("video_path", Path("temp.mp4")),
                prompt=prompt or f"Task: {task}",
            )

            # Simple confidence estimation (placeholder)
            confidence = self._estimate_confidence(result)

            if confidence >= self.confidence_threshold:
                return {"text": result, "confidence": confidence, "source": "student"}

        except Exception as e:
            logger.error("Student inference failed", error=str(e))

        # Fallback to teacher if enabled
        if self.fallback_to_teacher:
            logger.info("Falling back to teacher model")
            self._load_fallback_teacher()
            return await self._teacher_fallback.annotate(
                frames=frames, prompt=prompt, task=task, **kwargs
            )

        return {"text": "", "confidence": 0.0, "error": "Student failed, fallback disabled"}

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on response characteristics."""
        # Simple heuristic: longer, more structured responses are more confident
        length_score = min(len(text) / 500, 1.0)

        # Check for action keywords
        action_keywords = ["hand", "pick", "place", "hold", "cut", "pour", "move"]
        has_actions = any(kw in text.lower() for kw in action_keywords)

        # Check for temporal markers
        temporal_markers = ["then", "next", "after", "before", "while"]
        has_temporal = any(tm in text.lower() for tm in temporal_markers)

        # Combine scores
        score = length_score * 0.5 + has_actions * 0.3 + has_temporal * 0.2

        return min(score, 1.0)


def batch_inference(
    model_path: Path,
    video_paths: List[Path],
    prompts: Optional[List[str]] = None,
    batch_size: int = 4,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """Run batch inference on multiple videos."""
    engine = StudentInferenceEngine(model_path)

    results = []
    prompts = prompts or ["Describe this video in detail."] * len(video_paths)

    for i in range(0, len(video_paths), batch_size):
        batch_paths = video_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        for video_path, prompt in zip(batch_paths, batch_prompts):
            try:
                text = engine.generate(
                    video_path=video_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                results.append({
                    "video_path": str(video_path),
                    "text": text,
                    "status": "success",
                })
            except Exception as e:
                results.append({
                    "video_path": str(video_path),
                    "text": "",
                    "status": "failed",
                    "error": str(e),
                })

    return results
