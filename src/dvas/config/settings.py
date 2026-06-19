"""Configuration management for DVAS."""

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_NAME: str = "dvas"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
    DATA_ROOT: Path = Field(default=PROJECT_ROOT / "data")

    # API Keys for Teacher Models
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    TOGETHER_API_KEY: Optional[str] = None

    # Teacher Model Settings
    DEFAULT_TEACHER_MODEL: str = "gpt-5.5"
    TEACHER_MAX_CONCURRENT: int = 10
    TEACHER_RATE_LIMIT_RPM: int = 500
    TEACHER_TIMEOUT_SECONDS: int = 120

    # Student Model Settings
    STUDENT_MODEL_PATH: Optional[str] = None
    STUDENT_DEVICE: str = "auto"
    STUDENT_BATCH_SIZE: int = 1

    # Video Processing
    DEFAULT_FPS: int = 30
    DEFAULT_NUM_FRAMES: int = 16
    VIDEO_MAX_DURATION_SECONDS: float = 60.0
    FRAME_RESIZE: Optional[tuple] = (448, 448)

    # Data Storage
    ANNOTATION_FORMAT: str = "json"
    EXPORT_FORMATS: List[str] = Field(default=["llava", "openai"])

    # Quality Thresholds
    QUALITY_MIN_SCORE: float = 0.7
    QUALITY_AUTO_APPROVE_THRESHOLD: float = 0.9
    QUALITY_HUMAN_REVIEW_THRESHOLD: float = 0.6

    # EPIC-KITCHENS
    EPIC_KITCHENS_ROOT: Optional[Path] = None

    # Redis (for task queue)
    REDIS_URL: str = "redis://localhost:6379/0"

    # API Authentication
    API_KEY: Optional[str] = None  # Set to enable API key auth
    API_KEY_HEADER: str = "X-API-Key"
    ALLOW_UNAUTHENTICATED: bool = True  # Set False to require API key

    @property
    def data_paths(self) -> dict:
        """Get all data paths."""
        return {
            "raw": self.DATA_ROOT / "raw",
            "processed": self.DATA_ROOT / "processed",
            "annotations": self.DATA_ROOT / "annotations",
            "exports": self.DATA_ROOT / "exports",
        }

    @property
    def annotation_paths(self) -> dict:
        """Get annotation storage paths."""
        root = self.DATA_ROOT / "annotations"
        return {
            "gold": root / "gold",
            "model": root / "model",
            "reviewed": root / "reviewed",
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
