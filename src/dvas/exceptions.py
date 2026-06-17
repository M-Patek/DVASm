"""DVAS exceptions hierarchy."""

from typing import Any, Dict, Optional


class DVASException(Exception):
    """Base exception for DVAS."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "DVAS_001"
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(DVASException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DVAS_CFG_001", details=details)


class ValidationError(DVASException):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DVAS_VAL_001", details=details)
        self.field = field


class StorageError(DVASException):
    """Raised when storage operations fail."""

    def __init__(self, message: str, path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DVAS_STOR_001", details=details)
        self.path = path


class VideoProcessingError(DVASException):
    """Raised when video processing fails."""

    def __init__(
        self,
        message: str,
        video_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code="DVAS_VID_001", details=details)
        self.video_path = video_path


class APIError(DVASException):
    """Base exception for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code="DVAS_API_001", details=details)
        self.status_code = status_code


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=429, details=details)
        self.error_code = "DVAS_API_429"
        self.retry_after = retry_after


class APITimeoutError(APIError):
    """Raised when API call times out."""

    def __init__(self, message: str, timeout: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=504, details=details)
        self.error_code = "DVAS_API_504"
        self.timeout = timeout


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)
        self.error_code = "DVAS_API_401"


class ModelInferenceError(DVASException):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code="DVAS_MODEL_001", details=details)
        self.model_name = model_name


class PipelineError(DVASException):
    """Raised when pipeline execution fails."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code="DVAS_PIPE_001", details=details)
        self.stage = stage


class RetryExhaustedError(DVASException):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        last_error: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code="DVAS_RETRY_001", details=details)
        self.attempts = attempts
        self.last_error = last_error
