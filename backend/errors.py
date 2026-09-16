from __future__ import annotations


class ArtifactValidationError(RuntimeError):
    """Required production model artifacts are missing or incompatible."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class InvalidVideoError(ValueError):
    """The uploaded file cannot be decoded as a usable video."""


class InferencePipelineError(RuntimeError):
    """The vision model failed after the input passed video validation."""


class RulingPipelineError(RuntimeError):
    """Both the grounded explanation path and its safe fallback failed."""


class APIError(RuntimeError):
    """A public, sanitised HTTP error."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
