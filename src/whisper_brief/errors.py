"""Exception types. Everything the CLI reports to the user derives from WhisperBriefError."""

from __future__ import annotations


class WhisperBriefError(Exception):
    """Base class for errors this package raises on purpose."""


class ConfigError(WhisperBriefError):
    """Missing or invalid configuration, such as an absent API key."""


class FFmpegError(WhisperBriefError):
    """ffmpeg or ffprobe is missing, or exited with an error."""


class APIError(WhisperBriefError):
    """An HTTP call to a provider failed, after any retries."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class BriefValidationError(WhisperBriefError):
    """The model's brief could not be parsed or did not match the schema."""


class LocalBackendUnavailable(WhisperBriefError):
    """The local transcription backend was requested but faster-whisper is not installed."""
