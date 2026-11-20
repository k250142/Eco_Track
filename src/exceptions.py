"""Custom exception classes for EcoTrack (Lab 06)."""


class EcoTrackError(Exception):
    """Base class for all EcoTrack-specific errors."""


class DataValidationError(EcoTrackError):
    """Raised when a data file is missing, unreadable, or contains no valid records."""

    def __init__(self, message, file_path=None):
        super().__init__(message)
        self.file_path = file_path


class WeatherServiceError(EcoTrackError):
    """Raised when weather data cannot be retrieved and no cached copy exists."""
