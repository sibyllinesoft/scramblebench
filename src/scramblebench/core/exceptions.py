"""
Custom exception hierarchy for ScrambleBench.

This module defines a comprehensive exception hierarchy that provides
clear error categorization and helpful error messages for different
types of failures that can occur in the ScrambleBench framework.
"""

from typing import Any, Dict, List, Optional, Union
import logging


class ScrambleBenchError(Exception):
    """
    Base exception class for all ScrambleBench errors.
    
    This is the root exception that all other ScrambleBench exceptions
    inherit from. It provides common functionality like error categorization,
    logging integration, and structured error information.
    
    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        details: Additional error details
        context: Context information about where the error occurred
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.context = context or {}
        self.cause = cause
        
        # Log the error for debugging
        logger = logging.getLogger(__name__)
        logger.debug(f"Exception created: {self.error_code} - {message}", extra={
            'error_code': self.error_code,
            'details': self.details,
            'context': self.context
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'context': self.context,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        if self.details:
            return f"[{self.error_code}] {self.message} (details: {self.details})"
        return f"[{self.error_code}] {self.message}"


# Configuration Errors
class ConfigurationError(ScrambleBenchError):
    """Errors related to configuration validation and loading."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


# Model Errors  
class ModelError(ScrambleBenchError):
    """Base class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a specified model cannot be found or loaded."""
    pass


class ModelAPIError(ModelError):
    """Raised when model API calls fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data
        self.details.update({
            'status_code': status_code,
            'response_data': response_data
        })


class ModelTimeoutError(ModelError):
    """Raised when model operations timeout."""
    pass


class ModelQuotaExceededError(ModelError):
    """Raised when API quota or rate limits are exceeded."""
    pass


class InvalidModelResponseError(ModelError):
    """Raised when model returns invalid or unexpected response."""
    pass


# Data Errors
class DataError(ScrambleBenchError):
    """Base class for data-related errors."""
    pass


class DataNotFoundError(DataError):
    """Raised when required data cannot be found."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted or invalid."""
    pass


class DataLoadingError(DataError):
    """Raised when data loading operations fail."""
    pass


# Benchmark Errors
class BenchmarkError(ScrambleBenchError):
    """Base class for benchmark-related errors."""
    pass


class BenchmarkExecutionError(BenchmarkError):
    """Raised when benchmark execution fails."""
    pass


class BenchmarkValidationError(BenchmarkError):
    """Raised when benchmark validation fails."""
    pass


class InsufficientDataError(BenchmarkError):
    """Raised when there's insufficient data for benchmark execution."""
    pass


# Translation Errors
class TranslationError(ScrambleBenchError):
    """Base class for translation-related errors."""
    pass


class LanguageGenerationError(TranslationError):
    """Raised when language generation fails."""
    pass


class TranslationValidationError(TranslationError):
    """Raised when translation validation fails."""
    pass


class UnsupportedLanguageError(TranslationError):
    """Raised when an unsupported language is requested."""
    pass


# Evaluation Errors
class EvaluationError(ScrambleBenchError):
    """Base class for evaluation-related errors."""
    pass


class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation operations timeout."""
    pass


class MetricsComputationError(EvaluationError):
    """Raised when metrics computation fails."""
    pass


class InvalidEvaluationModeError(EvaluationError):
    """Raised when an invalid evaluation mode is specified."""
    pass


# IO Errors
class IOError(ScrambleBenchError):
    """Base class for input/output errors."""
    pass


class FileNotFoundError(IOError):
    """Raised when required files cannot be found."""
    pass


class PermissionError(IOError):
    """Raised when file/directory permissions are insufficient."""
    pass


class SerializationError(IOError):
    """Raised when serialization/deserialization fails."""
    pass


# Network Errors
class NetworkError(ScrambleBenchError):
    """Base class for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Raised when network connections fail."""
    pass


class APIError(NetworkError):
    """Raised when API calls fail due to network issues."""
    pass


class RateLimitError(NetworkError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.details['retry_after'] = retry_after


# Validation Errors
class ValidationError(ScrambleBenchError):
    """Base class for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    
    def __init__(
        self,
        message: str,
        schema_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.schema_errors = schema_errors or []
        self.details['schema_errors'] = self.schema_errors


class TypeValidationError(ValidationError):
    """Raised when type validation fails."""
    pass


class RangeValidationError(ValidationError):
    """Raised when value range validation fails."""
    pass


# Authentication/Authorization Errors
class AuthenticationError(ScrambleBenchError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(ScrambleBenchError):
    """Raised when authorization fails."""
    pass


# Resource Errors
class ResourceError(ScrambleBenchError):
    """Base class for resource-related errors."""
    pass


class ResourceExhaustionError(ResourceError):
    """Raised when system resources are exhausted."""
    pass


class MemoryError(ResourceError):
    """Raised when memory allocation fails."""
    pass


class DiskSpaceError(ResourceError):
    """Raised when disk space is insufficient."""
    pass


# Utility Functions for Error Handling
def wrap_exception(
    original_exception: Exception,
    new_exception_class: type,
    message: Optional[str] = None,
    **kwargs
) -> ScrambleBenchError:
    """
    Wrap an original exception in a ScrambleBench exception.
    
    Args:
        original_exception: The original exception to wrap
        new_exception_class: The ScrambleBench exception class to use
        message: Custom message (uses original message if None)
        **kwargs: Additional arguments for the new exception
        
    Returns:
        New ScrambleBench exception with original as cause
    """
    if message is None:
        message = str(original_exception)
    
    return new_exception_class(
        message=message,
        cause=original_exception,
        **kwargs
    )


def handle_common_exceptions(func):
    """
    Decorator to handle common exceptions and convert them to ScrambleBench exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise wrap_exception(e, FileNotFoundError, f"File not found: {e.filename}")
        except PermissionError as e:
            raise wrap_exception(e, PermissionError, f"Permission denied: {e.filename}")
        except ConnectionError as e:
            raise wrap_exception(e, ConnectionError, "Network connection failed")
        except TimeoutError as e:
            raise wrap_exception(e, ModelTimeoutError, "Operation timed out")
        except ValueError as e:
            raise wrap_exception(e, ValidationError, f"Value validation failed: {str(e)}")
        except TypeError as e:
            raise wrap_exception(e, TypeValidationError, f"Type validation failed: {str(e)}")
        except Exception as e:
            # For unknown exceptions, wrap in generic ScrambleBenchError
            raise wrap_exception(e, ScrambleBenchError, f"Unexpected error: {str(e)}")
    
    return wrapper


def validate_not_none(value: Any, name: str) -> Any:
    """
    Validate that a value is not None.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        The value if valid
        
    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def validate_type(value: Any, expected_type: type, name: str) -> Any:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of the value for error messages
        
    Returns:
        The value if valid
        
    Raises:
        TypeValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise TypeValidationError(
            f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: str = "value"
) -> Union[int, float]:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Returns:
        The value if valid
        
    Raises:
        RangeValidationError: If value is outside the allowed range
    """
    if min_value is not None and value < min_value:
        raise RangeValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise RangeValidationError(f"{name} must be <= {max_value}, got {value}")
    
    return value


# Exception Registry for programmatic access
EXCEPTION_REGISTRY = {
    # Configuration
    'configuration': ConfigurationError,
    'invalid_configuration': InvalidConfigurationError,
    'missing_configuration': MissingConfigurationError,
    
    # Models
    'model': ModelError,
    'model_not_found': ModelNotFoundError,
    'model_api': ModelAPIError,
    'model_timeout': ModelTimeoutError,
    'model_quota_exceeded': ModelQuotaExceededError,
    'invalid_model_response': InvalidModelResponseError,
    
    # Data
    'data': DataError,
    'data_not_found': DataNotFoundError,
    'data_validation': DataValidationError,
    'data_corruption': DataCorruptionError,
    'data_loading': DataLoadingError,
    
    # Benchmarks
    'benchmark': BenchmarkError,
    'benchmark_execution': BenchmarkExecutionError,
    'benchmark_validation': BenchmarkValidationError,
    'insufficient_data': InsufficientDataError,
    
    # Translation
    'translation': TranslationError,
    'language_generation': LanguageGenerationError,
    'translation_validation': TranslationValidationError,
    'unsupported_language': UnsupportedLanguageError,
    
    # Evaluation
    'evaluation': EvaluationError,
    'evaluation_timeout': EvaluationTimeoutError,
    'metrics_computation': MetricsComputationError,
    'invalid_evaluation_mode': InvalidEvaluationModeError,
    
    # IO
    'io': IOError,
    'file_not_found': FileNotFoundError,
    'permission': PermissionError,
    'serialization': SerializationError,
    
    # Network
    'network': NetworkError,
    'connection': ConnectionError,
    'api': APIError,
    'rate_limit': RateLimitError,
    
    # Validation
    'validation': ValidationError,
    'schema_validation': SchemaValidationError,
    'type_validation': TypeValidationError,
    'range_validation': RangeValidationError,
    
    # Auth
    'authentication': AuthenticationError,
    'authorization': AuthorizationError,
    
    # Resources
    'resource': ResourceError,
    'resource_exhaustion': ResourceExhaustionError,
    'memory': MemoryError,
    'disk_space': DiskSpaceError,
}


def get_exception_class(error_code: str) -> type:
    """
    Get exception class by error code.
    
    Args:
        error_code: Error code string
        
    Returns:
        Exception class
        
    Raises:
        KeyError: If error code is not found
    """
    return EXCEPTION_REGISTRY[error_code]


def create_exception(error_code: str, message: str, **kwargs) -> ScrambleBenchError:
    """
    Create exception by error code.
    
    Args:
        error_code: Error code string
        message: Error message
        **kwargs: Additional arguments for exception
        
    Returns:
        Created exception instance
    """
    exception_class = get_exception_class(error_code)
    return exception_class(message, error_code=error_code, **kwargs)