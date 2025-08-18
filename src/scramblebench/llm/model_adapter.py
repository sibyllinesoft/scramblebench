"""
Unified Model Adapter for ScrambleBench
=======================================

This module provides a standardized way to interact with different types of models
across the ScrambleBench framework. It eliminates the duplicate model interaction
patterns found in various benchmark implementations and provides a single,
consistent interface for model queries.

The ModelAdapter handles different model interfaces automatically and provides
enhanced error handling, logging, and performance monitoring capabilities.

Key Features:
    * Automatic interface detection for different model types
    * Consistent error handling and logging
    * Response validation and sanitization
    * Performance monitoring and metrics collection
    * Retry logic for transient failures
    * Support for both sync and async model interfaces
"""

from typing import Any, Dict, List, Optional, Union, Callable
import logging
import time
import asyncio
from dataclasses import dataclass
from enum import Enum
import inspect

from scramblebench.llm.model_interface import ModelInterface, ModelResponse
from scramblebench.core.exceptions import (
    ModelError, ModelTimeoutError, ModelAPIError, InvalidModelResponseError,
    ValidationError, handle_common_exceptions
)
from scramblebench.core.logging import get_logger, log_performance


class ModelInterfaceType(Enum):
    """Types of model interfaces detected."""
    SCRAMBLEBENCH_INTERFACE = "scramblebench_interface"  # Our ModelInterface
    GENERATE_METHOD = "generate_method"  # Has generate() method
    QUERY_METHOD = "query_method"  # Has query() method 
    CALLABLE = "callable"  # Callable object
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """
    Result of a model query with metadata.
    
    This standardizes the response format across different model types
    and provides additional metadata for debugging and analysis.
    """
    text: str
    success: bool
    error: Optional[str] = None
    interface_type: Optional[ModelInterfaceType] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None
    
    def __bool__(self) -> bool:
        """Allow truth testing based on success."""
        return self.success
    
    @property
    def is_empty(self) -> bool:
        """Check if the response text is empty or whitespace only."""
        return not self.text or not self.text.strip()


class ModelAdapter:
    """
    Unified adapter for interacting with different model types.
    
    This class provides a consistent interface for querying models regardless
    of their underlying implementation. It automatically detects the model's
    interface and uses the appropriate method for interaction.
    
    The adapter supports:
    - ScrambleBench ModelInterface implementations
    - Models with generate() methods
    - Models with query() methods
    - Callable models
    - Custom interface methods
    
    Examples:
        Basic usage:
        
        .. code-block:: python
        
            adapter = ModelAdapter()
            
            # Works with any model type
            result = adapter.query(model, "What is 2+2?")
            
            if result.success:
                print(f"Response: {result.text}")
                print(f"Time: {result.response_time:.2f}s")
            else:
                print(f"Error: {result.error}")
        
        With configuration:
        
        .. code-block:: python
        
            adapter = ModelAdapter(
                timeout=60,
                max_retries=3,
                logger=custom_logger
            )
            
            result = adapter.query(model, prompt)
        
        Batch processing:
        
        .. code-block:: python
        
            prompts = ["Question 1", "Question 2", "Question 3"]
            results = adapter.batch_query(model, prompts)
            
            for i, result in enumerate(results):
                print(f"Prompt {i}: {result.text}")
        
        Async usage:
        
        .. code-block:: python
        
            result = await adapter.query_async(model, prompt)
    """
    
    def __init__(
        self,
        timeout: Optional[float] = None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        logger: Optional[logging.Logger] = None,
        default_interface_method: Optional[str] = None,
        validate_responses: bool = True
    ):
        """
        Initialize the model adapter.
        
        Args:
            timeout: Maximum time to wait for model response (seconds)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            logger: Logger instance for debugging and monitoring
            default_interface_method: Default method name for custom interfaces
            validate_responses: Whether to validate and sanitize responses
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = get_logger(__name__) if logger is None else logger
        self.default_interface_method = default_interface_method
        self.validate_responses = validate_responses
        
        # Performance tracking
        self._query_count = 0
        self._total_response_time = 0.0
        self._error_count = 0
        
        # Interface cache to avoid repeated detection
        self._interface_cache: Dict[str, ModelInterfaceType] = {}
    
    @handle_common_exceptions
    @log_performance
    def query(
        self,
        model: Any,
        prompt: str,
        **kwargs
    ) -> QueryResult:
        """
        Query a model with a prompt and return the response.
        
        This is the main entry point for model interaction. It automatically
        detects the model's interface and uses the appropriate method.
        
        Args:
            model: The model to query (any type supported)
            prompt: The input prompt/question
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            QueryResult containing the response and metadata
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Empty or invalid prompt provided", 
                                error_code="empty_prompt",
                                context={"prompt_length": len(prompt) if prompt else 0})
        
        start_time = time.time()
        self._query_count += 1
        
        try:
            # Detect model interface
            interface_type = self._detect_interface(model)
            
            # Execute query with retries
            result = self._execute_with_retries(model, prompt, interface_type, **kwargs)
            
            # Add metadata
            response_time = time.time() - start_time
            result.response_time = response_time
            result.interface_type = interface_type
            
            # Update performance tracking
            self._total_response_time += response_time
            
            if not result.success:
                self._error_count += 1
            
            # Validate and sanitize response if enabled
            if self.validate_responses and result.success:
                result = self._validate_response(result)
            
            self.logger.debug(
                f"Model query completed: {result.success}, "
                f"time={response_time:.3f}s, "
                f"interface={interface_type.value}"
            )
            
            return result
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Model query failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return QueryResult(
                text="",
                success=False,
                error=error_msg,
                response_time=time.time() - start_time
            )
    
    def batch_query(
        self,
        model: Any,
        prompts: List[str],
        **kwargs
    ) -> List[QueryResult]:
        """
        Query a model with multiple prompts.
        
        Args:
            model: The model to query
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of QueryResult objects
        """
        results = []
        
        # Check if model supports batch processing
        if isinstance(model, ModelInterface):
            try:
                # Use model's batch processing if available
                responses = model.batch_generate(prompts, **kwargs)
                
                for prompt, response in zip(prompts, responses):
                    if response.error:
                        result = QueryResult(
                            text="",
                            success=False,
                            error=response.error,
                            interface_type=ModelInterfaceType.SCRAMBLEBENCH_INTERFACE
                        )
                    else:
                        result = QueryResult(
                            text=response.text,
                            success=True,
                            interface_type=ModelInterfaceType.SCRAMBLEBENCH_INTERFACE,
                            metadata=response.metadata,
                            raw_response=response.raw_response
                        )
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                self.logger.warning(f"Batch processing failed, falling back to individual queries: {e}")
        
        # Fall back to individual queries
        for prompt in prompts:
            result = self.query(model, prompt, **kwargs)
            results.append(result)
        
        return results
    
    async def query_async(
        self,
        model: Any,
        prompt: str,
        **kwargs
    ) -> QueryResult:
        """
        Asynchronous version of query method.
        
        Args:
            model: The model to query
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            QueryResult containing the response and metadata
        """
        # For sync models, run in thread pool
        if not self._is_async_model(model):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.query(model, prompt, **kwargs)
            )
        
        # For async models, implement async query logic
        # This would need to be expanded based on async model interfaces
        return self.query(model, prompt, **kwargs)
    
    def _detect_interface(self, model: Any) -> ModelInterfaceType:
        """
        Detect the type of interface the model supports.
        
        Args:
            model: The model to inspect
            
        Returns:
            Detected interface type
        """
        model_id = id(model)
        
        # Check cache first
        if model_id in self._interface_cache:
            return self._interface_cache[model_id]
        
        interface_type = ModelInterfaceType.UNKNOWN
        
        # Check for ScrambleBench ModelInterface
        if isinstance(model, ModelInterface):
            interface_type = ModelInterfaceType.SCRAMBLEBENCH_INTERFACE
        
        # Check for generate method
        elif hasattr(model, 'generate') and callable(getattr(model, 'generate')):
            interface_type = ModelInterfaceType.GENERATE_METHOD
        
        # Check for query method
        elif hasattr(model, 'query') and callable(getattr(model, 'query')):
            interface_type = ModelInterfaceType.QUERY_METHOD
        
        # Check if model is callable
        elif callable(model):
            interface_type = ModelInterfaceType.CALLABLE
        
        # Check for custom interface method
        elif (self.default_interface_method and 
              hasattr(model, self.default_interface_method) and
              callable(getattr(model, self.default_interface_method))):
            interface_type = ModelInterfaceType.GENERATE_METHOD  # Treat as generate
        
        # Cache the result
        self._interface_cache[model_id] = interface_type
        
        self.logger.debug(f"Detected interface type: {interface_type.value} for model {type(model)}")
        
        return interface_type
    
    def _execute_with_retries(
        self,
        model: Any,
        prompt: str,
        interface_type: ModelInterfaceType,
        **kwargs
    ) -> QueryResult:
        """
        Execute the model query with retry logic.
        
        Args:
            model: The model to query
            prompt: The input prompt
            interface_type: Detected interface type
            **kwargs: Additional parameters
            
        Returns:
            QueryResult from the successful execution
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._execute_query(model, prompt, interface_type, **kwargs)
                
                if result.success:
                    if attempt > 0:
                        self.logger.info(f"Query succeeded on attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Query attempt {attempt + 1} failed: {e}")
            
            # Wait before retrying (except on last attempt)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                self.logger.debug(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        
        return QueryResult(
            text="",
            success=False,
            error=f"Query failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )
    
    def _execute_query(
        self,
        model: Any,
        prompt: str,
        interface_type: ModelInterfaceType,
        **kwargs
    ) -> QueryResult:
        """
        Execute the actual model query based on interface type.
        
        Args:
            model: The model to query
            prompt: The input prompt
            interface_type: The interface type to use
            **kwargs: Additional parameters
            
        Returns:
            QueryResult from the execution
        """
        try:
            if interface_type == ModelInterfaceType.SCRAMBLEBENCH_INTERFACE:
                response = model.generate(prompt, **kwargs)
                
                if response.error:
                    return QueryResult(
                        text="",
                        success=False,
                        error=response.error,
                        raw_response=response
                    )
                else:
                    return QueryResult(
                        text=response.text,
                        success=True,
                        metadata=response.metadata,
                        raw_response=response
                    )
            
            elif interface_type == ModelInterfaceType.GENERATE_METHOD:
                text = model.generate(prompt, **kwargs)
                return QueryResult(
                    text=str(text),
                    success=True
                )
            
            elif interface_type == ModelInterfaceType.QUERY_METHOD:
                text = model.query(prompt, **kwargs)
                return QueryResult(
                    text=str(text),
                    success=True
                )
            
            elif interface_type == ModelInterfaceType.CALLABLE:
                text = model(prompt, **kwargs)
                return QueryResult(
                    text=str(text),
                    success=True
                )
            
            else:
                return QueryResult(
                    text="",
                    success=False,
                    error=f"Unsupported model interface: {interface_type.value}"
                )
                
        except Exception as e:
            return QueryResult(
                text="",
                success=False,
                error=f"Model execution failed: {str(e)}"
            )
    
    def _validate_response(self, result: QueryResult) -> QueryResult:
        """
        Validate and sanitize the model response.
        
        Args:
            result: The query result to validate
            
        Returns:
            Validated and possibly modified query result
        """
        if not result.success:
            return result
        
        # Check for empty responses
        if result.is_empty:
            self.logger.warning("Model returned empty response")
            return QueryResult(
                text="",
                success=False,
                error="Model returned empty response",
                response_time=result.response_time,
                interface_type=result.interface_type
            )
        
        # Basic text sanitization
        sanitized_text = result.text.strip()
        
        # Log if text was modified
        if sanitized_text != result.text:
            self.logger.debug("Response text was sanitized")
        
        return QueryResult(
            text=sanitized_text,
            success=True,
            response_time=result.response_time,
            interface_type=result.interface_type,
            metadata=result.metadata,
            raw_response=result.raw_response
        )
    
    def _is_async_model(self, model: Any) -> bool:
        """
        Check if the model supports async operations.
        
        Args:
            model: The model to check
            
        Returns:
            True if model appears to support async operations
        """
        # Check for async methods
        for method_name in ['generate', 'query', '__call__']:
            if hasattr(model, method_name):
                method = getattr(model, method_name)
                if inspect.iscoroutinefunction(method):
                    return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this adapter.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_response_time = (
            self._total_response_time / self._query_count
            if self._query_count > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / self._query_count
            if self._query_count > 0 else 0.0
        )
        
        return {
            'total_queries': self._query_count,
            'total_response_time': self._total_response_time,
            'average_response_time': avg_response_time,
            'total_errors': self._error_count,
            'error_rate': error_rate,
            'interface_cache_size': len(self._interface_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the interface detection cache."""
        self._interface_cache.clear()
        self.logger.debug("Interface cache cleared")
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._query_count = 0
        self._total_response_time = 0.0
        self._error_count = 0
        self.logger.debug("Performance statistics reset")


# Global default adapter instance
_default_adapter = None


def get_default_adapter() -> ModelAdapter:
    """
    Get the global default ModelAdapter instance.
    
    Returns:
        Default ModelAdapter instance
    """
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = ModelAdapter()
    return _default_adapter


def set_default_adapter(adapter: ModelAdapter) -> None:
    """
    Set the global default ModelAdapter instance.
    
    Args:
        adapter: The ModelAdapter instance to use as default
    """
    global _default_adapter
    _default_adapter = adapter


# Convenience functions using the default adapter
def query_model(model: Any, prompt: str, **kwargs) -> QueryResult:
    """
    Convenience function to query a model using the default adapter.
    
    Args:
        model: The model to query
        prompt: The input prompt
        **kwargs: Additional parameters
        
    Returns:
        QueryResult containing the response and metadata
    """
    return get_default_adapter().query(model, prompt, **kwargs)


def batch_query_model(model: Any, prompts: List[str], **kwargs) -> List[QueryResult]:
    """
    Convenience function for batch queries using the default adapter.
    
    Args:
        model: The model to query
        prompts: List of input prompts
        **kwargs: Additional parameters
        
    Returns:
        List of QueryResult objects
    """
    return get_default_adapter().batch_query(model, prompts, **kwargs)


# Legacy compatibility function
def query_model_legacy(model: Any, prompt: str) -> str:
    """
    Legacy compatibility function that returns just the text response.
    
    This function mimics the behavior of the old _query_model methods
    found in benchmark classes, providing backward compatibility.
    
    Args:
        model: The model to query
        prompt: The input prompt
        
    Returns:
        The response text
        
    Raises:
        ValueError: If the model doesn't have a recognized interface
        RuntimeError: If the query fails
    """
    result = query_model(model, prompt)
    
    if not result.success:
        if "does not have a recognized interface" in (result.error or ""):
            raise ValueError(result.error)
        else:
            raise RuntimeError(result.error)
    
    return result.text