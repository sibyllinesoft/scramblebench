"""
Abstract Model Interface for ScrambleBench
==========================================

This module defines the standardized interface that all Large Language Model (LLM)
implementations must follow to ensure consistent interaction patterns across different
model providers, types, and deployment configurations. It provides the foundation
for ScrambleBench's provider-agnostic model integration system.

Core Design Principles
----------------------

**Interface Uniformity:**
    All models expose identical methods regardless of their underlying API,
    enabling seamless provider switching and evaluation consistency.

**Type Safety:**
    Comprehensive type annotations and dataclass structures ensure reliable
    data flow and early error detection.

**Configuration Standardization:**
    Universal configuration schema allows consistent parameter management
    across different model providers and types.

**Error Handling:**
    Structured error reporting through :class:`ModelResponse` enables
    graceful degradation and detailed debugging information.

Interface Hierarchy
--------------------

**Abstract Base Classes:**
    * :class:`ModelInterface` - Main interface for all model implementations
    
**Configuration Classes:**
    * :class:`ModelConfig` - Universal model behavior configuration
    * :class:`ModelResponse` - Standardized response container
    
**Enumeration Types:**
    * :class:`ModelType` - Classification of different model capabilities

**Utility Classes:**
    * :class:`DummyModel` - Testing implementation for development
    * :class:`ModelFactory` - Centralized model creation and registration

Implementation Requirements
---------------------------

Any model implementation must provide:

**Initialization:**
    - Proper API key and connection management
    - Configuration validation and application
    - Health checks and connectivity testing

**Core Generation Methods:**
    - :meth:`generate` - Single prompt text generation
    - :meth:`chat` - Conversation-based interaction  
    - :meth:`batch_generate` - Batch processing for efficiency

**Utility Methods:**
    - :meth:`estimate_tokens` - Token counting for cost estimation
    - :meth:`validate_prompt` - Input validation and sanitation
    - :meth:`get_model_info` - Provider and capability information

Configuration Management
------------------------

The :class:`ModelConfig` provides unified parameter control:

**Sampling Parameters:**
    * ``temperature`` - Controls randomness (0.0 = deterministic, 2.0 = very creative)
    * ``top_p`` - Nucleus sampling threshold for diverse but coherent outputs
    * ``max_tokens`` - Maximum response length to prevent runaway generation

**Penalty Systems:**
    * ``frequency_penalty`` - Reduces repetitive text patterns
    * ``presence_penalty`` - Encourages diverse topic exploration
    * ``stop_sequences`` - Custom termination triggers

**Operational Controls:**
    * ``timeout`` - Request timeout for reliability
    * Custom provider-specific parameters via kwargs

Response Structure
------------------

The :class:`ModelResponse` standardizes all model outputs:

**Generated Content:**
    * ``text`` - Primary generated text content
    * ``raw_response`` - Complete provider response for debugging

**Metadata Tracking:**
    * Token usage statistics (prompt, completion, total)
    * Response timing and performance metrics
    * Model identification and configuration
    * Quality indicators (finish_reason, etc.)

**Error Information:**
    * Structured error messages for debugging
    * Provider-specific error codes and details
    * Retry suggestions and recovery guidance

Usage Patterns
--------------

**Direct Model Usage:**
    For simple text generation tasks:

    .. code-block:: python

        from scramblebench.llm.model_interface import ModelConfig
        
        # Configure model behavior
        config = ModelConfig(
            temperature=0.0,        # Deterministic output
            max_tokens=500,         # Limit response length
            frequency_penalty=0.1   # Reduce repetition
        )
        
        # Use with any provider implementation
        model = SomeModelImplementation("model-name", config=config)
        model.initialize()
        
        # Generate text
        response = model.generate("Explain photosynthesis")
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.text}")
            print(f"Tokens used: {response.metadata['total_tokens']}")

**Conversation Management:**
    For multi-turn interactions:

    .. code-block:: python

        messages = [
            {"role": "system", "content": "You are a helpful science tutor."},
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "Photosynthesis is..."},
            {"role": "user", "content": "Can you give an example?"}
        ]
        
        response = model.chat(messages)
        print(response.text)

**Batch Processing:**
    For efficient multi-prompt evaluation:

    .. code-block:: python

        prompts = [
            "Translate 'hello' to Spanish",
            "What is 2+2?", 
            "Name a primary color"
        ]
        
        responses = model.batch_generate(prompts, max_concurrent=3)
        for i, response in enumerate(responses):
            print(f"Prompt {i}: {response.text}")

**Configuration Updates:**
    For dynamic parameter adjustment:

    .. code-block:: python

        # Check current configuration
        current_config = model.get_config()
        print(f"Current temperature: {current_config['temperature']}")
        
        # Update specific parameters
        model.update_config(temperature=0.8, max_tokens=1000)
        
        # Configuration changes apply to subsequent requests
        response = model.generate("Be creative and write a poem")

Testing and Development
-----------------------

**Dummy Model Implementation:**
    The :class:`DummyModel` provides a complete implementation for testing:

    .. code-block:: python

        from scramblebench.llm.model_interface import DummyModel
        
        # Create dummy model with predefined responses
        dummy = DummyModel(
            model_name="test-model",
            responses=["Response 1", "Response 2", "Response 3"]
        )
        dummy.initialize()
        
        # Test evaluation code without API calls
        response = dummy.generate("Any prompt")
        print(response.text)  # Will cycle through predefined responses

**Validation Methods:**
    Built-in validation prevents common integration issues:

    .. code-block:: python

        # Validate prompts before submission
        prompt = "Very long prompt..." * 1000
        if model.validate_prompt(prompt):
            response = model.generate(prompt)
        else:
            print("Prompt too long for model context window")
        
        # Estimate costs before batch operations
        total_tokens = sum(model.estimate_tokens(p) for p in prompts)
        estimated_cost = total_tokens * cost_per_token
        print(f"Estimated cost: ${estimated_cost:.2f}")

Factory Pattern Integration
---------------------------

**Model Registration:**
    Providers register with the factory for discovery:

    .. code-block:: python

        from scramblebench.llm.model_interface import ModelFactory
        
        # Register custom implementation
        ModelFactory.register_model('custom_provider', CustomModelClass)
        
        # Create models through factory
        model = ModelFactory.create_model(
            provider='custom_provider',
            model_name='custom-model-1',
            config=config
        )

**Available Providers:**
    Query registered providers and capabilities:

    .. code-block:: python

        providers = ModelFactory.list_providers()
        print(f"Available providers: {providers}")

Error Handling Patterns
------------------------

**Response Validation:**
    Always check for errors in responses:

    .. code-block:: python

        response = model.generate("test prompt")
        
        if response.error:
            # Handle specific error types
            if "rate limit" in response.error.lower():
                print("Rate limited - waiting before retry")
                time.sleep(60)
            elif "timeout" in response.error.lower():
                print("Request timeout - model may be overloaded")
            else:
                print(f"Generation failed: {response.error}")
        else:
            # Process successful response
            process_response(response.text)

**Initialization Validation:**
    Verify model setup before use:

    .. code-block:: python

        model = SomeModelImplementation("model-name")
        
        if not model.initialize():
            print("Model initialization failed")
            # Check API keys, network connectivity, etc.
            return
        
        if not model.is_initialized:
            print("Model not ready for requests")
            return
        
        # Safe to use model
        response = model.generate("test")

See Also
--------

* :class:`OpenRouterClient` - Concrete implementation for OpenRouter API
* :doc:`../user_guide/configuration` - Model configuration guide  
* :doc:`../tutorials/custom_models` - Implementing custom model providers
* :doc:`../api/llm` - Complete LLM API reference
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging


class ModelType(Enum):
    """
    Classification enumeration for different types of language models.
    
    This enumeration categorizes models based on their primary interaction patterns
    and capabilities, enabling appropriate prompt formatting and expectation setting
    for different model architectures.
    
    :cvar CHAT: Chat-based conversational models that expect structured message formats
    :type CHAT: str
    :cvar COMPLETION: Text completion models that continue from prompts
    :type COMPLETION: str  
    :cvar INSTRUCTION: Instruction-following models optimized for task completion
    :type INSTRUCTION: str
    :cvar CODE: Code-specialized models for programming tasks
    :type CODE: str
    
    Examples:
        Model type classification:
        
        .. code-block:: python
        
            # Chat models (GPT-4, Claude, etc.)
            if model.model_type == ModelType.CHAT:
                messages = [{"role": "user", "content": prompt}]
                response = model.chat(messages)
            
            # Completion models (GPT-3, etc.) 
            elif model.model_type == ModelType.COMPLETION:
                response = model.generate(prompt)
            
            # Instruction models
            elif model.model_type == ModelType.INSTRUCTION:
                formatted_prompt = f"Instruction: {prompt}\\nResponse:"
                response = model.generate(formatted_prompt)
    
    Note:
        The model type affects prompt formatting and interaction patterns but
        all models expose the same interface methods for consistency.
    """
    CHAT = "chat"  # Chat-based models (GPT-4, Claude, etc.)
    COMPLETION = "completion"  # Text completion models (GPT-3, etc.)
    INSTRUCTION = "instruction"  # Instruction-following models
    CODE = "code"  # Code-specific models (Codex, etc.)


@dataclass
class ModelResponse:
    """
    Standardized response container for all model interactions.
    
    This dataclass provides a unified structure for capturing model outputs,
    metadata, and error information across different providers and model types.
    It enables consistent response handling and debugging capabilities.
    
    :param text: The primary generated text content from the model
    :type text: str
    :param metadata: Additional response information including tokens, timing, and model details
    :type metadata: Dict[str, Any]
    :param raw_response: Complete unprocessed response from the model API for debugging
    :type raw_response: Optional[Any]
    :param error: Error message if the generation failed, None if successful
    :type error: Optional[str]
    
    Common Metadata Fields:
        Standard metadata fields across providers:
        
        * ``model`` (str) - Model identifier used for generation
        * ``response_time`` (float) - Generation latency in seconds
        * ``prompt_tokens`` (int) - Number of tokens in the input prompt
        * ``completion_tokens`` (int) - Number of tokens in the generated response
        * ``total_tokens`` (int) - Sum of prompt and completion tokens
        * ``finish_reason`` (str) - Reason generation stopped (length, stop, etc.)
        * ``request_data`` (dict) - Parameters used for the request
    
    Examples:
        Processing successful responses:
        
        .. code-block:: python
        
            response = model.generate("Explain quantum computing")
            
            if response.error:
                print(f"Generation failed: {response.error}")
            else:
                print(f"Generated text: {response.text}")
                print(f"Tokens used: {response.metadata['total_tokens']}")
                print(f"Response time: {response.metadata['response_time']:.2f}s")
        
        Accessing provider-specific information:
        
        .. code-block:: python
        
            # Access raw response for detailed debugging
            if response.raw_response:
                provider_data = response.raw_response
                print(f"Provider response: {provider_data}")
            
            # Check generation completion reason
            finish_reason = response.metadata.get('finish_reason')
            if finish_reason == 'length':
                print("Response truncated due to length limit")
            elif finish_reason == 'stop':
                print("Response completed naturally")
    
    Error Handling:
        Response validation and error checking:
        
        .. code-block:: python
        
            def process_response(response: ModelResponse) -> str:
                if response.error:
                    # Handle specific error types
                    if "rate_limit" in response.error.lower():
                        raise RateLimitError(f"Rate limited: {response.error}")
                    elif "timeout" in response.error.lower():
                        raise TimeoutError(f"Request timeout: {response.error}")
                    else:
                        raise RuntimeError(f"Generation failed: {response.error}")
                
                # Validate successful response
                if not response.text.strip():
                    raise ValueError("Empty response generated")
                
                return response.text
    
    Note:
        Always check the ``error`` field before processing ``text`` content.
        The ``metadata`` dictionary may contain provider-specific fields
        beyond the standard fields listed above.
    """
    text: str
    metadata: Dict[str, Any]
    raw_response: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ModelConfig:
    """
    Universal configuration for model behavior and generation parameters.
    
    This dataclass provides a standardized way to configure model behavior
    across different providers and model types. All parameters use common
    ranges and semantics to ensure consistent behavior regardless of the
    underlying model implementation.
    
    :param temperature: Controls randomness in generation (0.0-2.0)
    :type temperature: float
    :param max_tokens: Maximum number of tokens to generate
    :type max_tokens: int
    :param top_p: Nucleus sampling threshold for token selection (0.0-1.0)
    :type top_p: float
    :param frequency_penalty: Penalty for token frequency to reduce repetition (-2.0-2.0)
    :type frequency_penalty: float
    :param presence_penalty: Penalty for token presence to encourage diversity (-2.0-2.0)  
    :type presence_penalty: float
    :param stop_sequences: List of strings that will stop generation when encountered
    :type stop_sequences: Optional[List[str]]
    :param timeout: Request timeout in seconds
    :type timeout: int
    
    Sampling Parameters:
        **Temperature (0.0-2.0):**
            Controls the randomness of token selection:
            
            * ``0.0`` - Deterministic, always selects most likely token
            * ``0.1-0.3`` - Very focused, minimal creativity
            * ``0.7`` - Balanced creativity and coherence (default)
            * ``1.0`` - High creativity, more unpredictable
            * ``1.5-2.0`` - Maximum creativity, may be incoherent
        
        **Top-p / Nucleus Sampling (0.0-1.0):**
            Limits token selection to top probability mass:
            
            * ``0.1`` - Very conservative, only highest probability tokens
            * ``0.9`` - Balanced selection from likely tokens
            * ``1.0`` - Consider all tokens (default)
    
    Penalty Systems:
        **Frequency Penalty (-2.0-2.0):**
            Reduces likelihood of repeating tokens based on their frequency:
            
            * ``0.0`` - No frequency penalty (default)
            * ``0.1-0.5`` - Mild reduction in repetition
            * ``1.0-2.0`` - Strong penalty against repetitive patterns
        
        **Presence Penalty (-2.0-2.0):**
            Reduces likelihood of repeating any token that has appeared:
            
            * ``0.0`` - No presence penalty (default)
            * ``0.1-0.5`` - Encourage topic diversity
            * ``1.0-2.0`` - Strong penalty for any repeated content
    
    Examples:
        Configuration for different use cases:
        
        .. code-block:: python
        
            # Deterministic generation for benchmarks
            benchmark_config = ModelConfig(
                temperature=0.0,        # No randomness
                max_tokens=500,         # Reasonable limit
                frequency_penalty=0.1   # Slight anti-repetition
            )
            
            # Creative text generation
            creative_config = ModelConfig(
                temperature=0.8,        # High creativity
                top_p=0.9,              # Diverse token selection
                presence_penalty=0.3,   # Encourage topic diversity
                max_tokens=2000         # Longer responses
            )
            
            # Code generation optimized
            code_config = ModelConfig(
                temperature=0.2,        # Low randomness for precision
                frequency_penalty=0.0,  # Allow code repetition
                stop_sequences=["```"]  # Stop at code block end
            )
            
            # Conversation with length limits
            chat_config = ModelConfig(
                temperature=0.7,        # Balanced
                max_tokens=150,         # Short responses
                stop_sequences=["User:", "Human:"]  # Stop at conversation markers
            )
    
    Operational Controls:
        **Token Limits:**
            Set appropriate limits for cost and performance:
            
            * Consider model's context window size
            * Balance response quality vs. token cost
            * Account for prompt tokens in total budget
        
        **Timeout Settings:**
            Configure timeouts based on use case:
            
            * ``10-30s`` - Interactive applications
            * ``60-120s`` - Batch processing
            * ``300s+`` - Complex generation tasks
        
        **Stop Sequences:**
            Define custom termination conditions:
            
            * Format-specific stops: ``["```", "---", "</code>"]``
            * Conversation stops: ``["Human:", "Assistant:", "User:"]``
            * Content stops: ``["\n\n\n", "THE_END", "STOP"]``
    
    Provider Compatibility:
        Configuration mapping across providers:
        
        .. code-block:: python
        
            config = ModelConfig(temperature=0.8, max_tokens=1000)
            
            # OpenAI API format
            openai_params = {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty
            }
            
            # Anthropic API format  
            anthropic_params = {
                "temperature": config.temperature,
                "max_tokens_to_sample": config.max_tokens,
                "top_p": config.top_p
            }
    
    Validation:
        Configuration validation and bounds checking:
        
        .. code-block:: python
        
            def validate_config(config: ModelConfig) -> bool:
                if not 0.0 <= config.temperature <= 2.0:
                    raise ValueError("Temperature must be between 0.0 and 2.0")
                
                if config.max_tokens <= 0:
                    raise ValueError("max_tokens must be positive")
                
                if not 0.0 <= config.top_p <= 1.0:
                    raise ValueError("top_p must be between 0.0 and 1.0")
                
                return True
    
    Note:
        Provider implementations may not support all parameters.
        Unsupported parameters are typically ignored with a warning logged.
        Check provider documentation for parameter support details.
    """
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    timeout: int = 30


class ModelInterface(ABC):
    """
    Abstract base class defining the standard interface for all LLM implementations.
    
    This interface provides a unified, provider-agnostic way to interact with different
    Large Language Models while hiding implementation details and provider-specific APIs.
    All model implementations must inherit from this class and implement its abstract methods
    to ensure consistent behavior across the ScrambleBench evaluation framework.
    
    Design Principles:
        **Consistency:** All models expose identical methods with consistent semantics
        **Error Handling:** Structured error reporting through ModelResponse objects
        **Configuration:** Universal configuration management via ModelConfig
        **Extensibility:** Simple registration mechanism for new providers
        **Performance:** Built-in batch processing and async support capabilities
    
    Implementation Requirements:
        Subclasses must implement all abstract methods:
        
        * :meth:`initialize` - Establish connections and validate configuration
        * :meth:`generate` - Single-prompt text generation
        * :meth:`chat` - Multi-turn conversation handling
        * :meth:`batch_generate` - Efficient batch processing
        * :meth:`get_model_info` - Provider and capability metadata
        * :meth:`estimate_tokens` - Token counting for cost estimation
    
    State Management:
        The interface maintains several important state properties:
        
        * ``model_name`` - Identifier for the specific model
        * ``config`` - Current configuration parameters
        * ``_initialized`` - Initialization status flag
        * ``_model_type`` - Classification of model capabilities
        * ``logger`` - Logging instance for debugging and monitoring
    
    Lifecycle:
        Standard model lifecycle and usage pattern:
        
        .. code-block:: python
        
            # 1. Create model instance
            model = SomeModelImplementation(
                model_name="provider/model-name",
                config=ModelConfig(temperature=0.0)
            )
            
            # 2. Initialize (connect, authenticate, validate)
            if not model.initialize():
                raise RuntimeError("Model initialization failed")
            
            # 3. Verify ready state
            assert model.is_initialized
            
            # 4. Use model for generation
            response = model.generate("Explain quantum computing")
            
            # 5. Process response
            if response.error:
                handle_error(response.error)
            else:
                process_text(response.text)
    
    Error Handling:
        Robust error handling patterns:
        
        .. code-block:: python
        
            try:
                response = model.generate(prompt)
                
                if response.error:
                    # Handle model-level errors
                    if "rate_limit" in response.error.lower():
                        # Implement backoff strategy
                        time.sleep(60)
                        response = model.generate(prompt)
                    else:
                        logger.error(f"Generation failed: {response.error}")
                        return None
                
                return response.text
                
            except Exception as e:
                # Handle implementation-level errors
                logger.error(f"Model interface error: {e}")
                return None
    
    Configuration Management:
        Dynamic configuration updates:
        
        .. code-block:: python
        
            # Get current configuration
            current = model.get_config()
            print(f"Current temperature: {current['temperature']}")
            
            # Update specific parameters
            model.update_config(
                temperature=0.8,
                max_tokens=2000,
                stop_sequences=["END", "STOP"]
            )
            
            # Changes apply to subsequent requests
            response = model.generate("Write a creative story")
    
    Batch Processing:
        Efficient handling of multiple prompts:
        
        .. code-block:: python
        
            prompts = [
                "Translate 'hello' to French",
                "What is 2+2?",
                "Name a programming language"
            ]
            
            # Batch generation with error handling
            responses = model.batch_generate(prompts, max_concurrent=5)
            
            for i, response in enumerate(responses):
                if response.error:
                    print(f"Prompt {i} failed: {response.error}")
                else:
                    print(f"Prompt {i}: {response.text}")
    
    Conversation Support:
        Multi-turn conversation management:
        
        .. code-block:: python
        
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ]
            
            response = model.chat(conversation)
            
            if not response.error:
                # Add assistant response to conversation
                conversation.append({
                    "role": "assistant", 
                    "content": response.text
                })
                
                # Continue conversation
                conversation.append({
                    "role": "user",
                    "content": "Give me an example."
                })
                
                follow_up = model.chat(conversation)
    
    Utility Methods:
        Helper methods for common operations:
        
        .. code-block:: python
        
            # Simple query interface
            answer = model.query("What is the capital of France?")
            
            # Token estimation for cost planning
            prompt = "Long prompt text..."
            tokens = model.estimate_tokens(prompt)
            print(f"Estimated tokens: {tokens}")
            
            # Prompt validation
            very_long_prompt = "..." * 10000
            if model.validate_prompt(very_long_prompt):
                response = model.generate(very_long_prompt)
            else:
                print("Prompt exceeds model limits")
            
            # Model information
            info = model.get_model_info()
            print(f"Max context: {info['max_context_length']}")
            
            # Callable interface
            result = model("Quick question: What is 2+2?")
    
    Provider Registration:
        Integration with the ModelFactory system:
        
        .. code-block:: python
        
            class CustomModelProvider(ModelInterface):
                # Implement all abstract methods
                def initialize(self) -> bool:
                    # Custom initialization logic
                    return True
                
                def generate(self, prompt: str, **kwargs) -> ModelResponse:
                    # Custom generation logic
                    return ModelResponse(text="Custom response", metadata={})
                
                # ... implement other abstract methods
            
            # Register with factory
            ModelFactory.register_model('custom', CustomModelProvider)
            
            # Create through factory
            model = ModelFactory.create_model(
                provider='custom',
                model_name='custom-model-1'
            )
    
    Thread Safety:
        Implementation considerations for concurrent usage:
        
        .. code-block:: python
        
            # Models should be thread-safe for read operations
            # Configuration updates should be synchronized
            import threading
            
            config_lock = threading.Lock()
            
            def safe_config_update(model, **kwargs):
                with config_lock:
                    model.update_config(**kwargs)
            
            # Batch processing is typically async-safe
            responses = await model.batch_generate(prompts)
    
    See Also:
        * :class:`ModelResponse` - Standard response container
        * :class:`ModelConfig` - Configuration management
        * :class:`ModelFactory` - Model creation and registration
        * :class:`OpenRouterClient` - Concrete implementation example
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model interface.
        
        Args:
            model_name: Name/identifier of the model
            config: Model configuration (creates default if None)
            logger: Logger instance (creates default if None)
        """
        self.model_name = model_name
        self.config = config or ModelConfig()
        self.logger = logger or logging.getLogger(f"scramblebench.model.{model_name}")
        self._model_type: Optional[ModelType] = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Get the model name."""
        return self.model_name
    
    @property
    def model_type(self) -> Optional[ModelType]:
        """Get the model type."""
        return self._model_type
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self._initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the model and establish connections.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse containing generated text and metadata
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """
        Have a conversation with the model using a message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse containing the model's reply
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ModelResponse]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of ModelResponse objects
        """
        pass
    
    def query(self, prompt: str, **kwargs) -> str:
        """
        Simple query interface that returns just the generated text.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        response = self.generate(prompt, **kwargs)
        if response.error:
            raise RuntimeError(f"Model query failed: {response.error}")
        return response.text
    
    def update_config(self, **kwargs) -> None:
        """
        Update model configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Dictionary containing current configuration
        """
        return {
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'top_p': self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty,
            'stop_sequences': self.config.stop_sequences,
            'timeout': self.config.timeout
        }
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information (name, type, capabilities, etc.)
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated token count
        """
        pass
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate that a prompt is acceptable for this model.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            True if prompt is valid, False otherwise
        """
        if not prompt or not prompt.strip():
            self.logger.warning("Empty prompt provided")
            return False
        
        # Check token count
        estimated_tokens = self.estimate_tokens(prompt)
        max_context = self.get_model_info().get('max_context_length', 4096)
        
        if estimated_tokens > max_context - self.config.max_tokens:
            self.logger.warning(
                f"Prompt too long: {estimated_tokens} tokens "
                f"(max context: {max_context}, max_tokens: {self.config.max_tokens})"
            )
            return False
        
        return True
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Make the model callable for simple usage.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        return self.query(prompt, **kwargs)
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"type={self.model_type.value if self.model_type else 'unknown'}, "
            f"initialized={self.is_initialized})"
        )


class DummyModel(ModelInterface):
    """
    Dummy model implementation for testing and development.
    
    This model returns predefined responses and can be used for
    testing benchmark functionality without actual LLM API calls.
    """
    
    def __init__(
        self,
        model_name: str = "dummy",
        responses: Optional[List[str]] = None,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the dummy model.
        
        Args:
            model_name: Name for the dummy model
            responses: List of predefined responses (cycles through them)
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(model_name, config, logger)
        self.responses = responses or ["This is a dummy response."]
        self.response_index = 0
        self._model_type = ModelType.CHAT
    
    def initialize(self) -> bool:
        """Initialize the dummy model."""
        self._initialized = True
        self.logger.info("Dummy model initialized")
        return True
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a dummy response."""
        if not self._initialized:
            return ModelResponse(
                text="",
                metadata={},
                error="Model not initialized"
            )
        
        # Get next response
        response_text = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        
        return ModelResponse(
            text=response_text,
            metadata={
                'prompt_length': len(prompt),
                'response_length': len(response_text),
                'response_index': self.response_index - 1,
                'model': self.model_name
            }
        )
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate a dummy chat response."""
        # Convert messages to a single prompt
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        return self.generate(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate dummy responses for multiple prompts."""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get dummy model information."""
        return {
            'name': self.model_name,
            'type': self.model_type.value,
            'provider': 'dummy',
            'max_context_length': 4096,
            'max_output_tokens': 2048,
            'supports_chat': True,
            'supports_batch': True
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using a simple heuristic."""
        # Very rough estimate: ~4 characters per token
        return len(text) // 4
    
    def set_responses(self, responses: List[str]) -> None:
        """Set new predefined responses."""
        self.responses = responses
        self.response_index = 0
        self.logger.info(f"Updated responses: {len(responses)} responses")
    
    def add_response(self, response: str) -> None:
        """Add a new response to the list."""
        self.responses.append(response)
        self.logger.debug(f"Added response: {response[:50]}...")


class ModelFactory:
    """
    Factory for creating model instances.
    
    Provides a centralized way to create and configure different model types
    while handling provider-specific initialization details.
    """
    
    _model_classes = {}
    
    @classmethod
    def register_model(cls, provider: str, model_class):
        """
        Register a model class for a provider.
        
        Args:
            provider: Provider name (e.g., 'openrouter', 'dummy')
            model_class: Model class to register
        """
        cls._model_classes[provider] = model_class
    
    @classmethod
    def create_model(
        cls,
        provider: str,
        model_name: str,
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> ModelInterface:
        """
        Create a model instance.
        
        Args:
            provider: Provider name
            model_name: Name of the model
            config: Model configuration
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized model instance
        """
        if provider not in cls._model_classes:
            raise ValueError(f"Unknown provider: {provider}")
        
        model_class = cls._model_classes[provider]
        model = model_class(model_name=model_name, config=config, **kwargs)
        
        if not model.initialize():
            raise RuntimeError(f"Failed to initialize model: {model_name}")
        
        return model
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._model_classes.keys())


# Register the dummy model
ModelFactory.register_model('dummy', DummyModel)