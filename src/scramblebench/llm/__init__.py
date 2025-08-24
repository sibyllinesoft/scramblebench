"""
LLM Integration Layer for ScrambleBench
========================================

This module provides a comprehensive integration layer for Large Language Models (LLMs)
used in benchmark evaluation. It defines standardized interfaces and implementations
for consistent interaction with diverse model providers while abstracting away
provider-specific API details.

Key Components
--------------

**Abstract Interfaces:**
    * :class:`ModelInterface` - Base interface for all LLM implementations
    * :class:`ModelResponse` - Standardized response format across providers
    * :class:`ModelConfig` - Universal configuration for model behavior

**Concrete Implementations:**
    * :class:`OpenRouterClient` - Integration with OpenRouter API for multi-provider access
    * :class:`OllamaClient` - Integration with Ollama for local LLM inference
    * :class:`DummyModel` - Testing implementation for development and validation

**Factory Pattern:**
    * :class:`ModelFactory` - Centralized model creation and registration system

Architecture Overview
---------------------

The LLM integration follows a plugin-based architecture where each provider
implements the :class:`ModelInterface` and registers with the :class:`ModelFactory`.
This enables:

- **Provider Independence**: Switch between providers without changing evaluation code
- **Consistent API**: All models expose the same interface regardless of underlying implementation
- **Extensibility**: New providers can be added by implementing the interface
- **Configuration Management**: Unified configuration system across all providers

Usage Patterns
---------------

**Basic Usage:**
    Create and use models directly:

    .. code-block:: python

        from scramblebench.llm import OpenRouterClient, ModelConfig
        
        # Configure model behavior
        config = ModelConfig(temperature=0.0, max_tokens=1000)
        
        # Create model instance
        model = OpenRouterClient("openai/gpt-4", config=config)
        model.initialize()
        
        # Generate text
        response = model.generate("Explain quantum computing")
        print(response.text)

**Factory Usage:**
    Use the factory for dynamic model creation:

    .. code-block:: python

        from scramblebench.llm import ModelFactory
        
        # Create model through factory
        model = ModelFactory.create_model(
            provider="openrouter",
            model_name="anthropic/claude-3-sonnet", 
            config=config
        )
        
        # Batch processing
        prompts = ["Question 1", "Question 2", "Question 3"]
        responses = model.batch_generate(prompts)

**Evaluation Integration:**
    Models integrate seamlessly with benchmark evaluation:

    .. code-block:: python

        from scramblebench.core import BaseBenchmark
        from scramblebench.llm import create_openrouter_client
        
        # Create model for evaluation
        model = create_openrouter_client("openai/gpt-4")
        
        # Use in benchmark evaluation
        benchmark = SomeBenchmark(...)
        results = benchmark.run(model, max_samples=100)

Provider Support
----------------

**OpenRouter Integration:**
    Access to 50+ models through unified API:
    
    - OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
    - Anthropic (Claude 3 Opus, Sonnet, Haiku)
    - Meta (Llama 2, Code Llama)
    - Google (Gemini Pro, PaLM)
    - Mistral (Mixtral, Mistral 7B)
    - And many more...

**Ollama Integration:**
    Local inference with open-source models:
    
    - Meta Llama (3.2 1B, 3B, 8B variants)
    - Microsoft Phi (3.8B, 14B efficient models)
    - Google Gemma (2B, 9B optimized models) 
    - Code Llama (specialized for programming)
    - No API keys required, complete privacy

**Rate Limiting & Error Handling:**
    Built-in protection for production use:
    
    - Automatic retry logic with exponential backoff
    - Rate limiting to respect API quotas
    - Comprehensive error handling and logging
    - Request/response validation

Configuration & Environment
---------------------------

**API Keys:**
    Configure via environment variables:
    
    .. code-block:: bash
    
        export OPENROUTER_API_KEY="your_api_key_here"

**Default Configuration:**
    Models use sensible defaults but can be customized:
    
    .. code-block:: python
    
        config = ModelConfig(
            temperature=0.7,      # Creativity/randomness (0.0-2.0)
            max_tokens=1000,      # Maximum response length  
            top_p=1.0,            # Nucleus sampling parameter
            frequency_penalty=0.0, # Reduce repetition
            presence_penalty=0.0,  # Encourage topic diversity
            timeout=30            # Request timeout (seconds)
        )

Error Handling & Debugging
---------------------------

**Logging Integration:**
    Comprehensive logging for debugging and monitoring:
    
    .. code-block:: python
    
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        model = OpenRouterClient("openai/gpt-4")
        # Model operations will now log detailed information

**Error Recovery:**
    Robust error handling with informative messages:
    
    .. code-block:: python
    
        response = model.generate("test prompt")
        if response.error:
            print(f"Generation failed: {response.error}")
        else:
            print(f"Success: {response.text}")

**Validation:**
    Built-in validation prevents common issues:
    
    .. code-block:: python
    
        # Automatic prompt validation
        is_valid = model.validate_prompt(very_long_prompt)
        
        # Token estimation
        token_count = model.estimate_tokens(prompt)

See Also
--------

* :doc:`../api/llm` - Complete LLM API reference
* :doc:`../user_guide/configuration` - Configuration guide
* :doc:`../tutorials/custom_models` - Adding custom model providers
* :doc:`../examples/basic_usage` - Usage examples
"""

from scramblebench.llm.model_interface import ModelInterface, ModelResponse, ModelConfig, ModelType, ModelFactory, DummyModel
from scramblebench.llm.openrouter_client import OpenRouterClient, create_openrouter_client
from scramblebench.llm.ollama_client import OllamaClient, create_ollama_client
from scramblebench.llm.model_adapter import ModelAdapter, query_model, batch_query_model

__all__ = [
    # Core interfaces and types
    "ModelInterface", "ModelResponse", "ModelConfig", "ModelType", "ModelFactory", "DummyModel",
    
    # Provider implementations
    "OpenRouterClient", "OllamaClient", 
    
    # Factory functions
    "create_openrouter_client", "create_ollama_client",
    
    # Model adapter
    "ModelAdapter", "query_model", "batch_query_model"
]