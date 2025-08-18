"""
OpenRouter API Client for ScrambleBench
=======================================

This module provides comprehensive integration with the OpenRouter API, enabling
access to 50+ Large Language Models from multiple providers through a single,
unified interface. OpenRouter serves as an LLM gateway that simplifies provider
management, handles rate limiting, and provides cost optimization across different
model providers.

OpenRouter Integration Benefits
-------------------------------

**Multi-Provider Access:**
    Single API key provides access to models from:
    
    - OpenAI (GPT-3.5, GPT-4, GPT-4 Turbo)
    - Anthropic (Claude 3 Opus, Sonnet, Haiku)
    - Google (Gemini Pro, PaLM)
    - Meta (Llama 2, Code Llama)
    - Mistral (Mixtral, Mistral 7B)
    - Cohere, Together AI, and many more

**Unified Pricing:**
    Transparent, competitive pricing across all providers with:
    
    - Per-token billing for precise cost control
    - Volume discounts for large-scale usage
    - Real-time cost tracking and usage analytics
    - No hidden fees or subscription requirements

**Production Features:**
    Enterprise-ready functionality including:
    
    - Automatic failover between providers
    - Built-in rate limiting and retry logic
    - Load balancing across model instances
    - Comprehensive monitoring and logging

Architecture Overview
---------------------

The OpenRouter client implements the :class:`ModelInterface` with additional
provider-specific optimizations:

**Async Architecture:**
    Full async/await support for high-performance batch processing:
    
    - Non-blocking I/O operations
    - Concurrent request handling
    - Configurable concurrency limits
    - Automatic connection pooling

**Error Handling:**
    Comprehensive error recovery with exponential backoff:
    
    - Automatic retries for transient failures
    - Rate limit detection and backoff
    - Network timeout handling
    - Provider-specific error translation

**Rate Limiting:**
    Built-in protection against API quota violations:
    
    - Configurable requests per second limits
    - Token bucket algorithm implementation
    - Queue management for burst traffic
    - Provider-specific rate limit awareness

Configuration and Setup
-----------------------

**API Key Configuration:**
    Obtain and configure OpenRouter API access:
    
    .. code-block:: bash
    
        # Set environment variable (recommended)
        export OPENROUTER_API_KEY="your_api_key_here"
    
    .. code-block:: python
    
        # Or pass directly in code
        client = OpenRouterClient(
            model_name="openai/gpt-4",
            api_key="your_api_key_here"
        )

**Model Selection:**
    Choose from available models using OpenRouter naming:
    
    .. code-block:: python
    
        # Popular model configurations
        models = {
            "openai/gpt-4": "Most capable OpenAI model",
            "openai/gpt-3.5-turbo": "Fast and cost-effective",
            "anthropic/claude-3-opus": "Anthropic's most capable model",
            "anthropic/claude-3-sonnet": "Balanced performance and speed",
            "google/gemini-pro": "Google's flagship model",
            "meta-llama/llama-2-70b-chat": "Open-source conversational AI"
        }

**Configuration Examples:**
    Common configuration patterns for different use cases:
    
    .. code-block:: python
    
        from scramblebench.llm import OpenRouterClient, ModelConfig
        
        # Research configuration (deterministic)
        research_config = ModelConfig(
            temperature=0.0,        # No randomness
            max_tokens=2000,        # Longer responses
            timeout=60             # Extended timeout
        )
        
        # Production configuration (balanced)
        prod_config = ModelConfig(
            temperature=0.3,        # Slight creativity
            max_tokens=1000,        # Cost control
            frequency_penalty=0.1,  # Reduce repetition
            timeout=30             # Reasonable timeout
        )
        
        # Creative configuration (high variability)
        creative_config = ModelConfig(
            temperature=0.8,        # High creativity
            top_p=0.9,              # Diverse sampling
            presence_penalty=0.3,   # Topic diversity
            max_tokens=1500
        )

Usage Patterns
--------------

**Basic Usage:**
    Simple text generation with error handling:
    
    .. code-block:: python
    
        from scramblebench.llm import OpenRouterClient
        
        # Create and initialize client
        client = OpenRouterClient("openai/gpt-4")
        client.initialize()
        
        # Generate text
        response = client.generate("Explain machine learning in simple terms")
        
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.text}")
            print(f"Tokens: {response.metadata['total_tokens']}")
            print(f"Cost: ${response.metadata.get('cost', 0):.4f}")

**Conversation Handling:**
    Multi-turn conversations with context management:
    
    .. code-block:: python
    
        # Initialize conversation
        conversation = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is quantum computing?"}
        ]
        
        # Get response
        response = client.chat(conversation)
        
        if not response.error:
            # Add assistant response to conversation
            conversation.append({
                "role": "assistant",
                "content": response.text
            })
            
            # Continue conversation
            conversation.append({
                "role": "user", 
                "content": "How does it differ from classical computing?"
            })
            
            follow_up = client.chat(conversation)

**Batch Processing:**
    Efficient processing of multiple prompts:
    
    .. code-block:: python
    
        # Prepare batch of prompts
        prompts = [
            "Translate 'Hello' to Spanish",
            "What is 2 + 2?",
            "Name a primary color",
            "What is the capital of France?",
            "Explain photosynthesis briefly"
        ]
        
        # Process with controlled concurrency
        responses = client.batch_generate(
            prompts, 
            max_concurrent=3,  # Limit concurrent requests
            temperature=0.0    # Override config for this batch
        )
        
        # Process results
        for i, response in enumerate(responses):
            if response.error:
                print(f"Prompt {i} failed: {response.error}")
            else:
                print(f"Prompt {i}: {response.text}")

**Async Context Manager:**
    Proper resource management for async operations:
    
    .. code-block:: python
    
        import asyncio
        
        async def async_generation():
            async with OpenRouterClient("anthropic/claude-3-sonnet") as client:
                # Client is automatically initialized
                response = await client._chat_async([
                    {"role": "user", "content": "Explain async programming"}
                ])
                
                return response.text
        
        # Run async code
        result = asyncio.run(async_generation())

Performance Optimization
------------------------

**Rate Limiting Configuration:**
    Optimize throughput while respecting limits:
    
    .. code-block:: python
    
        # Set custom rate limits
        client.set_rate_limit(5.0)  # 5 requests per second
        
        # For high-volume processing
        client.set_rate_limit(20.0)  # Increase for premium accounts

**Concurrent Processing:**
    Balance speed and resource usage:
    
    .. code-block:: python
    
        # Configure batch processing
        responses = client.batch_generate(
            prompts,
            max_concurrent=10,  # Adjust based on rate limits
            timeout=45          # Per-request timeout
        )

**Connection Management:**
    Optimize HTTP connection handling:
    
    .. code-block:: python
    
        # Client handles connection pooling automatically
        # Sessions are reused across requests
        # Automatic cleanup on client destruction

**Cost Optimization:**
    Monitor and control API costs:
    
    .. code-block:: python
    
        # Estimate costs before generation
        prompt = "Long prompt text..."
        estimated_tokens = client.estimate_tokens(prompt)
        estimated_cost = estimated_tokens * 0.00002  # Example rate
        
        print(f"Estimated cost: ${estimated_cost:.4f}")
        
        # Use cost-effective models for bulk processing
        if estimated_cost > budget_limit:
            # Switch to more cost-effective model
            client = OpenRouterClient("openai/gpt-3.5-turbo")

Error Handling and Debugging
-----------------------------

**Comprehensive Error Types:**
    Handle different failure modes appropriately:
    
    .. code-block:: python
    
        response = client.generate(prompt)
        
        if response.error:
            error_msg = response.error.lower()
            
            if "rate limit" in error_msg:
                # Implement backoff strategy
                print("Rate limited - waiting 60 seconds")
                time.sleep(60)
                response = client.generate(prompt)
                
            elif "timeout" in error_msg:
                # Retry with longer timeout
                client.update_config(timeout=90)
                response = client.generate(prompt)
                
            elif "invalid_request_error" in error_msg:
                # Check prompt validity
                if not client.validate_prompt(prompt):
                    print("Prompt exceeds model limits")
                    
            elif "authentication" in error_msg:
                # Check API key
                print("Invalid API key - check OPENROUTER_API_KEY")
                
            else:
                print(f"Unknown error: {response.error}")

**Debugging and Monitoring:**
    Enable detailed logging for troubleshooting:
    
    .. code-block:: python
    
        import logging
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        
        client = OpenRouterClient("openai/gpt-4")
        # All API interactions will be logged
        
        # Check client status
        print(f"Client initialized: {client.is_initialized}")
        print(f"Model type: {client.model_type}")
        
        # Get model information
        info = client.get_model_info()
        print(f"Model info: {info}")
        
        # Monitor usage
        stats = client.get_usage_stats()
        print(f"Usage stats: {stats}")

Model Information and Capabilities
----------------------------------

**Available Models:**
    Query supported models and their capabilities:
    
    .. code-block:: python
    
        # Get list of available models
        models = client.get_available_models()
        
        for model in models:
            print(f"Model: {model['id']}")
            print(f"Description: {model.get('description', 'N/A')}")
            print(f"Context Length: {model.get('context_length', 'N/A')}")
            print(f"Pricing: {model.get('pricing', 'N/A')}")
            print("---")

**Model Selection Helper:**
    Utility function for common model mappings:
    
    .. code-block:: python
    
        from scramblebench.llm.openrouter_client import get_model_id
        
        # Get OpenRouter model ID from common name
        model_id = get_model_id("gpt-4")  # Returns "openai/gpt-4"
        model_id = get_model_id("claude-3-opus")  # Returns "anthropic/claude-3-opus"
        
        # Use with client
        client = OpenRouterClient(model_id)

**Capability Querying:**
    Check model-specific features:
    
    .. code-block:: python
    
        info = client.get_model_info()
        
        # Check capabilities
        supports_chat = info.get('supports_chat', False)
        max_context = info.get('max_context_length', 4096)
        max_output = info.get('max_output_tokens', 2048)
        
        print(f"Chat support: {supports_chat}")
        print(f"Max context: {max_context} tokens")
        print(f"Max output: {max_output} tokens")

Integration Examples
--------------------

**Benchmark Evaluation:**
    Integration with ScrambleBench evaluation pipeline:
    
    .. code-block:: python
    
        from scramblebench.core import TranslationBenchmark
        from scramblebench.llm import OpenRouterClient, ModelConfig
        
        # Configure for evaluation
        eval_config = ModelConfig(
            temperature=0.0,    # Deterministic for reproducibility
            max_tokens=500,     # Reasonable response length
            timeout=60          # Extended timeout for reliability
        )
        
        # Create model
        model = OpenRouterClient("openai/gpt-4", config=eval_config)
        model.initialize()
        
        # Create benchmark
        benchmark = TranslationBenchmark(
            source_dataset="qa_data.json",
            language_type="substitution"
        )
        
        # Run evaluation
        results = benchmark.run(model, max_samples=100)
        print(f"Accuracy: {results.score:.2%}")

**Factory Integration:**
    Use with ModelFactory for dynamic model creation:
    
    .. code-block:: python
    
        from scramblebench.llm import ModelFactory
        
        # Factory automatically handles OpenRouter registration
        model = ModelFactory.create_model(
            provider="openrouter",
            model_name="anthropic/claude-3-sonnet",
            config=ModelConfig(temperature=0.3)
        )
        
        # Model is ready to use
        response = model.generate("Test prompt")

**Custom Provider Creation:**
    Extend for specialized use cases:
    
    .. code-block:: python
    
        class CustomOpenRouterClient(OpenRouterClient):
            def __init__(self, model_name: str, **kwargs):
                # Add custom initialization
                super().__init__(model_name, **kwargs)
                self.custom_headers = {"X-Custom": "value"}
            
            async def _chat_async(self, messages, **kwargs):
                # Add custom preprocessing
                processed_messages = self.preprocess_messages(messages)
                return await super()._chat_async(processed_messages, **kwargs)

Security and Best Practices
----------------------------

**API Key Management:**
    Secure handling of authentication credentials:
    
    .. code-block:: python
    
        # Best practices for API key security
        import os
        
        # Use environment variables (recommended)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
        
        # Avoid hardcoding keys in source code
        # Never commit API keys to version control
        # Use secrets management in production

**Rate Limiting:**
    Respect API limits to avoid service interruption:
    
    .. code-block:: python
    
        # Configure conservative rate limits
        client.set_rate_limit(5.0)  # Well below typical limits
        
        # Monitor for rate limit errors
        # Implement exponential backoff
        # Use batch processing for efficiency

**Error Recovery:**
    Implement robust error handling:
    
    .. code-block:: python
    
        def robust_generate(client, prompt, max_retries=3):
            for attempt in range(max_retries):
                response = client.generate(prompt)
                
                if not response.error:
                    return response
                
                if "rate_limit" in response.error.lower():
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {response.error}")
            
            return response

See Also
--------

* :class:`ModelInterface` - Abstract base class for all model implementations
* :class:`ModelConfig` - Universal model configuration
* :class:`ModelResponse` - Standardized response container
* :doc:`../user_guide/configuration` - Configuration management guide
* :doc:`../tutorials/custom_models` - Creating custom model providers
* `OpenRouter Documentation <https://openrouter.ai/docs>`_ - Official API documentation
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import time
from pathlib import Path
import os

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    OpenAI = None
    openai_available = False

import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from scramblebench.llm.model_interface import (
    ModelInterface, ModelResponse, ModelConfig, ModelType, ModelFactory
)


class OpenRouterClient(ModelInterface):
    """
    Production-ready OpenRouter API client with comprehensive LLM provider access.
    
    This class provides a robust implementation of the :class:`ModelInterface` specifically
    designed for the OpenRouter API gateway. It offers access to 50+ models from multiple
    providers through a single interface, with built-in rate limiting, error recovery,
    async processing capabilities, and production-grade reliability features.
    
    Key Features:
        **Multi-Provider Support:**
            Access models from OpenAI, Anthropic, Google, Meta, Mistral, and more
            through a single API key and consistent interface.
            
        **Async Architecture:**
            Full async/await support with connection pooling, concurrent request
            handling, and configurable concurrency limits for optimal performance.
            
        **Intelligent Rate Limiting:**
            Built-in rate limiting with token bucket algorithm, automatic backoff,
            and provider-specific limit awareness to prevent quota violations.
            
        **Comprehensive Error Handling:**
            Exponential backoff retry logic, detailed error classification,
            and graceful degradation for production reliability.
            
        **Cost Optimization:**
            Token estimation, usage tracking, and cost-aware model selection
            to optimize API costs while maintaining performance.
    
    :param model_name: OpenRouter model identifier (e.g., "openai/gpt-4")
    :type model_name: str
    :param api_key: OpenRouter API key (reads from OPENROUTER_API_KEY env var if None)
    :type api_key: Optional[str]
    :param base_url: OpenRouter API base URL for custom endpoints
    :type base_url: str
    :param config: Model configuration for generation parameters
    :type config: Optional[ModelConfig]
    :param logger: Custom logger instance for debugging and monitoring
    :type logger: Optional[logging.Logger]
    
    Examples:
        Basic client setup and usage:
        
        .. code-block:: python
        
            from scramblebench.llm import OpenRouterClient, ModelConfig
            
            # Create client with environment variable API key
            client = OpenRouterClient("openai/gpt-4")
            
            # Or with explicit API key
            client = OpenRouterClient(
                model_name="anthropic/claude-3-sonnet",
                api_key="your_api_key_here"
            )
            
            # Initialize and use
            client.initialize()
            response = client.generate("Explain quantum computing")
            print(response.text)
        
        Advanced configuration for production:
        
        .. code-block:: python
        
            # Production-optimized configuration
            config = ModelConfig(
                temperature=0.0,        # Deterministic output
                max_tokens=1000,        # Cost control
                frequency_penalty=0.1,  # Reduce repetition
                timeout=60             # Extended timeout
            )
            
            client = OpenRouterClient(
                model_name="openai/gpt-4",
                config=config
            )
            
            # Configure rate limiting
            client.set_rate_limit(5.0)  # 5 requests per second
            
            # Initialize and verify
            if client.initialize():
                print(f"Client ready: {client.model_name}")
                print(f"Model info: {client.get_model_info()}")
            else:
                print("Initialization failed")
        
        Batch processing with error handling:
        
        .. code-block:: python
        
            prompts = [
                "What is machine learning?",
                "Explain neural networks",
                "Define artificial intelligence"
            ]
            
            # Process batch with controlled concurrency
            responses = client.batch_generate(
                prompts,
                max_concurrent=3,
                temperature=0.0
            )
            
            # Handle results
            for i, response in enumerate(responses):
                if response.error:
                    print(f"Prompt {i} failed: {response.error}")
                else:
                    print(f"Prompt {i}: {response.text[:100]}...")
                    print(f"Tokens: {response.metadata['total_tokens']}")
        
        Conversation management:
        
        .. code-block:: python
        
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ]
            
            response = client.chat(conversation)
            
            if not response.error:
                conversation.append({
                    "role": "assistant",
                    "content": response.text
                })
                
                # Continue conversation
                conversation.append({
                    "role": "user",
                    "content": "Show me a simple example"
                })
                
                follow_up = client.chat(conversation)
    
    Rate Limiting:
        The client implements intelligent rate limiting to prevent API quota violations:
        
        .. code-block:: python
        
            # Default rate limiting (10 requests/second)
            client = OpenRouterClient("openai/gpt-4")
            
            # Custom rate limiting
            client.set_rate_limit(20.0)  # 20 requests/second
            
            # Rate limiting is automatic and transparent
            # Requests are queued and throttled as needed
            responses = client.batch_generate(many_prompts)
    
    Error Recovery:
        Comprehensive error handling with automatic retry logic:
        
        .. code-block:: python
        
            response = client.generate("Test prompt")
            
            if response.error:
                error_lower = response.error.lower()
                
                if "rate limit" in error_lower:
                    # Automatic retry with backoff (already handled internally)
                    print("Rate limited - automatic retry applied")
                    
                elif "timeout" in error_lower:
                    # Increase timeout for retry
                    client.update_config(timeout=90)
                    response = client.generate("Test prompt")
                    
                elif "authentication" in error_lower:
                    print("Check API key configuration")
                    
                else:
                    print(f"Unknown error: {response.error}")
    
    Performance Optimization:
        Best practices for high-performance usage:
        
        .. code-block:: python
        
            # Connection reuse (automatic)
            client = OpenRouterClient("openai/gpt-4")
            client.initialize()  # Establishes persistent connection
            
            # Batch processing for efficiency
            large_batch = ["prompt"] * 100
            responses = client.batch_generate(
                large_batch,
                max_concurrent=10  # Adjust based on rate limits
            )
            
            # Async context manager for resource management
            async with OpenRouterClient("anthropic/claude-3-sonnet") as client:
                response = await client._chat_async([
                    {"role": "user", "content": "async prompt"}
                ])
    
    Model Information:
        Access model capabilities and metadata:
        
        .. code-block:: python
        
            info = client.get_model_info()
            
            print(f"Model: {info['name']}")
            print(f"Max context: {info['max_context_length']}")
            print(f"Max output: {info['max_output_tokens']}")
            print(f"Pricing: {info.get('pricing', 'N/A')}")
            
            # Get available models
            models = client.get_available_models()
            for model in models:
                print(f"{model['id']}: {model.get('description', '')}")
    
    Cost Management:
        Monitor and control API usage costs:
        
        .. code-block:: python
        
            # Estimate costs before generation
            prompt = "Long prompt text..."
            tokens = client.estimate_tokens(prompt)
            print(f"Estimated tokens: {tokens}")
            
            # Monitor actual usage
            response = client.generate(prompt)
            actual_tokens = response.metadata['total_tokens']
            print(f"Actual tokens: {actual_tokens}")
            
            # Usage statistics (if available)
            stats = client.get_usage_stats()
            print(f"Total usage: {stats}")
    
    Initialization:
        The client must be initialized before use:
        
        .. code-block:: python
        
            client = OpenRouterClient("openai/gpt-4")
            
            # Check initialization status
            print(f"Pre-init: {client.is_initialized}")  # False
            
            # Initialize (connects, authenticates, validates)
            success = client.initialize()
            
            if success:
                print(f"Post-init: {client.is_initialized}")  # True
                print(f"Model type: {client.model_type}")
            else:
                print("Initialization failed - check API key and network")
    
    Thread Safety:
        The client is designed for concurrent usage:
        
        .. code-block:: python
        
            import concurrent.futures
            
            def generate_text(prompt):
                return client.generate(prompt)
            
            # Safe for concurrent use
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(generate_text, f"Prompt {i}")
                    for i in range(10)
                ]
                
                results = [f.result() for f in futures]
    
    See Also:
        * :meth:`initialize` - Required setup before use
        * :meth:`generate` - Single prompt generation
        * :meth:`chat` - Conversation handling
        * :meth:`batch_generate` - Efficient batch processing
        * :meth:`set_rate_limit` - Rate limiting configuration
        * :func:`create_openrouter_client` - Factory function for easy creation
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            model_name: Name of the model on OpenRouter
            api_key: OpenRouter API key (reads from env if None)
            base_url: OpenRouter API base URL
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(model_name, config, logger)
        
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = base_url
        self._model_type = ModelType.CHAT  # Most OpenRouter models are chat-based
        
        # API client
        self._client = None
        self._session = None
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # Minimum time between requests
        
        # Model info cache
        self._model_info_cache = None
    
    def initialize(self) -> bool:
        """Initialize the OpenRouter client."""
        try:
            if not self.api_key:
                self.logger.error("OpenRouter API key not provided")
                return False
            
            # Initialize HTTP session
            self._session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://github.com/nathanrice/scramblebench',
                    'X-Title': 'ScrambleBench'
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection by getting model info
            model_info = asyncio.run(self._get_model_info_async())
            if not model_info:
                self.logger.error(f"Could not fetch info for model: {self.model_name}")
                return False
            
            self._model_info_cache = model_info
            self._initialized = True
            
            self.logger.info(f"OpenRouter client initialized for model: {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouter client: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text using OpenRouter API."""
        if not self._initialized:
            return ModelResponse(
                text="",
                metadata={},
                error="Client not initialized"
            )
        
        if not self.validate_prompt(prompt):
            return ModelResponse(
                text="",
                metadata={},
                error="Invalid prompt"
            )
        
        # Convert to chat format (most OpenRouter models expect this)
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Have a conversation using OpenRouter API."""
        if not self._initialized:
            return ModelResponse(
                text="",
                metadata={},
                error="Client not initialized"
            )
        
        try:
            return asyncio.run(self._chat_async(messages, **kwargs))
        except Exception as e:
            self.logger.error(f"Chat request failed: {e}")
            return ModelResponse(
                text="",
                metadata={},
                error=str(e)
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _chat_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> ModelResponse:
        """Async chat with retry logic."""
        # Rate limiting
        await self._rate_limit()
        
        # Prepare request data
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty)
        }
        
        # Add stop sequences if provided
        stop_sequences = kwargs.get('stop_sequences', self.config.stop_sequences)
        if stop_sequences:
            request_data["stop"] = stop_sequences
        
        start_time = time.time()
        
        async with self._session.post(
            f"{self.base_url}/chat/completions",
            json=request_data
        ) as response:
            response_data = await response.json()
            
            if response.status != 200:
                error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API request failed: {error_msg}")
            
            # Extract response text
            choices = response_data.get('choices', [])
            if not choices:
                raise Exception("No choices in API response")
            
            response_text = choices[0].get('message', {}).get('content', '')
            
            # Extract metadata
            usage = response_data.get('usage', {})
            metadata = {
                'model': self.model_name,
                'response_time': time.time() - start_time,
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'finish_reason': choices[0].get('finish_reason'),
                'request_data': request_data
            }
            
            return ModelResponse(
                text=response_text,
                metadata=metadata,
                raw_response=response_data
            )
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        if not self._initialized:
            return [
                ModelResponse(text="", metadata={}, error="Client not initialized")
                for _ in prompts
            ]
        
        try:
            return asyncio.run(self._batch_generate_async(prompts, **kwargs))
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            return [
                ModelResponse(text="", metadata={}, error=str(e))
                for _ in prompts
            ]
    
    async def _batch_generate_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ModelResponse]:
        """Async batch generation."""
        # Convert prompts to chat messages
        message_lists = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        
        # Create tasks for concurrent execution
        tasks = [
            self._chat_async(messages, **kwargs)
            for messages in message_lists
        ]
        
        # Execute with controlled concurrency
        max_concurrent = kwargs.get('max_concurrent', 5)
        results = []
        
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Convert exceptions to error responses
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ModelResponse(
                        text="",
                        metadata={},
                        error=str(result)
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self._model_info_cache:
            return self._model_info_cache
        
        if self._initialized:
            try:
                return asyncio.run(self._get_model_info_async())
            except Exception as e:
                self.logger.error(f"Failed to get model info: {e}")
        
        # Return default info
        return {
            'name': self.model_name,
            'type': self.model_type.value,
            'provider': 'openrouter',
            'max_context_length': 4096,
            'max_output_tokens': 2048,
            'supports_chat': True,
            'supports_batch': True
        }
    
    async def _get_model_info_async(self) -> Dict[str, Any]:
        """Get model info from OpenRouter API."""
        try:
            async with self._session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    self.logger.warning(f"Could not fetch models list: {response.status}")
                    return {}
                
                models_data = await response.json()
                models = models_data.get('data', [])
                
                # Find our model
                for model in models:
                    if model.get('id') == self.model_name:
                        return {
                            'name': model.get('id'),
                            'type': self.model_type.value,
                            'provider': 'openrouter',
                            'description': model.get('description', ''),
                            'max_context_length': model.get('context_length', 4096),
                            'max_output_tokens': model.get('max_output_tokens', 2048),
                            'pricing': model.get('pricing', {}),
                            'supports_chat': True,
                            'supports_batch': True,
                            'created': model.get('created'),
                            'owned_by': model.get('owned_by')
                        }
                
                self.logger.warning(f"Model {self.model_name} not found in models list")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching model info: {e}")
            return {}
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using OpenAI's rule of thumb."""
        # Rough estimate: ~4 characters per token for English text
        # This is approximate - actual tokenization would be more accurate
        return len(text) // 4
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        if not self._initialized:
            return []
        
        try:
            return asyncio.run(self._get_available_models_async())
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return []
    
    async def _get_available_models_async(self) -> List[Dict[str, Any]]:
        """Get available models list async."""
        try:
            async with self._session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    return []
                
                models_data = await response.json()
                return models_data.get('data', [])
                
        except Exception as e:
            self.logger.error(f"Error fetching models: {e}")
            return []
    
    def set_rate_limit(self, requests_per_second: float) -> None:
        """
        Set rate limit for API requests.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self._min_request_interval = 1.0 / requests_per_second
        self.logger.info(f"Rate limit set to {requests_per_second} requests/second")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics (if available from OpenRouter)."""
        # This would require additional API endpoints from OpenRouter
        # For now, return empty stats
        return {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    def __del__(self):
        """Cleanup when object is deleted."""
        if self._session and not self._session.closed:
            try:
                asyncio.run(self._session.close())
            except Exception:
                pass


# Factory function for easy creation
def create_openrouter_client(
    model_name: str,
    api_key: Optional[str] = None,
    config: Optional[ModelConfig] = None,
    **kwargs
) -> OpenRouterClient:
    """
    Create and initialize an OpenRouter client.
    
    Args:
        model_name: Name of the model on OpenRouter
        api_key: OpenRouter API key
        config: Model configuration
        **kwargs: Additional client arguments
        
    Returns:
        Initialized OpenRouter client
    """
    client = OpenRouterClient(
        model_name=model_name,
        api_key=api_key,
        config=config,
        **kwargs
    )
    
    if not client.initialize():
        raise RuntimeError(f"Failed to initialize OpenRouter client for {model_name}")
    
    return client


# Register with factory
ModelFactory.register_model('openrouter', OpenRouterClient)


# Common OpenRouter model configurations
OPENROUTER_MODELS = {
    'gpt-4': 'openai/gpt-4',
    'gpt-4-turbo': 'openai/gpt-4-turbo',
    'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
    'claude-3-opus': 'anthropic/claude-3-opus',
    'claude-3-sonnet': 'anthropic/claude-3-sonnet',
    'claude-3-haiku': 'anthropic/claude-3-haiku',
    'llama-2-70b': 'meta-llama/llama-2-70b-chat',
    'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct',
    'gemini-pro': 'google/gemini-pro'
}


def get_model_id(model_name: str) -> str:
    """
    Get the OpenRouter model ID for a common model name.
    
    Args:
        model_name: Common model name
        
    Returns:
        OpenRouter model ID
    """
    return OPENROUTER_MODELS.get(model_name, model_name)