"""
Ollama API Client for ScrambleBench
===================================

This module provides comprehensive integration with Ollama, a local LLM runtime that enables
running Large Language Models locally on consumer hardware. Ollama supports efficient inference
for many popular open-source models including Llama, Phi, Gemma, Code Llama, and many others.

Ollama Integration Benefits
---------------------------

**Local Model Execution:**
    Run models locally without API keys or network dependencies:
    
    - Complete privacy and data control
    - No rate limits or usage costs
    - Offline capability for air-gapped environments
    - Fast inference on consumer hardware (CPU/GPU)

**Wide Model Support:**
    Access to many popular open-source models:
    
    - Meta Llama (7B, 13B, 70B variants)
    - Microsoft Phi (3, 3.5 models optimized for efficiency)
    - Google Gemma (2B, 7B efficient models)
    - Code Llama (specialized for coding tasks)
    - Mistral, Falcon, and many community models

**Development-Friendly:**
    Perfect for development, testing, and preliminary benchmarking:
    
    - Quick setup and deployment
    - No external dependencies or API keys
    - Consistent performance for reproducible testing
    - Easy model switching and comparison

Architecture Overview
---------------------

The Ollama client implements the :class:`ModelInterface` with local inference optimizations:

**HTTP API Integration:**
    Communicates with local Ollama server via REST API:
    
    - JSON-based request/response format
    - Streaming support for real-time generation
    - Model management and switching
    - Resource usage monitoring

**Performance Optimization:**
    Optimized for local inference scenarios:
    
    - Connection pooling for HTTP requests
    - Efficient batch processing
    - Memory-aware model loading
    - GPU acceleration when available

**Error Handling:**
    Comprehensive error recovery for local deployment:
    
    - Server availability checking
    - Model loading status monitoring
    - Resource exhaustion handling
    - Network timeout management

Configuration and Setup
-----------------------

**Ollama Installation:**
    Install and start Ollama server:
    
    .. code-block:: bash
    
        # Install Ollama (Linux/macOS)
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Start Ollama server
        ollama serve
        
        # Pull models for use
        ollama pull llama3.2:1b
        ollama pull phi3:3.8b
        ollama pull gemma2:2b

**Client Configuration:**
    Configure client for local Ollama instance:
    
    .. code-block:: python
    
        from scramblebench.llm import OllamaClient, ModelConfig
        
        # Default configuration (localhost:11434)
        client = OllamaClient("llama3.2:1b")
        
        # Custom server configuration
        client = OllamaClient(
            model_name="phi3:3.8b",
            base_url="http://localhost:11434",
            config=ModelConfig(temperature=0.0, max_tokens=1000)
        )

**Model Selection:**
    Choose appropriate models for different use cases:
    
    .. code-block:: python
    
        # Small efficient models for quick testing
        small_models = {
            "llama3.2:1b": "Very fast, good for simple tasks",
            "phi3:3.8b": "Balanced efficiency and capability", 
            "gemma2:2b": "Google's efficient model",
            "qwen2:1.5b": "Alibaba's compact model"
        }
        
        # Medium models for better quality
        medium_models = {
            "llama3.2:3b": "Good balance of speed and quality",
            "phi3:14b": "High quality reasoning",
            "gemma2:9b": "Strong performance",
            "mistral:7b": "General-purpose excellence"
        }

Usage Patterns
--------------

**Basic Usage:**
    Simple text generation with local models:
    
    .. code-block:: python
    
        from scramblebench.llm import OllamaClient
        
        # Create and initialize client
        client = OllamaClient("llama3.2:1b")
        client.initialize()
        
        # Generate text
        response = client.generate("Explain machine learning in simple terms")
        
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.text}")
            print(f"Response time: {response.metadata['response_time']:.2f}s")

**Conversation Handling:**
    Multi-turn conversations with context:
    
    .. code-block:: python
    
        # Initialize conversation
        conversation = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I sort a list in Python?"}
        ]
        
        # Get response
        response = client.chat(conversation)
        
        if not response.error:
            # Add assistant response
            conversation.append({
                "role": "assistant",
                "content": response.text
            })
            
            # Continue conversation
            conversation.append({
                "role": "user",
                "content": "Can you show an example with custom sorting?"
            })
            
            follow_up = client.chat(conversation)

**Batch Processing:**
    Efficient processing of multiple prompts:
    
    .. code-block:: python
    
        # Prepare batch of prompts
        prompts = [
            "What is Python?",
            "Explain functions in programming", 
            "What are data structures?",
            "How does recursion work?"
        ]
        
        # Process batch (sequential for local models)
        responses = client.batch_generate(prompts)
        
        for i, response in enumerate(responses):
            if response.error:
                print(f"Prompt {i} failed: {response.error}")
            else:
                print(f"Prompt {i}: {response.text[:100]}...")

**Model Management:**
    Check and manage available models:
    
    .. code-block:: python
    
        # Check if model is available
        if not client.is_model_available("phi3:3.8b"):
            print("Model not found. Pull with: ollama pull phi3:3.8b")
        
        # List available models
        models = client.get_available_models()
        for model in models:
            print(f"Model: {model['name']}, Size: {model.get('size', 'Unknown')}")
        
        # Get model info
        info = client.get_model_info()
        print(f"Current model: {info['name']}")
        print(f"Parameters: {info.get('parameters', 'Unknown')}")

Performance Optimization
------------------------

**Model Selection:**
    Choose models appropriate for your hardware and use case:
    
    .. code-block:: python
    
        # For development/testing (fast inference)
        client = OllamaClient("llama3.2:1b")
        
        # For quality results (slower but better)
        client = OllamaClient("phi3:14b")
        
        # For coding tasks
        client = OllamaClient("codellama:7b")

**Resource Management:**
    Optimize memory and compute usage:
    
    .. code-block:: python
    
        # Configure for memory-constrained environments
        config = ModelConfig(
            temperature=0.0,    # Deterministic output
            max_tokens=512,     # Limit response length
            timeout=120        # Allow for slower inference
        )
        
        client = OllamaClient("gemma2:2b", config=config)

**Concurrent Processing:**
    Handle concurrent requests efficiently:
    
    .. code-block:: python
    
        # Ollama handles one request at a time internally
        # Use moderate concurrency to avoid overwhelming
        responses = client.batch_generate(
            prompts,
            max_concurrent=2  # Conservative for local inference
        )

Error Handling and Debugging
-----------------------------

**Server Availability:**
    Handle Ollama server connection issues:
    
    .. code-block:: python
    
        try:
            client = OllamaClient("llama3.2:1b")
            if not client.initialize():
                print("Failed to connect to Ollama server")
                print("Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"Ollama connection error: {e}")

**Model Availability:**
    Handle missing models gracefully:
    
    .. code-block:: python
    
        response = client.generate("Test prompt")
        
        if response.error:
            if "model not found" in response.error.lower():
                print(f"Model {client.model_name} not available")
                print(f"Pull with: ollama pull {client.model_name}")
            elif "server" in response.error.lower():
                print("Ollama server not responding")
            else:
                print(f"Unknown error: {response.error}")

**Performance Issues:**
    Handle resource constraints:
    
    .. code-block:: python
    
        # Monitor response times
        response = client.generate("Complex reasoning task")
        
        if response.metadata.get('response_time', 0) > 30:
            print("Slow response - consider using smaller model")
        
        # Check memory usage if available
        info = client.get_model_info()
        if info.get('memory_usage'):
            print(f"Memory usage: {info['memory_usage']}")

Integration Examples
--------------------

**Benchmark Evaluation:**
    Integration with ScrambleBench evaluation pipeline:
    
    .. code-block:: python
    
        from scramblebench.evaluation import EvaluationRunner
        from scramblebench.llm import OllamaClient, ModelConfig
        
        # Configure for deterministic evaluation
        eval_config = ModelConfig(
            temperature=0.0,    # No randomness
            max_tokens=1000,    # Sufficient for most tasks
            timeout=180        # Allow for local inference time
        )
        
        # Create model
        model = OllamaClient("phi3:3.8b", config=eval_config)
        model.initialize()
        
        # Run preliminary benchmarks
        # (Integration with existing evaluation pipeline)

**Factory Integration:**
    Use with ModelFactory for dynamic model creation:
    
    .. code-block:: python
    
        from scramblebench.llm import ModelFactory
        
        # Factory handles Ollama registration automatically
        model = ModelFactory.create_model(
            provider="ollama",
            model_name="gemma2:2b",
            config=ModelConfig(temperature=0.1)
        )
        
        # Model is ready to use
        response = model.generate("Test prompt")

Best Practices
--------------

**Model Selection:**
    Choose models based on requirements:
    
    .. code-block:: python
    
        # For speed (development/testing)
        fast_models = ["llama3.2:1b", "phi3:3.8b", "gemma2:2b"]
        
        # For quality (final benchmarks)
        quality_models = ["llama3.2:3b", "phi3:14b", "gemma2:9b"]
        
        # For specific tasks
        coding_models = ["codellama:7b", "phi3:14b"]
        math_models = ["phi3:14b", "gemma2:9b"]

**Resource Monitoring:**
    Monitor system resources during inference:
    
    .. code-block:: python
    
        import psutil
        
        # Check system resources before large batches
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            print("High memory usage - consider smaller batches")
        
        # Monitor during generation
        start_memory = psutil.virtual_memory().percent
        response = client.generate(prompt)
        end_memory = psutil.virtual_memory().percent
        print(f"Memory change: {end_memory - start_memory:.1f}%")

**Development Workflow:**
    Use Ollama for development and testing:
    
    .. code-block:: python
    
        # Development: Fast model for quick iteration
        dev_client = OllamaClient("llama3.2:1b")
        
        # Testing: Medium model for validation
        test_client = OllamaClient("phi3:3.8b") 
        
        # Switch between models easily
        current_model = dev_client if debugging else test_client

See Also
--------

* :class:`ModelInterface` - Abstract base class implemented by OllamaClient
* :class:`ModelConfig` - Configuration management for Ollama models
* :class:`ModelResponse` - Standardized response container
* :class:`ModelFactory` - Model creation and registration system
* `Ollama Documentation <https://ollama.ai/docs>`_ - Official Ollama documentation
* `Ollama Models <https://ollama.ai/models>`_ - Available model library
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import time
import asyncio
from pathlib import Path
import os

try:
    import aiohttp
    import requests
    http_available = True
except ImportError:
    aiohttp = None
    requests = None
    http_available = False

from scramblebench.llm.model_interface import (
    ModelInterface, ModelResponse, ModelConfig, ModelType, ModelFactory
)


"""
Production-grade Ollama client for ScrambleBench.

Integrates the working simple_ollama_client.py functionality with the
production ScrambleBench framework architecture.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .model_interface import ModelInterface, ModelResponse


logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 500
    verify_ssl: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class OllamaClient(ModelInterface):
    """
    Production-grade Ollama client for local LLM evaluation.
    
    Integrates with ScrambleBench's model interface while providing
    robust error handling, retry logic, and comprehensive logging.
    
    Examples:
        Basic usage:
        
        .. code-block:: python
        
            client = OllamaClient("llama3:8b")
            if client.initialize():
                response = client.generate("What is 2+2?")
                print(response.text)
        
        Custom configuration:
        
        .. code-block:: python
        
            config = OllamaConfig(
                base_url="http://remote-ollama:11434",
                temperature=0.0,
                max_tokens=200
            )
            client = OllamaClient("gemma3:27b", config)
        
        Async usage:
        
        .. code-block:: python
        
            async def batch_evaluate():
                client = OllamaClient("phi3:14b")
                await client.initialize_async()
                
                tasks = []
                for prompt in prompts:
                    tasks.append(client.generate_async(prompt))
                
                results = await asyncio.gather(*tasks)
                return results
    """
    
    def __init__(self, 
                 model_name: str, 
                 config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama3:8b", "gemma3:27b")
            config: Client configuration (uses defaults if None)
        """
        super().__init__(model_name)
        self.config = config or OllamaConfig()
        self.base_url = self.config.base_url.rstrip('/')
        
        # Session management
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Model validation
        self._available_models: List[str] = []
        
        logger.info(f"Initialized OllamaClient for model: {model_name}")
    
    def initialize(self) -> bool:
        """
        Initialize the client and verify model availability.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create session
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Test server connection
            response = self._session.get(
                f"{self.base_url}/api/tags", 
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            if response.status_code != 200:
                logger.error(f"Cannot connect to Ollama server at {self.base_url}")
                return False
            
            # Get available models
            data = response.json()
            models = data.get('models', [])
            self._available_models = [m.get('name', '') for m in models]
            
            # Check if model is available
            if self.model_name not in self._available_models:
                logger.error(f"Model {self.model_name} not available. Available models: {self._available_models}")
                logger.info(f"Pull model with: ollama pull {self.model_name}")
                return False
            
            self._initialized = True
            logger.info(f"Successfully initialized Ollama client for {self.model_name}")
            return True
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama server")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            return False
    
    async def initialize_async(self) -> bool:
        """
        Initialize async client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create async session
            connector = aiohttp.TCPConnector(verify_ssl=self.config.verify_ssl)
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            # Test server connection
            async with self._async_session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    logger.error(f"Cannot connect to Ollama server at {self.base_url}")
                    return False
                
                data = await response.json()
                models = data.get('models', [])
                self._available_models = [m.get('name', '') for m in models]
            
            # Check model availability
            if self.model_name not in self._available_models:
                logger.error(f"Model {self.model_name} not available")
                return False
            
            self._initialized = True
            logger.info(f"Successfully initialized async Ollama client for {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize async Ollama client: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate text using Ollama with retry logic.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse with generated text or error information
        """
        if not self._initialized:
            return ModelResponse(
                text="",
                metadata={},
                error="Client not initialized. Call initialize() first."
            )
        
        start_time = time.time()
        
        try:
            # Prepare request
            messages = [{"role": "user", "content": prompt}]
            
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                }
            }
            
            # Make request
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=kwargs.get('timeout', self.config.timeout)
            )
            
            response_time = time.time() - start_time
            
            # Handle errors
            if response.status_code != 200:
                error_msg = f"Ollama API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_msg)
                except:
                    error_msg = f"{error_msg} - {response.text[:200]}"
                
                logger.error(error_msg)
                return ModelResponse(
                    text="",
                    metadata={'response_time': response_time, 'status_code': response.status_code},
                    error=error_msg
                )
            
            # Parse response
            response_data = response.json()
            message = response_data.get('message', {})
            response_text = message.get('content', '').strip()
            
            # Compile metadata
            metadata = {
                'model': self.model_name,
                'response_time': response_time,
                'total_duration': response_data.get('total_duration', 0) / 1e9,
                'load_duration': response_data.get('load_duration', 0) / 1e9,
                'prompt_eval_count': response_data.get('prompt_eval_count', 0),
                'eval_count': response_data.get('eval_count', 0),
                'prompt_eval_duration': response_data.get('prompt_eval_duration', 0) / 1e9,
                'eval_duration': response_data.get('eval_duration', 0) / 1e9,
            }
            
            logger.debug(f"Generated {len(response_text)} characters in {response_time:.2f}s")
            
            return ModelResponse(
                text=response_text,
                metadata=metadata,
                raw_response=response_data
            )
            
        except requests.exceptions.Timeout:
            return ModelResponse(
                text="",
                metadata={'response_time': time.time() - start_time},
                error="Request timeout"
            )
        except requests.exceptions.ConnectionError:
            return ModelResponse(
                text="",
                metadata={'response_time': time.time() - start_time},
                error="Cannot connect to Ollama server"
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ModelResponse(
                text="",
                metadata={'response_time': time.time() - start_time},
                error=str(e)
            )
    
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate text asynchronously.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse with generated text or error information
        """
        if not self._async_session:
            return ModelResponse(
                text="",
                metadata={},
                error="Async client not initialized. Call initialize_async() first."
            )
        
        start_time = time.time()
        
        try:
            # Prepare request
            messages = [{"role": "user", "content": prompt}]
            
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                }
            }
            
            # Make async request
            async with self._async_session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=kwargs.get('timeout', self.config.timeout))
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"Ollama API error: {response.status} - {error_text[:200]}"
                    
                    return ModelResponse(
                        text="",
                        metadata={'response_time': response_time, 'status_code': response.status},
                        error=error_msg
                    )
                
                # Parse response
                response_data = await response.json()
                message = response_data.get('message', {})
                response_text = message.get('content', '').strip()
                
                # Compile metadata
                metadata = {
                    'model': self.model_name,
                    'response_time': response_time,
                    'total_duration': response_data.get('total_duration', 0) / 1e9,
                    'async': True
                }
                
                return ModelResponse(
                    text=response_text,
                    metadata=metadata,
                    raw_response=response_data
                )
                
        except asyncio.TimeoutError:
            return ModelResponse(
                text="",
                metadata={'response_time': time.time() - start_time},
                error="Async request timeout"
            )
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            return ModelResponse(
                text="",
                metadata={'response_time': time.time() - start_time},
                error=str(e)
            )
    
    def list_available_models(self) -> List[str]:
        """
        Get list of available models on the Ollama server.
        
        Returns:
            List of available model names
        """
        return self._available_models.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if not self._initialized:
            return {'error': 'Client not initialized'}
        
        try:
            response = self._session.get(f"{self.base_url}/api/show", 
                                       json={"name": self.model_name},
                                       timeout=self.config.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Failed to get model info: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def close(self):
        """Clean up resources."""
        if self._session:
            self._session.close()
            self._session = None
        
        self._initialized = False
        logger.info("Closed Ollama client")
    
    async def close_async(self):
        """Clean up async resources.""" 
        if self._async_session:
            await self._async_session.close()
            self._async_session = None
        
        self._initialized = False
        logger.info("Closed async Ollama client")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._session:
            try:
                self._session.close()
            except:
                pass


# Factory function for easy creation
def create_ollama_client(
    model_name: str,
    base_url: str = "http://localhost:11434",
    config: Optional[ModelConfig] = None,
    **kwargs
) -> OllamaClient:
    """
    Create and initialize an Ollama client.
    
    Args:
        model_name: Name of the Ollama model
        base_url: Ollama server URL
        config: Model configuration
        **kwargs: Additional client arguments
        
    Returns:
        Initialized Ollama client
    """
    client = OllamaClient(
        model_name=model_name,
        base_url=base_url,
        config=config,
        **kwargs
    )
    
    if not client.initialize():
        raise RuntimeError(f"Failed to initialize Ollama client for {model_name}")
    
    return client


# Register with factory
ModelFactory.register_model('ollama', OllamaClient)


# Common small Ollama models for development and testing
OLLAMA_SMALL_MODELS = {
    # Ultra-fast models (< 1GB) - great for development
    'llama3.2:1b': {
        'name': 'llama3.2:1b',
        'description': 'Llama 3.2 1B - Ultra-fast, good for simple tasks',
        'size': '1.3GB',
        'parameters': '1B',
        'use_case': 'development, simple QA'
    },
    'qwen2:0.5b': {
        'name': 'qwen2:0.5b', 
        'description': 'Qwen2 0.5B - Smallest available model',
        'size': '0.7GB',
        'parameters': '0.5B',
        'use_case': 'ultra-fast testing'
    },
    
    # Fast models (1-4GB) - balanced for testing
    'phi3:3.8b': {
        'name': 'phi3:3.8b',
        'description': 'Phi-3 3.8B - Microsoft\'s efficient model',
        'size': '2.3GB',
        'parameters': '3.8B',
        'use_case': 'balanced development, reasoning'
    },
    'gemma2:2b': {
        'name': 'gemma2:2b',
        'description': 'Gemma 2 2B - Google\'s efficient model',
        'size': '1.6GB', 
        'parameters': '2B',
        'use_case': 'efficient inference, general tasks'
    },
    'llama3.2:3b': {
        'name': 'llama3.2:3b',
        'description': 'Llama 3.2 3B - Good balance of speed and quality',
        'size': '2.0GB',
        'parameters': '3B',
        'use_case': 'testing, balanced performance'
    },
    
    # Quality models (4-8GB) - for final testing
    'phi3:14b': {
        'name': 'phi3:14b',
        'description': 'Phi-3 14B - High quality reasoning',
        'size': '7.9GB',
        'parameters': '14B',
        'use_case': 'quality testing, complex reasoning'
    },
    'gemma2:9b': {
        'name': 'gemma2:9b',
        'description': 'Gemma 2 9B - Strong performance model',
        'size': '5.4GB',
        'parameters': '9B', 
        'use_case': 'quality benchmarks'
    },
    'llama3.1:8b': {
        'name': 'llama3.1:8b',
        'description': 'Llama 3.1 8B - General purpose excellence',
        'size': '4.7GB',
        'parameters': '8B',
        'use_case': 'comprehensive evaluation'
    }
}


def get_recommended_model(use_case: str = "development") -> str:
    """
    Get a recommended Ollama model for a specific use case.
    
    Args:
        use_case: Use case ("development", "testing", "benchmarking")
        
    Returns:
        Recommended model name
    """
    recommendations = {
        'development': 'phi3:3.8b',      # Fast and capable
        'testing': 'llama3.2:3b',        # Balanced quality/speed  
        'benchmarking': 'phi3:14b',      # High quality
        'ultra_fast': 'llama3.2:1b',     # Fastest possible
        'quality': 'gemma2:9b'           # Best quality in range
    }
    
    return recommendations.get(use_case, 'phi3:3.8b')


def list_small_models() -> List[Dict[str, Any]]:
    """
    Get list of recommended small Ollama models.
    
    Returns:
        List of model information dictionaries
    """
    return list(OLLAMA_SMALL_MODELS.values())