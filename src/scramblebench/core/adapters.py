"""
Model adapters for uniform interface to LLM providers.

Provides consistent interface for both Ollama and hosted providers (OpenAI, Anthropic, etc.)
with tokenization, completion, model info, and cost estimation capabilities.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import aiohttp
import tiktoken
from openai import AsyncOpenAI


@dataclass
class CompletionResult:
    """Result of a completion request."""
    text: str
    success: bool
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class ModelInfo:
    """Information about a model."""
    family: str  # e.g., "gpt-4", "claude-3", "llama2"
    n_params: Optional[int] = None  # Number of parameters
    context_length: Optional[int] = None
    supports_streaming: bool = True
    provider: str = ""


@dataclass 
class TokenizationResult:
    """Result of tokenization."""
    tokens: List[str]
    token_count: int
    token_ids: Optional[List[int]] = None


class CostEstimator:
    """Estimates API costs for different providers."""
    
    # Pricing per 1K tokens (input, output) as of 2024
    PRICING = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0015, 0.002),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
        # Ollama models are free
        "ollama": (0.0, 0.0),
    }
    
    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a completion."""
        # Extract base model name for pricing lookup
        base_model = self._extract_base_model(model_id)
        
        if base_model not in self.PRICING:
            return 0.0  # Unknown model, assume free (like Ollama)
        
        input_rate, output_rate = self.PRICING[base_model]
        
        input_cost = (input_tokens / 1000) * input_rate
        output_cost = (output_tokens / 1000) * output_rate
        
        return input_cost + output_cost
    
    def _extract_base_model(self, model_id: str) -> str:
        """Extract base model name for pricing lookup."""
        if "gpt-4" in model_id.lower():
            if "turbo" in model_id.lower():
                return "gpt-4-turbo"
            return "gpt-4"
        elif "gpt-3.5" in model_id.lower():
            return "gpt-3.5-turbo"
        elif "claude-3-opus" in model_id.lower():
            return "claude-3-opus"
        elif "claude-3-sonnet" in model_id.lower():
            return "claude-3-sonnet"
        elif "claude-3-haiku" in model_id.lower():
            return "claude-3-haiku"
        else:
            return "ollama"  # Default to free


class BaseModelAdapter(ABC):
    """Base class for all model adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_estimator = CostEstimator()
    
    @abstractmethod
    async def tokenize(self, text: str) -> TokenizationResult:
        """Tokenize text and return tokens with count."""
        pass
    
    @abstractmethod
    async def complete(self, prompt: str, **params) -> CompletionResult:
        """Generate completion for the given prompt."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        pass
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a completion."""
        model_id = self.config.get('model_id', 'unknown')
        return self.cost_estimator.estimate_cost(model_id, input_tokens, output_tokens)


class OllamaAdapter(BaseModelAdapter):
    """Adapter for Ollama models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model_id = config.get('model_id')
        self.timeout = config.get('timeout', 60)
        
        if not self.model_id:
            raise ValueError("model_id is required for OllamaAdapter")
    
    async def tokenize(self, text: str) -> TokenizationResult:
        """Tokenize using Ollama's tokenize endpoint."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/tokenize",
                    json={
                        "model": self.model_id,
                        "prompt": text
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = data.get('tokens', [])
                        return TokenizationResult(
                            tokens=[str(t) for t in tokens],
                            token_count=len(tokens),
                            token_ids=tokens if isinstance(tokens[0], int) else None
                        )
                    else:
                        # Fallback: estimate tokens
                        estimated_count = len(text.split()) * 1.3  # Rough estimate
                        return TokenizationResult(
                            tokens=text.split(),
                            token_count=int(estimated_count)
                        )
            except Exception:
                # Fallback: simple word-based tokenization
                tokens = text.split()
                return TokenizationResult(
                    tokens=tokens,
                    token_count=int(len(tokens) * 1.3)
                )
    
    async def complete(self, prompt: str, **params) -> CompletionResult:
        """Generate completion using Ollama."""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Extract relevant parameters
            temperature = params.get('temperature', 0.0)
            max_tokens = params.get('max_tokens', 1024)
            seed = params.get('seed')
            
            # Build Ollama request
            request_data = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            if seed is not None:
                request_data["options"]["seed"] = seed
            
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '')
                        
                        # Estimate tokens used
                        prompt_tokens = await self._estimate_tokens(prompt)
                        response_tokens = await self._estimate_tokens(response_text)
                        total_tokens = prompt_tokens + response_tokens
                        
                        return CompletionResult(
                            text=response_text,
                            success=True,
                            metadata={
                                "provider": "ollama",
                                "model": self.model_id,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": response_tokens,
                                "total_tokens": total_tokens,
                                "duration": time.time() - start_time,
                                "temperature": temperature,
                                "seed": seed
                            },
                            tokens_used=total_tokens,
                            cost_usd=0.0  # Ollama is free
                        )
                    else:
                        error_text = await response.text()
                        return CompletionResult(
                            text="",
                            success=False,
                            metadata={"provider": "ollama", "model": self.model_id},
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
                        
            except asyncio.TimeoutError:
                return CompletionResult(
                    text="",
                    success=False,
                    metadata={"provider": "ollama", "model": self.model_id},
                    error_message=f"Request timed out after {self.timeout}s"
                )
            except Exception as e:
                return CompletionResult(
                    text="",
                    success=False,
                    metadata={"provider": "ollama", "model": self.model_id},
                    error_message=f"Request failed: {str(e)}"
                )
    
    async def get_model_info(self) -> ModelInfo:
        """Get model information from Ollama."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/show",
                    json={"name": self.model_id},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_details = data.get('details', {})
                        
                        return ModelInfo(
                            family=self.model_id.split(':')[0],  # e.g., "llama2" from "llama2:7b"
                            n_params=model_details.get('parameter_size'),
                            context_length=model_details.get('context_length'),
                            supports_streaming=True,
                            provider="ollama"
                        )
            except Exception:
                pass
        
        # Fallback info
        return ModelInfo(
            family=self.model_id.split(':')[0],
            provider="ollama",
            supports_streaming=True
        )
    
    async def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Use tokenization if available, otherwise estimate
        try:
            result = await self.tokenize(text)
            return result.token_count
        except Exception:
            # Rough estimation: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.3)


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.model_id = config.get('model_id')
        self.organization = config.get('organization')
        
        if not self.api_key:
            raise ValueError("api_key is required for OpenAIAdapter")
        if not self.model_id:
            raise ValueError("model_id is required for OpenAIAdapter")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
        
        # Initialize tokenizer for this model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_id)
        except KeyError:
            # Fallback to a general tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def tokenize(self, text: str) -> TokenizationResult:
        """Tokenize using tiktoken."""
        try:
            tokens = self.tokenizer.encode(text)
            token_strings = [self.tokenizer.decode([t]) for t in tokens]
            
            return TokenizationResult(
                tokens=token_strings,
                token_count=len(tokens),
                token_ids=tokens
            )
        except Exception as e:
            # Fallback estimation
            estimated_count = len(text.split()) * 1.3
            return TokenizationResult(
                tokens=text.split(),
                token_count=int(estimated_count)
            )
    
    async def complete(self, prompt: str, **params) -> CompletionResult:
        """Generate completion using OpenAI."""
        start_time = time.time()
        
        # Extract relevant parameters
        temperature = params.get('temperature', 0.0)
        max_tokens = params.get('max_tokens', 1024)
        seed = params.get('seed')
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if seed is not None:
                request_params["seed"] = seed
            
            response = await self.client.chat.completions.create(**request_params)
            
            choice = response.choices[0]
            response_text = choice.message.content or ""
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            cost_usd = self.estimate_cost(prompt_tokens, completion_tokens)
            
            return CompletionResult(
                text=response_text,
                success=True,
                metadata={
                    "provider": "openai",
                    "model": self.model_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "duration": time.time() - start_time,
                    "temperature": temperature,
                    "seed": seed,
                    "finish_reason": choice.finish_reason
                },
                tokens_used=total_tokens,
                cost_usd=cost_usd
            )
            
        except Exception as e:
            return CompletionResult(
                text="",
                success=False,
                metadata={"provider": "openai", "model": self.model_id},
                error_message=f"OpenAI API error: {str(e)}"
            )
    
    async def get_model_info(self) -> ModelInfo:
        """Get OpenAI model information."""
        # Static info for known OpenAI models
        model_info_map = {
            "gpt-4": ModelInfo(family="gpt-4", context_length=8192, provider="openai"),
            "gpt-4-turbo": ModelInfo(family="gpt-4", context_length=128000, provider="openai"),
            "gpt-4-turbo-preview": ModelInfo(family="gpt-4", context_length=128000, provider="openai"),
            "gpt-3.5-turbo": ModelInfo(family="gpt-3.5", context_length=4096, provider="openai"),
            "gpt-3.5-turbo-16k": ModelInfo(family="gpt-3.5", context_length=16384, provider="openai"),
        }
        
        return model_info_map.get(
            self.model_id, 
            ModelInfo(family="gpt-unknown", provider="openai")
        )


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.model_id = config.get('model_id')
        self.base_url = config.get('base_url', 'https://api.anthropic.com')
        
        if not self.api_key:
            raise ValueError("api_key is required for AnthropicAdapter")
        if not self.model_id:
            raise ValueError("model_id is required for AnthropicAdapter")
    
    async def tokenize(self, text: str) -> TokenizationResult:
        """Estimate tokenization for Anthropic models."""
        # Anthropic doesn't expose tokenization, so we estimate
        # Claude uses a similar tokenizer to GPT models
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            tokens = tokenizer.encode(text)
            return TokenizationResult(
                tokens=[str(t) for t in tokens],
                token_count=len(tokens),
                token_ids=tokens
            )
        except Exception:
            # Fallback estimation
            estimated_count = len(text.split()) * 1.3
            return TokenizationResult(
                tokens=text.split(),
                token_count=int(estimated_count)
            )
    
    async def complete(self, prompt: str, **params) -> CompletionResult:
        """Generate completion using Anthropic API."""
        start_time = time.time()
        
        # Extract parameters
        temperature = params.get('temperature', 0.0)
        max_tokens = params.get('max_tokens', 1024)
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        request_data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract response text
                        content = data.get('content', [])
                        response_text = ""
                        if content and len(content) > 0:
                            response_text = content[0].get('text', '')
                        
                        # Estimate token usage
                        usage = data.get('usage', {})
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        
                        if not input_tokens:  # Estimate if not provided
                            input_tokens = (await self.tokenize(prompt)).token_count
                        if not output_tokens:  # Estimate if not provided
                            output_tokens = (await self.tokenize(response_text)).token_count
                        
                        cost_usd = self.estimate_cost(input_tokens, output_tokens)
                        
                        return CompletionResult(
                            text=response_text,
                            success=True,
                            metadata={
                                "provider": "anthropic",
                                "model": self.model_id,
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                                "duration": time.time() - start_time,
                                "temperature": temperature
                            },
                            tokens_used=input_tokens + output_tokens,
                            cost_usd=cost_usd
                        )
                    else:
                        error_text = await response.text()
                        return CompletionResult(
                            text="",
                            success=False,
                            metadata={"provider": "anthropic", "model": self.model_id},
                            error_message=f"HTTP {response.status}: {error_text}"
                        )
                        
            except Exception as e:
                return CompletionResult(
                    text="",
                    success=False,
                    metadata={"provider": "anthropic", "model": self.model_id},
                    error_message=f"Anthropic API error: {str(e)}"
                )
    
    async def get_model_info(self) -> ModelInfo:
        """Get Anthropic model information."""
        model_info_map = {
            "claude-3-opus-20240229": ModelInfo(
                family="claude-3", 
                context_length=200000, 
                provider="anthropic"
            ),
            "claude-3-sonnet-20240229": ModelInfo(
                family="claude-3", 
                context_length=200000, 
                provider="anthropic"
            ),
            "claude-3-haiku-20240307": ModelInfo(
                family="claude-3", 
                context_length=200000, 
                provider="anthropic"
            ),
        }
        
        return model_info_map.get(
            self.model_id,
            ModelInfo(family="claude-unknown", provider="anthropic")
        )


def create_adapter(provider: str, config: Dict[str, Any]) -> BaseModelAdapter:
    """Factory function to create appropriate adapter based on provider."""
    if provider.lower() == "ollama":
        return OllamaAdapter(config)
    elif provider.lower() == "openai":
        return OpenAIAdapter(config)
    elif provider.lower() == "anthropic":
        return AnthropicAdapter(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def test_adapter(adapter: BaseModelAdapter, test_prompt: str = "Hello, world!") -> Dict[str, Any]:
    """Test an adapter with a simple prompt."""
    results = {}
    
    # Test tokenization
    try:
        tokenization = await adapter.tokenize(test_prompt)
        results['tokenization'] = {
            'token_count': tokenization.token_count,
            'success': True
        }
    except Exception as e:
        results['tokenization'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test completion
    try:
        completion = await adapter.complete(test_prompt, temperature=0.0, max_tokens=50)
        results['completion'] = {
            'success': completion.success,
            'text_length': len(completion.text) if completion.text else 0,
            'tokens_used': completion.tokens_used,
            'cost_usd': completion.cost_usd,
            'error': completion.error_message
        }
    except Exception as e:
        results['completion'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test model info
    try:
        model_info = await adapter.get_model_info()
        results['model_info'] = {
            'family': model_info.family,
            'provider': model_info.provider,
            'context_length': model_info.context_length,
            'success': True
        }
    except Exception as e:
        results['model_info'] = {
            'success': False,
            'error': str(e)
        }
    
    return results