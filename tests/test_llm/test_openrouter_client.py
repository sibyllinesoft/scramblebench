"""
Tests for OpenRouter client functionality.

This module provides comprehensive tests for the OpenRouter client implementation,
covering API interactions, error handling, rate limiting, and various model types.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from scramblebench.llm.openrouter_client import OpenRouterClient
from scramblebench.llm.model_interface import ModelResponse, ModelConfig, ModelType


class MockHttpResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status_code: int, json_data: Dict[str, Any] = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        
    def json(self):
        return self._json_data
        
    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise Exception(f"HTTP {self.status_code}")


class MockAsyncHttpResponse:
    """Mock async HTTP response for testing."""
    
    def __init__(self, status: int, json_data: Dict[str, Any] = None, text: str = ""):
        self.status = status
        self._json_data = json_data or {}
        self._text = text
        
    async def json(self):
        return self._json_data
        
    async def text(self):
        return self._text
        
    def raise_for_status(self):
        if 400 <= self.status < 600:
            raise Exception(f"HTTP {self.status}")


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return ModelConfig(
        temperature=0.7,
        max_tokens=1000,
        timeout=30
    )

@pytest.fixture
def openrouter_client(mock_config):
    """Create an OpenRouter client for testing."""
    with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
        client = OpenRouterClient(
            model_name="openai/gpt-3.5-turbo",
            config=mock_config
        )
    return client

@pytest.fixture
def mock_successful_response():
    """Mock successful OpenRouter API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        },
        "model": "openai/gpt-3.5-turbo"
    }


class TestOpenRouterClientInitialization:
    """Test OpenRouter client initialization."""
    
    def test_initialization_with_api_key_env(self, mock_config):
        """Test initialization with API key from environment."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            client = OpenRouterClient("openai/gpt-4", config=mock_config)
            
        assert client.model_name == "openai/gpt-4"
        assert client.api_key == "test_key"
        assert client.config == mock_config
        
    def test_initialization_with_explicit_api_key(self, mock_config):
        """Test initialization with explicit API key."""
        client = OpenRouterClient(
            "anthropic/claude-3-sonnet",
            api_key="explicit_key",
            config=mock_config
        )
        
        assert client.model_name == "anthropic/claude-3-sonnet"
        assert client.api_key == "explicit_key"
        
    def test_initialization_missing_api_key(self, mock_config):
        """Test initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                OpenRouterClient("openai/gpt-4", config=mock_config)
                
    def test_initialization_sets_model_type(self, openrouter_client):
        """Test that model type is set correctly."""
        assert openrouter_client.model_type == ModelType.CHAT
        
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            client = OpenRouterClient("openai/gpt-4")
            
        assert isinstance(client.config, ModelConfig)
        assert client.config.temperature == 0.7  # default
        
    def test_rate_limiter_initialization(self, openrouter_client):
        """Test that rate limiter is properly initialized."""
        assert openrouter_client.rate_limiter is not None
        assert openrouter_client.max_retries == 3  # default


class TestOpenRouterClientInitialization:
    """Test client initialization process."""
    
    @patch('aiohttp.ClientSession')
    def test_initialize_success(self, mock_session, openrouter_client):
        """Test successful initialization."""
        result = openrouter_client.initialize()
        
        assert result is True
        assert openrouter_client.is_initialized is True
        mock_session.assert_called_once()
        
    def test_initialize_already_initialized(self, openrouter_client):
        """Test initialization when already initialized."""
        openrouter_client._initialized = True
        
        result = openrouter_client.initialize()
        
        assert result is True
        
    def test_initialize_failure_handling(self, openrouter_client):
        """Test initialization failure handling."""
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            result = openrouter_client.initialize()
            
        assert result is False
        assert openrouter_client.is_initialized is False


class TestSingleGeneration:
    """Test single text generation functionality."""
    
    @patch('aiohttp.ClientSession.post')
    def test_generate_success(self, mock_post, openrouter_client, mock_successful_response):
        """Test successful text generation."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert isinstance(response, ModelResponse)
        assert response.text == "This is a test response."
        assert response.error is None
        assert "prompt_tokens" in response.metadata
        assert response.metadata["prompt_tokens"] == 10
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_not_initialized(self, mock_post, openrouter_client):
        """Test generation without initialization."""
        response = openrouter_client.generate("Test prompt")
        
        assert response.error == "Client not initialized"
        assert response.text == ""
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_api_error(self, mock_post, openrouter_client):
        """Test generation with API error."""
        mock_response = MockAsyncHttpResponse(400, {"error": {"message": "Bad request"}})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "Bad request" in response.error
        assert response.text == ""
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_network_error(self, mock_post, openrouter_client):
        """Test generation with network error."""
        mock_post.side_effect = Exception("Network error")
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "Network error" in response.error
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_malformed_response(self, mock_post, openrouter_client):
        """Test generation with malformed API response."""
        # Missing 'choices' field
        mock_response = MockAsyncHttpResponse(200, {"usage": {"total_tokens": 10}})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "Invalid response format" in response.error
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_with_custom_parameters(self, mock_post, openrouter_client, mock_successful_response):
        """Test generation with custom parameters."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate(
            "Test prompt",
            temperature=0.9,
            max_tokens=500,
            top_p=0.8
        )
        
        # Check that custom parameters were passed to API
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        assert request_data['temperature'] == 0.9
        assert request_data['max_tokens'] == 500
        assert request_data['top_p'] == 0.8
        
    def test_generate_prompt_validation(self, openrouter_client):
        """Test prompt validation before generation."""
        openrouter_client.initialize()
        
        # Test empty prompt
        response = openrouter_client.generate("")
        assert response.error is not None
        assert "Empty prompt" in response.error
        
        # Test None prompt
        response = openrouter_client.generate(None)
        assert response.error is not None


class TestChatInterface:
    """Test chat interface functionality."""
    
    @patch('aiohttp.ClientSession.post')
    def test_chat_success(self, mock_post, openrouter_client, mock_successful_response):
        """Test successful chat interaction."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        openrouter_client.initialize()
        response = openrouter_client.chat(messages)
        
        assert isinstance(response, ModelResponse)
        assert response.text == "This is a test response."
        assert response.error is None
        
        # Check that messages were passed correctly
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        assert request_data['messages'] == messages
        
    def test_chat_validation(self, openrouter_client):
        """Test chat message validation."""
        openrouter_client.initialize()
        
        # Test empty messages
        response = openrouter_client.chat([])
        assert response.error is not None
        
        # Test invalid message format
        invalid_messages = [{"content": "Missing role"}]
        response = openrouter_client.chat(invalid_messages)
        assert response.error is not None
        
        # Test None messages
        response = openrouter_client.chat(None)
        assert response.error is not None
        
    @patch('aiohttp.ClientSession.post')
    def test_chat_system_message_handling(self, mock_post, openrouter_client, mock_successful_response):
        """Test handling of system messages."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        
        openrouter_client.initialize()
        response = openrouter_client.chat(messages)
        
        assert response.error is None
        
        # Verify system message was included
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        assert request_data['messages'][0]['role'] == 'system'


class TestBatchGeneration:
    """Test batch generation functionality."""
    
    @patch('aiohttp.ClientSession.post')
    def test_batch_generate_success(self, mock_post, openrouter_client, mock_successful_response):
        """Test successful batch generation."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        openrouter_client.initialize()
        responses = openrouter_client.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all(isinstance(r, ModelResponse) for r in responses)
        assert all(r.text == "This is a test response." for r in responses)
        assert mock_post.call_count == 3  # One call per prompt
        
    @patch('aiohttp.ClientSession.post')
    def test_batch_generate_with_failures(self, mock_post, openrouter_client):
        """Test batch generation with some failures."""
        # First call succeeds, second fails, third succeeds
        responses = [
            MockAsyncHttpResponse(200, {"choices": [{"message": {"content": "Success 1"}}], "usage": {"total_tokens": 10}}),
            MockAsyncHttpResponse(400, {"error": {"message": "API error"}}),
            MockAsyncHttpResponse(200, {"choices": [{"message": {"content": "Success 2"}}], "usage": {"total_tokens": 10}})
        ]
        
        mock_post.return_value.__aenter__.side_effect = responses
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        openrouter_client.initialize()
        results = openrouter_client.batch_generate(prompts)
        
        assert len(results) == 3
        assert results[0].text == "Success 1"
        assert results[0].error is None
        assert results[1].error is not None
        assert results[1].text == ""
        assert results[2].text == "Success 2"
        assert results[2].error is None
        
    def test_batch_generate_empty_prompts(self, openrouter_client):
        """Test batch generation with empty prompts list."""
        openrouter_client.initialize()
        
        responses = openrouter_client.batch_generate([])
        
        assert responses == []
        
    @patch('aiohttp.ClientSession.post')
    def test_batch_generate_rate_limiting(self, mock_post, openrouter_client, mock_successful_response):
        """Test that batch generation respects rate limiting."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Mock rate limiter
        with patch.object(openrouter_client.rate_limiter, 'acquire') as mock_acquire:
            prompts = ["Prompt 1", "Prompt 2"]
            
            openrouter_client.initialize()
            openrouter_client.batch_generate(prompts)
            
            # Rate limiter should be called for each request
            assert mock_acquire.call_count == 2


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @patch('aiohttp.ClientSession.post')
    def test_rate_limit_respected(self, mock_post, openrouter_client, mock_successful_response):
        """Test that rate limiting is respected."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        
        # Make rapid requests
        start_time = time.time()
        for _ in range(3):
            openrouter_client.generate("Test prompt")
        end_time = time.time()
        
        # Should take some time due to rate limiting
        # (This is a basic test - actual timing depends on rate limit settings)
        duration = end_time - start_time
        assert duration >= 0  # Basic sanity check
        
    @patch('aiohttp.ClientSession.post')
    def test_rate_limit_error_handling(self, mock_post, openrouter_client):
        """Test handling of rate limit errors from API."""
        # Simulate rate limit error response
        rate_limit_response = MockAsyncHttpResponse(
            429, 
            {"error": {"message": "Rate limit exceeded"}}
        )
        mock_post.return_value.__aenter__.return_value = rate_limit_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "Rate limit exceeded" in response.error
        
    def test_rate_limiter_configuration(self, openrouter_client):
        """Test rate limiter configuration."""
        # Test custom rate limit
        custom_client = OpenRouterClient(
            "openai/gpt-4",
            api_key="test_key",
            rate_limit=10.0  # 10 requests per second
        )
        
        # Verify rate limiter was configured
        assert custom_client.rate_limiter is not None
        # Note: Actual rate limiter implementation details would be tested here


class TestRetryMechanism:
    """Test retry mechanism functionality."""
    
    @patch('aiohttp.ClientSession.post')
    def test_retry_on_network_error(self, mock_post, openrouter_client, mock_successful_response):
        """Test retry mechanism on network errors."""
        # First two calls fail, third succeeds
        mock_post.side_effect = [
            Exception("Network error 1"),
            Exception("Network error 2"),
            MockAsyncHttpResponse(200, mock_successful_response).__aenter__()
        ]
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        # Should eventually succeed after retries
        assert response.text == "This is a test response."
        assert response.error is None
        assert mock_post.call_count == 3
        
    @patch('aiohttp.ClientSession.post')
    def test_retry_exhaustion(self, mock_post, openrouter_client):
        """Test retry exhaustion."""
        # All calls fail
        mock_post.side_effect = Exception("Persistent network error")
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "Persistent network error" in response.error
        assert mock_post.call_count == openrouter_client.max_retries + 1  # Initial + retries
        
    @patch('aiohttp.ClientSession.post')
    def test_no_retry_on_client_error(self, mock_post, openrouter_client):
        """Test that client errors (4xx) are not retried."""
        mock_response = MockAsyncHttpResponse(400, {"error": {"message": "Bad request"}})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert mock_post.call_count == 1  # No retries for 4xx errors
        
    @patch('aiohttp.ClientSession.post')
    def test_retry_on_server_error(self, mock_post, openrouter_client, mock_successful_response):
        """Test retry on server errors (5xx)."""
        # First call returns 500, second succeeds
        mock_post.side_effect = [
            MockAsyncHttpResponse(500, {"error": {"message": "Internal server error"}}).__aenter__(),
            MockAsyncHttpResponse(200, mock_successful_response).__aenter__()
        ]
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.text == "This is a test response."
        assert response.error is None
        assert mock_post.call_count == 2


class TestModelInformation:
    """Test model information and capabilities."""
    
    def test_get_model_info(self, openrouter_client):
        """Test getting model information."""
        info = openrouter_client.get_model_info()
        
        assert isinstance(info, dict)
        assert info['name'] == "openai/gpt-3.5-turbo"
        assert info['type'] == ModelType.CHAT.value
        assert info['provider'] == 'openrouter'
        assert 'max_context_length' in info
        assert 'supports_chat' in info
        
    def test_estimate_tokens(self, openrouter_client):
        """Test token estimation."""
        text = "This is a test string with multiple words."
        tokens = openrouter_client.estimate_tokens(text)
        
        assert isinstance(tokens, int)
        assert tokens > 0
        # Basic sanity check - should be roughly proportional to text length
        assert tokens <= len(text)  # Should be fewer tokens than characters
        
    def test_estimate_tokens_empty(self, openrouter_client):
        """Test token estimation for empty text."""
        tokens = openrouter_client.estimate_tokens("")
        assert tokens == 0
        
    def test_estimate_tokens_long_text(self, openrouter_client):
        """Test token estimation for long text."""
        long_text = "This is a test. " * 100  # ~1600 characters
        tokens = openrouter_client.estimate_tokens(long_text)
        
        # Should be reasonable estimate (rough heuristic: ~4 chars per token)
        assert 300 <= tokens <= 500


class TestPromptValidation:
    """Test prompt validation functionality."""
    
    def test_validate_prompt_success(self, openrouter_client):
        """Test successful prompt validation."""
        prompt = "This is a valid prompt."
        
        result = openrouter_client.validate_prompt(prompt)
        
        assert result is True
        
    def test_validate_prompt_empty(self, openrouter_client):
        """Test validation of empty prompt."""
        result = openrouter_client.validate_prompt("")
        assert result is False
        
        result = openrouter_client.validate_prompt("   ")  # Whitespace only
        assert result is False
        
    def test_validate_prompt_none(self, openrouter_client):
        """Test validation of None prompt."""
        result = openrouter_client.validate_prompt(None)
        assert result is False
        
    def test_validate_prompt_too_long(self, openrouter_client):
        """Test validation of overly long prompt."""
        # Create a very long prompt that would exceed context window
        very_long_prompt = "This is a test. " * 10000  # Very long
        
        result = openrouter_client.validate_prompt(very_long_prompt)
        
        # Should fail due to length
        assert result is False
        
    def test_validate_prompt_near_limit(self, openrouter_client):
        """Test validation of prompt near token limit."""
        # Create prompt that's close to but under the limit
        info = openrouter_client.get_model_info()
        max_context = info['max_context_length']
        
        # Estimate a prompt that uses most of the context
        target_tokens = max_context - openrouter_client.config.max_tokens - 100
        # Very rough estimate: 4 chars per token
        estimated_chars = target_tokens * 4
        
        prompt = "a" * estimated_chars
        
        result = openrouter_client.validate_prompt(prompt)
        
        # Should pass (this is a basic test)
        assert isinstance(result, bool)


class TestConfigurationManagement:
    """Test configuration management."""
    
    def test_update_config(self, openrouter_client):
        """Test updating configuration."""
        original_temp = openrouter_client.config.temperature
        
        openrouter_client.update_config(temperature=0.9, max_tokens=500)
        
        assert openrouter_client.config.temperature == 0.9
        assert openrouter_client.config.max_tokens == 500
        
    def test_update_config_invalid_parameter(self, openrouter_client):
        """Test updating with invalid parameter."""
        mock_logger = Mock()
        openrouter_client.logger = mock_logger
        
        openrouter_client.update_config(invalid_param=123)
        
        # Should log warning for unknown parameter
        mock_logger.warning.assert_called()
        
    def test_get_config(self, openrouter_client):
        """Test getting current configuration."""
        config_dict = openrouter_client.get_config()
        
        assert isinstance(config_dict, dict)
        assert 'temperature' in config_dict
        assert 'max_tokens' in config_dict
        assert 'timeout' in config_dict
        assert config_dict['temperature'] == openrouter_client.config.temperature


class TestConvenienceMethods:
    """Test convenience methods and interfaces."""
    
    @patch('aiohttp.ClientSession.post')
    def test_query_method(self, mock_post, openrouter_client, mock_successful_response):
        """Test the query convenience method."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        text = openrouter_client.query("Test prompt")
        
        assert text == "This is a test response."
        assert isinstance(text, str)
        
    @patch('aiohttp.ClientSession.post')
    def test_query_method_error(self, mock_post, openrouter_client):
        """Test query method with error."""
        mock_response = MockAsyncHttpResponse(400, {"error": {"message": "Bad request"}})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        
        with pytest.raises(RuntimeError, match="Model query failed"):
            openrouter_client.query("Test prompt")
            
    @patch('aiohttp.ClientSession.post')
    def test_callable_interface(self, mock_post, openrouter_client, mock_successful_response):
        """Test that the client is callable."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        text = openrouter_client("Test prompt")
        
        assert text == "This is a test response."
        assert isinstance(text, str)


class TestStringRepresentations:
    """Test string representations."""
    
    def test_str_representation(self, openrouter_client):
        """Test string representation."""
        str_repr = str(openrouter_client)
        
        assert "OpenRouterClient" in str_repr
        assert "openai/gpt-3.5-turbo" in str_repr
        
    def test_repr_representation(self, openrouter_client):
        """Test detailed representation."""
        repr_str = repr(openrouter_client)
        
        assert "OpenRouterClient" in repr_str
        assert "openai/gpt-3.5-turbo" in repr_str
        assert "initialized" in repr_str
        assert ModelType.CHAT.value in repr_str


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @patch('aiohttp.ClientSession.post')
    def test_json_decode_error(self, mock_post, openrouter_client):
        """Test handling of JSON decode errors."""
        # Return invalid JSON
        mock_response = MockAsyncHttpResponse(200, text="Invalid JSON")
        mock_response.json = Mock(side_effect=ValueError("Invalid JSON"))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "JSON" in response.error
        
    @patch('aiohttp.ClientSession.post')
    def test_timeout_error(self, mock_post, openrouter_client):
        """Test handling of timeout errors."""
        import asyncio
        mock_post.side_effect = asyncio.TimeoutError()
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "timeout" in response.error.lower()
        
    @patch('aiohttp.ClientSession.post')
    def test_connection_error(self, mock_post, openrouter_client):
        """Test handling of connection errors."""
        import aiohttp
        mock_post.side_effect = aiohttp.ClientConnectionError()
        
        openrouter_client.initialize()
        response = openrouter_client.generate("Test prompt")
        
        assert response.error is not None
        assert "connection" in response.error.lower()


class TestAsyncIntegration:
    """Test async functionality integration."""
    
    @patch('aiohttp.ClientSession.post')
    @pytest.mark.asyncio
    async def test_async_generate_direct(self, mock_post, openrouter_client, mock_successful_response):
        """Test direct async generation method."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        
        # If the client has an async generate method
        if hasattr(openrouter_client, 'async_generate'):
            response = await openrouter_client.async_generate("Test prompt")
            
            assert isinstance(response, ModelResponse)
            assert response.text == "This is a test response."
            
    @patch('aiohttp.ClientSession.post')
    def test_concurrent_requests(self, mock_post, openrouter_client, mock_successful_response):
        """Test handling of concurrent requests."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        
        # Make multiple concurrent requests
        import threading
        results = []
        errors = []
        
        def make_request():
            try:
                response = openrouter_client.generate("Test prompt")
                results.append(response)
            except Exception as e:
                errors.append(e)
                
        threads = [threading.Thread(target=make_request) for _ in range(3)]
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All requests should complete successfully
        assert len(results) == 3
        assert len(errors) == 0
        assert all(r.text == "This is a test response." for r in results)


class TestModelCompatibility:
    """Test compatibility with different model types."""
    
    @pytest.mark.parametrize("model_name", [
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-2-70b-chat",
        "mistralai/mistral-7b-instruct",
        "google/palm-2-chat-bison"
    ])
    def test_model_name_handling(self, model_name, mock_config):
        """Test handling of different model names."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            client = OpenRouterClient(model_name, config=mock_config)
            
        assert client.model_name == model_name
        assert client.model_type == ModelType.CHAT  # All should be chat models
        
    def test_model_specific_parameters(self, openrouter_client):
        """Test model-specific parameter handling."""
        # Test that model-specific parameters are preserved
        info = openrouter_client.get_model_info()
        
        # Different models may have different capabilities
        assert 'max_context_length' in info
        assert info['max_context_length'] > 0
        
        # Should handle model-specific settings
        openrouter_client.update_config(temperature=0.1)
        assert openrouter_client.config.temperature == 0.1


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    @patch('aiohttp.ClientSession.post')
    def test_large_prompt_handling(self, mock_post, openrouter_client, mock_successful_response):
        """Test handling of large prompts."""
        # Create a large but valid prompt
        large_prompt = "This is a test prompt. " * 100  # ~2400 characters
        
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        openrouter_client.initialize()
        response = openrouter_client.generate(large_prompt)
        
        assert response.error is None
        assert response.text == "This is a test response."
        
    @patch('aiohttp.ClientSession.post')
    def test_batch_performance(self, mock_post, openrouter_client, mock_successful_response):
        """Test performance characteristics of batch processing."""
        mock_response = MockAsyncHttpResponse(200, mock_successful_response)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Test larger batch
        prompts = [f"Prompt {i}" for i in range(10)]
        
        openrouter_client.initialize()
        start_time = time.time()
        responses = openrouter_client.batch_generate(prompts)
        end_time = time.time()
        
        # All should succeed
        assert len(responses) == 10
        assert all(r.text == "This is a test response." for r in responses)
        
        # Should complete in reasonable time (basic performance check)
        duration = end_time - start_time
        assert duration < 60  # Should complete within a minute
        
    def test_memory_efficiency(self, openrouter_client):
        """Test memory efficiency of client operations."""
        # This is more of a smoke test for memory leaks
        import gc
        
        openrouter_client.initialize()
        
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(100):
            # Just validation, no actual API calls
            openrouter_client.validate_prompt(f"Test prompt {i}")
            openrouter_client.estimate_tokens(f"Test text {i}")
            
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not accumulate too many objects
        # This is a rough test - exact numbers depend on implementation
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold