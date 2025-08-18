"""
Tests for model interface and factory functionality.

This module tests the abstract model interface, dummy model implementation,
and model factory system for creating and managing LLM instances.
"""

import pytest
from unittest.mock import Mock, patch

from scramblebench.llm.model_interface import (
    ModelInterface, ModelResponse, ModelConfig, ModelType,
    DummyModel, ModelFactory
)


class TestModelConfig:
    """Test suite for ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.stop_sequences is None
        assert config.timeout == 30
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelConfig(
            temperature=0.5,
            max_tokens=2000,
            top_p=0.8,
            timeout=60
        )
        
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.top_p == 0.8
        assert config.timeout == 60


class TestModelResponse:
    """Test suite for ModelResponse class."""
    
    def test_response_creation(self):
        """Test creation of model responses."""
        response = ModelResponse(
            text="Test response",
            metadata={"tokens": 10},
            raw_response={"data": "raw"}
        )
        
        assert response.text == "Test response"
        assert response.metadata == {"tokens": 10}
        assert response.raw_response == {"data": "raw"}
        assert response.error is None
    
    def test_error_response(self):
        """Test creation of error responses."""
        response = ModelResponse(
            text="",
            metadata={},
            error="API error"
        )
        
        assert response.text == ""
        assert response.error == "API error"


class TestDummyModel:
    """Test suite for DummyModel implementation."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a DummyModel instance for testing."""
        return DummyModel(
            model_name="test-dummy",
            responses=["Response 1", "Response 2", "Response 3"]
        )
    
    def test_dummy_model_initialization(self, dummy_model):
        """Test dummy model initialization."""
        assert dummy_model.model_name == "test-dummy"
        assert dummy_model.model_type == ModelType.CHAT
        assert not dummy_model.is_initialized
        
        # Initialize
        success = dummy_model.initialize()
        assert success
        assert dummy_model.is_initialized
    
    def test_dummy_model_generation(self, dummy_model):
        """Test text generation with dummy model."""
        dummy_model.initialize()
        
        # Generate responses
        response1 = dummy_model.generate("Test prompt 1")
        response2 = dummy_model.generate("Test prompt 2")
        response3 = dummy_model.generate("Test prompt 3")
        response4 = dummy_model.generate("Test prompt 4")  # Should cycle back
        
        assert response1.text == "Response 1"
        assert response2.text == "Response 2"
        assert response3.text == "Response 3"
        assert response4.text == "Response 1"  # Cycled back
        
        # Check metadata
        assert response1.metadata["model"] == "test-dummy"
        assert "response_index" in response1.metadata
    
    def test_dummy_model_chat(self, dummy_model):
        """Test chat functionality with dummy model."""
        dummy_model.initialize()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = dummy_model.chat(messages)
        assert response.text in dummy_model.responses
        assert response.error is None
    
    def test_dummy_model_batch_generation(self, dummy_model):
        """Test batch generation with dummy model."""
        dummy_model.initialize()
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = dummy_model.batch_generate(prompts)
        
        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, ModelResponse)
            assert response.text in dummy_model.responses
    
    def test_dummy_model_uninitialized(self, dummy_model):
        """Test behavior when model is not initialized."""
        response = dummy_model.generate("Test prompt")
        
        assert response.text == ""
        assert response.error == "Model not initialized"
    
    def test_dummy_model_query_interface(self, dummy_model):
        """Test the simple query interface."""
        dummy_model.initialize()
        
        result = dummy_model.query("Test prompt")
        assert isinstance(result, str)
        assert result in dummy_model.responses
    
    def test_dummy_model_callable_interface(self, dummy_model):
        """Test that model is callable."""
        dummy_model.initialize()
        
        result = dummy_model("Test prompt")
        assert isinstance(result, str)
        assert result in dummy_model.responses
    
    def test_dummy_model_config_update(self, dummy_model):
        """Test configuration updates."""
        dummy_model.initialize()
        
        # Update config
        dummy_model.update_config(temperature=0.5, max_tokens=1500)
        
        config = dummy_model.get_config()
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 1500
    
    def test_dummy_model_info(self, dummy_model):
        """Test model information retrieval."""
        info = dummy_model.get_model_info()
        
        assert info["name"] == "test-dummy"
        assert info["type"] == ModelType.CHAT.value
        assert info["provider"] == "dummy"
        assert "max_context_length" in info
        assert "supports_chat" in info
    
    def test_dummy_model_token_estimation(self, dummy_model):
        """Test token estimation."""
        text = "This is a test sentence for token estimation."
        tokens = dummy_model.estimate_tokens(text)
        
        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens == len(text) // 4  # Dummy implementation
    
    def test_dummy_model_prompt_validation(self, dummy_model):
        """Test prompt validation."""
        # Valid prompt
        assert dummy_model.validate_prompt("Valid prompt")
        
        # Empty prompt
        assert not dummy_model.validate_prompt("")
        assert not dummy_model.validate_prompt("   ")
    
    def test_dummy_model_response_management(self, dummy_model):
        """Test response management functionality."""
        dummy_model.initialize()
        
        # Set new responses
        new_responses = ["New response 1", "New response 2"]
        dummy_model.set_responses(new_responses)
        
        response = dummy_model.generate("Test")
        assert response.text in new_responses
        
        # Add additional response
        dummy_model.add_response("Additional response")
        assert "Additional response" in dummy_model.responses


class TestModelFactory:
    """Test suite for ModelFactory class."""
    
    def test_factory_model_registration(self):
        """Test model registration with factory."""
        # Custom model class for testing
        class TestModel(ModelInterface):
            def initialize(self):
                self._initialized = True
                return True
            
            def generate(self, prompt, **kwargs):
                return ModelResponse(text="test", metadata={})
            
            def chat(self, messages, **kwargs):
                return ModelResponse(text="test", metadata={})
            
            def batch_generate(self, prompts, **kwargs):
                return [ModelResponse(text="test", metadata={}) for _ in prompts]
            
            def get_model_info(self):
                return {"name": self.model_name}
            
            def estimate_tokens(self, text):
                return len(text) // 4
        
        # Register test model
        ModelFactory.register_model("test", TestModel)
        
        # Check it's registered
        assert "test" in ModelFactory.list_providers()
        
        # Create model instance
        model = ModelFactory.create_model("test", "test-model")
        assert isinstance(model, TestModel)
        assert model.is_initialized
    
    def test_factory_unknown_provider(self):
        """Test handling of unknown providers."""
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelFactory.create_model("unknown", "test-model")
    
    def test_factory_list_providers(self):
        """Test listing available providers."""
        providers = ModelFactory.list_providers()
        
        assert isinstance(providers, list)
        assert "dummy" in providers  # Should be registered by default
    
    def test_factory_create_dummy_model(self):
        """Test creating dummy model through factory."""
        model = ModelFactory.create_model("dummy", "test-model")
        
        assert isinstance(model, DummyModel)
        assert model.model_name == "test-model"
        assert model.is_initialized


class TestModelInterfaceAbstract:
    """Test suite for abstract ModelInterface behavior."""
    
    def test_abstract_interface_cannot_instantiate(self):
        """Test that abstract interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelInterface("test-model")
    
    def test_interface_properties(self):
        """Test interface properties using dummy model."""
        model = DummyModel("test-model")
        
        assert model.name == "test-model"
        assert model.model_type == ModelType.CHAT
        assert not model.is_initialized
        
        model.initialize()
        assert model.is_initialized
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ModelConfig(temperature=-1)  # Invalid temperature
        model = DummyModel("test", config=config)
        
        # Model should still initialize (dummy doesn't validate)
        assert model.initialize()
    
    def test_string_representations(self):
        """Test string representations of models."""
        model = DummyModel("test-model")
        
        str_repr = str(model)
        assert "DummyModel" in str_repr
        assert "test-model" in str_repr
        
        detailed_repr = repr(model)
        assert "DummyModel" in detailed_repr
        assert "test-model" in detailed_repr
        assert "initialized=False" in detailed_repr
    
    def test_query_error_handling(self):
        """Test error handling in query interface."""
        model = DummyModel("test")
        model.initialize()
        
        # Mock generate to return error
        error_response = ModelResponse(text="", metadata={}, error="Test error")
        
        with patch.object(model, 'generate', return_value=error_response):
            with pytest.raises(RuntimeError, match="Model query failed"):
                model.query("test prompt")