"""
Tests for the unified model adapter.
"""

import pytest
import time
from unittest.mock import Mock, patch
from scramblebench.llm.model_adapter import (
    ModelAdapter, QueryResult, ModelInterfaceType,
    query_model, batch_query_model, query_model_legacy,
    get_default_adapter, set_default_adapter
)


class TestQueryResult:
    """Test QueryResult dataclass functionality."""
    
    def test_query_result_creation(self):
        """Test creating QueryResult instances."""
        result = QueryResult(
            text="Response text",
            success=True,
            interface_type=ModelInterfaceType.GENERATE_METHOD,
            response_time=1.5
        )
        
        assert result.text == "Response text"
        assert result.success is True
        assert result.interface_type == ModelInterfaceType.GENERATE_METHOD
        assert result.response_time == 1.5
    
    def test_query_result_truth_testing(self):
        """Test truth testing based on success."""
        success_result = QueryResult("text", True)
        failed_result = QueryResult("", False, error="Failed")
        
        assert bool(success_result) is True
        assert bool(failed_result) is False
    
    def test_is_empty_property(self):
        """Test is_empty property."""
        empty_result = QueryResult("", True)
        whitespace_result = QueryResult("   \n  ", True)
        content_result = QueryResult("Content", True)
        
        assert empty_result.is_empty is True
        assert whitespace_result.is_empty is True
        assert content_result.is_empty is False


class TestModelAdapter:
    """Test ModelAdapter functionality."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ModelAdapter(
            timeout=30,
            max_retries=2,
            retry_delay=0.5,
            validate_responses=False
        )
        
        assert adapter.timeout == 30
        assert adapter.max_retries == 2
        assert adapter.retry_delay == 0.5
        assert adapter.validate_responses is False
    
    def test_detect_interface_scramblebench(self):
        """Test interface detection for ScrambleBench models."""
        # Mock a ScrambleBench interface
        mock_model = Mock()
        mock_model.__class__.__bases__ = [Mock()]
        
        adapter = ModelAdapter()
        
        # Mock isinstance check
        with patch('scramblebench.llm.model_adapter.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            interface_type = adapter._detect_interface(mock_model)
            assert interface_type == ModelInterfaceType.SCRAMBLEBENCH_INTERFACE
    
    def test_detect_interface_generate_method(self):
        """Test interface detection for models with generate method."""
        mock_model = Mock()
        mock_model.generate = Mock()
        
        adapter = ModelAdapter()
        interface_type = adapter._detect_interface(mock_model)
        assert interface_type == ModelInterfaceType.GENERATE_METHOD
    
    def test_detect_interface_query_method(self):
        """Test interface detection for models with query method."""
        mock_model = Mock()
        mock_model.query = Mock()
        
        adapter = ModelAdapter()
        interface_type = adapter._detect_interface(mock_model)
        assert interface_type == ModelInterfaceType.QUERY_METHOD
    
    def test_detect_interface_callable(self):
        """Test interface detection for callable models."""
        mock_model = Mock()
        
        adapter = ModelAdapter()
        interface_type = adapter._detect_interface(mock_model)
        assert interface_type == ModelInterfaceType.CALLABLE
    
    def test_detect_interface_caching(self):
        """Test interface detection caching."""
        mock_model = Mock()
        mock_model.generate = Mock()
        
        adapter = ModelAdapter()
        
        # First call should detect and cache
        interface_type1 = adapter._detect_interface(mock_model)
        assert interface_type1 == ModelInterfaceType.GENERATE_METHOD
        
        # Remove the generate method to test caching
        delattr(mock_model, 'generate')
        
        # Second call should use cache, not re-detect
        interface_type2 = adapter._detect_interface(mock_model)
        assert interface_type2 == ModelInterfaceType.GENERATE_METHOD
    
    def test_query_empty_prompt(self):
        """Test query with empty prompt."""
        mock_model = Mock()
        adapter = ModelAdapter()
        
        result = adapter.query(mock_model, "")
        assert result.success is False
        assert "Empty or invalid prompt" in result.error
    
    def test_query_generate_method_success(self):
        """Test successful query with generate method."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Generated response")
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Generated response"
        assert result.interface_type == ModelInterfaceType.GENERATE_METHOD
        assert result.response_time is not None
        mock_model.generate.assert_called_once_with("Test prompt")
    
    def test_query_query_method_success(self):
        """Test successful query with query method."""
        mock_model = Mock()
        mock_model.query = Mock(return_value="Query response")
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Query response"
        assert result.interface_type == ModelInterfaceType.QUERY_METHOD
        mock_model.query.assert_called_once_with("Test prompt")
    
    def test_query_callable_success(self):
        """Test successful query with callable model."""
        mock_model = Mock(return_value="Callable response")
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Callable response"
        assert result.interface_type == ModelInterfaceType.CALLABLE
        mock_model.assert_called_once_with("Test prompt")
    
    def test_query_with_kwargs(self):
        """Test query with additional keyword arguments."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Response")
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt", temperature=0.7, max_tokens=100)
        
        assert result.success is True
        mock_model.generate.assert_called_once_with("Test prompt", temperature=0.7, max_tokens=100)
    
    def test_query_model_exception(self):
        """Test query when model raises exception."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Model error"))
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is False
        assert "Model execution failed" in result.error
        assert "Model error" in result.error
    
    def test_query_unsupported_interface(self):
        """Test query with unsupported model interface."""
        mock_model = Mock(spec=[])  # No callable methods
        
        adapter = ModelAdapter()
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is False
        assert "Unsupported model interface" in result.error
    
    def test_query_with_retries(self):
        """Test query with retry logic."""
        mock_model = Mock()
        # First call fails, second succeeds
        mock_model.generate = Mock(side_effect=[Exception("Temporary error"), "Success"])
        
        adapter = ModelAdapter(max_retries=1, retry_delay=0.01)
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Success"
        assert mock_model.generate.call_count == 2
    
    def test_query_max_retries_exceeded(self):
        """Test query when max retries are exceeded."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Persistent error"))
        
        adapter = ModelAdapter(max_retries=1, retry_delay=0.01)
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is False
        assert "failed after 2 attempts" in result.error
        assert mock_model.generate.call_count == 2
    
    def test_response_validation(self):
        """Test response validation and sanitization."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="  Response with whitespace  ")
        
        adapter = ModelAdapter(validate_responses=True)
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Response with whitespace"  # Stripped
    
    def test_response_validation_empty_response(self):
        """Test validation with empty response."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="   ")
        
        adapter = ModelAdapter(validate_responses=True)
        result = adapter.query(mock_model, "Test prompt")
        
        assert result.success is False
        assert "empty response" in result.error
    
    def test_batch_query_fallback(self):
        """Test batch query fallback to individual queries."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=["Response 1", "Response 2"])
        
        adapter = ModelAdapter()
        results = adapter.batch_query(mock_model, ["Prompt 1", "Prompt 2"])
        
        assert len(results) == 2
        assert results[0].text == "Response 1"
        assert results[1].text == "Response 2"
        assert mock_model.generate.call_count == 2
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=["Success", Exception("Error")])
        
        adapter = ModelAdapter()
        
        # Successful query
        adapter.query(mock_model, "Test 1")
        
        # Failed query
        adapter.query(mock_model, "Test 2")
        
        stats = adapter.get_performance_stats()
        
        assert stats['total_queries'] == 2
        assert stats['total_errors'] == 1
        assert stats['error_rate'] == 0.5
        assert stats['total_response_time'] > 0
        assert stats['average_response_time'] > 0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Response")
        
        adapter = ModelAdapter()
        
        # Populate cache
        adapter.query(mock_model, "Test")
        assert len(adapter._interface_cache) > 0
        
        # Clear cache
        adapter.clear_cache()
        assert len(adapter._interface_cache) == 0
    
    def test_reset_stats(self):
        """Test statistics reset."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Response")
        
        adapter = ModelAdapter()
        
        # Generate some stats
        adapter.query(mock_model, "Test")
        assert adapter._query_count > 0
        
        # Reset stats
        adapter.reset_stats()
        assert adapter._query_count == 0
        assert adapter._total_response_time == 0.0
        assert adapter._error_count == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_query_model_function(self):
        """Test query_model convenience function."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Response")
        
        result = query_model(mock_model, "Test prompt")
        
        assert result.success is True
        assert result.text == "Response"
    
    def test_batch_query_model_function(self):
        """Test batch_query_model convenience function."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=["Response 1", "Response 2"])
        
        results = batch_query_model(mock_model, ["Prompt 1", "Prompt 2"])
        
        assert len(results) == 2
        assert results[0].text == "Response 1"
        assert results[1].text == "Response 2"
    
    def test_query_model_legacy_success(self):
        """Test legacy compatibility function success."""
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Response")
        
        response = query_model_legacy(mock_model, "Test prompt")
        assert response == "Response"
    
    def test_query_model_legacy_interface_error(self):
        """Test legacy function with interface error."""
        mock_model = Mock(spec=[])  # No supported interface
        
        with pytest.raises(ValueError, match="does not have a recognized interface"):
            query_model_legacy(mock_model, "Test prompt")
    
    def test_query_model_legacy_runtime_error(self):
        """Test legacy function with other errors."""
        mock_model = Mock()
        mock_model.generate = Mock(side_effect=Exception("Model error"))
        
        with pytest.raises(RuntimeError):
            query_model_legacy(mock_model, "Test prompt")
    
    def test_default_adapter_management(self):
        """Test default adapter getter and setter."""
        # Test getting default adapter
        default1 = get_default_adapter()
        assert isinstance(default1, ModelAdapter)
        
        # Test getting same instance
        default2 = get_default_adapter()
        assert default1 is default2
        
        # Test setting custom adapter
        custom_adapter = ModelAdapter(timeout=60)
        set_default_adapter(custom_adapter)
        
        default3 = get_default_adapter()
        assert default3 is custom_adapter
        assert default3.timeout == 60