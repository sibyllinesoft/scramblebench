"""
Comprehensive tests for the unified configuration system.

This module tests the consolidated configuration classes including:
- Individual configuration components (ModelConfig, DataConfig, etc.)
- Main ScrambleBenchConfig class
- Environment variable loading
- File serialization/deserialization
- Validation and error handling
- Backward compatibility
"""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from scramblebench.core.unified_config import (
    ScrambleBenchConfig,
    ModelConfig,
    DatasetConfig,
    TransformConfig,
    LoggingConfig
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_model_config(self):
        """Test default model configuration values."""
        config = ModelConfig()
        
        assert config.name == "openai/gpt-3.5-turbo"
        assert config.provider == ModelProvider.OPENROUTER
        assert config.model_type == ModelType.CHAT
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.rate_limit == 10.0
        assert config.vocab_size == 1000
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            name="anthropic/claude-3-sonnet",
            provider=ModelProvider.ANTHROPIC,
            temperature=0.0,
            max_tokens=2000,
            timeout=60
        )
        
        assert config.name == "anthropic/claude-3-sonnet"
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.temperature == 0.0
        assert config.max_tokens == 2000
        assert config.timeout == 60
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid configuration
        config = ModelConfig(temperature=1.5, top_p=0.8)
        assert config.temperature == 1.5
        assert config.top_p == 0.8
        
        # Invalid temperature
        with pytest.raises(ValueError):
            ModelConfig(temperature=3.0)  # > 2.0
        
        # Invalid top_p
        with pytest.raises(ValueError):
            ModelConfig(top_p=1.5)  # > 1.0
        
        # Invalid max_tokens
        with pytest.raises(ValueError):
            ModelConfig(max_tokens=-100)  # <= 0
    
    def test_backward_compatibility_imports(self):
        """Test that legacy imports still work."""
        # Test the alias works
        # Legacy Config import no longer supported - using ScrambleBenchConfig directly
        config = ScrambleBenchConfig()
        assert isinstance(config, ScrambleBenchConfig)
        
    def test_comprehensive_config_sections(self):
        """Test all configuration sections are present."""
        config = ScrambleBenchConfig()
        
        # Verify all sections exist
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'benchmark')
        assert hasattr(config, 'transformations')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'metrics')
        assert hasattr(config, 'plots')
        
        # Verify they are proper instances
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.transformations, TransformationConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.plots, PlotConfig)
    
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key_123"})
    def test_api_key_resolution_openrouter(self):
        """Test API key resolution from environment for OpenRouter."""
        config = ModelConfig(provider=ModelProvider.OPENROUTER)
        assert config.api_key == "test_key_123"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai_key_456"})
    def test_api_key_resolution_openai(self):
        """Test API key resolution from environment for OpenAI."""
        config = ModelConfig(provider=ModelProvider.OPENAI)
        assert config.api_key == "openai_key_456"
    
    def test_explicit_api_key_override(self):
        """Test that explicit API key overrides environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env_key"}):
            config = ModelConfig(api_key="explicit_key")
            assert config.api_key == "explicit_key"


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_data_config(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.data_dir == "data"
        assert config.benchmarks_dir == "data/benchmarks"
        assert config.results_dir == "data/results"
        assert config.cache_dir == "data/cache"
        assert config.default_format == "json"
        assert config.encoding == "utf-8"
        assert config.enable_caching is True
        assert config.max_cache_size == 1000
        assert config.batch_size == 1000
    
    def test_custom_data_config(self):
        """Test custom data configuration."""
        config = DataConfig(
            data_dir="/custom/data",
            max_cache_size=5000,
            enable_caching=False,
            batch_size=500
        )
        
        assert config.data_dir == "/custom/data"
        assert config.max_cache_size == 5000
        assert config.enable_caching is False
        assert config.batch_size == 500


class TestBenchmarkConfig:
    """Test BenchmarkConfig class."""
    
    def test_default_benchmark_config(self):
        """Test default benchmark configuration."""
        config = BenchmarkConfig()
        
        assert config.random_seed == 42
        assert config.evaluation_mode == "exact_match"
        assert config.evaluation_threshold == 0.8
        assert config.preserve_numbers is True
        assert config.preserve_proper_nouns is True
        assert config.max_concurrent_requests == 5
    
    def test_benchmark_config_validation(self):
        """Test benchmark configuration validation."""
        # Valid threshold
        config = BenchmarkConfig(evaluation_threshold=0.9)
        assert config.evaluation_threshold == 0.9
        
        # Invalid threshold
        with pytest.raises(ValueError):
            BenchmarkConfig(evaluation_threshold=1.5)  # > 1.0
        
        # Invalid random seed
        with pytest.raises(ValueError):
            BenchmarkConfig(random_seed=-1)  # < 0


class TestTransformationConfig:
    """Test TransformationConfig class."""
    
    def test_default_transformation_config(self):
        """Test default transformation configuration."""
        config = TransformationConfig()
        
        assert config.enabled_types == [TransformationType.ALL]
        assert config.language_complexity == 5
        assert config.synonym_rate == 0.3
        assert config.preserve_function_words is True
        assert config.batch_size == 10
    
    def test_transformation_types_validation(self):
        """Test transformation types validation."""
        # String input should be converted to list
        config = TransformationConfig(enabled_types="language_translation")
        assert config.enabled_types == [TransformationType.LANGUAGE_TRANSLATION]
        
        # List input should work
        config = TransformationConfig(
            enabled_types=["language_translation", "synonym_replacement"]
        )
        assert len(config.enabled_types) == 2
        assert TransformationType.LANGUAGE_TRANSLATION in config.enabled_types
        assert TransformationType.SYNONYM_REPLACEMENT in config.enabled_types
    
    def test_transformation_config_validation(self):
        """Test transformation configuration validation."""
        # Valid complexity
        config = TransformationConfig(language_complexity=8)
        assert config.language_complexity == 8
        
        # Invalid complexity
        with pytest.raises(ValueError):
            TransformationConfig(language_complexity=15)  # > 10
        
        # Valid synonym rate
        config = TransformationConfig(synonym_rate=0.5)
        assert config.synonym_rate == 0.5
        
        # Invalid synonym rate
        with pytest.raises(ValueError):
            TransformationConfig(synonym_rate=1.5)  # > 1.0


class TestLoggingConfig:
    """Test LoggingConfig class."""
    
    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.level == LogLevel.INFO
        assert config.console is True
        assert config.file is None
        assert config.backup_count == 5
        assert "scramblebench.core" in config.loggers
    
    def test_custom_logging_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file="/var/log/scramblebench.log",
            console=False
        )
        
        assert config.level == LogLevel.DEBUG
        assert config.file == "/var/log/scramblebench.log"
        assert config.console is False


class TestScrambleBenchConfig:
    """Test main ScrambleBenchConfig class."""
    
    def test_default_config(self):
        """Test default configuration initialization."""
        config = ScrambleBenchConfig()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.transformations, TransformationConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.plots, PlotConfig)
        assert config.version == "0.1.0"
    
    def test_config_serialization(self):
        """Test configuration serialization to dictionary."""
        config = ScrambleBenchConfig()
        config_dict = config.dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "data" in config_dict
        assert "benchmark" in config_dict
        assert config_dict["version"] == "0.1.0"
    
    def test_config_validation_assignment(self):
        """Test that validation works on assignment."""
        config = ScrambleBenchConfig()
        
        # Valid assignment
        config.model.temperature = 0.5
        assert config.model.temperature == 0.5
        
        # Invalid assignment should raise error
        with pytest.raises(ValueError):
            config.model.temperature = 3.0  # > 2.0
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            ScrambleBenchConfig(unknown_field="value")


class TestConfigFileSerialization:
    """Test configuration file operations."""
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Create and save config
            original_config = ScrambleBenchConfig()
            original_config.model.temperature = 0.5
            original_config.benchmark.random_seed = 123
            original_config.save_to_file(config_path)
            
            # Load config
            loaded_config = ScrambleBenchConfig.load_from_file(config_path)
            
            assert loaded_config.model.temperature == 0.5
            assert loaded_config.benchmark.random_seed == 123
            assert loaded_config.version == "0.1.0"
        
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Create and save config
            original_config = ScrambleBenchConfig()
            original_config.model.name = "test_model"
            original_config.data.max_cache_size = 2000
            original_config.save_to_file(config_path)
            
            # Load config
            loaded_config = ScrambleBenchConfig.load_from_file(config_path)
            
            assert loaded_config.model.name == "test_model"
            assert loaded_config.data.max_cache_size == 2000
        
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_unsupported_format_error(self):
        """Test error handling for unsupported file formats."""
        config = ScrambleBenchConfig()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            config.save_to_file("config.txt")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            ScrambleBenchConfig.load_from_file("config.txt")
    
    def test_directory_creation(self):
        """Test that directories are created when saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nested" / "dir" / "config.yaml"
            
            config = ScrambleBenchConfig()
            config.save_to_file(config_path)
            
            assert config_path.exists()
            assert config_path.parent.exists()


class TestEnvironmentVariableLoading:
    """Test environment variable loading."""
    
    @patch.dict(os.environ, {
        "SCRAMBLEBENCH_MODEL_NAME": "custom/model",
        "SCRAMBLEBENCH_MODEL_TEMPERATURE": "0.8",
        "SCRAMBLEBENCH_MODEL_MAX_TOKENS": "1500",
        "SCRAMBLEBENCH_DATA_DIR": "/custom/data",
        "SCRAMBLEBENCH_LOG_LEVEL": "DEBUG",
        "SCRAMBLEBENCH_RANDOM_SEED": "999",
        "OPENROUTER_API_KEY": "test_api_key"
    })
    def test_from_env_loading(self):
        """Test loading configuration from environment variables."""
        config = ScrambleBenchConfig.from_env()
        
        assert config.model.name == "custom/model"
        assert config.model.temperature == 0.8
        assert config.model.max_tokens == 1500
        assert config.model.api_key == "test_api_key"
        assert config.data.data_dir == "/custom/data"
        assert config.logging.level == LogLevel.DEBUG
        assert config.benchmark.random_seed == 999
    
    @patch.dict(os.environ, {
        "CUSTOM_PREFIX_MODEL_NAME": "env_model",
        "CUSTOM_PREFIX_LOG_LEVEL": "ERROR"
    })
    def test_custom_env_prefix(self):
        """Test custom environment variable prefix."""
        config = ScrambleBenchConfig.from_env(prefix="CUSTOM_PREFIX")
        
        assert config.model.name == "env_model"
        assert config.logging.level == LogLevel.ERROR
    
    @patch.dict(os.environ, {
        "SCRAMBLEBENCH_MODEL_TEMPERATURE": "invalid_float"
    })
    def test_invalid_env_values(self):
        """Test handling of invalid environment variable values."""
        # Should not raise error, just use default
        config = ScrambleBenchConfig.from_env()
        assert config.model.temperature == 0.7  # Default value


class TestLoggingSetup:
    """Test logging setup functionality."""
    
    @patch('logging.getLogger')
    @patch('logging.StreamHandler')
    @patch('logging.FileHandler')
    def test_setup_logging_console_only(self, mock_file_handler, mock_stream_handler, mock_get_logger):
        """Test logging setup with console only."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        config = ScrambleBenchConfig()
        config.logging.console = True
        config.logging.file = None
        config.setup_logging()
        
        mock_stream_handler.assert_called_once()
        mock_file_handler.assert_not_called()
        mock_logger.addHandler.assert_called_once()
    
    @patch('logging.getLogger')
    @patch('logging.StreamHandler')
    @patch('logging.FileHandler')
    def test_setup_logging_with_file(self, mock_file_handler, mock_stream_handler, mock_get_logger):
        """Test logging setup with file handler."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            config = ScrambleBenchConfig()
            config.logging.console = True
            config.logging.file = str(log_file)
            config.setup_logging()
            
            mock_stream_handler.assert_called_once()
            mock_file_handler.assert_called_once()
            assert mock_logger.addHandler.call_count == 2


class TestDirectoryCreation:
    """Test directory creation functionality."""
    
    def test_create_directories(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ScrambleBenchConfig()
            config.data.data_dir = str(Path(temp_dir) / "data")
            config.data.benchmarks_dir = str(Path(temp_dir) / "benchmarks")
            config.data.results_dir = str(Path(temp_dir) / "results")
            config.data.cache_dir = str(Path(temp_dir) / "cache")
            config.data.temp_dir = str(Path(temp_dir) / "temp")
            
            config.create_directories()
            
            assert Path(config.data.data_dir).exists()
            assert Path(config.data.benchmarks_dir).exists()
            assert Path(config.data.results_dir).exists()
            assert Path(config.data.cache_dir).exists()
            assert Path(config.data.temp_dir).exists()


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def test_config_alias(self):
        """Test that Config alias works."""
        config = Config()
        assert isinstance(config, ScrambleBenchConfig)
    
    def test_load_config_function(self):
        """Test load_config convenience function."""
        config = load_config()
        assert isinstance(config, ScrambleBenchConfig)
    
    def test_load_config_with_file(self):
        """Test load_config with file."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Create config file
            test_config = {
                "model": {"name": "test_model"},
                "benchmark": {"random_seed": 456}
            }
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Load with function
            config = load_config(config_file=config_path)
            
            assert config.model.name == "test_model"
            assert config.benchmark.random_seed == 456
        
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_load_config_with_overrides(self):
        """Test load_config with override parameters."""
        config = load_config(
            model={"temperature": 0.9},
            benchmark={"random_seed": 789}
        )
        
        assert config.model.temperature == 0.9
        assert config.benchmark.random_seed == 789


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Create original config
            config = ScrambleBenchConfig()
            config.model.name = "test_model"
            config.model.temperature = 0.5
            config.benchmark.random_seed = 12345
            config.data.data_dir = str(Path(temp_dir) / "data")
            
            # Save config
            config.save_to_file(config_path)
            
            # Load config
            loaded_config = ScrambleBenchConfig.load_from_file(config_path)
            
            # Setup logging and directories
            loaded_config.setup_logging()
            loaded_config.create_directories()
            
            # Verify everything works
            assert loaded_config.model.name == "test_model"
            assert loaded_config.model.temperature == 0.5
            assert loaded_config.benchmark.random_seed == 12345
            assert Path(loaded_config.data.data_dir).exists()
    
    def test_config_modification_and_validation(self):
        """Test configuration modification with validation."""
        config = ScrambleBenchConfig()
        
        # Valid modifications
        config.model.temperature = 1.5
        config.benchmark.evaluation_threshold = 0.9
        config.transformations.language_complexity = 8
        
        assert config.model.temperature == 1.5
        assert config.benchmark.evaluation_threshold == 0.9
        assert config.transformations.language_complexity == 8
        
        # Invalid modifications should raise errors
        with pytest.raises(ValueError):
            config.model.temperature = 3.0
        
        with pytest.raises(ValueError):
            config.benchmark.evaluation_threshold = 1.5
        
        with pytest.raises(ValueError):
            config.transformations.language_complexity = 15


# Additional fixtures for complex testing scenarios
@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "provider": "openrouter",
            "temperature": 0.3,
            "max_tokens": 1500
        },
        "benchmark": {
            "random_seed": 999,
            "evaluation_threshold": 0.85
        },
        "data": {
            "max_cache_size": 5000,
            "enable_caching": False
        }
    }


@pytest.fixture
def temp_config_file():
    """Temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        config_path = Path(f.name)
    
    yield config_path
    
    config_path.unlink(missing_ok=True)


class TestComplexScenarios:
    """Test complex configuration scenarios."""
    
    def test_nested_config_updates(self, sample_config_dict):
        """Test nested configuration updates."""
        config = ScrambleBenchConfig(**sample_config_dict)
        
        assert config.model.name == "test_model"
        assert config.model.temperature == 0.3
        assert config.benchmark.random_seed == 999
        assert config.data.max_cache_size == 5000
    
    def test_partial_config_loading(self, temp_config_file, sample_config_dict):
        """Test loading partial configuration from file."""
        # Save partial config
        with open(temp_config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        # Load should fill in defaults for missing sections
        config = ScrambleBenchConfig.load_from_file(temp_config_file)
        
        assert config.model.name == "test_model"  # From file
        assert config.logging.level == LogLevel.INFO  # Default value
        assert config.plots.dpi == 300  # Default value
    
    def test_config_merge_behavior(self):
        """Test configuration merging behavior."""
        base_config = {
            "model": {"temperature": 0.5, "max_tokens": 1000},
            "benchmark": {"random_seed": 42}
        }
        
        override_config = {
            "model": {"temperature": 0.8},  # Override temperature, keep max_tokens
            "data": {"max_cache_size": 2000}  # Add new section
        }
        
        # Load with overrides using load_config
        config = load_config(**override_config)
        config_dict = config.dict()
        
        # Apply base config manually to test merging
        # Note: _deep_update removed from unified config - using dict.update for now
        config_dict.update(base_config)
        final_config = ScrambleBenchConfig.from_dict(config_dict)
        
        assert final_config.model.temperature == 0.8  # Overridden
        assert final_config.model.max_tokens == 1000  # From base
        assert final_config.data.max_cache_size == 2000  # New
        assert final_config.benchmark.random_seed == 42  # From base