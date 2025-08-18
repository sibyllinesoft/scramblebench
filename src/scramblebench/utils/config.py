"""
Configuration management for ScrambleBench (DEPRECATED).

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`scramblebench.core.config` instead.
    The new unified configuration system provides better validation, type safety,
    and consistency across the entire framework.

This module provides a flexible configuration system that supports
YAML files, environment variables, and programmatic configuration
with validation and type checking.

Migration Guide:
    Replace imports from this module with the new unified system:
    
    .. code-block:: python
    
        # Old (deprecated)
        from scramblebench.utils.config import Config, ModelConfig
        
        # New (recommended)
        from scramblebench.core.config import ScrambleBenchConfig, ModelConfig
        
        # Or use the backward compatibility alias
        from scramblebench.core.config import Config
"""

from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import os
import yaml
import json
import logging
from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    random_seed: int = 42
    evaluation_mode: str = "exact_match"
    evaluation_threshold: float = 0.8
    preserve_numbers: bool = True
    preserve_proper_nouns: bool = True
    preserve_structure: bool = True
    preserve_entities: bool = True
    prompt_template: str = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_dir: str = "data"
    benchmarks_dir: str = "data/benchmarks"
    languages_dir: str = "data/languages"
    results_dir: str = "data/results"
    cache_dir: str = "data/cache"
    max_cache_size: int = 1000  # Maximum cached items


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    default_provider: str = "openrouter"
    default_model: str = "openai/gpt-3.5-turbo"
    timeout: int = 30
    max_retries: int = 3
    rate_limit: float = 10.0  # requests per second
    vocab_size: int = 1000


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True


class Config:
    """
    Flexible configuration management system for ScrambleBench.
    
    Supports loading from YAML files, environment variables,
    and programmatic configuration with validation and defaults.
    """
    
    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        load_env: bool = True
    ):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
            config_dict: Dictionary with configuration values
            load_env: Whether to load values from environment variables
        """
        self.logger = logging.getLogger("scramblebench.config")
        
        # Initialize with defaults
        self._config = self._get_default_config()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Update with provided dictionary
        if config_dict:
            self.update(config_dict)
        
        # Load environment variables
        if load_env:
            self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'benchmark': asdict(BenchmarkConfig()),
            'data': asdict(DataConfig()),
            'model': asdict(ModelConfig()),
            'logging': asdict(LoggingConfig())
        }
    
    def load_from_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            self.logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            if file_config:
                self._deep_update(self._config, file_config)
                self.logger.info(f"Loaded configuration from: {config_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def _load_from_env(self) -> None:
        """Load configuration values from environment variables."""
        env_mappings = {
            'SCRAMBLEBENCH_RANDOM_SEED': ('benchmark', 'random_seed', int),
            'SCRAMBLEBENCH_DATA_DIR': ('data', 'data_dir', str),
            'SCRAMBLEBENCH_MODEL_PROVIDER': ('model', 'default_provider', str),
            'SCRAMBLEBENCH_MODEL_NAME': ('model', 'default_model', str),
            'SCRAMBLEBENCH_LOG_LEVEL': ('logging', 'level', str),
            'OPENROUTER_API_KEY': ('model', 'api_key', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    typed_value = type_func(value)
                    if section not in self._config:
                        self._config[section] = {}
                    self._config[section][key] = typed_value
                    self.logger.debug(f"Loaded from env {env_var}: {key} = {typed_value}")
                except ValueError as e:
                    self.logger.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update target dictionary with source values."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        try:
            # Validate random seed
            seed = self.get('benchmark.random_seed')
            if not isinstance(seed, int) or seed < 0:
                raise ValueError(f"Invalid random_seed: {seed}")
            
            # Validate evaluation threshold
            threshold = self.get('benchmark.evaluation_threshold')
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError(f"Invalid evaluation_threshold: {threshold}")
            
            # Validate rate limit
            rate_limit = self.get('model.rate_limit')
            if not isinstance(rate_limit, (int, float)) or rate_limit <= 0:
                raise ValueError(f"Invalid rate_limit: {rate_limit}")
            
            # Validate directories exist (create if needed)
            for dir_key in ['data_dir', 'benchmarks_dir', 'languages_dir', 'results_dir', 'cache_dir']:
                dir_path = Path(self.get(f'data.{dir_key}'))
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'benchmark.random_seed')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                value = value[k]
            
            return value
        
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'benchmark.random_seed')
            value: Value to set
        """
        keys = key.split('.')
        target = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value
        self.logger.debug(f"Set config {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration updates
        """
        self._deep_update(self._config, config_dict)
        self.logger.debug("Configuration updated from dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        
        Returns:
            Deep copy of the configuration dictionary
        """
        return deepcopy(self._config)
    
    def save_to_file(self, config_file: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_file: Path to save the configuration
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            self.logger.info(f"Configuration saved to: {config_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def get_benchmark_config(self) -> BenchmarkConfig:
        """Get benchmark configuration as a dataclass."""
        config_dict = self.get('benchmark', {})
        return BenchmarkConfig(**config_dict)
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as a dataclass."""
        config_dict = self.get('data', {})
        return DataConfig(**config_dict)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as a dataclass."""
        config_dict = self.get('model', {})
        return ModelConfig(**config_dict)
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration as a dataclass."""
        config_dict = self.get('logging', {})
        return LoggingConfig(**config_dict)
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        logging_config = self.get_logging_config()
        
        # Set log level
        level = getattr(logging, logging_config.level.upper(), logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(logging_config.format)
        
        # Get root logger
        root_logger = logging.getLogger('scramblebench')
        root_logger.setLevel(level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if logging_config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if logging_config.file:
            log_path = Path(logging_config.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        self.logger.info("Logging configured")
    
    def create_subset(self, section: str) -> 'Config':
        """
        Create a new Config instance with only a specific section.
        
        Args:
            section: Section name to extract
            
        Returns:
            New Config instance with only the specified section
        """
        subset_config = self.get(section, {})
        return Config(config_dict={section: subset_config}, load_env=False)
    
    def merge(self, other: 'Config') -> 'Config':
        """
        Merge this configuration with another one.
        
        Args:
            other: Other Config instance to merge
            
        Returns:
            New Config instance with merged configuration
        """
        merged_config = self.to_dict()
        self._deep_update(merged_config, other.to_dict())
        
        return Config(config_dict=merged_config, load_env=False)
    
    def validate_section(self, section: str, schema: Dict[str, Type]) -> bool:
        """
        Validate a configuration section against a schema.
        
        Args:
            section: Section name to validate
            schema: Dictionary mapping keys to expected types
            
        Returns:
            True if validation passes, False otherwise
        """
        section_config = self.get(section, {})
        
        for key, expected_type in schema.items():
            if key not in section_config:
                self.logger.error(f"Missing required key in {section}: {key}")
                return False
            
            value = section_config[key]
            if not isinstance(value, expected_type):
                self.logger.error(
                    f"Invalid type for {section}.{key}: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )
                return False
        
        return True
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Config({self._config})"


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Path to configuration file
        **kwargs: Additional arguments for Config constructor
        
    Returns:
        Loaded Config instance
    """
    return Config(config_file=config_file, **kwargs)