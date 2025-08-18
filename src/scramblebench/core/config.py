"""
Unified Configuration System for ScrambleBench
==============================================

This module provides a comprehensive, unified configuration system that consolidates
all configuration needs across the ScrambleBench framework. It uses Pydantic for
validation while maintaining backward compatibility with existing code.

Design Principles:
    * Single source of truth for all configuration
    * Type safety with Pydantic validation
    * Environment variable support
    * Hierarchical configuration with sensible defaults
    * Easy serialization and deserialization
"""

from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import os
import yaml
import json
import logging
from pydantic import BaseModel, Field, validator
from enum import Enum


# Core Enumerations
class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class ModelType(str, Enum):
    """Types of supported models."""
    CHAT = "chat"
    COMPLETION = "completion"
    INSTRUCTION = "instruction"
    CODE = "code"


class TransformationType(str, Enum):
    """Types of transformations available."""
    LANGUAGE_TRANSLATION = "language_translation"
    PROPER_NOUN_SWAP = "proper_noun_swap"
    SYNONYM_REPLACEMENT = "synonym_replacement"
    PARAPHRASING = "paraphrasing"
    LONG_CONTEXT = "long_context"
    ALL = "all"


class EvaluationMode(str, Enum):
    """Evaluation modes."""
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    COMPREHENSIVE = "comprehensive"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Configuration Classes
class ModelConfig(BaseModel):
    """
    Unified model configuration supporting both evaluation and interface usage.
    
    This class consolidates model configuration from both the utils and evaluation
    modules, providing a single comprehensive configuration for all model interactions.
    """
    
    # Model identification
    name: str = Field(
        default="openai/gpt-3.5-turbo",
        description="Model name or identifier"
    )
    provider: ModelProvider = Field(
        default=ModelProvider.OPENROUTER,
        description="Model provider"
    )
    model_type: Optional[ModelType] = Field(
        default=ModelType.CHAT,
        description="Model type classification"
    )
    
    # Authentication
    api_key: Optional[str] = Field(
        default=None,
        description="API key (uses environment variable if None)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for repetition"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for repetition"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="List of stop sequences"
    )
    
    # Performance settings
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial retry delay in seconds"
    )
    rate_limit: float = Field(
        default=10.0,
        gt=0.0,
        description="Requests per second limit"
    )
    
    # Legacy compatibility fields
    vocab_size: int = Field(
        default=1000,
        gt=0,
        description="Vocabulary size (for language generation)"
    )
    
    class Config:
        use_enum_values = True
        validate_assignment = True
    
    @validator('api_key', pre=True, always=True)
    def resolve_api_key(cls, v, values):
        """Resolve API key from environment if not provided."""
        if v is None:
            provider = values.get('provider', ModelProvider.OPENROUTER)
            if provider == ModelProvider.OPENROUTER:
                v = os.getenv('OPENROUTER_API_KEY')
            elif provider == ModelProvider.OPENAI:
                v = os.getenv('OPENAI_API_KEY')
            elif provider == ModelProvider.ANTHROPIC:
                v = os.getenv('ANTHROPIC_API_KEY')
        return v


class DataConfig(BaseModel):
    """Configuration for data handling and storage."""
    
    data_dir: str = Field(
        default="data",
        description="Base data directory"
    )
    benchmarks_dir: str = Field(
        default="data/benchmarks",
        description="Benchmark datasets directory"
    )
    languages_dir: str = Field(
        default="data/languages",
        description="Constructed languages directory"
    )
    results_dir: str = Field(
        default="data/results",
        description="Results output directory"
    )
    cache_dir: str = Field(
        default="data/cache",
        description="Cache directory"
    )
    temp_dir: str = Field(
        default="/tmp/scramblebench",
        description="Temporary files directory"
    )
    
    # File format settings
    default_format: str = Field(
        default="json",
        description="Default file format"
    )
    supported_formats: List[str] = Field(
        default=["json", "jsonl", "csv", "parquet"],
        description="Supported file formats"
    )
    encoding: str = Field(
        default="utf-8",
        description="Default file encoding"
    )
    
    # Caching settings
    enable_caching: bool = Field(
        default=True,
        description="Enable data caching"
    )
    max_cache_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum cached items"
    )
    cache_ttl: int = Field(
        default=86400,
        gt=0,
        description="Cache TTL in seconds"
    )
    cache_compression: bool = Field(
        default=True,
        description="Enable cache compression"
    )
    
    # Loading settings
    batch_size: int = Field(
        default=1000,
        gt=0,
        description="Data loading batch size"
    )
    parallel_loading: bool = Field(
        default=True,
        description="Enable parallel data loading"
    )
    max_workers: int = Field(
        default=4,
        gt=0,
        description="Maximum worker threads"
    )
    chunk_size: int = Field(
        default=10000,
        gt=0,
        description="Data chunk size for processing"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""
    
    # Core settings
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    evaluation_mode: str = Field(
        default="exact_match",
        description="Evaluation mode"
    )
    evaluation_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Evaluation threshold"
    )
    
    # Preservation settings
    preserve_numbers: bool = Field(
        default=True,
        description="Preserve numbers in transformations"
    )
    preserve_proper_nouns: bool = Field(
        default=True,
        description="Preserve proper nouns in transformations"
    )
    preserve_structure: bool = Field(
        default=True,
        description="Preserve document structure"
    )
    preserve_entities: bool = Field(
        default=True,
        description="Preserve named entities"
    )
    
    # Prompt configuration
    prompt_template: str = Field(
        default="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        description="Default prompt template"
    )
    
    # Processing settings
    max_samples: Optional[int] = Field(
        default=None,
        description="Maximum samples per benchmark"
    )
    sample_strategy: str = Field(
        default="random",
        description="Sampling strategy"
    )
    
    # Performance settings
    max_concurrent_requests: int = Field(
        default=5,
        gt=0,
        description="Maximum concurrent requests"
    )
    save_interval: int = Field(
        default=50,
        gt=0,
        description="Save results every N samples"
    )


class TransformationConfig(BaseModel):
    """Configuration for text transformations."""
    
    enabled_types: List[TransformationType] = Field(
        default=[TransformationType.ALL],
        description="Types of transformations to apply"
    )
    
    # Language translation settings
    languages: List[str] = Field(
        default=["constructed_1", "constructed_2"],
        description="Languages to use for translation"
    )
    language_complexity: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Complexity of constructed languages"
    )
    
    # Proper noun swap settings
    proper_noun_strategy: str = Field(
        default="random",
        description="Strategy for proper noun replacement"
    )
    preserve_gender: bool = Field(
        default=True,
        description="Preserve gender in name replacements"
    )
    preserve_origin: bool = Field(
        default=False,
        description="Preserve cultural origin of names"
    )
    
    # Synonym replacement settings
    synonym_rate: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Proportion of words to replace with synonyms"
    )
    preserve_function_words: bool = Field(
        default=True,
        description="Preserve articles, prepositions, etc."
    )
    preserve_sentiment: bool = Field(
        default=True,
        description="Maintain text sentiment"
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for synonym selection"
    )
    
    # General settings
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    batch_size: int = Field(
        default=10,
        gt=0,
        description="Transformation batch size"
    )
    
    class Config:
        use_enum_values = True
    
    @validator('enabled_types', pre=True)
    def validate_transformation_types(cls, v):
        if isinstance(v, str):
            v = [v]
        return [TransformationType(t) if isinstance(t, str) else t for t in v]


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for logs"
    )
    
    # Output destinations
    console: bool = Field(
        default=True,
        description="Enable console logging"
    )
    file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    # File settings
    max_file_size: str = Field(
        default="10MB",
        description="Maximum log file size"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup log files"
    )
    
    # Logger-specific settings
    loggers: Dict[str, str] = Field(
        default_factory=lambda: {
            "scramblebench.core": "INFO",
            "scramblebench.translation": "INFO",
            "scramblebench.llm": "INFO",
            "scramblebench.evaluation": "INFO"
        },
        description="Logger-specific levels"
    )
    
    class Config:
        use_enum_values = True


class MetricsConfig(BaseModel):
    """Configuration for metrics calculation."""
    
    # Accuracy metrics
    calculate_exact_match: bool = Field(default=True)
    calculate_f1: bool = Field(default=True)
    calculate_bleu: bool = Field(default=False)
    calculate_rouge: bool = Field(default=False)
    
    # Robustness metrics
    calculate_degradation: bool = Field(default=True)
    degradation_threshold: float = Field(
        default=0.05, 
        description="Significant degradation threshold"
    )
    
    # Statistical tests
    significance_level: float = Field(
        default=0.05, 
        description="Statistical significance level"
    )
    multiple_comparisons_correction: str = Field(
        default="bonferroni", 
        description="Multiple comparisons correction method"
    )
    
    # Reporting
    confidence_intervals: bool = Field(default=True)
    effect_size: bool = Field(default=True)


class PlotConfig(BaseModel):
    """Configuration for plot generation."""
    
    # Output settings
    save_formats: List[str] = Field(
        default=["png", "pdf"], 
        description="Plot formats to save"
    )
    dpi: int = Field(default=300, description="Plot DPI")
    figsize: tuple = Field(default=(10, 6), description="Default figure size")
    
    # Style settings
    style: str = Field(default="seaborn-v0_8", description="Matplotlib style")
    color_palette: str = Field(default="Set2", description="Color palette")
    
    # Plot types
    generate_degradation_plots: bool = Field(default=True)
    generate_comparison_plots: bool = Field(default=True)
    generate_heatmaps: bool = Field(default=True)
    generate_distribution_plots: bool = Field(default=True)
    
    # Interactive plots
    generate_interactive: bool = Field(
        default=True, 
        description="Generate interactive Plotly plots"
    )


class ScrambleBenchConfig(BaseModel):
    """
    Main configuration class for ScrambleBench.
    
    This is the unified configuration that brings together all subsystem
    configurations into a single, coherent structure.
    """
    
    # Subsystem configurations
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration"
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data configuration"
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Benchmark configuration"
    )
    transformations: TransformationConfig = Field(
        default_factory=TransformationConfig,
        description="Transformation configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration"
    )
    plots: PlotConfig = Field(
        default_factory=PlotConfig,
        description="Plot configuration"
    )
    
    # Global settings
    version: str = Field(
        default="0.1.0",
        description="Configuration version"
    )
    created_at: Optional[str] = Field(
        default=None,
        description="Configuration creation timestamp"
    )
    
    class Config:
        validate_assignment = True
        extra = "forbid"  # Prevent additional fields
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(
                    self.dict(exclude_unset=True),
                    f,
                    default_flow_style=False,
                    indent=2
                )
            elif path.suffix.lower() == '.json':
                json.dump(
                    self.dict(exclude_unset=True),
                    f,
                    indent=2
                )
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> 'ScrambleBenchConfig':
        """Load configuration from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")
        
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "SCRAMBLEBENCH") -> 'ScrambleBenchConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Map environment variables to config paths
        env_mappings = {
            f'{prefix}_MODEL_NAME': ['model', 'name'],
            f'{prefix}_MODEL_PROVIDER': ['model', 'provider'],
            f'{prefix}_MODEL_TEMPERATURE': ['model', 'temperature'],
            f'{prefix}_MODEL_MAX_TOKENS': ['model', 'max_tokens'],
            f'{prefix}_MODEL_TIMEOUT': ['model', 'timeout'],
            f'{prefix}_DATA_DIR': ['data', 'data_dir'],
            f'{prefix}_RESULTS_DIR': ['data', 'results_dir'],
            f'{prefix}_LOG_LEVEL': ['logging', 'level'],
            f'{prefix}_RANDOM_SEED': ['benchmark', 'random_seed'],
            'OPENROUTER_API_KEY': ['model', 'api_key'],
            'OPENAI_API_KEY': ['model', 'api_key'],
            'ANTHROPIC_API_KEY': ['model', 'api_key'],
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the correct nested location
                target = config
                for key in path[:-1]:
                    target = getattr(target, key)
                
                # Set the value with appropriate type conversion
                current_value = getattr(target, path[-1])
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(target, path[-1], value)
        
        return config
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.logging
        
        # Set log level
        level = getattr(logging, log_config.level.value, logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.format,
            datefmt=log_config.date_format
        )
        
        # Get root logger
        root_logger = logging.getLogger('scramblebench')
        root_logger.setLevel(level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        if log_config.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_config.file:
            log_path = Path(log_config.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Configure specific loggers
        for logger_name, level_str in log_config.loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level_str, logging.INFO))
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.data.benchmarks_dir,
            self.data.languages_dir,
            self.data.results_dir,
            self.data.cache_dir,
            self.data.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Convenience functions for backward compatibility
def load_config(
    config_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> ScrambleBenchConfig:
    """
    Load configuration with backward compatibility.
    
    Args:
        config_file: Path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Loaded configuration instance
    """
    if config_file:
        config = ScrambleBenchConfig.load_from_file(config_file)
    else:
        config = ScrambleBenchConfig.from_env()
    
    # Apply any overrides
    if kwargs:
        config_dict = config.dict()
        _deep_update(config_dict, kwargs)
        config = ScrambleBenchConfig(**config_dict)
    
    return config


def _deep_update(target: dict, source: dict) -> None:
    """Deep update a dictionary."""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


# Backward compatibility aliases
Config = ScrambleBenchConfig