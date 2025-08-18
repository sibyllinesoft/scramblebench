"""
Configuration classes for the evaluation pipeline (DEPRECATED).

.. deprecated:: 0.2.0
    This module is deprecated. Use :mod:`scramblebench.core.config` instead.
    The new unified configuration system consolidates all configuration classes
    and provides better type safety and validation.

Migration Guide:
    Replace evaluation-specific configs with the unified system:
    
    .. code-block:: python
    
        # Old (deprecated)
        from scramblebench.evaluation.config import EvaluationConfig, ModelConfig
        
        # New (recommended)
        from scramblebench.core.config import ScrambleBenchConfig
        
        # Access evaluation-specific configs through the main config
        config = ScrambleBenchConfig()
        model_config = config.model
        transformation_config = config.transformations
        metrics_config = config.metrics
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from enum import Enum


class TransformationType(str, Enum):
    """Types of transformations available."""
    LANGUAGE_TRANSLATION = "language_translation"
    PROPER_NOUN_SWAP = "proper_noun_swap"
    SYNONYM_REPLACEMENT = "synonym_replacement"
    PARAPHRASING = "paraphrasing"
    LONG_CONTEXT = "long_context"
    ALL = "all"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class EvaluationMode(str, Enum):
    """Evaluation modes."""
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    COMPREHENSIVE = "comprehensive"


class ModelConfig(BaseModel):
    """Configuration for a model to evaluate."""
    name: str = Field(..., description="Model name or identifier")
    provider: ModelProvider = Field(..., description="Model provider")
    api_key: Optional[str] = Field(None, description="API key (uses env if None)")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Maximum output tokens")
    timeout: int = Field(60, description="Request timeout in seconds")
    rate_limit: float = Field(1.0, description="Requests per second")
    
    class Config:
        use_enum_values = True


class TransformationConfig(BaseModel):
    """Configuration for transformations."""
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
    
    # General settings
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    batch_size: int = Field(default=10, description="Transformation batch size")
    
    class Config:
        use_enum_values = True
    
    @validator('enabled_types', pre=True)
    def validate_transformation_types(cls, v):
        if isinstance(v, str):
            v = [v]
        return [TransformationType(t) if isinstance(t, str) else t for t in v]


class EvaluationConfig(BaseModel):
    """Main configuration for evaluation pipeline."""
    
    # Experiment settings
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Experiment description")
    mode: EvaluationMode = Field(
        default=EvaluationMode.COMPREHENSIVE,
        description="Evaluation mode"
    )
    
    # Input/Output paths
    benchmark_paths: List[Union[str, Path]] = Field(
        ..., description="Paths to benchmark datasets"
    )
    output_dir: Union[str, Path] = Field(
        default="results",
        description="Directory to save results"
    )
    
    # Models to evaluate
    models: List[ModelConfig] = Field(..., description="Models to evaluate")
    
    # Transformations
    transformations: TransformationConfig = Field(
        default_factory=TransformationConfig,
        description="Transformation settings"
    )
    
    # Evaluation settings
    max_samples: Optional[int] = Field(
        None, description="Maximum samples per benchmark (None for all)"
    )
    sample_seed: Optional[int] = Field(
        None, description="Seed for sampling benchmarks"
    )
    
    # Performance settings
    max_concurrent_requests: int = Field(
        default=5, description="Maximum concurrent API requests"
    )
    save_interval: int = Field(
        default=50, description="Save results every N samples"
    )
    
    # Analysis settings
    generate_plots: bool = Field(default=True, description="Generate analysis plots")
    calculate_significance: bool = Field(
        default=True, description="Calculate statistical significance"
    )
    
    class Config:
        use_enum_values = True
    
    @validator('benchmark_paths', pre=True)
    def validate_benchmark_paths(cls, v):
        if isinstance(v, (str, Path)):
            v = [v]
        return [Path(p) for p in v]
    
    @validator('output_dir', pre=True)
    def validate_output_dir(cls, v):
        return Path(v)
    
    def get_output_dir(self) -> Path:
        """Get the output directory as a Path object."""
        return self.output_dir
    
    def get_experiment_dir(self) -> Path:
        """Get the full experiment directory path."""
        return self.get_output_dir() / self.experiment_name
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


class MetricsConfig(BaseModel):
    """Configuration for metrics calculation."""
    
    # Accuracy metrics
    calculate_exact_match: bool = Field(default=True)
    calculate_f1: bool = Field(default=True)
    calculate_bleu: bool = Field(default=False)
    calculate_rouge: bool = Field(default=False)
    
    # Robustness metrics
    calculate_degradation: bool = Field(default=True)
    degradation_threshold: float = Field(default=0.05, description="Significant degradation threshold")
    
    # Statistical tests
    significance_level: float = Field(default=0.05, description="Statistical significance level")
    multiple_comparisons_correction: str = Field(
        default="bonferroni", description="Multiple comparisons correction method"
    )
    
    # Reporting
    confidence_intervals: bool = Field(default=True)
    effect_size: bool = Field(default=True)


class PlotConfig(BaseModel):
    """Configuration for plot generation."""
    
    # Output settings
    save_formats: List[str] = Field(default=["png", "pdf"], description="Plot formats to save")
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
    generate_interactive: bool = Field(default=True, description="Generate interactive Plotly plots")