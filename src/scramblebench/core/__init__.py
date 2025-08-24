"""
ScrambleBench Core Module.

This module contains the core functionality for the ScrambleBench evaluation system,
including configuration management, database operations, and evaluation orchestration.
"""

# Legacy compatibility
from scramblebench.core.benchmark import BaseBenchmark
from scramblebench.core.evaluator import Evaluator
from scramblebench.core.reporter import Reporter

# New unified system components
try:
    from .unified_config import ScrambleBenchConfig
    from .database import ScrambleBenchDatabase
    from .runner import EvaluationRunner
    from .adapters import create_adapter, BaseModelAdapter, OllamaAdapter, OpenAIAdapter, AnthropicAdapter
except ImportError:
    # Components not yet implemented
    ScrambleBenchConfig = None
    ScrambleBenchDatabase = None
    EvaluationRunner = None
    create_adapter = None
    BaseModelAdapter = None
    OllamaAdapter = None
    OpenAIAdapter = None
    AnthropicAdapter = None

__all__ = [
    # Legacy components
    "BaseBenchmark", "Evaluator", "Reporter",
    # New unified components
    "ScrambleBenchConfig", "ScrambleBenchDatabase", "EvaluationRunner",
    "create_adapter", "BaseModelAdapter", "OllamaAdapter", "OpenAIAdapter", "AnthropicAdapter"
]