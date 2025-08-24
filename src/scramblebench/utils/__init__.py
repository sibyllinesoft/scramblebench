"""
Utilities module for ScrambleBench.

This module provides configuration management, data loading,
and other utility functions used throughout the benchmarking system.
"""

# Backward compatibility - Config is deprecated, use ScrambleBenchConfig instead
from ..core.unified_config import ScrambleBenchConfig as Config
from scramblebench.utils.data_loader import DataLoader

__all__ = ["Config", "DataLoader"]