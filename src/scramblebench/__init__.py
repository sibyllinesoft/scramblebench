"""
ScrambleBench: LLM Benchmarking Toolkit

A comprehensive toolkit for evaluating Large Language Models through:
- Translation benchmarks using constructed languages
- Long context benchmarks with document transformation
- Standardized evaluation and reporting framework

Example usage:
    >>> from scramblebench import TranslationBenchmark, LongContextBenchmark
    >>> benchmark = TranslationBenchmark()
    >>> results = benchmark.run()
"""

__version__ = "0.1.0"
__author__ = "Nathan Rice"
__email__ = "nathan.alexander.rice@gmail.com"

from scramblebench.core.benchmark import BaseBenchmark
from scramblebench.core.evaluator import Evaluator
from scramblebench.core.reporter import Reporter
from scramblebench.translation.benchmark import TranslationBenchmark
from scramblebench.longcontext.benchmark import LongContextBenchmark
from scramblebench.llm.model_interface import ModelInterface
from scramblebench.utils.config import Config

__all__ = [
    "BaseBenchmark",
    "Evaluator", 
    "Reporter",
    "TranslationBenchmark",
    "LongContextBenchmark",
    "ModelInterface",
    "Config",
]