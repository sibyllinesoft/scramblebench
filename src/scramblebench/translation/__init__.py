"""
Translation benchmark module for ScrambleBench.

This module implements benchmarks using constructed languages to test
model performance on problems that have been transformed to avoid
training data contamination.
"""

from scramblebench.translation.benchmark import TranslationBenchmark
from scramblebench.translation.language_generator import LanguageGenerator
from scramblebench.translation.translator import ProblemTranslator

__all__ = ["TranslationBenchmark", "LanguageGenerator", "ProblemTranslator"]