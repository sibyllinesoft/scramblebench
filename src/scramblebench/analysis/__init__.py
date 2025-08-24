"""
Advanced Statistical Analysis Pipeline for ScrambleBench

This module implements the Step S8 statistical analysis framework specified in TODO.md,
providing academic-grade statistical modeling to infer scaling shapes without 
presupposing thresholds.

Key Features:
- GLMM (Generalized Linear Mixed Models) for hierarchical data
- GAM (Generalized Additive Models) for non-parametric scaling discovery
- Changepoint detection with model comparison
- Contamination vs brittleness separation analysis
- Bootstrap confidence intervals and multiple testing correction
- Academic publication-ready outputs
"""

from .statistical_models import (
    ScalingAnalyzer,
    GLMMAnalyzer, 
    GAMAnalyzer,
    ChangepointAnalyzer,
    ContaminationAnalyzer
)

from .model_comparison import (
    ModelComparison,
    ModelSelectionCriteria,
    AICWeights
)

from .academic_export import (
    AcademicExporter,
    PreregReport,
    LaTeXTableFormatter,
    CSVExporter
)

from .bootstrap_inference import (
    BootstrapAnalyzer,
    ConfidenceIntervals,
    PermutationTests
)

__all__ = [
    'ScalingAnalyzer',
    'GLMMAnalyzer',
    'GAMAnalyzer', 
    'ChangepointAnalyzer',
    'ContaminationAnalyzer',
    'ModelComparison',
    'ModelSelectionCriteria',
    'AICWeights',
    'AcademicExporter',
    'PreregReport',
    'LaTeXTableFormatter',
    'CSVExporter',
    'BootstrapAnalyzer',
    'ConfidenceIntervals',
    'PermutationTests'
]