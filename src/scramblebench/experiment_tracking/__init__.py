"""
ScrambleBench Experiment Tracking System

A comprehensive experiment management system designed for academic research 
in language dependency analysis. Provides robust tracking, monitoring, and 
reproducibility features for large-scale AI model evaluation experiments.

Core Features:
- Experiment queue management and status tracking
- Real-time progress monitoring with ETA calculations  
- Resource utilization tracking (API costs, compute hours)
- Statistical significance testing and A/B test framework
- Academic reproducibility validation and metadata tracking
- Database integration with existing ScrambleBench schema
- Export capabilities for academic publication and replication

Usage:
    from scramblebench.experiment_tracking import ExperimentTracker
    
    tracker = ExperimentTracker(database_url="postgresql://...")
    experiment_id = tracker.create_experiment(
        name="Language Dependency Analysis",
        config=config,
        researcher="Dr. Smith"
    )
    tracker.run_experiment(experiment_id)
"""

from .core import ExperimentTracker, ExperimentStatus, ExperimentMetadata
from .queue import ExperimentQueue, QueuedExperiment, QueueStatus
from .monitor import ExperimentMonitor, ProgressTracker, ResourceMonitor
from .statistics import StatisticalAnalyzer, SignificanceTest, ABTestFramework
from .reproducibility import ReproducibilityValidator, EnvironmentSnapshot
from .database import DatabaseManager, ExperimentDatabase
from .export import AcademicExporter, ReplicationPackage, PublicationData
from .cli import ExperimentCLI

__version__ = "1.0.0"
__author__ = "ScrambleBench Research Team"

# Core classes for easy import
__all__ = [
    "ExperimentTracker",
    "ExperimentQueue", 
    "ExperimentMonitor",
    "StatisticalAnalyzer",
    "ReproducibilityValidator",
    "DatabaseManager",
    "AcademicExporter",
    "ExperimentCLI",
    
    # Status and data classes
    "ExperimentStatus",
    "ExperimentMetadata",
    "QueuedExperiment",
    "QueueStatus",
    "ProgressTracker",
    "ResourceMonitor",
    "SignificanceTest",
    "ABTestFramework",
    "EnvironmentSnapshot",
    "ReplicationPackage",
    "PublicationData"
]

# Module metadata for reproducibility tracking
MODULE_INFO = {
    "version": __version__,
    "api_version": "2024.1",
    "database_schema_version": "1.0.0",
    "compatible_scramblebench_versions": [">=1.0.0"],
    "python_requirements": [">=3.9"],
    "dependencies": {
        "sqlalchemy": ">=2.0.0",
        "psycopg2-binary": ">=2.9.0", 
        "scipy": ">=1.11.0",
        "pandas": ">=2.0.0",
        "numpy": ">=1.24.0",
        "matplotlib": ">=3.7.0",
        "seaborn": ">=0.12.0"
    }
}