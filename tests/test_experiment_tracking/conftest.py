"""
Pytest fixtures and configuration for experiment tracking tests.

Provides fixtures for testing experiment management, queue operations,
monitoring, statistics, and database integration.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from scramblebench.experiment_tracking.core import (
    ExperimentTracker, ExperimentMetadata, ExperimentStatus
)
from scramblebench.experiment_tracking.queue import (
    ExperimentQueue, QueuedExperiment, ResourceRequirements, QueueStatus
)
from scramblebench.experiment_tracking.monitor import (
    ExperimentMonitor, ProgressTracker, ProgressSnapshot
)
from scramblebench.experiment_tracking.statistics import (
    StatisticalAnalyzer, SignificanceTest, ABTestResult, LanguageDependencyAnalysis
)
from scramblebench.experiment_tracking.reproducibility import ReproducibilityValidator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_experiment_dir():
    """Fixture providing temporary directory for experiment data."""
    temp_dir = tempfile.mkdtemp(prefix="test_experiments_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_database_url():
    """Fixture providing mock database URL for testing."""
    return "postgresql://test:test@localhost:5432/test_experiments"


@pytest.fixture
def mock_db_manager():
    """Fixture providing mock database manager for testing."""
    mock = AsyncMock()
    
    # Mock common database operations
    mock.create_experiment = AsyncMock(return_value=True)
    mock.get_experiment_metadata = AsyncMock(return_value=None)
    mock.get_experiment_config = AsyncMock(return_value=None)
    mock.update_experiment_status = AsyncMock(return_value=True)
    mock.update_experiment_metadata = AsyncMock(return_value=True)
    mock.save_experiment_results = AsyncMock(return_value=True)
    mock.save_experiment_error = AsyncMock(return_value=True)
    mock.list_experiments = AsyncMock(return_value=[])
    
    return mock


@pytest.fixture
def sample_experiment_config():
    """Fixture providing sample experiment configuration."""
    return {
        "models": [
            {"provider": "openrouter", "model": "gpt-4"},
            {"provider": "ollama", "model": "llama2"}
        ],
        "benchmark_paths": ["benchmarks/mmlu.json"],
        "max_samples": 100,
        "sample_seed": 42,
        "transforms": ["original", "scrambled"],
        "output_dir": "test_outputs"
    }


@pytest.fixture
def sample_experiment_metadata():
    """Fixture providing sample experiment metadata."""
    return ExperimentMetadata(
        experiment_id="exp-test-123",
        name="Test Experiment",
        description="A test experiment for validation",
        research_question="Does scrambling affect model performance?",
        hypothesis="Models will show decreased performance on scrambled text",
        researcher_name="Test Researcher",
        institution="Test University",
        git_commit_hash="abc123def456",
        git_branch="main",
        environment_snapshot={"python": "3.9.0", "torch": "1.9.0"},
        random_seed=42,
        config_hash="config-hash-123",
        created_at=datetime.now(),
        started_at=None,
        completed_at=None,
        status=ExperimentStatus.PLANNED,
        progress=0.0,
        current_stage="initialized"
    )


@pytest.fixture
def sample_queued_experiment():
    """Fixture providing sample queued experiment."""
    return QueuedExperiment(
        experiment_id="queue-exp-123",
        priority=1,
        depends_on=[],
        resource_requirements=ResourceRequirements(
            api_calls_per_hour=1000,
            memory_gb=4.0,
            storage_gb=2.0,
            estimated_duration_hours=1.0
        ),
        estimated_duration=timedelta(hours=1),
        queued_at=datetime.now(),
        status=QueueStatus.PENDING
    )


@pytest.fixture
def sample_evaluation_results():
    """Fixture providing sample evaluation results."""
    mock_results = Mock()
    mock_results.results = []
    
    # Generate sample evaluation data
    models = ["gpt-4", "claude-3", "llama-2"]
    transforms = ["original", "scrambled"]
    
    for model in models:
        for transform in transforms:
            for i in range(10):
                result = Mock()
                result.model_id = model
                result.transform = transform
                result.item_id = f"item-{i}"
                result.is_correct = i % 3 == 0  # 33% accuracy
                result.latency_ms = 100 + i * 10
                result.cost_usd = 0.001 * (i + 1)
                result.timestamp = datetime.now() + timedelta(seconds=i)
                mock_results.results.append(result)
    
    mock_results.total_evaluations = len(mock_results.results)
    mock_results.completed_evaluations = len(mock_results.results)
    mock_results.accuracy = sum(1 for r in mock_results.results if r.is_correct) / len(mock_results.results)
    mock_results.avg_latency = sum(r.latency_ms for r in mock_results.results) / len(mock_results.results)
    mock_results.total_cost = sum(r.cost_usd for r in mock_results.results)
    
    return mock_results


@pytest.fixture
def mock_evaluation_runner():
    """Fixture providing mock evaluation runner."""
    runner = AsyncMock()
    
    # Mock runner status
    runner.get_status = Mock(return_value={
        "current_stage": "running_evaluations",
        "completed": 50,
        "total": 100,
        "progress": 0.5
    })
    
    # Mock evaluation execution
    async def run_evaluation():
        await asyncio.sleep(0.1)  # Simulate work
        return Mock(
            results=[Mock(accuracy=0.85, latency=150)],
            accuracy=0.85,
            total_evaluations=100
        )
    
    runner.run_evaluation = run_evaluation
    
    return runner


@pytest.fixture
def mock_progress_tracker():
    """Fixture providing mock progress tracker."""
    tracker = AsyncMock()
    
    tracker.initialize = AsyncMock()
    tracker.update_progress = AsyncMock()
    tracker.get_current_progress = AsyncMock(return_value={
        "stage": "running_evaluations",
        "progress": 0.75,
        "eta": "5 minutes",
        "details": {"completed": 75, "total": 100}
    })
    
    return tracker


@pytest.fixture
def mock_statistical_analyzer():
    """Fixture providing mock statistical analyzer."""
    analyzer = AsyncMock()
    
    analyzer.compute_performance_metrics = AsyncMock()
    analyzer.analyze_thresholds = AsyncMock()
    analyzer.run_significance_tests = AsyncMock()
    
    # Mock statistical results
    analyzer.get_significance_results = AsyncMock(return_value=[
        SignificanceTest(
            test_name="t_test",
            p_value=0.001,
            statistic=3.45,
            significant=True,
            effect_size=0.75
        )
    ])
    
    return analyzer


@pytest.fixture
def mock_reproducibility_validator():
    """Fixture providing mock reproducibility validator."""
    validator = AsyncMock()
    
    validator.capture_environment = AsyncMock(return_value={
        "python_version": "3.9.0",
        "pip_packages": {"torch": "1.9.0", "numpy": "1.21.0"},
        "system_info": {"platform": "linux", "cpu_count": 8}
    })
    
    validator.get_git_info = AsyncMock(return_value={
        "commit_hash": "abc123def456789",
        "branch": "main",
        "is_dirty": False,
        "remote_url": "https://github.com/test/repo.git"
    })
    
    validator.validate_reproducibility = AsyncMock(return_value=True)
    
    return validator


@pytest.fixture
def experiment_queue():
    """Fixture providing ExperimentQueue instance."""
    return ExperimentQueue(
        max_concurrent=3,
        resource_limits={
            'total_api_calls_per_hour': 5000,
            'total_memory_gb': 16.0,
            'total_storage_gb': 50.0,
            'total_cost_per_day': 500.0
        }
    )


@pytest.fixture
def experiment_monitor(mock_db_manager):
    """Fixture providing ExperimentMonitor instance."""
    return ExperimentMonitor(mock_db_manager)


@pytest.fixture
def statistical_analyzer(mock_db_manager):
    """Fixture providing StatisticalAnalyzer instance."""
    return StatisticalAnalyzer(mock_db_manager)


@pytest.fixture
def progress_tracker(mock_db_manager):
    """Fixture providing ProgressTracker instance."""
    return ProgressTracker(
        experiment_id="test-exp-progress",
        db_manager=mock_db_manager
    )


@pytest.fixture
async def experiment_tracker(
    temp_experiment_dir, 
    mock_database_url,
    mock_db_manager,
    mock_reproducibility_validator,
    mock_statistical_analyzer
):
    """Fixture providing ExperimentTracker instance with mocked dependencies."""
    with patch('scramblebench.experiment_tracking.core.DatabaseManager', return_value=mock_db_manager), \
         patch('scramblebench.experiment_tracking.core.ReproducibilityValidator', return_value=mock_reproducibility_validator), \
         patch('scramblebench.experiment_tracking.core.StatisticalAnalyzer', return_value=mock_statistical_analyzer):
        
        tracker = ExperimentTracker(
            database_url=mock_database_url,
            data_dir=temp_experiment_dir,
            max_concurrent_experiments=2
        )
        
        yield tracker


@pytest.fixture
def complex_dependency_graph():
    """Fixture providing complex experiment dependency graph for testing."""
    experiments = [
        {"id": "exp-a", "depends_on": [], "priority": 3},
        {"id": "exp-b", "depends_on": ["exp-a"], "priority": 2},
        {"id": "exp-c", "depends_on": ["exp-a"], "priority": 1},
        {"id": "exp-d", "depends_on": ["exp-b", "exp-c"], "priority": 1},
        {"id": "exp-e", "depends_on": [], "priority": 2},
        {"id": "exp-f", "depends_on": ["exp-e"], "priority": 1}
    ]
    
    return experiments


@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing scenarios."""
    # Generate large dataset for performance testing
    num_experiments = 100
    experiments = []
    
    for i in range(num_experiments):
        exp = QueuedExperiment(
            experiment_id=f"perf-exp-{i}",
            priority=i % 5 + 1,  # Priorities 1-5
            depends_on=[f"perf-exp-{j}" for j in range(max(0, i-2), i)],  # Dependency chain
            resource_requirements=ResourceRequirements(
                api_calls_per_hour=100 * (i % 10 + 1),
                memory_gb=1.0 + (i % 8),
                storage_gb=0.5 + (i % 4),
                estimated_duration_hours=0.5 + (i % 6) * 0.5
            ),
            estimated_duration=timedelta(minutes=30 + i % 120),
            queued_at=datetime.now() + timedelta(seconds=i)
        )
        experiments.append(exp)
    
    return experiments


@pytest.fixture
def mock_experiment_results_data():
    """Fixture providing mock experiment results data for statistical analysis."""
    np.random.seed(42)  # For reproducible test data
    
    # Generate results for 3 models across 2 conditions
    models = ["model-a", "model-b", "model-c"]
    conditions = ["original", "scrambled"]
    
    results_data = []
    
    for model in models:
        for condition in conditions:
            # Generate performance scores with some realistic patterns
            base_performance = 0.8 if model == "model-a" else 0.75 if model == "model-b" else 0.7
            condition_penalty = 0.15 if condition == "scrambled" else 0.0
            
            scores = np.random.normal(
                loc=base_performance - condition_penalty,
                scale=0.1,
                size=50
            )
            scores = np.clip(scores, 0.0, 1.0)  # Ensure scores are in valid range
            
            for i, score in enumerate(scores):
                results_data.append({
                    "experiment_id": f"test-exp-{model}-{condition}",
                    "model": model,
                    "condition": condition,
                    "score": score,
                    "item_id": f"item-{i}",
                    "latency_ms": np.random.normal(200, 50),
                    "cost_usd": np.random.normal(0.01, 0.002)
                })
    
    return pd.DataFrame(results_data)


@pytest.fixture
def mock_language_dependency_data():
    """Fixture providing mock language dependency analysis data."""
    # Simulate language dependency analysis results
    models = ["gpt-4", "claude-3", "llama-2"]
    scramble_levels = list(range(8))  # 0-7 scrambling levels
    
    data = []
    for model in models:
        for level in scramble_levels:
            # Simulate decreasing performance with increased scrambling
            base_performance = 0.9 if model == "gpt-4" else 0.85 if model == "claude-3" else 0.8
            degradation = level * 0.08  # Performance degrades with scrambling
            performance = max(0.2, base_performance - degradation + np.random.normal(0, 0.02))
            
            data.append({
                "model": model,
                "scramble_level": level,
                "performance": performance,
                "threshold_detected": level > 2 and performance < base_performance * 0.8,
                "language_dependency_score": min(1.0, level * 0.125)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def resource_constraint_scenarios():
    """Fixture providing various resource constraint scenarios for testing."""
    scenarios = [
        {
            "name": "memory_constrained",
            "limits": {
                "total_api_calls_per_hour": 10000,
                "total_memory_gb": 4.0,  # Very limited memory
                "total_storage_gb": 100.0,
                "total_cost_per_day": 1000.0
            },
            "experiments": [
                {"memory_gb": 2.0, "should_fit": True},
                {"memory_gb": 3.0, "should_fit": False},  # Would exceed limit
                {"memory_gb": 1.0, "should_fit": True}
            ]
        },
        {
            "name": "api_rate_limited",
            "limits": {
                "total_api_calls_per_hour": 1000,  # Very limited API calls
                "total_memory_gb": 32.0,
                "total_storage_gb": 100.0,
                "total_cost_per_day": 1000.0
            },
            "experiments": [
                {"api_calls_per_hour": 500, "should_fit": True},
                {"api_calls_per_hour": 800, "should_fit": False},  # Would exceed limit
                {"api_calls_per_hour": 200, "should_fit": True}
            ]
        },
        {
            "name": "cost_constrained",
            "limits": {
                "total_api_calls_per_hour": 10000,
                "total_memory_gb": 32.0,
                "total_storage_gb": 100.0,
                "total_cost_per_day": 50.0  # Very limited budget
            },
            "experiments": [
                {"cost_limit": 25.0, "should_fit": True},
                {"cost_limit": 40.0, "should_fit": False},  # Would exceed limit
                {"cost_limit": 10.0, "should_fit": True}
            ]
        }
    ]
    
    return scenarios


@pytest.fixture
def error_simulation_helpers():
    """Fixture providing helpers for simulating various error conditions."""
    class ErrorSimulator:
        @staticmethod
        def network_error():
            from sqlalchemy.exc import OperationalError
            return OperationalError("Network connection failed", None, None)
        
        @staticmethod
        def database_error():
            from sqlalchemy.exc import IntegrityError
            return IntegrityError("Constraint violation", None, None)
        
        @staticmethod
        def timeout_error():
            import asyncio
            return asyncio.TimeoutError("Operation timed out")
        
        @staticmethod
        def resource_exhaustion():
            return RuntimeError("Insufficient resources available")
        
        @staticmethod
        def configuration_error():
            return ValueError("Invalid configuration provided")
        
        @staticmethod
        def permission_error():
            return PermissionError("Access denied to resource")
    
    return ErrorSimulator()


@pytest.fixture
def mock_system_resources():
    """Fixture providing mock system resource monitoring."""
    class MockSystemResources:
        def __init__(self):
            self.cpu_usage = 0.5
            self.memory_usage = 0.6
            self.disk_usage = 0.3
            self.network_io = {"bytes_sent": 1000000, "bytes_recv": 2000000}
        
        def update_usage(self, cpu=None, memory=None, disk=None):
            if cpu is not None:
                self.cpu_usage = cpu
            if memory is not None:
                self.memory_usage = memory
            if disk is not None:
                self.disk_usage = disk
        
        def get_usage(self):
            return {
                "cpu_percent": self.cpu_usage * 100,
                "memory_percent": self.memory_usage * 100,
                "disk_percent": self.disk_usage * 100,
                "network_io": self.network_io
            }
    
    return MockSystemResources()


@pytest.fixture
def experiment_lifecycle_validator():
    """Fixture providing experiment lifecycle validation utilities."""
    class LifecycleValidator:
        def __init__(self):
            self.valid_transitions = {
                ExperimentStatus.PLANNED: [ExperimentStatus.QUEUED, ExperimentStatus.CANCELLED],
                ExperimentStatus.QUEUED: [ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED],
                ExperimentStatus.RUNNING: [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.PAUSED, ExperimentStatus.CANCELLED],
                ExperimentStatus.PAUSED: [ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED],
                ExperimentStatus.COMPLETED: [],  # Terminal state
                ExperimentStatus.FAILED: [],     # Terminal state
                ExperimentStatus.CANCELLED: []   # Terminal state
            }
        
        def is_valid_transition(self, from_status: ExperimentStatus, to_status: ExperimentStatus) -> bool:
            return to_status in self.valid_transitions.get(from_status, [])
        
        def get_valid_next_states(self, current_status: ExperimentStatus) -> List[ExperimentStatus]:
            return self.valid_transitions.get(current_status, [])
        
        def is_terminal_state(self, status: ExperimentStatus) -> bool:
            return len(self.valid_transitions.get(status, [])) == 0
    
    return LifecycleValidator()


# Async test helpers

@pytest.fixture
def async_test_timeout():
    """Fixture providing timeout for async tests."""
    return 30.0  # 30 seconds timeout for async operations


@pytest.fixture
def mock_async_context_manager():
    """Fixture providing mock async context manager for testing."""
    class MockAsyncContextManager:
        def __init__(self, return_value=None):
            self.return_value = return_value
            self.entered = False
            self.exited = False
            self.exception = None
        
        async def __aenter__(self):
            self.entered = True
            return self.return_value
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            self.exception = exc_val
            return False
    
    return MockAsyncContextManager