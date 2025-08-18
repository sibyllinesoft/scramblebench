"""
Tests for core benchmark functionality.

This module provides comprehensive tests for the base benchmark class
and its implementations, covering initialization, execution, result handling,
and error conditions.
"""

import pytest
import logging
import time
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from dataclasses import asdict

from scramblebench.core.benchmark import BaseBenchmark, BenchmarkResult
from scramblebench.utils.config import Config


class MockBenchmark(BaseBenchmark):
    """Mock benchmark implementation for testing."""
    
    def __init__(self, name: str = "mock_benchmark", **kwargs):
        super().__init__(name, **kwargs)
        self.data = []
        self.prepared = False
        
    def prepare_data(self) -> None:
        """Mock data preparation."""
        self.data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
            {"question": "What is 5+5?", "answer": "10"}
        ]
        self.prepared = True
        
    def run_single_evaluation(self, model: Any, data_item: Any) -> Dict[str, Any]:
        """Mock single evaluation."""
        # Simulate model prediction
        predicted = model.predict(data_item["question"])
        correct = predicted == data_item["answer"]
        
        return {
            "predicted": predicted,
            "expected": data_item["answer"],
            "correct": correct,
            "score": 1.0 if correct else 0.0
        }
        
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock metrics computation."""
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "score": accuracy,
            "accuracy": accuracy,
            "total_samples": total,
            "correct_samples": correct
        }
        
    def get_evaluation_data(self, num_samples: int = None) -> List[Any]:
        """Mock evaluation data getter."""
        if not self.prepared:
            self.prepare_data()
            
        data = self.data.copy()
        if num_samples is not None:
            data = data[:num_samples]
        return data


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name: str = "mock_model", accuracy: float = 1.0):
        self.name = name
        self.accuracy = accuracy
        self.call_count = 0
        
    def predict(self, question: str) -> str:
        """Mock prediction method."""
        self.call_count += 1
        
        # Simple mock logic - perfect accuracy returns correct answers
        if self.accuracy >= 1.0:
            if "2+2" in question:
                return "4"
            elif "3+3" in question:
                return "6"
            elif "5+5" in question:
                return "10"
        
        # Imperfect accuracy returns wrong answers sometimes
        import random
        if random.random() < self.accuracy:
            if "2+2" in question:
                return "4"
            elif "3+3" in question:
                return "6"
            elif "5+5" in question:
                return "10"
        
        return "wrong"


# Fixtures
@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return Config({"results_dir": "test_results", "random_seed": 42})

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def benchmark(mock_config, mock_logger):
    """Create a mock benchmark instance."""
    return MockBenchmark(config=mock_config, logger=mock_logger)

@pytest.fixture
def perfect_model():
    """Create a perfect accuracy model."""
    return MockModel(accuracy=1.0)

@pytest.fixture
def imperfect_model():
    """Create an imperfect accuracy model."""
    return MockModel(accuracy=0.5)

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            score=0.85,
            metrics={"accuracy": 0.85, "precision": 0.90},
            metadata={"samples": 100},
            duration=45.2,
            timestamp=1234567890
        )
        
        assert result.benchmark_name == "test_benchmark"
        assert result.model_name == "test_model"
        assert result.score == 0.85
        assert result.metrics["accuracy"] == 0.85
        assert result.metadata["samples"] == 100
        assert result.duration == 45.2
        assert result.timestamp == 1234567890
        
    def test_benchmark_result_serialization(self):
        """Test that BenchmarkResult can be serialized."""
        result = BenchmarkResult(
            benchmark_name="test",
            model_name="model",
            score=0.75,
            metrics={"test": 1},
            metadata={"test": 2},
            duration=10.0,
            timestamp=1000.0
        )
        
        # Should be able to convert to dict
        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["benchmark_name"] == "test"
        assert result_dict["score"] == 0.75


class TestBaseBenchmarkInitialization:
    """Test benchmark initialization and setup."""
    
    def test_basic_initialization(self):
        """Test basic benchmark initialization."""
        benchmark = MockBenchmark("test_benchmark")
        
        assert benchmark.name == "test_benchmark"
        assert benchmark.config is not None
        assert benchmark.logger is not None
        assert len(benchmark.results) == 0
        
    def test_initialization_with_config(self, mock_config):
        """Test initialization with provided config."""
        benchmark = MockBenchmark("test", config=mock_config)
        
        assert benchmark.config is mock_config
        assert benchmark.config.get("random_seed") == 42
        
    def test_initialization_with_logger(self, mock_logger):
        """Test initialization with provided logger."""
        benchmark = MockBenchmark("test", logger=mock_logger)
        
        assert benchmark.logger is mock_logger
        
    def test_initialization_creates_default_config(self):
        """Test that default config is created when none provided."""
        benchmark = MockBenchmark("test")
        
        assert isinstance(benchmark.config, Config)
        
    def test_initialization_creates_default_logger(self):
        """Test that default logger is created when none provided."""
        benchmark = MockBenchmark("test")
        
        assert isinstance(benchmark.logger, logging.Logger)
        assert "scramblebench.test" in benchmark.logger.name
        
    def test_results_property_returns_copy(self, benchmark):
        """Test that results property returns a copy."""
        # Add a mock result
        mock_result = BenchmarkResult(
            benchmark_name="test",
            model_name="model",
            score=0.5,
            metrics={},
            metadata={},
            duration=1.0,
            timestamp=1000.0
        )
        benchmark._results.append(mock_result)
        
        results_copy = benchmark.results
        assert len(results_copy) == 1
        
        # Modifying copy shouldn't affect original
        results_copy.append(mock_result)
        assert len(benchmark.results) == 1


class TestBaseBenchmarkDataHandling:
    """Test benchmark data preparation and handling."""
    
    def test_prepare_data_is_called(self, benchmark, perfect_model):
        """Test that prepare_data is called during run."""
        assert not benchmark.prepared
        
        benchmark.run(perfect_model)
        
        assert benchmark.prepared
        
    def test_get_evaluation_data_returns_all_data(self, benchmark):
        """Test getting all evaluation data."""
        benchmark.prepare_data()
        data = benchmark.get_evaluation_data()
        
        assert len(data) == 3
        assert all("question" in item for item in data)
        
    def test_get_evaluation_data_limits_samples(self, benchmark):
        """Test limiting number of samples."""
        benchmark.prepare_data()
        data = benchmark.get_evaluation_data(num_samples=2)
        
        assert len(data) == 2
        
    def test_get_evaluation_data_handles_zero_samples(self, benchmark):
        """Test handling zero samples request."""
        benchmark.prepare_data()
        data = benchmark.get_evaluation_data(num_samples=0)
        
        assert len(data) == 0
        
    def test_get_evaluation_data_handles_oversized_request(self, benchmark):
        """Test handling request for more samples than available."""
        benchmark.prepare_data()
        data = benchmark.get_evaluation_data(num_samples=100)
        
        # Should return all available data
        assert len(data) == 3


class TestBaseBenchmarkExecution:
    """Test benchmark execution and evaluation."""
    
    def test_successful_run_perfect_model(self, benchmark, perfect_model):
        """Test successful benchmark run with perfect model."""
        result = benchmark.run(perfect_model)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "mock_benchmark"
        assert result.model_name == "mock_model"
        assert result.score == 1.0  # Perfect accuracy
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["total_samples"] == 3
        assert result.metrics["correct_samples"] == 3
        assert result.duration > 0
        assert result.timestamp > 0
        
    def test_successful_run_imperfect_model(self, benchmark):
        """Test benchmark run with imperfect model."""
        # Create a model that always returns wrong answers
        wrong_model = MockModel(accuracy=0.0)
        wrong_model.predict = lambda x: "wrong"
        
        result = benchmark.run(wrong_model)
        
        assert result.score == 0.0
        assert result.metrics["accuracy"] == 0.0
        assert result.metrics["correct_samples"] == 0
        
    def test_run_calls_all_required_methods(self, benchmark, perfect_model):
        """Test that run calls all required methods in order."""
        with patch.object(benchmark, 'prepare_data') as mock_prepare, \
             patch.object(benchmark, 'get_evaluation_data') as mock_get_data, \
             patch.object(benchmark, 'run_single_evaluation') as mock_single, \
             patch.object(benchmark, 'compute_metrics') as mock_compute, \
             patch.object(benchmark, 'save_result') as mock_save:
            
            mock_get_data.return_value = [{"test": "data"}]
            mock_single.return_value = {"correct": True, "score": 1.0}
            mock_compute.return_value = {"score": 1.0}
            
            benchmark.run(perfect_model)
            
            mock_prepare.assert_called_once()
            mock_get_data.assert_called_once()
            mock_single.assert_called_once()
            mock_compute.assert_called_once()
            mock_save.assert_called_once()
            
    def test_run_with_limited_samples(self, benchmark, perfect_model):
        """Test running with limited number of samples."""
        result = benchmark.run(perfect_model, num_samples=2)
        
        assert result.metadata["num_samples"] == 2
        assert result.metrics["total_samples"] == 2
        
    def test_run_saves_results_by_default(self, benchmark, perfect_model):
        """Test that results are saved by default."""
        with patch.object(benchmark, 'save_result') as mock_save:
            result = benchmark.run(perfect_model)
            
            mock_save.assert_called_once_with(result)
            
    def test_run_skips_save_when_requested(self, benchmark, perfect_model):
        """Test that saving can be skipped."""
        with patch.object(benchmark, 'save_result') as mock_save:
            benchmark.run(perfect_model, save_results=False)
            
            mock_save.assert_not_called()
            
    def test_run_stores_result_in_memory(self, benchmark, perfect_model):
        """Test that results are stored in memory."""
        assert len(benchmark.results) == 0
        
        benchmark.run(perfect_model)
        
        assert len(benchmark.results) == 1
        assert benchmark.results[0].benchmark_name == "mock_benchmark"
        
    def test_multiple_runs_accumulate_results(self, benchmark, perfect_model):
        """Test that multiple runs accumulate results."""
        benchmark.run(perfect_model)
        benchmark.run(perfect_model)
        
        assert len(benchmark.results) == 2
        
    def test_run_logs_progress(self, benchmark, perfect_model, mock_logger):
        """Test that run logs appropriate progress messages."""
        benchmark.logger = mock_logger
        
        benchmark.run(perfect_model)
        
        # Check that info messages were logged
        mock_logger.info.assert_has_calls([
            call("Starting benchmark: mock_benchmark"),
            call("Result saved: mock_benchmark"),
            call(
                "Benchmark completed: mock_benchmark | "
                "Score: 1.0000 | "
                f"Duration: {benchmark.results[0].duration:.2f}s"
            )
        ])


class TestBenchmarkMetrics:
    """Test metrics computation and handling."""
    
    def test_compute_metrics_perfect_accuracy(self, benchmark):
        """Test metrics computation with perfect accuracy."""
        results = [
            {"correct": True, "score": 1.0},
            {"correct": True, "score": 1.0},
            {"correct": True, "score": 1.0}
        ]
        
        metrics = benchmark.compute_metrics(results)
        
        assert metrics["score"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["total_samples"] == 3
        assert metrics["correct_samples"] == 3
        
    def test_compute_metrics_partial_accuracy(self, benchmark):
        """Test metrics computation with partial accuracy."""
        results = [
            {"correct": True, "score": 1.0},
            {"correct": False, "score": 0.0},
            {"correct": True, "score": 1.0}
        ]
        
        metrics = benchmark.compute_metrics(results)
        
        assert metrics["score"] == 2/3
        assert metrics["accuracy"] == 2/3
        assert metrics["total_samples"] == 3
        assert metrics["correct_samples"] == 2
        
    def test_compute_metrics_zero_accuracy(self, benchmark):
        """Test metrics computation with zero accuracy."""
        results = [
            {"correct": False, "score": 0.0},
            {"correct": False, "score": 0.0}
        ]
        
        metrics = benchmark.compute_metrics(results)
        
        assert metrics["score"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["total_samples"] == 2
        assert metrics["correct_samples"] == 0
        
    def test_compute_metrics_empty_results(self, benchmark):
        """Test metrics computation with empty results."""
        metrics = benchmark.compute_metrics([])
        
        assert metrics["score"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["total_samples"] == 0
        assert metrics["correct_samples"] == 0


class TestBenchmarkResultSaving:
    """Test result saving functionality."""
    
    @patch('scramblebench.core.benchmark.Path')
    def test_save_result_creates_directory(self, mock_path, benchmark):
        """Test that save_result creates results directory."""
        mock_result = BenchmarkResult(
            benchmark_name="test",
            model_name="model",
            score=0.5,
            metrics={},
            metadata={},
            duration=1.0,
            timestamp=1000.0
        )
        
        mock_path.return_value.mkdir = MagicMock()
        
        benchmark.save_result(mock_result)
        
        mock_path.assert_called_with("test_results")
        mock_path.return_value.mkdir.assert_called_with(parents=True, exist_ok=True)
        
    def test_save_result_logs_message(self, benchmark, mock_logger):
        """Test that save_result logs appropriate message."""
        benchmark.logger = mock_logger
        
        mock_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="model",
            score=0.5,
            metrics={},
            metadata={},
            duration=1.0,
            timestamp=1000.0
        )
        
        benchmark.save_result(mock_result)
        
        mock_logger.info.assert_called_with("Result saved: test_benchmark")
        
    def test_save_result_uses_config_directory(self):
        """Test that save_result uses directory from config."""
        custom_config = Config({"results_dir": "custom/results/path"})
        benchmark = MockBenchmark(config=custom_config)
        
        with patch('scramblebench.core.benchmark.Path') as mock_path:
            mock_result = BenchmarkResult(
                benchmark_name="test",
                model_name="model", 
                score=0.5,
                metrics={},
                metadata={},
                duration=1.0,
                timestamp=1000.0
            )
            
            benchmark.save_result(mock_result)
            
            mock_path.assert_called_with("custom/results/path")


class TestBenchmarkUtilityMethods:
    """Test utility methods and properties."""
    
    def test_clear_results(self, benchmark, perfect_model):
        """Test clearing stored results."""
        benchmark.run(perfect_model)
        assert len(benchmark.results) == 1
        
        benchmark.clear_results()
        assert len(benchmark.results) == 0
        
    def test_validate_config_with_valid_config(self, benchmark):
        """Test config validation with valid config."""
        assert benchmark.validate_config() is True
        
    def test_validate_config_with_none_config(self):
        """Test config validation with None config."""
        benchmark = MockBenchmark()
        benchmark.config = None
        
        assert benchmark.validate_config() is False
        
    def test_results_property_is_readonly(self, benchmark, perfect_model):
        """Test that results property returns immutable copy."""
        benchmark.run(perfect_model)
        
        results = benchmark.results
        original_length = len(results)
        
        # Try to modify the returned list
        results.clear()
        
        # Original should be unchanged
        assert len(benchmark.results) == original_length


class TestBenchmarkErrorHandling:
    """Test error handling and edge cases."""
    
    def test_run_with_model_prediction_failure(self, benchmark):
        """Test handling model prediction failures."""
        failing_model = MockModel()
        failing_model.predict = MagicMock(side_effect=Exception("Model failed"))
        
        with pytest.raises(Exception, match="Model failed"):
            benchmark.run(failing_model)
            
    def test_run_with_empty_data(self, benchmark, perfect_model):
        """Test running with empty evaluation data."""
        benchmark.data = []  # Set empty data
        
        result = benchmark.run(perfect_model)
        
        assert result.score == 0.0
        assert result.metrics["total_samples"] == 0
        
    def test_single_evaluation_result_structure(self, benchmark, perfect_model):
        """Test that single evaluation returns expected structure."""
        benchmark.prepare_data()
        data_item = benchmark.data[0]
        
        result = benchmark.run_single_evaluation(perfect_model, data_item)
        
        assert "predicted" in result
        assert "expected" in result
        assert "correct" in result
        assert "score" in result
        assert isinstance(result["correct"], bool)
        assert isinstance(result["score"], (int, float))
        
    def test_model_without_name_attribute(self, benchmark):
        """Test handling model without name attribute."""
        class NamelessModel:
            def predict(self, x):
                return "4"
                
        nameless_model = NamelessModel()
        result = benchmark.run(nameless_model)
        
        # Should use string representation
        assert result.model_name == str(nameless_model)


class TestBenchmarkTiming:
    """Test timing and performance measurement."""
    
    def test_benchmark_duration_measurement(self, benchmark, perfect_model):
        """Test that benchmark duration is measured correctly."""
        with patch('time.time', side_effect=[1000.0, 1002.5]):  # 2.5 second duration
            result = benchmark.run(perfect_model)
            
        assert result.duration == 2.5
        assert result.timestamp == 1000.0
        
    def test_duration_is_positive(self, benchmark, perfect_model):
        """Test that duration is always positive."""
        result = benchmark.run(perfect_model)
        
        assert result.duration >= 0
        
    def test_timestamp_is_recorded(self, benchmark, perfect_model):
        """Test that timestamp is recorded at start."""
        before_run = time.time()
        result = benchmark.run(perfect_model)
        after_run = time.time()
        
        assert before_run <= result.timestamp <= after_run


class TestBenchmarkConcurrency:
    """Test concurrent execution scenarios."""
    
    def test_multiple_benchmarks_independent(self):
        """Test that multiple benchmark instances are independent."""
        benchmark1 = MockBenchmark("bench1")
        benchmark2 = MockBenchmark("bench2")
        
        model = MockModel()
        
        result1 = benchmark1.run(model)
        result2 = benchmark2.run(model)
        
        assert result1.benchmark_name == "bench1"
        assert result2.benchmark_name == "bench2"
        assert len(benchmark1.results) == 1
        assert len(benchmark2.results) == 1
        
    def test_model_call_counting(self, benchmark):
        """Test that model calls are counted correctly."""
        model = MockModel()
        
        benchmark.run(model)
        
        # Should be called once per data item
        assert model.call_count == 3


class TestBenchmarkConfiguration:
    """Test configuration handling and validation."""
    
    def test_config_is_included_in_metadata(self, benchmark, perfect_model):
        """Test that config is included in result metadata."""
        result = benchmark.run(perfect_model)
        
        assert "config" in result.metadata
        assert isinstance(result.metadata["config"], dict)
        assert result.metadata["config"]["random_seed"] == 42
        
    def test_benchmark_respects_config_settings(self):
        """Test that benchmark respects configuration settings."""
        config = Config({
            "results_dir": "custom_results",
            "debug": True
        })
        
        benchmark = MockBenchmark(config=config)
        
        assert benchmark.config.get("results_dir") == "custom_results"
        assert benchmark.config.get("debug") is True
        
    def test_config_changes_affect_behavior(self):
        """Test that config changes affect benchmark behavior."""
        # Test with different results directories
        config1 = Config({"results_dir": "results1"})
        config2 = Config({"results_dir": "results2"})
        
        bench1 = MockBenchmark(config=config1)
        bench2 = MockBenchmark(config=config2)
        
        with patch('scramblebench.core.benchmark.Path') as mock_path:
            mock_result = BenchmarkResult(
                benchmark_name="test",
                model_name="model",
                score=0.5,
                metrics={},
                metadata={},
                duration=1.0,
                timestamp=1000.0
            )
            
            bench1.save_result(mock_result)
            bench2.save_result(mock_result)
            
            # Should be called with different paths
            assert mock_path.call_args_list[0][0][0] == "results1"
            assert mock_path.call_args_list[1][0][0] == "results2"