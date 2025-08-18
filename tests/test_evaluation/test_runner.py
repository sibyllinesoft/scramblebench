"""
Tests for evaluation runner functionality.

This module provides comprehensive tests for the EvaluationRunner class,
covering pipeline orchestration, transformation generation, model evaluation,
and results management.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

from scramblebench.evaluation.runner import EvaluationRunner
from scramblebench.evaluation.config import EvaluationConfig, ModelConfig, ModelProvider, TransformationConfig
from scramblebench.evaluation.results import EvaluationResults
from scramblebench.evaluation.openrouter_runner import EvaluationResult


class MockTransformationPipeline:
    """Mock transformation pipeline for testing."""
    
    def __init__(self, config, data_dir=None, logger=None):
        self.config = config
        self.data_dir = data_dir
        self.logger = logger
        self.generated_transformations = []
        
    async def generate_transformations(self, benchmark_data: List[Dict[str, Any]], max_samples: int = None) -> List[Dict[str, Any]]:
        """Mock transformation generation."""
        transformations = []
        for item in benchmark_data:
            # Create mock transformations
            if self.config.enabled_types:
                for transform_type in self.config.enabled_types:
                    transformed_item = item.copy()
                    transformed_item['transformed'] = True
                    transformed_item['transformation_type'] = transform_type
                    # Simple transformation - replace 'e' with 'æ'
                    if 'question' in transformed_item:
                        transformed_item['question'] = transformed_item['question'].replace('e', 'æ')
                    transformations.append(transformed_item)
            else:
                transformations.append(item)
                
        self.generated_transformations = transformations
        return transformations[:max_samples] if max_samples else transformations


class MockModelRunner:
    """Mock model runner for testing."""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.evaluation_results = []
        
    async def evaluate_benchmark_data(
        self, 
        original_data: List[Dict[str, Any]], 
        transformed_data: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Mock evaluation."""
        results = []
        
        for model in self.config.models:
            # Mock results for original data
            for item in original_data:
                result = EvaluationResult(
                    model_name=model.name,
                    question_id=item.get('id', 'unknown'),
                    question=item.get('question', ''),
                    expected_answer=item.get('answer', ''),
                    predicted_answer="Mock answer",
                    correct=True,
                    score=1.0,
                    response_time=0.5,
                    metadata={'transformation_type': 'original'}
                )
                results.append(result)
                
            # Mock results for transformed data
            for item in transformed_data:
                result = EvaluationResult(
                    model_name=model.name,
                    question_id=item.get('id', 'unknown'),
                    question=item.get('question', ''),
                    expected_answer=item.get('answer', ''),
                    predicted_answer="Mock transformed answer",
                    correct=False,  # Lower performance on transformed data
                    score=0.7,
                    response_time=0.6,
                    metadata={'transformation_type': item.get('transformation_type', 'unknown')}
                )
                results.append(result)
                
        self.evaluation_results = results
        return results


class MockResultsManager:
    """Mock results manager for testing."""
    
    def __init__(self, output_dir, logger=None):
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.saved_results = []
        
    def save_results(self, results: EvaluationResults) -> Path:
        """Mock results saving."""
        self.saved_results.append(results)
        return self.output_dir / f"results_{results.experiment_name}.json"
        
    def load_results(self, experiment_name: str) -> EvaluationResults:
        """Mock results loading."""
        # Return mock results
        return EvaluationResults(
            experiment_name=experiment_name,
            description="Mock results",
            model_results={},
            metrics={},
            metadata={}
        )
        
    def list_experiments(self) -> List[str]:
        """Mock experiment listing."""
        return ["exp1", "exp2", "exp3"]


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self):
        self.mock_data = [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is 3+3?", "answer": "6"},
            {"id": "q3", "question": "What is the capital of France?", "answer": "Paris"}
        ]
        
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Mock dataset loading."""
        return self.mock_data.copy()


class MockMetricsCalculator:
    """Mock metrics calculator for testing."""
    
    def __init__(self):
        pass
        
    def calculate_robustness_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Mock robustness metrics calculation."""
        return {
            "original_accuracy": 1.0,
            "transformed_accuracy": 0.7,
            "robustness_score": 0.7,
            "degradation": 0.3,
            "model_count": len(set(r.model_name for r in results)),
            "question_count": len(set(r.question_id for r in results))
        }
        
    def calculate_significance_tests(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Mock significance tests."""
        return {
            "t_test_pvalue": 0.001,
            "wilcoxon_pvalue": 0.002,
            "effect_size": 0.8,
            "significant": True
        }


class MockPlotGenerator:
    """Mock plot generator for testing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.generated_plots = []
        
    def generate_comparison_plots(self, results: EvaluationResults) -> List[Path]:
        """Mock plot generation."""
        plot_paths = [
            self.output_dir / "accuracy_comparison.png",
            self.output_dir / "robustness_heatmap.png"
        ]
        self.generated_plots.extend(plot_paths)
        return plot_paths
        
    def generate_degradation_analysis(self, results: EvaluationResults) -> Path:
        """Mock degradation analysis plot."""
        plot_path = self.output_dir / "degradation_analysis.png"
        self.generated_plots.append(plot_path)
        return plot_path


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_config():
    """Create a mock evaluation configuration."""
    return EvaluationConfig(
        experiment_name="test_experiment",
        description="Test evaluation",
        models=[
            ModelConfig(
                name="openai/gpt-3.5-turbo",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0
            ),
            ModelConfig(
                name="anthropic/claude-3-haiku",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0
            )
        ],
        benchmark_paths=["test_benchmark.json"],
        transformations=TransformationConfig(
            enabled_types=["language_translation", "synonym_replacement"],
            synonym_rate=0.3
        ),
        max_samples=10,
        generate_plots=True,
        calculate_significance=True
    )

@pytest.fixture
def evaluation_runner(mock_config, temp_dir):
    """Create an evaluation runner for testing."""
    runner = EvaluationRunner(mock_config, data_dir=temp_dir)
    
    # Replace components with mocks
    runner.transformation_pipeline = MockTransformationPipeline(
        mock_config.transformations, 
        data_dir=temp_dir
    )
    runner.results_manager = MockResultsManager(temp_dir)
    runner.data_loader = MockDataLoader()
    runner.model_runners = {
        "openrouter": MockModelRunner(mock_config)
    }
    
    return runner


class TestEvaluationRunnerInitialization:
    """Test evaluation runner initialization."""
    
    def test_initialization_basic(self, mock_config, temp_dir):
        """Test basic initialization."""
        runner = EvaluationRunner(mock_config, data_dir=temp_dir)
        
        assert runner.config == mock_config
        assert runner.data_dir == temp_dir
        assert runner.logger is not None
        assert runner.transformation_pipeline is not None
        assert runner.results_manager is not None
        assert runner.data_loader is not None
        
    def test_initialization_default_data_dir(self, mock_config):
        """Test initialization with default data directory."""
        runner = EvaluationRunner(mock_config)
        
        assert runner.data_dir == Path("data")
        
    def test_initialization_custom_logger(self, mock_config, temp_dir):
        """Test initialization with custom logger."""
        custom_logger = Mock()
        runner = EvaluationRunner(mock_config, data_dir=temp_dir, logger=custom_logger)
        
        assert runner.logger == custom_logger
        
    def test_model_runners_initialization(self, mock_config, temp_dir):
        """Test that model runners are properly initialized."""
        with patch('scramblebench.evaluation.runner.OpenRouterEvaluationRunner') as mock_runner_class:
            runner = EvaluationRunner(mock_config, data_dir=temp_dir)
            
        # Should create OpenRouter runner for OpenRouter models
        mock_runner_class.assert_called_once()
        assert "openrouter" in runner.model_runners
        
    def test_unsupported_provider_warning(self, temp_dir):
        """Test warning for unsupported model providers."""
        # Create config with unsupported provider
        config = EvaluationConfig(
            experiment_name="test",
            models=[
                ModelConfig(
                    name="unsupported/model",
                    provider="unsupported_provider",  # This should trigger warning
                    temperature=0.0
                )
            ],
            benchmark_paths=["test.json"],
            transformations=TransformationConfig()
        )
        
        mock_logger = Mock()
        runner = EvaluationRunner(config, data_dir=temp_dir, logger=mock_logger)
        
        # Should log warning about unsupported provider
        mock_logger.warning.assert_called()


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.mark.asyncio
    async def test_load_benchmark_data_single_file(self, evaluation_runner):
        """Test loading data from a single benchmark file."""
        # Mock the data loader to return test data
        evaluation_runner.data_loader.load_dataset = Mock(return_value=[
            {"id": "q1", "question": "Test question", "answer": "Test answer"}
        ])
        
        evaluation_runner.config.benchmark_paths = ["single_benchmark.json"]
        
        data = await evaluation_runner._load_benchmark_data()
        
        assert len(data) == 1
        assert data[0]["question"] == "Test question"
        
    @pytest.mark.asyncio
    async def test_load_benchmark_data_multiple_files(self, evaluation_runner):
        """Test loading data from multiple benchmark files."""
        def mock_load_dataset(path):
            if "benchmark1" in path:
                return [{"id": "q1", "question": "Question 1", "answer": "Answer 1"}]
            elif "benchmark2" in path:
                return [{"id": "q2", "question": "Question 2", "answer": "Answer 2"}]
            return []
            
        evaluation_runner.data_loader.load_dataset = Mock(side_effect=mock_load_dataset)
        evaluation_runner.config.benchmark_paths = ["benchmark1.json", "benchmark2.json"]
        
        data = await evaluation_runner._load_benchmark_data()
        
        assert len(data) == 2
        assert data[0]["question"] == "Question 1"
        assert data[1]["question"] == "Question 2"
        
    @pytest.mark.asyncio
    async def test_load_benchmark_data_with_max_samples(self, evaluation_runner):
        """Test loading data with max samples limit."""
        evaluation_runner.data_loader.load_dataset = Mock(return_value=[
            {"id": f"q{i}", "question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(10)
        ])
        
        evaluation_runner.config.benchmark_paths = ["large_benchmark.json"]
        evaluation_runner.config.max_samples = 5
        
        data = await evaluation_runner._load_benchmark_data()
        
        assert len(data) == 5
        
    @pytest.mark.asyncio
    async def test_load_benchmark_data_file_not_found(self, evaluation_runner):
        """Test handling of missing benchmark files."""
        evaluation_runner.data_loader.load_dataset = Mock(side_effect=FileNotFoundError("File not found"))
        evaluation_runner.config.benchmark_paths = ["missing_file.json"]
        
        with pytest.raises(FileNotFoundError):
            await evaluation_runner._load_benchmark_data()


class TestTransformationGeneration:
    """Test transformation generation."""
    
    @pytest.mark.asyncio
    async def test_generate_transformations_success(self, evaluation_runner):
        """Test successful transformation generation."""
        original_data = [
            {"id": "q1", "question": "What is test?", "answer": "Test answer"}
        ]
        
        transformations = await evaluation_runner._generate_transformations(original_data)
        
        assert len(transformations) > 0
        # Check that transformations were applied
        assert any(item.get('transformed') for item in transformations)
        
    @pytest.mark.asyncio
    async def test_generate_transformations_with_max_samples(self, evaluation_runner):
        """Test transformation generation with sample limit."""
        original_data = [
            {"id": f"q{i}", "question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(10)
        ]
        
        evaluation_runner.config.max_samples = 3
        
        transformations = await evaluation_runner._generate_transformations(original_data)
        
        # Should respect max_samples limit
        assert len(transformations) <= 6  # 3 samples * 2 transformation types
        
    @pytest.mark.asyncio
    async def test_generate_transformations_empty_data(self, evaluation_runner):
        """Test transformation generation with empty data."""
        transformations = await evaluation_runner._generate_transformations([])
        
        assert transformations == []
        
    @pytest.mark.asyncio
    async def test_generate_transformations_no_enabled_types(self, evaluation_runner):
        """Test transformation generation with no enabled transformation types."""
        evaluation_runner.transformation_pipeline.config.enabled_types = []
        
        original_data = [{"id": "q1", "question": "Test", "answer": "Answer"}]
        transformations = await evaluation_runner._generate_transformations(original_data)
        
        # Should return original data
        assert len(transformations) == 1
        assert transformations[0] == original_data[0]


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluate_models_success(self, evaluation_runner):
        """Test successful model evaluation."""
        original_data = [{"id": "q1", "question": "Test", "answer": "Answer"}]
        transformed_data = [{"id": "q1", "question": "Tæst", "answer": "Answer", "transformation_type": "language_translation"}]
        
        results = await evaluation_runner._evaluate_models(original_data, transformed_data)
        
        assert len(results) > 0
        assert all(isinstance(result, EvaluationResult) for result in results)
        
        # Should have results for both original and transformed data
        original_results = [r for r in results if r.metadata.get('transformation_type') == 'original']
        transformed_results = [r for r in results if r.metadata.get('transformation_type') != 'original']
        
        assert len(original_results) > 0
        assert len(transformed_results) > 0
        
    @pytest.mark.asyncio
    async def test_evaluate_models_multiple_providers(self, evaluation_runner):
        """Test evaluation with multiple model providers."""
        # Add another mock provider
        evaluation_runner.model_runners["another_provider"] = MockModelRunner(evaluation_runner.config)
        
        original_data = [{"id": "q1", "question": "Test", "answer": "Answer"}]
        transformed_data = [{"id": "q1", "question": "Tæst", "answer": "Answer"}]
        
        results = await evaluation_runner._evaluate_models(original_data, transformed_data)
        
        # Should have results from both providers
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_evaluate_models_empty_data(self, evaluation_runner):
        """Test evaluation with empty data."""
        results = await evaluation_runner._evaluate_models([], [])
        
        assert results == []
        
    @pytest.mark.asyncio
    async def test_evaluate_models_provider_failure(self, evaluation_runner):
        """Test handling of provider evaluation failures."""
        # Mock provider to raise exception
        evaluation_runner.model_runners["openrouter"].evaluate_benchmark_data = AsyncMock(
            side_effect=Exception("Provider failed")
        )
        
        original_data = [{"id": "q1", "question": "Test", "answer": "Answer"}]
        transformed_data = [{"id": "q1", "question": "Tæst", "answer": "Answer"}]
        
        with pytest.raises(Exception, match="Provider failed"):
            await evaluation_runner._evaluate_models(original_data, transformed_data)


class TestResultsProcessing:
    """Test results processing and analysis."""
    
    def test_process_results_basic(self, evaluation_runner):
        """Test basic results processing."""
        mock_results = [
            EvaluationResult(
                model_name="test_model",
                question_id="q1",
                question="Test",
                expected_answer="Answer",
                predicted_answer="Answer",
                correct=True,
                score=1.0,
                response_time=0.5,
                metadata={"transformation_type": "original"}
            )
        ]
        
        processed_results = evaluation_runner._process_results(mock_results)
        
        assert isinstance(processed_results, EvaluationResults)
        assert processed_results.experiment_name == evaluation_runner.config.experiment_name
        assert processed_results.description == evaluation_runner.config.description
        
    def test_process_results_with_metrics(self, evaluation_runner):
        """Test results processing with metrics calculation."""
        mock_results = [
            EvaluationResult(
                model_name="test_model",
                question_id="q1",
                question="Test",
                expected_answer="Answer",
                predicted_answer="Answer",
                correct=True,
                score=1.0,
                response_time=0.5,
                metadata={"transformation_type": "original"}
            )
        ]
        
        # Mock metrics calculator
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        processed_results = evaluation_runner._process_results(mock_results)
        
        assert 'robustness_metrics' in processed_results.metrics
        assert processed_results.metrics['robustness_metrics']['robustness_score'] == 0.7
        
    def test_process_results_with_significance_tests(self, evaluation_runner):
        """Test results processing with significance tests."""
        evaluation_runner.config.calculate_significance = True
        
        mock_results = [
            EvaluationResult(
                model_name="test_model",
                question_id="q1",
                question="Test",
                expected_answer="Answer",
                predicted_answer="Answer",
                correct=True,
                score=1.0,
                response_time=0.5,
                metadata={"transformation_type": "original"}
            )
        ]
        
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        processed_results = evaluation_runner._process_results(mock_results)
        
        assert 'significance_tests' in processed_results.metrics
        assert processed_results.metrics['significance_tests']['significant'] is True
        
    def test_process_results_empty(self, evaluation_runner):
        """Test processing of empty results."""
        processed_results = evaluation_runner._process_results([])
        
        assert isinstance(processed_results, EvaluationResults)
        assert processed_results.experiment_name == evaluation_runner.config.experiment_name


class TestPlotGeneration:
    """Test plot generation functionality."""
    
    def test_generate_plots_enabled(self, evaluation_runner, temp_dir):
        """Test plot generation when enabled."""
        evaluation_runner.config.generate_plots = True
        
        mock_results = EvaluationResults(
            experiment_name="test",
            description="Test results",
            model_results={},
            metrics={},
            metadata={}
        )
        
        # Mock plot generator
        evaluation_runner._plot_generator = MockPlotGenerator(temp_dir)
        
        plot_paths = evaluation_runner._generate_plots(mock_results)
        
        assert len(plot_paths) > 0
        assert all(isinstance(path, Path) for path in plot_paths)
        
    def test_generate_plots_disabled(self, evaluation_runner):
        """Test when plot generation is disabled."""
        evaluation_runner.config.generate_plots = False
        
        mock_results = EvaluationResults(
            experiment_name="test",
            description="Test results",
            model_results={},
            metrics={},
            metadata={}
        )
        
        plot_paths = evaluation_runner._generate_plots(mock_results)
        
        assert plot_paths == []
        
    def test_generate_plots_failure_handling(self, evaluation_runner, temp_dir):
        """Test handling of plot generation failures."""
        evaluation_runner.config.generate_plots = True
        
        mock_results = EvaluationResults(
            experiment_name="test",
            description="Test results",
            model_results={},
            metrics={},
            metadata={}
        )
        
        # Mock plot generator to raise exception
        mock_plot_generator = Mock()
        mock_plot_generator.generate_comparison_plots.side_effect = Exception("Plot generation failed")
        evaluation_runner._plot_generator = mock_plot_generator
        
        mock_logger = Mock()
        evaluation_runner.logger = mock_logger
        
        plot_paths = evaluation_runner._generate_plots(mock_results)
        
        # Should handle error gracefully
        assert plot_paths == []
        mock_logger.error.assert_called()


class TestFullEvaluationPipeline:
    """Test the complete evaluation pipeline."""
    
    @pytest.mark.asyncio
    async def test_run_evaluation_complete_pipeline(self, evaluation_runner):
        """Test running the complete evaluation pipeline."""
        # Mock all external dependencies
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        evaluation_runner._plot_generator = MockPlotGenerator(evaluation_runner.data_dir)
        
        results = await evaluation_runner.run_evaluation()
        
        assert isinstance(results, EvaluationResults)
        assert results.experiment_name == evaluation_runner.config.experiment_name
        
    @pytest.mark.asyncio
    async def test_run_evaluation_without_original(self, evaluation_runner):
        """Test running evaluation without original data."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        results = await evaluation_runner.run_evaluation(include_original=False)
        
        assert isinstance(results, EvaluationResults)
        
    @pytest.mark.asyncio
    async def test_run_evaluation_without_intermediate_saving(self, evaluation_runner):
        """Test running evaluation without intermediate result saving."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        results = await evaluation_runner.run_evaluation(save_intermediate=False)
        
        assert isinstance(results, EvaluationResults)
        
    @pytest.mark.asyncio
    async def test_run_evaluation_progress_tracking(self, evaluation_runner):
        """Test that evaluation progress is properly tracked."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        # Mock logger to check progress messages
        mock_logger = Mock()
        evaluation_runner.logger = mock_logger
        
        await evaluation_runner.run_evaluation()
        
        # Should log progress messages
        log_calls = mock_logger.info.call_args_list
        log_messages = [call[0][0] for call in log_calls]
        
        assert any("Starting evaluation" in msg for msg in log_messages)
        assert any("Loading benchmark data" in msg for msg in log_messages)
        assert any("Generating transformations" in msg for msg in log_messages)
        assert any("Evaluating models" in msg for msg in log_messages)
        assert any("Processing results" in msg for msg in log_messages)
        
    @pytest.mark.asyncio
    async def test_run_evaluation_timing(self, evaluation_runner):
        """Test that evaluation timing is tracked."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        results = await evaluation_runner.run_evaluation()
        
        assert 'timing' in results.metadata
        assert 'total_duration' in results.metadata['timing']
        assert results.metadata['timing']['total_duration'] > 0
        
    @pytest.mark.asyncio
    async def test_run_evaluation_stage_tracking(self, evaluation_runner):
        """Test that evaluation stages are properly tracked."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        # Check initial stage
        assert evaluation_runner._current_stage == "initialized"
        
        await evaluation_runner.run_evaluation()
        
        # Should end in completed stage
        assert evaluation_runner._current_stage == "completed"


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_data_loading_failure(self, evaluation_runner):
        """Test handling of data loading failures."""
        evaluation_runner.data_loader.load_dataset = Mock(
            side_effect=Exception("Data loading failed")
        )
        
        with pytest.raises(Exception, match="Data loading failed"):
            await evaluation_runner.run_evaluation()
            
    @pytest.mark.asyncio
    async def test_transformation_failure(self, evaluation_runner):
        """Test handling of transformation failures."""
        evaluation_runner.transformation_pipeline.generate_transformations = AsyncMock(
            side_effect=Exception("Transformation failed")
        )
        
        with pytest.raises(Exception, match="Transformation failed"):
            await evaluation_runner.run_evaluation()
            
    @pytest.mark.asyncio
    async def test_evaluation_failure(self, evaluation_runner):
        """Test handling of evaluation failures."""
        evaluation_runner.model_runners["openrouter"].evaluate_benchmark_data = AsyncMock(
            side_effect=Exception("Evaluation failed")
        )
        
        with pytest.raises(Exception, match="Evaluation failed"):
            await evaluation_runner.run_evaluation()
            
    @pytest.mark.asyncio
    async def test_results_saving_failure(self, evaluation_runner):
        """Test handling of results saving failures."""
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        evaluation_runner.results_manager.save_results = Mock(
            side_effect=Exception("Results saving failed")
        )
        
        mock_logger = Mock()
        evaluation_runner.logger = mock_logger
        
        # Should continue despite saving failure
        results = await evaluation_runner.run_evaluation()
        
        assert isinstance(results, EvaluationResults)
        mock_logger.error.assert_called()


class TestConfigurationHandling:
    """Test configuration handling and validation."""
    
    def test_config_validation_valid(self, evaluation_runner):
        """Test validation of valid configuration."""
        is_valid = evaluation_runner._validate_config()
        
        assert is_valid is True
        
    def test_config_validation_missing_models(self, temp_dir):
        """Test validation with missing models."""
        config = EvaluationConfig(
            experiment_name="test",
            models=[],  # Empty models list
            benchmark_paths=["test.json"],
            transformations=TransformationConfig()
        )
        
        runner = EvaluationRunner(config, data_dir=temp_dir)
        
        is_valid = runner._validate_config()
        
        assert is_valid is False
        
    def test_config_validation_missing_benchmarks(self, temp_dir):
        """Test validation with missing benchmark paths."""
        config = EvaluationConfig(
            experiment_name="test",
            models=[
                ModelConfig(name="test_model", provider=ModelProvider.OPENROUTER)
            ],
            benchmark_paths=[],  # Empty benchmark paths
            transformations=TransformationConfig()
        )
        
        runner = EvaluationRunner(config, data_dir=temp_dir)
        
        is_valid = runner._validate_config()
        
        assert is_valid is False
        
    def test_config_validation_invalid_transformation_config(self, temp_dir):
        """Test validation with invalid transformation configuration."""
        config = EvaluationConfig(
            experiment_name="test",
            models=[
                ModelConfig(name="test_model", provider=ModelProvider.OPENROUTER)
            ],
            benchmark_paths=["test.json"],
            transformations=TransformationConfig(
                synonym_rate=1.5  # Invalid rate > 1.0
            )
        )
        
        runner = EvaluationRunner(config, data_dir=temp_dir)
        
        is_valid = runner._validate_config()
        
        assert is_valid is False


class TestUtilityMethods:
    """Test utility methods and helpers."""
    
    def test_get_current_stage(self, evaluation_runner):
        """Test getting current evaluation stage."""
        assert evaluation_runner.get_current_stage() == "initialized"
        
        evaluation_runner._current_stage = "loading_data"
        assert evaluation_runner.get_current_stage() == "loading_data"
        
    def test_get_progress_info(self, evaluation_runner):
        """Test getting progress information."""
        progress = evaluation_runner.get_progress_info()
        
        assert isinstance(progress, dict)
        assert 'current_stage' in progress
        assert 'start_time' in progress
        assert 'elapsed_time' in progress
        
    def test_cancel_evaluation(self, evaluation_runner):
        """Test evaluation cancellation."""
        # This would test cancellation functionality if implemented
        # For now, just test that the method exists and can be called
        evaluation_runner.cancel_evaluation()
        
        # Check that cancellation state is set
        assert hasattr(evaluation_runner, '_cancelled')
        
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, evaluation_runner):
        """Test resource cleanup."""
        # Run evaluation to initialize resources
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        await evaluation_runner.run_evaluation()
        
        # Test cleanup
        evaluation_runner.cleanup_resources()
        
        # Verify cleanup occurred (implementation-dependent)
        assert hasattr(evaluation_runner, '_cleaned_up')


class TestConcurrencyAndPerformance:
    """Test concurrency and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_model_evaluation(self, evaluation_runner):
        """Test that model evaluation can handle concurrency."""
        # Add multiple model runners
        evaluation_runner.model_runners = {
            "provider1": MockModelRunner(evaluation_runner.config),
            "provider2": MockModelRunner(evaluation_runner.config),
            "provider3": MockModelRunner(evaluation_runner.config)
        }
        
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        original_data = [{"id": "q1", "question": "Test", "answer": "Answer"}]
        transformed_data = [{"id": "q1", "question": "Tæst", "answer": "Answer"}]
        
        # Measure time
        import time
        start_time = time.time()
        
        results = await evaluation_runner._evaluate_models(original_data, transformed_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time even with multiple providers
        assert duration < 5.0  # Should complete within 5 seconds
        assert len(results) > 0
        
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, evaluation_runner):
        """Test handling of large datasets."""
        # Create large mock dataset
        large_data = [
            {"id": f"q{i}", "question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(1000)
        ]
        
        evaluation_runner.data_loader.load_dataset = Mock(return_value=large_data)
        evaluation_runner.config.max_samples = 100  # Limit for testing
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        results = await evaluation_runner.run_evaluation()
        
        assert isinstance(results, EvaluationResults)
        # Should handle large datasets efficiently
        
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, evaluation_runner):
        """Test memory usage during evaluation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        evaluation_runner._metrics_calculator = MockMetricsCalculator()
        
        await evaluation_runner.run_evaluation()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB


class TestIntegrationWithExternalSystems:
    """Test integration with external systems and APIs."""
    
    @pytest.mark.asyncio
    async def test_openrouter_integration(self, evaluation_runner):
        """Test integration with OpenRouter API."""
        # This would test actual OpenRouter integration
        # For now, test that the integration point exists
        assert "openrouter" in evaluation_runner.model_runners
        
        openrouter_runner = evaluation_runner.model_runners["openrouter"]
        assert hasattr(openrouter_runner, 'evaluate_benchmark_data')
        
    def test_results_export_formats(self, evaluation_runner):
        """Test exporting results in different formats."""
        mock_results = EvaluationResults(
            experiment_name="test",
            description="Test results",
            model_results={},
            metrics={},
            metadata={}
        )
        
        # Test JSON export
        json_path = evaluation_runner.results_manager.save_results(mock_results)
        assert json_path.suffix == '.json'
        
        # Test that results were saved
        assert len(evaluation_runner.results_manager.saved_results) == 1
        
    def test_plot_integration(self, evaluation_runner, temp_dir):
        """Test integration with plotting system."""
        evaluation_runner.config.generate_plots = True
        evaluation_runner._plot_generator = MockPlotGenerator(temp_dir)
        
        mock_results = EvaluationResults(
            experiment_name="test",
            description="Test results",
            model_results={},
            metrics={},
            metadata={}
        )
        
        plot_paths = evaluation_runner._generate_plots(mock_results)
        
        assert len(plot_paths) > 0
        assert all(path.suffix == '.png' for path in plot_paths)