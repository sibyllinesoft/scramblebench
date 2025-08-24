"""
Tests for the evaluation pipeline components.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import pandas as pd

from scramblebench.core.unified_config import (
    EvaluationConfig, ModelConfig, TransformationConfig
)
from scramblebench.evaluation.transformation_pipeline import (
    TransformationPipeline, TransformationResult, TransformationSet
)
from scramblebench.evaluation.openrouter_runner import (
    OpenRouterEvaluationRunner, EvaluationResult
)
from scramblebench.evaluation.results import ResultsManager, EvaluationResults
from scramblebench.evaluation.metrics import MetricsCalculator
from scramblebench.evaluation.plotting import PlotGenerator
from scramblebench.evaluation.runner import EvaluationRunner


class TestEvaluationConfig:
    """Test configuration classes."""
    
    def test_basic_config_creation(self):
        """Test creating a basic evaluation configuration."""
        config = EvaluationConfig(
            experiment_name="test_experiment",
            benchmark_paths=["data/test.json"],
            models=[
                ModelConfig(
                    name="test/model",
                    provider=ModelProvider.OPENROUTER
                )
            ]
        )
        
        assert config.experiment_name == "test_experiment"
        assert len(config.models) == 1
        assert config.models[0].name == "test/model"
        assert config.transformations.enabled_types == [TransformationType.ALL]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid transformation type
        with pytest.raises(ValueError):
            TransformationConfig(enabled_types=["invalid_type"])
    
    def test_config_serialization(self):
        """Test configuration serialization to/from file."""
        config = EvaluationConfig(
            experiment_name="test_experiment",
            benchmark_paths=["data/test.json"],
            models=[
                ModelConfig(
                    name="test/model",
                    provider=ModelProvider.OPENROUTER,
                    temperature=0.5
                )
            ],
            transformations=TransformationConfig(
                enabled_types=[TransformationType.LANGUAGE_TRANSLATION],
                synonym_rate=0.3
            )
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_to_file(f.name)
            
            # Load back and verify
            loaded_config = EvaluationConfig.load_from_file(f.name)
            assert loaded_config.experiment_name == config.experiment_name
            assert loaded_config.models[0].temperature == 0.5
            assert loaded_config.transformations.synonym_rate == 0.3


class TestTransformationPipeline:
    """Test transformation pipeline."""
    
    @pytest.fixture
    def sample_problems(self):
        """Sample benchmark problems for testing."""
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "answer": "Paris"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "choices": ["Shakespeare", "Dickens", "Austen", "Wilde"],
                "answer": "Shakespeare"
            }
        ]
    
    @pytest.fixture
    def transformation_config(self):
        """Sample transformation configuration."""
        return TransformationConfig(
            enabled_types=[TransformationType.SYNONYM_REPLACEMENT],
            synonym_rate=0.3,
            seed=42
        )
    
    def test_pipeline_creation(self, transformation_config):
        """Test creating transformation pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TransformationPipeline(
                transformation_config,
                data_dir=Path(temp_dir)
            )
            
            assert pipeline.config == transformation_config
            assert pipeline.data_dir == Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_synonym_replacement(self, transformation_config, sample_problems):
        """Test synonym replacement transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TransformationPipeline(
                transformation_config,
                data_dir=Path(temp_dir)
            )
            
            # Test single transformation
            result = pipeline._apply_transformation(
                sample_problems[0],
                TransformationType.SYNONYM_REPLACEMENT
            )
            
            assert result is not None
            assert result.success
            assert result.transformation_type == TransformationType.SYNONYM_REPLACEMENT.value
            assert "replacements" in result.transformation_metadata
    
    @pytest.mark.asyncio
    async def test_generate_transformation_sets(self, transformation_config, sample_problems):
        """Test generating transformation sets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TransformationPipeline(
                transformation_config,
                data_dir=Path(temp_dir)
            )
            
            transformation_sets = await pipeline.generate_transformation_sets(
                sample_problems,
                ["problem_1", "problem_2"]
            )
            
            assert len(transformation_sets) == 2
            assert all(isinstance(ts, TransformationSet) for ts in transformation_sets)
            assert transformation_sets[0].problem_id == "problem_1"
    
    def test_transformation_stats(self, transformation_config, sample_problems):
        """Test transformation statistics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TransformationPipeline(
                transformation_config,
                data_dir=Path(temp_dir)
            )
            
            # Create mock transformation sets
            transformation_sets = [
                TransformationSet(
                    original_problem=sample_problems[0],
                    transformations=[
                        TransformationResult(
                            original_problem=sample_problems[0],
                            transformed_problem=sample_problems[0],
                            transformation_type="test_type",
                            transformation_metadata={},
                            success=True
                        )
                    ],
                    problem_id="problem_1"
                )
            ]
            
            stats = pipeline.get_transformation_stats(transformation_sets)
            
            assert stats["total_problems"] == 1
            assert stats["total_transformations"] == 1
            assert stats["successful_transformations"] == 1
            assert stats["overall_success_rate"] == 1.0


class TestOpenRouterRunner:
    """Test OpenRouter evaluation runner."""
    
    @pytest.fixture
    def evaluation_config(self):
        """Sample evaluation configuration."""
        return EvaluationConfig(
            experiment_name="test_evaluation",
            benchmark_paths=["data/test.json"],
            models=[
                ModelConfig(
                    name="test/model",
                    provider=ModelProvider.OPENROUTER,
                    api_key="test_key"
                )
            ]
        )
    
    @pytest.fixture
    def sample_transformation_sets(self):
        """Sample transformation sets for testing."""
        return [
            TransformationSet(
                original_problem={"question": "Test question?", "answer": "Test answer"},
                transformations=[
                    TransformationResult(
                        original_problem={"question": "Test question?", "answer": "Test answer"},
                        transformed_problem={"question": "Modified question?", "answer": "Test answer"},
                        transformation_type="test_type",
                        transformation_metadata={},
                        success=True
                    )
                ],
                problem_id="test_problem"
            )
        ]
    
    @patch('scramblebench.evaluation.openrouter_runner.create_openrouter_client')
    def test_runner_initialization(self, mock_create_client, evaluation_config):
        """Test OpenRouter runner initialization."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        runner = OpenRouterEvaluationRunner(evaluation_config)
        
        assert "test/model" in runner.model_clients
        mock_create_client.assert_called_once()
    
    def test_prompt_creation(self, evaluation_config):
        """Test prompt creation from problems."""
        with patch('scramblebench.evaluation.openrouter_runner.create_openrouter_client'):
            runner = OpenRouterEvaluationRunner(evaluation_config)
            
            # Test multiple choice format
            problem = {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "4"
            }
            
            prompt = runner._create_prompt(problem)
            assert "What is 2+2?" in prompt
            assert "A. 3" in prompt
            assert "B. 4" in prompt
            assert "Answer:" in prompt
    
    @pytest.mark.asyncio
    @patch('scramblebench.evaluation.openrouter_runner.create_openrouter_client')
    async def test_evaluate_transformation_sets(self, mock_create_client, evaluation_config, sample_transformation_sets):
        """Test evaluating transformation sets."""
        # Mock client setup
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.metadata = {"response_time": 1.0, "total_tokens": 10}
        mock_response.error = None
        
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client
        
        runner = OpenRouterEvaluationRunner(evaluation_config)
        
        # Mock the async response generation
        async def mock_generate_response_async(client, prompt):
            return mock_response
        
        runner._generate_response_async = mock_generate_response_async
        
        results = await runner.evaluate_transformation_sets(
            sample_transformation_sets,
            include_original=True
        )
        
        # Should have results for original + transformed versions
        assert len(results) == 2  # original + 1 transformation
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert any(r.transformation_type == "original" for r in results)


class TestResultsManager:
    """Test results management."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results."""
        return EvaluationResults(
            results=[
                EvaluationResult(
                    problem_id="test_problem",
                    transformation_type="original",
                    model_name="test/model",
                    original_problem={"question": "Test?"},
                    transformed_problem=None,
                    model_response="Test response",
                    response_metadata={"response_time": 1.0},
                    success=True,
                    evaluation_time=1.0
                )
            ],
            config=EvaluationConfig(
                experiment_name="test_experiment",
                benchmark_paths=["test.json"],
                models=[ModelConfig(name="test/model", provider=ModelProvider.OPENROUTER)]
            )
        )
    
    def test_results_to_dataframe(self, sample_results):
        """Test converting results to DataFrame."""
        df = sample_results.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "problem_id" in df.columns
        assert "model_name" in df.columns
        assert "success" in df.columns
    
    def test_results_filtering(self, sample_results):
        """Test filtering results."""
        filtered = sample_results.filter_by_model("test/model")
        assert len(filtered.results) == 1
        
        filtered = sample_results.filter_by_transformation("original")
        assert len(filtered.results) == 1
        
        filtered = sample_results.filter_by_model("nonexistent")
        assert len(filtered.results) == 0
    
    def test_success_rate_calculation(self, sample_results):
        """Test success rate calculation."""
        rate = sample_results.get_success_rate()
        assert rate == 1.0
        
        rate = sample_results.get_success_rate(model_name="test/model")
        assert rate == 1.0
        
        rate = sample_results.get_success_rate(transformation_type="original")
        assert rate == 1.0
    
    def test_results_manager_save_load(self, sample_results):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_manager = ResultsManager(Path(temp_dir))
            
            # Save results
            saved_files = results_manager.save_results(
                sample_results,
                "test_experiment",
                format="json"
            )
            
            assert "json" in saved_files
            assert "config" in saved_files
            assert "metadata" in saved_files
            
            # Load results
            loaded_results = results_manager.load_results("test_experiment")
            
            assert loaded_results.config.experiment_name == sample_results.config.experiment_name
            assert len(loaded_results.results) == len(sample_results.results)


class TestMetricsCalculator:
    """Test metrics calculation."""
    
    @pytest.fixture
    def sample_results_with_ground_truth(self):
        """Sample results with ground truth for testing."""
        results = EvaluationResults(
            results=[
                EvaluationResult(
                    problem_id="problem_1",
                    transformation_type="original",
                    model_name="model_a",
                    original_problem={"question": "Test?", "answer": "correct"},
                    transformed_problem=None,
                    model_response="correct",
                    response_metadata={},
                    success=True,
                    evaluation_time=1.0
                ),
                EvaluationResult(
                    problem_id="problem_1",
                    transformation_type="original",
                    model_name="model_b",
                    original_problem={"question": "Test?", "answer": "correct"},
                    transformed_problem=None,
                    model_response="wrong",
                    response_metadata={},
                    success=True,
                    evaluation_time=1.0
                ),
                EvaluationResult(
                    problem_id="problem_1",
                    transformation_type="synonym_replacement",
                    model_name="model_a",
                    original_problem={"question": "Test?", "answer": "correct"},
                    transformed_problem={"question": "Exam?", "answer": "correct"},
                    model_response="correct",
                    response_metadata={},
                    success=True,
                    evaluation_time=1.0
                )
            ],
            config=EvaluationConfig(
                experiment_name="test_experiment",
                benchmark_paths=["test.json"],
                models=[
                    ModelConfig(name="model_a", provider=ModelProvider.OPENROUTER),
                    ModelConfig(name="model_b", provider=ModelProvider.OPENROUTER)
                ]
            )
        )
        
        ground_truth = {"problem_1": "correct"}
        
        return results, ground_truth
    
    def test_accuracy_metrics_calculation(self, sample_results_with_ground_truth):
        """Test accuracy metrics calculation."""
        results, ground_truth = sample_results_with_ground_truth
        
        calculator = MetricsCalculator()
        accuracy_metrics = calculator.calculate_accuracy_metrics(results, ground_truth)
        
        assert "model_a" in accuracy_metrics
        assert "model_b" in accuracy_metrics
        
        # Model A should have perfect accuracy
        assert accuracy_metrics["model_a"].exact_match == 1.0
        
        # Model B should have 0% accuracy
        assert accuracy_metrics["model_b"].exact_match == 0.0
    
    def test_robustness_metrics_calculation(self, sample_results_with_ground_truth):
        """Test robustness metrics calculation."""
        results, ground_truth = sample_results_with_ground_truth
        
        calculator = MetricsCalculator()
        robustness_metrics = calculator.calculate_robustness_metrics(results, ground_truth)
        
        assert "model_a" in robustness_metrics
        
        # Model A should have no degradation (1.0 -> 1.0)
        model_a_metrics = robustness_metrics["model_a"]
        assert model_a_metrics.avg_degradation == 0.0
        assert len(model_a_metrics.significant_degradations) == 0
    
    def test_statistical_tests(self, sample_results_with_ground_truth):
        """Test statistical significance tests."""
        results, ground_truth = sample_results_with_ground_truth
        
        calculator = MetricsCalculator()
        statistical_tests = calculator.calculate_statistical_tests(results, ground_truth)
        
        assert "model_a" in statistical_tests.pairwise_comparisons
        assert "model_b" in statistical_tests.pairwise_comparisons["model_a"]
        assert "model_a" in statistical_tests.confidence_intervals
        assert "model_b" in statistical_tests.confidence_intervals
    
    def test_metrics_report_generation(self, sample_results_with_ground_truth):
        """Test comprehensive metrics report generation."""
        results, ground_truth = sample_results_with_ground_truth
        
        calculator = MetricsCalculator()
        report = calculator.generate_metrics_report(results, ground_truth)
        
        assert "experiment_info" in report
        assert "accuracy_metrics" in report
        assert "robustness_metrics" in report
        assert "statistical_tests" in report


class TestPlotGenerator:
    """Test plot generation."""
    
    @pytest.fixture
    def sample_results_for_plotting(self):
        """Sample results suitable for plotting."""
        return EvaluationResults(
            results=[
                EvaluationResult(
                    problem_id="problem_1",
                    transformation_type="original",
                    model_name="model_a",
                    original_problem={},
                    transformed_problem=None,
                    model_response="response",
                    response_metadata={"response_time": 1.0, "total_tokens": 100},
                    success=True,
                    evaluation_time=1.0
                ),
                EvaluationResult(
                    problem_id="problem_1",
                    transformation_type="synonym_replacement",
                    model_name="model_a",
                    original_problem={},
                    transformed_problem={},
                    model_response="response",
                    response_metadata={"response_time": 1.2, "total_tokens": 110},
                    success=True,
                    evaluation_time=1.2
                )
            ],
            config=EvaluationConfig(
                experiment_name="test_experiment",
                benchmark_paths=["test.json"],
                models=[ModelConfig(name="model_a", provider=ModelProvider.OPENROUTER)]
            )
        )
    
    def test_plot_generator_creation(self):
        """Test creating plot generator."""
        generator = PlotGenerator()
        assert generator is not None
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_model_comparison_plot(self, mock_show, mock_savefig, sample_results_for_plotting):
        """Test model comparison plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = PlotGenerator()
            
            result = generator.plot_model_comparison(
                sample_results_for_plotting,
                Path(temp_dir)
            )
            
            assert result.success
            # Should have files for each format
            assert len(result.file_paths) >= 1


class TestEvaluationRunner:
    """Test the main evaluation runner."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return EvaluationConfig(
            experiment_name="test_runner",
            benchmark_paths=["test.json"],
            models=[
                ModelConfig(
                    name="test/model",
                    provider=ModelProvider.OPENROUTER,
                    api_key="test_key"
                )
            ],
            max_samples=5,
            generate_plots=False,
            calculate_significance=False
        )
    
    def test_runner_creation(self, sample_config):
        """Test creating evaluation runner."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = EvaluationRunner(sample_config, data_dir=Path(temp_dir))
            
            assert runner.config == sample_config
            assert runner.data_dir == Path(temp_dir)
    
    @patch('scramblebench.evaluation.runner.DataLoader')
    @patch('scramblebench.evaluation.openrouter_runner.create_openrouter_client')
    def test_runner_status(self, mock_create_client, mock_data_loader, sample_config):
        """Test runner status tracking."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = EvaluationRunner(sample_config, data_dir=Path(temp_dir))
            
            status = runner.get_status()
            
            assert status["current_stage"] == "initialized"
            assert status["experiment_name"] == "test_runner"
            assert status["configured_models"] == 1
    
    def test_config_file_operations(self, sample_config):
        """Test configuration file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = EvaluationRunner(sample_config, data_dir=Path(temp_dir))
            
            # Save config
            config_path = runner.save_config()
            assert config_path.exists()
            
            # Load from config file
            new_runner = EvaluationRunner.from_config_file(config_path, Path(temp_dir))
            assert new_runner.config.experiment_name == sample_config.experiment_name


# Integration tests
class TestEvaluationIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_mock(self):
        """Test end-to-end evaluation with mocked components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample benchmark file
            benchmark_data = [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"}
            ]
            
            benchmark_file = Path(temp_dir) / "test_benchmark.json"
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_data, f)
            
            # Create configuration
            config = EvaluationConfig(
                experiment_name="integration_test",
                benchmark_paths=[str(benchmark_file)],
                output_dir=temp_dir,
                models=[
                    ModelConfig(
                        name="test/model",
                        provider=ModelProvider.OPENROUTER,
                        api_key="test_key"
                    )
                ],
                transformations=TransformationConfig(
                    enabled_types=[TransformationType.SYNONYM_REPLACEMENT],
                    synonym_rate=0.3
                ),
                max_samples=2,
                generate_plots=False,
                calculate_significance=False
            )
            
            # Mock the OpenRouter client
            with patch('scramblebench.evaluation.openrouter_runner.create_openrouter_client') as mock_create_client:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.text = "4"  # Mock correct answer
                mock_response.metadata = {"response_time": 1.0, "total_tokens": 10}
                mock_response.error = None
                
                mock_client.generate.return_value = mock_response
                mock_create_client.return_value = mock_client
                
                # Create and run evaluation
                runner = EvaluationRunner(config, data_dir=Path(temp_dir))
                
                # Mock the async response generation
                async def mock_generate_response_async(client, prompt):
                    return mock_response
                
                # Patch the async method
                with patch.object(
                    OpenRouterEvaluationRunner,
                    '_generate_response_async',
                    mock_generate_response_async
                ):
                    results = await runner.run_evaluation()
                
                # Verify results
                assert isinstance(results, EvaluationResults)
                assert len(results.results) > 0
                assert results.config.experiment_name == "integration_test"
                
                # Verify output files were created
                experiment_dir = Path(temp_dir) / "integration_test"
                assert experiment_dir.exists()
                assert (experiment_dir / "config.yaml").exists()
                assert (experiment_dir / "results.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])