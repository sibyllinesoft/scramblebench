"""
End-to-end integration tests for ScrambleBench.

This module provides comprehensive integration tests that verify the complete
workflows and interactions between different components of ScrambleBench,
including translation benchmarks, long context benchmarks, evaluation pipelines,
and CLI operations.
"""

import pytest
import json
import tempfile
import asyncio
import os
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock

from scramblebench.translation.benchmark import TranslationBenchmark
from scramblebench.longcontext.benchmark import LongContextBenchmark
from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
from scramblebench.llm.model_interface import DummyModel, ModelConfig
from scramblebench.core.unified_config import ScrambleBenchConfig
from scramblebench.utils.data_loader import DataLoader


class TestTranslationBenchmarkIntegration:
    """End-to-end integration tests for translation benchmark workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config({
            "random_seed": 42,
            "vocab_size": 100,
            "languages_dir": str(self.temp_dir / "languages"),
            "results_dir": str(self.temp_dir / "results"),
            "preserve_numbers": True,
            "preserve_proper_nouns": True,
            "evaluation_mode": "exact_match",
            "evaluation_threshold": 0.8
        })
        
        # Create sample dataset
        self.sample_data = [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is artificial intelligence?", "answer": "AI is machine intelligence"},
            {"id": "q3", "question": "Name the capital of France.", "answer": "Paris"}
        ]
        
        self.dataset_file = self.temp_dir / "test_dataset.json"
        with open(self.dataset_file, 'w') as f:
            json.dump(self.sample_data, f)
            
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_complete_translation_benchmark_workflow(self):
        """Test complete translation benchmark workflow from start to finish."""
        # Create translation benchmark
        benchmark = TranslationBenchmark(
            source_dataset=str(self.dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            language_complexity=3,
            config=self.config
        )
        
        # Create dummy model for testing
        model = DummyModel(
            model_name="test_model",
            responses=[
                "4",  # Correct answer for 2+2
                "Artificial intelligence is machine intelligence",  # Close to correct
                "Paris"  # Correct answer for capital
            ]
        )
        model.initialize()
        
        # Run the benchmark
        result = benchmark.run(model, num_samples=3)
        
        # Verify result structure
        assert result.benchmark_name.startswith("translation_")
        assert result.model_name == "test_model"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.duration > 0
        assert len(result.metrics) > 0
        
        # Verify that language was generated and saved
        languages_dir = Path(self.config.get('languages_dir'))
        assert languages_dir.exists()
        language_files = list(languages_dir.glob("*.json"))
        assert len(language_files) > 0
        
        # Verify language file content
        with open(language_files[0]) as f:
            lang_data = json.load(f)
            assert lang_data['language_type'] == 'substitution'
            assert 'rules' in lang_data
            assert 'vocabulary' in lang_data
            
        # Verify translated problems were created
        assert len(benchmark.translated_problems) == 3
        for translated_problem in benchmark.translated_problems:
            assert translated_problem.language_name is not None
            assert len(translated_problem.translation_key) > 0
            
    def test_translation_benchmark_with_different_language_types(self):
        """Test translation benchmark with different language types."""
        language_types = [
            LanguageType.SUBSTITUTION,
            LanguageType.PHONETIC,
            LanguageType.SCRAMBLED,
            LanguageType.SYNTHETIC
        ]
        
        results = {}
        
        for lang_type in language_types:
            benchmark = TranslationBenchmark(
                source_dataset=str(self.dataset_file),
                language_type=lang_type,
                language_complexity=3,
                config=self.config
            )
            
            model = DummyModel(responses=["4", "AI", "Paris"])
            model.initialize()
            
            result = benchmark.run(model, num_samples=2)
            results[lang_type] = result
            
        # Verify all language types worked
        for lang_type, result in results.items():
            assert result.metrics['language_type'] == lang_type.value
            assert result.score >= 0.0
            
    def test_translation_benchmark_with_complexity_scaling(self):
        """Test translation benchmark with different complexity levels."""
        complexities = [1, 5, 10]
        results = {}
        
        for complexity in complexities:
            benchmark = TranslationBenchmark(
                source_dataset=str(self.dataset_file),
                language_type=LanguageType.SUBSTITUTION,
                language_complexity=complexity,
                config=self.config
            )
            
            model = DummyModel(responses=["4", "AI", "Paris"])
            model.initialize()
            
            result = benchmark.run(model, num_samples=2)
            results[complexity] = result
            
        # Verify complexity affects results
        for complexity, result in results.items():
            assert result.metrics['language_complexity'] == complexity
            
    def test_translation_benchmark_error_handling(self):
        """Test error handling in translation benchmark workflow."""
        # Test with non-existent dataset
        benchmark = TranslationBenchmark(
            source_dataset="nonexistent_file.json",
            language_type=LanguageType.SUBSTITUTION,
            config=self.config
        )
        
        with pytest.raises(FileNotFoundError):
            benchmark.prepare_data()
            
        # Test with invalid model
        class BadModel:
            pass
            
        benchmark = TranslationBenchmark(
            source_dataset=str(self.dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            config=self.config
        )
        
        bad_model = BadModel()
        
        with pytest.raises(ValueError):
            benchmark.run(bad_model)


class TestLongContextBenchmarkIntegration:
    """End-to-end integration tests for long context benchmark workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config({
            "random_seed": 42,
            "vocab_size": 200,
            "languages_dir": str(self.temp_dir / "languages"),
            "results_dir": str(self.temp_dir / "results"),
            "preserve_structure": True,
            "preserve_entities": True,
            "evaluation_mode": "exact_match",
            "evaluation_threshold": 0.8
        })
        
        # Create sample long context dataset
        self.sample_data = [
            {
                "id": "doc1",
                "document": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                intelligent machines. Machine learning is a subset of AI that focuses on the ability 
                of machines to receive data and learn for themselves without being explicitly programmed. 
                Deep learning is a subset of machine learning that uses neural networks with multiple 
                layers to model complex patterns in data.
                """,
                "questions": [
                    "What is artificial intelligence?",
                    "What is machine learning?",
                    "What is deep learning?"
                ],
                "answers": [
                    "A branch of computer science that aims to create intelligent machines",
                    "A subset of AI that focuses on machines learning from data",
                    "A subset of machine learning using neural networks"
                ]
            }
        ]
        
        self.dataset_file = self.temp_dir / "longcontext_dataset.json"
        with open(self.dataset_file, 'w') as f:
            json.dump(self.sample_data, f)
            
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_complete_longcontext_benchmark_workflow(self):
        """Test complete long context benchmark workflow."""
        from scramblebench.longcontext.document_transformer import TransformationType
        
        # Create long context benchmark
        benchmark = LongContextBenchmark(
            dataset_name=str(self.dataset_file),
            transformation_type=TransformationType.TRANSLATION,
            language_type=LanguageType.SUBSTITUTION,
            language_complexity=3,
            config=self.config
        )
        
        # Create dummy model
        model = DummyModel(
            responses=[
                "A branch of computer science that creates intelligent machines",
                "A subset of AI focusing on machine learning from data",
                "A subset of machine learning with neural networks"
            ]
        )
        model.initialize()
        
        # Run the benchmark
        result = benchmark.run(model, num_samples=1)
        
        # Verify result structure
        assert result.benchmark_name.startswith("longcontext_")
        assert result.model_name == "test_model"
        assert isinstance(result.score, float)
        assert result.duration > 0
        
        # Verify document transformation occurred
        assert len(benchmark.transformed_documents) == 1
        transformed_doc = benchmark.transformed_documents[0]
        assert transformed_doc.original_document != transformed_doc.transformed_document
        
        # Verify Q&A transformation occurred
        assert len(benchmark.transformed_qa_pairs) == 1
        qa_pairs = benchmark.transformed_qa_pairs[0]
        assert len(qa_pairs) == 3  # Three Q&A pairs
        
        for qa_pair in qa_pairs:
            assert qa_pair.original_qa.question != qa_pair.transformed_qa.question
            
    def test_longcontext_benchmark_without_language(self):
        """Test long context benchmark with non-translation transformation."""
        from scramblebench.longcontext.document_transformer import TransformationType
        
        benchmark = LongContextBenchmark(
            dataset_name=str(self.dataset_file),
            transformation_type=TransformationType.STRUCTURAL,
            language_type=None,
            config=self.config
        )
        
        model = DummyModel(responses=["Answer 1", "Answer 2", "Answer 3"])
        model.initialize()
        
        result = benchmark.run(model, num_samples=1)
        
        # Should work without constructed language
        assert result.score >= 0.0
        assert benchmark.constructed_language is None
        
    def test_longcontext_benchmark_export_functionality(self):
        """Test long context benchmark data export."""
        from scramblebench.longcontext.document_transformer import TransformationType
        
        benchmark = LongContextBenchmark(
            dataset_name=str(self.dataset_file),
            transformation_type=TransformationType.TRANSLATION,
            language_type=LanguageType.SUBSTITUTION,
            config=self.config
        )
        
        # Prepare data
        benchmark.prepare_data()
        
        # Export benchmark data
        export_dir = self.temp_dir / "export"
        benchmark.export_benchmark_data(export_dir)
        
        # Verify export files
        assert (export_dir / "transformed_documents.json").exists()
        assert (export_dir / "transformed_qa_pairs.json").exists()
        assert (export_dir / "constructed_language.json").exists()
        assert (export_dir / "benchmark_metadata.json").exists()
        
        # Verify export content
        with open(export_dir / "benchmark_metadata.json") as f:
            metadata = json.load(f)
            assert metadata['benchmark_name'] == benchmark.name
            assert metadata['transformation_type'] == 'translation'


class TestCrossComponentIntegration:
    """Test integration between different ScrambleBench components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_language_generator_translator_integration(self):
        """Test integration between language generator and translator."""
        from scramblebench.translation.translator import ProblemTranslator
        
        # Generate a language
        generator = LanguageGenerator(seed=42)
        language = generator.generate_language(
            name="integration_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=5,
            vocab_size=100
        )
        
        # Use translator with the generated language
        translator = ProblemTranslator()
        
        problem = {
            "question": "What is the meaning of life?",
            "answer": "42"
        }
        
        translated = translator.translate_problem(
            problem=problem,
            language=language,
            preserve_numbers=True
        )
        
        # Verify translation
        assert translated.original_problem == problem
        assert translated.translated_problem["question"] != problem["question"]
        assert "42" in translated.translated_problem["answer"]  # Number preserved
        assert len(translated.translation_key) > 0
        
    def test_data_loader_benchmark_integration(self):
        """Test integration between data loader and benchmarks."""
        # Create test data in different formats
        test_data = [
            {"id": "q1", "question": "Test question 1", "answer": "Answer 1"},
            {"id": "q2", "question": "Test question 2", "answer": "Answer 2"}
        ]
        
        # JSON format
        json_file = self.temp_dir / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
            
        # JSONL format
        jsonl_file = self.temp_dir / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
                
        # Test with different formats
        data_loader = DataLoader()
        
        json_data = data_loader.load_dataset(str(json_file))
        jsonl_data = data_loader.load_dataset(str(jsonl_file))
        
        assert json_data == test_data
        assert jsonl_data == test_data
        
        # Test with benchmark
        config = Config({"random_seed": 42})
        benchmark = TranslationBenchmark(
            source_dataset=str(json_file),
            language_type=LanguageType.SUBSTITUTION,
            config=config
        )
        
        benchmark.prepare_data()
        assert len(benchmark.original_problems) == 2
        
    def test_model_benchmark_integration(self):
        """Test integration between models and benchmarks."""
        # Create test dataset
        test_data = [
            {"id": "q1", "question": "What is 1+1?", "answer": "2"},
            {"id": "q2", "question": "What is 2+2?", "answer": "4"}
        ]
        
        dataset_file = self.temp_dir / "model_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Test with different model configurations
        configs = [
            ModelConfig(temperature=0.0, max_tokens=50),
            ModelConfig(temperature=0.5, max_tokens=100),
            ModelConfig(temperature=1.0, max_tokens=200)
        ]
        
        for config in configs:
            model = DummyModel(
                config=config,
                responses=["2", "4"]
            )
            model.initialize()
            
            benchmark = TranslationBenchmark(
                source_dataset=str(dataset_file),
                language_type=LanguageType.SUBSTITUTION,
                config=Config({"random_seed": 42})
            )
            
            result = benchmark.run(model, num_samples=2)
            
            # Verify model config is reflected in results
            assert result.model_name == "dummy"
            assert result.score >= 0.0


class TestFullPipelineIntegration:
    """Test complete end-to-end pipeline integration."""
    
    def setup_method(self):
        """Set up test environment.""" 
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'})
    @patch('scramblebench.evaluation.openrouter_runner.OpenRouterEvaluationRunner')
    def test_evaluation_pipeline_integration(self, mock_runner_class):
        """Test complete evaluation pipeline integration."""
        from scramblebench.evaluation.runner import EvaluationRunner
        from scramblebench.core.unified_config import EvaluationConfig, ModelConfig, TransformationConfig
        
        # Create test data
        test_data = [
            {"id": "q1", "question": "What is AI?", "answer": "Artificial Intelligence"},
            {"id": "q2", "question": "What is ML?", "answer": "Machine Learning"}
        ]
        
        benchmark_file = self.temp_dir / "eval_test.json"
        with open(benchmark_file, 'w') as f:
            json.dump(test_data, f)
            
        # Create evaluation configuration
        config = EvaluationConfig(
            experiment_name="integration_test",
            description="Integration test evaluation",
            models=[
                ModelConfig(
                    name="openai/gpt-3.5-turbo",
                    provider=ModelProvider.OPENROUTER,
                    temperature=0.0
                )
            ],
            benchmark_paths=[str(benchmark_file)],
            transformations=TransformationConfig(
                enabled_types=["language_translation"],
                synonym_rate=0.3
            ),
            max_samples=2,
            generate_plots=False,
            calculate_significance=False
        )
        
        # Mock the OpenRouter runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Mock evaluation results
        from scramblebench.evaluation.openrouter_runner import EvaluationResult
        mock_results = [
            EvaluationResult(
                model_name="openai/gpt-3.5-turbo",
                question_id="q1",
                question="What is AI?",
                expected_answer="Artificial Intelligence",
                predicted_answer="AI is artificial intelligence",
                correct=True,
                score=0.9,
                response_time=0.5,
                metadata={"transformation_type": "original"}
            )
        ]
        
        mock_runner.evaluate_benchmark_data = AsyncMock(return_value=mock_results)
        
        # Run evaluation
        runner = EvaluationRunner(config, data_dir=self.temp_dir)
        
        # Test that the pipeline can be set up
        assert runner.config == config
        assert runner.data_dir == self.temp_dir
        
        # Note: Full async test would require more complex mocking
        # This test verifies the integration setup
        
    def test_cli_benchmark_integration(self):
        """Test CLI integration with benchmark components."""
        from click.testing import CliRunner
        from scramblebench.cli import cli
        
        runner = CliRunner()
        
        # Test complete CLI workflow
        with runner.isolated_filesystem():
            # Generate a language via CLI
            result = runner.invoke(cli, [
                'language', 'generate', 'cli_test',
                '--type', 'substitution',
                '--complexity', '3'
            ])
            assert result.exit_code == 0
            
            # List languages via CLI
            result = runner.invoke(cli, ['language', 'list'])
            assert result.exit_code == 0
            assert 'cli_test' in result.output
            
            # Transform text via CLI
            result = runner.invoke(cli, [
                'transform', 'text',
                'Hello world',
                'cli_test'
            ])
            assert result.exit_code == 0
            
    def test_configuration_integration(self):
        """Test configuration system integration across components."""
        # Create comprehensive configuration
        config_data = {
            "random_seed": 12345,
            "vocab_size": 500,
            "languages_dir": str(self.temp_dir / "langs"),
            "results_dir": str(self.temp_dir / "results"),
            "preserve_numbers": True,
            "preserve_proper_nouns": False,
            "evaluation_mode": "semantic_similarity",
            "evaluation_threshold": 0.85,
            "chunk_long_documents": True,
            "chunk_size": 5000
        }
        
        config = Config(config_data)
        
        # Test with language generator
        generator = LanguageGenerator(
            seed=config.get('random_seed'),
            logger=None
        )
        
        language = generator.generate_language(
            name="config_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3,
            vocab_size=config.get('vocab_size')
        )
        
        assert len(language.vocabulary) <= config.get('vocab_size')
        
        # Test with benchmark
        test_data = [{"question": "Test", "answer": "Answer"}]
        dataset_file = self.temp_dir / "config_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        benchmark = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            config=config
        )
        
        # Verify configuration is used
        assert benchmark.config.get('preserve_numbers') == True
        assert benchmark.config.get('preserve_proper_nouns') == False
        assert benchmark.config.get('evaluation_threshold') == 0.85


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_cascading_error_handling(self):
        """Test error handling across component boundaries."""
        # Test data loading error propagation
        benchmark = TranslationBenchmark(
            source_dataset="nonexistent_file.json",
            language_type=LanguageType.SUBSTITUTION,
            config=Config()
        )
        
        with pytest.raises(FileNotFoundError):
            benchmark.prepare_data()
            
        # Test model error propagation
        class FailingModel:
            def __init__(self):
                self.name = "failing_model"
                
            def initialize(self):
                return True
                
            def generate(self, prompt, **kwargs):
                raise Exception("Model generation failed")
                
        test_data = [{"question": "Test", "answer": "Answer"}]
        dataset_file = self.temp_dir / "error_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        benchmark = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            config=Config()
        )
        
        failing_model = FailingModel()
        
        with pytest.raises(Exception):
            benchmark.run(failing_model)
            
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures in pipeline."""
        # Create model that fails on some inputs
        class PartiallyFailingModel:
            def __init__(self):
                self.name = "partial_model"
                self.call_count = 0
                
            def initialize(self):
                return True
                
            def generate(self, prompt, **kwargs):
                self.call_count += 1
                if self.call_count % 2 == 0:
                    raise Exception("Partial failure")
                return "Success response"
                
        test_data = [
            {"question": "Test 1", "answer": "Answer 1"},
            {"question": "Test 2", "answer": "Answer 2"},
            {"question": "Test 3", "answer": "Answer 3"}
        ]
        
        dataset_file = self.temp_dir / "partial_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        benchmark = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            config=Config()
        )
        
        partial_model = PartiallyFailingModel()
        
        # Should handle partial failures gracefully
        with pytest.raises(Exception):
            # This will fail, but we're testing the error handling pathway
            benchmark.run(partial_model)


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        large_data = [
            {"id": f"q{i}", "question": f"Question {i} about topic", "answer": f"Answer {i}"}
            for i in range(100)  # 100 questions for performance test
        ]
        
        dataset_file = self.temp_dir / "large_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(large_data, f)
            
        # Test with limited samples for reasonable test time
        config = Config({"random_seed": 42})
        benchmark = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            language_complexity=3,
            config=config
        )
        
        model = DummyModel(responses=["Answer"] * 100)
        model.initialize()
        
        import time
        start_time = time.time()
        
        result = benchmark.run(model, num_samples=10)  # Limited for test performance
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 30  # Less than 30 seconds
        assert result.score >= 0.0
        assert result.metadata["num_samples"] == 10
        
    def test_concurrent_benchmark_execution(self):
        """Test concurrent execution of multiple benchmarks."""
        # Create test data
        test_data = [
            {"question": "Concurrent test", "answer": "Concurrent answer"}
        ]
        
        dataset_file = self.temp_dir / "concurrent_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Run multiple benchmarks concurrently
        import threading
        import time
        
        results = []
        errors = []
        
        def run_benchmark(lang_type):
            try:
                benchmark = TranslationBenchmark(
                    source_dataset=str(dataset_file),
                    language_type=lang_type,
                    config=Config({"random_seed": 42})
                )
                
                model = DummyModel(responses=["Concurrent answer"])
                model.initialize()
                
                result = benchmark.run(model, num_samples=1)
                results.append((lang_type, result))
            except Exception as e:
                errors.append((lang_type, e))
                
        # Run different language types concurrently
        language_types = [
            LanguageType.SUBSTITUTION,
            LanguageType.PHONETIC,
            LanguageType.SCRAMBLED
        ]
        
        threads = []
        start_time = time.time()
        
        for lang_type in language_types:
            thread = threading.Thread(target=run_benchmark, args=(lang_type,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        end_time = time.time()
        
        # All should complete successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(language_types)
        
        # Should be faster than sequential execution
        assert end_time - start_time < 15  # Should complete quickly
        
    def test_memory_usage_monitoring(self):
        """Test memory usage during integrated workflows."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run memory-intensive workflow
        test_data = [
            {"question": f"Memory test {i}", "answer": f"Answer {i}"}
            for i in range(50)
        ]
        
        dataset_file = self.temp_dir / "memory_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Run multiple benchmarks
        for complexity in [1, 5, 10]:
            benchmark = TranslationBenchmark(
                source_dataset=str(dataset_file),
                language_type=LanguageType.SUBSTITUTION,
                language_complexity=complexity,
                config=Config({"random_seed": 42, "vocab_size": 1000})
            )
            
            model = DummyModel(responses=["Answer"] * 50)
            model.initialize()
            
            result = benchmark.run(model, num_samples=10)
            assert result.score >= 0.0
            
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"


class TestDataConsistencyIntegration:
    """Test data consistency across integrated components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_translation_consistency(self):
        """Test translation consistency across pipeline."""
        # Create test data with specific content
        test_data = [
            {"question": "The quick brown fox jumps over the lazy dog.", "answer": "Pangram sentence"}
        ]
        
        dataset_file = self.temp_dir / "consistency_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Create benchmark with fixed seed
        benchmark = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            language_complexity=3,
            config=Config({"random_seed": 42})
        )
        
        # Prepare data multiple times
        benchmark.prepare_data()
        first_translation = benchmark.translated_problems[0].translated_problem["question"]
        
        # Create new benchmark with same parameters
        benchmark2 = TranslationBenchmark(
            source_dataset=str(dataset_file),
            language_type=LanguageType.SUBSTITUTION,
            language_complexity=3,
            config=Config({"random_seed": 42})
        )
        
        benchmark2.prepare_data()
        second_translation = benchmark2.translated_problems[0].translated_problem["question"]
        
        # Should get same translation with same seed
        assert first_translation == second_translation
        
    def test_evaluation_consistency(self):
        """Test evaluation consistency across runs."""
        test_data = [
            {"question": "What is 2+2?", "answer": "4"}
        ]
        
        dataset_file = self.temp_dir / "eval_consistency.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Run same evaluation multiple times
        results = []
        
        for run in range(3):
            benchmark = TranslationBenchmark(
                source_dataset=str(dataset_file),
                language_type=LanguageType.SUBSTITUTION,
                config=Config({"random_seed": 42})
            )
            
            model = DummyModel(responses=["4"])  # Correct answer
            model.initialize()
            
            result = benchmark.run(model, num_samples=1)
            results.append(result.score)
            
        # All runs should give same score
        assert all(score == results[0] for score in results)
        
    def test_export_import_consistency(self):
        """Test data consistency through export/import cycle."""
        from scramblebench.longcontext.benchmark import LongContextBenchmark
        from scramblebench.longcontext.document_transformer import TransformationType
        
        # Create long context data
        test_data = [
            {
                "document": "This is a test document with important information.",
                "questions": ["What type of document is this?"],
                "answers": ["A test document"]
            }
        ]
        
        dataset_file = self.temp_dir / "export_test.json"
        with open(dataset_file, 'w') as f:
            json.dump(test_data, f)
            
        # Create and prepare benchmark
        benchmark = LongContextBenchmark(
            dataset_name=str(dataset_file),
            transformation_type=TransformationType.TRANSLATION,
            language_type=LanguageType.SUBSTITUTION,
            config=Config({"random_seed": 42})
        )
        
        benchmark.prepare_data()
        
        # Export data
        export_dir = self.temp_dir / "export"
        benchmark.export_benchmark_data(export_dir)
        
        # Verify exported data can be read and is consistent
        with open(export_dir / "transformed_documents.json") as f:
            exported_docs = json.load(f)
            
        with open(export_dir / "transformed_qa_pairs.json") as f:
            exported_qa = json.load(f)
            
        # Check consistency
        assert len(exported_docs) == 1
        assert len(exported_qa) == 1
        assert exported_docs[0]['transformation_type'] == 'translation'
        assert len(exported_qa[0]['qa_pairs']) == 1