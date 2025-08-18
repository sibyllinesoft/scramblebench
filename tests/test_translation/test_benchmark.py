"""
Tests for translation benchmark functionality.

This module provides comprehensive tests for the TranslationBenchmark class,
covering initialization, data preparation, problem translation, evaluation,
and result computation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, List

from scramblebench.translation.benchmark import TranslationBenchmark
from scramblebench.translation.language_generator import (
    LanguageGenerator, ConstructedLanguage, LanguageType, LanguageRule
)
from scramblebench.translation.translator import TranslatedProblem, TranslationUnit
from scramblebench.core.evaluator import EvaluationResult, EvaluationMode
from scramblebench.utils.config import Config


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name: str = "mock_model", responses: Dict[str, str] = None):
        self.name = name
        self.responses = responses or {}
        self.call_count = 0
        
    def generate(self, prompt: str) -> str:
        """Mock generation method."""
        self.call_count += 1
        # Return specific response if configured, otherwise echo
        return self.responses.get(prompt, f"Answer for: {prompt}")
        
    def query(self, prompt: str) -> str:
        """Alternative interface for querying."""
        return self.generate(prompt)


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: List[Dict[str, Any]] = None):
        self.data = data or [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is 3+3?", "answer": "6"},
            {"id": "q3", "question": "What is the capital of France?", "answer": "Paris"}
        ]
        
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Mock dataset loading."""
        return self.data.copy()


class MockLanguageGenerator:
    """Mock language generator for testing."""
    
    def __init__(self, seed: int = 42, logger=None):
        self.seed = seed
        self.logger = logger
        self.generated_languages = {}
        
    def generate_language(
        self, 
        name: str, 
        language_type: LanguageType,
        complexity: int = 5,
        vocab_size: int = 1000
    ) -> ConstructedLanguage:
        """Mock language generation."""
        rules = [
            LanguageRule("a", "æ", "character", 1),
            LanguageRule("e", "ə", "character", 1),
            LanguageRule("hello", "halo", "word", 10)
        ]
        
        vocabulary = {"hello": "halo", "world": "warld", "what": "whæt"}
        
        language = ConstructedLanguage(
            name=name,
            language_type=language_type,
            rules=rules,
            vocabulary=vocabulary,
            metadata={"complexity": str(complexity)}
        )
        
        self.generated_languages[name] = language
        return language
        
    def save_language(self, language: ConstructedLanguage, path: Path) -> None:
        """Mock language saving."""
        pass


class MockProblemTranslator:
    """Mock problem translator for testing."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def translate_problem(
        self,
        problem: Dict[str, Any],
        language: ConstructedLanguage,
        preserve_numbers: bool = True,
        preserve_proper_nouns: bool = True
    ) -> TranslatedProblem:
        """Mock problem translation."""
        # Simple mock translation - replace 'e' with 'æ'
        translated_problem = {}
        translation_key = {}
        translation_units = []
        
        for key, value in problem.items():
            if isinstance(value, str):
                translated_value = value.replace("e", "æ").replace("hello", "halo")
                translated_problem[key] = translated_value
                if value != translated_value:
                    translation_key[value] = translated_value
                    translation_units.append(
                        TranslationUnit(
                            original=value,
                            translated=translated_value,
                            rule_type="character",
                            confidence=1.0
                        )
                    )
            else:
                translated_problem[key] = value
                
        return TranslatedProblem(
            original_problem=problem,
            translated_problem=translated_problem,
            translation_key=translation_key,
            translation_units=translation_units,
            language_name=language.name
        )
        
    def translate_answer(
        self,
        answer: str,
        translation_key: Dict[str, str],
        reverse: bool = False
    ) -> str:
        """Mock answer translation."""
        if reverse:
            # Translate back from constructed language
            return answer.replace("æ", "e").replace("halo", "hello")
        else:
            # Translate to constructed language
            return answer.replace("e", "æ").replace("hello", "halo")
            
    def verify_translation_consistency(
        self,
        translated_problem: TranslatedProblem
    ) -> Dict[str, Any]:
        """Mock translation verification."""
        return {"consistent": True, "issues": []}


class MockEvaluator:
    """Mock evaluator for testing."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def extract_answer_pattern(self, response: str) -> str:
        """Mock answer extraction."""
        # Simple extraction - return last word
        return response.strip().split()[-1] if response.strip() else response
        
    def evaluate_response(
        self,
        predicted: str,
        expected: str,
        mode: EvaluationMode,
        threshold: float = 0.8
    ) -> EvaluationResult:
        """Mock response evaluation."""
        correct = predicted.strip().lower() == expected.strip().lower()
        score = 1.0 if correct else 0.0
        
        return EvaluationResult(
            correct=correct,
            score=score,
            explanation=f"Predicted '{predicted}' vs expected '{expected}'",
            metadata={"mode": mode.value}
        )


# Fixtures
@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return Config({
        "random_seed": 42,
        "vocab_size": 100,
        "languages_dir": "test_languages",
        "results_dir": "test_results",
        "preserve_numbers": True,
        "preserve_proper_nouns": True,
        "evaluation_mode": "exact_match",
        "evaluation_threshold": 0.8
    })

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture  
def mock_benchmark(mock_config):
    """Create a mock translation benchmark."""
    benchmark = TranslationBenchmark(
        source_dataset="test_dataset",
        language_type=LanguageType.SUBSTITUTION,
        language_complexity=5,
        config=mock_config
    )
    
    # Replace components with mocks
    benchmark.data_loader = MockDataLoader()
    benchmark.language_generator = MockLanguageGenerator()
    benchmark.translator = MockProblemTranslator()
    benchmark.evaluator = MockEvaluator()
    
    return benchmark

@pytest.fixture
def perfect_model():
    """Create a model that gives perfect answers."""
    return MockModel(
        responses={
            "Whæt is 2+2?": "4",
            "Whæt is 3+3?": "6", 
            "Whæt is thæ cæpitæl of Fræncæ?": "Paris"
        }
    )

@pytest.fixture
def sample_translated_problems():
    """Create sample translated problems."""
    problems = []
    for i in range(3):
        problems.append(TranslatedProblem(
            original_problem={"id": f"q{i+1}", "question": f"Question {i+1}", "answer": f"Answer {i+1}"},
            translated_problem={"id": f"q{i+1}", "question": f"Qwæstion {i+1}", "answer": f"Answer {i+1}"},
            translation_key={"Question": "Qwæstion", "e": "æ"},
            translation_units=[
                TranslationUnit("Question", "Qwæstion", "word", 1.0),
                TranslationUnit("e", "æ", "character", 1.0)
            ],
            language_name="test_language"
        ))
    return problems


class TestTranslationBenchmarkInitialization:
    """Test translation benchmark initialization."""
    
    def test_basic_initialization(self):
        """Test basic benchmark initialization."""
        benchmark = TranslationBenchmark("test_dataset")
        
        assert benchmark.source_dataset == "test_dataset"
        assert benchmark.language_type == LanguageType.SUBSTITUTION
        assert benchmark.language_complexity == 5
        assert benchmark.name == "translation_test_dataset_substitution"
        
    def test_initialization_with_parameters(self, mock_config):
        """Test initialization with specific parameters."""
        benchmark = TranslationBenchmark(
            source_dataset="custom_dataset",
            language_type=LanguageType.PHONETIC,
            language_complexity=8,
            config=mock_config
        )
        
        assert benchmark.source_dataset == "custom_dataset"
        assert benchmark.language_type == LanguageType.PHONETIC
        assert benchmark.language_complexity == 8
        assert benchmark.config is mock_config
        
    def test_components_initialization(self, mock_benchmark):
        """Test that all components are properly initialized."""
        assert mock_benchmark.language_generator is not None
        assert mock_benchmark.translator is not None
        assert mock_benchmark.evaluator is not None
        assert mock_benchmark.data_loader is not None
        
    def test_initial_state(self, mock_benchmark):
        """Test initial benchmark state."""
        assert mock_benchmark.constructed_language is None
        assert mock_benchmark.translated_problems == []
        assert mock_benchmark.original_problems == []


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_loads_original_problems(self, mock_benchmark):
        """Test that prepare_data loads original problems."""
        mock_benchmark.prepare_data()
        
        assert len(mock_benchmark.original_problems) == 3
        assert mock_benchmark.original_problems[0]["question"] == "What is 2+2?"
        
    def test_prepare_data_generates_language(self, mock_benchmark):
        """Test that prepare_data generates constructed language."""
        mock_benchmark.prepare_data()
        
        assert mock_benchmark.constructed_language is not None
        assert mock_benchmark.constructed_language.language_type == LanguageType.SUBSTITUTION
        
    def test_prepare_data_translates_problems(self, mock_benchmark):
        """Test that prepare_data translates all problems."""
        mock_benchmark.prepare_data()
        
        assert len(mock_benchmark.translated_problems) == 3
        assert all(isinstance(p, TranslatedProblem) for p in mock_benchmark.translated_problems)
        
    def test_prepare_data_calls_verification(self, mock_benchmark):
        """Test that prepare_data calls translation verification."""
        with patch.object(mock_benchmark, '_verify_translations') as mock_verify:
            mock_benchmark.prepare_data()
            
        mock_verify.assert_called_once()
        
    def test_prepare_data_saves_language(self, mock_benchmark, temp_dir):
        """Test that prepare_data saves the generated language."""
        mock_benchmark.config.data["languages_dir"] = str(temp_dir)
        
        with patch.object(mock_benchmark.language_generator, 'save_language') as mock_save:
            mock_benchmark.prepare_data()
            
        mock_save.assert_called_once()
        
    def test_prepare_data_logging(self, mock_benchmark):
        """Test that prepare_data logs appropriate messages."""
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark.prepare_data()
        
        mock_logger.info.assert_has_calls([
            call("Preparing translation benchmark data for test_dataset"),
            call("Loaded 3 problems"),
            call("Translated 3 problems")
        ])


class TestEvaluationDataHandling:
    """Test evaluation data handling."""
    
    def test_get_evaluation_data_returns_all_problems(self, mock_benchmark, sample_translated_problems):
        """Test getting all evaluation data."""
        mock_benchmark.translated_problems = sample_translated_problems
        
        data = mock_benchmark.get_evaluation_data()
        
        assert len(data) == 3
        assert data == sample_translated_problems
        
    def test_get_evaluation_data_limits_samples(self, mock_benchmark, sample_translated_problems):
        """Test limiting number of samples."""
        mock_benchmark.translated_problems = sample_translated_problems
        
        data = mock_benchmark.get_evaluation_data(num_samples=2)
        
        assert len(data) == 2
        assert data == sample_translated_problems[:2]
        
    def test_get_evaluation_data_handles_zero_samples(self, mock_benchmark, sample_translated_problems):
        """Test handling zero samples."""
        mock_benchmark.translated_problems = sample_translated_problems
        
        data = mock_benchmark.get_evaluation_data(num_samples=0)
        
        assert len(data) == 0
        
    def test_get_evaluation_data_handles_oversized_request(self, mock_benchmark, sample_translated_problems):
        """Test handling request for more samples than available."""
        mock_benchmark.translated_problems = sample_translated_problems
        
        data = mock_benchmark.get_evaluation_data(num_samples=100)
        
        assert len(data) == 3  # All available


class TestSingleEvaluation:
    """Test single problem evaluation."""
    
    def test_run_single_evaluation_structure(self, mock_benchmark, perfect_model, sample_translated_problems):
        """Test that single evaluation returns proper structure."""
        mock_benchmark.prepare_data()
        
        result = mock_benchmark.run_single_evaluation(perfect_model, sample_translated_problems[0])
        
        required_fields = [
            'problem_id', 'translated_problem', 'original_answer',
            'model_response', 'extracted_answer', 'translated_answer',
            'correct', 'score', 'evaluation_explanation', 'response_time',
            'translation_key', 'language_name', 'metadata'
        ]
        
        for field in required_fields:
            assert field in result
            
    def test_run_single_evaluation_extracts_problem_text(self, mock_benchmark, perfect_model):
        """Test problem text extraction."""
        mock_benchmark.prepare_data()
        
        with patch.object(mock_benchmark, '_extract_problem_text', return_value="test problem") as mock_extract:
            translated_problem = sample_translated_problems()[0]
            mock_benchmark.run_single_evaluation(perfect_model, translated_problem)
            
        mock_extract.assert_called_once_with(translated_problem.translated_problem)
        
    def test_run_single_evaluation_queries_model(self, mock_benchmark, perfect_model):
        """Test that model is queried with problem text."""
        mock_benchmark.prepare_data()
        
        with patch.object(mock_benchmark, '_query_model', return_value="test response") as mock_query:
            translated_problem = sample_translated_problems()[0]
            mock_benchmark.run_single_evaluation(perfect_model, translated_problem)
            
        mock_query.assert_called_once()
        
    def test_run_single_evaluation_measures_response_time(self, mock_benchmark, perfect_model, sample_translated_problems):
        """Test that response time is measured."""
        mock_benchmark.prepare_data()
        
        result = mock_benchmark.run_single_evaluation(perfect_model, sample_translated_problems[0])
        
        assert 'response_time' in result
        assert isinstance(result['response_time'], (int, float))
        assert result['response_time'] >= 0
        
    def test_run_single_evaluation_translates_answer_back(self, mock_benchmark, perfect_model):
        """Test that model answer is translated back to original language."""
        mock_benchmark.prepare_data()
        
        with patch.object(mock_benchmark.translator, 'translate_answer', return_value="translated back") as mock_translate:
            translated_problem = sample_translated_problems()[0]
            result = mock_benchmark.run_single_evaluation(perfect_model, translated_problem)
            
        mock_translate.assert_called_once()
        args = mock_translate.call_args
        assert args[1]['reverse'] is True
        
    def test_run_single_evaluation_evaluates_answer(self, mock_benchmark, perfect_model):
        """Test that answer is properly evaluated."""
        mock_benchmark.prepare_data()
        
        with patch.object(mock_benchmark.evaluator, 'evaluate_response') as mock_evaluate:
            mock_evaluate.return_value = EvaluationResult(True, 1.0, "correct", {})
            
            translated_problem = sample_translated_problems()[0]
            result = mock_benchmark.run_single_evaluation(perfect_model, translated_problem)
            
        mock_evaluate.assert_called_once()
        assert result['correct'] is True
        assert result['score'] == 1.0


class TestTextExtraction:
    """Test text extraction utilities."""
    
    def test_extract_problem_text_question_field(self, mock_benchmark):
        """Test extracting problem text from question field."""
        problem = {"question": "What is 2+2?", "other": "ignored"}
        
        text = mock_benchmark._extract_problem_text(problem)
        
        assert text == "What is 2+2?"
        
    def test_extract_problem_text_problem_field(self, mock_benchmark):
        """Test extracting problem text from problem field."""
        problem = {"problem": "Solve this equation", "other": "ignored"}
        
        text = mock_benchmark._extract_problem_text(problem)
        
        assert text == "Solve this equation"
        
    def test_extract_problem_text_fallback_concatenation(self, mock_benchmark):
        """Test fallback to concatenating all string values."""
        problem = {"field1": "part1", "field2": "part2", "number": 42}
        
        text = mock_benchmark._extract_problem_text(problem)
        
        assert "part1" in text
        assert "part2" in text
        assert "42" not in text  # Non-string values ignored
        
    def test_extract_expected_answer_answer_field(self, mock_benchmark):
        """Test extracting expected answer from answer field."""
        problem = {"answer": "42", "question": "What is the answer?"}
        
        answer = mock_benchmark._extract_expected_answer(problem)
        
        assert answer == "42"
        
    def test_extract_expected_answer_solution_field(self, mock_benchmark):
        """Test extracting expected answer from solution field."""
        problem = {"solution": "The answer is 42", "question": "What?"}
        
        answer = mock_benchmark._extract_expected_answer(problem)
        
        assert answer == "The answer is 42"
        
    def test_extract_expected_answer_numeric_conversion(self, mock_benchmark):
        """Test converting numeric answers to strings."""
        problem = {"answer": 42}
        
        answer = mock_benchmark._extract_expected_answer(problem)
        
        assert answer == "42"
        
    def test_extract_expected_answer_fallback(self, mock_benchmark):
        """Test fallback when no answer field found."""
        problem = {"question": "What is this?"}
        
        answer = mock_benchmark._extract_expected_answer(problem)
        
        assert answer == "unknown"


class TestModelQuerying:
    """Test model querying functionality."""
    
    def test_query_model_generate_interface(self, mock_benchmark):
        """Test querying model with generate interface."""
        model = MockModel()
        
        response = mock_benchmark._query_model(model, "test question")
        
        assert response == "Answer for: test question"
        
    def test_query_model_query_interface(self, mock_benchmark):
        """Test querying model with query interface."""
        class QueryModel:
            def query(self, text):
                return f"Query response: {text}"
                
        model = QueryModel()
        
        response = mock_benchmark._query_model(model, "test question")
        
        assert response == "Query response: test question"
        
    def test_query_model_callable_interface(self, mock_benchmark):
        """Test querying callable model."""
        class CallableModel:
            def __call__(self, text):
                return f"Callable response: {text}"
                
        model = CallableModel()
        
        response = mock_benchmark._query_model(model, "test question")
        
        assert response == "Callable response: test question"
        
    def test_query_model_no_interface_error(self, mock_benchmark):
        """Test error when model has no recognized interface."""
        class BadModel:
            pass
            
        model = BadModel()
        
        with pytest.raises(ValueError, match="does not have a recognized interface"):
            mock_benchmark._query_model(model, "test question")


class TestMetricsComputation:
    """Test metrics computation."""
    
    def test_compute_metrics_perfect_accuracy(self, mock_benchmark):
        """Test metrics computation with perfect accuracy."""
        results = [
            {'correct': True, 'score': 1.0, 'response_time': 0.5, 'metadata': {'translation_units': 5, 'language_complexity': 3}},
            {'correct': True, 'score': 1.0, 'response_time': 0.6, 'metadata': {'translation_units': 4, 'language_complexity': 3}},
            {'correct': True, 'score': 1.0, 'response_time': 0.4, 'metadata': {'translation_units': 6, 'language_complexity': 3}}
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['score'] == 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['correct_answers'] == 3
        assert metrics['total_questions'] == 3
        assert metrics['avg_response_time'] == 0.5
        assert metrics['avg_translation_units'] == 5.0
        
    def test_compute_metrics_partial_accuracy(self, mock_benchmark):
        """Test metrics computation with partial accuracy."""
        results = [
            {'correct': True, 'score': 1.0, 'response_time': 0.5, 'metadata': {'translation_units': 5, 'language_complexity': 3}},
            {'correct': False, 'score': 0.0, 'response_time': 0.6, 'metadata': {'translation_units': 4, 'language_complexity': 3}},
            {'correct': True, 'score': 1.0, 'response_time': 0.4, 'metadata': {'translation_units': 6, 'language_complexity': 3}}
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['score'] == 2/3
        assert metrics['accuracy'] == 2/3
        assert metrics['correct_answers'] == 2
        assert metrics['total_questions'] == 3
        
    def test_compute_metrics_empty_results(self, mock_benchmark):
        """Test metrics computation with empty results."""
        metrics = mock_benchmark.compute_metrics([])
        
        assert metrics['score'] == 0.0
        
    def test_compute_metrics_includes_benchmark_info(self, mock_benchmark):
        """Test that metrics include benchmark-specific information."""
        results = [
            {'correct': True, 'score': 1.0, 'response_time': 0.5, 'metadata': {'translation_units': 5, 'language_complexity': 5}}
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['language_type'] == 'substitution'
        assert metrics['language_complexity'] == 5
        assert metrics['source_dataset'] == 'test_dataset'
        
    def test_compute_metrics_complexity_performance_analysis(self, mock_benchmark):
        """Test complexity performance analysis in metrics."""
        results = [
            {'correct': True, 'score': 1.0, 'response_time': 0.5, 'metadata': {'translation_units': 5, 'language_complexity': 3}},
            {'correct': True, 'score': 1.0, 'response_time': 0.6, 'metadata': {'translation_units': 4, 'language_complexity': 3}},
            {'correct': False, 'score': 0.0, 'response_time': 0.4, 'metadata': {'translation_units': 6, 'language_complexity': 5}}
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert 'complexity_performance' in metrics
        assert 3 in metrics['complexity_performance']
        assert 5 in metrics['complexity_performance']
        assert metrics['complexity_performance'][3] == 1.0  # Perfect for complexity 3
        assert metrics['complexity_performance'][5] == 0.0  # Poor for complexity 5


class TestTranslationVerification:
    """Test translation verification functionality."""
    
    def test_verify_translations_successful(self, mock_benchmark, sample_translated_problems):
        """Test successful translation verification."""
        mock_benchmark.translated_problems = sample_translated_problems
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark._verify_translations()
        
        mock_logger.info.assert_called_with("All translations verified successfully")
        
    def test_verify_translations_with_issues(self, mock_benchmark):
        """Test translation verification with issues."""
        # Mock translator to return issues
        mock_benchmark.translator.verify_translation_consistency = MagicMock(
            return_value={"consistent": False, "issues": ["test issue"]}
        )
        
        mock_benchmark.translated_problems = sample_translated_problems()
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark._verify_translations()
        
        mock_logger.warning.assert_called()


class TestTranslationKeyExport:
    """Test translation key export functionality."""
    
def test_export_translation_key_success(self, mock_benchmark, temp_dir, sample_translated_problems):
        """Test successful translation key export."""
        mock_benchmark.translated_problems = sample_translated_problems
        mock_benchmark.constructed_language = ConstructedLanguage(
            name="test_lang",
            language_type=LanguageType.SUBSTITUTION,
            rules=[LanguageRule("a", "æ", "character", 1)],
            vocabulary={"hello": "halo"},
            metadata={}
        )
        
        output_path = temp_dir / "translation_key.json"
        mock_benchmark.export_translation_key(output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        assert 'language_name' in data
        assert 'translation_key' in data
        assert 'language_rules' in data
        assert 'vocabulary' in data
        assert 'statistics' in data
        
    def test_export_translation_key_no_problems(self, mock_benchmark, temp_dir):
        """Test export when no translated problems available."""
        mock_benchmark.translated_problems = []
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        output_path = temp_dir / "empty_key.json"
        mock_benchmark.export_translation_key(output_path)
        
        mock_logger.warning.assert_called_with("No translated problems available for export")
        assert not output_path.exists()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_config_success(self, mock_benchmark):
        """Test successful config validation."""
        assert mock_benchmark.validate_config() is True
        
    def test_validate_config_missing_required_field(self, mock_benchmark):
        """Test config validation with missing required field."""
        # Remove required field
        del mock_benchmark.config.data['random_seed']
        
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        assert mock_benchmark.validate_config() is False
        mock_logger.error.assert_called_with("Missing required config field: random_seed")
        
    def test_validate_config_invalid_complexity(self, mock_benchmark):
        """Test config validation with invalid complexity."""
        mock_benchmark.language_complexity = 15  # Invalid (> 10)
        
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        assert mock_benchmark.validate_config() is False
        mock_logger.error.assert_called_with("Language complexity must be 1-10, got 15")
        
    def test_validate_config_complexity_too_low(self, mock_benchmark):
        """Test config validation with complexity too low."""
        mock_benchmark.language_complexity = 0  # Invalid (< 1)
        
        assert mock_benchmark.validate_config() is False


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_full_benchmark_run_workflow(self, mock_benchmark, perfect_model):
        """Test complete benchmark execution workflow."""
        result = mock_benchmark.run(perfect_model)
        
        # Check that all components were used
        assert len(mock_benchmark.original_problems) == 3
        assert mock_benchmark.constructed_language is not None
        assert len(mock_benchmark.translated_problems) == 3
        
        # Check result structure
        assert result.benchmark_name == "translation_test_dataset_substitution"
        assert result.model_name == "mock_model"
        assert isinstance(result.score, float)
        assert result.duration > 0
        
    def test_benchmark_run_with_limited_samples(self, mock_benchmark, perfect_model):
        """Test benchmark run with limited samples."""
        result = mock_benchmark.run(perfect_model, num_samples=2)
        
        assert result.metadata["num_samples"] == 2
        
    def test_benchmark_state_after_run(self, mock_benchmark, perfect_model):
        """Test benchmark state after successful run."""
        mock_benchmark.run(perfect_model)
        
        # State should be properly maintained
        assert len(mock_benchmark.results) == 1
        assert mock_benchmark.constructed_language is not None
        assert len(mock_benchmark.translated_problems) == 3


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_model_interface_error_propagation(self, mock_benchmark):
        """Test that model interface errors are properly propagated."""
        class BadModel:
            pass
            
        bad_model = BadModel()
        
        with pytest.raises(ValueError):
            mock_benchmark.run(bad_model)
            
    def test_empty_dataset_handling(self, mock_benchmark, perfect_model):
        """Test handling of empty datasets."""
        mock_benchmark.data_loader.data = []
        
        result = mock_benchmark.run(perfect_model)
        
        assert result.score == 0.0
        assert result.metrics["total_questions"] == 0
        
    def test_translation_failure_resilience(self, mock_benchmark, perfect_model):
        """Test resilience to translation failures."""
        # Mock translator to fail on some problems
        def failing_translate(problem, language, **kwargs):
            if "2+2" in str(problem):
                raise Exception("Translation failed")
            return mock_benchmark.translator.translate_problem(problem, language, **kwargs)
            
        original_translate = mock_benchmark.translator.translate_problem
        mock_benchmark.translator.translate_problem = failing_translate
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            mock_benchmark.run(perfect_model)


class TestPerformanceAndScaling:
    """Test performance and scaling considerations."""
    
    def test_large_dataset_simulation(self, mock_config):
        """Test with simulated large dataset."""
        # Create large mock dataset
        large_data = [{"id": f"q{i}", "question": f"Question {i}", "answer": f"Answer {i}"} 
                     for i in range(100)]
        
        benchmark = TranslationBenchmark("large_dataset", config=mock_config)
        benchmark.data_loader = MockDataLoader(large_data)
        benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.evaluator = MockEvaluator()
        
        # Should handle large datasets
        benchmark.prepare_data()
        
        assert len(benchmark.original_problems) == 100
        assert len(benchmark.translated_problems) == 100
        
    def test_memory_efficiency_with_samples(self, mock_benchmark, perfect_model):
        """Test memory efficiency when using limited samples."""
        # Run with limited samples
        result = mock_benchmark.run(perfect_model, num_samples=2)
        
        # Should only process requested samples
        assert result.metadata["num_samples"] == 2
        assert perfect_model.call_count == 2  # Only called for 2 samples


class TestParameterizedTesting:
    """Test with different parameter combinations."""
    
    @pytest.mark.parametrize("language_type", [
        LanguageType.SUBSTITUTION,
        LanguageType.PHONETIC,
        LanguageType.SCRAMBLED,
        LanguageType.SYNTHETIC
    ])
    def test_all_language_types(self, language_type, mock_config, perfect_model):
        """Test benchmark with all language types."""
        benchmark = TranslationBenchmark(
            "test_dataset",
            language_type=language_type,
            config=mock_config
        )
        
        # Replace with mocks
        benchmark.data_loader = MockDataLoader()
        benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.evaluator = MockEvaluator()
        
        result = benchmark.run(perfect_model)
        
        assert result.metrics["language_type"] == language_type.value
        
    @pytest.mark.parametrize("complexity", [1, 3, 5, 7, 10])
    def test_complexity_levels(self, complexity, mock_config, perfect_model):
        """Test benchmark with different complexity levels."""
        benchmark = TranslationBenchmark(
            "test_dataset",
            language_complexity=complexity,
            config=mock_config
        )
        
        # Replace with mocks
        benchmark.data_loader = MockDataLoader()
        benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.evaluator = MockEvaluator()
        
        result = benchmark.run(perfect_model)
        
        assert result.metrics["language_complexity"] == complexity
        
    @pytest.mark.parametrize("num_samples", [1, 2, 3, None])
    def test_sample_sizes(self, num_samples, mock_benchmark, perfect_model):
        """Test benchmark with different sample sizes."""
        result = mock_benchmark.run(perfect_model, num_samples=num_samples)
        
        expected_samples = 3 if num_samples is None else min(num_samples, 3)
        assert result.metadata["num_samples"] == expected_samples