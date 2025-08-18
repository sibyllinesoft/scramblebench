"""
Tests for long context benchmark functionality.

This module provides comprehensive tests for the LongContextBenchmark class,
covering document transformation, Q&A pair handling, evaluation, and
various transformation strategies.
"""

import pytest
import json
import tempfile
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, List, Tuple

from scramblebench.longcontext.benchmark import LongContextBenchmark
from scramblebench.longcontext.document_transformer import (
    DocumentTransformer, TransformedDocument, TransformationType
)
from scramblebench.longcontext.qa_transformer import (
    QATransformer, QAPair, TransformedQAPair, AnswerType
)
from scramblebench.translation.language_generator import (
    LanguageGenerator, ConstructedLanguage, LanguageType, LanguageRule
)
from scramblebench.translation.translator import ProblemTranslator
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
        # Look for specific responses based on question content
        for key, response in self.responses.items():
            if key in prompt:
                return response
        # Default response
        return "Default answer"


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: List[Dict[str, Any]] = None):
        self.data = data or [
            {
                "id": "doc1",
                "document": "This is a sample document about artificial intelligence. AI has many applications in modern technology. Machine learning is a subset of AI that focuses on pattern recognition.",
                "questions": ["What is AI?", "What is machine learning?"],
                "answers": ["Artificial intelligence", "A subset of AI that focuses on pattern recognition"]
            },
            {
                "id": "doc2", 
                "document": "The capital of France is Paris. Paris is known for the Eiffel Tower and the Louvre Museum. It has a population of over 2 million people.",
                "questions": ["What is the capital of France?", "What is Paris known for?"],
                "answers": ["Paris", "The Eiffel Tower and the Louvre Museum"]
            }
        ]
        
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Mock dataset loading."""
        return self.data.copy()


class MockLanguageGenerator:
    """Mock language generator for testing."""
    
    def __init__(self, seed: int = 42, logger=None):
        self.seed = seed
        self.logger = logger
        
    def generate_language(
        self, 
        name: str, 
        language_type: LanguageType,
        complexity: int = 5,
        vocab_size: int = 2000
    ) -> ConstructedLanguage:
        """Mock language generation."""
        rules = [
            LanguageRule("a", "æ", "character", 1),
            LanguageRule("e", "ə", "character", 1),
            LanguageRule("intelligence", "intəlligəncə", "word", 10)
        ]
        
        vocabulary = {
            "intelligence": "intəlligəncə",
            "artificial": "ærtificiæl", 
            "machine": "mæchinə",
            "learning": "ləærning"
        }
        
        return ConstructedLanguage(
            name=name,
            language_type=language_type,
            rules=rules,
            vocabulary=vocabulary,
            metadata={"complexity": str(complexity)}
        )
        
    def save_language(self, language: ConstructedLanguage, path: Path) -> None:
        """Mock language saving."""
        pass


class MockDocumentTransformer:
    """Mock document transformer for testing."""
    
    def __init__(self, translator=None, logger=None):
        self.translator = translator
        self.logger = logger
        
    def transform_document(
        self,
        document: str,
        transformation_type: TransformationType,
        language: ConstructedLanguage = None,
        preserve_structure: bool = True,
        preserve_entities: bool = True
    ) -> TransformedDocument:
        """Mock document transformation."""
        # Simple transformation - replace 'e' with 'æ'
        transformed = document.replace("e", "æ")
        
        return TransformedDocument(
            original_document=document,
            transformed_document=transformed,
            transformation_type=transformation_type,
            transformation_map={"e": "æ"},
            metadata={"document_id": "test_doc", "preserved_entities": []}
        )
        
    def get_transformation_stats(self, doc: TransformedDocument) -> Dict[str, Any]:
        """Mock transformation stats."""
        return {
            "length_ratio": len(doc.transformed_document) / len(doc.original_document),
            "transformation_count": len(doc.transformation_map),
            "entities_preserved": len(doc.metadata.get("preserved_entities", []))
        }


class MockQATransformer:
    """Mock Q&A transformer for testing."""
    
    def __init__(self, translator=None, logger=None):
        self.translator = translator
        self.logger = logger
        
    def transform_qa_pairs(
        self,
        qa_pairs: List[QAPair],
        transformed_document: TransformedDocument,
        language: ConstructedLanguage = None
    ) -> List[TransformedQAPair]:
        """Mock Q&A transformation."""
        transformed_pairs = []
        
        for qa in qa_pairs:
            # Simple transformation
            transformed_q = qa.question.replace("e", "æ")
            transformed_a = qa.answer.replace("e", "æ")
            
            transformed_qa = QAPair(
                question=transformed_q,
                answer=transformed_a,
                answer_type=qa.answer_type,
                context_required=qa.context_required,
                difficulty=qa.difficulty,
                metadata=qa.metadata
            )
            
            transformed_pairs.append(TransformedQAPair(
                original_qa=qa,
                transformed_qa=transformed_qa,
                answer_mapping={"e": "æ"},
                confidence=0.95,
                metadata={}
            ))
            
        return transformed_pairs
        
    def get_transformation_stats(self, qa_pairs: List[TransformedQAPair]) -> Dict[str, Any]:
        """Mock transformation stats."""
        return {
            "total_pairs": len(qa_pairs),
            "avg_confidence": sum(qa.confidence for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            "answer_types": list(set(qa.transformed_qa.answer_type for qa in qa_pairs))
        }
        
    def validate_transformation(self, qa_pair: TransformedQAPair, doc: TransformedDocument) -> Dict[str, Any]:
        """Mock transformation validation."""
        return {"valid": True, "issues": []}


class MockProblemTranslator:
    """Mock problem translator for testing."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def translate_answer(
        self,
        answer: str,
        translation_key: Dict[str, str],
        reverse: bool = False
    ) -> str:
        """Mock answer translation."""
        if reverse:
            # Translate back from constructed language
            return answer.replace("æ", "e")
        else:
            # Translate to constructed language
            return answer.replace("e", "æ")


class MockEvaluator:
    """Mock evaluator for testing."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def extract_answer_pattern(self, response: str) -> str:
        """Mock answer extraction."""
        # Return the response as-is for simplicity
        return response.strip()
        
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
        "vocab_size": 2000,
        "languages_dir": "test_languages",
        "results_dir": "test_results",
        "preserve_structure": True,
        "preserve_entities": True,
        "evaluation_mode": "exact_match",
        "evaluation_threshold": 0.8,
        "prompt_template": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    })

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_benchmark(mock_config):
    """Create a mock long context benchmark."""
    benchmark = LongContextBenchmark(
        dataset_name="test_dataset",
        transformation_type=TransformationType.TRANSLATION,
        language_type=LanguageType.SUBSTITUTION,
        language_complexity=5,
        config=mock_config
    )
    
    # Replace components with mocks
    benchmark.data_loader = MockDataLoader()
    benchmark.language_generator = MockLanguageGenerator()
    benchmark.translator = MockProblemTranslator()
    benchmark.document_transformer = MockDocumentTransformer()
    benchmark.qa_transformer = MockQATransformer()
    benchmark.evaluator = MockEvaluator()
    
    return benchmark

@pytest.fixture
def perfect_model():
    """Create a model that gives perfect answers."""
    return MockModel(
        responses={
            "What is AI?": "Artificial intelligence",
            "What is machine learning?": "A subset of AI that focuses on pattern recognition",
            "What is the capital of France?": "Paris",
            "What is Paris known for?": "The Eiffel Tower and the Louvre Museum"
        }
    )

@pytest.fixture
def sample_qa_pairs():
    """Create sample Q&A pairs."""
    return [
        QAPair(
            question="What is artificial intelligence?",
            answer="AI is a field of computer science",
            answer_type=AnswerType.ABSTRACTIVE.value,
            context_required=True,
            difficulty=3,
            metadata={"id": "q1"}
        ),
        QAPair(
            question="Is AI useful?",
            answer="Yes",
            answer_type=AnswerType.YES_NO.value,
            context_required=False,
            difficulty=1,
            metadata={"id": "q2"}
        )
    ]


class TestLongContextBenchmarkInitialization:
    """Test long context benchmark initialization."""
    
    def test_basic_initialization(self):
        """Test basic benchmark initialization."""
        benchmark = LongContextBenchmark("test_dataset")
        
        assert benchmark.dataset_name == "test_dataset"
        assert benchmark.transformation_type == TransformationType.TRANSLATION
        assert benchmark.language_type == LanguageType.SUBSTITUTION
        assert benchmark.language_complexity == 5
        assert benchmark.name == "longcontext_test_dataset_translation"
        
    def test_initialization_with_parameters(self, mock_config):
        """Test initialization with specific parameters."""
        benchmark = LongContextBenchmark(
            dataset_name="custom_dataset",
            transformation_type=TransformationType.PARAPHRASE,
            language_type=LanguageType.PHONETIC,
            language_complexity=8,
            config=mock_config
        )
        
        assert benchmark.dataset_name == "custom_dataset"
        assert benchmark.transformation_type == TransformationType.PARAPHRASE
        assert benchmark.language_type == LanguageType.PHONETIC
        assert benchmark.language_complexity == 8
        
    def test_initialization_without_language_type(self):
        """Test initialization without language type for non-translation transformations."""
        benchmark = LongContextBenchmark(
            dataset_name="test",
            transformation_type=TransformationType.STRUCTURAL,
            language_type=None
        )
        
        assert benchmark.language_type is None
        assert benchmark.language_generator is None
        
    def test_components_initialization(self, mock_benchmark):
        """Test that all components are properly initialized."""
        assert mock_benchmark.language_generator is not None
        assert mock_benchmark.translator is not None
        assert mock_benchmark.document_transformer is not None
        assert mock_benchmark.qa_transformer is not None
        assert mock_benchmark.evaluator is not None
        assert mock_benchmark.data_loader is not None
        
    def test_initial_state(self, mock_benchmark):
        """Test initial benchmark state."""
        assert mock_benchmark.constructed_language is None
        assert mock_benchmark.original_data == []
        assert mock_benchmark.transformed_documents == []
        assert mock_benchmark.transformed_qa_pairs == []


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_loads_original_data(self, mock_benchmark):
        """Test that prepare_data loads original data."""
        mock_benchmark.prepare_data()
        
        assert len(mock_benchmark.original_data) == 2
        assert mock_benchmark.original_data[0]["id"] == "doc1"
        
    def test_prepare_data_generates_language_for_translation(self, mock_benchmark):
        """Test that prepare_data generates language for translation transformation."""
        mock_benchmark.prepare_data()
        
        assert mock_benchmark.constructed_language is not None
        assert mock_benchmark.constructed_language.language_type == LanguageType.SUBSTITUTION
        
    def test_prepare_data_skips_language_for_non_translation(self, mock_config):
        """Test that language generation is skipped for non-translation transformations."""
        benchmark = LongContextBenchmark(
            dataset_name="test",
            transformation_type=TransformationType.STRUCTURAL,
            language_type=None,
            config=mock_config
        )
        benchmark.data_loader = MockDataLoader()
        benchmark.document_transformer = MockDocumentTransformer()
        benchmark.qa_transformer = MockQATransformer()
        
        benchmark.prepare_data()
        
        assert benchmark.constructed_language is None
        
    def test_prepare_data_transforms_documents(self, mock_benchmark):
        """Test that prepare_data transforms all documents."""
        mock_benchmark.prepare_data()
        
        assert len(mock_benchmark.transformed_documents) == 2
        assert all(isinstance(doc, TransformedDocument) for doc in mock_benchmark.transformed_documents)
        
    def test_prepare_data_transforms_qa_pairs(self, mock_benchmark):
        """Test that prepare_data transforms all Q&A pairs."""
        mock_benchmark.prepare_data()
        
        assert len(mock_benchmark.transformed_qa_pairs) == 2
        assert all(isinstance(qa_list, list) for qa_list in mock_benchmark.transformed_qa_pairs)
        assert all(isinstance(qa, TransformedQAPair) for qa_list in mock_benchmark.transformed_qa_pairs for qa in qa_list)
        
    def test_prepare_data_calls_validation(self, mock_benchmark):
        """Test that prepare_data calls transformation validation."""
        with patch.object(mock_benchmark, '_validate_transformations') as mock_validate:
            mock_benchmark.prepare_data()
            
        mock_validate.assert_called_once()
        
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
            call("Preparing long context benchmark data for test_dataset"),
            call("Loaded 2 documents with Q&A pairs"),
            call("Transformed 2 documents and 4 Q&A pairs")
        ])


class TestDocumentAndQAExtraction:
    """Test document and Q&A extraction utilities."""
    
    def test_extract_document_standard_field(self, mock_benchmark):
        """Test extracting document from standard field."""
        data_item = {"document": "Test document content", "other": "ignored"}
        
        doc = mock_benchmark._extract_document(data_item)
        
        assert doc == "Test document content"
        
    def test_extract_document_alternative_fields(self, mock_benchmark):
        """Test extracting document from alternative field names."""
        for field in ['context', 'passage', 'text', 'article']:
            data_item = {field: "Document content", "other": "ignored"}
            
            doc = mock_benchmark._extract_document(data_item)
            
            assert doc == "Document content"
            
    def test_extract_document_longest_field_fallback(self, mock_benchmark):
        """Test fallback to longest string field."""
        data_item = {
            "short": "brief",
            "long": "This is a much longer piece of text content",
            "number": 42
        }
        
        doc = mock_benchmark._extract_document(data_item)
        
        assert doc == "This is a much longer piece of text content"
        
    def test_extract_document_no_content_error(self, mock_benchmark):
        """Test error when no document content found."""
        data_item = {"number": 42, "list": [1, 2, 3]}
        
        with pytest.raises(ValueError, match="Could not extract document"):
            mock_benchmark._extract_document(data_item)
            
    def test_extract_qa_pairs_questions_answers_lists(self, mock_benchmark):
        """Test extracting Q&A pairs from questions/answers lists."""
        data_item = {
            "questions": ["Q1", "Q2"],
            "answers": ["A1", "A2"],
            "other": "ignored"
        }
        
        qa_pairs = mock_benchmark._extract_qa_pairs(data_item)
        
        assert len(qa_pairs) == 2
        assert qa_pairs[0].question == "Q1"
        assert qa_pairs[0].answer == "A1"
        assert qa_pairs[1].question == "Q2"
        assert qa_pairs[1].answer == "A2"
        
    def test_extract_qa_pairs_qa_pairs_list(self, mock_benchmark):
        """Test extracting from qa_pairs list structure."""
        data_item = {
            "qa_pairs": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"}
            ]
        }
        
        qa_pairs = mock_benchmark._extract_qa_pairs(data_item)
        
        assert len(qa_pairs) == 2
        assert qa_pairs[0].question == "Q1"
        assert qa_pairs[1].question == "Q2"
        
    def test_extract_qa_pairs_single_pair(self, mock_benchmark):
        """Test extracting single Q&A pair."""
        data_item = {"question": "What is this?", "answer": "A test"}
        
        qa_pairs = mock_benchmark._extract_qa_pairs(data_item)
        
        assert len(qa_pairs) == 1
        assert qa_pairs[0].question == "What is this?"
        assert qa_pairs[0].answer == "A test"
        
    def test_extract_qa_pairs_no_pairs_warning(self, mock_benchmark):
        """Test warning when no Q&A pairs found."""
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        data_item = {"document": "Just a document"}
        
        qa_pairs = mock_benchmark._extract_qa_pairs(data_item)
        
        assert len(qa_pairs) == 0
        mock_logger.warning.assert_called_with("No Q&A pairs found in data item")


class TestQAPairCreation:
    """Test Q&A pair creation and answer type detection."""
    
    def test_create_qa_pair_basic(self, mock_benchmark):
        """Test basic Q&A pair creation."""
        qa_pair = mock_benchmark._create_qa_pair(
            "What is AI?",
            "Artificial intelligence",
            {"id": "q1"}
        )
        
        assert qa_pair.question == "What is AI?"
        assert qa_pair.answer == "Artificial intelligence"
        assert qa_pair.metadata["id"] == "q1"
        
    def test_create_qa_pair_with_difficulty(self, mock_benchmark):
        """Test Q&A pair creation with difficulty."""
        qa_pair = mock_benchmark._create_qa_pair(
            "Question",
            "Answer",
            {"difficulty": "hard"}
        )
        
        assert qa_pair.difficulty == 5  # 'hard' maps to 5
        
    def test_create_qa_pair_numeric_difficulty(self, mock_benchmark):
        """Test Q&A pair creation with numeric difficulty."""
        qa_pair = mock_benchmark._create_qa_pair(
            "Question",
            "Answer", 
            {"difficulty": 7}
        )
        
        assert qa_pair.difficulty == 7
        
    def test_determine_answer_type_yes_no(self, mock_benchmark):
        """Test detection of yes/no answer type."""
        for answer in ["yes", "no", "true", "false"]:
            answer_type = mock_benchmark._determine_answer_type(answer, {})
            assert answer_type == AnswerType.YES_NO.value
            
    def test_determine_answer_type_numeric(self, mock_benchmark):
        """Test detection of numeric answer type."""
        for answer in ["42", "3.14", "100"]:
            answer_type = mock_benchmark._determine_answer_type(answer, {})
            assert answer_type == AnswerType.NUMERIC.value
            
    def test_determine_answer_type_multiple_choice(self, mock_benchmark):
        """Test detection of multiple choice answer type."""
        for answer in ["A", "B", "C", "D"]:
            answer_type = mock_benchmark._determine_answer_type(answer, {})
            assert answer_type == AnswerType.MULTIPLE_CHOICE.value
            
    def test_determine_answer_type_list(self, mock_benchmark):
        """Test detection of list answer type."""
        for answer in ["A, B, C", "red and blue", "cats, dogs"]:
            answer_type = mock_benchmark._determine_answer_type(answer, {})
            assert answer_type == AnswerType.LIST.value
            
    def test_determine_answer_type_extractive_short(self, mock_benchmark):
        """Test detection of extractive answer type for short answers."""
        answer = "The Eiffel Tower"  # 3 words, likely extractive
        answer_type = mock_benchmark._determine_answer_type(answer, {})
        assert answer_type == AnswerType.EXTRACTIVE.value
        
    def test_determine_answer_type_abstractive_long(self, mock_benchmark):
        """Test detection of abstractive answer type for long answers."""
        answer = "This is a very long answer that would likely be abstractive since it contains many words and concepts"
        answer_type = mock_benchmark._determine_answer_type(answer, {})
        assert answer_type == AnswerType.ABSTRACTIVE.value
        
    def test_determine_answer_type_from_metadata(self, mock_benchmark):
        """Test answer type detection from metadata."""
        answer_type = mock_benchmark._determine_answer_type(
            "Any answer",
            {"answer_type": "custom_type"}
        )
        assert answer_type == "custom_type"


class TestEvaluationDataHandling:
    """Test evaluation data handling."""
    
    def test_get_evaluation_data_returns_all_data(self, mock_benchmark):
        """Test getting all evaluation data."""
        mock_benchmark.prepare_data()
        
        data = mock_benchmark.get_evaluation_data()
        
        assert len(data) == 2
        assert all(isinstance(item, tuple) for item in data)
        assert all(len(item) == 2 for item in data)
        
    def test_get_evaluation_data_limits_samples(self, mock_benchmark):
        """Test limiting number of samples."""
        mock_benchmark.prepare_data()
        
        data = mock_benchmark.get_evaluation_data(num_samples=1)
        
        assert len(data) == 1
        
    def test_get_evaluation_data_handles_oversized_request(self, mock_benchmark):
        """Test handling request for more samples than available."""
        mock_benchmark.prepare_data()
        
        data = mock_benchmark.get_evaluation_data(num_samples=100)
        
        assert len(data) == 2  # All available


class TestSingleEvaluation:
    """Test single document evaluation."""
    
    def test_run_single_evaluation_structure(self, mock_benchmark, perfect_model):
        """Test that single evaluation returns proper structure."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        result = mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
        
        required_fields = [
            'document_id', 'document_length', 'original_document_length',
            'transformation_type', 'qa_count', 'accuracy', 'avg_score',
            'avg_response_time', 'total_response_time', 'qa_results',
            'document_stats', 'qa_stats', 'metadata'
        ]
        
        for field in required_fields:
            assert field in result
            
    def test_run_single_evaluation_processes_all_qa_pairs(self, mock_benchmark, perfect_model):
        """Test that all Q&A pairs are processed."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        result = mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
        
        # First document has 2 Q&A pairs
        assert result['qa_count'] == 2
        assert len(result['qa_results']) == 2
        
    def test_run_single_evaluation_measures_timing(self, mock_benchmark, perfect_model):
        """Test that response times are measured."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        result = mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
        
        assert 'avg_response_time' in result
        assert 'total_response_time' in result
        assert result['avg_response_time'] >= 0
        assert result['total_response_time'] >= 0
        
    def test_run_single_evaluation_constructs_prompts(self, mock_benchmark, perfect_model):
        """Test that prompts are properly constructed."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        with patch.object(mock_benchmark, '_construct_prompt', return_value="test prompt") as mock_construct:
            mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
            
        # Should be called once per Q&A pair (2 times for first document)
        assert mock_construct.call_count == 2
        
    def test_run_single_evaluation_queries_model(self, mock_benchmark, perfect_model):
        """Test that model is queried for each Q&A pair."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        initial_call_count = perfect_model.call_count
        mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
        
        # Should query model for each Q&A pair
        assert perfect_model.call_count == initial_call_count + 2
        
    def test_run_single_evaluation_handles_translation_back(self, mock_benchmark, perfect_model):
        """Test that answers are translated back when using constructed language."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        with patch.object(mock_benchmark.translator, 'translate_answer', return_value="translated back") as mock_translate:
            result = mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
            
        # Should attempt translation for each Q&A pair
        assert mock_translate.call_count >= 2


class TestPromptConstruction:
    """Test prompt construction functionality."""
    
    def test_construct_prompt_default_template(self, mock_benchmark):
        """Test prompt construction with default template."""
        context = "This is the context document."
        question = "What is this about?"
        
        prompt = mock_benchmark._construct_prompt(context, question)
        
        expected = "Context: This is the context document.\n\nQuestion: What is this about?\n\nAnswer:"
        assert prompt == expected
        
    def test_construct_prompt_custom_template(self, mock_benchmark):
        """Test prompt construction with custom template."""
        mock_benchmark.config.data["prompt_template"] = "Document: {context}\nQ: {question}\nA:"
        
        context = "Test context"
        question = "Test question?"
        
        prompt = mock_benchmark._construct_prompt(context, question)
        
        expected = "Document: Test context\nQ: Test question?\nA:"
        assert prompt == expected


class TestModelQuerying:
    """Test model querying functionality."""
    
    def test_query_model_generate_interface(self, mock_benchmark):
        """Test querying model with generate interface."""
        model = MockModel()
        
        response = mock_benchmark._query_model(model, "test prompt")
        
        assert response == "Default answer"
        
    def test_query_model_query_interface(self, mock_benchmark):
        """Test querying model with query interface."""
        class QueryModel:
            def query(self, text):
                return f"Query response: {text}"
                
        model = QueryModel()
        
        response = mock_benchmark._query_model(model, "test prompt")
        
        assert response == "Query response: test prompt"
        
    def test_query_model_callable_interface(self, mock_benchmark):
        """Test querying callable model."""
        class CallableModel:
            def __call__(self, text):
                return f"Callable response: {text}"
                
        model = CallableModel()
        
        response = mock_benchmark._query_model(model, "test prompt")
        
        assert response == "Callable response: test prompt"
        
    def test_query_model_no_interface_error(self, mock_benchmark):
        """Test error when model has no recognized interface."""
        class BadModel:
            pass
            
        model = BadModel()
        
        with pytest.raises(ValueError, match="does not have a recognized interface"):
            mock_benchmark._query_model(model, "test prompt")


class TestMetricsComputation:
    """Test metrics computation."""
    
    def test_compute_metrics_perfect_accuracy(self, mock_benchmark):
        """Test metrics computation with perfect accuracy."""
        results = [
            {
                'accuracy': 1.0, 'qa_count': 2,
                'qa_results': [
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.5, 'transformation_confidence': 0.9, 'metadata': {'difficulty': 3}},
                    {'correct': True, 'score': 1.0, 'answer_type': 'yes_no', 'response_time': 0.3, 'transformation_confidence': 0.95, 'metadata': {'difficulty': 1}}
                ],
                'document_length': 100
            }
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['score'] == 1.0
        assert metrics['overall_accuracy'] == 1.0
        assert metrics['document_accuracy'] == 1.0
        assert metrics['total_documents'] == 1
        assert metrics['total_qa_pairs'] == 2
        assert metrics['correct_answers'] == 2
        
    def test_compute_metrics_partial_accuracy(self, mock_benchmark):
        """Test metrics computation with partial accuracy."""
        results = [
            {
                'accuracy': 0.5, 'qa_count': 2,
                'qa_results': [
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.5, 'transformation_confidence': 0.9, 'metadata': {'difficulty': 3}},
                    {'correct': False, 'score': 0.0, 'answer_type': 'yes_no', 'response_time': 0.3, 'transformation_confidence': 0.95, 'metadata': {'difficulty': 1}}
                ],
                'document_length': 100
            }
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['score'] == 0.5
        assert metrics['overall_accuracy'] == 0.5
        assert metrics['correct_answers'] == 1
        
    def test_compute_metrics_empty_results(self, mock_benchmark):
        """Test metrics computation with empty results."""
        metrics = mock_benchmark.compute_metrics([])
        
        assert metrics['score'] == 0.0
        
    def test_compute_metrics_includes_performance_analysis(self, mock_benchmark):
        """Test that metrics include performance analysis by type and difficulty."""
        results = [
            {
                'accuracy': 1.0, 'qa_count': 3,
                'qa_results': [
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.5, 'transformation_confidence': 0.9, 'metadata': {'difficulty': 3}},
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.4, 'transformation_confidence': 0.95, 'metadata': {'difficulty': 3}},
                    {'correct': False, 'score': 0.0, 'answer_type': 'yes_no', 'response_time': 0.3, 'transformation_confidence': 0.8, 'metadata': {'difficulty': 1}}
                ],
                'document_length': 100
            }
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert 'answer_type_performance' in metrics
        assert 'difficulty_performance' in metrics
        assert 'extractive' in metrics['answer_type_performance']
        assert 'yes_no' in metrics['answer_type_performance']
        assert 3 in metrics['difficulty_performance']
        assert 1 in metrics['difficulty_performance']
        
        # Extractive should be perfect (2/2), yes_no should be 0 (0/1)
        assert metrics['answer_type_performance']['extractive'] == 1.0
        assert metrics['answer_type_performance']['yes_no'] == 0.0
        
    def test_compute_metrics_includes_benchmark_info(self, mock_benchmark):
        """Test that metrics include benchmark-specific information."""
        results = [
            {
                'accuracy': 1.0, 'qa_count': 1,
                'qa_results': [
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.5, 'transformation_confidence': 0.9, 'metadata': {'difficulty': 3}}
                ],
                'document_length': 100
            }
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert metrics['transformation_type'] == 'translation'
        assert metrics['language_type'] == 'substitution'
        assert metrics['language_complexity'] == 5
        assert metrics['dataset_name'] == 'test_dataset'
        
    def test_compute_metrics_document_length_analysis(self, mock_benchmark):
        """Test document length analysis in metrics."""
        results = [
            {
                'accuracy': 1.0, 'qa_count': 1,
                'qa_results': [
                    {'correct': True, 'score': 1.0, 'answer_type': 'extractive', 'response_time': 0.5, 'transformation_confidence': 0.9, 'metadata': {'difficulty': 3}}
                ],
                'document_length': 200
            },
            {
                'accuracy': 0.5, 'qa_count': 1,
                'qa_results': [
                    {'correct': False, 'score': 0.0, 'answer_type': 'extractive', 'response_time': 0.6, 'transformation_confidence': 0.8, 'metadata': {'difficulty': 3}}
                ],
                'document_length': 300
            }
        ]
        
        metrics = mock_benchmark.compute_metrics(results)
        
        assert 'document_length_stats' in metrics
        assert 'length_vs_performance' in metrics
        assert metrics['document_length_stats']['avg_length'] == 250
        assert metrics['document_length_stats']['min_length'] == 200
        assert metrics['document_length_stats']['max_length'] == 300
        
        assert len(metrics['length_vs_performance']) == 2
        assert metrics['length_vs_performance'][0]['length'] == 200
        assert metrics['length_vs_performance'][0]['accuracy'] == 1.0


class TestTransformationValidation:
    """Test transformation validation functionality."""
    
    def test_validate_transformations_successful(self, mock_benchmark):
        """Test successful transformation validation."""
        mock_benchmark.prepare_data()
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark._validate_transformations()
        
        mock_logger.info.assert_has_calls([
            call("Validating transformations"),
            call("All transformations validated successfully")
        ])
        
    def test_validate_transformations_with_document_issues(self, mock_benchmark):
        """Test validation with document transformation issues."""
        mock_benchmark.prepare_data()
        
        # Mock document transformer to return unusual length ratio
        def mock_get_stats(doc):
            return {'length_ratio': 0.3}  # Too low
            
        mock_benchmark.document_transformer.get_transformation_stats = mock_get_stats
        
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark._validate_transformations()
        
        # Should log warnings for unusual length ratios
        warning_calls = mock_logger.warning.call_args_list
        assert any("unusual length ratio" in str(call) for call in warning_calls)
        
    def test_validate_transformations_with_qa_issues(self, mock_benchmark):
        """Test validation with Q&A transformation issues."""
        mock_benchmark.prepare_data()
        
        # Mock Q&A transformer to return validation issues
        def mock_validate(qa_pair, doc):
            return {"valid": False, "issues": ["test issue"]}
            
        mock_benchmark.qa_transformer.validate_transformation = mock_validate
        
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        mock_benchmark._validate_transformations()
        
        # Should log warnings for Q&A validation issues
        warning_calls = mock_logger.warning.call_args_list
        assert any("validation failed" in str(call) for call in warning_calls)


class TestBenchmarkDataExport:
    """Test benchmark data export functionality."""
    
    def test_export_benchmark_data_creates_files(self, mock_benchmark, temp_dir):
        """Test that export creates all expected files."""
        mock_benchmark.prepare_data()
        
        mock_benchmark.export_benchmark_data(temp_dir)
        
        assert (temp_dir / "transformed_documents.json").exists()
        assert (temp_dir / "transformed_qa_pairs.json").exists()
        assert (temp_dir / "constructed_language.json").exists()
        assert (temp_dir / "benchmark_metadata.json").exists()
        
    def test_export_benchmark_data_document_content(self, mock_benchmark, temp_dir):
        """Test content of exported documents."""
        mock_benchmark.prepare_data()
        
        mock_benchmark.export_benchmark_data(temp_dir)
        
        with open(temp_dir / "transformed_documents.json", 'r') as f:
            docs_data = json.load(f)
            
        assert len(docs_data) == 2
        assert 'original_document' in docs_data[0]
        assert 'transformed_document' in docs_data[0]
        assert 'transformation_type' in docs_data[0]
        assert 'transformation_map' in docs_data[0]
        
    def test_export_benchmark_data_qa_content(self, mock_benchmark, temp_dir):
        """Test content of exported Q&A pairs."""
        mock_benchmark.prepare_data()
        
        mock_benchmark.export_benchmark_data(temp_dir)
        
        with open(temp_dir / "transformed_qa_pairs.json", 'r') as f:
            qa_data = json.load(f)
            
        assert len(qa_data) == 2  # 2 documents
        assert 'qa_pairs' in qa_data[0]
        assert len(qa_data[0]['qa_pairs']) == 2  # 2 Q&A pairs per document
        
        qa_pair = qa_data[0]['qa_pairs'][0]
        assert 'original_question' in qa_pair
        assert 'transformed_question' in qa_pair
        assert 'answer_type' in qa_pair
        assert 'confidence' in qa_pair
        
    def test_export_benchmark_data_metadata_content(self, mock_benchmark, temp_dir):
        """Test content of exported metadata."""
        mock_benchmark.prepare_data()
        
        mock_benchmark.export_benchmark_data(temp_dir)
        
        with open(temp_dir / "benchmark_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        assert metadata['benchmark_name'] == mock_benchmark.name
        assert metadata['dataset_name'] == 'test_dataset'
        assert metadata['transformation_type'] == 'translation'
        assert metadata['language_type'] == 'substitution'
        assert metadata['total_documents'] == 2
        assert metadata['total_qa_pairs'] == 4
        
    def test_export_benchmark_data_without_language(self, mock_config, temp_dir):
        """Test export when no constructed language is used."""
        benchmark = LongContextBenchmark(
            dataset_name="test",
            transformation_type=TransformationType.STRUCTURAL,
            language_type=None,
            config=mock_config
        )
        benchmark.data_loader = MockDataLoader()
        benchmark.document_transformer = MockDocumentTransformer()
        benchmark.qa_transformer = MockQATransformer()
        
        benchmark.prepare_data()
        benchmark.export_benchmark_data(temp_dir)
        
        # Language file should not exist
        assert not (temp_dir / "constructed_language.json").exists()
        # But other files should exist
        assert (temp_dir / "transformed_documents.json").exists()
        assert (temp_dir / "transformed_qa_pairs.json").exists()
        assert (temp_dir / "benchmark_metadata.json").exists()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_config_success(self, mock_benchmark):
        """Test successful config validation."""
        assert mock_benchmark.validate_config() is True
        
    def test_validate_config_translation_without_language(self, mock_config):
        """Test validation failure when translation transformation lacks language type."""
        benchmark = LongContextBenchmark(
            dataset_name="test",
            transformation_type=TransformationType.TRANSLATION,
            language_type=None,
            config=mock_config
        )
        
        mock_logger = MagicMock()
        benchmark.logger = mock_logger
        
        assert benchmark.validate_config() is False
        mock_logger.error.assert_called_with("Language type required for translation transformation")
        
    def test_validate_config_invalid_complexity(self, mock_benchmark):
        """Test validation failure with invalid complexity."""
        mock_benchmark.language_complexity = 15  # Invalid (> 10)
        
        mock_logger = MagicMock()
        mock_benchmark.logger = mock_logger
        
        assert mock_benchmark.validate_config() is False
        mock_logger.error.assert_called_with("Language complexity must be 1-10, got 15")
        
    def test_validate_config_complexity_too_low(self, mock_benchmark):
        """Test validation failure with complexity too low."""
        mock_benchmark.language_complexity = 0  # Invalid (< 1)
        
        assert mock_benchmark.validate_config() is False


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_full_benchmark_run_workflow(self, mock_benchmark, perfect_model):
        """Test complete benchmark execution workflow."""
        result = mock_benchmark.run(perfect_model)
        
        # Check that all components were used
        assert len(mock_benchmark.original_data) == 2
        assert mock_benchmark.constructed_language is not None
        assert len(mock_benchmark.transformed_documents) == 2
        assert len(mock_benchmark.transformed_qa_pairs) == 2
        
        # Check result structure
        assert result.benchmark_name == "longcontext_test_dataset_translation"
        assert result.model_name == "mock_model"
        assert isinstance(result.score, float)
        assert result.duration > 0
        
    def test_benchmark_run_with_limited_samples(self, mock_benchmark, perfect_model):
        """Test benchmark run with limited samples."""
        result = mock_benchmark.run(perfect_model, num_samples=1)
        
        assert result.metadata["num_samples"] == 1
        
    def test_benchmark_state_after_run(self, mock_benchmark, perfect_model):
        """Test benchmark state after successful run."""
        mock_benchmark.run(perfect_model)
        
        # State should be properly maintained
        assert len(mock_benchmark.results) == 1
        assert mock_benchmark.constructed_language is not None
        assert len(mock_benchmark.transformed_documents) == 2
        assert len(mock_benchmark.transformed_qa_pairs) == 2


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
        assert result.metrics["total_documents"] == 0
        
    def test_document_extraction_failure(self, mock_benchmark, perfect_model):
        """Test handling of document extraction failures."""
        # Set data with no extractable document
        mock_benchmark.data_loader.data = [{"number": 42}]
        
        with pytest.raises(ValueError):
            mock_benchmark.run(perfect_model)
            
    def test_translation_error_handling(self, mock_benchmark, perfect_model):
        """Test handling of translation errors."""
        mock_benchmark.prepare_data()
        eval_data = mock_benchmark.get_evaluation_data()
        
        # Mock translator to raise exception sometimes
        def failing_translate(answer, mapping, reverse=False):
            if "error" in answer:
                raise Exception("Translation failed")
            return answer.replace("æ", "e")
            
        mock_benchmark.translator.translate_answer = failing_translate
        
        # Should handle the error gracefully (catch exception and use direct comparison)
        result = mock_benchmark.run_single_evaluation(perfect_model, eval_data[0])
        
        # Should complete without crashing
        assert 'accuracy' in result


class TestParameterizedTesting:
    """Test with different parameter combinations."""
    
    @pytest.mark.parametrize("transformation_type", [
        TransformationType.TRANSLATION,
        TransformationType.PARAPHRASE,
        TransformationType.STRUCTURAL,
        TransformationType.HYBRID
    ])
    def test_all_transformation_types(self, transformation_type, mock_config, perfect_model):
        """Test benchmark with all transformation types."""
        # Only use language type for translation
        language_type = LanguageType.SUBSTITUTION if transformation_type == TransformationType.TRANSLATION else None
        
        benchmark = LongContextBenchmark(
            dataset_name="test_dataset",
            transformation_type=transformation_type,
            language_type=language_type,
            config=mock_config
        )
        
        # Replace with mocks
        benchmark.data_loader = MockDataLoader()
        if language_type:
            benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.document_transformer = MockDocumentTransformer()
        benchmark.qa_transformer = MockQATransformer()
        benchmark.evaluator = MockEvaluator()
        
        result = benchmark.run(perfect_model)
        
        assert result.metrics["transformation_type"] == transformation_type.value
        
    @pytest.mark.parametrize("complexity", [1, 3, 5, 7, 10])
    def test_complexity_levels(self, complexity, mock_config, perfect_model):
        """Test benchmark with different complexity levels."""
        benchmark = LongContextBenchmark(
            dataset_name="test_dataset",
            language_complexity=complexity,
            config=mock_config
        )
        
        # Replace with mocks
        benchmark.data_loader = MockDataLoader()
        benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.document_transformer = MockDocumentTransformer()
        benchmark.qa_transformer = MockQATransformer()
        benchmark.evaluator = MockEvaluator()
        
        result = benchmark.run(perfect_model)
        
        assert result.metrics["language_complexity"] == complexity
        
    @pytest.mark.parametrize("num_samples", [1, 2, None])
    def test_sample_sizes(self, num_samples, mock_benchmark, perfect_model):
        """Test benchmark with different sample sizes."""
        result = mock_benchmark.run(perfect_model, num_samples=num_samples)
        
        expected_samples = 2 if num_samples is None else min(num_samples, 2)
        assert result.metadata["num_samples"] == expected_samples


class TestPerformanceAndScaling:
    """Test performance and scaling considerations."""
    
    def test_large_document_simulation(self, mock_config):
        """Test with simulated large documents."""
        # Create data with large documents
        large_doc = "This is a very long document. " * 1000  # ~30k characters
        large_data = [
            {
                "id": f"doc{i}",
                "document": large_doc,
                "questions": [f"Question {i}"],
                "answers": [f"Answer {i}"]
            }
            for i in range(5)
        ]
        
        benchmark = LongContextBenchmark("large_dataset", config=mock_config)
        benchmark.data_loader = MockDataLoader(large_data)
        benchmark.language_generator = MockLanguageGenerator()
        benchmark.translator = MockProblemTranslator()
        benchmark.document_transformer = MockDocumentTransformer()
        benchmark.qa_transformer = MockQATransformer()
        benchmark.evaluator = MockEvaluator()
        
        # Should handle large documents
        benchmark.prepare_data()
        
        assert len(benchmark.original_data) == 5
        assert len(benchmark.transformed_documents) == 5
        assert all(len(doc.original_document) > 20000 for doc in benchmark.transformed_documents)
        
    def test_memory_efficiency_with_samples(self, mock_benchmark, perfect_model):
        """Test memory efficiency when using limited samples."""
        # Run with limited samples
        result = mock_benchmark.run(perfect_model, num_samples=1)
        
        # Should only process requested samples
        assert result.metadata["num_samples"] == 1
        # Model should only be called for Q&A pairs in 1 document (2 pairs)
        assert perfect_model.call_count == 2