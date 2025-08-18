"""
Tests for data extraction utilities.
"""

import pytest
from scramblebench.core.data_extractors import (
    DocumentExtractor, ProblemTextExtractor, AnswerExtractor, 
    QAPairExtractor, AnswerTypeDetector, PromptConstructor,
    create_extractors
)


class TestDocumentExtractor:
    """Test document extraction functionality."""
    
    def test_extract_with_standard_field(self):
        """Test extraction with standard document field."""
        extractor = DocumentExtractor()
        data = {"document": "This is a test document."}
        result = extractor.extract(data)
        assert result == "This is a test document."
    
    def test_extract_with_priority_fields(self):
        """Test extraction with different priority fields."""
        extractor = DocumentExtractor()
        
        # Test context field
        data = {"context": "Context text", "other": "Other text"}
        assert extractor.extract(data) == "Context text"
        
        # Test passage field
        data = {"passage": "Passage text", "random": "Random text"}
        assert extractor.extract(data) == "Passage text"
    
    def test_extract_longest_field_fallback(self):
        """Test fallback to longest string field."""
        extractor = DocumentExtractor()
        data = {
            "short": "Short",
            "longer": "This is a longer text field that should be selected",
            "number": 42
        }
        result = extractor.extract(data)
        assert result == "This is a longer text field that should be selected"
    
    def test_extract_failure(self):
        """Test extraction failure when no text found."""
        extractor = DocumentExtractor()
        data = {"number": 42, "list": [1, 2, 3]}
        
        with pytest.raises(ValueError, match="Could not extract document"):
            extractor.extract(data)
    
    def test_custom_field_priority(self):
        """Test custom field priority."""
        extractor = DocumentExtractor(field_priority=["custom", "document"])
        data = {"document": "Document text", "custom": "Custom text"}
        assert extractor.extract(data) == "Custom text"


class TestProblemTextExtractor:
    """Test problem text extraction functionality."""
    
    def test_extract_question_field(self):
        """Test extraction with question field."""
        extractor = ProblemTextExtractor()
        data = {"question": "What is 2+2?"}
        assert extractor.extract(data) == "What is 2+2?"
    
    def test_extract_concatenated_fallback(self):
        """Test concatenation fallback."""
        extractor = ProblemTextExtractor()
        data = {"part1": "What is", "part2": "2+2?", "number": 42}
        result = extractor.extract(data)
        assert "What is" in result and "2+2?" in result
    
    def test_extract_empty_data(self):
        """Test extraction with empty data."""
        extractor = ProblemTextExtractor()
        data = {"number": 42}
        assert extractor.extract(data) == ""


class TestAnswerExtractor:
    """Test answer extraction functionality."""
    
    def test_extract_answer_field(self):
        """Test extraction with answer field."""
        extractor = AnswerExtractor()
        data = {"answer": "4"}
        assert extractor.extract(data) == "4"
    
    def test_extract_numeric_conversion(self):
        """Test numeric answer conversion."""
        extractor = AnswerExtractor()
        data = {"answer": 42}
        assert extractor.extract(data) == "42"
    
    def test_extract_fallback(self):
        """Test fallback to unknown."""
        extractor = AnswerExtractor()
        data = {"other": "value"}
        assert extractor.extract(data) == "unknown"


class TestAnswerTypeDetector:
    """Test answer type detection functionality."""
    
    def test_detect_from_metadata(self):
        """Test detection from metadata."""
        detector = AnswerTypeDetector()
        result = detector.detect("Some answer", {"answer_type": "custom"})
        assert result == "custom"
    
    def test_detect_yes_no(self):
        """Test yes/no detection."""
        detector = AnswerTypeDetector()
        assert detector.detect("yes", {}) == "yes_no"
        assert detector.detect("NO", {}) == "yes_no"
        assert detector.detect("true", {}) == "yes_no"
        assert detector.detect("False", {}) == "yes_no"
    
    def test_detect_numeric(self):
        """Test numeric detection."""
        detector = AnswerTypeDetector()
        assert detector.detect("42", {}) == "numeric"
        assert detector.detect("3.14", {}) == "numeric"
        assert detector.detect("0", {}) == "numeric"
    
    def test_detect_multiple_choice(self):
        """Test multiple choice detection."""
        detector = AnswerTypeDetector()
        assert detector.detect("A", {}) == "multiple_choice"
        assert detector.detect("C", {}) == "multiple_choice"
    
    def test_detect_list(self):
        """Test list detection."""
        detector = AnswerTypeDetector()
        assert detector.detect("A, B, C", {}) == "list"
        assert detector.detect("cats and dogs", {}) == "list"
    
    def test_detect_extractive(self):
        """Test extractive detection (short answers)."""
        detector = AnswerTypeDetector()
        assert detector.detect("Paris", {}) == "extractive"
        assert detector.detect("blue car", {}) == "extractive"
    
    def test_detect_abstractive(self):
        """Test abstractive detection (long answers)."""
        detector = AnswerTypeDetector()
        long_answer = "This is a long answer that requires multiple sentences to explain the concept properly"
        assert detector.detect(long_answer, {}) == "abstractive"


class TestQAPairExtractor:
    """Test Q&A pair extraction functionality."""
    
    def test_extract_list_format(self):
        """Test extraction from questions/answers lists."""
        extractor = QAPairExtractor()
        data = {
            "questions": ["What is 2+2?", "What is 3+3?"],
            "answers": ["4", "6"]
        }
        result = extractor.extract(data)
        assert len(result) == 2
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["answer"] == "4"
        assert result[1]["question"] == "What is 3+3?"
        assert result[1]["answer"] == "6"
    
    def test_extract_nested_format(self):
        """Test extraction from qa_pairs array."""
        extractor = QAPairExtractor()
        data = {
            "qa_pairs": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"}
            ]
        }
        result = extractor.extract(data)
        assert len(result) == 2
        assert result[0]["question"] == "Q1"
    
    def test_extract_single_format(self):
        """Test extraction from single Q&A."""
        extractor = QAPairExtractor()
        data = {"question": "What is the capital of France?", "answer": "Paris"}
        result = extractor.extract(data)
        assert len(result) == 1
        assert result[0]["question"] == "What is the capital of France?"
        assert result[0]["answer"] == "Paris"
    
    def test_qa_pair_metadata(self):
        """Test Q&A pair metadata creation."""
        extractor = QAPairExtractor()
        data = {
            "question": "Test question",
            "answer": "yes",
            "difficulty": "easy",
            "id": "test_001"
        }
        result = extractor.extract(data)
        qa_pair = result[0]
        
        assert qa_pair["answer_type"] == "yes_no"
        assert qa_pair["difficulty"] == 1  # easy -> 1
        assert qa_pair["metadata"]["id"] == "test_001"


class TestPromptConstructor:
    """Test prompt construction functionality."""
    
    def test_default_template(self):
        """Test default template construction."""
        constructor = PromptConstructor()
        prompt = constructor.construct_qa_prompt("This is context", "What is this?")
        expected = "Context: This is context\n\nQuestion: What is this?\n\nAnswer:"
        assert prompt == expected
    
    def test_custom_template(self):
        """Test custom template."""
        template = "Given: {context}\nQ: {question}\nA:"
        constructor = PromptConstructor(template)
        prompt = constructor.construct_qa_prompt("Some context", "Some question")
        expected = "Given: Some context\nQ: Some question\nA:"
        assert prompt == expected
    
    def test_general_construct(self):
        """Test general construct method."""
        constructor = PromptConstructor("{greeting} {name}!")
        prompt = constructor.construct(greeting="Hello", name="World")
        assert prompt == "Hello World!"
    
    def test_problem_prompt(self):
        """Test problem prompt construction."""
        constructor = PromptConstructor()
        prompt = constructor.construct_problem_prompt("Solve this problem")
        assert prompt == "Solve this problem"


class TestCreateExtractors:
    """Test extractor factory function."""
    
    def test_create_extractors(self):
        """Test extractor factory creates all expected extractors."""
        extractors = create_extractors()
        
        expected_keys = {'document', 'problem_text', 'answer', 'qa_pairs'}
        assert set(extractors.keys()) == expected_keys
        
        assert isinstance(extractors['document'], DocumentExtractor)
        assert isinstance(extractors['problem_text'], ProblemTextExtractor)
        assert isinstance(extractors['answer'], AnswerExtractor)
        assert isinstance(extractors['qa_pairs'], QAPairExtractor)