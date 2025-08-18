"""
Data extraction utilities for benchmark implementations.

This module provides reusable data extraction logic that was previously
duplicated across multiple benchmark classes. The extractors handle common
data formats and provide consistent interfaces for benchmark data processing.
"""

from typing import Any, Dict, List, Optional
import re
from abc import ABC, abstractmethod


class DataExtractor(ABC):
    """
    Abstract base class for data extraction components.
    
    Defines the interface for extracting specific types of data
    from benchmark datasets in a consistent manner.
    """
    
    @abstractmethod
    def extract(self, data_item: Dict[str, Any]) -> Any:
        """
        Extract data from a data item.
        
        Args:
            data_item: Dictionary containing the data to extract from
            
        Returns:
            Extracted data in the appropriate format
            
        Raises:
            ValueError: If the data item format is invalid
        """
        pass


class DocumentExtractor(DataExtractor):
    """
    Extracts document text from various data formats.
    
    Handles common field names and formats used for document storage
    in different benchmark datasets.
    """
    
    def __init__(self, field_priority: Optional[List[str]] = None):
        """
        Initialize the document extractor.
        
        Args:
            field_priority: List of field names to try in order
        """
        self.field_priority = field_priority or [
            'document', 'context', 'passage', 'text', 'article', 'content'
        ]
    
    def extract(self, data_item: Dict[str, Any]) -> str:
        """
        Extract document text from a data item.
        
        Args:
            data_item: Dictionary containing document data
            
        Returns:
            Document text as string
            
        Raises:
            ValueError: If no document text can be found
        """
        # Try priority fields first
        for field in self.field_priority:
            if field in data_item and isinstance(data_item[field], str):
                return data_item[field]
        
        # Fallback: find the longest string field
        longest_field = ""
        for value in data_item.values():
            if isinstance(value, str) and len(value) > len(longest_field):
                longest_field = value
        
        if longest_field:
            return longest_field
        
        raise ValueError("Could not extract document from data item")


class ProblemTextExtractor(DataExtractor):
    """
    Extracts problem text from translated problems or questions.
    
    Handles various formats for storing problem statements,
    questions, and prompts in benchmark datasets.
    """
    
    def __init__(self, field_priority: Optional[List[str]] = None):
        """
        Initialize the problem text extractor.
        
        Args:
            field_priority: List of field names to try in order
        """
        self.field_priority = field_priority or [
            'question', 'problem', 'text', 'prompt', 'input', 'query'
        ]
    
    def extract(self, data_item: Dict[str, Any]) -> str:
        """
        Extract problem text from a data item.
        
        Args:
            data_item: Dictionary containing problem data
            
        Returns:
            Problem text as string
        """
        # Try priority fields first
        for field in self.field_priority:
            if field in data_item and isinstance(data_item[field], str):
                return data_item[field]
        
        # Fallback: concatenate all string values
        text_parts = []
        for value in data_item.values():
            if isinstance(value, str):
                text_parts.append(value)
        
        return ' '.join(text_parts) if text_parts else ""


class AnswerExtractor(DataExtractor):
    """
    Extracts expected answers from various data formats.
    
    Handles different field names and data types for answer storage
    in benchmark datasets.
    """
    
    def __init__(self, field_priority: Optional[List[str]] = None):
        """
        Initialize the answer extractor.
        
        Args:
            field_priority: List of field names to try in order
        """
        self.field_priority = field_priority or [
            'answer', 'solution', 'output', 'target', 'label', 'ground_truth'
        ]
    
    def extract(self, data_item: Dict[str, Any]) -> str:
        """
        Extract expected answer from a data item.
        
        Args:
            data_item: Dictionary containing answer data
            
        Returns:
            Answer as string
        """
        # Try priority fields first
        for field in self.field_priority:
            if field in data_item:
                answer = data_item[field]
                if isinstance(answer, str):
                    return answer
                else:
                    return str(answer)
        
        # Default fallback
        return "unknown"


class QAPairExtractor(DataExtractor):
    """
    Extracts Q&A pairs from various data formats.
    
    Handles different structures for storing question-answer pairs
    including list formats, nested objects, and single pair formats.
    """
    
    def __init__(self):
        """Initialize the Q&A pair extractor."""
        self.answer_type_detector = AnswerTypeDetector()
    
    def extract(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract Q&A pairs from a data item.
        
        Args:
            data_item: Dictionary containing Q&A data
            
        Returns:
            List of Q&A pair dictionaries
        """
        qa_pairs = []
        
        # Try list format (questions + answers arrays)
        if 'questions' in data_item and 'answers' in data_item:
            questions = data_item['questions']
            answers = data_item['answers']
            
            if isinstance(questions, list) and isinstance(answers, list):
                for q, a in zip(questions, answers):
                    qa_pairs.append(self._create_qa_pair(q, a, data_item))
        
        # Try nested format (qa_pairs array)
        elif 'qa_pairs' in data_item:
            for qa in data_item['qa_pairs']:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    qa_pairs.append(self._create_qa_pair(qa['question'], qa['answer'], qa))
        
        # Try single Q&A format
        elif 'question' in data_item and 'answer' in data_item:
            qa_pairs.append(self._create_qa_pair(
                data_item['question'], data_item['answer'], data_item
            ))
        
        return qa_pairs
    
    def _create_qa_pair(
        self,
        question: str,
        answer: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a standardized Q&A pair dictionary.
        
        Args:
            question: Question text
            answer: Answer text
            metadata: Additional metadata
            
        Returns:
            Standardized Q&A pair dictionary
        """
        # Determine answer type
        answer_type = self.answer_type_detector.detect(answer, metadata)
        
        # Determine difficulty
        difficulty = metadata.get('difficulty', 3)
        if isinstance(difficulty, str):
            difficulty_map = {'easy': 1, 'medium': 3, 'hard': 5}
            difficulty = difficulty_map.get(difficulty.lower(), 3)
        
        return {
            'question': question,
            'answer': answer,
            'answer_type': answer_type,
            'context_required': metadata.get('context_required', True),
            'difficulty': difficulty,
            'metadata': {
                'id': metadata.get('id', 'unknown'),
                'source': metadata.get('source', 'unknown')
            }
        }


class AnswerTypeDetector:
    """
    Detects the type of an answer using heuristics.
    
    Provides automatic classification of answer types based on
    content analysis and metadata hints.
    """
    
    def detect(self, answer: str, metadata: Dict[str, Any]) -> str:
        """
        Detect the type of an answer.
        
        Args:
            answer: Answer text to classify
            metadata: Additional metadata that may contain type hints
            
        Returns:
            Answer type as string
        """
        # Check metadata first
        if 'answer_type' in metadata:
            return metadata['answer_type']
        
        # Heuristic detection
        answer_lower = answer.lower().strip()
        
        if answer_lower in ['yes', 'no', 'true', 'false']:
            return 'yes_no'
        
        if re.match(r'^\d+(\.\d+)?$', answer.strip()):
            return 'numeric'
        
        if re.match(r'^[A-D]$', answer.strip()):
            return 'multiple_choice'
        
        if ',' in answer or ' and ' in answer:
            return 'list'
        
        # Check if it's likely extractive (common for reading comprehension)
        if len(answer.split()) <= 10:  # Short answers often extractive
            return 'extractive'
        
        # Default to abstractive
        return 'abstractive'


class PromptConstructor:
    """
    Constructs prompts for model evaluation.
    
    Provides templating and formatting functionality for creating
    consistent prompts across different benchmark types.
    """
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize the prompt constructor.
        
        Args:
            template: Template string with placeholders
        """
        self.template = template or "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    def construct(self, **kwargs) -> str:
        """
        Construct a prompt from template and arguments.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)
    
    def construct_qa_prompt(self, context: str, question: str) -> str:
        """
        Construct a Q&A prompt specifically.
        
        Args:
            context: Context text
            question: Question text
            
        Returns:
            Formatted Q&A prompt
        """
        return self.construct(context=context, question=question)
    
    def construct_problem_prompt(self, problem_text: str) -> str:
        """
        Construct a problem-solving prompt.
        
        Args:
            problem_text: Problem statement
            
        Returns:
        """
        return problem_text  # Simple pass-through for problem text


# Factory function for easy access to extractors
def create_extractors() -> Dict[str, DataExtractor]:
    """
    Create a set of standard data extractors.
    
    Returns:
        Dictionary mapping extractor names to instances
    """
    return {
        'document': DocumentExtractor(),
        'problem_text': ProblemTextExtractor(),
        'answer': AnswerExtractor(),
        'qa_pairs': QAPairExtractor(),
    }