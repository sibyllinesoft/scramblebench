"""
Q&A transformation for long context benchmarks.

This module handles the transformation of question-answer pairs
to align with transformed documents while maintaining evaluation
validity and avoiding training data contamination.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
import logging
from pathlib import Path

from scramblebench.translation.language_generator import ConstructedLanguage
from scramblebench.translation.translator import ProblemTranslator
from scramblebench.longcontext.document_transformer import TransformedDocument


@dataclass
class QAPair:
    """
    A question-answer pair with metadata.
    
    Attributes:
        question: The question text
        answer: The answer text
        answer_type: Type of answer (extractive, abstractive, multiple_choice, etc.)
        context_required: Whether the answer requires document context
        difficulty: Difficulty level (1-5)
        metadata: Additional Q&A metadata
    """
    question: str
    answer: str
    answer_type: str
    context_required: bool = True
    difficulty: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TransformedQAPair:
    """
    A Q&A pair that has been transformed to align with document changes.
    
    Attributes:
        original_qa: Original question-answer pair
        transformed_qa: Transformed question-answer pair
        answer_mapping: Mapping from transformed to original answer components
        confidence: Confidence in transformation quality (0-1)
        metadata: Additional transformation metadata
    """
    original_qa: QAPair
    transformed_qa: QAPair
    answer_mapping: Dict[str, str]
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnswerType(Enum):
    """Types of answers in Q&A pairs."""
    EXTRACTIVE = "extractive"  # Direct span from document
    ABSTRACTIVE = "abstractive"  # Generated/summarized answer
    MULTIPLE_CHOICE = "multiple_choice"  # One of several options
    YES_NO = "yes_no"  # Boolean answer
    NUMERIC = "numeric"  # Number or quantity
    LIST = "list"  # List of items


class QATransformer:
    """
    Transforms question-answer pairs for long context benchmarks.
    
    Handles the alignment of Q&A pairs with transformed documents
    while maintaining evaluation validity and answer correctness.
    """
    
    def __init__(
        self,
        translator: Optional[ProblemTranslator] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Q&A transformer.
        
        Args:
            translator: Problem translator for language transformations
            logger: Logger instance (creates default if None)
        """
        self.translator = translator or ProblemTranslator()
        self.logger = logger or logging.getLogger("scramblebench.qa_transformer")
    
    def transform_qa_pairs(
        self,
        qa_pairs: List[QAPair],
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage] = None
    ) -> List[TransformedQAPair]:
        """
        Transform Q&A pairs to align with a transformed document.
        
        Args:
            qa_pairs: List of original Q&A pairs
            transformed_document: Document that has been transformed
            language: Constructed language (if document was translated)
            
        Returns:
            List of transformed Q&A pairs
        """
        self.logger.info(f"Transforming {len(qa_pairs)} Q&A pairs")
        
        transformed_pairs = []
        
        for i, qa_pair in enumerate(qa_pairs):
            self.logger.debug(f"Transforming Q&A pair {i+1}/{len(qa_pairs)}")
            
            transformed_pair = self._transform_single_qa(
                qa_pair, transformed_document, language
            )
            
            transformed_pairs.append(transformed_pair)
        
        self.logger.info(f"Transformed {len(transformed_pairs)} Q&A pairs")
        return transformed_pairs
    
    def _transform_single_qa(
        self,
        qa_pair: QAPair,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> TransformedQAPair:
        """Transform a single Q&A pair."""
        # Transform question
        transformed_question = self._transform_question(
            qa_pair.question, transformed_document, language
        )
        
        # Transform answer based on type
        answer_type = AnswerType(qa_pair.answer_type)
        transformed_answer, answer_mapping, confidence = self._transform_answer(
            qa_pair.answer, answer_type, transformed_document, language
        )
        
        # Create transformed Q&A pair
        transformed_qa = QAPair(
            question=transformed_question,
            answer=transformed_answer,
            answer_type=qa_pair.answer_type,
            context_required=qa_pair.context_required,
            difficulty=qa_pair.difficulty,
            metadata={
                **qa_pair.metadata,
                'transformation_confidence': confidence,
                'transformation_type': transformed_document.transformation_type.value
            }
        )
        
        return TransformedQAPair(
            original_qa=qa_pair,
            transformed_qa=transformed_qa,
            answer_mapping=answer_mapping,
            confidence=confidence,
            metadata={
                'language_used': language.name if language else None,
                'document_transformation': transformed_document.transformation_type.value
            }
        )
    
    def _transform_question(
        self,
        question: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> str:
        """Transform a question to align with document transformation."""
        transformed_question = question
        
        # Apply document transformation mappings to question
        for transformed_term, original_term in transformed_document.transformation_map.items():
            # Use word boundaries for precise replacement
            pattern = r'\b' + re.escape(original_term) + r'\b'
            transformed_question = re.sub(
                pattern, transformed_term, transformed_question, flags=re.IGNORECASE
            )
        
        # If we have a language, apply full translation
        if language is not None:
            pseudo_problem = {"question": transformed_question}
            translated = self.translator.translate_problem(
                problem=pseudo_problem,
                language=language,
                preserve_numbers=True,
                preserve_proper_nouns=True
            )
            transformed_question = translated.translated_problem["question"]
        
        return transformed_question
    
    def _transform_answer(
        self,
        answer: str,
        answer_type: AnswerType,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """
        Transform an answer based on its type and document transformation.
        
        Returns:
            Tuple of (transformed_answer, answer_mapping, confidence)
        """
        if answer_type == AnswerType.EXTRACTIVE:
            return self._transform_extractive_answer(
                answer, transformed_document, language
            )
        elif answer_type == AnswerType.ABSTRACTIVE:
            return self._transform_abstractive_answer(
                answer, transformed_document, language
            )
        elif answer_type == AnswerType.MULTIPLE_CHOICE:
            return self._transform_multiple_choice_answer(
                answer, transformed_document, language
            )
        elif answer_type == AnswerType.YES_NO:
            return self._transform_yes_no_answer(
                answer, transformed_document, language
            )
        elif answer_type == AnswerType.NUMERIC:
            return self._transform_numeric_answer(
                answer, transformed_document, language
            )
        elif answer_type == AnswerType.LIST:
            return self._transform_list_answer(
                answer, transformed_document, language
            )
        else:
            # Default transformation
            return self._transform_generic_answer(
                answer, transformed_document, language
            )
    
    def _transform_extractive_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform an extractive answer (direct span from document)."""
        # Find the answer span in the original document
        original_doc = transformed_document.original_document
        answer_lower = answer.lower().strip()
        
        # Look for exact match first
        answer_start = original_doc.lower().find(answer_lower)
        
        if answer_start == -1:
            # Try fuzzy matching for slight variations
            confidence = 0.5
            self.logger.warning(f"Could not find exact match for extractive answer: {answer}")
        else:
            confidence = 1.0
        
        # Find corresponding span in transformed document
        transformed_answer = self._find_transformed_span(
            answer, original_doc, transformed_document.transformed_document,
            transformed_document.transformation_map
        )
        
        if transformed_answer is None:
            # Fallback: apply transformation mapping
            transformed_answer = answer
            for transformed_term, original_term in transformed_document.transformation_map.items():
                pattern = r'\b' + re.escape(original_term) + r'\b'
                transformed_answer = re.sub(
                    pattern, transformed_term, transformed_answer, flags=re.IGNORECASE
                )
            confidence *= 0.7
        
        # Apply language translation if available
        if language is not None:
            pseudo_problem = {"answer": transformed_answer}
            translated = self.translator.translate_problem(
                problem=pseudo_problem,
                language=language,
                preserve_numbers=True,
                preserve_proper_nouns=True
            )
            transformed_answer = translated.translated_problem["answer"]
        
        answer_mapping = {transformed_answer: answer}
        
        return transformed_answer, answer_mapping, confidence
    
    def _transform_abstractive_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform an abstractive answer (generated/summarized)."""
        transformed_answer = answer
        confidence = 0.9  # Abstractive answers are generally more robust
        
        # Apply document transformation mappings
        for transformed_term, original_term in transformed_document.transformation_map.items():
            pattern = r'\b' + re.escape(original_term) + r'\b'
            transformed_answer = re.sub(
                pattern, transformed_term, transformed_answer, flags=re.IGNORECASE
            )
        
        # Apply language translation if available
        if language is not None:
            pseudo_problem = {"answer": transformed_answer}
            translated = self.translator.translate_problem(
                problem=pseudo_problem,
                language=language,
                preserve_numbers=True,
                preserve_proper_nouns=True
            )
            transformed_answer = translated.translated_problem["answer"]
        
        answer_mapping = {transformed_answer: answer}
        
        return transformed_answer, answer_mapping, confidence
    
    def _transform_multiple_choice_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform a multiple choice answer."""
        # Multiple choice answers are typically just option letters/numbers
        # which don't need transformation unless they contain text
        
        if re.match(r'^[A-D]$', answer.strip()) or re.match(r'^\d+$', answer.strip()):
            # Simple option identifier - no transformation needed
            return answer, {answer: answer}, 1.0
        
        # If the answer contains text, transform it
        return self._transform_abstractive_answer(answer, transformed_document, language)
    
    def _transform_yes_no_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform a yes/no answer."""
        answer_lower = answer.lower().strip()
        
        if answer_lower in ['yes', 'no', 'true', 'false']:
            # Boolean answers typically don't need transformation
            # unless we're translating the language
            if language is not None:
                # Translate yes/no to the constructed language
                pseudo_problem = {"answer": answer}
                translated = self.translator.translate_problem(
                    problem=pseudo_problem,
                    language=language,
                    preserve_numbers=True,
                    preserve_proper_nouns=True
                )
                transformed_answer = translated.translated_problem["answer"]
                return transformed_answer, {transformed_answer: answer}, 1.0
            else:
                return answer, {answer: answer}, 1.0
        
        # If it's not a simple yes/no, treat as abstractive
        return self._transform_abstractive_answer(answer, transformed_document, language)
    
    def _transform_numeric_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform a numeric answer."""
        # Numeric answers typically don't need transformation
        # Numbers should be preserved across transformations
        
        # Check if it's purely numeric
        if re.match(r'^\d+(\.\d+)?$', answer.strip()):
            return answer, {answer: answer}, 1.0
        
        # If it contains text with numbers, transform the text part
        return self._transform_abstractive_answer(answer, transformed_document, language)
    
    def _transform_list_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Transform a list answer."""
        # Parse list items
        items = self._parse_list_items(answer)
        
        transformed_items = []
        answer_mapping = {}
        total_confidence = 0.0
        
        for item in items:
            transformed_item, item_mapping, confidence = self._transform_abstractive_answer(
                item, transformed_document, language
            )
            transformed_items.append(transformed_item)
            answer_mapping.update(item_mapping)
            total_confidence += confidence
        
        # Reconstruct list
        if len(items) > 1:
            transformed_answer = ', '.join(transformed_items[:-1]) + ', and ' + transformed_items[-1]
        elif len(items) == 1:
            transformed_answer = transformed_items[0]
        else:
            transformed_answer = answer
        
        avg_confidence = total_confidence / len(items) if items else 1.0
        
        return transformed_answer, answer_mapping, avg_confidence
    
    def _transform_generic_answer(
        self,
        answer: str,
        transformed_document: TransformedDocument,
        language: Optional[ConstructedLanguage]
    ) -> Tuple[str, Dict[str, str], float]:
        """Generic answer transformation fallback."""
        return self._transform_abstractive_answer(answer, transformed_document, language)
    
    def _find_transformed_span(
        self,
        original_span: str,
        original_document: str,
        transformed_document: str,
        transformation_map: Dict[str, str]
    ) -> Optional[str]:
        """
        Find the corresponding span in the transformed document.
        
        This is a simplified implementation - in practice, you might use
        more sophisticated alignment techniques.
        """
        # Apply transformation mapping to the span
        transformed_span = original_span
        
        for transformed_term, original_term in transformation_map.items():
            pattern = r'\b' + re.escape(original_term) + r'\b'
            transformed_span = re.sub(
                pattern, transformed_term, transformed_span, flags=re.IGNORECASE
            )
        
        # Check if the transformed span exists in the transformed document
        if transformed_span.lower() in transformed_document.lower():
            return transformed_span
        
        return None
    
    def _parse_list_items(self, list_text: str) -> List[str]:
        """Parse a list-formatted answer into individual items."""
        # Handle different list formats
        items = []
        
        # Try comma-separated
        if ',' in list_text:
            items = [item.strip() for item in list_text.split(',')]
        # Try semicolon-separated
        elif ';' in list_text:
            items = [item.strip() for item in list_text.split(';')]
        # Try "and" separated
        elif ' and ' in list_text:
            items = [item.strip() for item in list_text.split(' and ')]
        # Try numbered list
        elif re.search(r'\d+\.', list_text):
            items = [
                re.sub(r'^\d+\.\s*', '', item.strip())
                for item in re.split(r'\d+\.', list_text)
                if item.strip()
            ]
        # Try bullet points
        elif re.search(r'[•\-\*]', list_text):
            items = [
                re.sub(r'^[•\-\*]\s*', '', item.strip())
                for item in re.split(r'[•\-\*]', list_text)
                if item.strip()
            ]
        else:
            # Single item
            items = [list_text.strip()]
        
        return [item for item in items if item]
    
    def validate_transformation(
        self,
        transformed_qa: TransformedQAPair,
        transformed_document: TransformedDocument
    ) -> Dict[str, Any]:
        """
        Validate the quality of a Q&A transformation.
        
        Args:
            transformed_qa: Transformed Q&A pair to validate
            transformed_document: Associated transformed document
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'confidence': transformed_qa.confidence,
            'recommendations': []
        }
        
        # Check if answer exists in transformed document (for extractive answers)
        if transformed_qa.transformed_qa.answer_type == "extractive":
            answer_in_doc = transformed_qa.transformed_qa.answer.lower() in transformed_document.transformed_document.lower()
            if not answer_in_doc:
                validation['valid'] = False
                validation['issues'].append("Extractive answer not found in transformed document")
                validation['recommendations'].append("Verify answer span alignment")
        
        # Check transformation consistency
        if not transformed_qa.answer_mapping:
            validation['issues'].append("No answer mapping available")
            validation['recommendations'].append("Ensure answer transformation creates mapping")
        
        # Check confidence threshold
        if transformed_qa.confidence < 0.5:
            validation['issues'].append(f"Low transformation confidence: {transformed_qa.confidence}")
            validation['recommendations'].append("Consider manual review or different transformation")
        
        return validation
    
    def get_transformation_stats(
        self,
        transformed_qa_pairs: List[TransformedQAPair]
    ) -> Dict[str, Any]:
        """
        Get statistics about Q&A transformations.
        
        Args:
            transformed_qa_pairs: List of transformed Q&A pairs
            
        Returns:
            Dictionary containing transformation statistics
        """
        if not transformed_qa_pairs:
            return {}
        
        # Confidence statistics
        confidences = [qa.confidence for qa in transformed_qa_pairs]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Answer type distribution
        answer_types = {}
        for qa in transformed_qa_pairs:
            answer_type = qa.transformed_qa.answer_type
            answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
        
        # Difficulty distribution
        difficulties = {}
        for qa in transformed_qa_pairs:
            difficulty = qa.transformed_qa.difficulty
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        stats = {
            'total_qa_pairs': len(transformed_qa_pairs),
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'answer_type_distribution': answer_types,
            'difficulty_distribution': difficulties,
            'low_confidence_count': len([qa for qa in transformed_qa_pairs if qa.confidence < 0.7]),
            'high_confidence_count': len([qa for qa in transformed_qa_pairs if qa.confidence >= 0.9])
        }
        
        return stats