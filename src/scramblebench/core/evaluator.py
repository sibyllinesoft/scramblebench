"""
Evaluation framework for ScrambleBench.

This module provides standardized evaluation metrics and utilities
for assessing model performance across different benchmark types.
Supports both automatic and human evaluation modes.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import re
import difflib
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class EvaluationMode(Enum):
    """Supported evaluation modes."""
    EXACT_MATCH = "exact_match"
    SUBSTRING_MATCH = "substring_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """
    Result of a single evaluation.
    
    Attributes:
        correct: Whether the response was correct
        score: Numeric score (0.0 to 1.0)
        explanation: Human-readable explanation of the result
        metadata: Additional evaluation metadata
    """
    correct: bool
    score: float
    explanation: str
    metadata: Dict[str, Any]


class Evaluator:
    """
    Main evaluation engine for benchmark results.
    
    Provides multiple evaluation modes and standardized metrics
    computation for different types of benchmark tasks.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the evaluator.
        
        Args:
            logger: Logger instance (creates default if None)
        """
        self.logger = logger or logging.getLogger("scramblebench.evaluator")
        self._custom_evaluators: Dict[str, Callable] = {}
    
    def evaluate_response(
        self,
        predicted: str,
        expected: str,
        mode: EvaluationMode = EvaluationMode.EXACT_MATCH,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a model response against expected answer.
        
        Args:
            predicted: Model's predicted response
            expected: Expected correct response
            mode: Evaluation mode to use
            **kwargs: Additional arguments for specific evaluation modes
            
        Returns:
            EvaluationResult containing evaluation outcome
        """
        if mode == EvaluationMode.EXACT_MATCH:
            return self._exact_match_eval(predicted, expected)
        elif mode == EvaluationMode.SUBSTRING_MATCH:
            return self._substring_match_eval(predicted, expected)
        elif mode == EvaluationMode.FUZZY_MATCH:
            threshold = kwargs.get('threshold', 0.8)
            return self._fuzzy_match_eval(predicted, expected, threshold)
        elif mode == EvaluationMode.SEMANTIC_SIMILARITY:
            threshold = kwargs.get('threshold', 0.8)
            return self._semantic_similarity_eval(predicted, expected, threshold)
        elif mode == EvaluationMode.CUSTOM:
            evaluator_name = kwargs.get('evaluator_name')
            if evaluator_name not in self._custom_evaluators:
                raise ValueError(f"Custom evaluator '{evaluator_name}' not found")
            return self._custom_evaluators[evaluator_name](predicted, expected, **kwargs)
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
    
    def _exact_match_eval(self, predicted: str, expected: str) -> EvaluationResult:
        """Exact string match evaluation."""
        # Normalize whitespace and case
        pred_clean = predicted.strip().lower()
        exp_clean = expected.strip().lower()
        
        correct = pred_clean == exp_clean
        score = 1.0 if correct else 0.0
        
        explanation = "Exact match" if correct else f"Expected '{expected}', got '{predicted}'"
        
        return EvaluationResult(
            correct=correct,
            score=score,
            explanation=explanation,
            metadata={
                'predicted_clean': pred_clean,
                'expected_clean': exp_clean
            }
        )
    
    def _substring_match_eval(self, predicted: str, expected: str) -> EvaluationResult:
        """Substring containment evaluation."""
        pred_clean = predicted.strip().lower()
        exp_clean = expected.strip().lower()
        
        correct = exp_clean in pred_clean
        score = 1.0 if correct else 0.0
        
        explanation = (
            f"Expected substring '{expected}' found in response"
            if correct
            else f"Expected substring '{expected}' not found in response"
        )
        
        return EvaluationResult(
            correct=correct,
            score=score,
            explanation=explanation,
            metadata={
                'predicted_clean': pred_clean,
                'expected_clean': exp_clean
            }
        )
    
    def _fuzzy_match_eval(
        self, 
        predicted: str, 
        expected: str, 
        threshold: float
    ) -> EvaluationResult:
        """Fuzzy string matching evaluation using edit distance."""
        pred_clean = predicted.strip().lower()
        exp_clean = expected.strip().lower()
        
        # Calculate similarity using difflib
        similarity = difflib.SequenceMatcher(None, pred_clean, exp_clean).ratio()
        
        correct = similarity >= threshold
        score = similarity
        
        explanation = (
            f"Fuzzy match score: {similarity:.3f} (threshold: {threshold})"
            if correct
            else f"Fuzzy match score: {similarity:.3f} below threshold: {threshold}"
        )
        
        return EvaluationResult(
            correct=correct,
            score=score,
            explanation=explanation,
            metadata={
                'similarity': similarity,
                'threshold': threshold,
                'predicted_clean': pred_clean,
                'expected_clean': exp_clean
            }
        )
    
    def _semantic_similarity_eval(
        self, 
        predicted: str, 
        expected: str, 
        threshold: float
    ) -> EvaluationResult:
        """
        Semantic similarity evaluation.
        
        Note: This is a placeholder implementation. In practice,
        you would use embeddings from a sentence transformer model.
        """
        # Placeholder: use fuzzy matching as proxy for semantic similarity
        # In production, use sentence transformers or similar
        self.logger.warning(
            "Semantic similarity evaluation not fully implemented. "
            "Using fuzzy matching as proxy."
        )
        return self._fuzzy_match_eval(predicted, expected, threshold)
    
    def register_custom_evaluator(
        self, 
        name: str, 
        evaluator_func: Callable[[str, str], EvaluationResult]
    ) -> None:
        """
        Register a custom evaluation function.
        
        Args:
            name: Name to register the evaluator under
            evaluator_func: Function that takes (predicted, expected) and returns EvaluationResult
        """
        self._custom_evaluators[name] = evaluator_func
        self.logger.info(f"Registered custom evaluator: {name}")
    
    def compute_aggregate_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics from a list of evaluation results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing aggregate metrics
        """
        if not results:
            return {}
        
        # Basic metrics
        correct_count = sum(1 for r in results if r.correct)
        total_count = len(results)
        accuracy = correct_count / total_count
        
        # Score statistics
        scores = [r.score for r in results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        metrics = {
            'accuracy': accuracy,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'total_samples': total_count,
            'correct_samples': correct_count,
        }
        
        return metrics
    
    def compute_classification_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics (precision, recall, F1).
        
        Args:
            predictions: List of predicted labels
            ground_truth: List of ground truth labels
            labels: List of label names (optional)
            
        Returns:
            Dictionary containing classification metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        # Add per-class metrics if labels provided
        if labels:
            per_class_precision, per_class_recall, per_class_f1, _ = (
                precision_recall_fscore_support(
                    ground_truth, predictions, labels=labels, zero_division=0
                )
            )
            
            for i, label in enumerate(labels):
                metrics[f'precision_{label}'] = per_class_precision[i]
                metrics[f'recall_{label}'] = per_class_recall[i]
                metrics[f'f1_{label}'] = per_class_f1[i]
        
        return metrics
    
    def extract_answer_pattern(
        self, 
        text: str, 
        pattern: str = r"(?i)(?:answer|solution):\s*(.+?)(?:\n|$)"
    ) -> Optional[str]:
        """
        Extract answer from text using regex pattern.
        
        Args:
            text: Text to extract answer from
            pattern: Regex pattern to use for extraction
            
        Returns:
            Extracted answer string or None if not found
        """
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Normalized answer text
        """
        # Remove extra whitespace, normalize case
        normalized = re.sub(r'\s+', ' ', answer.strip().lower())
        
        # Remove common punctuation that doesn't affect meaning
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized