"""
Metrics computation utilities for benchmark results.

This module provides standardized metrics computation logic that was previously
duplicated across benchmark implementations. It offers both individual metrics
and composite analysis functions.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics
import logging


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    calculate_confidence_intervals: bool = True
    confidence_level: float = 0.95
    include_timing_metrics: bool = True
    include_difficulty_analysis: bool = True
    include_answer_type_analysis: bool = True


class BaseMetricsComputer:
    """
    Base class for computing benchmark metrics.
    
    Provides common metrics computation functionality that can be
    extended by specific benchmark implementations.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the metrics computer.
        
        Args:
            config: Configuration for metrics computation
            logger: Logger instance
        """
        self.config = config or MetricsConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_basic_accuracy_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute basic accuracy-based metrics.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing basic accuracy metrics
        """
        if not results:
            return {
                'score': 0.0,
                'accuracy': 0.0,
                'correct_count': 0,
                'total_count': 0
            }
        
        correct_count = sum(1 for r in results if r.get('correct', False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        metrics = {
            'score': accuracy,  # Primary score for most benchmarks
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
        }
        
        # Add score-based metrics if available
        if any('score' in r for r in results if isinstance(r.get('score'), (int, float))):
            scores = [r.get('score', 0.0) for r in results if isinstance(r.get('score'), (int, float))]
            if scores:
                metrics['mean_score'] = statistics.mean(scores)
                metrics['median_score'] = statistics.median(scores)
                if len(scores) > 1:
                    metrics['score_std'] = statistics.stdev(scores)
        
        return metrics
    
    def compute_timing_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute timing-related metrics.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing timing metrics
        """
        if not self.config.include_timing_metrics:
            return {}
        
        response_times = [
            r.get('response_time', 0.0) for r in results 
            if 'response_time' in r and isinstance(r['response_time'], (int, float))
        ]
        
        if not response_times:
            return {}
        
        metrics = {
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'total_response_time': sum(response_times),
        }
        
        if len(response_times) > 1:
            metrics['response_time_std'] = statistics.stdev(response_times)
        
        return metrics
    
    def compute_difficulty_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute performance by difficulty level.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing difficulty-based performance metrics
        """
        if not self.config.include_difficulty_analysis:
            return {}
        
        difficulty_groups = defaultdict(list)
        
        for result in results:
            # Try to extract difficulty from various locations
            difficulty = None
            if 'difficulty' in result:
                difficulty = result['difficulty']
            elif 'metadata' in result and isinstance(result['metadata'], dict):
                difficulty = result['metadata'].get('difficulty')
            
            if difficulty is not None:
                difficulty_groups[difficulty].append(result)
        
        if not difficulty_groups:
            return {}
        
        difficulty_performance = {}
        for difficulty, group_results in difficulty_groups.items():
            correct_count = sum(1 for r in group_results if r.get('correct', False))
            total_count = len(group_results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            # Calculate average score if available
            scores = [r.get('score', 0.0) for r in group_results if isinstance(r.get('score'), (int, float))]
            avg_score = statistics.mean(scores) if scores else accuracy
            
            difficulty_performance[str(difficulty)] = {
                'accuracy': accuracy,
                'avg_score': avg_score,
                'count': total_count,
                'correct_count': correct_count
            }
        
        return {'difficulty_performance': difficulty_performance}
    
    def compute_answer_type_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute performance by answer type.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing answer type performance metrics
        """
        if not self.config.include_answer_type_analysis:
            return {}
        
        type_groups = defaultdict(list)
        
        for result in results:
            # Try to extract answer type from various locations
            answer_type = None
            if 'answer_type' in result:
                answer_type = result['answer_type']
            elif 'metadata' in result and isinstance(result['metadata'], dict):
                answer_type = result['metadata'].get('answer_type')
            
            if answer_type is not None:
                type_groups[answer_type].append(result)
        
        if not type_groups:
            return {}
        
        answer_type_performance = {}
        for answer_type, group_results in type_groups.items():
            correct_count = sum(1 for r in group_results if r.get('correct', False))
            total_count = len(group_results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            # Calculate average score if available
            scores = [r.get('score', 0.0) for r in group_results if isinstance(r.get('score'), (int, float))]
            avg_score = statistics.mean(scores) if scores else accuracy
            
            answer_type_performance[str(answer_type)] = {
                'accuracy': accuracy,
                'avg_score': avg_score,
                'count': total_count,
                'correct_count': correct_count
            }
        
        return {'answer_type_performance': answer_type_performance}
    
    def compute_confidence_intervals(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute confidence intervals for key metrics.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing confidence interval estimates
        """
        if not self.config.calculate_confidence_intervals or len(results) < 2:
            return {}
        
        # For now, implement a simple bootstrap-style confidence interval
        # In a full implementation, you'd use proper statistical methods
        correct_values = [1 if r.get('correct', False) else 0 for r in results]
        
        if not correct_values:
            return {}
        
        mean_accuracy = statistics.mean(correct_values)
        
        # Simple approximation - in reality you'd use proper CI calculation
        n = len(correct_values)
        se = (mean_accuracy * (1 - mean_accuracy) / n) ** 0.5
        
        # Approximate 95% CI
        margin = 1.96 * se
        ci_lower = max(0.0, mean_accuracy - margin)
        ci_upper = min(1.0, mean_accuracy + margin)
        
        return {
            'confidence_intervals': {
                'accuracy': {
                    'mean': mean_accuracy,
                    'lower': ci_lower,
                    'upper': ci_upper,
                    'level': self.config.confidence_level
                }
            }
        }


class TranslationMetricsComputer(BaseMetricsComputer):
    """Specialized metrics computer for translation benchmarks."""
    
    def compute_translation_specific_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute metrics specific to translation benchmarks.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing translation-specific metrics
        """
        metrics = {}
        
        # Average translation units per problem
        translation_units = [
            r['metadata'].get('translation_units', 0) for r in results
            if 'metadata' in r and isinstance(r['metadata'], dict)
        ]
        
        if translation_units:
            metrics['avg_translation_units'] = statistics.mean(translation_units)
        
        # Language complexity analysis
        complexity_values = [
            r['metadata'].get('language_complexity', 0) for r in results
            if 'metadata' in r and isinstance(r['metadata'], dict)
        ]
        
        if complexity_values:
            metrics['language_complexity'] = complexity_values[0]  # Should be consistent
        
        # Extract source dataset info
        source_datasets = set()
        for result in results:
            if 'metadata' in result and 'source_dataset' in result['metadata']:
                source_datasets.add(result['metadata']['source_dataset'])
        
        if source_datasets:
            metrics['source_datasets'] = list(source_datasets)
        
        return metrics


class LongContextMetricsComputer(BaseMetricsComputer):
    """Specialized metrics computer for long context benchmarks."""
    
    def compute_long_context_specific_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute metrics specific to long context benchmarks.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing long context-specific metrics
        """
        metrics = {}
        
        # Document length analysis
        if any('document_length' in r for r in results):
            doc_lengths = [r['document_length'] for r in results if 'document_length' in r]
            if doc_lengths:
                metrics['document_length_stats'] = {
                    'avg_length': statistics.mean(doc_lengths),
                    'min_length': min(doc_lengths),
                    'max_length': max(doc_lengths),
                    'median_length': statistics.median(doc_lengths)
                }
                
                if len(doc_lengths) > 1:
                    metrics['document_length_stats']['std_length'] = statistics.stdev(doc_lengths)
        
        # Document-level accuracy if available
        if any('accuracy' in r for r in results):
            doc_accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
            if doc_accuracies:
                metrics['document_accuracy'] = {
                    'mean': statistics.mean(doc_accuracies),
                    'median': statistics.median(doc_accuracies)
                }
                
                if len(doc_accuracies) > 1:
                    metrics['document_accuracy']['std'] = statistics.stdev(doc_accuracies)
        
        # Q&A count analysis
        if any('qa_count' in r for r in results):
            qa_counts = [r['qa_count'] for r in results if 'qa_count' in r]
            if qa_counts:
                metrics['qa_stats'] = {
                    'avg_qa_per_document': statistics.mean(qa_counts),
                    'total_qa_pairs': sum(qa_counts),
                    'min_qa_per_document': min(qa_counts),
                    'max_qa_per_document': max(qa_counts)
                }
        
        # Length vs performance correlation
        length_performance_pairs = []
        for result in results:
            if 'document_length' in result and 'accuracy' in result:
                length_performance_pairs.append({
                    'length': result['document_length'],
                    'accuracy': result['accuracy']
                })
        
        if length_performance_pairs:
            metrics['length_vs_performance'] = length_performance_pairs
        
        return metrics


def create_metrics_computer(benchmark_type: str, config: Optional[MetricsConfig] = None) -> BaseMetricsComputer:
    """
    Factory function to create appropriate metrics computer.
    
    Args:
        benchmark_type: Type of benchmark ('translation', 'longcontext', or 'base')
        config: Configuration for metrics computation
        
    Returns:
        Appropriate metrics computer instance
    """
    if benchmark_type == 'translation':
        return TranslationMetricsComputer(config)
    elif benchmark_type == 'longcontext':
        return LongContextMetricsComputer(config)
    else:
        return BaseMetricsComputer(config)