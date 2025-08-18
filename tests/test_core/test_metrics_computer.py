"""
Tests for metrics computation utilities.
"""

import pytest
from scramblebench.core.metrics_computer import (
    BaseMetricsComputer, TranslationMetricsComputer, LongContextMetricsComputer,
    MetricsConfig, create_metrics_computer
)


class TestMetricsConfig:
    """Test metrics configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.calculate_confidence_intervals is True
        assert config.confidence_level == 0.95
        assert config.include_timing_metrics is True
        assert config.include_difficulty_analysis is True
        assert config.include_answer_type_analysis is True


class TestBaseMetricsComputer:
    """Test base metrics computation functionality."""
    
    def test_basic_accuracy_metrics_empty(self):
        """Test basic metrics with empty results."""
        computer = BaseMetricsComputer()
        metrics = computer.compute_basic_accuracy_metrics([])
        
        assert metrics['score'] == 0.0
        assert metrics['accuracy'] == 0.0
        assert metrics['correct_count'] == 0
        assert metrics['total_count'] == 0
    
    def test_basic_accuracy_metrics_with_data(self):
        """Test basic metrics with sample data."""
        computer = BaseMetricsComputer()
        results = [
            {'correct': True, 'score': 1.0},
            {'correct': False, 'score': 0.0},
            {'correct': True, 'score': 0.8},
            {'correct': False, 'score': 0.2}
        ]
        
        metrics = computer.compute_basic_accuracy_metrics(results)
        
        assert metrics['accuracy'] == 0.5  # 2/4
        assert metrics['correct_count'] == 2
        assert metrics['total_count'] == 4
        assert metrics['score'] == 0.5
        assert metrics['mean_score'] == 0.5  # (1.0 + 0.0 + 0.8 + 0.2) / 4
    
    def test_timing_metrics(self):
        """Test timing metrics computation."""
        computer = BaseMetricsComputer()
        results = [
            {'response_time': 1.0},
            {'response_time': 2.0},
            {'response_time': 1.5},
            {'other_field': 'value'}  # Should be ignored
        ]
        
        metrics = computer.compute_timing_metrics(results)
        
        assert metrics['avg_response_time'] == 1.5
        assert metrics['median_response_time'] == 1.5
        assert metrics['min_response_time'] == 1.0
        assert metrics['max_response_time'] == 2.0
        assert metrics['total_response_time'] == 4.5
    
    def test_timing_metrics_disabled(self):
        """Test timing metrics when disabled."""
        config = MetricsConfig(include_timing_metrics=False)
        computer = BaseMetricsComputer(config)
        results = [{'response_time': 1.0}]
        
        metrics = computer.compute_timing_metrics(results)
        assert metrics == {}
    
    def test_difficulty_analysis(self):
        """Test difficulty-based performance analysis."""
        computer = BaseMetricsComputer()
        results = [
            {'correct': True, 'difficulty': 1, 'score': 1.0},
            {'correct': True, 'difficulty': 1, 'score': 0.9},
            {'correct': False, 'difficulty': 3, 'score': 0.2},
            {'correct': True, 'difficulty': 3, 'score': 0.8},
            {'correct': False, 'metadata': {'difficulty': 5}, 'score': 0.1}
        ]
        
        metrics = computer.compute_difficulty_analysis(results)
        
        assert 'difficulty_performance' in metrics
        perf = metrics['difficulty_performance']
        
        # Difficulty 1: 2 correct, accuracy = 1.0, avg_score = 0.95
        assert perf['1']['accuracy'] == 1.0
        assert perf['1']['avg_score'] == 0.95
        assert perf['1']['count'] == 2
        
        # Difficulty 3: 1/2 correct, accuracy = 0.5, avg_score = 0.5
        assert perf['3']['accuracy'] == 0.5
        assert perf['3']['avg_score'] == 0.5
        
        # Difficulty 5: 0/1 correct, accuracy = 0.0, avg_score = 0.1
        assert perf['5']['accuracy'] == 0.0
        assert perf['5']['avg_score'] == 0.1
    
    def test_answer_type_analysis(self):
        """Test answer type performance analysis."""
        computer = BaseMetricsComputer()
        results = [
            {'correct': True, 'answer_type': 'yes_no', 'score': 1.0},
            {'correct': False, 'answer_type': 'yes_no', 'score': 0.0},
            {'correct': True, 'answer_type': 'numeric', 'score': 0.9},
            {'correct': True, 'metadata': {'answer_type': 'extractive'}, 'score': 0.8}
        ]
        
        metrics = computer.compute_answer_type_analysis(results)
        
        assert 'answer_type_performance' in metrics
        perf = metrics['answer_type_performance']
        
        # yes_no: 1/2 correct, accuracy = 0.5, avg_score = 0.5
        assert perf['yes_no']['accuracy'] == 0.5
        assert perf['yes_no']['avg_score'] == 0.5
        
        # numeric: 1/1 correct, accuracy = 1.0, avg_score = 0.9
        assert perf['numeric']['accuracy'] == 1.0
        assert perf['numeric']['avg_score'] == 0.9
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        computer = BaseMetricsComputer()
        results = [
            {'correct': True},
            {'correct': True},
            {'correct': False},
            {'correct': True}
        ]
        
        metrics = computer.compute_confidence_intervals(results)
        
        assert 'confidence_intervals' in metrics
        ci = metrics['confidence_intervals']['accuracy']
        
        assert 'mean' in ci
        assert 'lower' in ci
        assert 'upper' in ci
        assert ci['level'] == 0.95
        assert 0.0 <= ci['lower'] <= ci['mean'] <= ci['upper'] <= 1.0
    
    def test_confidence_intervals_insufficient_data(self):
        """Test confidence intervals with insufficient data."""
        computer = BaseMetricsComputer()
        results = [{'correct': True}]  # Only one result
        
        metrics = computer.compute_confidence_intervals(results)
        assert metrics == {}


class TestTranslationMetricsComputer:
    """Test translation-specific metrics computation."""
    
    def test_translation_specific_metrics(self):
        """Test translation-specific metrics calculation."""
        computer = TranslationMetricsComputer()
        results = [
            {
                'metadata': {
                    'translation_units': 10,
                    'language_complexity': 5,
                    'source_dataset': 'math_problems'
                }
            },
            {
                'metadata': {
                    'translation_units': 15,
                    'language_complexity': 5,
                    'source_dataset': 'math_problems'
                }
            }
        ]
        
        metrics = computer.compute_translation_specific_metrics(results)
        
        assert metrics['avg_translation_units'] == 12.5
        assert metrics['language_complexity'] == 5
        assert 'math_problems' in metrics['source_datasets']
    
    def test_translation_metrics_empty(self):
        """Test translation metrics with empty metadata."""
        computer = TranslationMetricsComputer()
        results = [{'other': 'data'}]
        
        metrics = computer.compute_translation_specific_metrics(results)
        assert len(metrics) == 0  # No translation-specific data found


class TestLongContextMetricsComputer:
    """Test long context-specific metrics computation."""
    
    def test_long_context_specific_metrics(self):
        """Test long context-specific metrics calculation."""
        computer = LongContextMetricsComputer()
        results = [
            {
                'document_length': 1000,
                'accuracy': 0.8,
                'qa_count': 5
            },
            {
                'document_length': 2000,
                'accuracy': 0.6,
                'qa_count': 8
            }
        ]
        
        metrics = computer.compute_long_context_specific_metrics(results)
        
        # Document length stats
        assert metrics['document_length_stats']['avg_length'] == 1500
        assert metrics['document_length_stats']['min_length'] == 1000
        assert metrics['document_length_stats']['max_length'] == 2000
        assert metrics['document_length_stats']['median_length'] == 1500
        
        # Document accuracy
        assert metrics['document_accuracy']['mean'] == 0.7
        assert metrics['document_accuracy']['median'] == 0.7
        
        # Q&A stats
        assert metrics['qa_stats']['avg_qa_per_document'] == 6.5
        assert metrics['qa_stats']['total_qa_pairs'] == 13
        assert metrics['qa_stats']['min_qa_per_document'] == 5
        assert metrics['qa_stats']['max_qa_per_document'] == 8
        
        # Length vs performance
        assert len(metrics['length_vs_performance']) == 2
        assert metrics['length_vs_performance'][0]['length'] == 1000
        assert metrics['length_vs_performance'][0]['accuracy'] == 0.8
    
    def test_long_context_metrics_partial_data(self):
        """Test long context metrics with partial data."""
        computer = LongContextMetricsComputer()
        results = [
            {'document_length': 1000},  # No accuracy
            {'accuracy': 0.8}  # No document_length
        ]
        
        metrics = computer.compute_long_context_specific_metrics(results)
        
        # Should only have document length stats, no accuracy or correlation
        assert 'document_length_stats' in metrics
        assert 'document_accuracy' not in metrics
        assert 'length_vs_performance' not in metrics


class TestCreateMetricsComputer:
    """Test metrics computer factory function."""
    
    def test_create_translation_computer(self):
        """Test creating translation metrics computer."""
        computer = create_metrics_computer('translation')
        assert isinstance(computer, TranslationMetricsComputer)
    
    def test_create_longcontext_computer(self):
        """Test creating long context metrics computer."""
        computer = create_metrics_computer('longcontext')
        assert isinstance(computer, LongContextMetricsComputer)
    
    def test_create_base_computer(self):
        """Test creating base metrics computer."""
        computer = create_metrics_computer('base')
        assert isinstance(computer, BaseMetricsComputer)
        
        # Test default fallback
        computer = create_metrics_computer('unknown')
        assert isinstance(computer, BaseMetricsComputer)
    
    def test_create_computer_with_config(self):
        """Test creating computer with custom config."""
        config = MetricsConfig(include_timing_metrics=False)
        computer = create_metrics_computer('base', config)
        assert computer.config.include_timing_metrics is False