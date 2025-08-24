"""Tests for bootstrap inference functionality in the analysis module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Note: Import paths will need to be updated when the analysis module is implemented
from scramblebench.analysis.bootstrap_inference import (
    BootstrapAnalyzer,
    ConfidenceInterval,
    PermutationTestResult,
    ConfidenceIntervals,
    PermutationTests
)


class TestConfidenceInterval:
    """Test ConfidenceInterval dataclass functionality."""
    
    def test_confidence_interval_creation(self):
        """Test ConfidenceInterval dataclass creation and attributes."""
        ci = ConfidenceInterval(
            lower=0.1,
            upper=0.9,
            level=0.95,
            method="percentile",
            n_bootstrap=1000,
            bootstrap_distribution=np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        )
        
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        assert ci.level == 0.95
        assert ci.method == "percentile"
        assert ci.n_bootstrap == 1000
        assert len(ci.bootstrap_distribution) == 5
    
    def test_confidence_interval_width(self):
        """Test confidence interval width calculation."""
        ci = ConfidenceInterval(
            lower=0.2,
            upper=0.8,
            level=0.95,
            method="percentile",
            n_bootstrap=1000
        )
        
        width = ci.upper - ci.lower
        assert width == 0.6


class TestPermutationTestResult:
    """Test PermutationTestResult dataclass functionality."""
    
    def test_permutation_test_result_creation(self):
        """Test PermutationTestResult dataclass creation and attributes."""
        result = PermutationTestResult(
            statistic=2.5,
            p_value=0.03,
            null_distribution=np.array([1.0, 1.5, 2.0, 2.2, 3.0]),
            method="two-sided",
            n_permutations=1000
        )
        
        assert result.statistic == 2.5
        assert result.p_value == 0.03
        assert len(result.null_distribution) == 5
        assert result.method == "two-sided"
        assert result.n_permutations == 1000
    
    def test_permutation_test_significance(self):
        """Test significance testing with different alpha levels."""
        result = PermutationTestResult(
            statistic=2.5,
            p_value=0.03,
            null_distribution=np.array([1.0, 1.5, 2.0, 2.2, 3.0]),
            method="two-sided",
            n_permutations=1000
        )
        
        # Significant at 0.05 level
        assert result.p_value < 0.05
        # Not significant at 0.01 level
        assert result.p_value > 0.01


class TestBootstrapAnalyzer:
    """Test BootstrapAnalyzer functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for bootstrap analysis."""
        np.random.seed(42)
        return pd.DataFrame({
            'score': np.random.normal(0.8, 0.1, 100),
            'model': np.repeat(['model_a', 'model_b'], 50),
            'difficulty': np.random.choice(['easy', 'medium', 'hard'], 100),
            'response_time': np.random.exponential(100, 100)
        })
    
    @pytest.fixture
    def bootstrap_analyzer(self):
        """Fixture providing BootstrapAnalyzer instance."""
        return BootstrapAnalyzer(
            n_bootstrap=100,  # Small number for fast tests
            confidence_level=0.95,
            random_seed=42
        )
    
    @pytest.fixture
    def mock_model_fit(self):
        """Fixture providing mock model fit object."""
        mock_fit = Mock()
        mock_fit.fixed_effects = {'intercept': 0.5, 'slope': 0.3}
        mock_fit.aic = 100.5
        mock_fit.bic = 105.2
        return mock_fit
    
    def test_bootstrap_analyzer_initialization(self):
        """Test BootstrapAnalyzer initialization with various parameters."""
        analyzer = BootstrapAnalyzer(
            n_bootstrap=500,
            confidence_level=0.90,
            random_seed=123,
            stratify_column='model'
        )
        
        assert analyzer.n_bootstrap == 500
        assert analyzer.confidence_level == 0.90
        assert analyzer.random_seed == 123
        assert analyzer.stratify_column == 'model'
    
    def test_bootstrap_model_parameters_success(self, bootstrap_analyzer, sample_data, mock_model_fit):
        """Test successful bootstrap parameter estimation."""
        with patch.object(bootstrap_analyzer, '_fit_model', return_value=mock_model_fit):
            results = bootstrap_analyzer.bootstrap_model_parameters(
                sample_data,
                model_type='linear',
                parameter_names=['intercept', 'slope']
            )
            
            assert len(results) == 2
            assert 'intercept' in results
            assert 'slope' in results
            assert all(isinstance(ci, ConfidenceInterval) for ci in results.values())
    
    def test_bootstrap_model_parameters_with_failures(self, bootstrap_analyzer, sample_data):
        """Test bootstrap parameter estimation with some failed fits."""
        def mock_fit_with_failures(data):
            # Simulate some failures
            if np.random.random() < 0.2:  # 20% failure rate
                raise ValueError("Model fitting failed")
            mock_fit = Mock()
            mock_fit.fixed_effects = {'intercept': np.random.normal(0.5, 0.1)}
            return mock_fit
        
        with patch.object(bootstrap_analyzer, '_fit_model', side_effect=mock_fit_with_failures):
            results = bootstrap_analyzer.bootstrap_model_parameters(
                sample_data,
                model_type='linear',
                parameter_names=['intercept']
            )
            
            assert 'intercept' in results
            assert isinstance(results['intercept'], ConfidenceInterval)
    
    def test_bootstrap_model_comparison_success(self, bootstrap_analyzer, sample_data):
        """Test successful bootstrap model comparison."""
        mock_fit_1 = Mock()
        mock_fit_1.aic = 100.0
        mock_fit_2 = Mock()
        mock_fit_2.aic = 105.0
        
        with patch.object(bootstrap_analyzer, '_fit_model') as mock_fit:
            mock_fit.side_effect = [mock_fit_1, mock_fit_2] * 100  # For bootstrap samples
            
            result = bootstrap_analyzer.bootstrap_model_comparison(
                sample_data,
                model1_type='linear',
                model2_type='quadratic',
                metric='aic'
            )
            
            assert isinstance(result, PermutationTestResult)
            assert result.statistic == 5.0  # Difference in AIC
            assert 0.0 <= result.p_value <= 1.0
    
    def test_bootstrap_scaling_correlation_list_inputs(self, bootstrap_analyzer):
        """Test bootstrap correlation analysis with list inputs."""
        scores = [0.7, 0.8, 0.9, 0.6, 0.85]
        sizes = [1000, 2000, 4000, 500, 1500]
        
        with patch('numpy.corrcoef', return_value=np.array([[1.0, 0.8], [0.8, 1.0]])):
            result = bootstrap_analyzer.bootstrap_scaling_correlation(
                scores, sizes, correlation_type='pearson'
            )
            
            assert isinstance(result, ConfidenceInterval)
            assert result.method in ['percentile', 'bca']
    
    def test_bootstrap_scaling_correlation_dataframe_input(self, bootstrap_analyzer, sample_data):
        """Test bootstrap correlation analysis with DataFrame input."""
        result = bootstrap_analyzer.bootstrap_scaling_correlation(
            sample_data, 
            score_column='score',
            size_column='response_time',
            correlation_type='spearman'
        )
        
        assert isinstance(result, dict)
        # Should contain results for different difficulty levels if stratified
    
    def test_permutation_test_success(self, bootstrap_analyzer, sample_data):
        """Test permutation test functionality."""
        group1 = sample_data[sample_data['model'] == 'model_a']['score'].values
        group2 = sample_data[sample_data['model'] == 'model_b']['score'].values
        
        result = bootstrap_analyzer.permutation_test(
            group1, group2,
            statistic='mean_difference',
            n_permutations=100
        )
        
        assert isinstance(result, PermutationTestResult)
        assert result.method == 'two-sided'
        assert len(result.null_distribution) == 100
        assert 0.0 <= result.p_value <= 1.0
    
    def test_stratified_bootstrap_sample(self, bootstrap_analyzer, sample_data):
        """Test stratified bootstrap sampling."""
        bootstrap_analyzer.stratify_column = 'model'
        
        boot_sample = bootstrap_analyzer._stratified_bootstrap_sample(sample_data)
        
        assert len(boot_sample) == len(sample_data)
        assert set(boot_sample.columns) == set(sample_data.columns)
        # Should preserve stratification structure
        original_counts = sample_data['model'].value_counts()
        boot_counts = boot_sample['model'].value_counts()
        assert boot_counts.sum() == original_counts.sum()
    
    def test_bootstrap_sample(self, bootstrap_analyzer, sample_data):
        """Test simple bootstrap sampling."""
        boot_sample = bootstrap_analyzer._bootstrap_sample(sample_data)
        
        assert len(boot_sample) == len(sample_data)
        assert set(boot_sample.columns) == set(sample_data.columns)
    
    def test_percentile_ci(self, bootstrap_analyzer):
        """Test percentile confidence interval calculation."""
        estimates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        original = 0.5
        
        lower, upper = bootstrap_analyzer._percentile_ci(estimates, original)
        
        assert lower < upper
        assert lower >= min(estimates)
        assert upper <= max(estimates)
    
    def test_bca_ci(self, bootstrap_analyzer):
        """Test bias-corrected and accelerated confidence interval."""
        estimates = np.random.normal(0.5, 0.1, 1000)
        original = 0.5
        
        lower, upper = bootstrap_analyzer._bca_ci(estimates, original, alpha=0.05)
        
        assert lower < upper
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
    
    def test_fit_model_linear(self, bootstrap_analyzer, sample_data):
        """Test linear model fitting."""
        with patch('scramblebench.analysis.statistical_models.ScalingAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_fit = Mock()
            mock_fit.fixed_effects = {'intercept': 0.5, 'slope': 0.3}
            mock_analyzer.fit_scaling_model.return_value = mock_fit
            mock_analyzer_class.return_value = mock_analyzer
            
            result = bootstrap_analyzer._fit_model(sample_data, 'linear')
            
            assert result == mock_fit
    
    def test_confidence_level_validation(self):
        """Test confidence level validation during initialization."""
        # Valid confidence level
        analyzer = BootstrapAnalyzer(confidence_level=0.95)
        assert analyzer.confidence_level == 0.95
        
        # Test edge cases
        with pytest.raises(ValueError):
            BootstrapAnalyzer(confidence_level=1.1)  # > 1
        
        with pytest.raises(ValueError):
            BootstrapAnalyzer(confidence_level=0.0)  # <= 0
    
    def test_random_seed_reproducibility(self, sample_data):
        """Test that random seed ensures reproducible results."""
        analyzer1 = BootstrapAnalyzer(n_bootstrap=50, random_seed=42)
        analyzer2 = BootstrapAnalyzer(n_bootstrap=50, random_seed=42)
        
        # Mock model fitting to return consistent results
        mock_fit = Mock()
        mock_fit.fixed_effects = {'intercept': 0.5}
        
        with patch.object(analyzer1, '_fit_model', return_value=mock_fit), \
             patch.object(analyzer2, '_fit_model', return_value=mock_fit):
            
            result1 = analyzer1.bootstrap_model_parameters(
                sample_data, 'linear', ['intercept']
            )
            result2 = analyzer2.bootstrap_model_parameters(
                sample_data, 'linear', ['intercept']
            )
            
            # Results should be identical with same seed
            assert result1['intercept'].lower == result2['intercept'].lower
            assert result1['intercept'].upper == result2['intercept'].upper


class TestConfidenceIntervals:
    """Test ConfidenceIntervals utility class."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is valid."""
        assert True
    
    # TODO: Add tests for ConfidenceIntervals utility functions
    # Example test structure:
    # def test_compute_bootstrap_ci(self):
    #     """Test bootstrap confidence interval computation."""
    #     pass
    # 
    # def test_compare_confidence_intervals(self):
    #     """Test confidence interval comparison methods."""
    #     pass
    # 
    # def test_confidence_interval_coverage(self):
    #     """Test confidence interval coverage properties."""
    #     pass


class TestPermutationTests:
    """Test PermutationTests utility class."""
    
    def test_placeholder(self):
        """Placeholder test to ensure test structure is valid."""
        assert True
    
    # TODO: Add tests for PermutationTests utility functions
    # Example test structure:
    # def test_two_sample_permutation_test(self):
    #     """Test two-sample permutation test."""
    #     pass
    # 
    # def test_paired_permutation_test(self):
    #     """Test paired permutation test."""
    #     pass
    # 
    # def test_permutation_test_statistics(self):
    #     """Test different permutation test statistics."""
    #     pass


class TestBootstrapInferenceIntegration:
    """Test integration between bootstrap components."""
    
    @pytest.fixture
    def sample_experiment_data(self):
        """Fixture providing realistic experiment data."""
        np.random.seed(42)
        models = ['gpt-4', 'claude-3', 'gemini-pro']
        difficulties = ['easy', 'medium', 'hard']
        
        data = []
        for model in models:
            for difficulty in difficulties:
                n_samples = 100
                # Simulate different performance by model and difficulty
                base_score = {'gpt-4': 0.9, 'claude-3': 0.85, 'gemini-pro': 0.8}[model]
                difficulty_penalty = {'easy': 0, 'medium': -0.1, 'hard': -0.2}[difficulty]
                
                scores = np.random.beta(
                    (base_score + difficulty_penalty) * 10,
                    (1 - (base_score + difficulty_penalty)) * 10,
                    n_samples
                )
                
                for i, score in enumerate(scores):
                    data.append({
                        'model': model,
                        'difficulty': difficulty,
                        'score': score,
                        'response_time': np.random.exponential(100),
                        'trial_id': i
                    })
        
        return pd.DataFrame(data)
    
    def test_full_bootstrap_analysis_workflow(self, sample_experiment_data):
        """Test complete bootstrap analysis workflow."""
        analyzer = BootstrapAnalyzer(n_bootstrap=50, random_seed=42)
        
        # Mock model fitting for the workflow
        def mock_model_fit(data):
            mock_fit = Mock()
            # Simulate model parameters based on data characteristics
            mean_score = data['score'].mean()
            mock_fit.fixed_effects = {
                'intercept': mean_score,
                'difficulty_effect': -0.1 if 'hard' in data['difficulty'].unique() else 0
            }
            mock_fit.aic = 100 + np.random.normal(0, 5)
            return mock_fit
        
        with patch.object(analyzer, '_fit_model', side_effect=mock_model_fit):
            # Test parameter bootstrapping
            param_results = analyzer.bootstrap_model_parameters(
                sample_experiment_data,
                model_type='mixed_effects',
                parameter_names=['intercept', 'difficulty_effect']
            )
            
            assert len(param_results) == 2
            assert all(isinstance(ci, ConfidenceInterval) for ci in param_results.values())
            
            # Test model comparison
            comparison_result = analyzer.bootstrap_model_comparison(
                sample_experiment_data,
                model1_type='linear',
                model2_type='mixed_effects',
                metric='aic'
            )
            
            assert isinstance(comparison_result, PermutationTestResult)
            assert comparison_result.n_permutations == 50
    
    def test_bootstrap_with_missing_data(self):
        """Test bootstrap analysis handles missing data appropriately."""
        data_with_na = pd.DataFrame({
            'score': [0.8, 0.9, np.nan, 0.7, 0.85],
            'model': ['a', 'a', 'b', 'b', 'a'],
            'size': [100, 200, 300, np.nan, 250]
        })
        
        analyzer = BootstrapAnalyzer(n_bootstrap=10)
        
        # Should handle missing data gracefully
        boot_sample = analyzer._bootstrap_sample(data_with_na)
        assert len(boot_sample) == len(data_with_na)
    
    def test_bootstrap_small_sample_warning(self, sample_analysis_config):
        """Test warning when bootstrap sample size is small."""
        small_data = pd.DataFrame({
            'score': [0.8, 0.9, 0.7],  # Very small sample
            'model': ['a', 'a', 'b']
        })
        
        analyzer = BootstrapAnalyzer(n_bootstrap=100)
        
        mock_fit = Mock()
        mock_fit.fixed_effects = {'intercept': 0.8}
        
        with patch.object(analyzer, '_fit_model', return_value=mock_fit), \
             patch('scramblebench.analysis.bootstrap_inference.logger') as mock_logger:
            
            results = analyzer.bootstrap_model_parameters(
                small_data, 'linear', ['intercept']
            )
            
            # Should log warning about small sample size
            assert mock_logger.warning.called