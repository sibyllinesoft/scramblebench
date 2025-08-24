"""
Tests for statistical analysis functionality.

Tests statistical analysis, significance testing, A/B testing,
and language dependency analysis for experiment results.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from scipy import stats

from scramblebench.experiment_tracking.statistics import (
    StatisticalAnalyzer, SignificanceTest, ABTestResult, LanguageDependencyAnalysis
)


class TestSignificanceTest:
    """Test cases for SignificanceTest data class."""
    
    def test_significance_test_creation(self):
        """Test creating SignificanceTest instance."""
        test = SignificanceTest(
            test_name="t_test",
            p_value=0.001,
            statistic=3.45,
            significant=True,
            effect_size=0.75,
            confidence_interval=(0.6, 0.9)
        )
        
        assert test.test_name == "t_test"
        assert test.p_value == 0.001
        assert test.statistic == 3.45
        assert test.significant is True
        assert test.effect_size == 0.75
        assert test.confidence_interval == (0.6, 0.9)
    
    def test_significance_test_significance_determination(self):
        """Test significance determination based on p-value."""
        significant_test = SignificanceTest(
            test_name="test",
            p_value=0.01,
            statistic=2.5,
            significant=True
        )
        
        not_significant_test = SignificanceTest(
            test_name="test",
            p_value=0.1,
            statistic=1.2,
            significant=False
        )
        
        assert significant_test.significant is True
        assert not_significant_test.significant is False


class TestABTestResult:
    """Test cases for ABTestResult data class."""
    
    def test_ab_test_result_creation(self):
        """Test creating ABTestResult instance."""
        result = ABTestResult(
            control_group="original",
            treatment_group="scrambled",
            control_mean=0.85,
            treatment_mean=0.72,
            effect_size=-0.13,
            p_value=0.002,
            significant=True,
            confidence_interval=(-0.2, -0.06),
            sample_size_control=100,
            sample_size_treatment=100
        )
        
        assert result.control_group == "original"
        assert result.treatment_group == "scrambled"
        assert result.control_mean == 0.85
        assert result.treatment_mean == 0.72
        assert result.effect_size == -0.13
        assert result.significant is True


class TestLanguageDependencyAnalysis:
    """Test cases for LanguageDependencyAnalysis data class."""
    
    def test_language_dependency_analysis_creation(self):
        """Test creating LanguageDependencyAnalysis instance."""
        analysis = LanguageDependencyAnalysis(
            model="gpt-4",
            threshold_level=3,
            performance_drop=0.25,
            dependency_score=0.8,
            confidence=0.95
        )
        
        assert analysis.model == "gpt-4"
        assert analysis.threshold_level == 3
        assert analysis.performance_drop == 0.25
        assert analysis.dependency_score == 0.8
        assert analysis.confidence == 0.95


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer functionality."""
    
    def test_statistical_analyzer_initialization(self, mock_db_manager):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer(mock_db_manager)
        
        assert analyzer.db_manager is mock_db_manager
        assert hasattr(analyzer, 'alpha')
        assert analyzer.alpha == 0.05  # Default significance level
    
    @pytest.mark.asyncio
    async def test_compute_performance_metrics(self, statistical_analyzer, sample_evaluation_results):
        """Test computing basic performance metrics."""
        experiment_id = "test-exp-metrics"
        
        await statistical_analyzer.compute_performance_metrics(experiment_id, sample_evaluation_results)
        
        # Verify database save was called
        statistical_analyzer.db_manager.save_performance_metrics.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_run_t_test_comparison(self, statistical_analyzer):
        """Test running t-test for model comparison."""
        # Mock data for two groups
        group_a_scores = np.random.normal(0.8, 0.1, 50)
        group_b_scores = np.random.normal(0.7, 0.1, 50)
        
        with patch('scipy.stats.ttest_ind') as mock_ttest:
            mock_ttest.return_value = (2.5, 0.01)  # statistic, p-value
            
            result = await statistical_analyzer.run_t_test_comparison(
                group_a_scores, group_b_scores, 
                "model_a", "model_b"
            )
            
            assert isinstance(result, SignificanceTest)
            assert result.test_name == "t_test"
            assert result.statistic == 2.5
            assert result.p_value == 0.01
            assert result.significant is True  # p < 0.05
    
    @pytest.mark.asyncio
    async def test_run_mann_whitney_test(self, statistical_analyzer):
        """Test running Mann-Whitney U test for non-parametric comparison."""
        # Mock data
        group_a_scores = np.array([0.9, 0.8, 0.85, 0.75, 0.82])
        group_b_scores = np.array([0.7, 0.65, 0.72, 0.68, 0.71])
        
        with patch('scipy.stats.mannwhitneyu') as mock_mwu:
            mock_mwu.return_value = (5.0, 0.02)  # statistic, p-value
            
            result = await statistical_analyzer.run_mann_whitney_test(
                group_a_scores, group_b_scores,
                "original", "scrambled"
            )
            
            assert isinstance(result, SignificanceTest)
            assert result.test_name == "mann_whitney"
            assert result.statistic == 5.0
            assert result.p_value == 0.02
            assert result.significant is True
    
    @pytest.mark.asyncio
    async def test_calculate_effect_size_cohens_d(self, statistical_analyzer):
        """Test calculating Cohen's d effect size."""
        group_a = np.array([0.8, 0.82, 0.78, 0.85, 0.79])
        group_b = np.array([0.7, 0.72, 0.68, 0.75, 0.71])
        
        effect_size = await statistical_analyzer.calculate_effect_size(group_a, group_b, method="cohens_d")
        
        assert isinstance(effect_size, float)
        assert effect_size > 0  # group_a should have higher mean
    
    @pytest.mark.asyncio
    async def test_bootstrap_confidence_interval(self, statistical_analyzer):
        """Test bootstrap confidence interval calculation."""
        data = np.random.normal(0.8, 0.1, 100)
        
        ci = await statistical_analyzer.bootstrap_confidence_interval(
            data, confidence_level=0.95, n_bootstrap=1000
        )
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound
        assert 0.7 < ci[0] < 0.9  # Reasonable bounds for mean ~0.8
        assert 0.7 < ci[1] < 0.9
    
    @pytest.mark.asyncio
    async def test_analyze_thresholds_simple(self, statistical_analyzer):
        """Test threshold analysis for language dependency."""
        experiment_id = "threshold-test"
        
        # Mock evaluation results with performance degradation
        mock_results = Mock()
        mock_results.results = []
        
        # Simulate performance drop at scramble level 3
        for level in range(8):  # 8 scramble levels
            for i in range(10):  # 10 samples per level
                result = Mock()
                result.model_id = "test-model"
                result.transform = f"scramble_{level}"
                result.scramble_level = level
                result.is_correct = level < 3 or (level >= 3 and i % 3 == 0)  # Performance drops at level 3
                mock_results.results.append(result)
        
        with patch.object(statistical_analyzer, '_detect_threshold', return_value=3) as mock_detect:
            await statistical_analyzer.analyze_thresholds(experiment_id, mock_results)
            
            mock_detect.assert_called()
            statistical_analyzer.db_manager.save_threshold_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_threshold_change_point(self, statistical_analyzer):
        """Test threshold detection using change point analysis."""
        # Create data with clear change point at position 3
        performance_data = np.array([0.9, 0.88, 0.91, 0.6, 0.62, 0.58, 0.61, 0.59])
        
        threshold = await statistical_analyzer._detect_threshold(performance_data, method="changepoint")
        
        assert threshold == 3  # Should detect change at position 3
    
    @pytest.mark.asyncio
    async def test_detect_threshold_statistical(self, statistical_analyzer):
        """Test threshold detection using statistical method."""
        # Create data with statistical difference after position 2
        performance_data = np.array([0.9, 0.88, 0.91, 0.6, 0.62, 0.58, 0.61, 0.59])
        
        with patch('scipy.stats.ttest_ind') as mock_ttest:
            # Mock significant difference
            mock_ttest.return_value = (3.2, 0.01)
            
            threshold = await statistical_analyzer._detect_threshold(performance_data, method="statistical")
            
            assert isinstance(threshold, int)
            assert 0 <= threshold < len(performance_data)
    
    @pytest.mark.asyncio
    async def test_run_ab_test(self, statistical_analyzer, mock_experiment_results_data):
        """Test A/B test analysis between conditions."""
        experiment_id = "ab-test-exp"
        
        # Filter data for A/B test
        control_data = mock_experiment_results_data[
            mock_experiment_results_data['condition'] == 'original'
        ]['score'].values
        
        treatment_data = mock_experiment_results_data[
            mock_experiment_results_data['condition'] == 'scrambled'
        ]['score'].values
        
        result = await statistical_analyzer.run_ab_test(
            experiment_id,
            control_data,
            treatment_data,
            control_name="original",
            treatment_name="scrambled"
        )
        
        assert isinstance(result, ABTestResult)
        assert result.control_group == "original"
        assert result.treatment_group == "scrambled"
        assert isinstance(result.p_value, float)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.significant, bool)
    
    @pytest.mark.asyncio
    async def test_run_significance_tests_comprehensive(self, statistical_analyzer, sample_evaluation_results):
        """Test comprehensive significance testing suite."""
        experiment_id = "significance-test-exp"
        
        with patch.object(statistical_analyzer, 'run_t_test_comparison') as mock_ttest, \
             patch.object(statistical_analyzer, 'run_mann_whitney_test') as mock_mwu, \
             patch.object(statistical_analyzer, 'run_ab_test') as mock_ab:
            
            # Mock test results
            mock_ttest.return_value = SignificanceTest("t_test", 0.01, 2.5, True)
            mock_mwu.return_value = SignificanceTest("mann_whitney", 0.02, 5.0, True)
            mock_ab.return_value = ABTestResult("original", "scrambled", 0.8, 0.7, -0.1, 0.01, True, (-0.2, 0), 50, 50)
            
            await statistical_analyzer.run_significance_tests(experiment_id, sample_evaluation_results)
            
            # Verify various tests were run
            assert mock_ttest.call_count > 0
            assert mock_mwu.call_count > 0
            assert mock_ab.call_count > 0
    
    @pytest.mark.asyncio
    async def test_analyze_language_dependency(self, statistical_analyzer, mock_language_dependency_data):
        """Test language dependency analysis."""
        experiment_id = "lang-dep-test"
        
        with patch.object(statistical_analyzer, '_detect_threshold', return_value=3):
            results = await statistical_analyzer.analyze_language_dependency(
                experiment_id, mock_language_dependency_data
            )
            
            assert isinstance(results, list)
            assert len(results) > 0
            
            for result in results:
                assert isinstance(result, LanguageDependencyAnalysis)
                assert result.model in ["gpt-4", "claude-3", "llama-2"]
                assert 0 <= result.dependency_score <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_statistical_power(self, statistical_analyzer):
        """Test statistical power calculation."""
        effect_size = 0.5
        sample_size = 50
        alpha = 0.05
        
        power = await statistical_analyzer.calculate_statistical_power(
            effect_size, sample_size, alpha
        )
        
        assert isinstance(power, float)
        assert 0 <= power <= 1
        # With effect size 0.5 and n=50, power should be reasonably high
        assert power > 0.5
    
    @pytest.mark.asyncio
    async def test_multiple_comparison_correction(self, statistical_analyzer):
        """Test multiple comparison correction (Bonferroni, FDR)."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
        
        # Bonferroni correction
        corrected_bonf = await statistical_analyzer.multiple_comparison_correction(
            p_values, method="bonferroni"
        )
        
        assert len(corrected_bonf) == len(p_values)
        assert all(corrected_bonf[i] >= p_values[i] for i in range(len(p_values)))  # Should increase p-values
        
        # FDR correction
        corrected_fdr = await statistical_analyzer.multiple_comparison_correction(
            p_values, method="fdr"
        )
        
        assert len(corrected_fdr) == len(p_values)
        # FDR should be less conservative than Bonferroni
        assert all(corrected_fdr[i] <= corrected_bonf[i] for i in range(len(p_values)))
    
    @pytest.mark.asyncio
    async def test_bayesian_analysis(self, statistical_analyzer):
        """Test Bayesian analysis for model comparison."""
        group_a_scores = np.random.normal(0.8, 0.1, 50)
        group_b_scores = np.random.normal(0.7, 0.1, 50)
        
        # Mock Bayesian analysis result
        with patch.object(statistical_analyzer, '_run_bayesian_test') as mock_bayes:
            mock_bayes.return_value = {
                "bayes_factor": 15.2,
                "credible_interval": (0.05, 0.15),
                "probability_superior": 0.95
            }
            
            result = await statistical_analyzer.bayesian_analysis(
                group_a_scores, group_b_scores,
                "model_a", "model_b"
            )
            
            assert "bayes_factor" in result
            assert "credible_interval" in result
            assert "probability_superior" in result
            
            # Strong evidence (BF > 10) for difference
            assert result["bayes_factor"] > 10


class TestStatisticalAnalyzerIntegration:
    """Integration tests for statistical analysis with database."""
    
    @pytest.mark.asyncio
    async def test_full_statistical_pipeline(self, statistical_analyzer, mock_experiment_results_data):
        """Test complete statistical analysis pipeline."""
        experiment_id = "full-pipeline-test"
        
        # Create mock evaluation results from dataframe
        mock_results = Mock()
        mock_results.results = []
        
        for _, row in mock_experiment_results_data.iterrows():
            result = Mock()
            result.model_id = row['model']
            result.transform = row['condition']
            result.score = row['score']
            result.is_correct = row['score'] > 0.75
            result.latency_ms = row['latency_ms']
            result.cost_usd = row['cost_usd']
            mock_results.results.append(result)
        
        with patch.object(statistical_analyzer, '_detect_threshold', return_value=2):
            # Run full analysis pipeline
            await statistical_analyzer.compute_performance_metrics(experiment_id, mock_results)
            await statistical_analyzer.analyze_thresholds(experiment_id, mock_results)
            await statistical_analyzer.run_significance_tests(experiment_id, mock_results)
            
            # Verify database calls were made
            statistical_analyzer.db_manager.save_performance_metrics.assert_called()
            statistical_analyzer.db_manager.save_threshold_analysis.assert_called()
            statistical_analyzer.db_manager.save_significance_test_results.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_analysis(self, statistical_analyzer, sample_evaluation_results):
        """Test error handling in statistical analysis."""
        experiment_id = "error-test-exp"
        
        # Mock database to raise error
        statistical_analyzer.db_manager.save_performance_metrics.side_effect = Exception("Database error")
        
        # Should handle error gracefully without raising
        await statistical_analyzer.compute_performance_metrics(experiment_id, sample_evaluation_results)
        
        # Verify error was logged but not raised
        statistical_analyzer.db_manager.save_performance_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_empty_results_handling(self, statistical_analyzer):
        """Test handling of empty or insufficient results."""
        experiment_id = "empty-results-test"
        
        # Empty results
        empty_results = Mock()
        empty_results.results = []
        
        # Should handle gracefully
        await statistical_analyzer.compute_performance_metrics(experiment_id, empty_results)
        
        # Verify appropriate handling (may or may not call database depending on implementation)
    
    @pytest.mark.asyncio
    async def test_single_model_analysis(self, statistical_analyzer):
        """Test analysis with only single model (no comparisons possible)."""
        experiment_id = "single-model-test"
        
        # Results with only one model
        single_model_results = Mock()
        single_model_results.results = []
        
        for i in range(20):
            result = Mock()
            result.model_id = "single-model"
            result.transform = "original"
            result.score = 0.8 + np.random.normal(0, 0.05)
            result.is_correct = result.score > 0.75
            single_model_results.results.append(result)
        
        # Should handle single model case
        await statistical_analyzer.compute_performance_metrics(experiment_id, single_model_results)
        await statistical_analyzer.run_significance_tests(experiment_id, single_model_results)
        
        # Verify basic metrics were computed even without comparisons
        statistical_analyzer.db_manager.save_performance_metrics.assert_called()


class TestStatisticalUtilities:
    """Test cases for statistical utility functions."""
    
    @pytest.mark.asyncio
    async def test_normality_testing(self, statistical_analyzer):
        """Test normality testing for choosing appropriate statistical tests."""
        # Normal data
        normal_data = np.random.normal(0.8, 0.1, 100)
        is_normal = await statistical_analyzer._test_normality(normal_data)
        assert isinstance(is_normal, bool)
        # Should likely be normal with large sample
        
        # Non-normal data (uniform)
        uniform_data = np.random.uniform(0.5, 1.0, 100)
        is_uniform_normal = await statistical_analyzer._test_normality(uniform_data)
        assert isinstance(is_uniform_normal, bool)
    
    @pytest.mark.asyncio
    async def test_outlier_detection(self, statistical_analyzer):
        """Test outlier detection in performance data."""
        # Data with outliers
        data_with_outliers = np.concatenate([
            np.random.normal(0.8, 0.05, 95),  # Normal data
            np.array([0.1, 0.2, 0.3, 0.95, 0.99])  # Outliers
        ])
        
        outliers = await statistical_analyzer.detect_outliers(data_with_outliers)
        
        assert isinstance(outliers, np.ndarray)
        assert len(outliers) > 0  # Should detect some outliers
        assert len(outliers) < len(data_with_outliers)  # But not all points
    
    @pytest.mark.asyncio
    async def test_sample_size_calculation(self, statistical_analyzer):
        """Test sample size calculation for desired statistical power."""
        effect_size = 0.5
        power = 0.8
        alpha = 0.05
        
        sample_size = await statistical_analyzer.calculate_required_sample_size(
            effect_size, power, alpha
        )
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        # For medium effect size (0.5), should need reasonable sample size
        assert 20 < sample_size < 200
    
    @pytest.mark.asyncio
    async def test_confidence_interval_coverage(self, statistical_analyzer):
        """Test confidence interval coverage properties."""
        # Generate multiple samples and check CI coverage
        true_mean = 0.8
        n_samples = 100
        coverage_count = 0
        n_experiments = 50
        
        for _ in range(n_experiments):
            sample = np.random.normal(true_mean, 0.1, n_samples)
            ci = await statistical_analyzer.bootstrap_confidence_interval(
                sample, confidence_level=0.95
            )
            
            if ci[0] <= true_mean <= ci[1]:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_experiments
        # Should be close to 0.95 (allow some variance)
        assert 0.85 <= coverage_rate <= 1.0