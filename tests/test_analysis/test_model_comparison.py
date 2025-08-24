"""
Tests for model_comparison.py

Tests the model comparison and selection framework including:
- ModelComparison: Main comparison class
- ModelSelectionCriteria: Selection criteria dataclass
- AICWeights: AIC weighting functionality
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import scipy.stats as stats

from scramblebench.analysis.model_comparison import (
    ModelComparison,
    ModelSelectionCriteria, 
    AICWeights
)

from scramblebench.analysis.statistical_models import ModelFit


class TestAICWeights:
    """Tests for AICWeights dataclass"""
    
    def test_aic_weights_creation(self):
        """Test AICWeights object creation"""
        weights = AICWeights(
            models=['model1', 'model2', 'model3'],
            aic_values=[100.0, 105.0, 110.0],
            delta_aic=[0.0, 5.0, 10.0], 
            weights=[0.6, 0.3, 0.1],
            cumulative_weights=[0.6, 0.9, 1.0]
        )
        
        assert weights.models == ['model1', 'model2', 'model3']
        assert weights.aic_values == [100.0, 105.0, 110.0]
        assert weights.weights == [0.6, 0.3, 0.1]
        assert abs(sum(weights.weights) - 1.0) < 1e-6
        assert weights.cumulative_weights == [0.6, 0.9, 1.0]
        
    def test_is_clear_winner(self):
        """Test clear winner detection"""
        # Strong winner case
        strong_weights = AICWeights(
            models=['winner', 'loser1', 'loser2'],
            aic_values=[100.0, 115.0, 120.0],
            delta_aic=[0.0, 15.0, 20.0],
            weights=[0.95, 0.04, 0.01],
            cumulative_weights=[0.95, 0.99, 1.0]
        )
        
        is_clear, winner = strong_weights.is_clear_winner(0.9)
        assert is_clear is True
        assert winner == 'winner'
        
        # No clear winner case
        weak_weights = AICWeights(
            models=['model1', 'model2', 'model3'],
            aic_values=[100.0, 101.0, 102.0],
            delta_aic=[0.0, 1.0, 2.0],
            weights=[0.5, 0.3, 0.2],
            cumulative_weights=[0.5, 0.8, 1.0]
        )
        
        is_clear, winner = weak_weights.is_clear_winner(0.9)
        assert is_clear is False
        assert winner == 'model1'  # Still returns best model
        
    def test_get_top_models(self):
        """Test top models selection by cumulative weight"""
        weights = AICWeights(
            models=['model1', 'model2', 'model3', 'model4'],
            aic_values=[100.0, 105.0, 110.0, 120.0],
            delta_aic=[0.0, 5.0, 10.0, 20.0],
            weights=[0.6, 0.25, 0.1, 0.05],
            cumulative_weights=[0.6, 0.85, 0.95, 1.0]
        )
        
        # Get top models covering 90% of weight
        top_90 = weights.get_top_models(0.9)
        assert len(top_90) == 2  # First two models cover 85%, need third for 95%
        assert top_90[0] == 'model1'
        assert top_90[1] == 'model2'
        
        # Get top models covering 95%
        top_95 = weights.get_top_models(0.95)
        assert len(top_95) == 3
        assert top_95 == ['model1', 'model2', 'model3']


class TestModelSelectionCriteria:
    """Tests for ModelSelectionCriteria dataclass"""
    
    def test_model_selection_criteria_creation(self):
        """Test ModelSelectionCriteria object creation"""
        criteria = ModelSelectionCriteria(
            model_name="test_model",
            aic=100.5,
            bic=105.2,
            aic_weight=0.6,
            bic_weight=0.7,
            aic_rank=1,
            bic_rank=1,
            cv_score=0.85,
            cv_rank=2,
            evidence_ratio_aic=1.0,
            evidence_ratio_bic=1.0
        )
        
        assert criteria.model_name == "test_model"
        assert criteria.aic == 100.5
        assert criteria.bic == 105.2
        assert criteria.aic_weight == 0.6
        assert criteria.bic_weight == 0.7
        assert criteria.composite_rank == 1.5  # (1+1+2)/3
        
    def test_composite_rank_calculation(self):
        """Test composite rank calculation with missing CV"""
        criteria = ModelSelectionCriteria(
            model_name="test",
            aic=100.0,
            bic=105.0,
            aic_weight=0.5,
            bic_weight=0.4,
            aic_rank=2,
            bic_rank=3,
            cv_score=None,
            cv_rank=None,
            evidence_ratio_aic=2.0,
            evidence_ratio_bic=2.5
        )
        
        # With no CV rank, should be average of AIC and BIC ranks
        assert criteria.composite_rank == 2.5  # (2+3)/2


class TestModelComparison:
    """Tests for ModelComparison class"""
    
    @pytest.fixture
    def sample_model_fits(self):
        """Create sample model fits for testing"""
        fits = {
            'linear': ModelFit(
                model_name="linear",
                aic=120.0,
                bic=125.0,
                log_likelihood=-58.0,
                fixed_effects={"intercept": 1.0, "slope": 0.1},
                converged=True,
                n_observations=100,
                deviance=116.0
            ),
            'segmented': ModelFit(
                model_name="segmented",
                aic=115.0,
                bic=123.0,
                log_likelihood=-55.0,
                fixed_effects={"intercept": 1.0, "slope1": 0.05, "slope2": 0.25, "breakpoint": 9.0},
                converged=True,
                n_observations=100,
                deviance=110.0
            ),
            'gam': ModelFit(
                model_name="gam",
                aic=118.0,
                bic=128.0,
                log_likelihood=-56.0,
                fixed_effects={"smooth_term": "nonparametric"},
                converged=True,
                n_observations=100,
                deviance=112.0
            )
        }
        return fits
        
    @pytest.fixture
    def non_converged_fits(self):
        """Create model fits with convergence issues"""
        fits = {
            'good_model': ModelFit(
                model_name="good_model",
                aic=100.0,
                bic=105.0,
                log_likelihood=-48.0,
                fixed_effects={"intercept": 1.0},
                converged=True,
                n_observations=100
            ),
            'bad_model': ModelFit(
                model_name="bad_model",
                aic=200.0,
                bic=210.0,
                log_likelihood=-98.0,
                fixed_effects={"intercept": 0.5},
                converged=False,
                n_observations=100
            )
        }
        return fits
        
    def test_model_comparison_init(self):
        """Test ModelComparison initialization"""
        comparison = ModelComparison(alpha=0.01)
        assert comparison.alpha == 0.01
        
    def test_compare_models_success(self, sample_model_fits):
        """Test successful model comparison"""
        comparison = ModelComparison(alpha=0.05)
        result = comparison.compare_models(sample_model_fits)
        
        # Verify result structure
        assert 'n_models_compared' in result
        assert 'information_criteria' in result
        assert 'aic_weights' in result
        assert 'bic_weights' in result
        assert 'model_rankings' in result
        assert 'best_model' in result
        
        assert result['n_models_compared'] == 3
        
        # Verify information criteria
        ic = result['information_criteria']
        assert 'linear' in ic
        assert 'segmented' in ic
        assert 'gam' in ic
        
        # Verify delta AIC is calculated correctly
        assert ic['segmented']['delta_aic'] == 0.0  # Best model
        assert ic['gam']['delta_aic'] == 3.0  # 118 - 115
        assert ic['linear']['delta_aic'] == 5.0  # 120 - 115
        
        # Verify AIC weights structure
        aic_weights = result['aic_weights']
        assert isinstance(aic_weights, AICWeights)
        assert len(aic_weights.models) == 3
        assert aic_weights.models[0] == 'segmented'  # Best model first
        
        # Weights should sum to 1
        assert abs(sum(aic_weights.weights) - 1.0) < 1e-6
        
        # Best model should be segmented (lowest AIC)
        assert result['best_model']['model_name'] == 'segmented'
        
    def test_compare_models_insufficient_models(self):
        """Test comparison with insufficient models"""
        comparison = ModelComparison()
        
        single_model = {
            'only_model': ModelFit(
                model_name="only",
                aic=100.0,
                bic=105.0,
                log_likelihood=-48.0,
                fixed_effects={},
                converged=True,
                n_observations=50
            )
        }
        
        with pytest.raises(ValueError, match="Need at least 2 models"):
            comparison.compare_models(single_model)
            
    def test_compare_models_convergence_filtering(self, non_converged_fits):
        """Test handling of non-converged models"""
        comparison = ModelComparison()
        
        # Should warn but still proceed
        with patch.object(comparison.logger, 'warning') as mock_warn:
            result = comparison.compare_models(non_converged_fits)
            mock_warn.assert_called_once()
            
        # Should still return results
        assert result['n_models_compared'] == 2
        
    def test_calculate_information_criteria(self, sample_model_fits):
        """Test information criteria calculation"""
        comparison = ModelComparison()
        result = comparison._calculate_information_criteria(sample_model_fits)
        
        # Verify structure
        for model_name in sample_model_fits.keys():
            assert model_name in result
            
            model_ic = result[model_name]
            assert 'aic' in model_ic
            assert 'bic' in model_ic
            assert 'delta_aic' in model_ic
            assert 'delta_bic' in model_ic
            assert 'n_params' in model_ic
            
        # Verify delta calculations
        min_aic = min(fit.aic for fit in sample_model_fits.values())
        assert result['segmented']['delta_aic'] == 0.0  # segmented has min AIC (115)
        assert result['linear']['delta_aic'] == 120.0 - min_aic
        
    def test_calculate_aic_weights(self, sample_model_fits):
        """Test AIC weights calculation"""
        comparison = ModelComparison()
        weights = comparison._calculate_aic_weights(sample_model_fits)
        
        assert isinstance(weights, AICWeights)
        assert len(weights.models) == 3
        
        # First model should be best (lowest AIC)
        assert weights.models[0] == 'segmented'
        
        # Weights should sum to 1
        assert abs(sum(weights.weights) - 1.0) < 1e-6
        
        # Weights should be in descending order
        assert weights.weights[0] >= weights.weights[1] >= weights.weights[2]
        
        # Cumulative weights should be non-decreasing
        for i in range(1, len(weights.cumulative_weights)):
            assert weights.cumulative_weights[i] >= weights.cumulative_weights[i-1]
            
    def test_calculate_bic_weights(self, sample_model_fits):
        """Test BIC weights calculation (should use same logic as AIC)"""
        comparison = ModelComparison()
        bic_weights = comparison._calculate_bic_weights(sample_model_fits)
        
        assert isinstance(bic_weights, AICWeights)  # Reuses same structure
        assert len(bic_weights.models) == 3
        
        # Weights should sum to 1
        assert abs(sum(bic_weights.weights) - 1.0) < 1e-6
        
    def test_rank_models(self, sample_model_fits):
        """Test model ranking"""
        comparison = ModelComparison()
        
        ic_results = comparison._calculate_information_criteria(sample_model_fits)
        aic_weights = comparison._calculate_aic_weights(sample_model_fits)
        bic_weights = comparison._calculate_bic_weights(sample_model_fits)
        
        rankings = comparison._rank_models(sample_model_fits, ic_results, aic_weights, bic_weights)
        
        # Verify structure
        for model_name in sample_model_fits.keys():
            assert model_name in rankings
            criteria = rankings[model_name]
            assert isinstance(criteria, ModelSelectionCriteria)
            assert criteria.model_name == model_name
            assert 1 <= criteria.aic_rank <= 3
            assert 1 <= criteria.bic_rank <= 3
            
        # Best AIC model should have rank 1
        best_aic_model = min(sample_model_fits.keys(), key=lambda k: sample_model_fits[k].aic)
        assert rankings[best_aic_model].aic_rank == 1
        
    def test_likelihood_ratio_tests(self, sample_model_fits):
        """Test likelihood ratio tests"""
        comparison = ModelComparison(alpha=0.05)
        lr_tests = comparison._likelihood_ratio_tests(sample_model_fits)
        
        # Should find linear vs segmented comparison
        if 'linear_vs_segmented' in lr_tests:
            test = lr_tests['linear_vs_segmented']
            
            assert test['null_model'] == 'linear'
            assert test['alternative_model'] == 'segmented'
            assert 'lr_statistic' in test
            assert 'p_value' in test
            assert test['lr_statistic'] >= 0
            assert 0 <= test['p_value'] <= 1
            
    def test_select_best_model(self, sample_model_fits):
        """Test best model selection"""
        comparison = ModelComparison()
        
        ic_results = comparison._calculate_information_criteria(sample_model_fits)
        aic_weights = comparison._calculate_aic_weights(sample_model_fits)
        bic_weights = comparison._calculate_bic_weights(sample_model_fits)
        rankings = comparison._rank_models(sample_model_fits, ic_results, aic_weights, bic_weights)
        
        best_model = comparison._select_best_model(rankings)
        
        # Verify structure
        assert 'model_name' in best_model
        assert 'selection_criteria' in best_model
        assert 'is_clear_winner' in best_model
        assert 'confidence_level' in best_model
        
        # Should select model with best composite rank
        best_composite_rank = min(criteria.composite_rank for criteria in rankings.values())
        expected_best = [name for name, criteria in rankings.items() 
                        if criteria.composite_rank == best_composite_rank][0]
        assert best_model['model_name'] == expected_best
        
    def test_assess_evidence_strength(self, sample_model_fits):
        """Test evidence strength assessment"""
        comparison = ModelComparison()
        
        aic_weights = comparison._calculate_aic_weights(sample_model_fits)
        bic_weights = comparison._calculate_bic_weights(sample_model_fits)
        
        evidence = comparison._assess_evidence_strength(aic_weights, bic_weights)
        
        # Verify structure
        assert 'strength' in evidence
        assert 'recommendation' in evidence
        assert 'aic_evidence' in evidence
        assert 'bic_evidence' in evidence
        assert 'aic_bic_agreement' in evidence
        
        assert evidence['strength'] in ['weak', 'moderate', 'strong', 'very_strong']
        
    def test_create_summary_table(self, sample_model_fits):
        """Test summary table creation"""
        comparison = ModelComparison()
        
        ic_results = comparison._calculate_information_criteria(sample_model_fits)
        aic_weights = comparison._calculate_aic_weights(sample_model_fits)
        bic_weights = comparison._calculate_bic_weights(sample_model_fits)
        rankings = comparison._rank_models(sample_model_fits, ic_results, aic_weights, bic_weights)
        
        summary_table = comparison._create_summary_table(rankings)
        
        assert isinstance(summary_table, pd.DataFrame)
        assert len(summary_table) == 3
        
        # Check column names
        expected_cols = ['Model', 'AIC', 'BIC', 'AIC Weight', 'BIC Weight', 
                        'AIC Rank', 'BIC Rank', 'Composite Rank']
        for col in expected_cols:
            assert col in summary_table.columns
            
    def test_model_averaging_predictions(self, sample_model_fits):
        """Test model averaging predictions"""
        comparison = ModelComparison()
        
        # Mock predictions for each model 
        for model_name, fit in sample_model_fits.items():
            fit.predictions = np.array([0.5, 0.6, 0.7, 0.8])  # Mock predictions
            
        aic_weights = comparison._calculate_aic_weights(sample_model_fits)
        X_new = np.array([[1, 2, 3, 4]]).T  # Mock new data
        
        weighted_pred, pred_var = comparison.model_averaging_predictions(
            sample_model_fits, aic_weights, X_new, threshold=0.95
        )
        
        # Basic validation
        assert len(weighted_pred) == 4
        assert len(pred_var) == 4
        assert np.all(weighted_pred >= 0)
        assert np.all(pred_var >= 0)


# Integration tests
class TestModelComparisonIntegration:
    """Integration tests for complete model comparison workflow"""
    
    def test_full_comparison_workflow(self):
        """Test complete comparison workflow"""
        # Create realistic model fits with different performance levels
        model_fits = {
            'linear': ModelFit(
                model_name="linear", aic=150.0, bic=155.0, log_likelihood=-73.0,
                fixed_effects={"intercept": 1.0, "slope": 0.1}, converged=True, n_observations=100
            ),
            'segmented': ModelFit(
                model_name="segmented", aic=140.0, bic=148.0, log_likelihood=-66.0,
                fixed_effects={"intercept": 1.0, "slope1": 0.05, "slope2": 0.3, "breakpoint": 9.2}, 
                converged=True, n_observations=100
            ),
            'gam': ModelFit(
                model_name="gam", aic=145.0, bic=155.0, log_likelihood=-69.0,
                fixed_effects={"smooth_term": "nonparametric"}, converged=True, n_observations=100
            ),
            'failed_model': ModelFit(
                model_name="failed", aic=1000.0, bic=1010.0, log_likelihood=-498.0,
                fixed_effects={}, converged=False, n_observations=100
            )
        }
        
        comparison = ModelComparison(alpha=0.05)
        
        with patch.object(comparison.logger, 'warning'):  # Suppress convergence warnings
            results = comparison.compare_models(model_fits)
            
        # Verify complete workflow
        assert results['best_model']['model_name'] == 'segmented'  # Lowest AIC
        
        # Verify evidence assessment
        evidence = results['evidence_assessment']
        assert evidence['strength'] in ['weak', 'moderate', 'strong', 'very_strong']
        
        # Verify model rankings are complete
        rankings = results['model_rankings']
        for model_name in ['linear', 'segmented', 'gam', 'failed_model']:
            assert model_name in rankings
            
        # Verify likelihood ratio tests were attempted
        if results['likelihood_ratio_tests']:
            for test_name, test_result in results['likelihood_ratio_tests'].items():
                assert 'p_value' in test_result
                assert 'lr_statistic' in test_result
                
    def test_model_comparison_with_ties(self):
        """Test comparison when models have similar performance"""
        # Create models with very similar AIC values
        tied_fits = {
            'model_a': ModelFit(
                model_name="model_a", aic=100.0, bic=105.0, log_likelihood=-48.0,
                fixed_effects={"param1": 1.0}, converged=True, n_observations=100
            ),
            'model_b': ModelFit(
                model_name="model_b", aic=100.1, bic=105.2, log_likelihood=-48.05,
                fixed_effects={"param1": 0.95}, converged=True, n_observations=100
            ),
            'model_c': ModelFit(
                model_name="model_c", aic=100.2, bic=105.5, log_likelihood=-48.1,
                fixed_effects={"param1": 1.05}, converged=True, n_observations=100
            )
        }
        
        comparison = ModelComparison()
        results = comparison.compare_models(tied_fits)
        
        # With tied models, no clear winner should be detected
        assert results['evidence_assessment']['strength'] in ['weak', 'moderate']
        
        # AIC weights should be more evenly distributed
        aic_weights = results['aic_weights'].weights
        max_weight = max(aic_weights)
        assert max_weight < 0.7  # No single model dominates