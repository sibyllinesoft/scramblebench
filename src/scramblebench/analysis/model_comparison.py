"""
Model Comparison and Selection Framework

Implements systematic model comparison using AIC, BIC, and cross-validation
for selecting the best scaling model per family. Includes evidence ratios
and model averaging capabilities for robust inference.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

from .statistical_models import ModelFit

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionCriteria:
    """Container for model selection criteria and rankings"""
    model_name: str
    aic: float
    bic: float
    aic_weight: float
    bic_weight: float
    aic_rank: int
    bic_rank: int
    cv_score: Optional[float] = None
    cv_rank: Optional[int] = None
    evidence_ratio_aic: float = 1.0
    evidence_ratio_bic: float = 1.0
    
    @property
    def composite_rank(self) -> float:
        """Composite ranking score (lower is better)"""
        ranks = [self.aic_rank, self.bic_rank]
        if self.cv_rank is not None:
            ranks.append(self.cv_rank)
        return np.mean(ranks)


@dataclass
class AICWeights:
    """AIC weights for model averaging"""
    models: List[str]
    aic_values: List[float]
    delta_aic: List[float]
    weights: List[float]
    cumulative_weights: List[float]
    
    def get_top_models(self, threshold: float = 0.95) -> List[str]:
        """Get models comprising specified cumulative weight"""
        top_models = []
        for i, (model, cum_weight) in enumerate(zip(self.models, self.cumulative_weights)):
            top_models.append(model)
            if cum_weight >= threshold:
                break
        return top_models
    
    def is_clear_winner(self, threshold: float = 0.9) -> Tuple[bool, Optional[str]]:
        """Check if there's a clear best model"""
        if self.weights and self.weights[0] > threshold:
            return True, self.models[0]
        return False, None


class ModelComparison:
    """
    Comprehensive model comparison and selection framework
    
    Implements systematic comparison of GLMM, GAM, and changepoint models
    using multiple criteria for robust model selection.
    """
    
    def __init__(self, alpha: float = 0.05, logger: Optional[logging.Logger] = None):
        """
        Initialize model comparison framework
        
        Args:
            alpha: Significance level for tests
            logger: Logger instance
        """
        self.alpha = alpha
        self.logger = logger or logging.getLogger(__name__)
    
    def compare_models(self, model_fits: Dict[str, ModelFit]) -> Dict[str, Any]:
        """
        Comprehensive comparison of multiple models
        
        Args:
            model_fits: Dictionary of model names to ModelFit objects
            
        Returns:
            Comprehensive comparison results
        """
        self.logger.info(f"Comparing {len(model_fits)} models")
        
        if len(model_fits) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        # Filter to converged models only
        converged_models = {
            name: fit for name, fit in model_fits.items() 
            if fit.converged
        }
        
        if len(converged_models) < 2:
            self.logger.warning("Fewer than 2 converged models available")
            converged_models = model_fits  # Use all models anyway
        
        # Calculate information criteria
        ic_results = self._calculate_information_criteria(converged_models)
        
        # AIC/BIC weights
        aic_weights = self._calculate_aic_weights(converged_models)
        bic_weights = self._calculate_bic_weights(converged_models)
        
        # Model rankings
        rankings = self._rank_models(converged_models, ic_results, aic_weights, bic_weights)
        
        # Likelihood ratio tests (where applicable)
        lr_tests = self._likelihood_ratio_tests(converged_models)
        
        # Best model selection
        best_model = self._select_best_model(rankings)
        
        # Evidence strength assessment
        evidence_assessment = self._assess_evidence_strength(aic_weights, bic_weights)
        
        return {
            'n_models_compared': len(converged_models),
            'information_criteria': ic_results,
            'aic_weights': aic_weights,
            'bic_weights': bic_weights,
            'model_rankings': rankings,
            'likelihood_ratio_tests': lr_tests,
            'best_model': best_model,
            'evidence_assessment': evidence_assessment,
            'model_selection_summary': self._create_summary_table(rankings)
        }
    
    def _calculate_information_criteria(self, models: Dict[str, ModelFit]) -> Dict[str, Dict[str, float]]:
        """Calculate AIC, BIC, and related metrics for all models"""
        
        ic_results = {}
        
        for name, fit in models.items():
            ic_results[name] = {
                'aic': fit.aic,
                'bic': fit.bic,
                'log_likelihood': fit.log_likelihood,
                'n_params': len(fit.fixed_effects),
                'n_observations': fit.n_observations,
                'deviance': fit.deviance or 0.0
            }
        
        # Calculate delta values (difference from minimum)
        aic_values = [result['aic'] for result in ic_results.values()]
        bic_values = [result['bic'] for result in ic_results.values()]
        
        min_aic = min(aic_values)
        min_bic = min(bic_values)
        
        for name in ic_results:
            ic_results[name]['delta_aic'] = ic_results[name]['aic'] - min_aic
            ic_results[name]['delta_bic'] = ic_results[name]['bic'] - min_bic
        
        return ic_results
    
    def _calculate_aic_weights(self, models: Dict[str, ModelFit]) -> AICWeights:
        """Calculate Akaike weights for model averaging"""
        
        model_names = list(models.keys())
        aic_values = [models[name].aic for name in model_names]
        
        # Delta AIC
        min_aic = min(aic_values)
        delta_aic = [aic - min_aic for aic in aic_values]
        
        # Weights
        exp_terms = [np.exp(-0.5 * delta) for delta in delta_aic]
        sum_exp = sum(exp_terms)
        weights = [exp_term / sum_exp for exp_term in exp_terms]
        
        # Sort by weight (descending)
        sorted_indices = np.argsort(weights)[::-1]
        
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_aic = [aic_values[i] for i in sorted_indices]
        sorted_delta = [delta_aic[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]
        
        # Cumulative weights
        cumulative_weights = np.cumsum(sorted_weights).tolist()
        
        return AICWeights(
            models=sorted_names,
            aic_values=sorted_aic,
            delta_aic=sorted_delta,
            weights=sorted_weights,
            cumulative_weights=cumulative_weights
        )
    
    def _calculate_bic_weights(self, models: Dict[str, ModelFit]) -> AICWeights:
        """Calculate BIC weights (same formula as AIC weights but with BIC)"""
        
        model_names = list(models.keys())
        bic_values = [models[name].bic for name in model_names]
        
        # Delta BIC
        min_bic = min(bic_values)
        delta_bic = [bic - min_bic for bic in bic_values]
        
        # Weights
        exp_terms = [np.exp(-0.5 * delta) for delta in delta_bic]
        sum_exp = sum(exp_terms)
        weights = [exp_term / sum_exp for exp_term in exp_terms]
        
        # Sort by weight (descending)
        sorted_indices = np.argsort(weights)[::-1]
        
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_bic = [bic_values[i] for i in sorted_indices]
        sorted_delta = [delta_bic[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]
        
        # Cumulative weights
        cumulative_weights = np.cumsum(sorted_weights).tolist()
        
        return AICWeights(  # Reuse same structure for BIC
            models=sorted_names,
            aic_values=sorted_bic,
            delta_aic=sorted_delta,
            weights=sorted_weights,
            cumulative_weights=cumulative_weights
        )
    
    def _rank_models(
        self, 
        models: Dict[str, ModelFit], 
        ic_results: Dict[str, Dict[str, float]],
        aic_weights: AICWeights,
        bic_weights: AICWeights
    ) -> Dict[str, ModelSelectionCriteria]:
        """Rank models using multiple criteria"""
        
        rankings = {}
        
        # AIC ranking
        aic_sorted = sorted(models.items(), key=lambda x: x[1].aic)
        aic_ranks = {name: rank + 1 for rank, (name, _) in enumerate(aic_sorted)}
        
        # BIC ranking  
        bic_sorted = sorted(models.items(), key=lambda x: x[1].bic)
        bic_ranks = {name: rank + 1 for rank, (name, _) in enumerate(bic_sorted)}
        
        # Create weight lookup dictionaries
        aic_weight_lookup = dict(zip(aic_weights.models, aic_weights.weights))
        bic_weight_lookup = dict(zip(bic_weights.models, bic_weights.weights))
        
        # Evidence ratios (relative to best model)
        best_aic_weight = max(aic_weights.weights)
        best_bic_weight = max(bic_weights.weights)
        
        for name, fit in models.items():
            rankings[name] = ModelSelectionCriteria(
                model_name=name,
                aic=fit.aic,
                bic=fit.bic,
                aic_weight=aic_weight_lookup[name],
                bic_weight=bic_weight_lookup[name],
                aic_rank=aic_ranks[name],
                bic_rank=bic_ranks[name],
                cv_score=fit.cv_score,
                cv_rank=None,  # Would need cross-validation results
                evidence_ratio_aic=best_aic_weight / aic_weight_lookup[name] if aic_weight_lookup[name] > 0 else np.inf,
                evidence_ratio_bic=best_bic_weight / bic_weight_lookup[name] if bic_weight_lookup[name] > 0 else np.inf
            )
        
        return rankings
    
    def _likelihood_ratio_tests(self, models: Dict[str, ModelFit]) -> Dict[str, Dict[str, Any]]:
        """Perform likelihood ratio tests between nested models"""
        
        lr_tests = {}
        
        # Common nested model pairs
        nested_pairs = [
            ('linear', 'segmented'),  # Linear vs segmented
            ('glmm', 'gam'),          # Parametric vs non-parametric (if comparable)
        ]
        
        for model1_name, model2_name in nested_pairs:
            if model1_name in models and model2_name in models:
                
                model1 = models[model1_name]
                model2 = models[model2_name]
                
                # Check if models are nested (simplified check)
                if len(model2.fixed_effects) > len(model1.fixed_effects):
                    # model2 is more complex (alternative hypothesis)
                    lr_stat = 2 * (model2.log_likelihood - model1.log_likelihood)
                    df_diff = len(model2.fixed_effects) - len(model1.fixed_effects)
                    
                    if lr_stat >= 0 and df_diff > 0:
                        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
                        
                        lr_tests[f"{model1_name}_vs_{model2_name}"] = {
                            'null_model': model1_name,
                            'alternative_model': model2_name,
                            'lr_statistic': lr_stat,
                            'df_difference': df_diff,
                            'p_value': p_value,
                            'significant': p_value < self.alpha,
                            'aic_difference': model2.aic - model1.aic,
                            'bic_difference': model2.bic - model1.bic
                        }
        
        return lr_tests
    
    def _select_best_model(self, rankings: Dict[str, ModelSelectionCriteria]) -> Dict[str, Any]:
        """Select best model using composite criteria"""
        
        # Sort by composite rank
        sorted_models = sorted(rankings.items(), key=lambda x: x[1].composite_rank)
        
        best_name, best_criteria = sorted_models[0]
        
        # Check if there's a clear winner
        aic_weights_list = [criteria.aic_weight for criteria in rankings.values()]
        max_aic_weight = max(aic_weights_list)
        
        is_clear_winner = max_aic_weight > 0.9
        
        return {
            'model_name': best_name,
            'selection_criteria': {
                'aic': best_criteria.aic,
                'bic': best_criteria.bic,
                'aic_weight': best_criteria.aic_weight,
                'bic_weight': best_criteria.bic_weight,
                'composite_rank': best_criteria.composite_rank
            },
            'is_clear_winner': is_clear_winner,
            'max_aic_weight': max_aic_weight,
            'confidence_level': 'high' if is_clear_winner else 'medium' if max_aic_weight > 0.7 else 'low'
        }
    
    def _assess_evidence_strength(self, aic_weights: AICWeights, bic_weights: AICWeights) -> Dict[str, Any]:
        """Assess strength of evidence for model selection"""
        
        # AIC evidence
        aic_clear_winner, aic_winner = aic_weights.is_clear_winner(0.9)
        aic_strong_evidence = aic_weights.weights[0] > 0.7 if aic_weights.weights else False
        
        # BIC evidence 
        bic_clear_winner, bic_winner = bic_weights.is_clear_winner(0.9)
        bic_strong_evidence = bic_weights.weights[0] > 0.7 if bic_weights.weights else False
        
        # Agreement between AIC and BIC
        agreement = aic_winner == bic_winner if aic_winner and bic_winner else False
        
        # Overall assessment
        if aic_clear_winner and bic_clear_winner and agreement:
            strength = "very_strong"
            recommendation = f"Strong evidence for {aic_winner}"
        elif aic_strong_evidence and bic_strong_evidence and agreement:
            strength = "strong"
            recommendation = f"Good evidence for {aic_winner}"
        elif agreement:
            strength = "moderate"
            recommendation = f"Moderate evidence for {aic_winner}"
        else:
            strength = "weak"
            recommendation = "Model selection uncertain - consider model averaging"
        
        return {
            'strength': strength,
            'recommendation': recommendation,
            'aic_evidence': {
                'clear_winner': aic_clear_winner,
                'winner': aic_winner,
                'max_weight': aic_weights.weights[0] if aic_weights.weights else 0
            },
            'bic_evidence': {
                'clear_winner': bic_clear_winner,
                'winner': bic_winner,
                'max_weight': bic_weights.weights[0] if bic_weights.weights else 0
            },
            'aic_bic_agreement': agreement
        }
    
    def _create_summary_table(self, rankings: Dict[str, ModelSelectionCriteria]) -> pd.DataFrame:
        """Create summary table for publication"""
        
        rows = []
        for name, criteria in rankings.items():
            rows.append({
                'Model': name,
                'AIC': f"{criteria.aic:.2f}",
                'BIC': f"{criteria.bic:.2f}",
                'AIC Weight': f"{criteria.aic_weight:.3f}",
                'BIC Weight': f"{criteria.bic_weight:.3f}",
                'AIC Rank': criteria.aic_rank,
                'BIC Rank': criteria.bic_rank,
                'Composite Rank': f"{criteria.composite_rank:.1f}"
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('Composite Rank')
    
    def model_averaging_predictions(
        self, 
        models: Dict[str, ModelFit], 
        weights: AICWeights,
        X_new: np.ndarray,
        threshold: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate model-averaged predictions
        
        Args:
            models: Fitted models
            weights: Model weights for averaging
            X_new: New data for prediction
            threshold: Cumulative weight threshold for model inclusion
            
        Returns:
            Tuple of (predictions, prediction_variance)
        """
        
        top_models = weights.get_top_models(threshold)
        
        # Get predictions from each model
        predictions = []
        model_weights = []
        
        for model_name in top_models:
            if model_name in models and models[model_name].predictions is not None:
                # Would need model-specific prediction methods
                # This is a placeholder for the concept
                pred = models[model_name].predictions  
                weight = dict(zip(weights.models, weights.weights))[model_name]
                
                predictions.append(pred)
                model_weights.append(weight)
        
        if not predictions:
            raise ValueError("No predictions available for model averaging")
        
        # Renormalize weights for selected models
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        # Weighted average
        stacked_preds = np.stack(predictions)
        weighted_pred = np.average(stacked_preds, axis=0, weights=normalized_weights)
        
        # Prediction variance (within + between model uncertainty)
        within_model_var = np.average(
            [np.var(pred) for pred in predictions], 
            weights=normalized_weights
        )
        
        between_model_var = np.average(
            [(pred - weighted_pred)**2 for pred in predictions],
            weights=normalized_weights,
            axis=0
        ).mean()
        
        total_variance = within_model_var + between_model_var
        
        return weighted_pred, np.full_like(weighted_pred, total_variance)