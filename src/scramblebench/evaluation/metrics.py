"""
Metrics calculation for ScrambleBench evaluations.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score
import json

from .results import EvaluationResults
from .config import MetricsConfig


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics."""
    exact_match: float
    f1_score: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None


@dataclass
class RobustnessMetrics:
    """Container for robustness metrics."""
    degradation_scores: Dict[str, float]  # transformation_type -> degradation
    significant_degradations: List[str]
    avg_degradation: float
    max_degradation: float
    min_degradation: float


@dataclass
class StatisticalTests:
    """Container for statistical test results."""
    pairwise_comparisons: Dict[str, Dict[str, float]]  # model1 -> model2 -> p_value
    effect_sizes: Dict[str, Dict[str, float]]  # model1 -> model2 -> effect_size
    confidence_intervals: Dict[str, Tuple[float, float]]  # model -> (lower, upper)


class MetricsCalculator:
    """
    Calculator for various evaluation metrics including accuracy, robustness,
    and statistical significance tests.
    """
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            config: Metrics configuration
            logger: Logger instance
        """
        self.config = config or MetricsConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_accuracy_metrics(
        self,
        results: EvaluationResults,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, AccuracyMetrics]:
        """
        Calculate accuracy metrics for all models.
        
        Args:
            results: Evaluation results
            ground_truth: Dictionary mapping problem_id to correct answer
            
        Returns:
            Dictionary mapping model names to accuracy metrics
        """
        df = results.to_dataframe()
        model_metrics = {}
        
        for model_name in results.get_model_names():
            model_df = df[df['model_name'] == model_name]
            
            if ground_truth:
                metrics = self._calculate_accuracy_with_ground_truth(model_df, ground_truth)
            else:
                # Calculate accuracy based on success rate (API success)
                metrics = AccuracyMetrics(
                    exact_match=model_df['success'].mean()
                )
            
            model_metrics[model_name] = metrics
        
        return model_metrics
    
    def calculate_robustness_metrics(
        self,
        results: EvaluationResults,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, RobustnessMetrics]:
        """
        Calculate robustness metrics by comparing performance on original vs transformed problems.
        
        Args:
            results: Evaluation results
            ground_truth: Dictionary mapping problem_id to correct answer
            
        Returns:
            Dictionary mapping model names to robustness metrics
        """
        df = results.to_dataframe()
        model_metrics = {}
        
        for model_name in results.get_model_names():
            model_df = df[df['model_name'] == model_name]
            
            # Get original performance
            original_df = model_df[model_df['transformation_type'] == 'original']
            if original_df.empty:
                self.logger.warning(f"No original results found for model {model_name}")
                continue
            
            original_accuracy = self._get_accuracy_score(original_df, ground_truth)
            
            # Calculate degradation for each transformation type
            degradation_scores = {}
            transformation_types = [t for t in model_df['transformation_type'].unique() if t != 'original']
            
            for transform_type in transformation_types:
                transform_df = model_df[model_df['transformation_type'] == transform_type]
                if transform_df.empty:
                    continue
                
                transform_accuracy = self._get_accuracy_score(transform_df, ground_truth)
                degradation = original_accuracy - transform_accuracy
                degradation_scores[transform_type] = degradation
            
            # Identify significant degradations
            significant_degradations = [
                t for t, d in degradation_scores.items()
                if d > self.config.degradation_threshold
            ]
            
            if degradation_scores:
                metrics = RobustnessMetrics(
                    degradation_scores=degradation_scores,
                    significant_degradations=significant_degradations,
                    avg_degradation=np.mean(list(degradation_scores.values())),
                    max_degradation=max(degradation_scores.values()),
                    min_degradation=min(degradation_scores.values())
                )
            else:
                metrics = RobustnessMetrics(
                    degradation_scores={},
                    significant_degradations=[],
                    avg_degradation=0.0,
                    max_degradation=0.0,
                    min_degradation=0.0
                )
            
            model_metrics[model_name] = metrics
        
        return model_metrics
    
    def calculate_statistical_tests(
        self,
        results: EvaluationResults,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> StatisticalTests:
        """
        Calculate statistical significance tests between models.
        
        Args:
            results: Evaluation results
            ground_truth: Dictionary mapping problem_id to correct answer
            
        Returns:
            Statistical test results
        """
        df = results.to_dataframe()
        models = results.get_model_names()
        
        # Prepare data for comparisons
        model_scores = {}
        for model_name in models:
            model_df = df[df['model_name'] == model_name]
            scores = self._get_problem_level_scores(model_df, ground_truth)
            model_scores[model_name] = scores
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        effect_sizes = {}
        
        for i, model1 in enumerate(models):
            pairwise_comparisons[model1] = {}
            effect_sizes[model1] = {}
            
            for j, model2 in enumerate(models):
                if i != j:
                    scores1 = model_scores[model1]
                    scores2 = model_scores[model2]
                    
                    # Align scores by problem_id
                    aligned_scores = self._align_scores(scores1, scores2)
                    
                    if len(aligned_scores) > 0:
                        s1_values = [s[0] for s in aligned_scores]
                        s2_values = [s[1] for s in aligned_scores]
                        
                        # Statistical test (paired t-test)
                        if len(s1_values) > 1:
                            t_stat, p_value = stats.ttest_rel(s1_values, s2_values)
                            
                            # Effect size (Cohen's d for paired samples)
                            diff = np.array(s1_values) - np.array(s2_values)
                            effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                        else:
                            p_value = 1.0
                            effect_size = 0.0
                        
                        pairwise_comparisons[model1][model2] = p_value
                        effect_sizes[model1][model2] = effect_size
        
        # Apply multiple comparisons correction
        if self.config.multiple_comparisons_correction == "bonferroni":
            pairwise_comparisons = self._apply_bonferroni_correction(pairwise_comparisons)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for model_name in models:
            scores = list(model_scores[model_name].values())
            if len(scores) > 1:
                mean_score = np.mean(scores)
                sem = stats.sem(scores)
                ci = stats.t.interval(
                    1 - self.config.significance_level,
                    len(scores) - 1,
                    loc=mean_score,
                    scale=sem
                )
                confidence_intervals[model_name] = ci
            else:
                confidence_intervals[model_name] = (0.0, 1.0)
        
        return StatisticalTests(
            pairwise_comparisons=pairwise_comparisons,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_accuracy_with_ground_truth(
        self,
        model_df: pd.DataFrame,
        ground_truth: Dict[str, str]
    ) -> AccuracyMetrics:
        """Calculate accuracy metrics using ground truth."""
        exact_matches = 0
        total_predictions = 0
        
        # For F1 calculation
        true_labels = []
        pred_labels = []
        
        for _, row in model_df.iterrows():
            problem_id = row['problem_id']
            if problem_id not in ground_truth:
                continue
            
            true_answer = ground_truth[problem_id]
            pred_answer = row['model_response']
            
            # Exact match
            if self._normalize_answer(pred_answer) == self._normalize_answer(true_answer):
                exact_matches += 1
            
            total_predictions += 1
            
            # For F1 (treating as binary classification)
            true_labels.append(true_answer)
            pred_labels.append(pred_answer)
        
        exact_match = exact_matches / total_predictions if total_predictions > 0 else 0.0
        
        # Calculate F1 if requested and possible
        f1 = None
        if self.config.calculate_f1 and len(set(true_labels)) <= 10:  # Only for reasonable number of classes
            try:
                # Convert to binary if too many classes
                if len(set(true_labels)) > 2:
                    true_binary = [1 if t == p else 0 for t, p in zip(true_labels, pred_labels)]
                    pred_binary = [1] * len(pred_labels)  # All predictions are "positive"
                    f1 = f1_score(true_binary, pred_binary, average='binary')
                else:
                    f1 = f1_score(true_labels, pred_labels, average='weighted')
            except Exception as e:
                self.logger.warning(f"Could not calculate F1 score: {e}")
        
        return AccuracyMetrics(
            exact_match=exact_match,
            f1_score=f1
        )
    
    def _get_accuracy_score(
        self,
        df: pd.DataFrame,
        ground_truth: Optional[Dict[str, str]]
    ) -> float:
        """Get accuracy score for a dataframe."""
        if ground_truth:
            correct = 0
            total = 0
            for _, row in df.iterrows():
                problem_id = row['problem_id']
                if problem_id in ground_truth:
                    if self._normalize_answer(row['model_response']) == self._normalize_answer(ground_truth[problem_id]):
                        correct += 1
                    total += 1
            return correct / total if total > 0 else 0.0
        else:
            # Use success rate as proxy
            return df['success'].mean()
    
    def _get_problem_level_scores(
        self,
        df: pd.DataFrame,
        ground_truth: Optional[Dict[str, str]]
    ) -> Dict[str, float]:
        """Get accuracy score for each problem."""
        scores = {}
        
        for _, row in df.iterrows():
            problem_id = row['problem_id']
            
            if ground_truth and problem_id in ground_truth:
                score = 1.0 if self._normalize_answer(row['model_response']) == self._normalize_answer(ground_truth[problem_id]) else 0.0
            else:
                score = 1.0 if row['success'] else 0.0
            
            scores[problem_id] = score
        
        return scores
    
    def _align_scores(
        self,
        scores1: Dict[str, float],
        scores2: Dict[str, float]
    ) -> List[Tuple[float, float]]:
        """Align scores by problem ID."""
        aligned = []
        
        for problem_id in scores1:
            if problem_id in scores2:
                aligned.append((scores1[problem_id], scores2[problem_id]))
        
        return aligned
    
    def _apply_bonferroni_correction(
        self,
        pairwise_comparisons: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Apply Bonferroni correction to p-values."""
        # Count total comparisons
        total_comparisons = sum(len(comparisons) for comparisons in pairwise_comparisons.values())
        
        # Apply correction
        corrected = {}
        for model1, comparisons in pairwise_comparisons.items():
            corrected[model1] = {}
            for model2, p_value in comparisons.items():
                corrected_p = min(p_value * total_comparisons, 1.0)
                corrected[model1][model2] = corrected_p
        
        return corrected
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not isinstance(answer, str):
            return str(answer).lower().strip()
        
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', answer.lower().strip())
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def generate_metrics_report(
        self,
        results: EvaluationResults,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        
        Args:
            results: Evaluation results
            ground_truth: Optional ground truth answers
            
        Returns:
            Comprehensive metrics report
        """
        report = {
            "experiment_info": {
                "total_results": len(results.results),
                "models": results.get_model_names(),
                "transformation_types": results.get_transformation_types(),
                "has_ground_truth": ground_truth is not None
            }
        }
        
        # Accuracy metrics
        if self.config.calculate_exact_match:
            accuracy_metrics = self.calculate_accuracy_metrics(results, ground_truth)
            report["accuracy_metrics"] = {
                model: {
                    "exact_match": metrics.exact_match,
                    "f1_score": metrics.f1_score
                }
                for model, metrics in accuracy_metrics.items()
            }
        
        # Robustness metrics
        if self.config.calculate_degradation:
            robustness_metrics = self.calculate_robustness_metrics(results, ground_truth)
            report["robustness_metrics"] = {
                model: {
                    "avg_degradation": metrics.avg_degradation,
                    "max_degradation": metrics.max_degradation,
                    "significant_degradations": metrics.significant_degradations,
                    "degradation_by_transformation": metrics.degradation_scores
                }
                for model, metrics in robustness_metrics.items()
            }
        
        # Statistical tests
        if len(results.get_model_names()) > 1:
            statistical_tests = self.calculate_statistical_tests(results, ground_truth)
            report["statistical_tests"] = {
                "pairwise_comparisons": statistical_tests.pairwise_comparisons,
                "effect_sizes": statistical_tests.effect_sizes,
                "confidence_intervals": {
                    model: {"lower": ci[0], "upper": ci[1]}
                    for model, ci in statistical_tests.confidence_intervals.items()
                }
            }
        
        return report
    
    def save_metrics_report(
        self,
        report: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """Save metrics report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Saved metrics report to {output_path}")