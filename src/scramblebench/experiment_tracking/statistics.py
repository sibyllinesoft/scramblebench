"""
Statistical Analysis Framework for Experiment Tracking

Advanced statistical analysis system for academic research including 
A/B testing, significance testing, effect size calculations, and 
language dependency coefficient analysis.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

from ..evaluation.results import EvaluationResults
from .database import DatabaseManager


class TestType(Enum):
    """Types of statistical tests"""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    CORRELATION = "correlation"
    REGRESSION = "regression"


class EffectSize(Enum):
    """Effect size measures"""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    ETA_SQUARED = "eta_squared"
    CRAMERS_V = "cramers_v"


@dataclass
class SignificanceTest:
    """Results from a statistical significance test"""
    test_type: TestType
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: EffectSize
    confidence_interval: Tuple[float, float]
    
    # Test details
    degrees_of_freedom: Optional[int] = None
    sample_size_1: int = 0
    sample_size_2: int = 0
    
    # Interpretation
    is_significant: bool = False
    significance_level: float = 0.05
    power: Optional[float] = None
    interpretation: str = ""
    
    # Effect size interpretation
    effect_magnitude: str = ""  # "small", "medium", "large"
    
    # Raw data summary
    group1_mean: Optional[float] = None
    group1_std: Optional[float] = None 
    group2_mean: Optional[float] = None
    group2_std: Optional[float] = None


@dataclass
class ABTestResult:
    """A/B test comparison results"""
    experiment_id: str
    comparison_name: str
    control_group: str
    treatment_group: str
    
    # Primary metric
    metric_name: str
    control_value: float
    treatment_value: float
    lift: float  # (treatment - control) / control
    
    # Statistical test
    significance_test: SignificanceTest
    
    # Sample sizes
    control_samples: int
    treatment_samples: int
    
    # Confidence and power
    confidence_level: float = 0.95
    statistical_power: Optional[float] = None
    minimum_detectable_effect: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_significant_improvement(self) -> bool:
        """Check if treatment shows significant improvement"""
        return (self.significance_test.is_significant and 
                self.lift > 0)
    
    @property
    def recommendation(self) -> str:
        """Get recommendation based on results"""
        if self.is_significant_improvement:
            return f"Deploy treatment: {self.lift*100:.2f}% improvement (p={self.significance_test.p_value:.4f})"
        elif self.significance_test.is_significant and self.lift < 0:
            return f"Do not deploy: {abs(self.lift)*100:.2f}% degradation (p={self.significance_test.p_value:.4f})"
        else:
            return f"Inconclusive: {self.lift*100:.2f}% change (p={self.significance_test.p_value:.4f}, not significant)"


@dataclass
class LanguageDependencyAnalysis:
    """Analysis of language dependency for a model"""
    experiment_id: str
    model_id: str
    model_name: str
    
    # Baseline performance (0% scrambling)
    baseline_accuracy: float
    baseline_samples: int
    
    # Performance across scrambling levels
    scrambling_performance: Dict[float, float]  # level -> accuracy
    
    # Dependency metrics  
    dependency_coefficient: float  # 0 = no dependency, 1 = complete dependency
    threshold_value: Optional[float]  # Scrambling level where performance drops significantly
    
    # Statistical measures
    correlation_coefficient: float  # Correlation between scrambling and performance
    correlation_p_value: float
    
    # Robustness metrics
    robustness_score: float  # Overall resistance to scrambling
    degradation_rate: float  # Rate of performance loss per scrambling unit
    
    # Confidence intervals
    dependency_ci_lower: float
    dependency_ci_upper: float
    
    # Analysis metadata
    computed_at: datetime = field(default_factory=datetime.now)
    statistical_method: str = "spearman_correlation"


class StatisticalAnalyzer:
    """
    Advanced statistical analysis system for experiment results
    
    Provides comprehensive statistical analysis capabilities including:
    - Significance testing (t-tests, Mann-Whitney U, ANOVA, etc.)
    - Effect size calculations
    - A/B testing framework
    - Language dependency analysis
    - Multiple comparison corrections
    - Statistical power analysis
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        default_alpha: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize statistical analyzer
        
        Args:
            db_manager: Database manager for data access
            default_alpha: Default significance level
            logger: Logger instance
        """
        self.db_manager = db_manager
        self.default_alpha = default_alpha
        self.logger = logger or logging.getLogger(__name__)
        
        # Suppress some statistical warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats')
    
    async def compute_performance_metrics(
        self,
        experiment_id: str,
        results: EvaluationResults
    ) -> None:
        """
        Compute and store comprehensive performance metrics
        
        Args:
            experiment_id: Experiment ID
            results: Evaluation results to analyze
        """
        self.logger.info(f"Computing performance metrics for experiment {experiment_id}")
        
        try:
            # Convert results to DataFrame for analysis
            df = self._results_to_dataframe(results)
            
            if df.empty:
                self.logger.warning(f"No data to analyze for experiment {experiment_id}")
                return
            
            # Group by key dimensions
            groups = ['model_id', 'benchmark_id', 'scrambling_method', 'scrambling_intensity']
            available_groups = [g for g in groups if g in df.columns]
            
            if not available_groups:
                self.logger.warning("No grouping columns found in results")
                return
            
            # Calculate metrics for each group
            metrics = df.groupby(available_groups).agg({
                'is_correct': ['count', 'sum', 'mean', 'std'],
                'response_time_ms': ['mean', 'median', 'std'],
                'partial_credit': ['mean', 'std'] if 'partial_credit' in df.columns else None,
                'cost': ['sum', 'mean'] if 'cost' in df.columns else None
            }).round(6)
            
            # Flatten column names
            metrics.columns = ['_'.join(col).strip('_') for col in metrics.columns]
            
            # Calculate confidence intervals
            for idx in metrics.index:
                group_data = df[df.set_index(available_groups).index == idx]
                accuracy_data = group_data['is_correct']
                
                if len(accuracy_data) > 1:
                    ci_lower, ci_upper = self._calculate_binomial_ci(
                        accuracy_data.sum(), len(accuracy_data)
                    )
                    metrics.loc[idx, 'accuracy_ci_lower'] = ci_lower
                    metrics.loc[idx, 'accuracy_ci_upper'] = ci_upper
            
            # Save to database
            await self._save_performance_metrics(experiment_id, metrics)
            
            self.logger.info(f"Computed metrics for {len(metrics)} groups")
            
        except Exception as e:
            self.logger.error(f"Error computing performance metrics: {e}")
            raise
    
    async def analyze_thresholds(
        self,
        experiment_id: str,
        results: EvaluationResults,
        threshold_definition: str = "50% of baseline performance"
    ) -> Dict[str, Any]:
        """
        Analyze language dependency thresholds
        
        Args:
            experiment_id: Experiment ID
            results: Evaluation results
            threshold_definition: Definition of performance threshold
            
        Returns:
            Dictionary of threshold analysis results
        """
        self.logger.info(f"Analyzing thresholds for experiment {experiment_id}")
        
        try:
            df = self._results_to_dataframe(results)
            threshold_results = {}
            
            # Group by model and benchmark
            for (model_id, benchmark_id), model_data in df.groupby(['model_id', 'benchmark_id']):
                
                # Calculate baseline performance (0% scrambling or original questions)
                baseline_data = model_data[
                    (model_data.get('scrambling_intensity', 0) == 0) |
                    model_data.get('is_original', False)
                ]
                
                if baseline_data.empty:
                    continue
                
                baseline_accuracy = baseline_data['is_correct'].mean()
                
                # Get performance across scrambling levels
                scrambling_data = model_data[model_data.get('scrambling_intensity', 0) > 0]
                
                if scrambling_data.empty:
                    continue
                
                # Calculate language dependency
                dependency_analysis = await self._calculate_language_dependency(
                    model_data, baseline_accuracy
                )
                
                # Find threshold point
                threshold_value = self._find_threshold_point(
                    scrambling_data, baseline_accuracy, threshold_ratio=0.5
                )
                
                # Store results
                key = f"{model_id}_{benchmark_id}"
                threshold_results[key] = {
                    'model_id': model_id,
                    'benchmark_id': benchmark_id,
                    'baseline_accuracy': baseline_accuracy,
                    'threshold_value': threshold_value,
                    'dependency_coefficient': dependency_analysis['dependency_coefficient'],
                    'correlation_coefficient': dependency_analysis['correlation_coefficient'],
                    'correlation_p_value': dependency_analysis['correlation_p_value'],
                    'is_significant': dependency_analysis['correlation_p_value'] < self.default_alpha
                }
                
                # Save to database
                await self._save_threshold_analysis(experiment_id, threshold_results[key])
            
            self.logger.info(f"Analyzed thresholds for {len(threshold_results)} model-benchmark pairs")
            return threshold_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing thresholds: {e}")
            raise
    
    async def run_significance_tests(
        self,
        experiment_id: str,
        results: EvaluationResults,
        comparisons: Optional[List[Tuple[str, str]]] = None
    ) -> List[SignificanceTest]:
        """
        Run statistical significance tests
        
        Args:
            experiment_id: Experiment ID
            results: Evaluation results
            comparisons: Optional list of (group1, group2) comparisons
            
        Returns:
            List of significance test results
        """
        self.logger.info(f"Running significance tests for experiment {experiment_id}")
        
        try:
            df = self._results_to_dataframe(results)
            test_results = []
            
            # If no specific comparisons, test all model pairs
            if not comparisons:
                models = df['model_id'].unique()
                comparisons = [(m1, m2) for i, m1 in enumerate(models) 
                              for m2 in models[i+1:]]
            
            for group1, group2 in comparisons:
                # Get data for each group
                data1 = df[df['model_id'] == group1]['is_correct']
                data2 = df[df['model_id'] == group2]['is_correct']
                
                if len(data1) < 10 or len(data2) < 10:
                    continue  # Need minimum sample size
                
                # Run t-test
                test_result = self._run_t_test(
                    data1, data2, f"{group1} vs {group2}"
                )
                test_results.append(test_result)
                
                # Save to database
                await self._save_significance_test(experiment_id, test_result)
            
            # Apply multiple comparison correction
            if len(test_results) > 1:
                corrected_results = self._apply_multiple_comparison_correction(test_results)
                test_results = corrected_results
            
            self.logger.info(f"Completed {len(test_results)} significance tests")
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error running significance tests: {e}")
            raise
    
    async def run_ab_test(
        self,
        experiment_id: str,
        control_group: str,
        treatment_group: str,
        metric_name: str = "accuracy",
        confidence_level: float = 0.95
    ) -> ABTestResult:
        """
        Run A/B test comparison
        
        Args:
            experiment_id: Experiment ID
            control_group: Control group identifier
            treatment_group: Treatment group identifier
            metric_name: Metric to compare
            confidence_level: Confidence level for test
            
        Returns:
            A/B test results
        """
        self.logger.info(f"Running A/B test: {control_group} vs {treatment_group}")
        
        try:
            # Get data from database
            control_data = await self.db_manager.get_experiment_data(
                experiment_id, filters={'group': control_group}
            )
            treatment_data = await self.db_manager.get_experiment_data(
                experiment_id, filters={'group': treatment_group}
            )
            
            if not control_data or not treatment_data:
                raise ValueError("Insufficient data for A/B test")
            
            # Extract metric values
            control_values = [d[metric_name] for d in control_data if metric_name in d]
            treatment_values = [d[metric_name] for d in treatment_data if metric_name in d]
            
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
            lift = (treatment_mean - control_mean) / control_mean
            
            # Run significance test
            significance_test = self._run_t_test(
                treatment_values, control_values,
                f"A/B Test: {treatment_group} vs {control_group}"
            )
            
            # Create A/B test result
            ab_result = ABTestResult(
                experiment_id=experiment_id,
                comparison_name=f"{treatment_group}_vs_{control_group}",
                control_group=control_group,
                treatment_group=treatment_group,
                metric_name=metric_name,
                control_value=control_mean,
                treatment_value=treatment_mean,
                lift=lift,
                significance_test=significance_test,
                control_samples=len(control_values),
                treatment_samples=len(treatment_values),
                confidence_level=confidence_level
            )
            
            # Calculate statistical power
            ab_result.statistical_power = self._calculate_statistical_power(
                control_values, treatment_values, self.default_alpha
            )
            
            # Save to database
            await self._save_ab_test_result(ab_result)
            
            return ab_result
            
        except Exception as e:
            self.logger.error(f"Error running A/B test: {e}")
            raise
    
    def _results_to_dataframe(self, results: EvaluationResults) -> pd.DataFrame:
        """Convert evaluation results to pandas DataFrame"""
        data = []
        
        for result in results.results:
            # Extract basic information
            row = {
                'experiment_id': results.config.experiment_name,
                'model_id': getattr(result, 'model_id', 'unknown'),
                'is_correct': result.success,
                'response_time_ms': getattr(result, 'response_time_ms', None),
                'cost': getattr(result, 'cost', 0.0)
            }
            
            # Add transformation information if available
            if hasattr(result, 'transformation_info'):
                info = result.transformation_info
                row.update({
                    'scrambling_method': info.get('method', 'unknown'),
                    'scrambling_intensity': info.get('intensity', 0),
                    'is_original': info.get('intensity', 0) == 0
                })
            
            # Add question information if available
            if hasattr(result, 'question_info'):
                info = result.question_info
                row.update({
                    'benchmark_id': info.get('benchmark', 'unknown'),
                    'domain': info.get('domain', 'unknown'),
                    'difficulty': info.get('difficulty', 'unknown')
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    async def _calculate_language_dependency(
        self,
        model_data: pd.DataFrame,
        baseline_accuracy: float
    ) -> Dict[str, float]:
        """Calculate language dependency metrics"""
        
        # Get scrambling levels and accuracies
        scrambling_levels = model_data.get('scrambling_intensity', pd.Series([0]))
        accuracies = model_data['is_correct']
        
        # Group by scrambling level
        level_performance = model_data.groupby('scrambling_intensity')['is_correct'].mean()
        
        if len(level_performance) < 2:
            return {
                'dependency_coefficient': 0.0,
                'correlation_coefficient': 0.0,
                'correlation_p_value': 1.0
            }
        
        # Calculate dependency coefficient (1 - average_retention)
        avg_scrambled_accuracy = level_performance[level_performance.index > 0].mean()
        dependency_coefficient = 1.0 - (avg_scrambled_accuracy / baseline_accuracy) if baseline_accuracy > 0 else 0.0
        dependency_coefficient = max(0.0, min(1.0, dependency_coefficient))
        
        # Calculate correlation between scrambling level and performance
        levels = level_performance.index.values
        perfs = level_performance.values
        
        if len(levels) > 2:
            corr_coef, p_value = spearmanr(levels, perfs)
        else:
            corr_coef, p_value = 0.0, 1.0
        
        return {
            'dependency_coefficient': dependency_coefficient,
            'correlation_coefficient': corr_coef,
            'correlation_p_value': p_value
        }
    
    def _find_threshold_point(
        self,
        scrambling_data: pd.DataFrame,
        baseline_accuracy: float,
        threshold_ratio: float = 0.5
    ) -> Optional[float]:
        """Find scrambling level where performance drops below threshold"""
        
        if scrambling_data.empty:
            return None
        
        threshold_value = baseline_accuracy * threshold_ratio
        
        # Group by scrambling level
        level_performance = scrambling_data.groupby('scrambling_intensity')['is_correct'].mean()
        
        # Find first level below threshold
        below_threshold = level_performance[level_performance < threshold_value]
        
        if not below_threshold.empty:
            return below_threshold.index.min()
        
        return None
    
    def _run_t_test(
        self,
        data1: Union[pd.Series, np.ndarray, List],
        data2: Union[pd.Series, np.ndarray, List],
        test_name: str
    ) -> SignificanceTest:
        """Run independent samples t-test"""
        
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        # Remove NaN values
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if len(data1) < 2 or len(data2) < 2:
            raise ValueError("Insufficient data for t-test")
        
        # Run t-test
        statistic, p_value = ttest_ind(data1, data2, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0.0
        
        # Calculate confidence interval for difference in means
        se = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
        df = len(data1) + len(data2) - 2
        t_crit = stats.t.ppf(1 - self.default_alpha/2, df)
        mean_diff = np.mean(data1) - np.mean(data2)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        # Interpret effect size
        effect_magnitude = self._interpret_cohens_d(abs(cohens_d))
        
        return SignificanceTest(
            test_type=TestType.T_TEST,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            effect_size_type=EffectSize.COHENS_D,
            confidence_interval=(ci_lower, ci_upper),
            degrees_of_freedom=int(df),
            sample_size_1=len(data1),
            sample_size_2=len(data2),
            is_significant=p_value < self.default_alpha,
            significance_level=self.default_alpha,
            interpretation=f"{'Significant' if p_value < self.default_alpha else 'Not significant'} difference between groups",
            effect_magnitude=effect_magnitude,
            group1_mean=np.mean(data1),
            group1_std=np.std(data1, ddof=1),
            group2_mean=np.mean(data2),
            group2_std=np.std(data2, ddof=1)
        )
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_binomial_ci(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate binomial confidence interval"""
        if trials == 0:
            return 0.0, 0.0
        
        alpha = 1 - confidence
        p = successes / trials
        
        # Wilson score interval
        z = stats.norm.ppf(1 - alpha/2)
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return max(0.0, center - margin), min(1.0, center + margin)
    
    def _apply_multiple_comparison_correction(
        self,
        test_results: List[SignificanceTest]
    ) -> List[SignificanceTest]:
        """Apply Bonferroni correction for multiple comparisons"""
        
        p_values = [test.p_value for test in test_results]
        rejected, p_adjusted, _, _ = multipletests(
            p_values, alpha=self.default_alpha, method='bonferroni'
        )
        
        # Update test results
        corrected_results = []
        for i, test in enumerate(test_results):
            corrected_test = test
            corrected_test.p_value = p_adjusted[i]
            corrected_test.is_significant = rejected[i]
            corrected_test.interpretation += f" (Bonferroni corrected)"
            corrected_results.append(corrected_test)
        
        return corrected_results
    
    def _calculate_statistical_power(
        self,
        data1: List[float],
        data2: List[float],
        alpha: float
    ) -> float:
        """Calculate statistical power for t-test"""
        
        # This is a simplified power calculation
        # In practice, you might use statsmodels.stats.power for more accurate calculations
        
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Effect size
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
        
        # Power calculation (simplified)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        
        return max(0.0, min(1.0, power))
    
    async def _save_performance_metrics(
        self,
        experiment_id: str,
        metrics: pd.DataFrame
    ) -> None:
        """Save performance metrics to database"""
        # This would save to the performance_metrics table
        # Implementation depends on the DatabaseManager
        pass
    
    async def _save_threshold_analysis(
        self,
        experiment_id: str,
        threshold_data: Dict[str, Any]
    ) -> None:
        """Save threshold analysis to database"""
        # This would save to the threshold_analyses table
        pass
    
    async def _save_significance_test(
        self,
        experiment_id: str,
        test_result: SignificanceTest
    ) -> None:
        """Save significance test results to database"""
        # This would save to the statistical_analyses table
        pass
    
    async def _save_ab_test_result(self, ab_result: ABTestResult) -> None:
        """Save A/B test results to database"""
        # This would save A/B test results for tracking
        pass


class ABTestFramework:
    """
    Comprehensive A/B testing framework for experiment comparisons
    """
    
    def __init__(
        self,
        statistical_analyzer: StatisticalAnalyzer,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize A/B test framework
        
        Args:
            statistical_analyzer: Statistical analyzer instance
            logger: Logger instance
        """
        self.stats = statistical_analyzer
        self.logger = logger or logging.getLogger(__name__)
        
        # Active A/B tests
        self.active_tests: Dict[str, ABTestResult] = {}
    
    async def create_ab_test(
        self,
        experiment_id: str,
        test_name: str,
        control_group: str,
        treatment_group: str,
        primary_metric: str,
        minimum_sample_size: int = 100,
        maximum_duration_days: int = 30,
        minimum_detectable_effect: float = 0.05
    ) -> str:
        """
        Create a new A/B test
        
        Args:
            experiment_id: Parent experiment ID
            test_name: Name for the A/B test
            control_group: Control group identifier
            treatment_group: Treatment group identifier
            primary_metric: Primary metric to optimize
            minimum_sample_size: Minimum samples per group
            maximum_duration_days: Maximum test duration
            minimum_detectable_effect: Minimum effect size to detect
            
        Returns:
            A/B test ID
        """
        test_id = f"{experiment_id}_{test_name}"
        
        # Initialize test tracking
        self.active_tests[test_id] = {
            'experiment_id': experiment_id,
            'test_name': test_name,
            'control_group': control_group,
            'treatment_group': treatment_group,
            'primary_metric': primary_metric,
            'minimum_sample_size': minimum_sample_size,
            'minimum_detectable_effect': minimum_detectable_effect,
            'start_date': datetime.now(),
            'status': 'active'
        }
        
        self.logger.info(f"Created A/B test {test_id}")
        return test_id
    
    async def analyze_ab_test(self, test_id: str) -> ABTestResult:
        """
        Analyze an active A/B test
        
        Args:
            test_id: A/B test ID
            
        Returns:
            A/B test results
        """
        if test_id not in self.active_tests:
            raise ValueError(f"A/B test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        # Run the analysis
        result = await self.stats.run_ab_test(
            experiment_id=test_config['experiment_id'],
            control_group=test_config['control_group'],
            treatment_group=test_config['treatment_group'],
            metric_name=test_config['primary_metric']
        )
        
        return result
    
    async def get_ab_test_recommendations(
        self,
        test_id: str
    ) -> Dict[str, Any]:
        """
        Get recommendations for an A/B test
        
        Args:
            test_id: A/B test ID
            
        Returns:
            Dictionary with recommendations and analysis
        """
        result = await self.analyze_ab_test(test_id)
        
        recommendations = {
            'test_id': test_id,
            'recommendation': result.recommendation,
            'confidence': 'high' if result.significance_test.p_value < 0.01 else 'medium' if result.significance_test.p_value < 0.05 else 'low',
            'effect_size': result.significance_test.effect_magnitude,
            'statistical_power': result.statistical_power,
            'sample_sizes': {
                'control': result.control_samples,
                'treatment': result.treatment_samples
            },
            'next_steps': self._generate_next_steps(result)
        }
        
        return recommendations
    
    def _generate_next_steps(self, result: ABTestResult) -> List[str]:
        """Generate next steps based on A/B test results"""
        steps = []
        
        if result.is_significant_improvement:
            steps.append("Deploy treatment to all users")
            steps.append("Monitor for any negative side effects")
            steps.append("Plan follow-up experiments to further optimize")
        
        elif result.significance_test.is_significant and result.lift < 0:
            steps.append("Do not deploy treatment - shows degradation")
            steps.append("Analyze why treatment performed worse")
            steps.append("Consider alternative treatments")
        
        else:
            if result.statistical_power and result.statistical_power < 0.8:
                steps.append("Increase sample size to achieve adequate power")
            
            steps.append("Continue collecting data")
            steps.append("Consider segmentation analysis")
            steps.append("Review test design for potential issues")
        
        return steps