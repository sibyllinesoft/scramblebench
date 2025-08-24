"""
Bootstrap Inference and Confidence Intervals

Implements bootstrap methods for parameter inference, confidence intervals,
and permutation tests. Provides robust statistical inference for scaling
analysis with proper multiple testing correction.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
import warnings

from .statistical_models import ModelFit

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results"""
    parameter: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str  # 'percentile', 'bca', 'normal'
    n_bootstrap: int
    bootstrap_distribution: Optional[np.ndarray] = None
    
    @property
    def width(self) -> float:
        """Width of confidence interval"""
        return self.upper - self.lower
    
    @property
    def contains_zero(self) -> bool:
        """Whether confidence interval contains zero"""
        return self.lower <= 0 <= self.upper
    
    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class PermutationTestResult:
    """Results from permutation test"""
    test_name: str
    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int
    effect_size: Optional[float] = None
    confidence_level: float = 0.95
    
    @property
    def is_significant(self) -> bool:
        """Whether test is significant at 0.05 level"""
        return self.p_value < 0.05
    
    def get_percentile(self, percentile: float) -> float:
        """Get percentile of null distribution"""
        return np.percentile(self.null_distribution, percentile)


class BootstrapAnalyzer:
    """
    Comprehensive bootstrap analysis for statistical inference
    
    Provides bootstrap confidence intervals, bias correction,
    and permutation tests for robust inference about scaling patterns.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize bootstrap analyzer
        
        Args:
            n_bootstrap: Number of bootstrap resamples
            confidence_level: Confidence level for intervals
            random_seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)
        self.logger = logger or logging.getLogger(__name__)
    
    def bootstrap_model_parameters(
        self, 
        data: pd.DataFrame,
        model_fit_function: Callable[[pd.DataFrame], ModelFit],
        parameter_names: Optional[List[str]] = None
    ) -> Dict[str, ConfidenceInterval]:
        """
        Bootstrap confidence intervals for model parameters
        
        Args:
            data: Analysis dataset
            model_fit_function: Function that fits model and returns ModelFit
            parameter_names: Parameter names to bootstrap (default: all fixed effects)
            
        Returns:
            Dictionary of confidence intervals by parameter name
        """
        self.logger.info(f"Bootstrapping model parameters with {self.n_bootstrap} resamples")
        
        # Fit original model to get parameter names
        original_fit = model_fit_function(data)
        
        if parameter_names is None:
            parameter_names = list(original_fit.fixed_effects.keys())
        
        # Storage for bootstrap estimates
        bootstrap_estimates = {param: [] for param in parameter_names}
        successful_boots = 0
        
        for i in range(self.n_bootstrap):
            try:
                # Bootstrap resample
                boot_data = self._stratified_bootstrap_sample(data)
                
                # Fit model
                boot_fit = model_fit_function(boot_data)
                
                if boot_fit.converged:
                    for param in parameter_names:
                        if param in boot_fit.fixed_effects:
                            bootstrap_estimates[param].append(boot_fit.fixed_effects[param])
                    successful_boots += 1
                    
            except Exception as e:
                # Skip failed bootstrap samples
                continue
        
        self.logger.info(f"Successful bootstrap samples: {successful_boots}/{self.n_bootstrap}")
        
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for param in parameter_names:
            estimates = bootstrap_estimates[param]
            
            if len(estimates) < 50:  # Minimum for reliable CI
                self.logger.warning(f"Too few successful bootstraps for {param}: {len(estimates)}")
                continue
            
            original_estimate = original_fit.fixed_effects.get(param, 0.0)
            
            # Calculate different types of confidence intervals
            percentile_ci = self._percentile_ci(estimates, original_estimate)
            bca_ci = self._bias_corrected_accelerated_ci(
                estimates, original_estimate, data, model_fit_function, param
            )
            
            # Use BCa if available, otherwise percentile
            ci_method = 'bca' if bca_ci else 'percentile'
            ci_result = bca_ci if bca_ci else percentile_ci
            
            confidence_intervals[param] = ConfidenceInterval(
                parameter=param,
                estimate=original_estimate,
                lower=ci_result[0],
                upper=ci_result[1],
                confidence_level=self.confidence_level,
                method=ci_method,
                n_bootstrap=len(estimates),
                bootstrap_distribution=np.array(estimates)
            )
        
        return confidence_intervals
    
    def bootstrap_model_comparison(
        self,
        data: pd.DataFrame,
        model1_function: Callable[[pd.DataFrame], ModelFit],
        model2_function: Callable[[pd.DataFrame], ModelFit],
        comparison_metric: str = 'aic'
    ) -> PermutationTestResult:
        """
        Bootstrap comparison of two models
        
        Args:
            data: Analysis dataset
            model1_function: First model fitting function
            model2_function: Second model fitting function
            comparison_metric: Metric to compare ('aic', 'bic', 'log_likelihood')
            
        Returns:
            Bootstrap comparison results
        """
        self.logger.info(f"Bootstrapping model comparison using {comparison_metric}")
        
        # Fit original models
        original_fit1 = model1_function(data)
        original_fit2 = model2_function(data)
        
        original_diff = getattr(original_fit1, comparison_metric) - getattr(original_fit2, comparison_metric)
        
        # Bootstrap distribution of difference
        bootstrap_diffs = []
        
        for i in range(self.n_bootstrap):
            try:
                boot_data = self._stratified_bootstrap_sample(data)
                
                boot_fit1 = model1_function(boot_data)
                boot_fit2 = model2_function(boot_data)
                
                if boot_fit1.converged and boot_fit2.converged:
                    boot_diff = getattr(boot_fit1, comparison_metric) - getattr(boot_fit2, comparison_metric)
                    bootstrap_diffs.append(boot_diff)
                    
            except Exception:
                continue
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # P-value: proportion of bootstrap differences more extreme than observed
        if comparison_metric in ['aic', 'bic']:  # Lower is better
            p_value = np.mean(bootstrap_diffs >= abs(original_diff))
        else:  # log_likelihood: higher is better
            p_value = np.mean(np.abs(bootstrap_diffs) >= abs(original_diff))
        
        return PermutationTestResult(
            test_name=f"Bootstrap model comparison ({comparison_metric})",
            observed_statistic=original_diff,
            null_distribution=bootstrap_diffs,
            p_value=p_value,
            n_permutations=len(bootstrap_diffs)
        )
    
    def bootstrap_scaling_correlation(
        self,
        data: pd.DataFrame,
        parameter_col: str = 'logN',
        outcome_col: str = 'is_correct',
        groupby_cols: Optional[List[str]] = None
    ) -> Dict[str, ConfidenceInterval]:
        """
        Bootstrap confidence intervals for scaling correlations
        
        Args:
            data: Analysis dataset
            parameter_col: Column with parameter measure (e.g., logN)
            outcome_col: Column with outcome measure
            groupby_cols: Optional grouping columns
            
        Returns:
            Confidence intervals for correlations
        """
        self.logger.info("Bootstrapping scaling correlations")
        
        if groupby_cols is None:
            # Global correlation
            original_corr, _ = stats.spearmanr(data[parameter_col], data[outcome_col])
            
            bootstrap_corrs = []
            for i in range(self.n_bootstrap):
                boot_data = self._bootstrap_sample(data)
                boot_corr, _ = stats.spearmanr(boot_data[parameter_col], boot_data[outcome_col])
                if not np.isnan(boot_corr):
                    bootstrap_corrs.append(boot_corr)
            
            ci = self._percentile_ci(bootstrap_corrs, original_corr)
            
            return {
                'global_correlation': ConfidenceInterval(
                    parameter='spearman_correlation',
                    estimate=original_corr,
                    lower=ci[0],
                    upper=ci[1],
                    confidence_level=self.confidence_level,
                    method='percentile',
                    n_bootstrap=len(bootstrap_corrs)
                )
            }
        
        else:
            # Group-wise correlations
            results = {}
            
            for group_name, group_data in data.groupby(groupby_cols):
                if len(group_data) < 10:  # Minimum sample size
                    continue
                
                original_corr, _ = stats.spearmanr(group_data[parameter_col], group_data[outcome_col])
                
                bootstrap_corrs = []
                for i in range(self.n_bootstrap):
                    boot_indices = self.rng.choice(len(group_data), len(group_data), replace=True)
                    boot_group = group_data.iloc[boot_indices]
                    boot_corr, _ = stats.spearmanr(boot_group[parameter_col], boot_group[outcome_col])
                    if not np.isnan(boot_corr):
                        bootstrap_corrs.append(boot_corr)
                
                if len(bootstrap_corrs) > 50:
                    ci = self._percentile_ci(bootstrap_corrs, original_corr)
                    
                    group_key = str(group_name) if isinstance(group_name, (str, int, float)) else '_'.join(map(str, group_name))
                    
                    results[group_key] = ConfidenceInterval(
                        parameter=f'correlation_{group_key}',
                        estimate=original_corr,
                        lower=ci[0],
                        upper=ci[1],
                        confidence_level=self.confidence_level,
                        method='percentile',
                        n_bootstrap=len(bootstrap_corrs)
                    )
            
            return results
    
    def permutation_test_scaling_difference(
        self,
        data: pd.DataFrame,
        group_col: str,
        outcome_col: str = 'is_correct',
        n_permutations: Optional[int] = None
    ) -> PermutationTestResult:
        """
        Permutation test for differences in scaling patterns between groups
        
        Args:
            data: Analysis dataset
            group_col: Column defining groups to compare
            outcome_col: Outcome variable
            n_permutations: Number of permutations (default: use bootstrap setting)
            
        Returns:
            Permutation test results
        """
        if n_permutations is None:
            n_permutations = self.n_bootstrap
        
        self.logger.info(f"Running permutation test with {n_permutations} permutations")
        
        groups = data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Permutation test requires exactly 2 groups")
        
        group1_data = data[data[group_col] == groups[0]]
        group2_data = data[data[group_col] == groups[1]]
        
        # Original difference in means
        original_diff = group1_data[outcome_col].mean() - group2_data[outcome_col].mean()
        
        # Permutation distribution
        combined_data = data.copy()
        null_diffs = []
        
        for i in range(n_permutations):
            # Permute group labels
            permuted_labels = self.rng.permutation(combined_data[group_col].values)
            combined_data[f'{group_col}_perm'] = permuted_labels
            
            # Calculate difference for permuted data
            perm_group1 = combined_data[combined_data[f'{group_col}_perm'] == groups[0]]
            perm_group2 = combined_data[combined_data[f'{group_col}_perm'] == groups[1]]
            
            perm_diff = perm_group1[outcome_col].mean() - perm_group2[outcome_col].mean()
            null_diffs.append(perm_diff)
        
        null_diffs = np.array(null_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(original_diff))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(group1_data) - 1) * group1_data[outcome_col].var() + 
             (len(group2_data) - 1) * group2_data[outcome_col].var()) /
            (len(group1_data) + len(group2_data) - 2)
        )
        
        effect_size = original_diff / pooled_std if pooled_std > 0 else 0.0
        
        return PermutationTestResult(
            test_name=f"Permutation test: {groups[0]} vs {groups[1]}",
            observed_statistic=original_diff,
            null_distribution=null_diffs,
            p_value=p_value,
            n_permutations=n_permutations,
            effect_size=effect_size
        )
    
    def _stratified_bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create stratified bootstrap sample preserving group structure"""
        
        # Stratify by key variables to preserve structure
        strata_cols = []
        for col in ['model_family', 'domain', 'condition_f']:
            if col in data.columns:
                strata_cols.append(col)
        
        if not strata_cols:
            return self._bootstrap_sample(data)
        
        # Sample within strata
        bootstrap_samples = []
        
        for strata_values, strata_data in data.groupby(strata_cols):
            if len(strata_data) > 0:
                boot_indices = self.rng.choice(len(strata_data), len(strata_data), replace=True)
                bootstrap_samples.append(strata_data.iloc[boot_indices])
        
        if bootstrap_samples:
            return pd.concat(bootstrap_samples, ignore_index=True)
        else:
            return self._bootstrap_sample(data)
    
    def _bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple bootstrap sample"""
        boot_indices = self.rng.choice(len(data), len(data), replace=True)
        return data.iloc[boot_indices].reset_index(drop=True)
    
    def _percentile_ci(self, bootstrap_estimates: List[float], original_estimate: float) -> Tuple[float, float]:
        """Calculate percentile confidence interval"""
        
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        lower = np.percentile(bootstrap_estimates, lower_percentile)
        upper = np.percentile(bootstrap_estimates, upper_percentile)
        
        return (lower, upper)
    
    def _bias_corrected_accelerated_ci(
        self,
        bootstrap_estimates: List[float],
        original_estimate: float,
        data: pd.DataFrame,
        model_fit_function: Callable,
        parameter_name: str
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate bias-corrected and accelerated (BCa) confidence interval
        
        This is more accurate than percentile CI but computationally intensive.
        """
        
        try:
            bootstrap_estimates = np.array(bootstrap_estimates)
            
            # Bias correction
            n_below = np.sum(bootstrap_estimates < original_estimate)
            bias_correction = stats.norm.ppf(n_below / len(bootstrap_estimates))
            
            # Acceleration (via jackknife)
            n = len(data)
            jackknife_estimates = []
            
            # Use subset for computational efficiency
            jackknife_indices = self.rng.choice(n, min(n, 200), replace=False)
            
            for i in jackknife_indices:
                try:
                    # Leave-one-out sample
                    jackknife_data = data.drop(data.index[i])
                    jackknife_fit = model_fit_function(jackknife_data)
                    
                    if jackknife_fit.converged and parameter_name in jackknife_fit.fixed_effects:
                        jackknife_estimates.append(jackknife_fit.fixed_effects[parameter_name])
                        
                except Exception:
                    continue
            
            if len(jackknife_estimates) < 50:
                return None  # Fall back to percentile
            
            jackknife_estimates = np.array(jackknife_estimates)
            jackknife_mean = np.mean(jackknife_estimates)
            
            # Acceleration parameter
            numerator = np.sum((jackknife_mean - jackknife_estimates) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_estimates) ** 2)) ** 1.5
            
            if denominator == 0:
                acceleration = 0
            else:
                acceleration = numerator / denominator
            
            # Adjusted percentiles
            alpha = 1 - self.confidence_level
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
            
            alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2)))
            alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2)))
            
            # Ensure valid percentiles
            alpha_1 = max(0.001, min(0.999, alpha_1))
            alpha_2 = max(0.001, min(0.999, alpha_2))
            
            lower = np.percentile(bootstrap_estimates, 100 * alpha_1)
            upper = np.percentile(bootstrap_estimates, 100 * alpha_2)
            
            return (lower, upper)
            
        except Exception as e:
            self.logger.warning(f"BCa CI calculation failed: {e}, using percentile CI")
            return None


class ConfidenceIntervals:
    """
    Container and utilities for confidence interval collections
    """
    
    def __init__(self, intervals: Dict[str, ConfidenceInterval]):
        """Initialize with dictionary of confidence intervals"""
        self.intervals = intervals
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for display/export"""
        
        rows = []
        for name, ci in self.intervals.items():
            rows.append({
                'Parameter': ci.parameter,
                'Estimate': ci.estimate,
                'CI_Lower': ci.lower,
                'CI_Upper': ci.upper,
                'CI_Width': ci.width,
                'Contains_Zero': ci.contains_zero,
                'Confidence_Level': ci.confidence_level,
                'Method': ci.method,
                'N_Bootstrap': ci.n_bootstrap
            })
        
        return pd.DataFrame(rows)
    
    def significant_parameters(self, alpha: float = 0.05) -> List[str]:
        """Get parameters with confidence intervals not containing zero"""
        
        significant = []
        for name, ci in self.intervals.items():
            if not ci.contains_zero:
                significant.append(name)
        
        return significant
    
    def summary_table(self) -> str:
        """Create formatted summary table"""
        
        df = self.to_dataframe()
        
        # Format for display
        df['CI'] = df.apply(lambda row: f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]", axis=1)
        display_cols = ['Parameter', 'Estimate', 'CI', 'Contains_Zero', 'Method']
        
        return df[display_cols].to_string(index=False)


class PermutationTests:
    """Container and utilities for permutation test collections"""
    
    def __init__(self, tests: Dict[str, PermutationTestResult]):
        """Initialize with dictionary of permutation tests"""
        self.tests = tests
    
    def apply_multiple_testing_correction(self, method: str = 'bonferroni') -> 'PermutationTests':
        """Apply multiple testing correction to p-values"""
        
        from statsmodels.stats.multitest import multipletests
        
        p_values = [test.p_value for test in self.tests.values()]
        test_names = list(self.tests.keys())
        
        if len(p_values) <= 1:
            return self  # No correction needed
        
        # Apply correction
        rejected, p_corrected, _, _ = multipletests(p_values, method=method)
        
        # Create corrected tests
        corrected_tests = {}
        for i, (name, original_test) in enumerate(self.tests.items()):
            # Create new test result with corrected p-value
            corrected_test = PermutationTestResult(
                test_name=f"{original_test.test_name} ({method} corrected)",
                observed_statistic=original_test.observed_statistic,
                null_distribution=original_test.null_distribution,
                p_value=p_corrected[i],
                n_permutations=original_test.n_permutations,
                effect_size=original_test.effect_size
            )
            corrected_tests[name] = corrected_test
        
        return PermutationTests(corrected_tests)
    
    def significant_tests(self, alpha: float = 0.05) -> List[str]:
        """Get names of significant tests"""
        return [name for name, test in self.tests.items() if test.p_value < alpha]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis"""
        
        rows = []
        for name, test in self.tests.items():
            rows.append({
                'Test': test.test_name,
                'Observed_Statistic': test.observed_statistic,
                'P_Value': test.p_value,
                'Significant': test.is_significant,
                'Effect_Size': test.effect_size,
                'N_Permutations': test.n_permutations
            })
        
        return pd.DataFrame(rows)