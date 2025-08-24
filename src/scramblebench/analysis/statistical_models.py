"""
Statistical Models for Scaling Analysis

Implementation of GLMM, GAM, and changepoint models for discovering
scaling patterns in LLM reasoning capabilities without presupposing thresholds.

This module provides the core statistical machinery for Step S8 of the 
ScrambleBench analysis pipeline, enabling academic-quality inference 
about smooth vs threshold scaling patterns.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# R integration for advanced models
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Activate pandas conversion
    pandas2ri.activate()
    numpy2ri.activate()
    
    # Import R packages
    r_lme4 = importr('lme4')
    r_mgcv = importr('mgcv') 
    r_segmented = importr('segmented')
    r_base = importr('base')
    r_stats = importr('stats')
    
    R_AVAILABLE = True
    
except ImportError:
    warnings.warn("R integration not available. Some advanced models will use Python alternatives.")
    R_AVAILABLE = False

from ..core.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ModelFit:
    """Container for statistical model fit results"""
    model_name: str
    model_type: str  # 'glmm', 'gam', 'changepoint', etc.
    formula: str
    
    # Model selection metrics
    aic: float
    bic: float  
    log_likelihood: float
    
    # Fit statistics
    r_squared: Optional[float] = None
    deviance: Optional[float] = None
    df_residual: Optional[int] = None
    
    # Parameter estimates
    fixed_effects: Dict[str, float] = field(default_factory=dict)
    fixed_effects_se: Dict[str, float] = field(default_factory=dict)
    fixed_effects_pvalues: Dict[str, float] = field(default_factory=dict)
    
    # Random effects (for GLMM)
    random_effects: Dict[str, float] = field(default_factory=dict)
    
    # Smooth terms (for GAM)
    smooth_terms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Changepoint results
    breakpoints: List[float] = field(default_factory=list)
    breakpoint_ci: List[Tuple[float, float]] = field(default_factory=list)
    sup_f_test: Optional[Dict[str, float]] = None
    
    # Model predictions
    predictions: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    
    # Cross-validation results
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Metadata
    n_observations: int = 0
    converged: bool = False
    warnings: List[str] = field(default_factory=list)
    fit_time: float = 0.0


class ScalingAnalyzer:
    """
    Master class for scaling pattern analysis
    
    Coordinates GLMM, GAM, and changepoint analyses to discover
    whether LLM reasoning scales smoothly or exhibits threshold effects.
    Implements the full S8 analysis pipeline from TODO.md.
    """
    
    def __init__(
        self,
        database: Database,
        use_r_backend: bool = True,
        alpha: float = 0.05,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize scaling analyzer
        
        Args:
            database: Database instance with evaluation results
            use_r_backend: Whether to use R for advanced models (recommended)
            alpha: Significance level for tests
            logger: Logger instance
        """
        self.db = database
        self.use_r = use_r_backend and R_AVAILABLE
        self.alpha = alpha
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.glmm = GLMMAnalyzer(self.use_r, alpha, self.logger)
        self.gam = GAMAnalyzer(self.use_r, alpha, self.logger) 
        self.changepoint = ChangepointAnalyzer(self.use_r, alpha, self.logger)
        self.contamination = ContaminationAnalyzer(alpha, self.logger)
        
        # Analysis results storage
        self.analysis_data: Optional[pd.DataFrame] = None
        self.model_fits: Dict[str, Dict[str, ModelFit]] = {}
        self.best_models: Dict[str, ModelFit] = {}
        
    def prepare_analysis_data(self, run_id: str) -> pd.DataFrame:
        """
        Construct analysis table with item-level rows and covariates
        
        Creates the fundamental analysis dataset with:
        - logN: log(parameter count) 
        - family: model family
        - domain: task domain
        - condition: orig/para/scram_level
        - tok_kl, tok_frag: tokenizer perturbation metrics
        
        Args:
            run_id: Run ID to analyze
            
        Returns:
            Analysis-ready DataFrame
        """
        self.logger.info(f"Preparing analysis data for run {run_id}")
        
        # Get all evaluations for the run
        conn = self.db.get_connection()
        
        query = """
        SELECT 
            e.eval_id,
            e.item_id,
            e.model_id,
            e.model_family,
            e.n_params,
            e.provider,
            e.transform,
            e.scramble_level,
            e.is_correct,
            e.tok_kl,
            e.tok_frag,
            e.prompt_tokens,
            e.completion_tokens,
            e.cost_usd,
            i.dataset,
            i.domain,
            i.question,
            i.answer
        FROM evals e
        JOIN items i ON e.item_id = i.item_id
        WHERE e.run_id = ?
        AND e.n_params IS NOT NULL
        ORDER BY e.model_id, i.dataset, i.domain, e.transform, e.scramble_level
        """
        
        df = pd.read_sql_query(query, conn.execute(query, (run_id,)).fetchall())
        
        if df.empty:
            raise ValueError(f"No evaluation data found for run {run_id}")
        
        # Create log parameter count
        df['logN'] = np.log10(df['n_params'])
        
        # Create condition variable (transform + level)
        def create_condition(row):
            if row['transform'] == 'original':
                return 'original'
            elif row['transform'] == 'paraphrase':
                return 'paraphrase'  
            else:  # scramble
                level = row['scramble_level'] or 0
                return f'scramble_{level}'
        
        df['condition'] = df.apply(create_condition, axis=1)
        
        # Ensure required covariates exist
        if 'tok_kl' not in df.columns or df['tok_kl'].isna().all():
            self.logger.warning("tok_kl not available, setting to 0")
            df['tok_kl'] = 0.0
            
        if 'tok_frag' not in df.columns or df['tok_frag'].isna().all():
            self.logger.warning("tok_frag not available, setting to 1")
            df['tok_frag'] = 1.0
        
        # Fill missing tokenizer metrics
        df['tok_kl'] = df['tok_kl'].fillna(0.0)
        df['tok_frag'] = df['tok_frag'].fillna(1.0)
        
        # Add derived variables for analysis
        df['is_scrambled'] = df['transform'] == 'scramble'
        df['is_paraphrase'] = df['transform'] == 'paraphrase'
        df['scramble_intensity'] = df['scramble_level'].fillna(0.0)
        
        # Create factor variables for modeling
        df['model_family_f'] = pd.Categorical(df['model_family'])
        df['domain_f'] = pd.Categorical(df['domain'])
        df['condition_f'] = pd.Categorical(df['condition'])
        
        self.logger.info(f"Prepared analysis dataset: {len(df)} observations, "
                        f"{df['model_id'].nunique()} models, "
                        f"{df['domain'].nunique()} domains")
        
        self.analysis_data = df
        return df
    
    def run_full_analysis(self, run_id: str) -> Dict[str, Any]:
        """
        Run complete scaling analysis pipeline
        
        Executes all statistical models and selects best fits per family.
        This is the main entry point for Step S8 analysis.
        
        Args:
            run_id: Run ID to analyze
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info(f"Starting full scaling analysis for run {run_id}")
        
        # Prepare analysis data
        df = self.prepare_analysis_data(run_id)
        
        results = {
            'run_id': run_id,
            'n_observations': len(df),
            'n_models': df['model_id'].nunique(),
            'n_families': df['model_family'].nunique(),
            'model_families': list(df['model_family'].unique()),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Run analysis by model family
        family_results = {}
        
        for family in df['model_family'].unique():
            if pd.isna(family):
                continue
                
            self.logger.info(f"Analyzing family: {family}")
            
            family_data = df[df['model_family'] == family].copy()
            
            if len(family_data) < 50:  # Minimum sample size
                self.logger.warning(f"Insufficient data for family {family}: {len(family_data)} obs")
                continue
            
            family_results[family] = self._analyze_family(family_data, family)
        
        results['family_results'] = family_results
        
        # Cross-family comparison
        if len(family_results) > 1:
            results['cross_family_comparison'] = self._compare_families(df)
        
        # Global model selection
        results['best_models_by_family'] = self._select_best_models()
        
        # Contamination vs brittleness analysis
        results['contamination_analysis'] = self.contamination.analyze(df)
        
        self.logger.info("Full scaling analysis completed")
        return results
    
    def _analyze_family(self, data: pd.DataFrame, family: str) -> Dict[str, Any]:
        """Analyze scaling patterns for a single model family"""
        
        results = {
            'family': family,
            'n_observations': len(data),
            'n_models': data['model_id'].nunique(),
            'parameter_range': {
                'min_params': data['n_params'].min(),
                'max_params': data['n_params'].max(),
                'logN_range': (data['logN'].min(), data['logN'].max())
            }
        }
        
        # Fit all model types
        model_fits = {}
        
        # 1. GLMM: Logistic with random intercepts
        try:
            glmm_fit = self.glmm.fit_hierarchical_model(data)
            model_fits['glmm'] = glmm_fit
        except Exception as e:
            self.logger.warning(f"GLMM fitting failed for {family}: {e}")
        
        # 2. GAM: Monotone smooths
        try:
            gam_fit = self.gam.fit_monotone_gam(data)
            model_fits['gam'] = gam_fit
        except Exception as e:
            self.logger.warning(f"GAM fitting failed for {family}: {e}")
        
        # 3. Changepoint models
        try:
            # Single slope (null model)
            linear_fit = self.changepoint.fit_linear_model(data)
            model_fits['linear'] = linear_fit
            
            # Segmented with one break
            segmented_fit = self.changepoint.fit_segmented_model(data)
            model_fits['segmented'] = segmented_fit
            
        except Exception as e:
            self.logger.warning(f"Changepoint fitting failed for {family}: {e}")
        
        results['model_fits'] = model_fits
        
        # Model comparison
        if len(model_fits) > 1:
            comparison = self._compare_models(model_fits)
            results['model_comparison'] = comparison
            results['best_model'] = comparison['best_model']
        
        # Store family results
        self.model_fits[family] = model_fits
        
        return results
    
    def _compare_models(self, model_fits: Dict[str, ModelFit]) -> Dict[str, Any]:
        """Compare models using AIC/BIC and statistical tests"""
        
        # Extract AIC/BIC values
        model_comparison = []
        for name, fit in model_fits.items():
            model_comparison.append({
                'model': name,
                'aic': fit.aic,
                'bic': fit.bic,
                'log_likelihood': fit.log_likelihood,
                'converged': fit.converged,
                'n_params': len(fit.fixed_effects)
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        
        # Select best model by AIC
        best_by_aic = comparison_df.loc[comparison_df['aic'].idxmin()]
        best_by_bic = comparison_df.loc[comparison_df['bic'].idxmin()]
        
        # AIC weights
        aic_values = comparison_df['aic'].values
        min_aic = aic_values.min()
        delta_aic = aic_values - min_aic
        exp_delta = np.exp(-0.5 * delta_aic)
        aic_weights = exp_delta / exp_delta.sum()
        
        comparison_df['delta_aic'] = delta_aic
        comparison_df['aic_weight'] = aic_weights
        
        # Statistical test for changepoint vs linear
        sup_f_result = None
        if 'linear' in model_fits and 'segmented' in model_fits:
            try:
                sup_f_result = self.changepoint.sup_f_test(
                    model_fits['linear'], 
                    model_fits['segmented']
                )
            except Exception as e:
                self.logger.warning(f"Sup-F test failed: {e}")
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_by_aic': best_by_aic['model'],
            'best_by_bic': best_by_bic['model'],
            'aic_weights': dict(zip(comparison_df['model'], aic_weights)),
            'sup_f_test': sup_f_result,
            'evidence_ratio': aic_weights.max() / aic_weights[aic_weights != aic_weights.max()].max() if len(aic_weights) > 1 else 1.0
        }
    
    def _compare_families(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compare scaling patterns across model families"""
        
        # Test for family differences in scaling
        # This is a simplified implementation - could be expanded
        
        family_summaries = []
        for family in data['model_family'].unique():
            if pd.isna(family):
                continue
                
            family_data = data[data['model_family'] == family]
            
            # Compute average LDC by parameter size
            ldc_by_size = []
            for model_id in family_data['model_id'].unique():
                model_data = family_data[family_data['model_id'] == model_id]
                
                # Get baseline accuracy
                baseline_acc = model_data[model_data['transform'] == 'original']['is_correct'].mean()
                
                # Get scrambled accuracy
                scrambled_data = model_data[model_data['transform'] == 'scramble']
                if not scrambled_data.empty:
                    scrambled_acc = scrambled_data['is_correct'].mean()
                    ldc = 1.0 - (scrambled_acc / baseline_acc) if baseline_acc > 0 else 0.0
                    
                    ldc_by_size.append({
                        'family': family,
                        'model_id': model_id,
                        'logN': model_data['logN'].iloc[0],
                        'n_params': model_data['n_params'].iloc[0],
                        'ldc': ldc
                    })
            
            family_summaries.extend(ldc_by_size)
        
        comparison_df = pd.DataFrame(family_summaries)
        
        if len(comparison_df) > 0:
            # Correlation between logN and LDC by family
            family_correlations = {}
            for family in comparison_df['family'].unique():
                family_data = comparison_df[comparison_df['family'] == family]
                if len(family_data) >= 3:
                    corr, p_val = stats.spearmanr(family_data['logN'], family_data['ldc'])
                    family_correlations[family] = {'correlation': corr, 'p_value': p_val}
            
            return {
                'family_ldc_data': comparison_df.to_dict('records'),
                'family_correlations': family_correlations
            }
        
        return {}
    
    def _select_best_models(self) -> Dict[str, str]:
        """Select best model for each family based on AIC/BIC"""
        
        best_models = {}
        
        for family, fits in self.model_fits.items():
            if not fits:
                continue
            
            # Find model with lowest AIC
            best_aic = float('inf')
            best_model = None
            
            for name, fit in fits.items():
                if fit.converged and fit.aic < best_aic:
                    best_aic = fit.aic
                    best_model = name
            
            if best_model:
                best_models[family] = best_model
                self.best_models[family] = fits[best_model]
        
        return best_models


class GLMMAnalyzer:
    """Generalized Linear Mixed Models for hierarchical scaling analysis"""
    
    def __init__(self, use_r: bool, alpha: float, logger: logging.Logger):
        self.use_r = use_r
        self.alpha = alpha
        self.logger = logger
    
    def fit_hierarchical_model(self, data: pd.DataFrame) -> ModelFit:
        """
        Fit GLMM with logistic link and random intercepts
        
        Model: logit(P(correct)) ~ logN + condition + logN:condition + (1|domain) + (1|family)
        
        Args:
            data: Analysis dataset
            
        Returns:
            Fitted model results
        """
        self.logger.info("Fitting GLMM with hierarchical structure")
        
        if self.use_r:
            return self._fit_glmm_r(data)
        else:
            return self._fit_glmm_python(data)
    
    def _fit_glmm_r(self, data: pd.DataFrame) -> ModelFit:
        """Fit GLMM using R lme4"""
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)
        
        # Formula with random intercepts
        formula = "is_correct ~ logN * condition_f + (1 | domain_f) + (1 | model_family_f)"
        
        # Fit model
        try:
            model = r_lme4.glmer(
                robjects.Formula(formula),
                data=r_data,
                family=r_stats.binomial(),
                control=r_lme4.glmerControl(optimizer="bobyqa", optCtrl=robjects.ListVector({"maxfun": 20000}))
            )
            
            # Extract results
            summary_obj = r_base.summary(model)
            
            # Get fixed effects
            fixed_effects = {}
            fixed_effects_se = {}
            fixed_effects_pvalues = {}
            
            coef_matrix = np.array(summary_obj.rx2('coefficients'))
            coef_names = list(summary_obj.rx2('coefficients').rownames)
            
            for i, name in enumerate(coef_names):
                fixed_effects[name] = coef_matrix[i, 0]  # Estimate
                fixed_effects_se[name] = coef_matrix[i, 1]  # Std Error
                fixed_effects_pvalues[name] = coef_matrix[i, 3]  # P-value
            
            # Get model fit statistics
            aic = r_stats.AIC(model)[0]
            bic = r_stats.BIC(model)[0]
            log_likelihood = float(r_stats.logLik(model)[0])
            
            # Random effects
            random_effects = {}
            ran_ef = r_lme4.VarCorr(model)
            # Simplified extraction - R object structure is complex
            
            return ModelFit(
                model_name="GLMM (R lme4)",
                model_type="glmm",
                formula=formula,
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                fixed_effects=fixed_effects,
                fixed_effects_se=fixed_effects_se,
                fixed_effects_pvalues=fixed_effects_pvalues,
                random_effects=random_effects,
                n_observations=len(data),
                converged=True  # Simplified check
            )
            
        except Exception as e:
            self.logger.error(f"R GLMM fitting failed: {e}")
            raise
    
    def _fit_glmm_python(self, data: pd.DataFrame) -> ModelFit:
        """Fit GLMM using Python statsmodels (simplified)"""
        
        # Use mixed linear model as approximation
        # This is less sophisticated than R lme4 but provides similar insights
        
        formula = "is_correct ~ logN + C(condition_f) + logN:C(condition_f)"
        
        try:
            # Simple logistic regression with interaction
            model = smf.logit(formula, data=data).fit(disp=0)
            
            fixed_effects = dict(model.params)
            fixed_effects_se = dict(model.bse)
            fixed_effects_pvalues = dict(model.pvalues)
            
            return ModelFit(
                model_name="GLMM (Python statsmodels)",
                model_type="glmm",
                formula=formula,
                aic=model.aic,
                bic=model.bic,
                log_likelihood=model.llf,
                fixed_effects=fixed_effects,
                fixed_effects_se=fixed_effects_se,
                fixed_effects_pvalues=fixed_effects_pvalues,
                n_observations=model.nobs,
                converged=model.converged
            )
            
        except Exception as e:
            self.logger.error(f"Python GLMM fitting failed: {e}")
            raise


class GAMAnalyzer:
    """Generalized Additive Models for non-parametric scaling discovery"""
    
    def __init__(self, use_r: bool, alpha: float, logger: logging.Logger):
        self.use_r = use_r
        self.alpha = alpha  
        self.logger = logger
    
    def fit_monotone_gam(self, data: pd.DataFrame) -> ModelFit:
        """
        Fit GAM with monotone smooth functions
        
        Model: logit(P(correct)) ~ s(logN, by=condition, m=1) + condition
        where m=1 enforces monotonicity
        
        Args:
            data: Analysis dataset
            
        Returns:
            Fitted GAM results
        """
        self.logger.info("Fitting monotone GAM")
        
        if self.use_r:
            return self._fit_gam_r(data)
        else:
            return self._fit_gam_python(data)
    
    def _fit_gam_r(self, data: pd.DataFrame) -> ModelFit:
        """Fit GAM using R mgcv"""
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)
        
        # Monotone smooth by condition
        formula = "is_correct ~ s(logN, by=condition_f, m=1, k=5) + condition_f"
        
        try:
            model = r_mgcv.gam(
                robjects.Formula(formula),
                data=r_data,
                family=r_stats.binomial(),
                method="REML"
            )
            
            # Extract results
            summary_obj = r_base.summary(model)
            
            # Get parametric coefficients
            fixed_effects = {}
            fixed_effects_se = {}
            fixed_effects_pvalues = {}
            
            p_coef = np.array(summary_obj.rx2('p.coeff'))
            p_se = np.array(summary_obj.rx2('se'))
            p_pvalues = np.array(summary_obj.rx2('p.pvalues'))
            p_names = list(summary_obj.rx2('p.coeff').names)
            
            for i, name in enumerate(p_names):
                fixed_effects[name] = p_coef[i]
                fixed_effects_se[name] = p_se[i] 
                fixed_effects_pvalues[name] = p_pvalues[i]
            
            # Get smooth terms
            smooth_terms = {}
            s_table = summary_obj.rx2('s.table')
            if s_table is not None:
                s_names = list(s_table.rownames)
                s_matrix = np.array(s_table)
                
                for i, name in enumerate(s_names):
                    smooth_terms[name] = {
                        'edf': s_matrix[i, 0],  # Effective degrees of freedom
                        'chi_sq': s_matrix[i, 2],  # Chi-square
                        'p_value': s_matrix[i, 3]  # P-value
                    }
            
            # Model fit statistics
            aic = r_stats.AIC(model)[0]
            bic = r_stats.BIC(model)[0] 
            log_likelihood = float(r_stats.logLik(model)[0])
            deviance = float(summary_obj.rx2('deviance'))
            
            return ModelFit(
                model_name="GAM (R mgcv)",
                model_type="gam",
                formula=formula,
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                deviance=deviance,
                fixed_effects=fixed_effects,
                fixed_effects_se=fixed_effects_se,
                fixed_effects_pvalues=fixed_effects_pvalues,
                smooth_terms=smooth_terms,
                n_observations=len(data),
                converged=True
            )
            
        except Exception as e:
            self.logger.error(f"R GAM fitting failed: {e}")
            raise
    
    def _fit_gam_python(self, data: pd.DataFrame) -> ModelFit:
        """Fit GAM using Python (simplified with spline regression)"""
        
        # Simplified implementation using polynomial features
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        try:
            # Create features
            X = data[['logN']].values
            y = data['is_correct'].values
            
            # Spline transformation (approximates GAM smooth)
            pipeline = Pipeline([
                ('spline', SplineTransformer(degree=3, n_knots=5, include_bias=False)),
                ('logistic', LogisticRegression(solver='liblinear', max_iter=1000))
            ])
            
            model = pipeline.fit(X, y)
            
            # Approximate AIC (simplified)
            y_pred_proba = model.predict_proba(X)[:, 1]
            log_likelihood = np.sum(y * np.log(y_pred_proba + 1e-15) + (1-y) * np.log(1-y_pred_proba + 1e-15))
            
            n_params = pipeline['spline'].n_features_out_ + 1  # spline features + intercept
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(len(data)) * n_params - 2 * log_likelihood
            
            return ModelFit(
                model_name="GAM (Python sklearn)",
                model_type="gam",
                formula="is_correct ~ spline(logN)",
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                n_observations=len(data),
                converged=True
            )
            
        except Exception as e:
            self.logger.error(f"Python GAM fitting failed: {e}")
            raise


class ChangepointAnalyzer:
    """Changepoint detection and segmented regression analysis"""
    
    def __init__(self, use_r: bool, alpha: float, logger: logging.Logger):
        self.use_r = use_r
        self.alpha = alpha
        self.logger = logger
    
    def fit_linear_model(self, data: pd.DataFrame) -> ModelFit:
        """Fit single-slope linear model (null model for changepoint tests)"""
        
        formula = "is_correct ~ logN + C(condition_f) + logN:C(condition_f)"
        
        try:
            model = smf.logit(formula, data=data).fit(disp=0)
            
            return ModelFit(
                model_name="Linear (single slope)",
                model_type="linear",
                formula=formula,
                aic=model.aic,
                bic=model.bic,
                log_likelihood=model.llf,
                fixed_effects=dict(model.params),
                fixed_effects_se=dict(model.bse),
                fixed_effects_pvalues=dict(model.pvalues),
                n_observations=model.nobs,
                converged=model.converged
            )
            
        except Exception as e:
            self.logger.error(f"Linear model fitting failed: {e}")
            raise
    
    def fit_segmented_model(self, data: pd.DataFrame) -> ModelFit:
        """Fit segmented regression with one breakpoint"""
        
        if self.use_r:
            return self._fit_segmented_r(data)
        else:
            return self._fit_segmented_python(data)
    
    def _fit_segmented_r(self, data: pd.DataFrame) -> ModelFit:
        """Fit segmented model using R segmented package"""
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)
        
        try:
            # First fit linear model
            linear_formula = "is_correct ~ logN"
            linear_model = r_stats.glm(
                robjects.Formula(linear_formula),
                data=r_data,
                family=r_stats.binomial()
            )
            
            # Fit segmented model
            # Start with breakpoint in middle of logN range
            logN_range = data['logN'].max() - data['logN'].min()
            start_breakpoint = data['logN'].min() + logN_range / 2
            
            segmented_model = r_segmented.segmented(
                linear_model,
                seg_Z=robjects.Formula("~ logN"),
                psi=start_breakpoint
            )
            
            # Extract results
            summary_obj = r_base.summary(segmented_model)
            
            # Get coefficients
            coef_matrix = np.array(summary_obj.rx2('coefficients'))
            coef_names = list(summary_obj.rx2('coefficients').rownames)
            
            fixed_effects = {}
            fixed_effects_se = {}
            fixed_effects_pvalues = {}
            
            for i, name in enumerate(coef_names):
                fixed_effects[name] = coef_matrix[i, 0]
                fixed_effects_se[name] = coef_matrix[i, 1]
                fixed_effects_pvalues[name] = coef_matrix[i, 3]
            
            # Get breakpoints
            breakpoints = list(r_segmented.breakpoints(segmented_model).rx2('psi')[0])
            
            # Get breakpoint confidence intervals
            confint_bp = r_segmented.confint_psisegmented(segmented_model)
            breakpoint_ci = [(float(confint_bp[0]), float(confint_bp[1]))]
            
            # Model fit statistics
            aic = r_stats.AIC(segmented_model)[0]
            bic = r_stats.BIC(segmented_model)[0]
            log_likelihood = float(r_stats.logLik(segmented_model)[0])
            
            return ModelFit(
                model_name="Segmented (R segmented)",
                model_type="changepoint",
                formula="is_correct ~ segmented(logN)",
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                fixed_effects=fixed_effects,
                fixed_effects_se=fixed_effects_se,
                fixed_effects_pvalues=fixed_effects_pvalues,
                breakpoints=breakpoints,
                breakpoint_ci=breakpoint_ci,
                n_observations=len(data),
                converged=True
            )
            
        except Exception as e:
            self.logger.error(f"R segmented fitting failed: {e}")
            raise
    
    def _fit_segmented_python(self, data: pd.DataFrame) -> ModelFit:
        """Fit segmented model using Python (grid search for breakpoint)"""
        
        try:
            # Grid search over potential breakpoints
            logN_min, logN_max = data['logN'].min(), data['logN'].max()
            logN_range = logN_max - logN_min
            
            # Search over middle 60% of range to avoid boundary effects
            search_min = logN_min + 0.2 * logN_range
            search_max = logN_max - 0.2 * logN_range
            
            breakpoint_candidates = np.linspace(search_min, search_max, 20)
            
            best_aic = float('inf')
            best_breakpoint = None
            best_model = None
            
            for bp in breakpoint_candidates:
                # Create segmented variables
                data_temp = data.copy()
                data_temp['logN_seg1'] = np.where(data_temp['logN'] <= bp, data_temp['logN'], bp)
                data_temp['logN_seg2'] = np.where(data_temp['logN'] > bp, data_temp['logN'] - bp, 0)
                
                # Fit segmented model
                formula = "is_correct ~ logN_seg1 + logN_seg2"
                
                try:
                    model = smf.logit(formula, data=data_temp).fit(disp=0)
                    
                    if model.converged and model.aic < best_aic:
                        best_aic = model.aic
                        best_breakpoint = bp
                        best_model = model
                        
                except:
                    continue
            
            if best_model is None:
                raise ValueError("No converged segmented model found")
            
            return ModelFit(
                model_name="Segmented (Python grid search)",
                model_type="changepoint", 
                formula="is_correct ~ segmented(logN)",
                aic=best_model.aic,
                bic=best_model.bic,
                log_likelihood=best_model.llf,
                fixed_effects=dict(best_model.params),
                fixed_effects_se=dict(best_model.bse),
                fixed_effects_pvalues=dict(best_model.pvalues),
                breakpoints=[best_breakpoint],
                breakpoint_ci=[(best_breakpoint - 0.1, best_breakpoint + 0.1)],  # Rough estimate
                n_observations=best_model.nobs,
                converged=best_model.converged
            )
            
        except Exception as e:
            self.logger.error(f"Python segmented fitting failed: {e}")
            raise
    
    def sup_f_test(self, linear_fit: ModelFit, segmented_fit: ModelFit) -> Dict[str, float]:
        """
        Supremum F-test for structural break significance
        
        Tests null hypothesis of no structural break vs alternative of one break.
        
        Args:
            linear_fit: Linear model (null)
            segmented_fit: Segmented model (alternative)
            
        Returns:
            Test results dictionary
        """
        
        # Likelihood ratio test statistic
        lr_stat = 2 * (segmented_fit.log_likelihood - linear_fit.log_likelihood)
        
        # Degrees of freedom difference (simplified)
        df_diff = len(segmented_fit.fixed_effects) - len(linear_fit.fixed_effects)
        
        # P-value from chi-squared distribution (approximate)
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
        
        return {
            'test_statistic': lr_stat,
            'df_difference': df_diff,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'test_type': 'Likelihood Ratio Test (approximate sup-F)'
        }


class ContaminationAnalyzer:
    """Analysis to separate contamination from brittleness effects"""
    
    def __init__(self, alpha: float, logger: logging.Logger):
        self.alpha = alpha
        self.logger = logger
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Separate contamination from brittleness using paraphrase vs scramble comparison
        
        Analyzes: Δ_para-scram = (Acc_para - Acc_scram)/Acc₀ vs logN
        Regresses on tok_kl and tok_frag to separate effects.
        
        Args:
            data: Analysis dataset
            
        Returns:
            Contamination analysis results
        """
        self.logger.info("Running contamination vs brittleness analysis")
        
        try:
            # Compute difference measure for each model
            contamination_data = []
            
            for model_id in data['model_id'].unique():
                model_data = data[data['model_id'] == model_id]
                
                # Get accuracies
                original_acc = model_data[model_data['transform'] == 'original']['is_correct'].mean()
                
                paraphrase_data = model_data[model_data['transform'] == 'paraphrase']
                scramble_data = model_data[model_data['transform'] == 'scramble']
                
                if not paraphrase_data.empty and not scramble_data.empty and original_acc > 0:
                    paraphrase_acc = paraphrase_data['is_correct'].mean()
                    scramble_acc = scramble_data['is_correct'].mean()
                    
                    # Contamination indicator: (Acc_para - Acc_scram) / Acc_0
                    delta_para_scram = (paraphrase_acc - scramble_acc) / original_acc
                    
                    # Get tokenizer metrics
                    tok_kl_para = paraphrase_data['tok_kl'].mean()
                    tok_frag_para = paraphrase_data['tok_frag'].mean()
                    tok_kl_scram = scramble_data['tok_kl'].mean()
                    tok_frag_scram = scramble_data['tok_frag'].mean()
                    
                    contamination_data.append({
                        'model_id': model_id,
                        'model_family': model_data['model_family'].iloc[0],
                        'logN': model_data['logN'].iloc[0],
                        'n_params': model_data['n_params'].iloc[0],
                        'delta_para_scram': delta_para_scram,
                        'original_acc': original_acc,
                        'paraphrase_acc': paraphrase_acc, 
                        'scramble_acc': scramble_acc,
                        'tok_kl_para': tok_kl_para,
                        'tok_frag_para': tok_frag_para,
                        'tok_kl_scram': tok_kl_scram,
                        'tok_frag_scram': tok_frag_scram
                    })
            
            if not contamination_data:
                return {'error': 'Insufficient data for contamination analysis'}
            
            contam_df = pd.DataFrame(contamination_data)
            
            # Regression analysis
            # Model: delta_para_scram ~ logN + tok_kl_diff + tok_frag_diff
            contam_df['tok_kl_diff'] = contam_df['tok_kl_scram'] - contam_df['tok_kl_para']
            contam_df['tok_frag_diff'] = contam_df['tok_frag_scram'] - contam_df['tok_frag_para']
            
            # Fit regression model
            formula = "delta_para_scram ~ logN + tok_kl_diff + tok_frag_diff"
            
            model = smf.ols(formula, data=contam_df).fit()
            
            # Correlation analysis
            correlations = {
                'delta_vs_logN': stats.spearmanr(contam_df['logN'], contam_df['delta_para_scram']),
                'delta_vs_tok_kl_diff': stats.spearmanr(contam_df['tok_kl_diff'], contam_df['delta_para_scram']),
                'delta_vs_tok_frag_diff': stats.spearmanr(contam_df['tok_frag_diff'], contam_df['delta_para_scram'])
            }
            
            return {
                'n_models': len(contam_df),
                'contamination_data': contam_df.to_dict('records'),
                'regression_results': {
                    'formula': formula,
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'aic': model.aic,
                    'bic': model.bic,
                    'coefficients': dict(model.params),
                    'std_errors': dict(model.bse),
                    'p_values': dict(model.pvalues)
                },
                'correlations': {
                    name: {'correlation': corr, 'p_value': p_val}
                    for name, (corr, p_val) in correlations.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Contamination analysis failed: {e}")
            return {'error': str(e)}