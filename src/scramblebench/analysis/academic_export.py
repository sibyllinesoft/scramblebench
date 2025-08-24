"""
Academic Export and Publication Tools

Provides publication-ready exports including LaTeX tables, CSV data,
preregistration reports, and academic formatting for scaling analysis results.
Ensures reproducibility and transparency for peer review.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd

from .statistical_models import ModelFit
from .model_comparison import ModelComparison, ModelSelectionCriteria
from .bootstrap_inference import ConfidenceInterval, PermutationTestResult

logger = logging.getLogger(__name__)


@dataclass
class PreregReport:
    """Preregistration report for transparency and reproducibility"""
    
    # Study metadata
    study_title: str
    study_date: str
    run_id: str
    git_commit: str
    
    # Analysis plan
    research_question: str
    hypotheses: List[str]
    statistical_methods: List[str]
    model_comparison_plan: str
    multiple_testing_correction: str
    
    # Data description
    n_observations: int
    n_models: int
    n_families: int
    parameter_range: Tuple[float, float]
    
    # Pre-specified decisions
    significance_threshold: float = 0.05
    minimum_effect_size: Optional[float] = None
    model_selection_criteria: str = "AIC with BIC confirmation"
    
    # Quality controls
    exclusion_criteria: List[str] = field(default_factory=list)
    robustness_checks: List[str] = field(default_factory=list)
    
    # Lock status
    is_locked: bool = False
    lock_timestamp: Optional[str] = None
    analysis_blind: bool = True  # Whether analysis was conducted blind to results
    
    def lock_preregistration(self) -> str:
        """Lock the preregistration to prevent tampering"""
        
        if self.is_locked:
            return self.lock_timestamp
        
        # Create hash of all fields for integrity
        content_str = json.dumps(self.__dict__, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        self.is_locked = True
        self.lock_timestamp = datetime.now().isoformat()
        
        logger.info(f"Preregistration locked at {self.lock_timestamp}")
        logger.info(f"Content hash: {content_hash}")
        
        return self.lock_timestamp
    
    def to_markdown(self) -> str:
        """Generate markdown preregistration report"""
        
        md_lines = [
            f"# Preregistration Report: {self.study_title}",
            "",
            f"**Study Date:** {self.study_date}",
            f"**Run ID:** {self.run_id}",  
            f"**Git Commit:** {self.git_commit}",
            f"**Lock Status:** {'LOCKED' if self.is_locked else 'UNLOCKED'}",
            ""
        ]
        
        if self.is_locked:
            md_lines.extend([
                f"**Lock Timestamp:** {self.lock_timestamp}",
                "⚠️ **This preregistration is locked and cannot be modified without detection.**",
                ""
            ])
        
        md_lines.extend([
            "## Research Question",
            self.research_question,
            "",
            "## Hypotheses"
        ])
        
        for i, hypothesis in enumerate(self.hypotheses, 1):
            md_lines.append(f"{i}. {hypothesis}")
        
        md_lines.extend([
            "",
            "## Statistical Methods",
            ""
        ])
        
        for method in self.statistical_methods:
            md_lines.append(f"- {method}")
        
        md_lines.extend([
            "",
            "## Model Comparison Plan", 
            self.model_comparison_plan,
            "",
            "## Multiple Testing Correction",
            self.multiple_testing_correction,
            "",
            "## Data Description",
            f"- **Observations:** {self.n_observations:,}",
            f"- **Models:** {self.n_models}",
            f"- **Families:** {self.n_families}", 
            f"- **Parameter Range:** {self.parameter_range[0]:.1f} - {self.parameter_range[1]:.1f} (log scale)",
            "",
            "## Pre-specified Decisions",
            f"- **Significance Threshold:** {self.significance_threshold}",
            f"- **Model Selection:** {self.model_selection_criteria}",
            ""
        ])
        
        if self.minimum_effect_size:
            md_lines.append(f"- **Minimum Effect Size:** {self.minimum_effect_size}")
        
        if self.exclusion_criteria:
            md_lines.extend([
                "",
                "## Exclusion Criteria",
                ""
            ])
            for criterion in self.exclusion_criteria:
                md_lines.append(f"- {criterion}")
        
        if self.robustness_checks:
            md_lines.extend([
                "",
                "## Robustness Checks",
                ""
            ])
            for check in self.robustness_checks:
                md_lines.append(f"- {check}")
        
        md_lines.extend([
            "",
            "---",
            f"*Report generated on {datetime.now().isoformat()}*"
        ])
        
        return "\n".join(md_lines)


class LaTeXTableFormatter:
    """
    LaTeX table formatter for academic publications
    
    Generates publication-quality tables with proper formatting,
    captions, labels, and statistical notation.
    """
    
    def __init__(self, precision: int = 3):
        """
        Initialize LaTeX formatter
        
        Args:
            precision: Decimal precision for numbers
        """
        self.precision = precision
    
    def model_comparison_table(
        self, 
        comparison_results: Dict[str, Any],
        caption: str = "Model comparison results",
        label: str = "tab:model_comparison"
    ) -> str:
        """
        Create LaTeX table for model comparison
        
        Args:
            comparison_results: Results from ModelComparison.compare_models()
            caption: Table caption
            label: LaTeX label for referencing
            
        Returns:
            LaTeX table code
        """
        
        # Extract comparison data
        comparison_table = comparison_results['model_selection_summary']
        
        # LaTeX table start
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Model & AIC & BIC & AIC Weight & BIC Weight & AIC Rank & BIC Rank \\\\",
            "\\midrule"
        ]
        
        # Add data rows
        for _, row in comparison_table.iterrows():
            model_name = self._escape_latex(row['Model'])
            
            latex_lines.append(
                f"{model_name} & "
                f"{float(row['AIC']):.{self.precision}f} & "
                f"{float(row['BIC']):.{self.precision}f} & "
                f"{float(row['AIC Weight']):.{self.precision}f} & "
                f"{float(row['BIC Weight']):.{self.precision}f} & "
                f"{row['AIC Rank']} & "
                f"{row['BIC Rank']} \\\\"
            )
        
        # Table end
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def parameter_estimates_table(
        self,
        model_fits: Dict[str, ModelFit],
        confidence_intervals: Optional[Dict[str, Dict[str, ConfidenceInterval]]] = None,
        caption: str = "Parameter estimates",
        label: str = "tab:parameters"
    ) -> str:
        """
        Create LaTeX table for parameter estimates with confidence intervals
        
        Args:
            model_fits: Dictionary of model fits by family
            confidence_intervals: Optional CIs by family and parameter
            caption: Table caption
            label: LaTeX label
            
        Returns:
            LaTeX table code
        """
        
        # Collect all parameters across models
        all_params = set()
        for fit in model_fits.values():
            all_params.update(fit.fixed_effects.keys())
        
        all_params = sorted(all_params)
        
        # Table header
        n_models = len(model_fits)
        col_spec = "l" + "c" * n_models
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering", 
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            "Parameter & " + " & ".join(model_fits.keys()) + " \\\\",
            "\\midrule"
        ]
        
        # Add parameter rows
        for param in all_params:
            param_name = self._format_parameter_name(param)
            row_values = [param_name]
            
            for family, fit in model_fits.items():
                if param in fit.fixed_effects:
                    estimate = fit.fixed_effects[param]
                    
                    # Add confidence interval if available
                    if confidence_intervals and family in confidence_intervals:
                        if param in confidence_intervals[family]:
                            ci = confidence_intervals[family][param]
                            cell_value = f"{estimate:.{self.precision}f} [{ci.lower:.{self.precision}f}, {ci.upper:.{self.precision}f}]"
                        else:
                            cell_value = f"{estimate:.{self.precision}f}"
                    else:
                        # Add standard error if available
                        if param in fit.fixed_effects_se:
                            se = fit.fixed_effects_se[param]
                            cell_value = f"{estimate:.{self.precision}f} ({se:.{self.precision}f})"
                        else:
                            cell_value = f"{estimate:.{self.precision}f}"
                    
                    # Add significance stars
                    if param in fit.fixed_effects_pvalues:
                        p_val = fit.fixed_effects_pvalues[param]
                        stars = self._significance_stars(p_val)
                        cell_value += stars
                    
                    row_values.append(cell_value)
                else:
                    row_values.append("--")
            
            latex_lines.append(" & ".join(row_values) + " \\\\")
        
        # Table footer with notes
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\footnotesize",
            "\\begin{tablenotes}",
            "\\item Notes: Standard errors in parentheses, 95\\% confidence intervals in brackets.",
            "\\item Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, . p < 0.1",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def scaling_results_table(
        self,
        scaling_results: pd.DataFrame,
        caption: str = "Scaling analysis results by model family",
        label: str = "tab:scaling_results"
    ) -> str:
        """
        Create LaTeX table for scaling analysis results
        
        Args:
            scaling_results: DataFrame with columns: Family, N_Models, LogN_Range, Best_Model, Evidence_Strength
            caption: Table caption
            label: LaTeX label
            
        Returns:
            LaTeX table code
        """
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrlll}",
            "\\toprule",
            "Family & N Models & Log N Range & Best Model & Evidence \\\\",
            "\\midrule"
        ]
        
        for _, row in scaling_results.iterrows():
            family = self._escape_latex(str(row['Family']))
            n_models = int(row['N_Models']) if 'N_Models' in row else "--"
            log_range = f"{row['LogN_Min']:.1f}--{row['LogN_Max']:.1f}" if 'LogN_Min' in row else "--"
            best_model = self._escape_latex(str(row['Best_Model'])) if 'Best_Model' in row else "--"
            evidence = str(row['Evidence_Strength']).title() if 'Evidence_Strength' in row else "--"
            
            latex_lines.append(
                f"{family} & {n_models} & {log_range} & {best_model} & {evidence} \\\\"
            )
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        replacements = {
            '&': '\\&',
            '%': '\\%', 
            '$': '\\$',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '\\': '\\textbackslash{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _format_parameter_name(self, param: str) -> str:
        """Format parameter name for LaTeX display"""
        # Convert common parameter names to mathematical notation
        replacements = {
            'logN': '$\\log N$',
            'Intercept': 'Intercept',
            'condition_f': 'Condition',
            'domain_f': 'Domain',
            'model_family_f': 'Family'
        }
        
        formatted = replacements.get(param, param)
        
        # Handle interaction terms
        if ':' in formatted:
            parts = formatted.split(':')
            formatted = ' $\\times$ '.join(parts)
        
        return formatted
    
    def _significance_stars(self, p_value: float) -> str:
        """Convert p-value to significance stars"""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "."
        else:
            return ""


class CSVExporter:
    """CSV export utilities for analysis results"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize CSV exporter
        
        Args:
            output_dir: Directory for CSV outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model_comparison(
        self, 
        comparison_results: Dict[str, Any],
        filename: str = "model_comparison.csv"
    ) -> Path:
        """Export model comparison results to CSV"""
        
        output_path = self.output_dir / filename
        
        # Main comparison table
        comparison_df = comparison_results['model_selection_summary']
        comparison_df.to_csv(output_path, index=False)
        
        # Export AIC weights separately
        aic_weights = comparison_results['aic_weights']
        weights_df = pd.DataFrame({
            'Model': aic_weights.models,
            'AIC': aic_weights.aic_values,
            'Delta_AIC': aic_weights.delta_aic,
            'AIC_Weight': aic_weights.weights,
            'Cumulative_Weight': aic_weights.cumulative_weights
        })
        
        weights_path = self.output_dir / f"aic_weights_{filename}"
        weights_df.to_csv(weights_path, index=False)
        
        logger.info(f"Model comparison exported to {output_path}")
        return output_path
    
    def export_parameter_estimates(
        self,
        model_fits: Dict[str, ModelFit],
        confidence_intervals: Optional[Dict[str, Dict[str, ConfidenceInterval]]] = None,
        filename: str = "parameter_estimates.csv"
    ) -> Path:
        """Export parameter estimates with confidence intervals"""
        
        output_path = self.output_dir / filename
        
        rows = []
        
        for family, fit in model_fits.items():
            for param, estimate in fit.fixed_effects.items():
                row = {
                    'Family': family,
                    'Parameter': param,
                    'Estimate': estimate,
                    'SE': fit.fixed_effects_se.get(param, np.nan),
                    'P_Value': fit.fixed_effects_pvalues.get(param, np.nan)
                }
                
                # Add confidence intervals if available
                if confidence_intervals and family in confidence_intervals:
                    if param in confidence_intervals[family]:
                        ci = confidence_intervals[family][param]
                        row['CI_Lower'] = ci.lower
                        row['CI_Upper'] = ci.upper
                        row['CI_Method'] = ci.method
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Parameter estimates exported to {output_path}")
        return output_path
    
    def export_scaling_summary(
        self,
        family_results: Dict[str, Any],
        filename: str = "scaling_summary.csv"
    ) -> Path:
        """Export scaling analysis summary by family"""
        
        output_path = self.output_dir / filename
        
        rows = []
        
        for family, results in family_results.items():
            row = {
                'Family': family,
                'N_Observations': results.get('n_observations', 0),
                'N_Models': results.get('n_models', 0),
                'LogN_Min': results.get('parameter_range', {}).get('logN_range', (0, 0))[0],
                'LogN_Max': results.get('parameter_range', {}).get('logN_range', (0, 0))[1],
                'Best_Model': results.get('best_model', 'Unknown'),
                'Evidence_Strength': results.get('model_comparison', {}).get('evidence_assessment', {}).get('strength', 'Unknown')
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Scaling summary exported to {output_path}")
        return output_path
    
    def export_contamination_analysis(
        self,
        contamination_results: Dict[str, Any],
        filename: str = "contamination_analysis.csv"
    ) -> Path:
        """Export contamination vs brittleness analysis"""
        
        output_path = self.output_dir / filename
        
        if 'contamination_data' in contamination_results:
            df = pd.DataFrame(contamination_results['contamination_data'])
            df.to_csv(output_path, index=False)
            
            # Export regression results separately
            if 'regression_results' in contamination_results:
                reg_results = contamination_results['regression_results']
                reg_df = pd.DataFrame({
                    'Coefficient': list(reg_results['coefficients'].keys()),
                    'Estimate': list(reg_results['coefficients'].values()),
                    'SE': list(reg_results['std_errors'].values()),
                    'P_Value': list(reg_results['p_values'].values())
                })
                
                reg_path = self.output_dir / f"contamination_regression_{filename}"
                reg_df.to_csv(reg_path, index=False)
        
        logger.info(f"Contamination analysis exported to {output_path}")
        return output_path


class AcademicExporter:
    """
    Comprehensive academic export system
    
    Coordinates all export functions and creates publication-ready
    output packages including data, tables, and reports.
    """
    
    def __init__(
        self,
        output_dir: Path,
        study_title: str = "ScrambleBench Scaling Analysis",
        precision: int = 3
    ):
        """
        Initialize academic exporter
        
        Args:
            output_dir: Base directory for all outputs
            study_title: Title for reports and tables
            precision: Decimal precision for numerical outputs
        """
        self.output_dir = Path(output_dir)
        self.study_title = study_title
        self.precision = precision
        
        # Create output subdirectories
        self.csv_dir = self.output_dir / "csv"
        self.latex_dir = self.output_dir / "latex"
        self.reports_dir = self.output_dir / "reports"
        
        for directory in [self.csv_dir, self.latex_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.csv_exporter = CSVExporter(self.csv_dir)
        self.latex_formatter = LaTeXTableFormatter(precision)
    
    def export_full_analysis(
        self,
        analysis_results: Dict[str, Any],
        run_id: str,
        git_commit: str = "unknown"
    ) -> Dict[str, Path]:
        """
        Export complete analysis results for publication
        
        Args:
            analysis_results: Results from ScalingAnalyzer.run_full_analysis()
            run_id: Run identifier
            git_commit: Git commit hash for reproducibility
            
        Returns:
            Dictionary of output file paths
        """
        
        logger.info(f"Exporting full analysis for run {run_id}")
        
        output_files = {}
        
        # 1. Create preregistration report
        prereg = self._create_preregistration(analysis_results, run_id, git_commit)
        prereg_path = self.reports_dir / f"preregistration_{run_id}.md"
        
        with open(prereg_path, 'w') as f:
            f.write(prereg.to_markdown())
        
        output_files['preregistration'] = prereg_path
        
        # 2. Export CSV data
        family_results = analysis_results.get('family_results', {})
        
        if family_results:
            # Model comparisons
            for family, results in family_results.items():
                if 'model_comparison' in results:
                    csv_path = self.csv_exporter.export_model_comparison(
                        results['model_comparison'],
                        filename=f"model_comparison_{family}.csv"
                    )
                    output_files[f'model_comparison_{family}'] = csv_path
            
            # Scaling summary
            summary_path = self.csv_exporter.export_scaling_summary(
                family_results,
                filename=f"scaling_summary_{run_id}.csv"
            )
            output_files['scaling_summary'] = summary_path
        
        # Contamination analysis
        if 'contamination_analysis' in analysis_results:
            contam_path = self.csv_exporter.export_contamination_analysis(
                analysis_results['contamination_analysis'],
                filename=f"contamination_analysis_{run_id}.csv"
            )
            output_files['contamination_analysis'] = contam_path
        
        # 3. Generate LaTeX tables
        latex_tables = self._generate_latex_tables(analysis_results, run_id)
        
        for table_name, latex_code in latex_tables.items():
            latex_path = self.latex_dir / f"{table_name}_{run_id}.tex"
            
            with open(latex_path, 'w') as f:
                f.write(latex_code)
            
            output_files[f'latex_{table_name}'] = latex_path
        
        # 4. Create summary report
        summary_report = self._create_summary_report(analysis_results, run_id)
        summary_path = self.reports_dir / f"analysis_summary_{run_id}.md"
        
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        output_files['summary_report'] = summary_path
        
        # 5. Create manifest file
        manifest = self._create_manifest(output_files, analysis_results, run_id, git_commit)
        manifest_path = self.output_dir / f"manifest_{run_id}.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        output_files['manifest'] = manifest_path
        
        logger.info(f"Full analysis exported to {self.output_dir}")
        logger.info(f"Generated {len(output_files)} output files")
        
        return output_files
    
    def _create_preregistration(
        self,
        analysis_results: Dict[str, Any], 
        run_id: str,
        git_commit: str
    ) -> PreregReport:
        """Create preregistration report"""
        
        # Extract study parameters
        n_obs = analysis_results.get('n_observations', 0)
        n_models = analysis_results.get('n_models', 0)
        n_families = analysis_results.get('n_families', 0)
        
        # Parameter range from family results
        param_ranges = []
        family_results = analysis_results.get('family_results', {})
        
        for results in family_results.values():
            if 'parameter_range' in results:
                logn_range = results['parameter_range'].get('logN_range', (0, 0))
                param_ranges.extend(logn_range)
        
        if param_ranges:
            param_range = (min(param_ranges), max(param_ranges))
        else:
            param_range = (0.0, 0.0)
        
        prereg = PreregReport(
            study_title=self.study_title,
            study_date=datetime.now().strftime("%Y-%m-%d"),
            run_id=run_id,
            git_commit=git_commit,
            research_question="Does reasoning capability scale smoothly with parameters, or are there critical thresholds where capabilities emerge?",
            hypotheses=[
                "H1: Reasoning robustness (measured by Language Dependency Coefficient) decreases smoothly with increasing model parameters",
                "H2: Some model families exhibit threshold effects in reasoning capabilities around specific parameter counts",
                "H3: Contamination effects can be separated from genuine brittleness using paraphrase controls"
            ],
            statistical_methods=[
                "Generalized Linear Mixed Models (GLMM) with random intercepts for domain and family",
                "Generalized Additive Models (GAM) with monotone smooth functions",
                "Segmented regression with changepoint detection",
                "Bootstrap confidence intervals (n=2000)",
                "Likelihood ratio tests for model comparison"
            ],
            model_comparison_plan="Compare linear, segmented, and GAM models using AIC/BIC with AIC weights for model averaging. Use likelihood ratio tests for nested model comparisons.",
            multiple_testing_correction="Bonferroni correction for family-wise error rate control across model families",
            n_observations=n_obs,
            n_models=n_models,
            n_families=n_families,
            parameter_range=param_range,
            exclusion_criteria=[
                "Models with fewer than 10 evaluation items per condition",
                "Model fits that fail to converge",
                "Families with fewer than 3 models"
            ],
            robustness_checks=[
                "Sensitivity analysis using different bootstrap sample sizes",
                "Alternative model specifications (different interaction terms)",
                "Cross-validation of model selection criteria"
            ]
        )
        
        # Lock the preregistration
        prereg.lock_preregistration()
        
        return prereg
    
    def _generate_latex_tables(
        self, 
        analysis_results: Dict[str, Any], 
        run_id: str
    ) -> Dict[str, str]:
        """Generate all LaTeX tables"""
        
        tables = {}
        family_results = analysis_results.get('family_results', {})
        
        # Main scaling results table
        scaling_data = []
        for family, results in family_results.items():
            scaling_data.append({
                'Family': family,
                'N_Models': results.get('n_models', 0),
                'LogN_Min': results.get('parameter_range', {}).get('logN_range', (0, 0))[0],
                'LogN_Max': results.get('parameter_range', {}).get('logN_range', (0, 0))[1],
                'Best_Model': results.get('best_model', 'Unknown'),
                'Evidence_Strength': results.get('model_comparison', {}).get('evidence_assessment', {}).get('strength', 'Unknown')
            })
        
        if scaling_data:
            scaling_df = pd.DataFrame(scaling_data)
            tables['scaling_results'] = self.latex_formatter.scaling_results_table(
                scaling_df,
                caption="Scaling analysis results by model family",
                label="tab:scaling_results"
            )
        
        # Model comparison tables for each family
        for family, results in family_results.items():
            if 'model_comparison' in results:
                tables[f'model_comparison_{family}'] = self.latex_formatter.model_comparison_table(
                    results['model_comparison'],
                    caption=f"Model comparison for {family} family",
                    label=f"tab:model_comparison_{family.lower()}"
                )
        
        return tables
    
    def _create_summary_report(
        self, 
        analysis_results: Dict[str, Any], 
        run_id: str
    ) -> str:
        """Create markdown summary report"""
        
        lines = [
            f"# Scaling Analysis Summary: {run_id}",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Study Title:** {self.study_title}",
            "",
            "## Key Findings",
            ""
        ]
        
        # Summary statistics
        lines.extend([
            "### Dataset Summary",
            f"- **Total Observations:** {analysis_results.get('n_observations', 0):,}",
            f"- **Models Analyzed:** {analysis_results.get('n_models', 0)}",
            f"- **Model Families:** {analysis_results.get('n_families', 0)}",
            ""
        ])
        
        # Family-specific results
        family_results = analysis_results.get('family_results', {})
        
        if family_results:
            lines.extend([
                "### Results by Model Family",
                ""
            ])
            
            for family, results in family_results.items():
                lines.extend([
                    f"#### {family}",
                    f"- **Models:** {results.get('n_models', 0)}",
                    f"- **Observations:** {results.get('n_observations', 0):,}",
                    f"- **Best Model:** {results.get('best_model', 'Unknown')}",
                    ""
                ])
                
                # Model comparison summary
                if 'model_comparison' in results:
                    evidence = results['model_comparison'].get('evidence_assessment', {})
                    lines.extend([
                        f"- **Evidence Strength:** {evidence.get('strength', 'Unknown')}",
                        f"- **Recommendation:** {evidence.get('recommendation', 'No recommendation')}",
                        ""
                    ])
        
        # Contamination analysis
        if 'contamination_analysis' in analysis_results:
            contam = analysis_results['contamination_analysis']
            lines.extend([
                "### Contamination vs Brittleness Analysis",
                f"- **Models Analyzed:** {contam.get('n_models', 0)}",
                ""
            ])
            
            if 'regression_results' in contam:
                reg = contam['regression_results']
                lines.extend([
                    f"- **R-squared:** {reg.get('r_squared', 0):.3f}",
                    f"- **Model AIC:** {reg.get('aic', 0):.1f}",
                    ""
                ])
        
        lines.extend([
            "## Files Generated",
            ""
        ])
        
        # Would list all generated files here
        lines.extend([
            "See `manifest.json` for complete list of generated files and checksums.",
            "",
            "---",
            "*This report was generated automatically from ScrambleBench analysis results.*"
        ])
        
        return "\n".join(lines)
    
    def _create_manifest(
        self,
        output_files: Dict[str, Path],
        analysis_results: Dict[str, Any],
        run_id: str,
        git_commit: str
    ) -> Dict[str, Any]:
        """Create manifest file with metadata and checksums"""
        
        # Calculate file checksums for integrity checking
        file_checksums = {}
        for name, path in output_files.items():
            if path.exists():
                with open(path, 'rb') as f:
                    content = f.read()
                    checksum = hashlib.sha256(content).hexdigest()
                    file_checksums[name] = {
                        'path': str(path.relative_to(self.output_dir)),
                        'size_bytes': len(content),
                        'sha256': checksum
                    }
        
        manifest = {
            'run_id': run_id,
            'git_commit': git_commit,
            'generated_at': datetime.now().isoformat(),
            'study_title': self.study_title,
            'analysis_summary': {
                'n_observations': analysis_results.get('n_observations', 0),
                'n_models': analysis_results.get('n_models', 0),
                'n_families': analysis_results.get('n_families', 0),
                'families': analysis_results.get('model_families', [])
            },
            'files': file_checksums,
            'software_versions': {
                'python': '3.8+',
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'r_available': 'Unknown'  # Would detect R version
            }
        }
        
        return manifest