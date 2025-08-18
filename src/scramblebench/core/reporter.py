"""
Results reporting and visualization for ScrambleBench.

This module provides comprehensive reporting capabilities including
statistical summaries, visualizations, and export functionality
for benchmark results.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
import logging
from datetime import datetime
import time

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from scramblebench.core.benchmark import BenchmarkResult


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: Path = Path("data/reports")
    include_charts: bool = True
    include_raw_data: bool = True
    format: str = "html"  # html, json, csv, markdown
    
    
class Reporter:
    """
    Comprehensive reporting system for benchmark results.
    
    Generates statistical summaries, visualizations, and exports
    in multiple formats for analysis and presentation.
    """
    
    def __init__(
        self, 
        config: Optional[ReportConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the reporter.
        
        Args:
            config: Report configuration (creates default if None)
            logger: Logger instance (creates default if None)
        """
        self.config = config or ReportConfig()
        self.logger = logger or logging.getLogger("scramblebench.reporter")
        self.console = Console()
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self, 
        results: List[BenchmarkResult],
        title: str = "ScrambleBench Results",
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report from benchmark results.
        
        Args:
            results: List of benchmark results to report on
            title: Title for the report
            save: Whether to save the report to disk
            
        Returns:
            Dictionary containing the generated report data
        """
        if not results:
            self.logger.warning("No results provided for report generation")
            return {}
        
        self.logger.info(f"Generating report for {len(results)} results")
        
        # Create report data structure
        report_data = {
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_summary(results),
            'detailed_results': [asdict(result) for result in results],
            'statistics': self._compute_statistics(results),
            'comparisons': self._generate_comparisons(results),
            'metadata': {
                'total_results': len(results),
                'benchmarks': list(set(r.benchmark_name for r in results)),
                'models': list(set(r.model_name for r in results)),
            }
        }
        
        # Generate formatted outputs
        if save:
            self._save_report(report_data, title)
        
        # Display console summary
        self._display_console_summary(results, title)
        
        return report_data
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate high-level summary statistics."""
        scores = [r.score for r in results]
        durations = [r.duration for r in results]
        
        summary = {
            'overall_score': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
            },
            'performance': {
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations),
                'fastest': np.min(durations),
                'slowest': np.max(durations),
            },
            'benchmarks': self._summarize_by_benchmark(results),
            'models': self._summarize_by_model(results),
        }
        
        return summary
    
    def _summarize_by_benchmark(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize results grouped by benchmark."""
        benchmark_groups = {}
        
        for result in results:
            benchmark = result.benchmark_name
            if benchmark not in benchmark_groups:
                benchmark_groups[benchmark] = []
            benchmark_groups[benchmark].append(result)
        
        summary = {}
        for benchmark, bench_results in benchmark_groups.items():
            scores = [r.score for r in bench_results]
            summary[benchmark] = {
                'count': len(bench_results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores),
                'models_tested': list(set(r.model_name for r in bench_results)),
            }
        
        return summary
    
    def _summarize_by_model(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize results grouped by model."""
        model_groups = {}
        
        for result in results:
            model = result.model_name
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        summary = {}
        for model, model_results in model_groups.items():
            scores = [r.score for r in model_results]
            summary[model] = {
                'count': len(model_results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores),
                'benchmarks_tested': list(set(r.benchmark_name for r in model_results)),
            }
        
        return summary
    
    def _compute_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute detailed statistical analysis."""
        df = pd.DataFrame([asdict(r) for r in results])
        
        statistics = {
            'score_distribution': {
                'percentiles': {
                    '25th': np.percentile(df['score'], 25),
                    '50th': np.percentile(df['score'], 50),
                    '75th': np.percentile(df['score'], 75),
                    '90th': np.percentile(df['score'], 90),
                    '95th': np.percentile(df['score'], 95),
                }
            },
            'correlations': self._compute_correlations(df),
            'outliers': self._identify_outliers(df),
        }
        
        return statistics
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Extract interesting correlations
        correlations = {}
        if 'score' in corr_matrix.columns and 'duration' in corr_matrix.columns:
            correlations['score_duration'] = corr_matrix.loc['score', 'duration']
        
        return correlations
    
    def _identify_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Identify outlier results using IQR method."""
        outliers = {}
        
        for col in ['score', 'duration']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ].index.tolist()
                
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    def _generate_comparisons(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comparative analysis between models and benchmarks."""
        comparisons = {}
        
        # Model comparison
        models = list(set(r.model_name for r in results))
        if len(models) > 1:
            model_comparison = {}
            for model in models:
                model_results = [r for r in results if r.model_name == model]
                model_scores = [r.score for r in model_results]
                model_comparison[model] = {
                    'mean_score': np.mean(model_scores),
                    'count': len(model_results),
                }
            
            # Rank models by performance
            ranked_models = sorted(
                model_comparison.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )
            
            comparisons['model_ranking'] = [
                {'model': model, **stats} for model, stats in ranked_models
            ]
        
        # Benchmark comparison
        benchmarks = list(set(r.benchmark_name for r in results))
        if len(benchmarks) > 1:
            benchmark_comparison = {}
            for benchmark in benchmarks:
                bench_results = [r for r in results if r.benchmark_name == benchmark]
                bench_scores = [r.score for r in bench_results]
                benchmark_comparison[benchmark] = {
                    'mean_score': np.mean(bench_scores),
                    'count': len(bench_results),
                    'difficulty': 1.0 - np.mean(bench_scores),  # Inverse of mean score
                }
            
            comparisons['benchmark_difficulty'] = sorted(
                benchmark_comparison.items(),
                key=lambda x: x[1]['difficulty'],
                reverse=True
            )
        
        return comparisons
    
    def _save_report(self, report_data: Dict[str, Any], title: str) -> None:
        """Save report to disk in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{title.lower().replace(' ', '_')}_{timestamp}"
        
        if self.config.format in ['json', 'all']:
            json_path = self.config.output_dir / f"{filename_base}.json"
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"JSON report saved: {json_path}")
        
        if self.config.format in ['csv', 'all']:
            csv_path = self.config.output_dir / f"{filename_base}.csv"
            df = pd.DataFrame(report_data['detailed_results'])
            df.to_csv(csv_path, index=False)
            self.logger.info(f"CSV report saved: {csv_path}")
        
        if self.config.format in ['html', 'all']:
            html_path = self.config.output_dir / f"{filename_base}.html"
            self._generate_html_report(report_data, html_path)
            self.logger.info(f"HTML report saved: {html_path}")
    
    def _generate_html_report(self, report_data: Dict[str, Any], path: Path) -> None:
        """Generate HTML report (placeholder implementation)."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{report_data['title']}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">Overall Mean Score: {report_data['summary']['overall_score']['mean']:.4f}</div>
                <div class="metric">Total Duration: {report_data['summary']['performance']['total_duration']:.2f}s</div>
                <div class="metric">Results Count: {report_data['metadata']['total_results']}</div>
            </div>
            <h2>Detailed Results</h2>
            <p>See JSON export for detailed results and statistics.</p>
        </body>
        </html>
        """
        
        with open(path, 'w') as f:
            f.write(html_content)
    
    def _display_console_summary(
        self, 
        results: List[BenchmarkResult], 
        title: str
    ) -> None:
        """Display summary in console using Rich formatting."""
        # Create title panel
        title_panel = Panel(
            Text(title, style="bold blue"),
            title="ScrambleBench Report",
            style="blue"
        )
        self.console.print(title_panel)
        
        # Summary statistics table
        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        scores = [r.score for r in results]
        durations = [r.duration for r in results]
        
        summary_table.add_row("Total Results", str(len(results)))
        summary_table.add_row("Mean Score", f"{np.mean(scores):.4f}")
        summary_table.add_row("Median Score", f"{np.median(scores):.4f}")
        summary_table.add_row("Score Std Dev", f"{np.std(scores):.4f}")
        summary_table.add_row("Best Score", f"{np.max(scores):.4f}")
        summary_table.add_row("Worst Score", f"{np.min(scores):.4f}")
        summary_table.add_row("Total Duration", f"{sum(durations):.2f}s")
        summary_table.add_row("Avg Duration", f"{np.mean(durations):.2f}s")
        
        self.console.print(summary_table)
        
        # Model performance table if multiple models
        models = list(set(r.model_name for r in results))
        if len(models) > 1:
            model_table = Table(title="Model Performance")
            model_table.add_column("Model", style="cyan")
            model_table.add_column("Mean Score", style="green")
            model_table.add_column("Results Count", style="yellow")
            
            for model in models:
                model_results = [r for r in results if r.model_name == model]
                mean_score = np.mean([r.score for r in model_results])
                model_table.add_row(model, f"{mean_score:.4f}", str(len(model_results)))
            
            self.console.print(model_table)
    
    def compare_models(
        self, 
        results: List[BenchmarkResult],
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed comparison between specific models.
        
        Args:
            results: List of benchmark results
            models: Specific models to compare (None for all)
            
        Returns:
            Detailed comparison data
        """
        if models is None:
            models = list(set(r.model_name for r in results))
        
        comparison = {}
        
        for model in models:
            model_results = [r for r in results if r.model_name == model]
            if not model_results:
                continue
            
            scores = [r.score for r in model_results]
            durations = [r.duration for r in model_results]
            
            comparison[model] = {
                'count': len(model_results),
                'score_stats': {
                    'mean': np.mean(scores),
                    'median': np.median(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                },
                'duration_stats': {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'total': sum(durations),
                },
                'benchmarks': list(set(r.benchmark_name for r in model_results)),
            }
        
        return comparison