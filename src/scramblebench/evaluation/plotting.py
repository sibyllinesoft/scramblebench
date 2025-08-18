"""
Plotting and visualization for ScrambleBench evaluation results.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

from .results import EvaluationResults
from .metrics import MetricsCalculator, AccuracyMetrics, RobustnessMetrics
from .config import PlotConfig


@dataclass
class PlotResult:
    """Result of a plotting operation."""
    success: bool
    file_paths: List[Path]
    error: Optional[str] = None


class PlotGenerator:
    """
    Generator for evaluation plots and visualizations.
    
    Creates various plot types including performance comparisons,
    degradation analysis, and statistical visualizations.
    """
    
    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the plot generator.
        
        Args:
            config: Plot configuration
            logger: Logger instance
        """
        self.config = config or PlotConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Set matplotlib style
        try:
            plt.style.use(self.config.style)
        except Exception:
            self.logger.warning(f"Could not set style {self.config.style}, using default")
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = self.config.figsize
        plt.rcParams['savefig.dpi'] = self.config.dpi
    
    def generate_all_plots(
        self,
        results: EvaluationResults,
        output_dir: Path,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, PlotResult]:
        """
        Generate all configured plot types.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
            ground_truth: Optional ground truth answers
            
        Returns:
            Dictionary mapping plot types to results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_results = {}
        
        # Performance comparison plots
        if self.config.generate_comparison_plots:
            try:
                result = self.plot_model_comparison(results, output_dir, ground_truth)
                plot_results["model_comparison"] = result
            except Exception as e:
                self.logger.error(f"Error generating model comparison plot: {e}")
                plot_results["model_comparison"] = PlotResult(False, [], str(e))
        
        # Degradation analysis plots
        if self.config.generate_degradation_plots:
            try:
                result = self.plot_robustness_analysis(results, output_dir, ground_truth)
                plot_results["robustness_analysis"] = result
            except Exception as e:
                self.logger.error(f"Error generating robustness analysis plot: {e}")
                plot_results["robustness_analysis"] = PlotResult(False, [], str(e))
        
        # Heatmaps
        if self.config.generate_heatmaps:
            try:
                result = self.plot_performance_heatmap(results, output_dir, ground_truth)
                plot_results["performance_heatmap"] = result
            except Exception as e:
                self.logger.error(f"Error generating performance heatmap: {e}")
                plot_results["performance_heatmap"] = PlotResult(False, [], str(e))
        
        # Distribution plots
        if self.config.generate_distribution_plots:
            try:
                result = self.plot_response_time_distribution(results, output_dir)
                plot_results["response_time_distribution"] = result
            except Exception as e:
                self.logger.error(f"Error generating response time distribution: {e}")
                plot_results["response_time_distribution"] = PlotResult(False, [], str(e))
        
        # Interactive plots
        if self.config.generate_interactive:
            try:
                result = self.plot_interactive_dashboard(results, output_dir, ground_truth)
                plot_results["interactive_dashboard"] = result
            except Exception as e:
                self.logger.error(f"Error generating interactive dashboard: {e}")
                plot_results["interactive_dashboard"] = PlotResult(False, [], str(e))
        
        return plot_results
    
    def plot_model_comparison(
        self,
        results: EvaluationResults,
        output_dir: Path,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> PlotResult:
        """Generate model comparison plots."""
        df = results.to_dataframe()
        models = results.get_model_names()
        
        # Calculate accuracy for each model
        metrics_calc = MetricsCalculator()
        accuracy_metrics = metrics_calc.calculate_accuracy_metrics(results, ground_truth)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Overall accuracy comparison
        model_names = list(accuracy_metrics.keys())
        accuracies = [metrics.exact_match for metrics in accuracy_metrics.values()]
        
        bars = ax1.bar(model_names, accuracies, color=sns.color_palette(self.config.color_palette, len(model_names)))
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        # Plot 2: Success rate by transformation type
        success_rates = []
        transformation_types = [t for t in results.get_transformation_types() if t != 'original']
        
        for model in model_names:
            model_rates = []
            for transform_type in transformation_types:
                model_transform_df = df[(df['model_name'] == model) & (df['transformation_type'] == transform_type)]
                rate = model_transform_df['success'].mean() if not model_transform_df.empty else 0
                model_rates.append(rate)
            success_rates.append(model_rates)
        
        if transformation_types:
            x = np.arange(len(transformation_types))
            width = 0.8 / len(model_names)
            
            for i, (model, rates) in enumerate(zip(model_names, success_rates)):
                offset = (i - len(model_names)/2) * width + width/2
                ax2.bar(x + offset, rates, width, label=model,
                       color=sns.color_palette(self.config.color_palette, len(model_names))[i])
            
            ax2.set_title('Success Rate by Transformation Type')
            ax2.set_ylabel('Success Rate')
            ax2.set_xlabel('Transformation Type')
            ax2.set_xticks(x)
            ax2.set_xticklabels(transformation_types, rotation=45, ha='right')
            ax2.legend()
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plots
        file_paths = []
        for format in self.config.save_formats:
            file_path = output_dir / f"model_comparison.{format}"
            plt.savefig(file_path, dpi=self.config.dpi, bbox_inches='tight')
            file_paths.append(file_path)
        
        plt.close()
        
        return PlotResult(True, file_paths)
    
    def plot_robustness_analysis(
        self,
        results: EvaluationResults,
        output_dir: Path,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> PlotResult:
        """Generate robustness analysis plots."""
        metrics_calc = MetricsCalculator()
        robustness_metrics = metrics_calc.calculate_robustness_metrics(results, ground_truth)
        
        models = list(robustness_metrics.keys())
        if not models:
            return PlotResult(False, [], "No robustness metrics available")
        
        # Get all transformation types
        all_transformations = set()
        for metrics in robustness_metrics.values():
            all_transformations.update(metrics.degradation_scores.keys())
        
        transformation_types = sorted(list(all_transformations))
        
        if not transformation_types:
            return PlotResult(False, [], "No transformation data available")
        
        # Create degradation plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Degradation heatmap
        degradation_matrix = []
        for model in models:
            model_degradations = []
            for transform_type in transformation_types:
                degradation = robustness_metrics[model].degradation_scores.get(transform_type, 0)
                model_degradations.append(degradation)
            degradation_matrix.append(model_degradations)
        
        im = ax1.imshow(degradation_matrix, cmap='Reds', aspect='auto')
        ax1.set_xticks(range(len(transformation_types)))
        ax1.set_xticklabels(transformation_types, rotation=45, ha='right')
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.set_title('Performance Degradation by Model and Transformation')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Degradation (Original - Transformed)')
        
        # Add text annotations
        for i, model in enumerate(models):
            for j, transform_type in enumerate(transformation_types):
                degradation = degradation_matrix[i][j]
                text = ax1.text(j, i, f'{degradation:.3f}', ha='center', va='center',
                               color='white' if degradation > 0.5 else 'black')
        
        # Plot 2: Average degradation by model
        avg_degradations = [robustness_metrics[model].avg_degradation for model in models]
        bars = ax2.bar(models, avg_degradations, color=sns.color_palette(self.config.color_palette, len(models)))
        ax2.set_title('Average Performance Degradation by Model')
        ax2.set_ylabel('Average Degradation')
        ax2.set_xlabel('Model')
        
        # Add value labels
        for bar, degradation in zip(bars, avg_degradations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{degradation:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plots
        file_paths = []
        for format in self.config.save_formats:
            file_path = output_dir / f"robustness_analysis.{format}"
            plt.savefig(file_path, dpi=self.config.dpi, bbox_inches='tight')
            file_paths.append(file_path)
        
        plt.close()
        
        return PlotResult(True, file_paths)
    
    def plot_performance_heatmap(
        self,
        results: EvaluationResults,
        output_dir: Path,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> PlotResult:
        """Generate performance heatmap."""
        df = results.to_dataframe()
        
        # Create pivot table for heatmap
        if ground_truth:
            # Calculate actual accuracy
            accuracy_data = []
            for _, row in df.iterrows():
                problem_id = row['problem_id']
                if problem_id in ground_truth:
                    correct = (self._normalize_answer(row['model_response']) == 
                              self._normalize_answer(ground_truth[problem_id]))
                    accuracy_data.append({
                        'model_name': row['model_name'],
                        'transformation_type': row['transformation_type'],
                        'accuracy': 1.0 if correct else 0.0
                    })
            
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data)
                pivot_table = accuracy_df.groupby(['model_name', 'transformation_type'])['accuracy'].mean().unstack()
            else:
                # Fallback to success rate
                pivot_table = df.groupby(['model_name', 'transformation_type'])['success'].mean().unstack()
        else:
            # Use success rate
            pivot_table = df.groupby(['model_name', 'transformation_type'])['success'].mean().unstack()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Performance Score'})
        plt.title('Model Performance Heatmap')
        plt.ylabel('Model')
        plt.xlabel('Transformation Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plots
        file_paths = []
        for format in self.config.save_formats:
            file_path = output_dir / f"performance_heatmap.{format}"
            plt.savefig(file_path, dpi=self.config.dpi, bbox_inches='tight')
            file_paths.append(file_path)
        
        plt.close()
        
        return PlotResult(True, file_paths)
    
    def plot_response_time_distribution(
        self,
        results: EvaluationResults,
        output_dir: Path
    ) -> PlotResult:
        """Generate response time distribution plots."""
        df = results.to_dataframe()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Response time distribution by model
        models = results.get_model_names()
        for model in models:
            model_df = df[df['model_name'] == model]
            ax1.hist(model_df['response_time'], alpha=0.7, label=model, bins=30)
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Response Time Distribution by Model')
        ax1.legend()
        
        # Plot 2: Response time box plot by model
        response_times_by_model = [df[df['model_name'] == model]['response_time'].values 
                                  for model in models]
        ax2.boxplot(response_times_by_model, labels=models)
        ax2.set_ylabel('Response Time (seconds)')
        ax2.set_title('Response Time Box Plot by Model')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Token usage vs response time
        ax3.scatter(df['total_tokens'], df['response_time'], alpha=0.6)
        ax3.set_xlabel('Total Tokens')
        ax3.set_ylabel('Response Time (seconds)')
        ax3.set_title('Token Usage vs Response Time')
        
        # Plot 4: Success rate vs response time
        successful_df = df[df['success'] == True]
        failed_df = df[df['success'] == False]
        
        ax4.hist(successful_df['response_time'], alpha=0.7, label='Successful', bins=30, color='green')
        ax4.hist(failed_df['response_time'], alpha=0.7, label='Failed', bins=30, color='red')
        ax4.set_xlabel('Response Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Response Time by Success Status')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plots
        file_paths = []
        for format in self.config.save_formats:
            file_path = output_dir / f"response_time_distribution.{format}"
            plt.savefig(file_path, dpi=self.config.dpi, bbox_inches='tight')
            file_paths.append(file_path)
        
        plt.close()
        
        return PlotResult(True, file_paths)
    
    def plot_interactive_dashboard(
        self,
        results: EvaluationResults,
        output_dir: Path,
        ground_truth: Optional[Dict[str, str]] = None
    ) -> PlotResult:
        """Generate interactive Plotly dashboard."""
        df = results.to_dataframe()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Model Performance Comparison', 'Response Time Distribution',
                           'Success Rate by Transformation', 'Token Usage Analysis'],
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Model performance comparison
        models = results.get_model_names()
        success_rates = [df[df['model_name'] == model]['success'].mean() for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=success_rates, name='Success Rate',
                   text=[f'{rate:.3f}' for rate in success_rates],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Plot 2: Response time distribution
        for model in models:
            model_df = df[df['model_name'] == model]
            fig.add_trace(
                go.Histogram(x=model_df['response_time'], name=model, opacity=0.7),
                row=1, col=2
            )
        
        # Plot 3: Success rate by transformation
        transformation_types = results.get_transformation_types()
        for model in models:
            transform_success_rates = []
            for transform_type in transformation_types:
                transform_df = df[(df['model_name'] == model) & (df['transformation_type'] == transform_type)]
                rate = transform_df['success'].mean() if not transform_df.empty else 0
                transform_success_rates.append(rate)
            
            fig.add_trace(
                go.Bar(x=transformation_types, y=transform_success_rates, name=model),
                row=2, col=1
            )
        
        # Plot 4: Token usage analysis
        fig.add_trace(
            go.Scatter(x=df['total_tokens'], y=df['response_time'],
                      mode='markers', name='Token Usage vs Time',
                      text=df['model_name'], hovertemplate='Model: %{text}<br>Tokens: %{x}<br>Time: %{y}s'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="ScrambleBench Evaluation Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save interactive plot
        file_paths = []
        html_path = output_dir / "interactive_dashboard.html"
        fig.write_html(html_path)
        file_paths.append(html_path)
        
        return PlotResult(True, file_paths)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        import re
        if not isinstance(answer, str):
            return str(answer).lower().strip()
        
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', answer.lower().strip())
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized