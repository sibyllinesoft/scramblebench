"""Tests for publication visualizer functionality in the analysis module."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os
from pathlib import Path

# Note: Import paths will need to be updated when the analysis module is implemented
from scramblebench.analysis.publication_visualizer import (
    PublicationVisualizer,
    ScalingPlots,
    ComparisonPlots,
    ConfidenceIntervalPlots,
    ThemeManager
)


class TestPublicationVisualizer:
    """Test publication-ready visualization functionality."""
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Fixture providing sample benchmark data for visualization."""
        return pd.DataFrame({
            'model': ['GPT-4', 'Claude-3', 'Gemini-Pro', 'Llama-2-70B'],
            'score': [0.85, 0.82, 0.78, 0.75],
            'ci_lower': [0.83, 0.80, 0.76, 0.73],
            'ci_upper': [0.87, 0.84, 0.80, 0.77],
            'category': ['Large', 'Large', 'Large', 'Large'],
            'parameters': [175e9, 100e9, 80e9, 70e9],
            'cost_per_token': [0.03, 0.025, 0.02, 0.015]
        })
    
    @pytest.fixture
    def sample_scaling_data(self):
        """Fixture providing sample scaling data for visualization."""
        return pd.DataFrame({
            'model_size': [1e9, 5e9, 10e9, 50e9, 100e9, 175e9],
            'performance': [0.45, 0.62, 0.71, 0.82, 0.85, 0.87],
            'ci_lower': [0.42, 0.59, 0.68, 0.79, 0.82, 0.84],
            'ci_upper': [0.48, 0.65, 0.74, 0.85, 0.88, 0.90],
            'model_name': ['Small-1B', 'Medium-5B', 'Large-10B', 'XL-50B', 'XXL-100B', 'XXXL-175B']
        })
    
    @pytest.fixture
    def publication_visualizer(self, temp_output_dir):
        """Fixture providing PublicationVisualizer instance."""
        return PublicationVisualizer(
            output_dir=str(temp_output_dir),
            theme='publication',
            dpi=300,
            figsize=(10, 6)
        )
    
    def test_publication_visualizer_initialization(self, temp_output_dir):
        """Test PublicationVisualizer initialization with various parameters."""
        visualizer = PublicationVisualizer(
            output_dir=str(temp_output_dir),
            theme='nature',
            dpi=600,
            figsize=(12, 8),
            color_palette='colorblind'
        )
        
        assert visualizer.output_dir == str(temp_output_dir)
        assert visualizer.theme == 'nature'
        assert visualizer.dpi == 600
        assert visualizer.figsize == (12, 8)
        assert visualizer.color_palette == 'colorblind'
    
    def test_create_comparison_plot(self, publication_visualizer, sample_benchmark_data):
        """Test creation of model comparison plots."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            fig, ax = publication_visualizer.create_comparison_plot(
                sample_benchmark_data,
                x_column='model',
                y_column='score',
                title='Model Performance Comparison',
                error_columns=['ci_lower', 'ci_upper']
            )
            
            assert fig is not None
            assert ax is not None
            # Should save the figure
            mock_savefig.assert_called_once()
    
    def test_create_confidence_interval_plot(self, publication_visualizer, sample_benchmark_data):
        """Test confidence interval visualization."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = publication_visualizer.create_confidence_interval_plot(
                sample_benchmark_data,
                x_column='model',
                y_column='score',
                ci_columns=['ci_lower', 'ci_upper'],
                title='Model Performance with 95% CI'
            )
            
            assert fig is not None
            assert ax is not None
            # Verify error bars or confidence intervals are plotted
    
    def test_create_scaling_plot(self, publication_visualizer, sample_scaling_data):
        """Test scaling law visualization."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = publication_visualizer.create_scaling_plot(
                sample_scaling_data,
                x_column='model_size',
                y_column='performance',
                log_scale=True,
                fit_line=True,
                title='Performance Scaling with Model Size'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_save_publication_figure(self, publication_visualizer):
        """Test saving figures in publication-ready format."""
        # Create a simple figure for testing
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_path = publication_visualizer.output_dir + '/test_figure'
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            publication_visualizer.save_figure(
                fig,
                'test_figure',
                formats=['png', 'pdf', 'svg']
            )
            
            # Should save in multiple formats
            assert mock_savefig.call_count == 3
    
    def test_plot_styling_consistency(self, publication_visualizer, sample_benchmark_data):
        """Test consistent plot styling for publications."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = publication_visualizer.create_comparison_plot(
                sample_benchmark_data,
                x_column='model',
                y_column='score',
                apply_theme=True
            )
            
            # Check that theme styling is applied
            # (In real implementation, would check specific style properties)
            assert fig is not None
    
    def test_color_palette_consistency(self, publication_visualizer):
        """Test color palette consistency across visualizations."""
        colors = publication_visualizer.get_color_palette(n_colors=4)
        
        assert len(colors) == 4
        assert all(isinstance(color, (str, tuple)) for color in colors)
    
    def test_subplot_layout_creation(self, publication_visualizer, sample_benchmark_data, sample_scaling_data):
        """Test creation of multi-panel figures."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig = publication_visualizer.create_multi_panel_figure(
                panels=[
                    {
                        'type': 'comparison',
                        'data': sample_benchmark_data,
                        'x': 'model',
                        'y': 'score',
                        'title': 'Model Comparison'
                    },
                    {
                        'type': 'scaling',
                        'data': sample_scaling_data,
                        'x': 'model_size',
                        'y': 'performance',
                        'title': 'Scaling Law'
                    }
                ],
                layout=(1, 2),
                figsize=(16, 6)
            )
            
            assert fig is not None


class TestScalingPlots:
    """Test scaling visualization functionality."""
    
    @pytest.fixture
    def scaling_plotter(self):
        """Fixture providing ScalingPlots instance."""
        return ScalingPlots(theme='publication')
    
    @pytest.fixture
    def scaling_data_with_fit(self):
        """Fixture providing scaling data with fitted parameters."""
        return {
            'data': pd.DataFrame({
                'log_size': np.log10([1e9, 5e9, 10e9, 50e9, 100e9]),
                'performance': [0.45, 0.62, 0.71, 0.82, 0.85],
                'ci_lower': [0.42, 0.59, 0.68, 0.79, 0.82],
                'ci_upper': [0.48, 0.65, 0.74, 0.85, 0.88]
            }),
            'fit_params': {'alpha': 0.15, 'beta': 0.45, 'r_squared': 0.92}
        }
    
    def test_power_law_fit_visualization(self, scaling_plotter, scaling_data_with_fit):
        """Test power law fit visualization."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = scaling_plotter.plot_power_law_fit(
                scaling_data_with_fit['data'],
                x_column='log_size',
                y_column='performance',
                fit_params=scaling_data_with_fit['fit_params'],
                show_equation=True
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_residuals_plot(self, scaling_plotter, scaling_data_with_fit):
        """Test residuals analysis plot."""
        # Add residuals to the data
        data = scaling_data_with_fit['data'].copy()
        data['residuals'] = np.random.normal(0, 0.02, len(data))
        
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = scaling_plotter.plot_residuals(
                data,
                x_column='log_size',
                residuals_column='residuals'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_scaling_with_bands(self, scaling_plotter, scaling_data_with_fit):
        """Test scaling plot with confidence bands."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = scaling_plotter.plot_scaling_with_bands(
                scaling_data_with_fit['data'],
                x_column='log_size',
                y_column='performance',
                ci_columns=['ci_lower', 'ci_upper'],
                show_fit=True
            )
            
            assert fig is not None
            assert ax is not None


class TestComparisonPlots:
    """Test comparison visualization functionality."""
    
    @pytest.fixture
    def comparison_plotter(self):
        """Fixture providing ComparisonPlots instance."""
        return ComparisonPlots(theme='publication')
    
    def test_bar_chart_with_error_bars(self, comparison_plotter, sample_benchmark_data):
        """Test bar chart with error bars."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = comparison_plotter.create_bar_chart(
                sample_benchmark_data,
                x_column='model',
                y_column='score',
                error_columns=['ci_lower', 'ci_upper'],
                title='Model Performance Comparison'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_violin_plot_comparison(self, comparison_plotter):
        """Test violin plot for distribution comparison."""
        # Create sample distribution data
        np.random.seed(42)
        distribution_data = pd.DataFrame({
            'model': np.repeat(['GPT-4', 'Claude-3', 'Gemini-Pro'], 100),
            'score': np.concatenate([
                np.random.normal(0.85, 0.05, 100),
                np.random.normal(0.82, 0.04, 100),
                np.random.normal(0.78, 0.06, 100)
            ])
        })
        
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = comparison_plotter.create_violin_plot(
                distribution_data,
                x_column='model',
                y_column='score',
                title='Score Distribution by Model'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_forest_plot(self, comparison_plotter, sample_benchmark_data):
        """Test forest plot for effect size visualization."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = comparison_plotter.create_forest_plot(
                sample_benchmark_data,
                labels_column='model',
                estimates_column='score',
                ci_columns=['ci_lower', 'ci_upper'],
                title='Effect Sizes with Confidence Intervals'
            )
            
            assert fig is not None
            assert ax is not None


class TestConfidenceIntervalPlots:
    """Test confidence interval visualization functionality."""
    
    @pytest.fixture
    def ci_plotter(self):
        """Fixture providing ConfidenceIntervalPlots instance."""
        return ConfidenceIntervalPlots(theme='publication')
    
    def test_bootstrap_distribution_plot(self, ci_plotter):
        """Test bootstrap distribution visualization."""
        # Generate sample bootstrap distribution
        np.random.seed(42)
        bootstrap_samples = np.random.normal(0.85, 0.03, 1000)
        
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = ci_plotter.plot_bootstrap_distribution(
                bootstrap_samples,
                original_estimate=0.85,
                confidence_level=0.95,
                title='Bootstrap Distribution of Performance Estimate'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_ci_comparison_plot(self, ci_plotter, sample_benchmark_data):
        """Test confidence interval comparison plot."""
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = ci_plotter.plot_ci_comparison(
                sample_benchmark_data,
                labels_column='model',
                estimates_column='score',
                ci_columns=['ci_lower', 'ci_upper'],
                title='Performance Estimates with 95% CI'
            )
            
            assert fig is not None
            assert ax is not None
    
    def test_coverage_probability_plot(self, ci_plotter):
        """Test confidence interval coverage probability visualization."""
        # Generate sample coverage data
        coverage_data = pd.DataFrame({
            'confidence_level': [0.80, 0.85, 0.90, 0.95, 0.99],
            'actual_coverage': [0.82, 0.86, 0.91, 0.94, 0.98],
            'lower_bound': [0.79, 0.83, 0.88, 0.91, 0.95],
            'upper_bound': [0.85, 0.89, 0.94, 0.97, 1.01]
        })
        
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            
            fig, ax = ci_plotter.plot_coverage_probability(
                coverage_data,
                nominal_column='confidence_level',
                actual_column='actual_coverage',
                title='Confidence Interval Coverage Analysis'
            )
            
            assert fig is not None
            assert ax is not None


class TestThemeManager:
    """Test theme management functionality."""
    
    def test_theme_manager_initialization(self):
        """Test ThemeManager initialization."""
        theme_manager = ThemeManager(default_theme='publication')
        
        assert theme_manager.default_theme == 'publication'
    
    def test_apply_publication_theme(self):
        """Test application of publication theme."""
        theme_manager = ThemeManager()
        
        with patch('matplotlib.pyplot.rcParams') as mock_rcparams:
            theme_manager.apply_theme('publication')
            
            # Should modify matplotlib rcParams
            assert mock_rcparams.__setitem__.called
    
    def test_apply_nature_theme(self):
        """Test application of Nature journal theme."""
        theme_manager = ThemeManager()
        
        with patch('matplotlib.pyplot.rcParams'):
            theme_config = theme_manager.get_theme_config('nature')
            
            assert 'font.family' in theme_config
            assert 'figure.dpi' in theme_config
            assert 'axes.linewidth' in theme_config
    
    def test_custom_theme_registration(self):
        """Test registration of custom themes."""
        theme_manager = ThemeManager()
        
        custom_theme = {
            'font.family': 'Times New Roman',
            'font.size': 14,
            'figure.figsize': (8, 6),
            'axes.labelsize': 12
        }
        
        theme_manager.register_custom_theme('custom_journal', custom_theme)
        
        assert 'custom_journal' in theme_manager.available_themes()
    
    def test_color_palette_management(self):
        """Test color palette management."""
        theme_manager = ThemeManager()
        
        # Test predefined palettes
        colorblind_palette = theme_manager.get_color_palette('colorblind', n_colors=5)
        assert len(colorblind_palette) == 5
        
        # Test custom palette registration
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        theme_manager.register_color_palette('custom', custom_colors)
        
        assert 'custom' in theme_manager.available_palettes()


class TestPublicationVisualizerIntegration:
    """Test integration between visualization components."""
    
    def test_comprehensive_figure_generation(self, temp_output_dir):
        """Test comprehensive figure generation for publication."""
        # Create comprehensive test data
        model_data = pd.DataFrame({
            'model': ['GPT-4', 'Claude-3', 'Gemini-Pro'],
            'accuracy': [0.87, 0.84, 0.81],
            'ci_lower': [0.85, 0.82, 0.79],
            'ci_upper': [0.89, 0.86, 0.83],
            'parameters': [175e9, 100e9, 80e9]
        })
        
        scaling_data = pd.DataFrame({
            'log_params': [9.0, 9.5, 10.0, 10.5, 11.0, 11.24],
            'performance': [0.65, 0.72, 0.78, 0.83, 0.86, 0.87]
        })
        
        visualizer = PublicationVisualizer(
            output_dir=str(temp_output_dir),
            theme='publication'
        )
        
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig') as mock_savefig:
            
            # Generate comprehensive figure set
            figures = visualizer.generate_publication_figures(
                {
                    'model_comparison': model_data,
                    'scaling_analysis': scaling_data
                },
                figure_types=['comparison', 'scaling', 'combined']
            )
            
            assert len(figures) >= 2  # At least comparison and scaling figures
            assert mock_savefig.called
    
    def test_export_figure_metadata(self, temp_output_dir):
        """Test export of figure metadata for publications."""
        visualizer = PublicationVisualizer(output_dir=str(temp_output_dir))
        
        figure_metadata = {
            'figure_1': {
                'title': 'Model Performance Comparison',
                'caption': 'Comparison of model performance with 95% confidence intervals',
                'data_source': 'benchmark_results.csv',
                'statistical_tests': ['one-way ANOVA', 'Tukey HSD']
            }
        }
        
        with patch('json.dump') as mock_json_dump:
            visualizer.export_figure_metadata(figure_metadata)
            
            mock_json_dump.assert_called_once()
    
    def test_batch_figure_export(self, temp_output_dir):
        """Test batch export of multiple figures."""
        visualizer = PublicationVisualizer(output_dir=str(temp_output_dir))
        
        # Create multiple figures
        figures = {}
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            figures[f'figure_{i}'] = fig
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            visualizer.batch_save_figures(
                figures,
                formats=['png', 'pdf'],
                dpi=300
            )
            
            # Should save each figure in multiple formats
            expected_calls = len(figures) * 2  # 2 formats per figure
            assert mock_savefig.call_count == expected_calls