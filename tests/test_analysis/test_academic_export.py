"""Tests for academic export functionality in the analysis module."""

import pytest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

# Note: Import paths will need to be updated when the analysis module is implemented
from scramblebench.analysis.academic_export import (
    PreregReport,
    LaTeXTableFormatter,
    CSVExporter,
    AcademicExporter
)


class TestPreregReport:
    """Test preregistration report functionality."""
    
    @pytest.fixture
    def sample_preregistration(self):
        """Fixture providing sample preregistration data."""
        return {
            'title': 'Large Language Model Scaling Analysis',
            'authors': ['Alice Smith', 'Bob Johnson', 'Carol Davis'],
            'institution': 'Test University',
            'date': '2024-08-23',
            'research_question': 'How does model size affect performance on reasoning tasks?',
            'hypotheses': [
                'Larger models will show better performance',
                'Performance gains will follow a power law'
            ],
            'methodology': {
                'models': ['gpt-4', 'claude-3', 'gemini-pro'],
                'tasks': ['reasoning', 'comprehension', 'generation'],
                'metrics': ['accuracy', 'response_time'],
                'sample_size': 1000
            },
            'analysis_plan': {
                'primary_analysis': 'Linear mixed-effects model',
                'covariates': ['task_difficulty', 'prompt_length'],
                'multiple_comparisons': 'Bonferroni correction'
            },
            'power_analysis': {
                'effect_size': 0.3,
                'alpha': 0.05,
                'power': 0.8
            }
        }
    
    def test_prereg_report_creation(self, sample_preregistration):
        """Test PreregReport creation and basic functionality."""
        report = PreregReport(**sample_preregistration)
        
        assert report.title == 'Large Language Model Scaling Analysis'
        assert len(report.authors) == 3
        assert len(report.hypotheses) == 2
        assert report.methodology['sample_size'] == 1000
    
    def test_generate_markdown_report(self, sample_preregistration, temp_output_dir):
        """Test markdown preregistration report generation."""
        report = PreregReport(**sample_preregistration)
        
        output_path = temp_output_dir / "preregistration.md"
        
        with patch('builtins.open', mock_open()) as mock_file:
            report.generate_markdown_report(str(output_path))
            
            mock_file.assert_called_once_with(str(output_path), 'w', encoding='utf-8')
            written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            
            # Check key sections are included
            assert 'Large Language Model Scaling Analysis' in written_content
            assert 'Research Question' in written_content
            assert 'Hypotheses' in written_content
            assert 'Methodology' in written_content
            assert 'Analysis Plan' in written_content
    
    def test_generate_json_export(self, sample_preregistration, temp_output_dir):
        """Test JSON export of preregistration."""
        report = PreregReport(**sample_preregistration)
        
        output_path = temp_output_dir / "preregistration.json"
        
        with patch('builtins.open', mock_open()) as mock_file:
            report.export_json(str(output_path))
            
            mock_file.assert_called_once_with(str(output_path), 'w', encoding='utf-8')
            
            # Verify JSON structure would be written
            assert mock_file().write.called


class TestLaTeXTableFormatter:
    """Test LaTeX table formatting functionality."""
    
    @pytest.fixture
    def sample_results_data(self):
        """Fixture providing sample analysis results for table formatting."""
        return pd.DataFrame({
            'Model': ['GPT-4', 'Claude-3', 'Gemini-Pro'],
            'Accuracy': [0.85, 0.82, 0.78],
            'CI_Lower': [0.83, 0.80, 0.76],
            'CI_Upper': [0.87, 0.84, 0.80],
            'P_Value': [0.001, 0.003, 0.008],
            'Effect_Size': [0.45, 0.38, 0.32]
        })
    
    def test_latex_formatter_initialization(self):
        """Test LaTeXTableFormatter initialization."""
        formatter = LaTeXTableFormatter(
            table_style='booktabs',
            decimal_places=3,
            caption_placement='bottom'
        )
        
        assert formatter.table_style == 'booktabs'
        assert formatter.decimal_places == 3
        assert formatter.caption_placement == 'bottom'
    
    def test_format_comparison_table(self, sample_results_data):
        """Test LaTeX formatting of model comparison table."""
        formatter = LaTeXTableFormatter(table_style='booktabs', decimal_places=3)
        
        latex_output = formatter.format_comparison_table(
            sample_results_data,
            caption="Model Performance Comparison",
            label="tab:model_comparison"
        )
        
        assert '\\begin{table}' in latex_output
        assert '\\begin{tabular}' in latex_output
        assert '\\toprule' in latex_output  # booktabs style
        assert '\\bottomrule' in latex_output
        assert 'GPT-4' in latex_output
        assert '0.850' in latex_output  # Formatted to 3 decimal places
        assert '\\caption{Model Performance Comparison}' in latex_output
        assert '\\label{tab:model_comparison}' in latex_output
    
    def test_format_confidence_intervals(self, sample_results_data):
        """Test confidence interval formatting in LaTeX."""
        formatter = LaTeXTableFormatter(decimal_places=2)
        
        latex_output = formatter.format_confidence_intervals_table(
            sample_results_data,
            ci_columns=['CI_Lower', 'CI_Upper'],
            main_column='Accuracy'
        )
        
        assert '[0.83, 0.87]' in latex_output or '(0.83, 0.87)' in latex_output
        assert 'Accuracy' in latex_output
        assert '\\begin{tabular}' in latex_output
    
    def test_format_significance_stars(self):
        """Test significance star formatting."""
        formatter = LaTeXTableFormatter()
        
        # Test different significance levels
        assert formatter._format_significance_stars(0.001) == '***'
        assert formatter._format_significance_stars(0.01) == '**'
        assert formatter._format_significance_stars(0.05) == '*'
        assert formatter._format_significance_stars(0.1) == ''
    
    def test_escape_latex_special_chars(self):
        """Test LaTeX special character escaping."""
        formatter = LaTeXTableFormatter()
        
        text_with_specials = "Model_1 & Model#2 $100% Better"
        escaped = formatter._escape_latex(text_with_specials)
        
        assert '\\&' in escaped  # & should be escaped
        assert '\\#' in escaped  # # should be escaped
        assert '\\$' in escaped  # $ should be escaped
        assert '\\_' in escaped  # _ should be escaped
    
    def test_custom_column_formatting(self, sample_results_data):
        """Test custom column formatting options."""
        formatter = LaTeXTableFormatter(decimal_places=4)
        
        latex_output = formatter.format_comparison_table(
            sample_results_data,
            column_formats={'Accuracy': '.4f', 'P_Value': '.4f'},
            bold_best=True
        )
        
        # Should have custom formatting applied
        assert '0.8500' in latex_output  # 4 decimal places for accuracy
        assert '\\textbf{' in latex_output  # Bold formatting for best values


class TestCSVExporter:
    """Test CSV export functionality."""
    
    @pytest.fixture
    def sample_analysis_results(self):
        """Fixture providing comprehensive analysis results for export."""
        return {
            'model_comparison': pd.DataFrame({
                'model': ['gpt-4', 'claude-3', 'gemini-pro'],
                'accuracy': [0.85, 0.82, 0.78],
                'ci_lower': [0.83, 0.80, 0.76],
                'ci_upper': [0.87, 0.84, 0.80],
                'p_value': [0.001, 0.003, 0.008]
            }),
            'scaling_analysis': {
                'parameters': ['intercept', 'log_size'],
                'coefficients': [0.5, 0.15],
                'standard_errors': [0.02, 0.01],
                'confidence_intervals': [(0.46, 0.54), (0.13, 0.17)]
            },
            'metadata': {
                'analysis_date': '2024-08-23',
                'n_bootstrap': 1000,
                'confidence_level': 0.95
            }
        }
    
    def test_csv_exporter_initialization(self):
        """Test CSVExporter initialization."""
        exporter = CSVExporter(
            include_metadata=True,
            precision=4,
            separator=';'
        )
        
        assert exporter.include_metadata is True
        assert exporter.precision == 4
        assert exporter.separator == ';'
    
    def test_export_single_dataframe(self, sample_analysis_results, temp_output_dir):
        """Test exporting a single DataFrame to CSV."""
        exporter = CSVExporter(precision=3)
        
        output_path = temp_output_dir / "model_comparison.csv"
        
        exporter.export_dataframe(
            sample_analysis_results['model_comparison'],
            str(output_path)
        )
        
        # Verify the CSV file would be created
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            exporter.export_dataframe(
                sample_analysis_results['model_comparison'],
                str(output_path)
            )
            mock_to_csv.assert_called_once()
    
    def test_export_multiple_sheets(self, sample_analysis_results, temp_output_dir):
        """Test exporting multiple data sheets."""
        exporter = CSVExporter(include_metadata=True)
        
        output_dir = temp_output_dir / "analysis_results"
        
        # Mock os.makedirs to avoid actual directory creation
        with patch('os.makedirs'), \
             patch('pandas.DataFrame.to_csv') as mock_to_csv, \
             patch('builtins.open', mock_open()) as mock_file:
            
            exporter.export_analysis_results(
                sample_analysis_results,
                str(output_dir)
            )
            
            # Should create CSV files for DataFrames
            assert mock_to_csv.called
            
            # Should create metadata file if include_metadata is True
            mock_file.assert_called()
    
    def test_format_confidence_intervals_for_csv(self):
        """Test confidence interval formatting for CSV export."""
        exporter = CSVExporter()
        
        ci_data = pd.DataFrame({
            'parameter': ['intercept', 'slope'],
            'estimate': [0.5, 0.3],
            'ci_lower': [0.45, 0.25],
            'ci_upper': [0.55, 0.35]
        })
        
        formatted = exporter._format_confidence_intervals(ci_data)
        
        assert 'CI_95' in formatted.columns or 'confidence_interval' in formatted.columns
    
    def test_handle_missing_values(self):
        """Test handling of missing values in CSV export."""
        exporter = CSVExporter(handle_nan='replace')
        
        data_with_nan = pd.DataFrame({
            'values': [1.0, np.nan, 3.0, np.nan],
            'labels': ['a', 'b', 'c', 'd']
        })
        
        # Should handle NaN values appropriately
        processed = exporter._preprocess_dataframe(data_with_nan)
        
        # Verify NaN handling
        assert processed is not None


class TestAcademicExporter:
    """Test comprehensive academic export functionality."""
    
    @pytest.fixture
    def comprehensive_results(self):
        """Fixture providing comprehensive analysis results."""
        return {
            'experiment_info': {
                'title': 'Scaling Laws in Large Language Models',
                'authors': ['Dr. Alice Smith', 'Prof. Bob Johnson'],
                'date': '2024-08-23',
                'version': '1.0'
            },
            'model_performance': pd.DataFrame({
                'model': ['GPT-4', 'Claude-3', 'Gemini-Pro', 'Llama-2-70B'],
                'accuracy': [0.87, 0.85, 0.82, 0.79],
                'ci_lower': [0.85, 0.83, 0.80, 0.77],
                'ci_upper': [0.89, 0.87, 0.84, 0.81],
                'p_value': [0.001, 0.002, 0.005, 0.01],
                'effect_size': [0.52, 0.48, 0.42, 0.38]
            }),
            'scaling_parameters': {
                'alpha': 0.15,
                'alpha_ci': (0.13, 0.17),
                'beta': -0.05,
                'beta_ci': (-0.07, -0.03),
                'r_squared': 0.85,
                'aic': 245.3,
                'bic': 251.7
            },
            'bootstrap_results': {
                'n_bootstrap': 1000,
                'confidence_level': 0.95,
                'bias_corrected': True
            }
        }
    
    def test_academic_exporter_initialization(self):
        """Test AcademicExporter initialization."""
        exporter = AcademicExporter(
            output_format=['latex', 'csv', 'json'],
            include_plots=True,
            citation_style='apa'
        )
        
        assert 'latex' in exporter.output_format
        assert 'csv' in exporter.output_format
        assert exporter.include_plots is True
        assert exporter.citation_style == 'apa'
    
    def test_generate_comprehensive_report(self, comprehensive_results, temp_output_dir):
        """Test comprehensive academic report generation."""
        exporter = AcademicExporter(output_format=['latex', 'csv'])
        
        # Mock the individual export methods
        with patch.object(exporter, '_generate_latex_report') as mock_latex, \
             patch.object(exporter, '_export_csv_data') as mock_csv, \
             patch.object(exporter, '_generate_bibtex') as mock_bibtex:
            
            exporter.generate_comprehensive_report(
                comprehensive_results,
                str(temp_output_dir),
                include_bibtex=True
            )
            
            mock_latex.assert_called_once()
            mock_csv.assert_called_once()
            mock_bibtex.assert_called_once()
    
    def test_latex_report_generation(self, comprehensive_results, temp_output_dir):
        """Test LaTeX report generation with tables and formatting."""
        exporter = AcademicExporter()
        
        output_path = temp_output_dir / "analysis_report.tex"
        
        with patch('builtins.open', mock_open()) as mock_file:
            exporter._generate_latex_report(
                comprehensive_results,
                str(output_path)
            )
            
            mock_file.assert_called()
            written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            
            # Check for LaTeX document structure
            assert '\\documentclass' in written_content
            assert '\\begin{document}' in written_content
            assert '\\end{document}' in written_content
    
    def test_bibtex_generation(self, comprehensive_results):
        """Test BibTeX citation generation."""
        exporter = AcademicExporter(citation_style='apa')
        
        with patch('builtins.open', mock_open()) as mock_file:
            exporter._generate_bibtex(
                comprehensive_results,
                "references.bib"
            )
            
            mock_file.assert_called_with("references.bib", 'w', encoding='utf-8')
            written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
            
            # Check for BibTeX structure
            assert '@article{' in written_content or '@misc{' in written_content
            assert 'author = {' in written_content
            assert 'title = {' in written_content
    
    def test_supplementary_materials_export(self, comprehensive_results, temp_output_dir):
        """Test export of supplementary materials."""
        exporter = AcademicExporter(include_supplementary=True)
        
        supp_dir = temp_output_dir / "supplementary"
        
        with patch('os.makedirs'), \
             patch('pandas.DataFrame.to_csv'), \
             patch('json.dump'):
            
            exporter._export_supplementary_materials(
                comprehensive_results,
                str(supp_dir)
            )
            
            # Should create supplementary materials directory and files
            # Actual file creation is mocked for testing
    
    def test_format_for_journal_submission(self, comprehensive_results, temp_output_dir):
        """Test formatting for specific journal submission requirements."""
        exporter = AcademicExporter(journal_format='nature')
        
        # Mock journal-specific formatting
        with patch.object(exporter, '_apply_nature_formatting') as mock_nature:
            formatted_results = exporter.format_for_journal(
                comprehensive_results,
                journal='nature'
            )
            
            mock_nature.assert_called_once()
            assert formatted_results is not None


class TestAcademicExportIntegration:
    """Test integration between academic export components."""
    
    def test_end_to_end_export_workflow(self, temp_output_dir):
        """Test complete academic export workflow."""
        # Create comprehensive mock results
        results = {
            'experiment_metadata': {
                'title': 'LLM Scaling Study',
                'authors': ['Alice Smith', 'Bob Johnson'],
                'date': '2024-08-23'
            },
            'analysis_results': pd.DataFrame({
                'model': ['GPT-4', 'Claude-3'],
                'score': [0.85, 0.82],
                'ci_lower': [0.83, 0.80],
                'ci_upper': [0.87, 0.84]
            }),
            'statistical_tests': {
                'anova_p': 0.003,
                'effect_size': 0.45,
                'power': 0.89
            }
        }
        
        # Initialize exporter with multiple formats
        exporter = AcademicExporter(
            output_format=['latex', 'csv', 'json'],
            include_plots=False,  # Skip plots for testing
            citation_style='apa'
        )
        
        # Mock all file operations
        with patch('builtins.open', mock_open()), \
             patch('pandas.DataFrame.to_csv'), \
             patch('json.dump'), \
             patch('os.makedirs'):
            
            # Test the full export process
            exporter.generate_comprehensive_report(
                results,
                str(temp_output_dir),
                include_bibtex=True,
                include_supplementary=True
            )
            
            # Verify that export methods were called appropriately
            # (Actual verification depends on mock assertions in the methods)
    
    def test_export_error_handling(self, temp_output_dir):
        """Test error handling in academic export processes."""
        exporter = AcademicExporter()
        
        # Test with invalid data
        invalid_results = {'invalid': 'data'}
        
        with pytest.raises((ValueError, TypeError, KeyError)):
            exporter.generate_comprehensive_report(
                invalid_results,
                str(temp_output_dir)
            )
    
    def test_template_customization(self, temp_output_dir):
        """Test customization of export templates."""
        exporter = AcademicExporter()
        
        custom_template = {
            'latex_preamble': '\\usepackage{custom}',
            'table_style': 'fancy',
            'figure_placement': 'H'
        }
        
        # Test template customization
        exporter.set_custom_template(custom_template)
        
        assert exporter.custom_template == custom_template