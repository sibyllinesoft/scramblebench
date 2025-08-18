"""
Comprehensive test suite for ScrambleBench CLI functionality.

This module provides extensive tests for all CLI commands and options,
covering language generation, text transformation, batch processing,
evaluation, and utility functions.
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import yaml

import pytest
from click.testing import CliRunner

from scramblebench.cli import cli, CLIContext
from scramblebench.translation.language_generator import LanguageType


class TestCLISetup:
    """Test CLI setup and context management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_cli_context_initialization(self):
        """Test CLI context initialization."""
        with self.runner.isolated_filesystem():
            ctx = CLIContext()
            
            assert ctx.verbose is False
            assert ctx.quiet is False
            assert ctx.output_format == "text"
            assert isinstance(ctx.data_dir, Path)
            assert isinstance(ctx.languages_dir, Path)
            assert isinstance(ctx.results_dir, Path)
            
    def test_cli_context_directory_creation(self):
        """Test that CLI context creates necessary directories."""
        with self.runner.isolated_filesystem():
            ctx = CLIContext()
            
            assert ctx.languages_dir.exists()
            assert ctx.results_dir.exists()
            
    def test_global_options(self):
        """Test global CLI options."""
        # Test verbose flag
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
        
        # Test quiet flag
        result = self.runner.invoke(cli, ['--quiet', '--help'])
        assert result.exit_code == 0
        
        # Test output format
        result = self.runner.invoke(cli, ['--output-format', 'json', '--help'])
        assert result.exit_code == 0
        
        # Test data directory
        result = self.runner.invoke(cli, ['--data-dir', str(self.temp_dir), '--help'])
        assert result.exit_code == 0


class TestLanguageCommands:
    """Test language generation and management commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_language_generate_substitution(self):
        """Test generating a substitution language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_substitution',
            '--type', 'substitution',
            '--complexity', '3',
            '--vocab-size', '100',
            '--seed', '42'
        ])
        
        assert result.exit_code == 0
        assert 'test_substitution' in result.output
        
        # Check that language file was created
        lang_file = self.temp_dir / 'languages' / 'test_substitution.json'
        assert lang_file.exists()
        
        # Verify file content
        with open(lang_file) as f:
            data = json.load(f)
        
        assert data['name'] == 'test_substitution'
        assert data['language_type'] == 'substitution'
        assert len(data['rules']) > 0
        assert len(data['vocabulary']) > 0
        
    def test_language_generate_phonetic(self):
        """Test generating a phonetic language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_phonetic',
            '--type', 'phonetic',
            '--complexity', '5',
            '--vocab-size', '200'
        ])
        
        assert result.exit_code == 0
        
        lang_file = self.temp_dir / 'languages' / 'test_phonetic.json'
        assert lang_file.exists()
        
        with open(lang_file) as f:
            data = json.load(f)
        
        assert data['language_type'] == 'phonetic'
        
    def test_language_generate_scrambled(self):
        """Test generating a scrambled language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_scrambled',
            '--type', 'scrambled',
            '--complexity', '4'
        ])
        
        assert result.exit_code == 0
        
        lang_file = self.temp_dir / 'languages' / 'test_scrambled.json'
        assert lang_file.exists()
        
    def test_language_generate_synthetic(self):
        """Test generating a synthetic language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_synthetic',
            '--type', 'synthetic',
            '--complexity', '7',
            '--vocab-size', '500'
        ])
        
        assert result.exit_code == 0
        
        lang_file = self.temp_dir / 'languages' / 'test_synthetic.json'
        assert lang_file.exists()
        
    def test_language_generate_invalid_type(self):
        """Test generating language with invalid type."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_invalid',
            '--type', 'invalid_type'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid language type' in result.output
        
    def test_language_generate_invalid_complexity(self):
        """Test generating language with invalid complexity."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_invalid',
            '--type', 'substitution',
            '--complexity', '15'  # Invalid (> 10)
        ])
        
        assert result.exit_code != 0
        
    def test_language_list_empty(self):
        """Test listing languages when none exist."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        assert 'No languages found' in result.output or len(result.output.strip()) == 0
        
    def test_language_list_with_languages(self):
        """Test listing languages when some exist."""
        # First generate a language
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_list',
            '--type', 'substitution'
        ])
        
        # Then list languages
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        assert 'test_list' in result.output
        
    def test_language_list_json_format(self):
        """Test listing languages in JSON format."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_json',
            '--type', 'substitution'
        ])
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--output-format', 'json',
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, (list, dict))
        
    def test_language_info_existing(self):
        """Test getting info for existing language."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_info',
            '--type', 'substitution'
        ])
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'info', 'test_info'
        ])
        
        assert result.exit_code == 0
        assert 'test_info' in result.output
        assert 'substitution' in result.output
        
    def test_language_info_nonexistent(self):
        """Test getting info for non-existent language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'info', 'nonexistent_language'
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower()
        
    def test_language_delete_existing(self):
        """Test deleting existing language."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_delete',
            '--type', 'substitution'
        ])
        
        # Verify it exists
        lang_file = self.temp_dir / 'languages' / 'test_delete.json'
        assert lang_file.exists()
        
        # Delete it
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'delete', 'test_delete'
        ], input='y\n')  # Confirm deletion
        
        assert result.exit_code == 0
        assert not lang_file.exists()
        
    def test_language_delete_nonexistent(self):
        """Test deleting non-existent language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'delete', 'nonexistent'
        ])
        
        assert result.exit_code != 0
        
    def test_language_delete_cancel(self):
        """Test canceling language deletion."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'test_cancel',
            '--type', 'substitution'
        ])
        
        lang_file = self.temp_dir / 'languages' / 'test_cancel.json'
        assert lang_file.exists()
        
        # Try to delete but cancel
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'delete', 'test_cancel'
        ], input='n\n')  # Cancel deletion
        
        assert result.exit_code == 0
        assert lang_file.exists()  # Should still exist


class TestTransformCommands:
    """Test text transformation commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_transform_text_simple(self):
        """Test simple text transformation."""
        # First generate a language
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'transform_test',
            '--type', 'substitution'
        ])
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'transform', 'text',
            'Hello world',
            'transform_test'
        ])
        
        assert result.exit_code == 0
        # Output should contain transformed text (different from input)
        assert 'Hello world' not in result.output or 'transformed' in result.output.lower()
        
    def test_transform_text_nonexistent_language(self):
        """Test text transformation with non-existent language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'transform', 'text',
            'Hello world',
            'nonexistent_language'
        ])
        
        assert result.exit_code != 0
        
    def test_transform_proper_nouns_random(self):
        """Test proper noun transformation with random strategy."""
        result = self.runner.invoke(cli, [
            'transform', 'proper-nouns',
            'John went to New York',
            '--strategy', 'random'
        ])
        
        assert result.exit_code == 0
        # Should contain transformed text
        
    def test_transform_proper_nouns_swap(self):
        """Test proper noun transformation with swap strategy."""
        result = self.runner.invoke(cli, [
            'transform', 'proper-nouns',
            'Alice talked to Bob',
            '--strategy', 'swap'
        ])
        
        assert result.exit_code == 0
        
    def test_transform_proper_nouns_invalid_strategy(self):
        """Test proper noun transformation with invalid strategy."""
        result = self.runner.invoke(cli, [
            'transform', 'proper-nouns',
            'John went to New York',
            '--strategy', 'invalid_strategy'
        ])
        
        assert result.exit_code != 0
        
    def test_transform_synonyms(self):
        """Test synonym replacement transformation."""
        result = self.runner.invoke(cli, [
            'transform', 'synonyms',
            'The big cat ran quickly',
            '--rate', '0.5'
        ])
        
        assert result.exit_code == 0
        
    def test_transform_synonyms_invalid_rate(self):
        """Test synonym replacement with invalid rate."""
        result = self.runner.invoke(cli, [
            'transform', 'synonyms',
            'Test text',
            '--rate', '1.5'  # Invalid (> 1.0)
        ])
        
        assert result.exit_code != 0
        
    def test_transform_synonyms_zero_rate(self):
        """Test synonym replacement with zero rate."""
        result = self.runner.invoke(cli, [
            'transform', 'synonyms',
            'Test text',
            '--rate', '0.0'
        ])
        
        assert result.exit_code == 0
        # Should return original text unchanged
        assert 'Test text' in result.output


class TestBatchCommands:
    """Test batch processing commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample data files
        self.sample_json = self.temp_dir / 'sample.json'
        self.sample_jsonl = self.temp_dir / 'sample.jsonl'
        
        sample_data = [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
        
        with open(self.sample_json, 'w') as f:
            json.dump(sample_data, f)
            
        with open(self.sample_jsonl, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
                
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_batch_extract_vocab_json(self):
        """Test vocabulary extraction from JSON file."""
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(self.sample_json)
        ])
        
        assert result.exit_code == 0
        assert 'AI' in result.output
        assert 'ML' in result.output
        
    def test_batch_extract_vocab_jsonl(self):
        """Test vocabulary extraction from JSONL file."""
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(self.sample_jsonl)
        ])
        
        assert result.exit_code == 0
        
    def test_batch_extract_vocab_nonexistent_file(self):
        """Test vocabulary extraction from non-existent file."""
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(self.temp_dir / 'nonexistent.json')
        ])
        
        assert result.exit_code != 0
        
    def test_batch_extract_vocab_with_output_file(self):
        """Test vocabulary extraction with output file."""
        output_file = self.temp_dir / 'vocab_output.txt'
        
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(self.sample_json),
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check output file content
        with open(output_file) as f:
            content = f.read()
            assert 'AI' in content
            
    def test_batch_extract_vocab_min_frequency(self):
        """Test vocabulary extraction with minimum frequency."""
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(self.sample_json),
            '--min-frequency', '2'
        ])
        
        assert result.exit_code == 0
        
    def test_batch_transform_questions_json(self):
        """Test transforming questions in JSON file."""
        # First generate a language
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'batch_test',
            '--type', 'substitution'
        ])
        
        output_file = self.temp_dir / 'transformed_questions.json'
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'batch', 'transform-questions',
            str(self.sample_json),
            'batch_test',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Check output file content
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            
    def test_batch_transform_questions_preserve_answers(self):
        """Test transforming questions while preserving answers."""
        # Generate a language
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'preserve_test',
            '--type', 'substitution'
        ])
        
        output_file = self.temp_dir / 'transformed_preserve.json'
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'batch', 'transform-questions',
            str(self.sample_json),
            'preserve_test',
            '--output', str(output_file),
            '--preserve-answers'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()


class TestEvaluationCommands:
    """Test evaluation pipeline commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample benchmark file
        self.benchmark_file = self.temp_dir / 'test_benchmark.json'
        benchmark_data = [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is the capital of France?", "answer": "Paris"}
        ]
        
        with open(self.benchmark_file, 'w') as f:
            json.dump(benchmark_data, f)
            
        # Create sample config file
        self.config_file = self.temp_dir / 'eval_config.yaml'
        config_data = {
            'experiment_name': 'test_evaluation',
            'description': 'Test evaluation run',
            'models': [
                {
                    'name': 'openai/gpt-3.5-turbo',
                    'provider': 'openrouter',
                    'temperature': 0.0
                }
            ],
            'benchmark_paths': [str(self.benchmark_file)],
            'transformations': {
                'enabled_types': ['language_translation'],
                'synonym_rate': 0.3
            },
            'max_samples': 10,
            'generate_plots': False,
            'calculate_significance': False
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
            
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'})
    @patch('scramblebench.evaluation.runner.EvaluationRunner')
    def test_evaluate_run_with_config(self, mock_runner_class):
        """Test running evaluation with config file."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'run',
            '--config', str(self.config_file)
        ])
        
        assert result.exit_code == 0
        mock_runner_class.assert_called_once()
        
    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key'})
    @patch('scramblebench.evaluation.runner.EvaluationRunner')
    def test_evaluate_run_with_parameters(self, mock_runner_class):
        """Test running evaluation with command-line parameters."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'run',
            '--models', 'openai/gpt-3.5-turbo',
            '--benchmarks', str(self.benchmark_file),
            '--experiment-name', 'cli_test',
            '--max-samples', '5'
        ])
        
        assert result.exit_code == 0
        
    def test_evaluate_run_missing_api_key(self):
        """Test evaluation run without API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = self.runner.invoke(cli, [
                'evaluate', 'run',
                '--models', 'openai/gpt-3.5-turbo',
                '--benchmarks', str(self.benchmark_file)
            ])
            
            assert result.exit_code != 0
            assert 'API key' in result.output
            
    def test_evaluate_run_missing_config_and_params(self):
        """Test evaluation run without config or required parameters."""
        result = self.runner.invoke(cli, [
            'evaluate', 'run'
        ])
        
        assert result.exit_code != 0
        
    @patch('scramblebench.evaluation.results.ResultsManager')
    def test_evaluate_analyze(self, mock_results_manager):
        """Test evaluation analysis command."""
        mock_manager = Mock()
        mock_results_manager.return_value = mock_manager
        
        # Mock results
        mock_results = Mock()
        mock_results.experiment_name = 'test_experiment'
        mock_results.metrics = {'accuracy': 0.85}
        mock_manager.load_results.return_value = mock_results
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'analyze',
            'test_experiment'
        ])
        
        assert result.exit_code == 0
        
    def test_evaluate_analyze_nonexistent_experiment(self):
        """Test analyzing non-existent experiment."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'analyze',
            'nonexistent_experiment'
        ])
        
        assert result.exit_code != 0
        
    @patch('scramblebench.evaluation.results.ResultsManager')
    def test_evaluate_list(self, mock_results_manager):
        """Test listing evaluation experiments."""
        mock_manager = Mock()
        mock_results_manager.return_value = mock_manager
        mock_manager.list_experiments.return_value = ['exp1', 'exp2', 'exp3']
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'list'
        ])
        
        assert result.exit_code == 0
        assert 'exp1' in result.output
        assert 'exp2' in result.output
        assert 'exp3' in result.output
        
    @patch('scramblebench.evaluation.results.ResultsManager')
    def test_evaluate_compare(self, mock_results_manager):
        """Test comparing evaluation experiments."""
        mock_manager = Mock()
        mock_results_manager.return_value = mock_manager
        
        # Mock results for comparison
        mock_results1 = Mock()
        mock_results1.experiment_name = 'exp1'
        mock_results1.metrics = {'accuracy': 0.85}
        
        mock_results2 = Mock()
        mock_results2.experiment_name = 'exp2'
        mock_results2.metrics = {'accuracy': 0.82}
        
        mock_manager.load_results.side_effect = [mock_results1, mock_results2]
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'evaluate', 'compare',
            'exp1', 'exp2'
        ])
        
        assert result.exit_code == 0


class TestOutputFormats:
    """Test different output formats."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_text_output_format(self):
        """Test text output format."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--output-format', 'text',
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        # Text format should not be JSON
        try:
            json.loads(result.output)
            assert False, "Output should not be JSON"
        except json.JSONDecodeError:
            pass  # Expected for text format
            
    def test_json_output_format(self):
        """Test JSON output format."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'json_test',
            '--type', 'substitution'
        ])
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--output-format', 'json',
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, (list, dict))
        
    def test_yaml_output_format(self):
        """Test YAML output format."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--output-format', 'yaml',
            'language', 'generate', 'yaml_test',
            '--type', 'substitution'
        ])
        
        assert result.exit_code == 0
        # Should be valid YAML (basic check)
        assert 'yaml_test' in result.output


class TestVerbosityAndQuietModes:
    """Test verbose and quiet output modes."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_verbose_mode(self):
        """Test verbose output mode."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--verbose',
            'language', 'generate', 'verbose_test',
            '--type', 'substitution'
        ])
        
        assert result.exit_code == 0
        # Verbose mode should include additional information
        assert len(result.output) > 0
        
    def test_quiet_mode(self):
        """Test quiet output mode."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            '--quiet',
            'language', 'generate', 'quiet_test',
            '--type', 'substitution'
        ])
        
        assert result.exit_code == 0
        # Quiet mode should have minimal output
        
    def test_verbose_and_quiet_conflict(self):
        """Test that verbose and quiet flags conflict."""
        result = self.runner.invoke(cli, [
            '--verbose',
            '--quiet',
            'language', 'list'
        ])
        
        # Should handle the conflict (implementation-dependent behavior)
        assert result.exit_code in [0, 2]  # 0 for success, 2 for parameter error


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(cli, ['invalid_command'])
        
        assert result.exit_code != 0
        assert 'No such command' in result.output
        
    def test_invalid_subcommand(self):
        """Test invalid subcommand handling."""
        result = self.runner.invoke(cli, ['language', 'invalid_subcommand'])
        
        assert result.exit_code != 0
        
    def test_missing_required_argument(self):
        """Test missing required argument."""
        result = self.runner.invoke(cli, ['language', 'generate'])
        
        assert result.exit_code != 0
        assert 'Missing argument' in result.output
        
    def test_invalid_option_value(self):
        """Test invalid option value."""
        result = self.runner.invoke(cli, [
            'language', 'generate', 'test',
            '--complexity', 'invalid'
        ])
        
        assert result.exit_code != 0
        
    def test_file_permission_error(self):
        """Test file permission error handling."""
        # Create a read-only directory
        readonly_dir = self.temp_dir / 'readonly'
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            result = self.runner.invoke(cli, [
                '--data-dir', str(readonly_dir),
                'language', 'generate', 'permission_test',
                '--type', 'substitution'
            ])
            
            # Should handle permission error gracefully
            assert result.exit_code != 0
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
            
    def test_disk_space_error_simulation(self):
        """Test handling of disk space errors (simulated)."""
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            result = self.runner.invoke(cli, [
                '--data-dir', str(self.temp_dir),
                'language', 'generate', 'space_test',
                '--type', 'substitution'
            ])
            
            assert result.exit_code != 0
            assert 'space' in result.output.lower() or 'error' in result.output.lower()


class TestConfigurationHandling:
    """Test configuration file handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_valid_yaml_config(self):
        """Test loading valid YAML configuration."""
        config_file = self.temp_dir / 'valid_config.yaml'
        config_data = {
            'experiment_name': 'test_config',
            'models': [{'name': 'test_model', 'provider': 'openrouter'}],
            'benchmark_paths': ['test.json'],
            'transformations': {'enabled_types': ['language_translation']}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
            
        # Test that config file can be loaded (basic validation)
        result = self.runner.invoke(cli, [
            'evaluate', 'run',
            '--config', str(config_file),
            '--dry-run'  # If such option exists
        ])
        
        # Should not fail on config loading
        assert 'yaml' not in result.output.lower() or result.exit_code == 0
        
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration."""
        config_file = self.temp_dir / 'invalid_config.yaml'
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
            
        result = self.runner.invoke(cli, [
            'evaluate', 'run',
            '--config', str(config_file)
        ])
        
        assert result.exit_code != 0
        
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        result = self.runner.invoke(cli, [
            'evaluate', 'run',
            '--config', str(self.temp_dir / 'nonexistent.yaml')
        ])
        
        assert result.exit_code != 0


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_large_vocabulary_extraction(self):
        """Test vocabulary extraction from large dataset."""
        # Create large dataset
        large_data = [
            {"question": f"Question {i} with unique words", "answer": f"Answer {i}"}
            for i in range(1000)
        ]
        
        large_file = self.temp_dir / 'large_dataset.json'
        with open(large_file, 'w') as f:
            json.dump(large_data, f)
            
        result = self.runner.invoke(cli, [
            'batch', 'extract-vocab',
            str(large_file),
            '--min-frequency', '2'
        ])
        
        assert result.exit_code == 0
        
    def test_batch_transformation_performance(self):
        """Test performance of batch transformations."""
        # Create moderately large dataset
        data = [
            {"question": f"What is the answer to question {i}?", "answer": f"Answer {i}"}
            for i in range(100)
        ]
        
        data_file = self.temp_dir / 'batch_data.json'
        with open(data_file, 'w') as f:
            json.dump(data, f)
            
        # Generate language
        self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'language', 'generate', 'perf_test',
            '--type', 'substitution'
        ])
        
        output_file = self.temp_dir / 'batch_output.json'
        
        import time
        start_time = time.time()
        
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.temp_dir),
            'batch', 'transform-questions',
            str(data_file),
            'perf_test',
            '--output', str(output_file)
        ])
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert result.exit_code == 0
        assert duration < 30  # Should complete within 30 seconds
        assert output_file.exists()


class TestHelpAndDocumentation:
    """Test help and documentation features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        
    def test_main_help(self):
        """Test main help command."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'ScrambleBench' in result.output or 'CLI' in result.output
        assert 'language' in result.output
        assert 'transform' in result.output
        assert 'batch' in result.output
        assert 'evaluate' in result.output
        
    def test_language_help(self):
        """Test language command help."""
        result = self.runner.invoke(cli, ['language', '--help'])
        
        assert result.exit_code == 0
        assert 'generate' in result.output
        assert 'list' in result.output
        assert 'info' in result.output
        
    def test_transform_help(self):
        """Test transform command help."""
        result = self.runner.invoke(cli, ['transform', '--help'])
        
        assert result.exit_code == 0
        assert 'text' in result.output
        assert 'proper-nouns' in result.output
        assert 'synonyms' in result.output
        
    def test_batch_help(self):
        """Test batch command help."""
        result = self.runner.invoke(cli, ['batch', '--help'])
        
        assert result.exit_code == 0
        assert 'extract-vocab' in result.output
        assert 'transform-questions' in result.output
        
    def test_evaluate_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(cli, ['evaluate', '--help'])
        
        assert result.exit_code == 0
        assert 'run' in result.output
        assert 'analyze' in result.output
        assert 'list' in result.output
        
    def test_subcommand_help(self):
        """Test subcommand help messages."""
        result = self.runner.invoke(cli, ['language', 'generate', '--help'])
        
        assert result.exit_code == 0
        assert 'type' in result.output
        assert 'complexity' in result.output
        assert 'vocab-size' in result.output