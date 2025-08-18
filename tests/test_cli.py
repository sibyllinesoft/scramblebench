"""
Test suite for ScrambleBench CLI functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scramblebench.cli import cli
from scramblebench.translation.language_generator import LanguageType


class TestLanguageCommands:
    """Test language generation and management commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
    def test_generate_language_substitution(self):
        """Test generating a substitution language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'test_lang',
            '--type', 'substitution',
            '--complexity', '3',
            '--vocab-size', '100',
            '--seed', '42'
        ])
        
        assert result.exit_code == 0
        assert 'test_lang' in result.output
        
        # Check that language file was created
        lang_file = self.data_dir / 'languages' / 'test_lang.json'
        assert lang_file.exists()
        
        # Verify file content
        with open(lang_file) as f:
            data = json.load(f)
        
        assert data['name'] == 'test_lang'
        assert data['language_type'] == 'substitution'
        assert len(data['rules']) > 0
        assert len(data['vocabulary']) > 0
    
    def test_generate_language_all_types(self):
        """Test generating all language types."""
        for lang_type in LanguageType:
            result = self.runner.invoke(cli, [
                '--data-dir', str(self.data_dir),
                'language', 'generate', f'test_{lang_type.value}',
                '--type', lang_type.value,
                '--complexity', '2',
                '--vocab-size', '50',
                '--seed', '123'
            ])
            
            assert result.exit_code == 0, f"Failed for type {lang_type.value}: {result.output}"
    
    def test_list_languages_empty(self):
        """Test listing languages when none exist."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        assert 'No languages found' in result.output
    
    def test_list_languages_with_content(self):
        """Test listing languages after creating some."""
        # Create test languages
        for i, lang_type in enumerate(['substitution', 'phonetic']):
            self.runner.invoke(cli, [
                '--data-dir', str(self.data_dir),
                'language', 'generate', f'test_lang_{i}',
                '--type', lang_type,
                '--complexity', '2',
                '--seed', str(i)
            ])
        
        # List languages
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        assert 'test_lang_0' in result.output
        assert 'test_lang_1' in result.output
    
    def test_list_languages_json_format(self):
        """Test listing languages in JSON format."""
        # Create a test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'json_test',
            '--type', 'substitution',
            '--complexity', '2'
        ])
        
        # List in JSON format
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'list',
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        data = json.loads(result.output)
        assert 'languages' in data
        assert len(data['languages']) == 1
        assert data['languages'][0]['name'] == 'json_test'
    
    def test_show_language(self):
        """Test showing language details."""
        # Create test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'show_test',
            '--type', 'phonetic',
            '--complexity', '3'
        ])
        
        # Show language details
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'show', 'show_test'
        ])
        
        assert result.exit_code == 0
        assert 'show_test' in result.output
        assert 'phonetic' in result.output
    
    def test_show_language_with_rules(self):
        """Test showing language with rules details."""
        # Create test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'rules_test',
            '--type', 'substitution',
            '--complexity', '2'
        ])
        
        # Show with rules
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'show', 'rules_test',
            '--show-rules',
            '--limit', '10'
        ])
        
        assert result.exit_code == 0
        assert 'rules_test' in result.output
    
    def test_show_nonexistent_language(self):
        """Test showing a language that doesn't exist."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'show', 'nonexistent'
        ])
        
        assert result.exit_code == 1
        assert 'not found' in result.output
    
    def test_delete_language(self):
        """Test deleting a language."""
        # Create test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'delete_test',
            '--type', 'substitution'
        ])
        
        # Verify it exists
        lang_file = self.data_dir / 'languages' / 'delete_test.json'
        assert lang_file.exists()
        
        # Delete with force
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'delete', 'delete_test',
            '--force'
        ])
        
        assert result.exit_code == 0
        assert not lang_file.exists()


class TestTransformCommands:
    """Test text transformation commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create a test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'test_transform',
            '--type', 'substitution',
            '--complexity', '3',
            '--seed', '42'
        ])
    
    def test_transform_text(self):
        """Test transforming text with a language."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'transform', 'text',
            'Hello world',
            'test_transform'
        ])
        
        assert result.exit_code == 0
        assert 'Original:' in result.output
        assert 'Translated:' in result.output
    
    def test_transform_text_json_output(self):
        """Test transforming text with JSON output."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            '--output-format', 'json',
            'transform', 'text',
            'Test sentence',
            'test_transform'
        ])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        data = json.loads(result.output)
        assert 'original' in data
        assert 'translated' in data
        assert 'language' in data
        assert data['original'] == 'Test sentence'
        assert data['language'] == 'test_transform'
    
    def test_proper_noun_replacement(self):
        """Test proper noun replacement."""
        result = self.runner.invoke(cli, [
            'transform', 'proper-nouns',
            'John Smith went to New York',
            '--strategy', 'random',
            '--seed', '123'
        ])
        
        assert result.exit_code == 0
        assert 'Original:' in result.output
        assert 'Transformed:' in result.output
    
    def test_synonym_replacement(self):
        """Test synonym replacement."""
        result = self.runner.invoke(cli, [
            'transform', 'synonyms',
            'The big dog ran fast',
            '--replacement-rate', '0.5',
            '--seed', '456'
        ])
        
        assert result.exit_code == 0
        assert 'Original:' in result.output
        assert 'Transformed:' in result.output
    
    def test_synonym_replacement_invalid_rate(self):
        """Test synonym replacement with invalid rate."""
        result = self.runner.invoke(cli, [
            'transform', 'synonyms',
            'Test text',
            '--replacement-rate', '1.5'
        ])
        
        assert result.exit_code == 1
        assert 'between 0.0 and 1.0' in result.output


class TestBatchCommands:
    """Test batch processing commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create sample benchmark data
        self.benchmark_data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "Geography"
            },
            {
                "question": "How many legs does a spider have?",
                "answer": "8",
                "category": "Biology"
            }
        ]
        
        # Save test data
        benchmark_dir = self.data_dir / 'benchmarks'
        benchmark_dir.mkdir(parents=True)
        self.benchmark_file = benchmark_dir / 'test_data.json'
        
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.benchmark_data, f)
    
    def test_extract_vocabulary(self):
        """Test vocabulary extraction from benchmark file."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'batch', 'extract-vocab',
            str(self.benchmark_file),
            '--min-freq', '1',
            '--max-words', '100'
        ])
        
        assert result.exit_code == 0
        assert 'Vocabulary extracted:' in result.output
        
        # Check that vocabulary file was created
        vocab_files = list((self.data_dir / 'results').glob('*vocabulary.json'))
        assert len(vocab_files) > 0
    
    def test_transform_batch(self):
        """Test batch transformation of problems."""
        # First create a test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'batch_test',
            '--type', 'substitution',
            '--complexity', '2',
            '--seed', '42'
        ])
        
        # Transform batch
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'batch', 'transform',
            str(self.benchmark_file),
            'batch_test',
            '--batch-size', '10'
        ])
        
        assert result.exit_code == 0
        assert 'Transformed' in result.output
        
        # Check that output file was created
        output_files = list((self.data_dir / 'results').glob('*transformed.json'))
        assert len(output_files) > 0


class TestUtilityCommands:
    """Test utility commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create test language
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'util_test',
            '--type', 'phonetic',
            '--complexity', '4',
            '--seed', '42'
        ])
    
    def test_language_stats(self):
        """Test getting language statistics."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'util', 'stats', 'util_test'
        ])
        
        assert result.exit_code == 0
        assert 'Language Statistics:' in result.output
        assert 'util_test' in result.output
    
    def test_export_rules_json(self):
        """Test exporting rules as JSON."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'util', 'export-rules', 'util_test',
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert 'Rules exported' in result.output
        
        # Check that rules file was created
        rules_files = list((self.data_dir / 'results').glob('*rules.json'))
        assert len(rules_files) > 0
    
    def test_validate_transformation(self):
        """Test validating a transformation."""
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'util', 'validate', 'util_test',
            'This is a test sentence'
        ])
        
        assert result.exit_code == 0
        assert 'Original:' in result.output
        assert 'Translated:' in result.output


class TestGlobalOptions:
    """Test global CLI options."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
    
    def test_verbose_mode(self):
        """Test verbose output mode."""
        result = self.runner.invoke(cli, [
            '--verbose',
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'verbose_test',
            '--type', 'substitution',
            '--complexity', '2'
        ])
        
        assert result.exit_code == 0
        # In verbose mode, should see data directory info
        assert str(self.data_dir) in result.output
    
    def test_quiet_mode(self):
        """Test quiet output mode."""
        result = self.runner.invoke(cli, [
            '--quiet',
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'quiet_test',
            '--type', 'substitution',
            '--complexity', '2'
        ])
        
        assert result.exit_code == 0
        # In quiet mode, output should be minimal
        assert len(result.output.strip()) < 100
    
    def test_json_output_format(self):
        """Test JSON output format."""
        # Generate a language first
        self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            'language', 'generate', 'json_output_test',
            '--type', 'substitution'
        ])
        
        # List languages in JSON format
        result = self.runner.invoke(cli, [
            '--data-dir', str(self.data_dir),
            '--output-format', 'json',
            'language', 'list'
        ])
        
        assert result.exit_code == 0
        
        # Should be valid JSON
        data = json.loads(result.output)
        assert 'languages' in data
    
    def test_help_commands(self):
        """Test help output for various commands."""
        # Main help
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'ScrambleBench CLI' in result.output
        
        # Language command help
        result = self.runner.invoke(cli, ['language', '--help'])
        assert result.exit_code == 0
        assert 'Language generation' in result.output
        
        # Transform command help
        result = self.runner.invoke(cli, ['transform', '--help'])
        assert result.exit_code == 0
        assert 'Text transformation' in result.output