"""
Comprehensive CLI interface for the ScrambleBench language generation system.

This module provides command-line interface capabilities for:
- Language generation and management
- Batch processing of benchmark questions
- Text transformations using various strategies
- Utility functions for language analysis and export
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.text import Text
from rich import print as rprint
import yaml

from scramblebench.translation.language_generator import (
    LanguageGenerator,
    LanguageType,
    ConstructedLanguage
)
from scramblebench.translation.translator import (
    ProblemTranslator,
    TranslatedProblem
)
from scramblebench.translation.text_transformer import (
    TextTransformer,
    ProperNounSwapper,
    SynonymReplacer,
    VocabularyExtractor
)
from scramblebench.utils.data_loader import DataLoader

# Initialize console for rich output
console = Console()

# Global CLI context
class CLIContext:
    """Shared context for CLI commands."""
    
    def __init__(self):
        self.verbose = False
        self.quiet = False
        self.output_format = "text"
        self.data_dir = Path("data")
        self.languages_dir = Path("data/languages")
        self.results_dir = Path("data/results")
        
        # Ensure directories exist
        self.languages_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

pass_context = click.make_pass_decorator(CLIContext, ensure=True)


def echo_success(message: str, ctx: CLIContext):
    """Echo success message with appropriate formatting."""
    if not ctx.quiet:
        if ctx.output_format == "json":
            click.echo(json.dumps({"status": "success", "message": message}))
        else:
            rprint(f"[green]✓[/green] {message}")


def echo_error(message: str, ctx: CLIContext):
    """Echo error message with appropriate formatting."""
    if ctx.output_format == "json":
        click.echo(json.dumps({"status": "error", "message": message}))
    else:
        rprint(f"[red]✗[/red] {message}")


def echo_info(message: str, ctx: CLIContext):
    """Echo info message with appropriate formatting."""
    if not ctx.quiet and ctx.verbose:
        if ctx.output_format == "json":
            click.echo(json.dumps({"status": "info", "message": message}))
        else:
            rprint(f"[blue]ℹ[/blue] {message}")


def validate_language_type(ctx, param, value):
    """Validate language type parameter."""
    if value:
        try:
            return LanguageType(value)
        except ValueError:
            valid_types = [t.value for t in LanguageType]
            raise click.BadParameter(
                f"Invalid language type. Must be one of: {', '.join(valid_types)}"
            )
    return value


def validate_complexity(ctx, param, value):
    """Validate complexity parameter."""
    if value is not None and not (1 <= value <= 10):
        raise click.BadParameter("Complexity must be between 1 and 10")
    return value


@click.group()
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Suppress non-essential output'
)
@click.option(
    '--output-format',
    type=click.Choice(['text', 'json', 'yaml']),
    default='text',
    help='Output format for results'
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("data"),
    help='Base data directory'
)
@pass_context
def cli(ctx: CLIContext, verbose: bool, quiet: bool, output_format: str, data_dir: Path):
    """
    ScrambleBench CLI - Language generation and transformation toolkit.
    
    Generate constructed languages, transform benchmark problems,
    and perform batch processing operations.
    """
    ctx.verbose = verbose
    ctx.quiet = quiet
    ctx.output_format = output_format
    ctx.data_dir = data_dir
    ctx.languages_dir = data_dir / "languages"
    ctx.results_dir = data_dir / "results"
    
    # Ensure directories exist
    ctx.languages_dir.mkdir(parents=True, exist_ok=True)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose and not quiet:
        echo_info(f"Using data directory: {data_dir}", ctx)


# Language Generation Commands
@cli.group(name='language')
def language_group():
    """Language generation and management commands."""
    pass


@language_group.command(name='generate')
@click.argument('name')
@click.option(
    '--type', 'lang_type',
    type=click.Choice([t.value for t in LanguageType]),
    required=True,
    callback=validate_language_type,
    help='Type of language to generate'
)
@click.option(
    '--complexity',
    type=int,
    default=5,
    callback=validate_complexity,
    help='Complexity level (1-10, default: 5)'
)
@click.option(
    '--vocab-size',
    type=int,
    default=1000,
    help='Vocabulary size (default: 1000)'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@click.option(
    '--save/--no-save',
    default=True,
    help='Save generated language to file'
)
@pass_context
def generate_language(
    ctx: CLIContext,
    name: str,
    lang_type: LanguageType,
    complexity: int,
    vocab_size: int,
    seed: Optional[int],
    save: bool
):
    """Generate a new constructed language."""
    try:
        echo_info(f"Generating {lang_type.value} language '{name}'...", ctx)
        
        generator = LanguageGenerator(seed=seed)
        language = generator.generate_language(
            name=name,
            language_type=lang_type,
            complexity=complexity,
            vocab_size=vocab_size
        )
        
        if save:
            output_path = ctx.languages_dir / f"{name}.json"
            generator.save_language(language, output_path)
            echo_success(f"Language saved to {output_path}", ctx)
        
        # Output language information
        if ctx.output_format == "json":
            output = {
                "name": language.name,
                "type": language.language_type.value,
                "complexity": complexity,
                "vocab_size": len(language.vocabulary),
                "rules_count": len(language.rules),
                "metadata": language.metadata
            }
            click.echo(json.dumps(output, indent=2))
        elif ctx.output_format == "yaml":
            output = {
                "name": language.name,
                "type": language.language_type.value,
                "complexity": complexity,
                "vocab_size": len(language.vocabulary),
                "rules_count": len(language.rules),
                "metadata": language.metadata
            }
            click.echo(yaml.dump(output, default_flow_style=False))
        else:
            if not ctx.quiet:
                table = Table(title=f"Generated Language: {name}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")
                
                table.add_row("Name", language.name)
                table.add_row("Type", language.language_type.value)
                table.add_row("Complexity", str(complexity))
                table.add_row("Vocabulary Size", str(len(language.vocabulary)))
                table.add_row("Rules Count", str(len(language.rules)))
                
                console.print(table)
        
    except Exception as e:
        echo_error(f"Failed to generate language: {e}", ctx)
        sys.exit(1)


@language_group.command(name='list')
@click.option(
    '--format', 'list_format',
    type=click.Choice(['table', 'simple', 'json', 'yaml']),
    help='Output format (overrides global format)'
)
@pass_context
def list_languages(ctx: CLIContext, list_format: Optional[str]):
    """List all available languages."""
    try:
        # Scan languages directory
        language_files = list(ctx.languages_dir.glob("*.json"))
        
        if not language_files:
            if ctx.output_format == "json" or list_format == "json":
                click.echo(json.dumps({"languages": []}))
            elif ctx.output_format == "yaml" or list_format == "yaml":
                click.echo(yaml.dump({"languages": []}))
            else:
                echo_info("No languages found", ctx)
            return
        
        languages = []
        for lang_file in language_files:
            try:
                generator = LanguageGenerator()
                language = generator.load_language(lang_file)
                languages.append({
                    "name": language.name,
                    "type": language.language_type.value,
                    "vocab_size": len(language.vocabulary),
                    "rules_count": len(language.rules),
                    "file": str(lang_file.name),
                    "complexity": language.metadata.get("complexity", "unknown")
                })
            except Exception as e:
                echo_error(f"Failed to load {lang_file}: {e}", ctx)
        
        # Output results
        output_fmt = list_format or ctx.output_format
        
        if output_fmt == "json":
            click.echo(json.dumps({"languages": languages}, indent=2))
        elif output_fmt == "yaml":
            click.echo(yaml.dump({"languages": languages}, default_flow_style=False))
        elif output_fmt == "simple":
            for lang in languages:
                click.echo(f"{lang['name']} ({lang['type']}) - {lang['vocab_size']} words")
        else:
            # Table format
            table = Table(title="Available Languages")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Complexity", style="green")
            table.add_column("Vocab Size", style="blue")
            table.add_column("Rules", style="magenta")
            table.add_column("File", style="white")
            
            for lang in languages:
                table.add_row(
                    lang["name"],
                    lang["type"],
                    str(lang["complexity"]),
                    str(lang["vocab_size"]),
                    str(lang["rules_count"]),
                    lang["file"]
                )
            
            console.print(table)
            
    except Exception as e:
        echo_error(f"Failed to list languages: {e}", ctx)
        sys.exit(1)


@language_group.command(name='show')
@click.argument('name')
@click.option(
    '--show-rules',
    is_flag=True,
    help='Show language rules in detail'
)
@click.option(
    '--show-vocabulary',
    is_flag=True,
    help='Show vocabulary mappings'
)
@click.option(
    '--limit',
    type=int,
    default=20,
    help='Limit number of rules/vocabulary items to show'
)
@pass_context
def show_language(
    ctx: CLIContext,
    name: str,
    show_rules: bool,
    show_vocabulary: bool,
    limit: int
):
    """Show detailed information about a language."""
    try:
        lang_file = ctx.languages_dir / f"{name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        if ctx.output_format == "json":
            output = {
                "name": language.name,
                "type": language.language_type.value,
                "metadata": language.metadata,
                "vocab_size": len(language.vocabulary),
                "rules_count": len(language.rules)
            }
            
            if show_rules:
                output["rules"] = [
                    {
                        "source": rule.source,
                        "target": rule.target,
                        "type": rule.rule_type,
                        "priority": rule.priority
                    }
                    for rule in language.rules[:limit]
                ]
            
            if show_vocabulary:
                vocab_items = list(language.vocabulary.items())[:limit]
                output["vocabulary"] = dict(vocab_items)
            
            click.echo(json.dumps(output, indent=2))
            
        elif ctx.output_format == "yaml":
            output = {
                "name": language.name,
                "type": language.language_type.value,
                "metadata": language.metadata,
                "vocab_size": len(language.vocabulary),
                "rules_count": len(language.rules)
            }
            
            if show_rules:
                output["rules"] = [
                    {
                        "source": rule.source,
                        "target": rule.target,
                        "type": rule.rule_type,
                        "priority": rule.priority
                    }
                    for rule in language.rules[:limit]
                ]
            
            if show_vocabulary:
                vocab_items = list(language.vocabulary.items())[:limit]
                output["vocabulary"] = dict(vocab_items)
            
            click.echo(yaml.dump(output, default_flow_style=False))
            
        else:
            # Rich table output
            table = Table(title=f"Language Details: {language.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Name", language.name)
            table.add_row("Type", language.language_type.value)
            table.add_row("Vocabulary Size", str(len(language.vocabulary)))
            table.add_row("Rules Count", str(len(language.rules)))
            
            for key, value in language.metadata.items():
                table.add_row(f"Meta: {key}", str(value))
            
            console.print(table)
            
            if show_rules:
                rules_table = Table(title="Language Rules (Sample)")
                rules_table.add_column("Source", style="yellow")
                rules_table.add_column("Target", style="green")
                rules_table.add_column("Type", style="blue")
                rules_table.add_column("Priority", style="magenta")
                
                for rule in language.rules[:limit]:
                    rules_table.add_row(
                        rule.source,
                        rule.target,
                        rule.rule_type,
                        str(rule.priority)
                    )
                
                console.print(rules_table)
            
            if show_vocabulary:
                vocab_table = Table(title="Vocabulary Mappings (Sample)")
                vocab_table.add_column("Original", style="yellow")
                vocab_table.add_column("Translated", style="green")
                
                vocab_items = list(language.vocabulary.items())[:limit]
                for original, translated in vocab_items:
                    vocab_table.add_row(original, translated)
                
                console.print(vocab_table)
        
    except Exception as e:
        echo_error(f"Failed to show language: {e}", ctx)
        sys.exit(1)


@language_group.command(name='delete')
@click.argument('name')
@click.option(
    '--force',
    is_flag=True,
    help='Force deletion without confirmation'
)
@pass_context
def delete_language(ctx: CLIContext, name: str, force: bool):
    """Delete a language file."""
    try:
        lang_file = ctx.languages_dir / f"{name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{name}' not found", ctx)
            sys.exit(1)
        
        if not force:
            if not click.confirm(f"Delete language '{name}'?"):
                echo_info("Deletion cancelled", ctx)
                return
        
        lang_file.unlink()
        echo_success(f"Language '{name}' deleted", ctx)
        
    except Exception as e:
        echo_error(f"Failed to delete language: {e}", ctx)
        sys.exit(1)


# Batch Processing Commands
@cli.group(name='batch')
def batch_group():
    """Batch processing commands for benchmark data."""
    pass


@batch_group.command(name='extract-vocab')
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for vocabulary (default: auto-generated)'
)
@click.option(
    '--min-freq',
    type=int,
    default=2,
    help='Minimum word frequency to include'
)
@click.option(
    '--max-words',
    type=int,
    default=5000,
    help='Maximum number of words to extract'
)
@pass_context
def extract_vocabulary(
    ctx: CLIContext,
    input_file: Path,
    output: Optional[Path],
    min_freq: int,
    max_words: int
):
    """Extract vocabulary from benchmark question files."""
    try:
        echo_info(f"Extracting vocabulary from {input_file}...", ctx)
        
        # Load benchmark data
        data_loader = DataLoader()
        questions = data_loader.load_benchmark_file(input_file)
        
        # Extract vocabulary
        extractor = VocabularyExtractor(
            min_frequency=min_freq,
            max_vocabulary_size=max_words
        )
        
        vocabulary = extractor.extract_from_problems(questions)
        
        # Generate output filename if not provided
        if not output:
            output = ctx.results_dir / f"{input_file.stem}_vocabulary.json"
        
        # Save vocabulary
        with open(output, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        echo_success(f"Vocabulary extracted: {len(vocabulary['words'])} words saved to {output}", ctx)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "extracted_words": len(vocabulary['words']),
                "output_file": str(output),
                "statistics": vocabulary.get('statistics', {})
            }))
        elif not ctx.quiet:
            table = Table(title="Vocabulary Extraction Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Words Extracted", str(len(vocabulary['words'])))
            table.add_row("Output File", str(output))
            table.add_row("Min Frequency", str(min_freq))
            table.add_row("Max Words", str(max_words))
            
            console.print(table)
            
    except Exception as e:
        echo_error(f"Failed to extract vocabulary: {e}", ctx)
        sys.exit(1)


@batch_group.command(name='transform')
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('language_name')
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for transformed problems'
)
@click.option(
    '--preserve-numbers/--transform-numbers',
    default=True,
    help='Whether to preserve numeric values'
)
@click.option(
    '--preserve-proper-nouns/--transform-proper-nouns',
    default=True,
    help='Whether to preserve proper nouns'
)
@click.option(
    '--batch-size',
    type=int,
    default=100,
    help='Batch size for processing'
)
@pass_context
def transform_batch(
    ctx: CLIContext,
    input_file: Path,
    language_name: str,
    output: Optional[Path],
    preserve_numbers: bool,
    preserve_proper_nouns: bool,
    batch_size: int
):
    """Transform a batch of problems using a constructed language."""
    try:
        # Load language
        lang_file = ctx.languages_dir / f"{language_name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{language_name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        echo_info(f"Loading problems from {input_file}...", ctx)
        
        # Load benchmark data
        data_loader = DataLoader()
        problems = data_loader.load_benchmark_file(input_file)
        
        echo_info(f"Transforming {len(problems)} problems...", ctx)
        
        # Initialize translator
        translator = ProblemTranslator()
        
        # Process in batches with progress bar
        transformed_problems = []
        for i in track(range(0, len(problems), batch_size), description="Processing batches..."):
            batch = problems[i:i + batch_size]
            
            for problem in batch:
                translated = translator.translate_problem(
                    problem,
                    language,
                    preserve_numbers=preserve_numbers,
                    preserve_proper_nouns=preserve_proper_nouns
                )
                transformed_problems.append({
                    'original': translated.original_problem,
                    'translated': translated.translated_problem,
                    'translation_key': translated.translation_key,
                    'metadata': translated.metadata
                })
        
        # Generate output filename if not provided
        if not output:
            output = ctx.results_dir / f"{input_file.stem}_{language_name}_transformed.json"
        
        # Save results
        with open(output, 'w') as f:
            json.dump({
                'language': language_name,
                'source_file': str(input_file),
                'transformation_settings': {
                    'preserve_numbers': preserve_numbers,
                    'preserve_proper_nouns': preserve_proper_nouns
                },
                'problems': transformed_problems
            }, f, indent=2)
        
        echo_success(f"Transformed {len(transformed_problems)} problems, saved to {output}", ctx)
        
    except Exception as e:
        echo_error(f"Failed to transform batch: {e}", ctx)
        sys.exit(1)


# Text Transformation Commands
@cli.group(name='transform')
def transform_group():
    """Text transformation commands."""
    pass


@transform_group.command(name='text')
@click.argument('text')
@click.argument('language_name')
@click.option(
    '--preserve-numbers/--transform-numbers',
    default=True,
    help='Whether to preserve numeric values'
)
@click.option(
    '--preserve-proper-nouns/--transform-proper-nouns',
    default=True,
    help='Whether to preserve proper nouns'
)
@pass_context
def transform_text(
    ctx: CLIContext,
    text: str,
    language_name: str,
    preserve_numbers: bool,
    preserve_proper_nouns: bool
):
    """Transform a single text string using a constructed language."""
    try:
        # Load language
        lang_file = ctx.languages_dir / f"{language_name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{language_name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        # Transform text
        translator = ProblemTranslator()
        problem = {"text": text}
        
        translated = translator.translate_problem(
            problem,
            language,
            preserve_numbers=preserve_numbers,
            preserve_proper_nouns=preserve_proper_nouns
        )
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "original": text,
                "translated": translated.translated_problem["text"],
                "translation_key": translated.translation_key,
                "language": language_name
            }, indent=2))
        else:
            if not ctx.quiet:
                rprint(f"[yellow]Original:[/yellow] {text}")
                rprint(f"[green]Translated:[/green] {translated.translated_problem['text']}")
            else:
                click.echo(translated.translated_problem["text"])
                
    except Exception as e:
        echo_error(f"Failed to transform text: {e}", ctx)
        sys.exit(1)


@transform_group.command(name='proper-nouns')
@click.argument('text')
@click.option(
    '--strategy',
    type=click.Choice(['random', 'thematic', 'phonetic']),
    default='random',
    help='Strategy for replacing proper nouns'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@pass_context
def swap_proper_nouns(
    ctx: CLIContext,
    text: str,
    strategy: str,
    seed: Optional[int]
):
    """Swap proper nouns in text with alternatives."""
    try:
        swapper = ProperNounSwapper(strategy=strategy, seed=seed)
        result = swapper.transform_text(text)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "original": text,
                "transformed": result.transformed_text,
                "replacements": result.replacements,
                "strategy": strategy
            }, indent=2))
        else:
            if not ctx.quiet:
                rprint(f"[yellow]Original:[/yellow] {text}")
                rprint(f"[green]Transformed:[/green] {result.transformed_text}")
                
                if result.replacements:
                    table = Table(title="Replacements")
                    table.add_column("Original", style="yellow")
                    table.add_column("Replacement", style="green")
                    
                    for original, replacement in result.replacements.items():
                        table.add_row(original, replacement)
                    
                    console.print(table)
            else:
                click.echo(result.transformed_text)
                
    except Exception as e:
        echo_error(f"Failed to swap proper nouns: {e}", ctx)
        sys.exit(1)


@transform_group.command(name='synonyms')
@click.argument('text')
@click.option(
    '--replacement-rate',
    type=float,
    default=0.3,
    help='Proportion of words to replace (0.0-1.0)'
)
@click.option(
    '--preserve-function-words',
    is_flag=True,
    default=True,
    help='Preserve function words (articles, prepositions, etc.)'
)
@click.option(
    '--seed',
    type=int,
    help='Random seed for reproducibility'
)
@pass_context
def replace_synonyms(
    ctx: CLIContext,
    text: str,
    replacement_rate: float,
    preserve_function_words: bool,
    seed: Optional[int]
):
    """Replace words with synonyms."""
    try:
        if not (0.0 <= replacement_rate <= 1.0):
            echo_error("Replacement rate must be between 0.0 and 1.0", ctx)
            sys.exit(1)
        
        replacer = SynonymReplacer(
            replacement_rate=replacement_rate,
            preserve_function_words=preserve_function_words,
            seed=seed
        )
        result = replacer.transform_text(text)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "original": text,
                "transformed": result.transformed_text,
                "replacements": result.replacements,
                "replacement_rate": replacement_rate
            }, indent=2))
        else:
            if not ctx.quiet:
                rprint(f"[yellow]Original:[/yellow] {text}")
                rprint(f"[green]Transformed:[/green] {result.transformed_text}")
                
                if result.replacements:
                    table = Table(title="Synonym Replacements")
                    table.add_column("Original", style="yellow")
                    table.add_column("Synonym", style="green")
                    
                    for original, synonym in result.replacements.items():
                        table.add_row(original, synonym)
                    
                    console.print(table)
            else:
                click.echo(result.transformed_text)
                
    except Exception as e:
        echo_error(f"Failed to replace synonyms: {e}", ctx)
        sys.exit(1)


# Utility Commands
@cli.group(name='util')
def util_group():
    """Utility commands for analysis and export."""
    pass


@util_group.command(name='stats')
@click.argument('language_name')
@pass_context
def language_stats(ctx: CLIContext, language_name: str):
    """Show detailed statistics for a language."""
    try:
        lang_file = ctx.languages_dir / f"{language_name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{language_name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        # Calculate statistics
        rule_types = {}
        for rule in language.rules:
            rule_types[rule.rule_type] = rule_types.get(rule.rule_type, 0) + 1
        
        vocab_length_dist = {}
        for word in language.vocabulary.keys():
            length = len(word)
            vocab_length_dist[length] = vocab_length_dist.get(length, 0) + 1
        
        stats = {
            "name": language.name,
            "type": language.language_type.value,
            "total_rules": len(language.rules),
            "rule_types": rule_types,
            "vocabulary_size": len(language.vocabulary),
            "vocabulary_length_distribution": vocab_length_dist,
            "metadata": language.metadata
        }
        
        if ctx.output_format == "json":
            click.echo(json.dumps(stats, indent=2))
        elif ctx.output_format == "yaml":
            click.echo(yaml.dump(stats, default_flow_style=False))
        else:
            table = Table(title=f"Language Statistics: {language.name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Language Name", language.name)
            table.add_row("Language Type", language.language_type.value)
            table.add_row("Total Rules", str(len(language.rules)))
            table.add_row("Vocabulary Size", str(len(language.vocabulary)))
            
            console.print(table)
            
            # Rule types table
            if rule_types:
                rule_table = Table(title="Rule Types Distribution")
                rule_table.add_column("Rule Type", style="yellow")
                rule_table.add_column("Count", style="green")
                
                for rule_type, count in rule_types.items():
                    rule_table.add_row(rule_type, str(count))
                
                console.print(rule_table)
            
            # Vocabulary length distribution
            if vocab_length_dist:
                length_table = Table(title="Vocabulary Length Distribution")
                length_table.add_column("Word Length", style="blue")
                length_table.add_column("Count", style="green")
                
                for length in sorted(vocab_length_dist.keys()):
                    length_table.add_row(str(length), str(vocab_length_dist[length]))
                
                console.print(length_table)
        
    except Exception as e:
        echo_error(f"Failed to get language statistics: {e}", ctx)
        sys.exit(1)


@util_group.command(name='export-rules')
@click.argument('language_name')
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for rules (default: auto-generated)'
)
@click.option(
    '--format', 'export_format',
    type=click.Choice(['json', 'yaml', 'csv']),
    default='json',
    help='Export format'
)
@pass_context
def export_rules(
    ctx: CLIContext,
    language_name: str,
    output: Optional[Path],
    export_format: str
):
    """Export language rules to a file."""
    try:
        lang_file = ctx.languages_dir / f"{language_name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{language_name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        # Generate output filename if not provided
        if not output:
            output = ctx.results_dir / f"{language_name}_rules.{export_format}"
        
        rules_data = []
        for rule in language.rules:
            rules_data.append({
                "source": rule.source,
                "target": rule.target,
                "type": rule.rule_type,
                "priority": rule.priority,
                "conditions": rule.conditions
            })
        
        # Export in specified format
        if export_format == "json":
            with open(output, 'w') as f:
                json.dump({
                    "language": language_name,
                    "rules": rules_data
                }, f, indent=2)
        elif export_format == "yaml":
            with open(output, 'w') as f:
                yaml.dump({
                    "language": language_name,
                    "rules": rules_data
                }, f, default_flow_style=False)
        elif export_format == "csv":
            import csv
            with open(output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["source", "target", "type", "priority"])
                writer.writeheader()
                for rule_data in rules_data:
                    writer.writerow({
                        "source": rule_data["source"],
                        "target": rule_data["target"],
                        "type": rule_data["type"],
                        "priority": rule_data["priority"]
                    })
        
        echo_success(f"Rules exported to {output}", ctx)
        
    except Exception as e:
        echo_error(f"Failed to export rules: {e}", ctx)
        sys.exit(1)


@util_group.command(name='validate')
@click.argument('language_name')
@click.argument('text')
@pass_context
def validate_transformation(ctx: CLIContext, language_name: str, text: str):
    """Validate that a transformation can be reversed."""
    try:
        # Load language
        lang_file = ctx.languages_dir / f"{language_name}.json"
        if not lang_file.exists():
            echo_error(f"Language '{language_name}' not found", ctx)
            sys.exit(1)
        
        generator = LanguageGenerator()
        language = generator.load_language(lang_file)
        
        # Transform text
        translator = ProblemTranslator()
        problem = {"text": text}
        
        translated = translator.translate_problem(problem, language)
        
        # Verify translation consistency
        verification = translator.verify_translation_consistency(translated)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "original": text,
                "translated": translated.translated_problem["text"],
                "verification": verification
            }, indent=2))
        else:
            rprint(f"[yellow]Original:[/yellow] {text}")
            rprint(f"[green]Translated:[/green] {translated.translated_problem['text']}")
            
            if verification["consistent"]:
                rprint("[green]✓ Translation is consistent[/green]")
            else:
                rprint("[red]✗ Translation has issues:[/red]")
                for issue in verification["issues"]:
                    rprint(f"  - {issue}")
            
            # Show statistics
            stats_table = Table(title="Validation Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in verification["statistics"].items():
                stats_table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(stats_table)
        
    except Exception as e:
        echo_error(f"Failed to validate transformation: {e}", ctx)
        sys.exit(1)


# Evaluation Commands
@cli.group(name='evaluate')
def evaluate_group():
    """Evaluation pipeline commands for comprehensive LLM benchmarking."""
    pass


@evaluate_group.command(name='run')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file for evaluation'
)
@click.option(
    '--models',
    type=str,
    help='Comma-separated list of model names (for quick setup)'
)
@click.option(
    '--benchmarks',
    type=str,
    help='Comma-separated list of benchmark file paths (for quick setup)'
)
@click.option(
    '--experiment-name',
    type=str,
    help='Name of the experiment (for quick setup)'
)
@click.option(
    '--transformations',
    type=str,
    help='Comma-separated list of transformation types (for quick setup)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default="results",
    help='Output directory for results'
)
@click.option(
    '--max-samples',
    type=int,
    help='Maximum number of samples per benchmark'
)
@click.option(
    '--no-plots',
    is_flag=True,
    help='Skip plot generation'
)
@click.option(
    '--include-original/--no-original',
    default=True,
    help='Include evaluation of original (untransformed) problems'
)
@pass_context
def run_evaluation(
    ctx: CLIContext,
    config: Optional[Path],
    models: Optional[str],
    benchmarks: Optional[str],
    experiment_name: Optional[str],
    transformations: Optional[str],
    output_dir: Path,
    max_samples: Optional[int],
    no_plots: bool,
    include_original: bool
):
    """Run a comprehensive evaluation experiment."""
    import asyncio
    
    try:
        if config:
            # Load from configuration file
            from scramblebench.evaluation import EvaluationRunner
            
            echo_info(f"Loading configuration from {config}", ctx)
            runner = EvaluationRunner.from_config_file(config, ctx.data_dir)
            
        elif models and benchmarks and experiment_name:
            # Quick setup mode
            from scramblebench.evaluation import run_quick_evaluation
            
            model_list = [m.strip() for m in models.split(',')]
            benchmark_list = [Path(b.strip()) for b in benchmarks.split(',')]
            transform_list = [t.strip() for t in transformations.split(',')] if transformations else None
            
            echo_info(f"Quick setup: {len(model_list)} models, {len(benchmark_list)} benchmarks", ctx)
            
            results = asyncio.run(run_quick_evaluation(
                benchmark_paths=benchmark_list,
                models=model_list,
                experiment_name=experiment_name,
                output_dir=str(output_dir),
                transformations=transform_list
            ))
            
            echo_success(f"Evaluation completed: {experiment_name}", ctx)
            
            if ctx.output_format == "json":
                click.echo(json.dumps({
                    "experiment_name": experiment_name,
                    "total_results": len(results.results),
                    "successful_results": sum(1 for r in results.results if r.success),
                    "output_dir": str(output_dir / experiment_name)
                }))
            
            return
            
        else:
            echo_error("Either provide --config file or --models, --benchmarks, and --experiment-name", ctx)
            sys.exit(1)
        
        # Run evaluation
        echo_info("Starting evaluation...", ctx)
        results = asyncio.run(runner.run_evaluation(
            include_original=include_original,
            save_intermediate=True
        ))
        
        echo_success(f"Evaluation completed: {runner.config.experiment_name}", ctx)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "experiment_name": runner.config.experiment_name,
                "total_results": len(results.results),
                "successful_results": sum(1 for r in results.results if r.success),
                "output_dir": str(runner.config.get_experiment_dir())
            }))
        elif not ctx.quiet:
            table = Table(title="Evaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Experiment", runner.config.experiment_name)
            table.add_row("Total Results", str(len(results.results)))
            table.add_row("Successful", str(sum(1 for r in results.results if r.success)))
            table.add_row("Output Directory", str(runner.config.get_experiment_dir()))
            
            console.print(table)
        
    except Exception as e:
        echo_error(f"Evaluation failed: {e}", ctx)
        sys.exit(1)


@evaluate_group.command(name='config')
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option(
    '--template',
    type=click.Choice(['basic', 'comprehensive', 'robustness']),
    default='basic',
    help='Configuration template to generate'
)
@pass_context
def create_config(ctx: CLIContext, output_path: Path, template: str):
    """Create a sample evaluation configuration file."""
    try:
        from scramblebench.evaluation.runner import create_sample_config
        from scramblebench.evaluation.config import (
            EvaluationConfig, ModelConfig, TransformationConfig, 
            ModelProvider, TransformationType, EvaluationMode
        )
        
        if template == 'basic':
            config = EvaluationConfig(
                experiment_name="basic_evaluation",
                description="Basic evaluation with core transformations",
                benchmark_paths=["data/benchmarks/sample.json"],
                models=[
                    ModelConfig(
                        name="anthropic/claude-3-sonnet",
                        provider=ModelProvider.OPENROUTER,
                        temperature=0.0
                    )
                ],
                transformations=TransformationConfig(
                    enabled_types=[TransformationType.LANGUAGE_TRANSLATION]
                ),
                max_samples=50
            )
        elif template == 'comprehensive':
            config = EvaluationConfig(
                experiment_name="comprehensive_evaluation",
                description="Comprehensive evaluation with all transformations",
                mode=EvaluationMode.COMPREHENSIVE,
                benchmark_paths=["data/benchmarks/collected/*/easy/*.json"],
                models=[
                    ModelConfig(name="anthropic/claude-3-sonnet", provider=ModelProvider.OPENROUTER),
                    ModelConfig(name="openai/gpt-4", provider=ModelProvider.OPENROUTER),
                    ModelConfig(name="meta-llama/llama-2-70b-chat", provider=ModelProvider.OPENROUTER)
                ],
                transformations=TransformationConfig(
                    enabled_types=[TransformationType.ALL]
                ),
                max_samples=200,
                generate_plots=True,
                calculate_significance=True
            )
        else:  # robustness
            config = EvaluationConfig(
                experiment_name="robustness_evaluation",
                description="Robustness-focused evaluation",
                mode=EvaluationMode.ROBUSTNESS,
                benchmark_paths=["data/benchmarks/collected/*/medium/*.json"],
                models=[
                    ModelConfig(name="anthropic/claude-3-sonnet", provider=ModelProvider.OPENROUTER),
                    ModelConfig(name="openai/gpt-4", provider=ModelProvider.OPENROUTER)
                ],
                transformations=TransformationConfig(
                    enabled_types=[
                        TransformationType.LANGUAGE_TRANSLATION,
                        TransformationType.PROPER_NOUN_SWAP,
                        TransformationType.SYNONYM_REPLACEMENT
                    ],
                    synonym_rate=0.5
                ),
                max_samples=100
            )
        
        config.save_to_file(output_path)
        echo_success(f"Configuration template '{template}' saved to {output_path}", ctx)
        
    except Exception as e:
        echo_error(f"Failed to create configuration: {e}", ctx)
        sys.exit(1)


@evaluate_group.command(name='list')
@click.option(
    '--format', 'list_format',
    type=click.Choice(['table', 'simple', 'json']),
    help='Output format (overrides global format)'
)
@pass_context
def list_experiments(ctx: CLIContext, list_format: Optional[str]):
    """List all evaluation experiments."""
    try:
        from scramblebench.evaluation import ResultsManager
        
        results_manager = ResultsManager(ctx.results_dir)
        experiments = results_manager.list_experiments()
        
        if not experiments:
            echo_info("No experiments found", ctx)
            return
        
        output_fmt = list_format or ctx.output_format
        
        if output_fmt == "json":
            click.echo(json.dumps({"experiments": experiments}))
        elif output_fmt == "simple":
            for exp in experiments:
                click.echo(exp)
        else:
            # Table format
            table = Table(title="Evaluation Experiments")
            table.add_column("Experiment", style="cyan")
            table.add_column("Status", style="green")
            
            for exp in experiments:
                table.add_row(exp, "Complete")
            
            console.print(table)
        
    except Exception as e:
        echo_error(f"Failed to list experiments: {e}", ctx)
        sys.exit(1)


@evaluate_group.command(name='analyze')
@click.argument('experiment_name')
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Directory to save analysis outputs'
)
@click.option(
    '--generate-plots',
    is_flag=True,
    default=True,
    help='Generate analysis plots'
)
@click.option(
    '--metrics-only',
    is_flag=True,
    help='Generate only metrics report (skip plots)'
)
@pass_context
def analyze_experiment(
    ctx: CLIContext,
    experiment_name: str,
    output_dir: Optional[Path],
    generate_plots: bool,
    metrics_only: bool
):
    """Analyze results from a completed experiment."""
    try:
        from scramblebench.evaluation import ResultsManager, MetricsCalculator, PlotGenerator
        
        results_manager = ResultsManager(ctx.results_dir)
        
        echo_info(f"Loading experiment: {experiment_name}", ctx)
        results = results_manager.load_results(experiment_name)
        
        if not output_dir:
            output_dir = ctx.results_dir / experiment_name / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate metrics report
        echo_info("Calculating metrics...", ctx)
        metrics_calc = MetricsCalculator()
        metrics_report = metrics_calc.generate_metrics_report(results)
        
        metrics_path = output_dir / "metrics_report.json"
        metrics_calc.save_metrics_report(metrics_report, metrics_path)
        
        echo_success(f"Metrics report saved to {metrics_path}", ctx)
        
        # Generate plots if requested
        if generate_plots and not metrics_only:
            echo_info("Generating plots...", ctx)
            
            plots_dir = output_dir / "plots"
            plot_generator = PlotGenerator()
            
            plot_results = plot_generator.generate_all_plots(results, plots_dir)
            
            successful_plots = sum(1 for r in plot_results.values() if r.success)
            echo_success(f"Generated {successful_plots} plot types in {plots_dir}", ctx)
        
        if ctx.output_format == "json":
            click.echo(json.dumps({
                "experiment": experiment_name,
                "analysis_dir": str(output_dir),
                "metrics_available": True,
                "plots_generated": generate_plots and not metrics_only
            }))
        
    except Exception as e:
        echo_error(f"Failed to analyze experiment: {e}", ctx)
        sys.exit(1)


@evaluate_group.command(name='compare')
@click.argument('experiment_names', nargs=-1, required=True)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    help='Output file for comparison report'
)
@click.option(
    '--metric',
    type=click.Choice(['accuracy', 'robustness', 'efficiency']),
    default='accuracy',
    help='Primary metric for comparison'
)
@pass_context
def compare_experiments(
    ctx: CLIContext,
    experiment_names: Tuple[str, ...],
    output: Optional[Path],
    metric: str
):
    """Compare results from multiple experiments."""
    try:
        from scramblebench.evaluation import ResultsManager
        
        results_manager = ResultsManager(ctx.results_dir)
        
        echo_info(f"Comparing {len(experiment_names)} experiments", ctx)
        comparison_df = results_manager.compare_experiments(list(experiment_names))
        
        if output:
            comparison_df.to_csv(output, index=False)
            echo_success(f"Comparison saved to {output}", ctx)
        
        if ctx.output_format == "json":
            click.echo(comparison_df.to_json(orient='records', indent=2))
        elif not ctx.quiet:
            # Display as table
            table = Table(title="Experiment Comparison")
            
            for col in comparison_df.columns:
                table.add_column(col, style="cyan" if col == "experiment" else "white")
            
            for _, row in comparison_df.iterrows():
                table.add_row(*[str(val) for val in row])
            
            console.print(table)
        
    except Exception as e:
        echo_error(f"Failed to compare experiments: {e}", ctx)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()