# ScrambleBench to Language Dependency Atlas Migration Guide

This guide helps you migrate from experimental ScrambleBench scripts to the production-grade Language Dependency Atlas framework.

## Overview

The Language Dependency Atlas framework provides:

- **Production-grade architecture**: Modular, tested, and maintainable codebase
- **Unified configuration**: YAML-based configuration for 100% reproducibility
- **Comprehensive analysis**: Statistical analysis and academic reporting
- **Database storage**: SQLite-based experiment tracking and result storage
- **CLI integration**: Command-line interface for all operations
- **Academic output**: Publication-ready reports in HTML, LaTeX, and PDF

## Migration Paths

### 1. Automatic Migration

The framework provides automated migration tools for common experimental scripts:

#### Migrate Individual Scripts

```bash
# Convert a script to YAML configuration
python -m language_dependency_atlas.cli migrate script precision_threshold_test.py

# Run the migrated experiment directly
python -m language_dependency_atlas.cli migrate precision gemma3:27b
```

#### Batch Migration

```bash
# Migrate all experimental scripts in current directory
python -m language_dependency_atlas.cli migrate batch --scripts-dir . --output-dir configs/experiments/migrated

# Run all migrated experiments
python -m language_dependency_atlas.cli migrate run-batch --config-dir configs/experiments/migrated
```

### 2. Manual Configuration

For custom experimental setups, create YAML configurations manually:

```bash
# Create a new configuration
python -m language_dependency_atlas.cli config create my_experiment --type threshold_analysis --models "llama3:8b,gemma3:27b"

# Validate the configuration
python -m language_dependency_atlas.cli config validate configs/experiments/my_experiment.yaml

# Run the experiment
python -m language_dependency_atlas.cli experiment run configs/experiments/my_experiment.yaml
```

## Common Migration Scenarios

### Scenario 1: Precision Threshold Testing

**Before (experimental script):**
```python
# precision_threshold_test.py
client = OllamaClient()
model_name = "gemma3:27b"
domains = ["mathematics", "logic"]
# ... manual testing logic
```

**After (production framework):**
```yaml
# configs/experiments/precision_test.yaml
experiment_name: "precision_threshold_analysis"
experiment_type: "threshold_analysis"
models:
  - name: "gemma3:27b"
    provider: "ollama"
    temperature: 0.1
testing:
  domains: ["mathematics", "logic"]
  difficulty_levels: [1, 2, 3]
  questions_per_domain_per_difficulty: 10
scrambling:
  levels: [0, 1, 2, 3, 4, 5]
analysis:
  contamination_detection: true
  generate_plots: true
```

**Run with:**
```bash
python -m language_dependency_atlas.cli experiment run configs/experiments/precision_test.yaml
```

### Scenario 2: Threshold Explorer

**Before:**
```python
# gemma27b_threshold_explorer.py
for level in range(0, 6):
    # Manual scrambling and evaluation
    results[level] = evaluate_model(scrambled_questions)
```

**After:**
```bash
# Direct migration command
python -m language_dependency_atlas.cli migrate explorer gemma3:27b --start-level 0 --end-level 5 --step-size 1
```

### Scenario 3: Multi-Model Comparison

**Before (multiple scripts):**
```python
# Multiple separate scripts for different models
# model_a_test.py, model_b_test.py, etc.
```

**After (single config):**
```yaml
experiment_name: "multi_model_comparison"
experiment_type: "comparative_analysis"
models:
  - name: "llama3:8b"
    provider: "ollama"
  - name: "gemma3:27b" 
    provider: "ollama"
  - name: "phi3:3.8b"
    provider: "ollama"
testing:
  domains: ["mathematics", "logic", "reading_comprehension", "common_sense"]
  questions_per_domain_per_difficulty: 20
analysis:
  multiple_comparison_correction: true
  effect_size_calculation: true
```

## Configuration Templates

### Quick Testing Template
```yaml
# configs/experiments/quick_test.yaml
experiment_name: "quick_evaluation"
experiment_type: "threshold_analysis"
models:
  - name: "phi3:3.8b"
    provider: "ollama"
    temperature: 0.1
    max_tokens: 50
    timeout: 20
testing:
  domains: ["mathematics", "logic"]
  difficulty_levels: [1, 2]
  questions_per_domain_per_difficulty: 5
  parallel_execution: false
scrambling:
  levels: [0, 3, 5]
  consistency_validation: false
analysis:
  generate_latex_tables: false
  contamination_detection: false
output_dir: "results/quick_test"
save_intermediate_results: false
```

### Academic Publication Template
```yaml
# configs/experiments/academic_study.yaml
experiment_name: "language_dependency_academic_study"
experiment_type: "comparative_analysis"
description: "Systematic comparison of language dependency across model architectures"
researcher: "Research Team"
institution: "University"
models:
  - name: "llama3:8b"
    provider: "ollama"
    temperature: 0.0
  - name: "gemma3:27b"
    provider: "ollama"
    temperature: 0.0
testing:
  domains: ["mathematics", "logic", "reading_comprehension", "common_sense", "spatial_reasoning"]
  difficulty_levels: [1, 2, 3]
  questions_per_domain_per_difficulty: 30
  random_sampling: true
  random_seed: 42
  contamination_detection: true
scrambling:
  levels: [0, 1, 2, 3, 4, 5]
  preserve_numbers: true
  preserve_punctuation: true
  consistency_validation: true
analysis:
  confidence_level: 0.95
  significance_threshold: 0.01
  effect_size_calculation: true
  multiple_comparison_correction: true
  contamination_threshold: 0.85
  generate_latex_tables: true
  generate_plots: true
  include_raw_responses: true
output_dir: "results/academic_study"
save_intermediate_results: true
cache_responses: true
```

### Contamination Detection Template
```yaml
# configs/experiments/contamination_detection.yaml
experiment_name: "contamination_detection_protocol"
experiment_type: "contamination_detection"
description: "Systematic detection of training data contamination"
models:
  - name: "gpt-3.5-turbo"
    provider: "openrouter"
    api_key: "${OPENROUTER_API_KEY}"
    temperature: 0.0
  - name: "llama3:8b"
    provider: "ollama"
    temperature: 0.0
testing:
  domains: ["mathematics", "reading_comprehension", "common_sense"]
  questions_per_domain_per_difficulty: 30
  contamination_detection: true
scrambling:
  levels: [0, 1, 3, 4, 5]
  consistency_validation: true
analysis:
  confidence_level: 0.99
  significance_threshold: 0.001
  contamination_threshold: 0.80
  baseline_performance_required: true
  generate_latex_tables: true
  include_raw_responses: true
```

## Database Schema

The production framework stores results in SQLite databases with the following structure:

```sql
-- Experiments table
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config TEXT NOT NULL,  -- JSON configuration
    status TEXT NOT NULL,  -- 'running', 'completed', 'error'
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_questions INTEGER,
    completed_questions INTEGER,
    error_message TEXT
);

-- Results table  
CREATE TABLE results (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    model_name TEXT,
    domain TEXT,
    difficulty INTEGER,
    scrambling_level INTEGER,
    question_text TEXT,
    scrambled_question TEXT,
    response TEXT,
    is_correct BOOLEAN,
    response_time REAL,
    timestamp TIMESTAMP,
    metadata TEXT,  -- JSON metadata
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

-- Analysis table
CREATE TABLE analysis (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    analysis_type TEXT,
    results TEXT,  -- JSON results
    timestamp TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);
```

## API Reference

### Python API

```python
from language_dependency_atlas import ExperimentRunner, ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml("configs/experiments/my_experiment.yaml")

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

# Generate report
from language_dependency_atlas.analysis import AcademicReporter
reporter = AcademicReporter(config)
report_path = reporter.generate_comprehensive_report(results)
```

### CLI Reference

```bash
# Configuration management
python -m language_dependency_atlas.cli config create <name> [options]
python -m language_dependency_atlas.cli config validate <config-file>
python -m language_dependency_atlas.cli config show <config-file>

# Experiment management
python -m language_dependency_atlas.cli experiment run <config-file>
python -m language_dependency_atlas.cli experiment list [--database <db>]
python -m language_dependency_atlas.cli experiment status <experiment-id>

# Analysis and reporting
python -m language_dependency_atlas.cli analysis report <experiment-id>
python -m language_dependency_atlas.cli analysis compare <exp-id-1> <exp-id-2>
python -m language_dependency_atlas.cli analysis contamination <experiment-id>

# Migration utilities
python -m language_dependency_atlas.cli migrate script <script-path>
python -m language_dependency_atlas.cli migrate batch [--scripts-dir <dir>]
python -m language_dependency_atlas.cli migrate precision <model-name>
python -m language_dependency_atlas.cli migrate explorer <model-name>

# Project initialization
python -m language_dependency_atlas.cli init <project-name>
```

## Integration with ScrambleBench CLI

The Language Dependency Atlas is integrated into the main ScrambleBench CLI:

```bash
# Access through ScrambleBench CLI
scramblebench atlas --help
scramblebench atlas config create my_experiment
scramblebench atlas experiment run configs/experiments/my_experiment.yaml
scramblebench atlas migrate precision gemma3:27b
```

## Best Practices

### 1. Configuration Management

- **Version control**: Store all YAML configurations in version control
- **Environment variables**: Use environment variables for API keys and sensitive data
- **Validation**: Always validate configurations before running experiments
- **Templates**: Start with provided templates and customize as needed

### 2. Experiment Design

- **Reproducibility**: Set random seeds and document all parameters
- **Sample sizes**: Use appropriate sample sizes for statistical significance
- **Controls**: Include baseline and control conditions
- **Documentation**: Document experiment rationale and methodology

### 3. Result Management

- **Storage**: Use the built-in SQLite database for result storage
- **Backup**: Regularly backup experiment databases
- **Validation**: Validate results against known benchmarks
- **Reporting**: Generate academic reports for publication and sharing

### 4. Performance Optimization

- **Parallel execution**: Enable parallel execution for faster results
- **Caching**: Enable response caching to avoid redundant API calls
- **Resource management**: Monitor system resources during experiments
- **Incremental analysis**: Use intermediate result saving for long experiments

## Troubleshooting

### Common Issues

1. **Configuration validation errors**
   - Check YAML syntax
   - Verify all required fields are present
   - Validate model names and providers

2. **Model connection errors**
   - Ensure Ollama is running for local models
   - Verify API keys for cloud providers
   - Check network connectivity

3. **Database errors**
   - Ensure write permissions to output directory
   - Check disk space availability
   - Verify SQLite installation

4. **Memory issues with large experiments**
   - Enable parallel execution with appropriate worker limits
   - Use intermediate result saving
   - Consider breaking large experiments into smaller batches

### Performance Tuning

```yaml
# Optimize for speed
testing:
  parallel_execution: true
  max_concurrent_requests: 4
  timeout_per_question: 15
  questions_per_domain_per_difficulty: 5

# Optimize for accuracy
testing:
  parallel_execution: false
  max_concurrent_requests: 1
  timeout_per_question: 60
  questions_per_domain_per_difficulty: 50
  verification_required: true
```

## Support and Documentation

- **GitHub Repository**: [ScrambleBench GitHub](https://github.com/user/scramblebench)
- **Documentation**: See `docs/` directory for detailed documentation
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Community**: Join discussions in GitHub Discussions

## Next Steps

1. **Choose your migration path**: Automatic migration for common scripts, manual configuration for custom setups
2. **Start with templates**: Use provided configuration templates as starting points
3. **Validate early**: Test configurations with small sample sizes first
4. **Scale gradually**: Increase experiment scope once basic functionality is working
5. **Generate reports**: Use academic reporting features for publication-ready output

The Language Dependency Atlas framework provides a robust, production-ready foundation for LLM evaluation research. The migration tools and comprehensive documentation ensure a smooth transition from experimental scripts to a maintainable research platform.