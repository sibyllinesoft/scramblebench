# ScrambleBench Evaluation Pipeline Guide

The ScrambleBench evaluation pipeline provides comprehensive benchmarking capabilities for Large Language Models (LLMs) with transformation-based robustness testing.

## Overview

The evaluation pipeline integrates all ScrambleBench components to:

1. **Generate Transformations**: Create transformed versions of benchmark questions using various strategies
2. **Evaluate Models**: Run multiple LLMs on both original and transformed questions via OpenRouter API
3. **Store Results**: Save evaluation data in structured formats for analysis
4. **Calculate Metrics**: Compute accuracy, robustness, and statistical significance measures
5. **Generate Visualizations**: Create plots and interactive dashboards for result analysis

## Quick Start

### 1. Installation

Ensure you have the evaluation dependencies installed:

```bash
# Install ScrambleBench with evaluation dependencies
pip install -e .

# Or if using uv
uv pip install -e .
```

### 2. Set up API Keys

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 3. Run a Quick Evaluation

```bash
# Quick evaluation with minimal configuration
scramblebench evaluate run \
  --models "anthropic/claude-3-haiku,openai/gpt-3.5-turbo" \
  --benchmarks "data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json" \
  --experiment-name "quick_test" \
  --transformations "language_translation,synonym_replacement"
```

### 4. Using Configuration Files

For more complex evaluations, use configuration files:

```bash
# Generate a sample configuration
scramblebench evaluate config configs/my_evaluation.yaml --template comprehensive

# Run evaluation with configuration
scramblebench evaluate run --config configs/my_evaluation.yaml
```

## CLI Commands

### `scramblebench evaluate run`

Run a comprehensive evaluation experiment.

**Options:**
- `--config PATH`: Configuration file for evaluation
- `--models LIST`: Comma-separated list of model names (for quick setup)
- `--benchmarks LIST`: Comma-separated list of benchmark file paths
- `--experiment-name NAME`: Name of the experiment
- `--transformations LIST`: Comma-separated list of transformation types
- `--output-dir PATH`: Output directory for results (default: results)
- `--max-samples INT`: Maximum number of samples per benchmark
- `--no-plots`: Skip plot generation
- `--include-original/--no-original`: Include evaluation of original problems

**Examples:**

```bash
# Quick setup
scramblebench evaluate run \
  --models "anthropic/claude-3-sonnet,openai/gpt-4" \
  --benchmarks "data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json" \
  --experiment-name "logic_test" \
  --max-samples 50

# Using configuration file
scramblebench evaluate run --config configs/comprehensive_evaluation.yaml

# Without plotting (faster)
scramblebench evaluate run --config configs/quick_test.yaml --no-plots
```

### `scramblebench evaluate config`

Generate sample configuration files.

```bash
# Basic configuration
scramblebench evaluate config configs/basic.yaml --template basic

# Comprehensive configuration
scramblebench evaluate config configs/comprehensive.yaml --template comprehensive

# Robustness-focused configuration
scramblebench evaluate config configs/robustness.yaml --template robustness
```

### `scramblebench evaluate list`

List all evaluation experiments.

```bash
# Table format
scramblebench evaluate list

# Simple list
scramblebench evaluate list --format simple

# JSON output
scramblebench evaluate list --format json
```

### `scramblebench evaluate analyze`

Analyze results from a completed experiment.

```bash
# Full analysis with plots
scramblebench evaluate analyze my_experiment

# Metrics only (no plots)
scramblebench evaluate analyze my_experiment --metrics-only

# Custom output directory
scramblebench evaluate analyze my_experiment --output-dir analysis_results/
```

### `scramblebench evaluate compare`

Compare results from multiple experiments.

```bash
# Compare experiments
scramblebench evaluate compare experiment1 experiment2 experiment3

# Save comparison to file
scramblebench evaluate compare exp1 exp2 --output comparison.csv

# Focus on specific metric
scramblebench evaluate compare exp1 exp2 --metric robustness
```

## Configuration

### Basic Configuration Structure

```yaml
experiment_name: my_evaluation
description: Description of the evaluation
mode: comprehensive  # accuracy, robustness, comprehensive

# Input datasets
benchmark_paths:
  - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
  - data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json

output_dir: results

# Models to evaluate
models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0
    max_tokens: 2048
    timeout: 60
    rate_limit: 1.0

# Transformation settings
transformations:
  enabled_types:
    - language_translation
    - synonym_replacement
  languages:
    - constructed_agglutinative_1
    - constructed_fusional_1
  synonym_rate: 0.3

# Evaluation settings
max_samples: 100
generate_plots: true
calculate_significance: true
```

### Available Transformation Types

- `language_translation`: Translate to constructed languages
- `proper_noun_swap`: Replace proper nouns with alternatives
- `synonym_replacement`: Replace words with synonyms
- `paraphrasing`: Rephrase questions (future feature)
- `long_context`: Add irrelevant context (future feature)
- `all`: Enable all available transformations

### Supported Models

All models available through OpenRouter are supported. Common examples:

- `anthropic/claude-3-sonnet`
- `anthropic/claude-3-haiku`
- `openai/gpt-4`
- `openai/gpt-3.5-turbo`
- `meta-llama/llama-2-70b-chat`
- `mistralai/mixtral-8x7b-instruct`

## Output Structure

Each evaluation creates the following directory structure:

```
results/
└── experiment_name/
    ├── config.yaml                 # Experiment configuration
    ├── metadata.json              # Experiment metadata
    ├── results.json               # Detailed results (JSON format)
    ├── results.parquet            # Results data (Parquet format)
    ├── summary.json               # Summary statistics
    ├── transformations.json       # Transformation details
    ├── metrics_report.json        # Comprehensive metrics
    └── plots/                     # Generated visualizations
        ├── model_comparison.png
        ├── robustness_analysis.png
        ├── performance_heatmap.png
        ├── response_time_distribution.png
        ├── interactive_dashboard.html
        └── plot_summary.json
```

## Metrics and Analysis

### Accuracy Metrics

- **Exact Match**: Percentage of responses that exactly match expected answers
- **F1 Score**: Token-level F1 score for partial credit
- **Success Rate**: Percentage of successful API calls

### Robustness Metrics

- **Performance Degradation**: Difference between original and transformed performance
- **Significant Degradations**: Transformations causing >5% performance drop
- **Robustness Score**: Average performance across all transformations

### Statistical Tests

- **Pairwise Comparisons**: Statistical significance tests between models
- **Effect Sizes**: Cohen's d for practical significance
- **Confidence Intervals**: Uncertainty estimates for performance metrics

## Visualization Types

### Static Plots (PNG/PDF)

1. **Model Comparison**: Bar chart comparing model accuracies
2. **Robustness Analysis**: Heatmap of performance degradation by transformation
3. **Performance Heatmap**: Model performance across transformation types
4. **Response Time Distribution**: Analysis of API response times

### Interactive Plots (HTML)

1. **Interactive Dashboard**: Multi-panel Plotly dashboard with filtering
2. **Detailed Comparisons**: Drill-down capabilities for specific models/transformations

## Best Practices

### 1. Start Small

Begin with quick tests to validate your setup:

```bash
scramblebench evaluate run \
  --models "anthropic/claude-3-haiku" \
  --benchmarks "data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json" \
  --experiment-name "test_run" \
  --max-samples 10 \
  --no-plots
```

### 2. Use Configuration Files for Complex Evaluations

For reproducible research, always use configuration files:

```bash
# Generate template
scramblebench evaluate config configs/my_study.yaml --template comprehensive

# Edit the configuration file as needed
# Then run:
scramblebench evaluate run --config configs/my_study.yaml
```

### 3. Monitor Rate Limits

Set appropriate rate limits in your configuration to avoid API throttling:

```yaml
models:
  - name: openai/gpt-4
    rate_limit: 1.0  # 1 request per second
  - name: anthropic/claude-3-haiku
    rate_limit: 2.0  # 2 requests per second
```

### 4. Save Intermediate Results

Enable intermediate saving for long evaluations:

```yaml
save_interval: 50  # Save every 50 samples
```

### 5. Use Appropriate Sample Sizes

- Development/testing: 10-50 samples
- Pilot studies: 100-200 samples
- Full evaluations: 500+ samples

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **Rate Limiting Errors**
   - Reduce `rate_limit` values in configuration
   - Decrease `max_concurrent_requests`

3. **Memory Issues with Large Evaluations**
   - Reduce `batch_size` in transformation settings
   - Use `max_samples` to limit evaluation size
   - Save results in Parquet format only

4. **Plot Generation Failures**
   - Install plotting dependencies: `pip install matplotlib seaborn plotly`
   - Use `--no-plots` flag to skip plotting
   - Check available disk space

### Performance Optimization

1. **Parallel Processing**
   ```yaml
   max_concurrent_requests: 5  # Adjust based on API limits
   ```

2. **Batch Transformations**
   ```yaml
   transformations:
     batch_size: 20  # Process transformations in batches
   ```

3. **Efficient Storage**
   - Use Parquet format for large datasets
   - Enable compression in results storage

## Examples

### Example 1: Model Comparison Study

```yaml
experiment_name: model_comparison_study
description: Comparing state-of-the-art models on logic reasoning
mode: accuracy

benchmark_paths:
  - data/benchmarks/collected/01_logic_reasoning/medium/collected_samples.json

models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0
  - name: openai/gpt-4
    provider: openrouter
    temperature: 0.0
  - name: meta-llama/llama-2-70b-chat
    provider: openrouter
    temperature: 0.0

transformations:
  enabled_types:
    - language_translation
  languages:
    - constructed_agglutinative_1

max_samples: 200
calculate_significance: true
```

### Example 2: Robustness Analysis

```yaml
experiment_name: robustness_analysis
description: Comprehensive robustness testing across transformation types
mode: robustness

benchmark_paths:
  - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
  - data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json

models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0

transformations:
  enabled_types:
    - all
  synonym_rate: 0.4
  language_complexity: 7

max_samples: 150
generate_plots: true
```

Run these examples with:

```bash
scramblebench evaluate run --config configs/model_comparison.yaml
scramblebench evaluate run --config configs/robustness_analysis.yaml
```

## Advanced Usage

### Programmatic API

For advanced users, the evaluation pipeline can be used programmatically:

```python
import asyncio
from scramblebench.evaluation import EvaluationRunner, EvaluationConfig

# Load configuration
config = EvaluationConfig.load_from_file("configs/my_evaluation.yaml")

# Create runner
runner = EvaluationRunner(config)

# Run evaluation
results = asyncio.run(runner.run_evaluation())

# Access results
print(f"Total evaluations: {len(results.results)}")
print(f"Success rate: {results.get_success_rate()}")

# Generate analysis
from scramblebench.evaluation import MetricsCalculator, PlotGenerator

metrics_calc = MetricsCalculator()
metrics = metrics_calc.generate_metrics_report(results)

plot_gen = PlotGenerator()
plots = plot_gen.generate_all_plots(results, "output/plots/")
```

### Custom Transformations

To add custom transformation types, extend the `TransformationPipeline` class:

```python
from scramblebench.evaluation.transformation_pipeline import TransformationPipeline

class CustomTransformationPipeline(TransformationPipeline):
    def _apply_custom_transformation(self, problem):
        # Implement your custom transformation
        transformed_problem = problem.copy()
        # ... transformation logic ...
        return TransformationResult(
            original_problem=problem,
            transformed_problem=transformed_problem,
            transformation_type="custom_type",
            transformation_metadata={"method": "custom"},
            success=True
        )
```

This guide provides a comprehensive overview of the ScrambleBench evaluation pipeline. For specific use cases or advanced configurations, refer to the configuration examples and API documentation.