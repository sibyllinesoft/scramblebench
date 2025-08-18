# ScrambleBench Evaluation Pipeline

A comprehensive evaluation system for LLM robustness testing with transformation-based benchmarking.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install ScrambleBench with evaluation dependencies
pip install -e .

# Required for plotting
pip install matplotlib seaborn plotly
```

### 2. Set API Key

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 3. Run Quick Evaluation

```bash
# Generate a sample configuration
scramblebench evaluate config configs/quick_test.yaml --template basic

# Run evaluation
scramblebench evaluate run --config configs/quick_test.yaml
```

## 📋 Features

### Core Capabilities

- **🔄 Transformation Pipeline**: Generate transformed versions of benchmark questions
  - Language translation to constructed languages
  - Proper noun swapping
  - Synonym replacement
  - Paraphrasing (future)
  - Long context insertion (future)

- **🤖 Model Evaluation**: Test multiple LLMs via OpenRouter API
  - Support for 50+ models (Claude, GPT, Llama, Mixtral, etc.)
  - Async evaluation with rate limiting
  - Comprehensive error handling
  - Progress tracking

- **📊 Results Management**: Structured storage and analysis
  - JSON and Parquet output formats
  - Experiment organization
  - Metadata tracking
  - Reproducible configurations

- **📈 Metrics & Analysis**: Statistical evaluation
  - Accuracy metrics (exact match, F1)
  - Robustness metrics (performance degradation)
  - Statistical significance testing
  - Effect size calculations

- **📊 Visualization**: Comprehensive plotting
  - Model comparison charts
  - Robustness heatmaps
  - Performance distributions
  - Interactive dashboards

## 🎯 Usage Examples

### Command Line Interface

```bash
# Quick evaluation
scramblebench evaluate run \
  --models "anthropic/claude-3-sonnet,openai/gpt-4" \
  --benchmarks "data/benchmarks/logic_reasoning.json" \
  --experiment-name "model_comparison" \
  --transformations "language_translation,synonym_replacement"

# Using configuration file
scramblebench evaluate run --config configs/comprehensive.yaml

# List experiments
scramblebench evaluate list

# Analyze results
scramblebench evaluate analyze model_comparison

# Compare experiments
scramblebench evaluate compare exp1 exp2 exp3
```

### Python API

```python
import asyncio
from scramblebench.evaluation import run_quick_evaluation

# Quick evaluation
results = await run_quick_evaluation(
    benchmark_paths=["data/benchmarks/logic_reasoning.json"],
    models=["anthropic/claude-3-sonnet", "openai/gpt-4"],
    experiment_name="api_test",
    transformations=["language_translation"]
)

print(f"Success rate: {results.get_success_rate():.2%}")
```

### Configuration File

```yaml
experiment_name: comprehensive_evaluation
description: Full robustness testing across models and transformations

benchmark_paths:
  - data/benchmarks/logic_reasoning.json
  - data/benchmarks/reading_comprehension.json

models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0
    max_tokens: 2048
  - name: openai/gpt-4
    provider: openrouter
    temperature: 0.0
    max_tokens: 2048

transformations:
  enabled_types:
    - language_translation
    - synonym_replacement
    - proper_noun_swap
  languages:
    - constructed_agglutinative_1
    - constructed_fusional_1
  synonym_rate: 0.3

max_samples: 200
generate_plots: true
calculate_significance: true
```

## 📊 Output Structure

```
results/
└── experiment_name/
    ├── config.yaml                 # Experiment configuration
    ├── metadata.json              # Experiment metadata  
    ├── results.json               # Detailed results
    ├── results.parquet            # Results data (efficient format)
    ├── summary.json               # Summary statistics
    ├── transformations.json       # Transformation details
    ├── metrics_report.json        # Comprehensive metrics
    └── plots/                     # Visualizations
        ├── model_comparison.png
        ├── robustness_analysis.png
        ├── performance_heatmap.png
        └── interactive_dashboard.html
```

## 🔧 Advanced Configuration

### Transformation Settings

```yaml
transformations:
  enabled_types:
    - language_translation
    - synonym_replacement
    - proper_noun_swap
  
  # Language translation
  languages:
    - constructed_agglutinative_1
    - constructed_fusional_1
  language_complexity: 6
  
  # Synonym replacement
  synonym_rate: 0.4
  preserve_function_words: true
  
  # Proper noun swapping
  proper_noun_strategy: random
  
  # Reproducibility
  seed: 42
  batch_size: 20
```

### Model Configuration

```yaml
models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0
    max_tokens: 2048
    timeout: 90
    rate_limit: 1.0  # requests per second
    
  - name: openai/gpt-4
    provider: openrouter
    temperature: 0.0
    max_tokens: 2048
    timeout: 90
    rate_limit: 1.0
```

### Performance Settings

```yaml
# Evaluation performance
max_concurrent_requests: 5
save_interval: 50

# Sampling
max_samples: 500
sample_seed: 123

# Analysis
generate_plots: true
calculate_significance: true
```

## 📈 Metrics Explained

### Accuracy Metrics

- **Exact Match**: Percentage of responses exactly matching expected answers
- **F1 Score**: Token-level F1 for partial credit evaluation
- **Success Rate**: Percentage of successful API calls

### Robustness Metrics

- **Performance Degradation**: Original accuracy - Transformed accuracy
- **Significant Degradations**: Transformations causing >5% performance drop
- **Average Degradation**: Mean degradation across all transformations

### Statistical Tests

- **Pairwise Comparisons**: t-tests between model performances
- **Effect Sizes**: Cohen's d for practical significance
- **Confidence Intervals**: 95% confidence bounds for metrics

## 🎨 Visualization Types

### Static Plots

1. **Model Comparison**: Bar charts comparing overall accuracy
2. **Robustness Heatmap**: Performance degradation by model/transformation
3. **Performance Matrix**: Success rates across conditions
4. **Response Time Analysis**: API latency distributions

### Interactive Plots

1. **Dashboard**: Multi-panel overview with filtering
2. **Drill-down Views**: Detailed analysis by model/transformation
3. **Time Series**: Performance over evaluation timeline

## 🔍 Troubleshooting

### Common Issues

**API Key Error**
```bash
export OPENROUTER_API_KEY="your_key_here"
```

**Rate Limiting**
```yaml
models:
  - rate_limit: 0.5  # Reduce to 0.5 requests/second
max_concurrent_requests: 2  # Reduce concurrency
```

**Memory Issues**
```yaml
transformations:
  batch_size: 10  # Reduce batch size
max_samples: 100  # Limit sample size
```

**Plot Generation Errors**
```bash
pip install matplotlib seaborn plotly  # Install dependencies
# Or use --no-plots flag
```

### Performance Tips

1. **Start Small**: Begin with 10-50 samples for testing
2. **Use Rate Limits**: Respect API limits to avoid throttling
3. **Batch Processing**: Configure appropriate batch sizes
4. **Parallel Models**: Evaluate multiple models simultaneously
5. **Save Intermediate**: Enable saving for long evaluations

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/test_evaluation/ -v

# Run with coverage
pytest tests/test_evaluation/ --cov=scramblebench.evaluation
```

## 📚 Documentation

- **[Evaluation Guide](EVALUATION_GUIDE.md)**: Comprehensive usage guide
- **[API Documentation](docs/api/)**: Detailed API reference
- **[Examples](examples/)**: Example scripts and configurations
- **[Configuration Reference](docs/configuration.md)**: Complete config options

## 🤝 Contributing

1. **Add Transformation Types**: Extend `TransformationPipeline`
2. **Add Model Providers**: Implement new `ModelInterface` 
3. **Add Metrics**: Extend `MetricsCalculator`
4. **Add Visualizations**: Extend `PlotGenerator`

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenRouter for LLM API access
- Matplotlib/Seaborn/Plotly for visualizations
- Pandas/NumPy for data processing
- Pydantic for configuration validation

---

For detailed usage instructions, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)