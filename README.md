# ScrambleBench

**The contamination-resistant LLM evaluation toolkit that gives you confidence in your benchmark results.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Why ScrambleBench?

**Training data contamination is destroying the credibility of LLM evaluation.** When models have seen your test data during training, benchmark scores become meaningless. Traditional benchmarks are compromised, making it impossible to accurately measure true model capabilities or compare different approaches.

**ScrambleBench solves this problem completely.** By transforming existing benchmarks into novel forms that preserve logical structure while eliminating memorization advantages, you get reliable evaluation results you can trust. Whether you're:

- <kbd>microscope</kbd> **Researchers** needing clean evaluation data for papers
- <kbd>building-2</kbd> **Enterprises** selecting models for production deployments  
- <kbd>rocket</kbd> **Startups** optimizing AI systems for specific tasks
- <kbd>graduation-cap</kbd> **Academics** studying model capabilities without contamination bias

ScrambleBench provides the contamination-resistant evaluation framework you need.

## How It Works

ScrambleBench uses two revolutionary approaches to eliminate training data contamination:

1. **<kbd>globe</kbd> Translation Benchmarks**: Transform problems into systematically constructed languages that preserve logical structure while making memorization impossible
2. **<kbd>book-open</kbd> Long Context Benchmarks**: Intelligently modify documents and Q&A pairs through translation and transformation while maintaining semantic content

**The result?** Clean, reliable benchmarks that measure true model reasoning rather than memorization.

## Quick Start

Get up and running with ScrambleBench in minutes:

### Installation

```bash
# Clone the repository
git clone https://github.com/sibyllinesoft/scramblebench.git
cd scramblebench

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Install development dependencies
uv sync --group dev

# Install with NLP capabilities for better text processing
pip install -e ".[nlp]"
```

### 30-Second Demo

```bash
# Generate a constructed language
scramblebench language generate mylang --type substitution --complexity 5

# Transform text using the language
scramblebench transform text "What is the capital of France?" mylang

# Run evaluation (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-key"
scramblebench evaluate run \
  --models "anthropic/claude-3-haiku" \
  --benchmarks "data/benchmarks/logic_reasoning.json" \
  --experiment-name "demo" \
  --max-samples 5
```

### Basic Python Usage

```python
from scramblebench import TranslationBenchmark
from scramblebench.llm import OpenRouterClient
from scramblebench.translation.language_generator import LanguageType

# Create a contamination-resistant benchmark
benchmark = TranslationBenchmark(
    source_dataset="simple_qa",
    language_type=LanguageType.SUBSTITUTION,
    language_complexity=5
)

# Initialize your model
model = OpenRouterClient(
    model_name="openai/gpt-4",
    api_key="your-openrouter-key"
)

# Get reliable, contamination-free results
result = benchmark.run(model, num_samples=50)
print(f"True model accuracy: {result.score:.2%}")
```

## Features

### <kbd>globe</kbd> Translation Benchmarks
- **Constructed Language Generation**: Create systematic artificial languages with varying complexity
- **Problem Translation**: Transform benchmark problems while maintaining solvability
- **Multiple Language Types**: Substitution, phonetic, scrambled, and synthetic languages
- **Translation Key Export**: Full mapping for verification and analysis

### <kbd>book-open</kbd> Long Context Benchmarks  
- **Document Transformation**: Multiple strategies (translation, paraphrase, structural reordering)
- **Q&A Alignment**: Intelligent transformation of questions and answers to match document changes
- **Preservation Controls**: Configurable preservation of numbers, entities, and structure
- **Answer Type Support**: Extractive, abstractive, multiple choice, and more

### <kbd>plug</kbd> LLM Integration
- **OpenRouter Support**: Access to 100+ models through unified API
- **Model Interface**: Standardized interface for any LLM provider  
- **Rate Limiting**: Built-in rate limiting and retry logic
- **Async Support**: Efficient batch processing and concurrent evaluation

### <kbd>bar-chart-3</kbd> Evaluation Pipeline
- **Comprehensive Robustness Testing**: End-to-end evaluation with transformation-based testing
- **Multi-Model Support**: Evaluate multiple LLMs simultaneously via OpenRouter API
- **Statistical Analysis**: Accuracy, robustness, and significance testing
- **Rich Visualizations**: Model comparisons, degradation analysis, interactive dashboards
- **Results Management**: Structured storage, experiment tracking, and comparison tools

### <kbd>settings</kbd> Configuration & Data
- **Flexible Configuration**: YAML-based config with environment variable support
- **Multiple Data Formats**: JSON, JSONL, CSV, Parquet, HuggingFace datasets
- **Intelligent Caching**: Automatic caching for improved performance
- **Extensible Loaders**: Easy integration of custom data sources

### <kbd>terminal</kbd> Command Line Interface
- **Comprehensive CLI**: Full-featured command-line interface with Click framework
- **Language Management**: Generate, list, and manage constructed languages
- **Batch Processing**: Extract vocabularies and transform benchmark datasets  
- **Text Transformations**: Apply proper noun swapping and synonym replacement
- **Multiple Output Formats**: Support for text, JSON, and YAML output formats

## Evaluation Pipeline

ScrambleBench includes a comprehensive evaluation pipeline for robustness testing:

### Quick Evaluation

```bash
# Set up API key
export OPENROUTER_API_KEY="your_key_here"

# Quick evaluation with CLI
scramblebench evaluate run \
  --models "anthropic/claude-3-haiku" \
  --benchmarks "data/benchmarks/logic_reasoning.json" \
  --experiment-name "quick_test" \
  --transformations "language_translation" \
  --max-samples 10
```

### Configuration-Based Evaluation

```yaml
# configs/robustness_test.yaml
experiment_name: robustness_evaluation
description: Comprehensive robustness testing

benchmark_paths:
  - data/benchmarks/logic_reasoning.json
  - data/benchmarks/reading_comprehension.json

models:
  - name: anthropic/claude-3-sonnet
    provider: openrouter
    temperature: 0.0
  - name: openai/gpt-4
    provider: openrouter
    temperature: 0.0

transformations:
  enabled_types:
    - language_translation
    - synonym_replacement
    - proper_noun_swap
  synonym_rate: 0.3

max_samples: 100
generate_plots: true
calculate_significance: true
```

```bash
# Run evaluation with configuration
scramblebench evaluate run --config configs/robustness_test.yaml

# Analyze results
scramblebench evaluate analyze robustness_evaluation

# Compare multiple experiments
scramblebench evaluate compare exp1 exp2 exp3
```

### Key Features

- **Multi-Model Testing**: Evaluate multiple LLMs simultaneously
- **Transformation Pipeline**: Language translation, synonym replacement, proper noun swapping
- **Statistical Analysis**: Accuracy, robustness metrics, significance testing
- **Rich Visualizations**: Performance comparisons, degradation heatmaps, interactive dashboards
- **Results Management**: Structured storage, experiment tracking, reproducible configurations

See [`README_EVALUATION.md`](README_EVALUATION.md) and [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) for comprehensive evaluation documentation.

## Project Structure

```
scramblebench/
├── src/scramblebench/           # Main package
│   ├── core/                    # Core benchmark framework
│   │   ├── benchmark.py         # Base benchmark class
│   │   ├── evaluator.py         # Evaluation framework
│   │   └── reporter.py          # Results reporting
│   ├── translation/             # Translation benchmarks
│   │   ├── language_generator.py # Constructed language generation
│   │   ├── translator.py        # Problem translation
│   │   └── benchmark.py         # Translation benchmark
│   ├── longcontext/             # Long context benchmarks  
│   │   ├── document_transformer.py # Document transformation
│   │   ├── qa_transformer.py    # Q&A transformation
│   │   └── benchmark.py         # Long context benchmark
│   ├── llm/                     # LLM integration
│   │   ├── model_interface.py   # Abstract model interface
│   │   └── openrouter_client.py # OpenRouter implementation
│   ├── evaluation/              # Evaluation pipeline
│   │   ├── config.py            # Evaluation configuration
│   │   ├── transformation_pipeline.py # Transformation generation
│   │   ├── openrouter_runner.py # Model evaluation runner
│   │   ├── results.py           # Results management
│   │   ├── metrics.py           # Metrics calculation
│   │   ├── plotting.py          # Visualization generation
│   │   └── runner.py            # Main evaluation orchestrator
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── data_loader.py       # Data loading utilities
├── tests/                       # Test suite
├── configs/                     # Configuration files
├── data/                        # Data directory
│   ├── benchmarks/              # Benchmark datasets
│   ├── languages/               # Generated languages
│   └── results/                 # Benchmark results
└── docs/                        # Documentation
```

## Configuration

ScrambleBench uses YAML configuration files with environment variable support:

```yaml
# config.yaml
benchmark:
  random_seed: 42
  evaluation_mode: "exact_match"
  evaluation_threshold: 0.8

model:
  default_provider: "openrouter"  
  default_model: "openai/gpt-3.5-turbo"
  timeout: 30
  rate_limit: 10.0

data:
  benchmarks_dir: "data/benchmarks"
  results_dir: "data/results"
  max_cache_size: 1000
```

Environment variables:
```bash
export OPENROUTER_API_KEY="your-api-key"
export SCRAMBLEBENCH_LOG_LEVEL="INFO"
export SCRAMBLEBENCH_DATA_DIR="./data"
```

## Benchmark Types

### Translation Benchmarks

Transform problems into constructed languages to avoid training data contamination:

**Language Types:**
- **Substitution**: Simple character/word substitution ciphers
- **Phonetic**: Phonetically plausible transformations  
- **Scrambled**: Systematic character scrambling with rules
- **Synthetic**: Fully artificial languages with grammar

**Example:**
```
Original: "What is the capital of France?"
Translated: "Whot is the kepitol of Fronke?" (substitution)
```

### Long Context Benchmarks

Transform long documents while preserving information content:

**Transformation Types:**
- **Translation**: Convert to constructed languages
- **Paraphrase**: Rephrase content while preserving meaning
- **Structural**: Reorder paragraphs and sections
- **Hybrid**: Combine multiple transformation strategies

## Data Formats

ScrambleBench supports multiple data formats:

### Basic Q&A Format (JSON/JSONL)
```json
{
  "id": "qa_001",
  "question": "What is 2 + 2?", 
  "answer": "4",
  "metadata": {
    "difficulty": "easy",
    "category": "math"
  }
}
```

### Reading Comprehension Format
```json
{
  "id": "rc_001",
  "document": "Long document text...",
  "question": "Question about the document",
  "answer": "Expected answer",
  "answer_type": "extractive"
}
```

### Long Context Format
```json
{
  "id": "doc_001", 
  "document": "Very long document...",
  "questions": [
    {
      "question": "Question 1",
      "answer": "Answer 1",
      "answer_type": "extractive"
    }
  ]
}
```

## Advanced Usage

### Custom Model Integration

```python
from scramblebench.llm.model_interface import ModelInterface, ModelResponse

class CustomModel(ModelInterface):
    def initialize(self):
        # Initialize your model
        return True
    
    def generate(self, prompt, **kwargs):
        # Your model inference logic
        response_text = your_model.generate(prompt)
        return ModelResponse(
            text=response_text,
            metadata={"custom": "metadata"}
        )
    
    # Implement other required methods...

# Register with factory
from scramblebench.llm.model_interface import ModelFactory
ModelFactory.register_model("custom", CustomModel)
```

### Custom Evaluation Metrics

```python
from scramblebench.core.evaluator import Evaluator, EvaluationResult

def custom_evaluator(predicted, expected, **kwargs):
    # Your custom evaluation logic
    score = your_evaluation_function(predicted, expected)
    return EvaluationResult(
        correct=score > 0.8,
        score=score,
        explanation=f"Custom score: {score}",
        metadata={}
    )

# Register custom evaluator
evaluator = Evaluator()
evaluator.register_custom_evaluator("custom", custom_evaluator)
```

### Batch Processing

```python
# Process multiple models
models = [
    OpenRouterClient("openai/gpt-4"),
    OpenRouterClient("anthropic/claude-3-sonnet"),
    OpenRouterClient("meta-llama/llama-2-70b-chat")
]

results = []
for model in models:
    result = benchmark.run(model)
    results.append(result)

# Generate comparative report
from scramblebench.core.reporter import Reporter
reporter = Reporter()
report = reporter.generate_report(results, title="Model Comparison")
```

## API Reference

### Core Classes

- **`BaseBenchmark`**: Abstract base class for all benchmarks
- **`TranslationBenchmark`**: Benchmark using constructed languages
- **`LongContextBenchmark`**: Benchmark for long document understanding
- **`Evaluator`**: Evaluation framework with multiple modes
- **`Reporter`**: Results reporting and visualization
- **`Config`**: Configuration management system
- **`DataLoader`**: Flexible data loading utilities

### Model Integration

- **`ModelInterface`**: Abstract interface for LLM integration
- **`OpenRouterClient`**: OpenRouter API client implementation
- **`DummyModel`**: Testing model for development
- **`ModelFactory`**: Factory for creating model instances

### Transformation Components

- **`LanguageGenerator`**: Constructed language generation
- **`ProblemTranslator`**: Problem translation system
- **`DocumentTransformer`**: Document transformation system
- **`QATransformer`**: Q&A transformation system

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/sibyllinesoft/scramblebench.git
cd scramblebench

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check src/
black --check src/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_translation/
pytest tests/test_longcontext/

# Run with coverage
pytest --cov=scramblebench

# Run integration tests
pytest -m integration
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Examples

See the `examples/` directory for complete examples:

- **Translation Benchmark Example**: End-to-end translation benchmark
- **Long Context Example**: Document transformation and evaluation
- **Custom Model Example**: Integrating custom models
- **Batch Evaluation Example**: Comparing multiple models
- **Configuration Examples**: Various configuration scenarios
- **CLI Demo Script**: Interactive demonstration of CLI features (`examples/cli_demo.py`)

## Troubleshooting

### Common Issues

**ImportError with optional dependencies:**
```bash
# Install optional dependencies
uv sync --group all
# Or specific groups
uv sync --group pandas --group datasets
```

**OpenRouter API issues:**
```bash
# Set API key
export OPENROUTER_API_KEY="your-key"
# Or in config file
echo "model:\n  api_key: your-key" >> config.yaml
```

**Memory issues with large documents:**
```yaml
# In config file
longcontext:
  chunk_long_documents: true
  chunk_size: 5000
```

**Slow evaluation:**
```yaml
# Increase parallelism and caching
model:
  rate_limit: 20.0
data:
  max_cache_size: 5000
```

### Getting Help

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/sibyllinesoft/scramblebench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sibyllinesoft/scramblebench/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ScrambleBench in your research, please cite:

```bibtex
@software{scramblebench2024,
  title={ScrambleBench: Contamination-Resistant LLM Evaluation Through Constructed Languages},
  author={Rice, Nathan},
  year={2024},
  url={https://github.com/sibyllinesoft/scramblebench}
}
```

## Acknowledgments

- OpenRouter for providing access to multiple LLM providers
- The research community for highlighting training data contamination issues
- Contributors and users who help improve the toolkit

---

**ScrambleBench** - Reliable LLM evaluation through contamination-resistant benchmarking.