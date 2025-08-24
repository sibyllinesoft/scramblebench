# Suggested Commands for ScrambleBench Development

## Development Setup
```bash
# Install dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Install in development mode
pip install -e .

# Install with NLP capabilities
pip install -e ".[nlp]"
```

## Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scramblebench

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow

# Run specific test modules
pytest tests/test_translation/
pytest tests/test_core/
```

## Code Quality Commands
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
ruff check --fix src/ tests/

# Type checking  
mypy src/

# Run pre-commit hooks
pre-commit run --all-files

# Security scanning
bandit -r src/
```

## CLI Usage Examples
```bash
# Generate constructed language
scramblebench language generate mylang --type substitution --complexity 5

# Transform text
scramblebench transform text "What is the capital of France?" mylang

# Run evaluation
export OPENROUTER_API_KEY="your-key"
scramblebench evaluate run --models "anthropic/claude-3-haiku" --benchmarks "data/benchmarks/logic_reasoning.json" --experiment-name "demo" --max-samples 5

# Run with configuration
scramblebench evaluate run --config configs/robustness_test.yaml
```

## Experimental Scripts (Legacy - To Be Refactored)
```bash
# Simple Ollama client test
python simple_ollama_client.py

# Precision threshold testing
python precision_threshold_test.py

# Comparative analysis
python comparative_model_analysis.py

# Atlas demo
python language_dependency_atlas_demo.py
```

## Build and Package
```bash
# Build package
python -m build

# Install from source
pip install -e .
```

## Utility Commands
```bash
# Standard Linux utilities
ls -la
find . -name "*.py" -type f
grep -r "pattern" src/
cd directory_name

# Git operations
git status
git add .
git commit -m "message"
git push origin main
```