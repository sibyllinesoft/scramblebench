# ScrambleBench Project Overview

## Purpose
ScrambleBench is a contamination-resistant LLM evaluation toolkit designed to solve training data contamination problems in benchmark evaluation. The project focuses on two main approaches:

1. **Translation Benchmarks**: Transform problems into constructed languages that preserve logical structure while eliminating memorization
2. **Long Context Benchmarks**: Intelligently modify documents through translation and transformation while maintaining semantic content

The project is currently in Phase 2 transformation from proof-of-concept experimental scripts to production-grade research framework suitable for academic publication.

## Tech Stack
- **Python 3.9+**: Primary language with modern type hints
- **Core Dependencies**: OpenAI, Transformers, PyTorch, Pandas, NumPy, Pydantic v2
- **API Integration**: OpenRouter for multiple LLM access, Ollama for local models
- **CLI Framework**: Typer/Click for command-line interface
- **Testing**: pytest with asyncio support, coverage reporting
- **Visualization**: Matplotlib, Seaborn, Plotly for results analysis
- **Config Management**: YAML-based configuration with environment variable support
- **Development Tools**: Black, Ruff, MyPy, pre-commit hooks

## Project Structure
```
scramblebench/
├── src/scramblebench/           # Main package (production code)
├── tests/                       # Comprehensive test suite
├── language_dependency_atlas/   # Experimental framework (legacy)
├── configs/                     # Configuration files
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── *.py (root level)           # Experimental scripts to be refactored
└── *.json                      # Result files from experiments
```

## Current State
**Production Framework**: Comprehensive src/scramblebench/ with CLI, evaluation pipeline, and LLM integration
**Experimental Scripts**: Multiple standalone .py files implementing proof-of-concept functionality
**Research Infrastructure**: formal_methodology.md, academic paper outline, comprehensive documentation

## Key Research Focus Areas
- Language dependency measurement via graduated scrambling
- Contamination detection through constructed languages  
- Model capability assessment across scales
- Statistical analysis of reasoning vs memorization