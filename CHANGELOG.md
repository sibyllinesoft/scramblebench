# Changelog

All notable changes to ScrambleBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Enhanced README with improved badges and documentation
- GitHub community files (CONTRIBUTING.md, SECURITY.md)
- Comprehensive release preparation

## [0.1.0] - 2025-08-24

### Added
- **Translation Benchmarks**: Systematic constructed language generation for contamination-resistant evaluation
- **Long Context Benchmarks**: Document transformation pipeline with Q&A adaptation
- **Multi-Provider LLM Support**: OpenRouter and Ollama integration with unified interface
- **Statistical Analysis Framework**: Bootstrap inference, model comparison, and visualization tools
- **Experiment Tracking**: Comprehensive experiment management with reproducibility features
- **Database Integration**: SQLAlchemy ORM with Alembic migrations for result persistence
- **CLI Interface**: Rich command-line interface with multiple benchmark and analysis commands
- **Academic Export**: Publication-ready visualizations and LaTeX table generation
- **Contamination Detection**: Advanced methods for identifying training data contamination
- **Performance Benchmarking**: Built-in performance monitoring and optimization tools

### Core Components
- **src/scramblebench/core/**: Domain logic, benchmark framework, and evaluation pipeline
- **src/scramblebench/evaluation/**: Evaluation runners, metrics computation, and result analysis  
- **src/scramblebench/llm/**: LLM provider abstractions and client implementations
- **src/scramblebench/transforms/**: Text transformation pipeline including paraphrasing and scrambling
- **src/scramblebench/translation/**: Translation benchmark generation and language construction
- **src/scramblebench/longcontext/**: Long context benchmark creation and document transformation
- **src/scramblebench/analysis/**: Statistical analysis, visualization, and academic reporting
- **src/scramblebench/utils/**: Shared utilities and helper functions

### Testing & Quality Assurance
- **43 test files** with comprehensive unit and integration test coverage
- **Pre-commit hooks** with Black, Ruff, MyPy, and Bandit
- **CI/CD pipeline** with multi-platform testing (Ubuntu, Windows, macOS)
- **Type checking** with strict MyPy configuration
- **Documentation** with Sphinx and ReadTheDocs integration
- **Performance profiling** and benchmark validation

### Documentation
- **Comprehensive README** with quick start guide and examples
- **API documentation** with auto-generated reference
- **Configuration examples** for various use cases
- **Research methodology** documentation for academic use
- **Architecture guides** and contribution guidelines

### Configuration & Examples
- **YAML configuration system** with environment variable support
- **Multiple example configurations** for different evaluation scenarios
- **Benchmark dataset collection** with quality assessment criteria
- **Visualization templates** and publication-ready figures

## [0.0.1-alpha] - Development

### Added
- Initial project structure and core architecture
- Basic translation and long context benchmark prototypes
- Experimental LLM integration and evaluation framework
- Research and development infrastructure

---

## Release Notes

### Version 0.1.0 - Initial Public Release

ScrambleBench 0.1.0 represents the culmination of extensive research and development into contamination-resistant LLM evaluation. This release provides:

**🔬 Research-Grade Evaluation**: Academic-quality statistical analysis with bootstrap inference, effect size calculations, and publication-ready visualizations.

**🌍 Translation Benchmarks**: Revolutionary approach using systematically constructed languages that preserve logical structure while eliminating memorization advantages.

**📚 Long Context Evaluation**: Intelligent document transformation that maintains semantic content while creating novel evaluation scenarios.

**🔌 Multi-Provider Support**: Unified interface for OpenRouter, Ollama, and other LLM providers with consistent evaluation protocols.

**📊 Comprehensive Analysis**: Advanced contamination detection, performance benchmarking, and statistical modeling capabilities.

**🛠️ Production Ready**: Professional CI/CD pipeline, comprehensive testing, type checking, and documentation standards.

This release establishes ScrambleBench as a foundational tool for reliable LLM evaluation in both academic and industrial settings.

---

## Upgrade Guide

### From Development to 0.1.0

This is the initial public release. For users upgrading from development versions:

1. **Configuration Changes**: Review configuration file format changes in `configs/examples/`
2. **Database Migrations**: Run `scramblebench db upgrade` to update database schema
3. **Dependencies**: Update all dependencies with `pip install -U scramblebench[dev]`
4. **CLI Changes**: Review new CLI commands and options with `scramblebench --help`

### Future Releases

Upgrade guides for future releases will be provided here with specific migration instructions and breaking change documentation.

---

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for information on how to contribute to ScrambleBench.

## Support

For questions, bug reports, and feature requests, please use [GitHub Issues](https://github.com/sibyllinesoft/scramblebench/issues).