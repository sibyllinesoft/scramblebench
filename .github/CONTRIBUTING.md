# Contributing to ScrambleBench

Thank you for your interest in contributing to ScrambleBench! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up development environment**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/scramblebench.git
   cd scramblebench
   pip install -e ".[dev]"
   pre-commit install
   ```
4. **Create a feature branch**: `git checkout -b feature/your-feature-name`
5. **Make your changes** following our guidelines below
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Installation
```bash
# Install development dependencies
pip install -e ".[dev,docs,nlp]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import scramblebench; print('✅ Installation successful')"
```

### Running Tests
```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/test_core/  # Specific directory
```

## 📝 Code Standards

### Code Style
We use automated code formatting and linting:

- **Black**: Code formatting
- **Ruff**: Fast linting and import sorting  
- **MyPy**: Type checking
- **Bandit**: Security analysis

Pre-commit hooks will automatically run these tools. You can also run them manually:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### Documentation
- Use **Google-style docstrings** for all public functions and classes
- Include **type hints** for all function parameters and return values
- Add **examples** in docstrings for complex functions
- Update relevant **README** sections for new features

### Testing Requirements
- **Write tests** for all new functionality
- Maintain **>90% test coverage** for new code
- Include both **unit** and **integration** tests
- Use descriptive test names: `test_translation_benchmark_handles_empty_input`

## 🐛 Reporting Issues

### Bug Reports
Use the **Bug Report** template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment information (Python version, OS, etc.)
- Relevant log output or error messages

### Feature Requests
Use the **Feature Request** template and describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered
- Any relevant research or examples

## 🔄 Pull Request Process

1. **Create descriptive PR title**: `Add translation benchmark for constructed languages`
2. **Fill out PR template** completely
3. **Ensure all checks pass**:
   - ✅ Tests pass
   - ✅ Code style checks pass
   - ✅ Type checking passes
   - ✅ Security scan passes
4. **Update documentation** if needed
5. **Add/update tests** for your changes
6. **Request review** from maintainers

### PR Guidelines
- **Keep PRs focused**: One feature or fix per PR
- **Write clear commit messages**: Use conventional commits format
- **Update CHANGELOG.md** for user-facing changes
- **Test thoroughly**: Include edge cases and error conditions

## 🏗️ Architecture Guidelines

ScrambleBench follows clean architecture principles:

```
src/scramblebench/
├── core/          # Domain logic and interfaces
├── evaluation/    # Evaluation pipeline
├── llm/          # LLM provider integrations
├── transforms/   # Text transformations
├── analysis/     # Statistical analysis
└── utils/        # Shared utilities
```

### Key Principles
- **Separation of concerns**: Keep domain logic separate from I/O
- **Type safety**: Use type hints and generic types
- **Error handling**: Use explicit error types and meaningful messages
- **Performance**: Profile performance-critical code
- **Documentation**: Code should be self-documenting with clear names

## 🎯 Areas for Contribution

### High-Impact Areas
- **New LLM providers**: Add support for additional providers
- **Evaluation metrics**: Implement new statistical measures
- **Text transformations**: Add novel scrambling/transformation methods
- **Visualization**: Enhance analysis and reporting capabilities
- **Documentation**: Improve guides, tutorials, and examples

### Good First Issues
Look for issues labeled `good first issue` or `help wanted` on GitHub.

## 📚 Resources

- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory  
- **Configuration**: `/configs/examples/` directory
- **Architecture**: See `docs/development/architecture.rst`

## 🤝 Community

- **Be respectful**: Follow our Code of Conduct
- **Be patient**: Maintainers review PRs in their spare time
- **Be collaborative**: Engage constructively in discussions
- **Be helpful**: Help other contributors when you can

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check docs first for common questions

## ⚖️ Legal

By contributing to ScrambleBench, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ScrambleBench! Your efforts help advance LLM evaluation and research. 🚀