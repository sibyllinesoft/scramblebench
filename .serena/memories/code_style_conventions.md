# Code Style and Conventions

## Python Code Style
- **Line Length**: 88 characters (Black standard, some configs show 100)
- **Formatting**: Black with isort for import sorting
- **Type Hints**: Comprehensive type annotations using modern Python 3.9+ syntax
- **Docstrings**: Google-style docstrings for all public functions/classes
- **Import Style**: Absolute imports, grouped by standard/third-party/local

## Code Quality Standards
- **Linting**: Ruff with comprehensive rule set (E, W, F, I, B, C4, UP)
- **Type Checking**: MyPy with strict configuration (disallow_untyped_defs, etc.)
- **Security**: Bandit security scanning, detect-secrets for secret detection
- **Testing**: pytest with 90%+ coverage target, unit/integration/e2e tests

## Naming Conventions
- **Classes**: PascalCase (e.g., `SimpleOllamaClient`, `ModelResponse`)
- **Functions/Variables**: snake_case (e.g., `generate_text`, `model_name`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMEOUT`)
- **Files/Modules**: snake_case (e.g., `simple_ollama_client.py`)

## Architecture Patterns
- **Dataclasses**: Used for data structures (e.g., `ModelResponse`)
- **Protocols/Interfaces**: Abstract base classes for extensibility
- **Error Handling**: Comprehensive exception handling with meaningful error messages
- **Configuration**: YAML-based with Pydantic models for validation
- **Factory Pattern**: Used for model instantiation (`ModelFactory`)

## Documentation Standards
- **README.md**: Comprehensive with examples, installation, usage
- **Docstrings**: Required for all public APIs with parameters, returns, examples
- **Type Annotations**: All function signatures must include complete type information
- **Comments**: Minimal inline comments, prefer self-documenting code

## Testing Philosophy
- **Test Coverage**: Minimum 90% line coverage
- **Test Types**: Unit (core logic), Integration (API interactions), E2E (full workflow)
- **Mocking**: Minimal mocking, prefer real implementations with test containers
- **Fixtures**: Reusable test data and setup