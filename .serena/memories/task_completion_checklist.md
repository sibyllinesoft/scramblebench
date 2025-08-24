# Task Completion Checklist

When completing any development task in ScrambleBench, ensure the following steps are performed:

## Code Quality Verification
- [ ] **Format Code**: Run `black src/ tests/` and `isort src/ tests/`
- [ ] **Lint Code**: Run `ruff check src/ tests/` and fix any issues
- [ ] **Type Check**: Run `mypy src/` and resolve type errors
- [ ] **Security Scan**: Run `bandit -r src/` and address security issues

## Testing Requirements
- [ ] **Unit Tests**: Add/update unit tests for new functionality
- [ ] **Integration Tests**: Add integration tests for API interactions
- [ ] **Test Coverage**: Ensure >90% coverage with `pytest --cov=scramblebench`
- [ ] **All Tests Pass**: Run `pytest` and ensure all tests pass

## Documentation Updates
- [ ] **Docstrings**: Add comprehensive docstrings for new functions/classes
- [ ] **Type Annotations**: Ensure all functions have complete type annotations
- [ ] **README Updates**: Update README.md if public API changes
- [ ] **Config Examples**: Update configuration examples if needed

## Pre-commit Validation
- [ ] **Pre-commit Hooks**: Run `pre-commit run --all-files`
- [ ] **Commit Message**: Follow conventional commit format
- [ ] **No Secrets**: Ensure no secrets or API keys are committed

## Experimental Code Migration (Phase 2 Focus)
- [ ] **Legacy Scripts**: Identify experimental scripts that need refactoring
- [ ] **Framework Integration**: Move working code into src/scramblebench/
- [ ] **Configuration**: Convert hardcoded values to YAML configuration
- [ ] **CLI Integration**: Add commands to scramblebench CLI
- [ ] **Test Migration**: Move/create tests in tests/ directory

## Research Reproducibility
- [ ] **Configuration Files**: Ensure experiments can be reproduced from config
- [ ] **Data Versioning**: Track experiment metadata and environment info  
- [ ] **Result Validation**: Verify output formats match expected schema
- [ ] **Documentation**: Update methodology documentation if needed

## Performance Considerations
- [ ] **Memory Usage**: Check for memory leaks in long-running processes
- [ ] **API Rate Limits**: Respect rate limits for external APIs
- [ ] **Caching**: Implement appropriate caching for expensive operations
- [ ] **Async Support**: Use async/await for I/O bound operations when possible