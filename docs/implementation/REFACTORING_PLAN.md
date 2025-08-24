# ScrambleBench Config Refactoring Plan

## Phase 1: Simple Import Replacements (COMPLETED)
These files only need import statement updates:

### Files Updated:
- ✅ `src/scramblebench/cli.py` - Updated imports and API calls

## Phase 2: Complex Structural Changes (IN PROGRESS) 
These files require significant restructuring due to API differences:

### Test Files (Require Config Structure Updates):
- `tests/test_evaluation/test_runner.py` - Uses old EvaluationConfig structure
- `tests/test_evaluation/test_evaluation_pipeline.py` - Complex test configurations
- `tests/test_integration.py` - Integration test configs

### Script Files (Require Config Migration):
- `run_ollama_benchmark.py` - Loads YAML configs with old structure
- `run_ollama_e2e_test.py` - Similar YAML config loading
- `run_scrambled_comparison.py` - Creates EvaluationConfig programmatically

### Documentation Files (Lower Priority):
- `docs/user_guide/*.rst` - Example code using old imports
- `docs/tutorials/*.rst` - Tutorial examples

## Phase 3: Legacy Files that Import from Deprecated Configs (POSTPONED)
These files import from deprecated configs but may be using systems that work:

- Files using `utils.config.Config` class - mostly test files
- Files using `core.config` classes - these may still work since core/config.py is comprehensive

## Identified Issues:

1. **API Mismatch**: The unified_config uses dataclasses with different field names and structure
2. **Enum vs String**: Old system uses `ModelProvider` enum, new uses strings
3. **Configuration Structure**: EvaluationConfig vs ScrambleBenchConfig have different hierarchies
4. **Test Complexity**: Many tests create complex configurations that don't map 1:1

## Strategy (REVISED):

The unified_config.py uses a completely different structure (dataclasses vs Pydantic, different field names, different hierarchy). This requires a different approach:

### Phase 1: Import Updates with Backward Compatibility ✅
1. ✅ Update CLI imports that just need class name changes
2. 🔄 Create backward compatibility utilities in unified_config.py
3. 🔄 Update files that can work with simple import changes

### Phase 2: Create Migration Utilities 🔄
1. Create a `from_evaluation_config()` class method in ScrambleBenchConfig
2. Create utilities to convert old YAML format to new format
3. Add deprecated import warnings but maintain compatibility

### Phase 3: Gradual Migration ⏸️
1. Migrate script files to use new config format one by one
2. Update test files to use new structure
3. Update documentation examples

### Phase 4: Remove Deprecated Files ⏸️
1. Only remove deprecated config files after all migration is complete
2. Ensure no breaking changes for users

## Files Requiring YAML Config Format Updates:
- Configuration YAML files in configs/ directory need to match unified_config format
- Script files loading these configs need to use new API
