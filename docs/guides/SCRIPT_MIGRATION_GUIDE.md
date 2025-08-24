# Script Migration Guide

This document outlines the migration of standalone scripts to unified CLI commands in ScrambleBench. The CLI refactor consolidates functionality into a structured command hierarchy while preserving all existing capabilities.

## Migration Overview

### New CLI Structure

The ScrambleBench CLI now provides a unified interface with the following command groups:

```
scramblebench
├── evaluate       # Core evaluation pipeline
├── paraphrase     # Paraphrasing pipeline
├── models         # Model management
├── analyze        # Statistical analysis and modeling
├── visualize      # Visualization and figures
├── init-project   # Project initialization
├── create-config  # Configuration generation
├── smoke-test     # Integration testing
├── scaling-survey # Scaling analysis
├── test           # Testing and validation (NEW)
├── benchmark      # Benchmark running (NEW) 
├── analysis       # Advanced analysis (NEW)
└── visualize      # Publication figures (NEW)
```

## Migrated Scripts

### Testing & Validation Scripts → `scramblebench test`

| Original Script | New Command | Description |
|---|---|---|
| `test_contamination_system.py` | `scramblebench test contamination` | Test contamination detection system |
| `validate_implementation.py` | `scramblebench test setup --component all` | Validate setup and dependencies |
| `test_benchmark_setup.py` | `scramblebench test setup --component data` | Test benchmark data availability |
| `validate_s8_setup.py` | `scramblebench test setup --component config` | Validate configuration files |
| `test_s8_basic.py` | `scramblebench test setup --component imports` | Test basic imports |

#### Examples:

```bash
# Quick contamination system validation
scramblebench test contamination --quick

# Full contamination detection analysis  
scramblebench test contamination --config configs/evaluation/contamination_detection_gemma3_4b.yaml

# Test all components
scramblebench test setup --component all

# Test only Ollama integration
scramblebench test setup --component ollama
```

### Benchmark Running Scripts → `scramblebench benchmark`

| Original Script | New Command | Description |
|---|---|---|
| `run_ollama_benchmark.py` | `scramblebench benchmark ollama` | Run Ollama benchmarks |
| `run_custom_ollama_benchmarks.py` | `scramblebench benchmark run` | Run custom benchmark configs |
| `run_ollama_test.py` | `scramblebench benchmark ollama --quick` | Quick Ollama test |
| `run_ollama_e2e_test.py` | `scramblebench benchmark run --config <config>` | End-to-end benchmark test |

#### Examples:

```bash
# Run Ollama benchmark with default model
scramblebench benchmark ollama

# Run with specific model
scramblebench benchmark ollama --model gemma:7b

# Quick test (10 samples)
scramblebench benchmark ollama --quick

# Run custom benchmark configuration
scramblebench benchmark run --config configs/evaluation/custom_config.yaml

# Override model and samples
scramblebench benchmark run --config <config> --model llama2:7b --max-samples 50
```

### Analysis & Comparison Scripts → `scramblebench analysis`

| Original Script | New Command | Description |
|---|---|---|
| `comparative_model_analysis.py` | `scramblebench analysis compare` | Compare model performance |
| `contamination_analyzer.py` | `scramblebench analysis contamination` | Analyze contamination results |
| `compare_model_results.py` | `scramblebench analysis compare --models <list>` | Compare specific models |
| `simple_comparative_analysis.py` | `scramblebench analysis compare --output <file>` | Simple comparison with output |

#### Examples:

```bash
# Compare all models in results directory
scramblebench analysis compare --results-dir results/

# Compare specific models
scramblebench analysis compare --models gpt-4 claude-3 gemini-pro

# Save comparison results
scramblebench analysis compare --output comparison_results.json

# Analyze contamination with custom threshold
scramblebench analysis contamination --threshold 0.01

# Analyze specific contamination results file
scramblebench analysis contamination --results-file results/contamination_results.json
```

### Visualization Scripts → `scramblebench visualize`

| Original Script | New Command | Description |
|---|---|---|
| `create_publication_figures.py` | `scramblebench visualize publication` | Generate publication figures |
| `create_publication_visualizations.py` | `scramblebench visualize publication` | Create publication visualizations |
| `visualize_contamination_results.py` | `scramblebench visualize publication --data-dir <dir>` | Visualize contamination results |
| `generate_figures_batch.py` | `scramblebench visualize publication --format pdf` | Batch figure generation |

#### Examples:

```bash
# Generate publication figures from current directory
scramblebench visualize publication

# Specify data directory and output format
scramblebench visualize publication --data-dir results/ --format pdf

# High-resolution figures
scramblebench visualize publication --dpi 600 --output-dir publication_figures/

# Generate SVG figures
scramblebench visualize publication --format svg
```

## Benefits of CLI Migration

### 1. Unified Interface
- Single entry point for all ScrambleBench functionality
- Consistent command structure and options
- Integrated help system (`scramblebench --help`, `scramblebench test --help`)

### 2. Better User Experience  
- Rich terminal output with colors and tables
- Progress indicators and status messages
- JSON output for programmatic use (`--output-format json`)

### 3. Configuration Management
- Centralized configuration handling
- Command-line overrides for config parameters
- Consistent path resolution and error handling

### 4. Error Handling
- Comprehensive error reporting with context
- Verbose mode for debugging (`--verbose`)
- Graceful handling of missing dependencies

### 5. Integration
- Database integration for result tracking
- Consistent result formats across commands
- Built-in validation and sanity checks

## Usage Patterns

### Basic Usage
```bash
# Get help
scramblebench --help
scramblebench test --help

# Run with verbose output
scramblebench --verbose test setup

# JSON output for scripting
scramblebench --output-format json analysis compare
```

### Configuration Overrides
```bash
# Override data directory
scramblebench --data-dir /custom/path test setup

# Override specific config parameters
scramblebench benchmark run --config base.yaml --model custom-model --max-samples 100
```

### Output Management
```bash
# Quiet mode (minimal output)
scramblebench --quiet benchmark ollama

# Verbose debugging
scramblebench --verbose analysis contamination

# Custom output directory
scramblebench visualize publication --output-dir /path/to/figures/
```

## Migration Checklist

For users migrating from standalone scripts to CLI commands:

### 1. Update Scripts/Workflows
- [ ] Replace standalone script calls with CLI commands
- [ ] Update configuration file paths if needed
- [ ] Test new commands with existing data

### 2. Configuration Updates
- [ ] Verify configuration files work with new CLI
- [ ] Update any hardcoded paths in configs
- [ ] Test configuration overrides

### 3. Output Handling
- [ ] Update result file parsing if needed
- [ ] Verify output formats match expectations
- [ ] Test JSON output mode for automated workflows

### 4. Error Handling
- [ ] Update error handling in wrapper scripts
- [ ] Test failure modes and recovery procedures
- [ ] Verify logging and debugging capabilities

## Backward Compatibility

### Preserved Scripts
Some scripts are preserved for specific use cases:

- **Shell scripts** (`.sh`): Preserved for system integration
- **Demo scripts**: Kept for examples and documentation
- **Legacy analysis**: Archived scripts moved to `scripts/legacy/`

### Script Wrapper (Optional)
For teams requiring gradual migration, create wrapper scripts:

```bash
#!/bin/bash
# Legacy wrapper for run_ollama_benchmark.py
echo "⚠️  run_ollama_benchmark.py is deprecated. Use: scramblebench benchmark ollama"
scramblebench benchmark ollama "$@"
```

## Troubleshooting

### Common Issues

1. **Configuration not found**
   ```bash
   # Ensure configs directory exists and is properly structured
   scramblebench test setup --component config
   ```

2. **Import errors**
   ```bash
   # Test all imports
   scramblebench test setup --component imports
   ```

3. **Missing data**
   ```bash
   # Verify data directory setup
   scramblebench test setup --component data
   ```

4. **Ollama connection issues**
   ```bash
   # Test Ollama integration
   scramblebench test setup --component ollama
   ```

### Getting Help

```bash
# General help
scramblebench --help

# Command-specific help
scramblebench test --help
scramblebench benchmark --help
scramblebench analysis --help

# Verbose error output
scramblebench --verbose <command>
```

## Future Enhancements

The unified CLI provides a foundation for:

- **Plugin system**: Extensible command architecture
- **Configuration templates**: Standard config generation
- **Result management**: Database integration and querying  
- **Workflow automation**: Chained command execution
- **Report generation**: Automated analysis reports

---

This migration consolidates ScrambleBench's functionality while maintaining full backward compatibility and improving the overall user experience.