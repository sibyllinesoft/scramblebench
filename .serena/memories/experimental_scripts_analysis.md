# Experimental Scripts Analysis

## Current Experimental Scripts (Root Directory)
These scripts represent proof-of-concept implementations that need to be integrated into the production framework:

### Core Working Scripts
1. **simple_ollama_client.py** - Working Ollama integration with ModelResponse dataclass
2. **precision_threshold_test.py** - Academic-rigorous testing framework for measuring reasoning boundaries
3. **gemma27b_threshold_explorer.py** - Model-specific threshold analysis
4. **comparative_model_analysis.py** - Cross-model comparison framework
5. **language_dependency_atlas_demo.py** - Main experimental framework demo

### Specialized Analysis Scripts  
6. **complete_dictionary_scramble.py** - Dictionary-based text scrambling
7. **double_scramble_test.py** - Multi-level scrambling analysis
8. **fair_dictionary_scramble.py** - Balanced scrambling methodology
9. **devastating_dictionary_analysis.py** - Extreme scrambling analysis
10. **paper_analysis.py** - Academic paper summary generation

### Support Scripts
11. **contamination_analyzer.py** - Training data contamination detection
12. **debug_scoring.py** - Scoring system debugging utilities
13. **quick_summary.py** - Result summarization utilities
14. **visualize_contamination_results.py** - Results visualization

## Key Components to Extract

### 1. Model Integration Layer
- SimpleOllamaClient class with proper error handling
- ModelResponse dataclass structure  
- Session management and timeout handling

### 2. Experiment Framework
- PrecisionThresholdTester design pattern
- Systematic test case structure
- Academic rigor in hypothesis testing

### 3. Text Transformation Engine
- Dictionary-based scrambling algorithms
- Multi-level transformation pipelines
- Semantic preservation metrics

### 4. Analysis and Reporting
- Statistical analysis frameworks
- Academic paper generation
- Visualization and plotting utilities

### 5. Configuration Management
- YAML-based configuration patterns
- Environment variable integration
- Experiment reproducibility tracking

## Integration Priority
1. **High Priority**: simple_ollama_client.py, precision_threshold_test.py
2. **Medium Priority**: comparative_model_analysis.py, contamination_analyzer.py
3. **Lower Priority**: Specialized analysis and debugging scripts

## Refactoring Strategy
- Extract core classes into src/scramblebench/
- Convert scripts to CLI commands
- Move hardcoded configs to YAML files
- Add comprehensive tests for all functionality