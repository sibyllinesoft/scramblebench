# 🔬 ScrambleBench Contamination Detection System

## Overview

This enhanced contamination detection system demonstrates ScrambleBench's core capability: **detecting potential training data contamination** by comparing model performance on original questions versus semantically equivalent but surface-form different ("scrambled") versions.

The system provides:
- ✅ **Side-by-side evaluation** of original and scrambled questions
- ✅ **Multiple scrambling techniques** (translation, synonym replacement, proper noun swapping)
- ✅ **Statistical significance testing** for performance differences
- ✅ **Comprehensive contamination resistance analysis**
- ✅ **Detailed reporting** with actionable insights

## 🧠 Core Concept: Contamination Detection Through Scrambling

### The Problem: Training Data Contamination
When models are trained on data that overlaps with evaluation benchmarks, they may **memorize specific surface forms** rather than learning genuine understanding. This leads to:
- Inflated evaluation scores
- Poor generalization to new formulations
- Unreliable assessment of true capabilities

### The Solution: Semantic-Preserving Scrambling
ScrambleBench creates **semantically equivalent but surface-different** versions of questions through:

1. **Constructed Language Translation** - Convert to artificial languages while preserving logical structure
2. **Synonym Replacement** - Replace words with synonyms maintaining meaning
3. **Proper Noun Swapping** - Replace names with thematically appropriate alternatives

### Detection Logic
- **Large Performance Drop** → Likely contamination (model memorized exact wording)
- **Stable Performance** → Clean model (understands semantic content)

## 🗂️ System Components

### 1. Main Analysis Engine
**File:** `run_scrambled_comparison.py`
- Orchestrates the complete contamination detection pipeline
- Loads benchmark questions and generates scrambled versions
- Evaluates models on both original and scrambled questions
- Performs statistical analysis and generates reports

### 2. Advanced Analysis Module
**File:** `contamination_analyzer.py`
- Deep statistical analysis of contamination patterns
- Outlier detection and consistency testing
- Advanced visualizations and insights generation
- Vulnerability matrix creation

### 3. Pipeline Orchestration
**File:** `run_contamination_detection.sh`
- Complete automated pipeline with prerequisite checks
- Progress monitoring and error handling
- Comprehensive report generation
- Results presentation and cleanup

### 4. Configuration
**File:** `configs/evaluation/contamination_detection_gemma3_4b.yaml`
- Comprehensive configuration for gemma3:4b analysis
- Multiple transformation types and parameters
- Statistical analysis settings

## 🚀 Quick Start

### Prerequisites
1. **Ollama** installed and running
2. **gemma3:4b** model available (`ollama pull gemma3:4b`)
3. **Python dependencies** (numpy, pandas, matplotlib, etc.)

### Run Analysis
```bash
# Full comprehensive analysis
./run_contamination_detection.sh

# Quick test (5 samples, 1 transformation)
./run_contamination_detection.sh --quick

# Skip prerequisite checks
./run_contamination_detection.sh --skip-checks
```

### Alternative: Direct Python Execution
```bash
# Basic analysis
python run_scrambled_comparison.py --config configs/evaluation/contamination_detection_gemma3_4b.yaml

# Quick test
python run_scrambled_comparison.py --quick --verbose

# Advanced analysis (after basic analysis)
python contamination_analyzer.py data/reports/contamination_detection_gemma3_4b/contamination_report.json
```

## 📊 Understanding Results

### Contamination Resistance Score
- **0.8 - 1.0**: ✅ **Excellent** resistance (low contamination risk)
- **0.6 - 0.8**: ⚠️ **Good** resistance (moderate contamination risk)
- **0.0 - 0.6**: 🚨 **Poor** resistance (high contamination risk)

### Performance Drop Analysis
- **< 5%**: Normal variation, no concern
- **5% - 15%**: Moderate concern, investigate further
- **> 15%**: Significant concern, likely contamination

### Statistical Significance
- **p < 0.05**: Statistically significant performance drop (evidence of contamination)
- **p ≥ 0.05**: No significant evidence of contamination

## 📁 Output Files

After running the analysis, you'll find:

```
data/reports/contamination_detection_gemma3_4b/
├── contamination_report.json          # Main analysis results
├── detailed_results.json              # Raw evaluation data
├── CONTAMINATION_SUMMARY.md           # Human-readable summary
├── FINAL_CONTAMINATION_REPORT.md      # Comprehensive report
├── advanced_contamination_analysis.json # Advanced insights
├── advanced_analysis/                 # Visualizations
│   ├── resistance_radar_chart.png
│   ├── performance_analysis.png
│   ├── vulnerability_heatmap.png
│   └── statistical_analysis.png
└── config.yaml                       # Analysis configuration
```

## 🔍 Interpreting Contamination Indicators

### High Contamination Risk Indicators
- **Large performance drops** (>15%) on scrambled questions
- **Statistical significance** (p < 0.05) of performance differences
- **Low resistance scores** (<0.6) across multiple transformations
- **Consistent pattern** of degradation across transformation types

### Low Contamination Risk Indicators
- **Stable performance** (<5% drop) on scrambled questions
- **High resistance scores** (>0.8) across transformations
- **No statistical significance** in performance differences
- **Consistent robustness** across different scrambling methods

## 🎯 Transformation Types Explained

### 1. Language Translation
**Purpose:** Tests if model relies on specific language patterns
**Method:** Converts questions to constructed languages while preserving logical structure
**Interpretation:** Large drops suggest over-reliance on English surface forms

### 2. Synonym Replacement
**Purpose:** Tests vocabulary robustness
**Method:** Replaces 35% of words with synonyms
**Interpretation:** Drops indicate memorization of specific word choices

### 3. Proper Noun Swapping
**Purpose:** Tests dependence on specific names/entities
**Method:** Replaces proper nouns with thematically appropriate alternatives
**Interpretation:** Should have minimal impact if model understands logic vs memorizes examples

## 📈 Advanced Analysis Features

The system includes sophisticated analysis capabilities:

### Statistical Analysis
- **Outlier Detection**: Identifies unusual results requiring investigation
- **Effect Size Calculation**: Measures practical significance of differences
- **Confidence Intervals**: Provides uncertainty bounds on measurements
- **Consistency Testing**: Evaluates reliability across transformations

### Pattern Recognition
- **Surface Form Sensitivity**: Quantifies reliance on exact wording
- **Transformation Hierarchy**: Validates expected difficulty relationships
- **Performance Clustering**: Identifies unusual groupings in results

### Visualization Suite
- **Resistance Radar Charts**: Multi-dimensional contamination resistance view
- **Performance Distributions**: Statistical distribution of results
- **Vulnerability Heatmaps**: Risk assessment by transformation type
- **Significance Plots**: Statistical validation visualization

## 🛠️ Customization Options

### Modify Transformations
Edit `contamination_detection_gemma3_4b.yaml`:
```yaml
transformations:
  enabled_types:
    - language_translation
    - synonym_replacement  
    - proper_noun_swap
  
  # Adjust scrambling intensity
  synonym_rate: 0.35           # 35% of words replaced
  language_complexity: 5       # Medium complexity languages
```

### Adjust Sample Size
```yaml
max_samples: 25              # Number of questions to analyze
```

### Modify Statistical Thresholds
```yaml
metrics_config:
  degradation_threshold: 0.05  # 5% degradation threshold
  significance_level: 0.05     # 95% confidence level
```

## 🔧 Troubleshooting

### Common Issues

**Error: "Model not available"**
```bash
ollama pull gemma3:4b
```

**Error: "Ollama service not running"**
```bash
ollama serve
```

**Error: "Python dependencies missing"**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

**Error: "Benchmark file not found"**
- Ensure `data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json` exists
- Run data collection scripts if needed

### Performance Issues
- Use `--quick` mode for testing (5 samples instead of 25)
- Reduce `max_samples` in configuration
- Ensure sufficient disk space for results

## 🔬 Research Background

This implementation is based on research in:
- **Data Contamination Detection** in language model evaluation
- **Semantic-Preserving Transformations** for robustness testing
- **Statistical Methods** for contamination assessment

### Key Principles
1. **Semantic Equivalence**: Transformations preserve logical meaning
2. **Surface Diversity**: Changes appearance while maintaining content
3. **Statistical Rigor**: Proper hypothesis testing for contamination claims
4. **Practical Utility**: Actionable insights for model development

## 📚 Example Use Cases

### 1. Model Development
- Test new models before deployment
- Compare contamination resistance across architectures
- Validate training data cleanliness

### 2. Benchmark Validation
- Assess benchmark integrity
- Identify potentially compromised evaluation sets
- Establish contamination baselines

### 3. Research Applications
- Study memorization vs generalization in language models
- Evaluate data cleaning techniques effectiveness
- Compare robustness across model families

## 🎓 Educational Value

This system demonstrates:
- **Contamination Detection Methodology**: Complete pipeline from concept to results
- **Statistical Analysis**: Proper significance testing and effect size calculation
- **Visualization Techniques**: Comprehensive result presentation
- **Software Engineering**: Modular, maintainable analysis system

## 🔮 Future Extensions

Potential enhancements:
- **Additional Scrambling Methods**: Paraphrasing, structural changes
- **Multi-Model Comparison**: Parallel analysis across models
- **Interactive Dashboard**: Web-based results exploration
- **Continuous Monitoring**: Automated contamination tracking

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/` directory
3. Examine detailed error messages in console output
4. Verify all prerequisites are properly installed

---

**🎯 The Goal**: Provide a comprehensive, production-ready system for detecting training data contamination using ScrambleBench's semantic-preserving transformation methodology.

This system enables researchers and practitioners to confidently assess whether their models demonstrate genuine understanding or potentially problematic memorization of training examples.