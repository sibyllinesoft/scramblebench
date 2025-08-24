# S8 Statistical Analysis - Quick Start Guide

## 🚀 Getting Started with S8

The S8 analysis pipeline provides academic-grade statistical modeling to discover LLM scaling patterns without presupposing thresholds. This guide will get you up and running quickly.

## ⚡ Installation

### 1. Install Dependencies

Run the installation script to install required Python packages:

```bash
# Review the script first
cat install-s8-deps.sh

# Make it executable and run
chmod +x install-s8-deps.sh
./install-s8-deps.sh
```

### 2. Verify Installation

Test that all S8 components are working:

```bash
python3 test_s8_basic.py
```

You should see:
```
✓ ALL TESTS PASSED - S8 analysis components ready!
```

## 📊 Try the Demo

Run the complete S8 pipeline on synthetic data:

```bash
# Generate synthetic data and run full analysis
python3 s8_analysis_demo.py --output-dir s8_demo_results

# Check the results
ls s8_demo_results/
```

This creates:
- **CSV exports**: `scaling_summary.csv`, `parameter_estimates.csv`  
- **LaTeX tables**: `scaling_results.tex`, `model_comparison_*.tex`
- **Reports**: `preregistration_*.md`, `analysis_summary_*.md`
- **Manifest**: Complete file listing with checksums

## 🔬 Analyze Your Data

### Command Line Interface

```bash
# Basic analysis of completed evaluation run
scramblebench analyze fit --run-id your_run_id_here

# Advanced analysis with all features
scramblebench analyze fit \
    --run-id your_run_id_here \
    --use-r \
    --bootstrap-samples 5000 \
    --export-latex \
    --export-csv \
    --prereg-report

# Compare multiple runs
scramblebench analyze compare-runs --run-ids run1,run2,run3

# Quick summary
scramblebench analyze summary --run-id your_run_id_here
```

### Python API

```python
from scramblebench.analysis import ScalingAnalyzer, AcademicExporter
from scramblebench.core.database import Database

# Initialize
db = Database("scramblebench.duckdb")
analyzer = ScalingAnalyzer(db, use_r_backend=True)

# Run analysis
results = analyzer.run_full_analysis("your_run_id")

# Export results
exporter = AcademicExporter("output_directory")
files = exporter.export_full_analysis(results, "your_run_id", "git_commit")

# Access findings
for family, best_model in results['best_models_by_family'].items():
    print(f"{family}: {best_model}")
```

## 🧪 Optional: R Backend Setup

For advanced statistical models, install R integration:

```bash
# Install Python-R bridge
python3 -m pip install --user rpy2

# Install R packages (requires R to be installed)
R -e "install.packages(c('lme4', 'mgcv', 'segmented'))"

# Test R integration
python3 -c "
import rpy2.robjects as ro
ro.r('library(lme4)')
print('R backend ready!')
"
```

## 📈 Understanding Results

### Model Types
- **Linear**: Simple linear scaling relationship
- **GLMM**: Hierarchical models accounting for domain/family structure  
- **GAM**: Non-parametric smooth curves detecting non-linear patterns
- **Segmented**: Changepoint detection for threshold effects

### Model Selection
- **AIC/BIC**: Information criteria for model comparison
- **Evidence Strength**: Categorical assessment (very_strong, strong, moderate, weak)
- **Model Weights**: Probability each model is best

### Key Metrics
- **RRS**: Reasoning Robustness Score = Acc_scrambled / Acc_original
- **LDC**: Language Dependency Coefficient = 1 - RRS
- **Δ_para-scram**: Differential metric separating contamination from brittleness

## 🔍 Troubleshooting

### Common Issues

**Import errors**: Install missing packages from requirements-s8.txt
```bash
python3 -m pip install --user -r requirements-s8.txt
```

**R backend fails**: Check R installation and packages
```bash
R --version
R -e "library(lme4); library(mgcv); library(segmented)"
```

**Database not found**: Ensure you have run evaluations first
```bash
scramblebench list-runs  # Check available runs
```

**Memory issues**: Reduce bootstrap samples or use Python backend
```bash
scramblebench analyze fit --run-id ID --bootstrap-samples 1000
```

### Getting Help

1. **Check logs**: S8 provides detailed logging of all operations
2. **Use Python backend**: Add `--no-r` flag to avoid R dependencies
3. **Reduce complexity**: Start with single family analysis
4. **Check data**: Verify your evaluation run contains required data

## 📚 Advanced Usage

### Custom Analysis

```python
# Analyze specific model families
analyzer = ScalingAnalyzer(db)
data = analyzer.prepare_analysis_data(run_id)
llama_data = data[data['model_family'] == 'llama']
results = analyzer._analyze_family(llama_data, 'llama')

# Custom contamination analysis  
contamination = ContaminationAnalyzer()
contamination_results = contamination.analyze_contamination_effects(data)

# Bootstrap specific parameters
bootstrap = BootstrapAnalyzer(n_bootstrap=10000)
cis = bootstrap.bootstrap_model_parameters(data, custom_fit_function)
```

### Research Applications

1. **Scaling Pattern Discovery**: Smooth vs threshold scaling
2. **Architecture Comparison**: Family-specific patterns
3. **Contamination Detection**: Memorization vs reasoning
4. **Capability Prediction**: Forecasting at different scales

## 🎯 Next Steps

1. **Run S8 Demo**: `python3 s8_analysis_demo.py --output-dir demo`
2. **Analyze Your Data**: Use your ScrambleBench evaluation runs
3. **Generate Papers**: Export LaTeX tables for publications
4. **Iterate**: Use bootstrap CIs to validate findings

The S8 pipeline transforms raw evaluation data into publication-quality statistical insights about LLM scaling patterns.

---

**Ready to discover how intelligence scales? Start with the demo!**

```bash
python3 s8_analysis_demo.py --output-dir my_first_s8_analysis
```