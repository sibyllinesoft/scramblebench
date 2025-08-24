# Contamination Detection Results Visualization

This document explains how to use the contamination detection visualization tool to analyze model performance on scrambled vs original questions.

## Overview

The `visualize_contamination_results.py` script reads JSON results from contamination tests and creates comprehensive visualizations showing:

- **Performance Comparison**: Original vs scrambled question performance
- **Category Analysis**: Which question types are most vulnerable to contamination
- **Risk Assessment**: Identification of questions showing potential memorization
- **Visual Charts**: Bar charts and performance comparisons
- **Summary Reports**: Detailed analysis with actionable insights

## Usage

### Basic Usage

```bash
# Use the default test results
python3 visualize_contamination_results.py

# Analyze a specific results file
python3 visualize_contamination_results.py data/results/contamination_tests/your_test_file.json
```

### What the Script Produces

1. **Console Output**: Rich, colorized tables and analysis
2. **Summary Report**: Text file with detailed statistics
3. **Visual Charts**: PNG charts showing performance comparisons
4. **Key Insights**: Actionable recommendations based on findings

## Understanding the Results

### Performance Drop Categories

- **🚨 HIGH DROP (>50%)**: Severe performance drops suggest possible memorization
- **⚠️ Moderate Drop (30-50%)**: Significant drops warrant investigation
- **✓ Minor Drop (<30%)**: Normal performance variation
- **✅ No Change (0%)**: Consistent performance indicates robust reasoning

### Contamination Risk Levels

- **🔴 CRITICAL (>70% drop)**: Strong contamination signal - investigate training data
- **🟠 HIGH (50-70% drop)**: Significant drop indicates potential contamination
- **🟡 MODERATE (30-50% drop)**: Moderate drop warrants investigation
- **🟢 LOW (<30% drop)**: Minor performance variation within normal range

### Category Vulnerability Analysis

- **🔴 HIGH RISK**: Categories showing >40% average performance drop
- **🟡 MODERATE**: Categories with 20-40% average drop
- **🟢 LOW RISK**: Categories with 5-20% drop
- **🛡️ ROBUST**: Categories showing <5% drop or improved performance

## Interpreting Results

### Good Signs (Low Contamination Risk)
- Consistent performance across original and scrambled questions
- No significant performance drops (>30%)
- Robust performance across all question categories
- Low overall contamination rate (<20%)

### Warning Signs (Potential Contamination)
- Large performance drops (>50%) on specific questions
- Certain categories showing consistent vulnerability
- High contamination rate (>20%)
- Perfect scores on original but poor scores on scrambled versions

### What to Do with Results

#### If No Contamination Detected ✅
- Model shows good generalization ability
- Continue monitoring with additional test cases
- Consider expanding test coverage to more domains

#### If Contamination Suspected 🚨
- Review training data for potential overlaps with test questions
- Investigate high-drop questions for common patterns
- Consider retraining with better data filtering
- Expand contamination testing with different scrambling methods

## Technical Details

### Test Methodology
The contamination detection uses several scrambling methods:
- **Synonym Replacement**: Key words replaced with synonyms
- **Word Order Changes**: Sentence structure modifications
- **Minor Word Substitution**: Similar meaning word swaps

### Scoring Method
- **Keyword Matching**: Responses evaluated based on presence of key concepts
- **Contamination Threshold**: Performance drops >30% trigger contamination flags
- **Statistical Analysis**: Category-wise and overall performance comparison

### Output Files

```
data/results/contamination_analysis/
├── contamination_analysis_YYYYMMDD_HHMMSS.txt  # Detailed text report
├── contamination_analysis_YYYYMMDD_HHMMSS.png  # Visual charts
```

## Example Interpretation

```
Model: gemma3:4b
Contamination Rate: 20.0% (2/10 questions)
Average Performance Drop: 11.4%

Key Findings:
- Logic reasoning shows 37.5% average drop (MODERATE risk)
- Mathematical reasoning shows 22.2% average drop (MODERATE risk)
- Pattern recognition shows 0% drop (ROBUST)

Recommendations:
- Investigate logic_02 and math_03 questions (75% and 67% drops)
- Check training data for similar logical reasoning patterns
- Model appears generally robust but has specific vulnerabilities
```

## Customization

You can modify the script to:
- Change contamination thresholds
- Add new visualization types
- Customize color schemes and styling
- Export results in different formats
- Add statistical significance tests

## Dependencies

- `rich`: For colorized console output and tables
- `matplotlib`: For chart generation
- `pathlib`, `json`, `datetime`: Standard library modules

No additional data science libraries (pandas, seaborn) required - the script uses built-in Python functionality for maximum compatibility.