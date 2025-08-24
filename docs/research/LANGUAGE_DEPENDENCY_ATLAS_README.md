# Language Dependency Atlas 🧠

**A Comprehensive LLM Reasoning Benchmarking Suite**

Based on the shocking discovery that models perform WORSE when given translation help, the Language Dependency Atlas provides systematic measurement of how language abstraction affects reasoning across cognitive domains.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# 1. Setup and validation
python setup_language_dependency_atlas.py

# 2. Run quick demo
python language_dependency_atlas_demo.py --quick --model phi3:3.8b

# 3. Run comprehensive benchmark
python language_dependency_atlas_demo.py --model phi3:14b
```

## 📖 Overview

The Language Dependency Atlas systematically tests LLM reasoning robustness by applying **6 graduated levels of language scrambling** across **7 cognitive domains**. This reveals whether models truly understand concepts or simply pattern-match on familiar language.

### Key Innovation: Graduated Scrambling

- **Level 0**: Original text (baseline)
- **Level 1**: Simple synonyms (`easy` → `simple`, `big` → `large`)
- **Level 2**: Word order shuffling with meaning preserved
- **Level 3**: Moderate vocabulary replacement (50% words changed)
- **Level 4**: Complete dictionary substitution (100% words changed but consistent)
- **Level 5**: Abstract symbols + numerals (`What is 5 + 3?` → `⚡ ∆ 5 + 3?`)

### Cognitive Domains Tested

1. **Mathematics** - Arithmetic, algebra, geometry, word problems
2. **Logic** - Syllogisms, propositional logic, fallacy detection
3. **Reading Comprehension** - Fact extraction, inference, analysis
4. **Common Sense** - Physical and social reasoning
5. **Spatial Reasoning** - Mental rotation, arrangements
6. **Temporal Reasoning** - Time sequences, durations
7. **Causal Reasoning** - Cause-effect, predictions

## 🎯 What It Measures

### Core Metrics

- **Language Dependency Coefficient** (0-1): How much performance relies on natural language patterns
- **Contamination Resistance Score** (0-1): Robustness to unfamiliar language
- **Critical Scrambling Threshold**: Level where performance breaks down significantly
- **Domain-Specific Dependencies**: Which reasoning types are most language-dependent

### Statistical Rigor

- Statistical significance testing with multiple comparison corrections
- Confidence intervals and effect size calculations
- Contamination pattern detection
- Performance contour mapping across domains and scrambling levels

## 📊 Example Results

```
Model: phi3:3.8b
Overall Language Dependency: 73.4%
Contamination Resistance: 41.2%

Domain Performance:
Mathematics     | Original: 82.1% | Abstract: 23.4% | Dependency: 71.5%
Logic           | Original: 69.3% | Abstract: 31.2% | Dependency: 55.0%
Reading Comp.   | Original: 91.5% | Abstract: 15.8% | Dependency: 82.7%
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.9+**
- **Ollama** for local model inference
- Required packages: `numpy`, `scipy`, `matplotlib`, `seaborn`, `plotly`

### Installation

```bash
# 1. Clone the repository (if not already done)
git clone <repository-url>
cd scramblebench

# 2. Install Python dependencies
pip install numpy scipy matplotlib seaborn plotly pandas requests pyyaml

# 3. Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# 4. Pull recommended models
ollama pull phi3:3.8b      # Fast, balanced
ollama pull phi3:14b       # High quality
ollama pull llama3.2:1b    # Ultra-fast testing

# 5. Validate setup
python setup_language_dependency_atlas.py
```

## 🎮 Usage Examples

### Basic Usage

```python
from language_dependency_atlas import LanguageDependencyAtlas
from scramblebench.llm import OllamaClient

# Initialize model and atlas
model = OllamaClient("phi3:3.8b")
model.initialize()

atlas = LanguageDependencyAtlas(model)

# Run comprehensive benchmark
results = atlas.run_comprehensive_benchmark(
    domains=['mathematics', 'logic'], 
    questions_per_domain=25
)

# Analyze results
analysis = atlas.analyze_results(results)
print(f"Language dependency: {analysis.overall_language_dependency:.1%}")
```

### Advanced Configuration

```python
from language_dependency_atlas import BenchmarkConfig
from language_dependency_atlas.core.models import DomainType, ScramblingLevel

# Custom configuration
config = BenchmarkConfig(
    domains=[DomainType.MATHEMATICS, DomainType.LOGIC],
    scrambling_levels=[ScramblingLevel.ORIGINAL, ScramblingLevel.COMPLETE],
    questions_per_domain_per_difficulty=50,
    parallel_execution=True,
    confidence_level=0.99
)

atlas = LanguageDependencyAtlas(model, config)
results = atlas.run_benchmark()
```

### Visualization & Reporting

```python
from language_dependency_atlas.analysis import ContourMapper, ReportGenerator

# Generate interactive visualizations
mapper = ContourMapper(interactive=True)
contour_html = mapper.generate_performance_contour(analysis.contour_data)

# Generate comprehensive reports
generator = ReportGenerator()
reports = generator.generate_comprehensive_report(
    analysis, 
    output_dir=Path("results/"),
    include_visualizations=True
)
```

## 📈 Output & Analysis

### Comprehensive Reports

The system generates multiple output formats:

- **HTML Dashboard**: Interactive analysis with visualizations
- **Executive Summary**: Markdown summary with key findings
- **CSV Data**: Raw results for further analysis
- **JSON Export**: Complete data for integration
- **LaTeX Tables**: Research-ready publication tables

### Visualizations

- **Performance Contour Maps**: 2D heatmaps showing accuracy across domains and scrambling levels
- **3D Dependency Surfaces**: Interactive 3D visualization of language dependency
- **Degradation Curves**: Line plots showing performance drops by scrambling level
- **Statistical Summaries**: Effect sizes, confidence intervals, significance tests

### Key Insights Generated

- Identification of most/least language-dependent domains
- Detection of potential training data contamination
- Statistical significance of performance differences
- Recommendations for model improvement
- Critical thresholds where reasoning breaks down

## 🔬 Scientific Applications

### Research Uses

- **Model Evaluation**: Systematic assessment of reasoning robustness
- **Contamination Detection**: Identify over-reliance on training patterns  
- **Cognitive Assessment**: Map model reasoning across human-like domains
- **Comparison Studies**: Standardized benchmarking across different models

### Publication Ready

- Rigorous statistical methodology with proper corrections
- LaTeX table generation for research papers
- Comprehensive methodology documentation
- Reproducible benchmarking protocols

## 🎛️ Configuration Options

### Model Configuration

```yaml
model:
  name: "phi3:3.8b"
  temperature: 0.0    # Deterministic for benchmarking
  max_tokens: 1000
  timeout: 30
```

### Benchmark Configuration

```yaml
benchmark:
  domains: ["mathematics", "logic", "reading_comprehension"]
  scrambling_levels: [0, 1, 2, 3, 4, 5]
  questions_per_domain_per_difficulty: 25
  parallel_execution: true
  max_concurrent_requests: 3
```

### Analysis Configuration

```yaml
analysis:
  confidence_level: 0.95
  statistical_significance_threshold: 0.05
  multiple_comparison_correction: "bonferroni"
```

## 🤝 Integration with ScrambleBench

The Language Dependency Atlas seamlessly integrates with the existing ScrambleBench infrastructure:

- **Ollama Client**: Leverages robust local model integration
- **Evaluation Pipeline**: Extends existing benchmarking framework
- **Result Storage**: Compatible with existing result formats
- **Configuration System**: Uses familiar YAML configuration

## 🚀 Performance

### Benchmarking Speed

- **Quick Demo**: ~2-5 minutes (Mathematics + Logic, 5 questions each)
- **Comprehensive**: ~15-30 minutes (4 domains, 25 questions each)
- **Full Suite**: ~45-90 minutes (7 domains, 25 questions each)

### Recommended Models by Use Case

```bash
# Ultra-fast development testing
ollama pull llama3.2:1b      # <30 seconds for quick demo

# Balanced development/testing  
ollama pull phi3:3.8b        # ~2-5 minutes for quick demo

# High-quality benchmarking
ollama pull phi3:14b         # ~10-15 minutes for comprehensive test
```

## 📚 Documentation Structure

```
language_dependency_atlas/
├── core/                    # Core framework components
│   ├── framework.py        # Main orchestration 
│   ├── scrambling.py       # Text scrambling engine
│   ├── question_bank.py    # Question management
│   └── models.py          # Data models
├── domains/                # Domain-specific implementations
│   ├── mathematics.py     # Math reasoning questions
│   ├── logic.py          # Logic puzzles and syllogisms
│   └── ...               # Other cognitive domains
├── analysis/              # Statistical analysis and visualization
│   ├── statistics.py     # Rigorous statistical tests
│   ├── visualization.py  # Contour maps and plots
│   └── reporting.py      # Multi-format report generation
└── config/               # Configuration templates
```

## 🤔 Interpreting Results

### Language Dependency Coefficient

- **0.0-0.3**: Robust reasoning, minimal language dependence
- **0.3-0.6**: Moderate dependence on familiar language patterns  
- **0.6-0.8**: High dependence, may struggle with novel phrasings
- **0.8-1.0**: Extreme dependence, likely memorizing training patterns

### Contamination Indicators

Watch for these warning signs:
- Non-monotonic degradation (performance increasing at higher scrambling levels)
- Unusually high performance on abstract symbol questions
- Perfect accuracy on "trick" questions that should confuse reasoning models

### Statistical Significance

- **p < 0.05**: Statistically significant differences between scrambling levels
- **Effect size > 0.8**: Large practical significance
- **Confidence intervals**: Measure reliability of language dependency estimates

## 🔍 Example Findings

### Typical Model Behaviors

**Pattern-Matching Models**:
- High language dependency (>70%)
- Severe performance drops at Level 3+ scrambling
- Strong performance on memorized question types

**Robust Reasoning Models**:
- Low language dependency (<40%)
- Gradual performance degradation across levels
- Consistent performance on novel formulations

### Domain-Specific Insights

**Mathematics**: Often shows highest contamination risk due to standardized problem formats

**Logic**: Most reliable for measuring pure reasoning capabilities

**Reading Comprehension**: Extremely language-dependent by nature

## 📝 Contributing

We welcome contributions to expand the Language Dependency Atlas:

1. **New Domains**: Add specialized reasoning domains
2. **Question Banks**: Contribute verified questions with metadata
3. **Analysis Methods**: Enhance statistical and visualization capabilities
4. **Model Integrations**: Support additional model providers

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the robust ScrambleBench infrastructure
- Inspired by research on language model reasoning limitations
- Statistical methodology based on cognitive psychology benchmarking

## 🐛 Troubleshooting

### Common Issues

**"Model not found"**:
```bash
ollama pull phi3:3.8b  # Pull the specific model
ollama list            # Check available models
```

**"Connection refused"**:
```bash
ollama serve           # Start Ollama server
```

**"Import errors"**:
```bash
pip install numpy scipy matplotlib seaborn plotly pandas requests pyyaml
```

### Performance Issues

- **Slow inference**: Try smaller models (`llama3.2:1b`)
- **Memory issues**: Reduce `max_concurrent_requests`
- **Timeout errors**: Increase `timeout_per_question`

### Getting Help

1. Run the setup validation: `python setup_language_dependency_atlas.py`
2. Check the logs in `language_dependency_atlas.log`
3. Try the quick test mode: `--quick` flag
4. Review configuration in `language_dependency_config_example.yaml`

---

**Ready to map the language dependency landscape of your models? Start with the quick demo and explore the depths of LLM reasoning capabilities!** 🧠✨