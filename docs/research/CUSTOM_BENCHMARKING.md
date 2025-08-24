# Custom Ollama Benchmarking Guide

This guide provides comprehensive instructions for running custom benchmarks on specific Ollama models using ScrambleBench's evaluation framework.

## 🎯 Target Models

The custom benchmark script is specifically designed to test these models:

- **gemma3:4b** - Google's efficient 4B parameter model
- **gpt-oss:20b** - Large open source GPT model (20B parameters)

## 🚀 Quick Start

### Prerequisites

1. **Ollama Server Running**
   ```bash
   ollama serve
   ```

2. **Models Available**
   ```bash
   ollama pull gemma3:4b
   ollama pull gpt-oss:20b
   ```

3. **ScrambleBench Environment**
   ```bash
   source venv/bin/activate  # or your environment activation
   ```

### Simple Usage

```bash
# Run benchmark on gemma3:4b
./run_benchmarks.sh gemma3

# Run benchmark on gpt-oss:20b  
./run_benchmarks.sh gpt-oss

# Run both models with 20 questions each
./run_benchmarks.sh both 20

# List available models
./run_benchmarks.sh list
```

### Advanced Usage

```bash
# Direct Python script with full control
python run_custom_ollama_benchmarks.py --model gemma3:4b --samples 25

# Custom configuration
python run_custom_ollama_benchmarks.py \
  --model gpt-oss:20b \
  --samples 30 \
  --temperature 0.2 \
  --max-tokens 600 \
  --timeout 180 \
  --verbose
```

## 📊 Benchmark Content

The benchmark suite includes **15 comprehensive questions** covering:

### Logic & Reasoning (4 questions)
- **Easy**: Basic syllogistic reasoning
- **Medium**: Conditional logic (modus ponens)
- **Hard**: Complex multi-constraint logic puzzles

### Mathematical Reasoning (3 questions)
- **Easy**: Basic arithmetic with word problems
- **Medium**: Proportional reasoning and ratios
- **Hard**: Multi-step algebra with rates and percentages

### Pattern Recognition (2 questions)
- **Easy**: Simple numeric sequences
- **Medium**: Complex patterns (perfect squares)

### Common Sense Reasoning (2 questions)
- **Easy**: Practical decision making
- **Medium**: Causal reasoning and explanations

### Language Understanding (1 question)
- **Medium**: Reading comprehension and information extraction

### Creative Reasoning (1 question)
- **Medium**: Analogical reasoning

### Problem Solving (1 question)
- **Hard**: Multi-step water jug problem

### Meta-Cognition (1 question)
- **Medium**: Conceptual understanding evaluation

## 📈 Scoring System

The benchmark uses sophisticated scoring methods based on question type:

- **Keyword Matching**: Basic content evaluation
- **Answer Accuracy**: Specific correct answers required
- **Multi-step Accuracy**: Complex problems requiring multiple correct elements
- **Practical Reasoning**: Real-world applicability assessment
- **Causal Understanding**: Quality of explanatory reasoning
- **Solution Completeness**: Comprehensive problem-solving evaluation

### Scoring Ranges
- **0.8-1.0**: Excellent performance
- **0.6-0.8**: Good performance  
- **0.4-0.6**: Fair performance
- **0.2-0.4**: Poor performance
- **0.0-0.2**: Very poor performance

## 📁 Output Structure

Results are saved in model-specific directories:

```
data/results/custom_benchmarks/
├── gemma3_4b/
│   ├── detailed_results_gemma3_4b_20250823_143052.json
│   └── ollama_integration_demo_results_20250823_143105.json
└── gpt-oss_20b/
    ├── detailed_results_gpt-oss_20b_20250823_150234.json
    └── ollama_integration_demo_results_20250823_150247.json
```

### Report Contents

Each benchmark run generates:

1. **Detailed Results JSON**
   - Individual question responses
   - Scoring breakdowns
   - Timing information
   - Model metadata

2. **Summary Report JSON**
   - Overall performance statistics
   - Category-wise analysis
   - Error analysis
   - Top/bottom performing questions

## 🔧 Configuration Options

### Model Parameters
- `--temperature`: Response randomness (0.0-2.0, default: 0.1)
- `--max-tokens`: Maximum response length (default: 500)
- `--timeout`: Request timeout in seconds (default: 120)

### Benchmark Parameters  
- `--samples`: Number of questions to evaluate (default: all 15)
- `--verbose`: Enable detailed logging
- `--list-models`: Show available models

## 📊 Performance Analysis

### Key Metrics Tracked
- **Overall Score**: Average across all questions
- **Pass Rate**: Percentage scoring ≥ 0.6
- **Category Performance**: Breakdown by question type
- **Response Time**: Average time per question
- **Error Rate**: Failed generations

### Comparative Analysis
When running both models, compare:
- Overall accuracy differences
- Category-specific strengths/weaknesses  
- Response time efficiency
- Error patterns and reliability

## 🛠️ Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Pull the model
   ollama pull gemma3:4b
   ollama pull gpt-oss:20b
   ```

2. **Ollama Server Not Running**
   ```bash
   # Start Ollama
   ollama serve
   ```

3. **Import Errors**
   ```bash
   # Activate ScrambleBench environment
   source venv/bin/activate
   
   # Verify installation
   python -c "import scramblebench; print('✅ ScrambleBench available')"
   ```

4. **Slow Response Times**
   - Reduce `--samples` for faster testing
   - Increase `--timeout` for large models
   - Monitor system resources (RAM/CPU usage)

### Performance Optimization

For faster benchmarking:
```bash
# Fewer questions for quick testing
python run_custom_ollama_benchmarks.py --model gemma3:4b --samples 5

# Reduced token limits for faster responses  
python run_custom_ollama_benchmarks.py --model gemma3:4b --max-tokens 200
```

For comprehensive evaluation:
```bash
# Full test suite with generous timeouts
python run_custom_ollama_benchmarks.py --model gpt-oss:20b --timeout 300
```

## 📊 Expected Results

### Performance Baselines

Based on model capabilities, expect:

**gemma3:4b**
- Overall Score: 0.4-0.7 range
- Strengths: Basic logic, simple math
- Challenges: Complex reasoning, multi-step problems
- Response Time: 2-10 seconds per question

**gpt-oss:20b** 
- Overall Score: 0.6-0.8 range  
- Strengths: Complex reasoning, language understanding
- Challenges: May be slower, resource intensive
- Response Time: 5-30 seconds per question

### Category Expectations

| Category | gemma3:4b | gpt-oss:20b |
|----------|-----------|-------------|
| Logic Reasoning | 0.5-0.7 | 0.7-0.9 |
| Math Reasoning | 0.4-0.6 | 0.6-0.8 |
| Pattern Recognition | 0.6-0.8 | 0.8-0.9 |
| Common Sense | 0.5-0.7 | 0.7-0.9 |

## 🎯 Next Steps

After running benchmarks:

1. **Analyze Results**: Compare performance across categories
2. **Identify Patterns**: Look for consistent strengths/weaknesses
3. **Model Selection**: Choose appropriate model for specific use cases
4. **Fine-tuning**: Consider if specific areas need improvement

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review ScrambleBench documentation
3. Examine detailed log output with `--verbose`
4. Verify Ollama model compatibility

---

**Happy Benchmarking!** 🚀