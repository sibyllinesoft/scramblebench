# ScrambleBench Experiment Tracking System

A comprehensive experiment management system designed for academic research in language dependency analysis. Provides robust tracking, monitoring, and reproducibility features for large-scale AI model evaluation experiments.

## Overview

The ScrambleBench Experiment Tracking System enables researchers to:

- **Manage Large-Scale Experiments**: Handle thousands of model evaluations across multiple benchmarks and scrambling levels
- **Ensure Reproducibility**: Complete environment capture and validation for academic replication
- **Statistical Analysis**: Advanced statistical testing including A/B tests, significance analysis, and effect size calculations
- **Academic Publication**: Export data and generate publication-ready tables, figures, and documentation
- **Real-Time Monitoring**: Track progress, resource utilization, and performance with alerting
- **Queue Management**: Intelligent experiment scheduling with priority handling and resource constraints

## Key Features

### 🔬 Academic Research Support
- **Reproducibility Validation**: Complete environment snapshots with validation
- **Replication Packages**: Self-contained packages for experiment replication
- **Statistical Analysis**: Comprehensive statistical testing with multiple comparison correction
- **Publication Export**: LaTeX tables, CSV data, and publication-ready documentation
- **Citation Support**: Automatic generation of citation information and metadata

### ⚡ Scalable Experiment Management
- **Queue System**: Priority-based scheduling with dependency resolution
- **Resource Management**: CPU, memory, API rate limit, and cost tracking
- **Progress Monitoring**: Real-time progress tracking with ETA calculations
- **Error Handling**: Automatic retry logic and failure recovery
- **Batch Processing**: Efficient parallel execution of large experiment sets

### 📊 Real-Time Monitoring & Analytics  
- **Dashboard Interface**: Web-based monitoring dashboard
- **Performance Tracking**: Response times, accuracy, cost, and resource utilization
- **Alert System**: Configurable alerts for performance issues and failures
- **Statistical Insights**: Language dependency coefficients and threshold analysis
- **A/B Testing**: Built-in framework for model comparison and significance testing

### 🔄 Integration & Extensibility
- **ScrambleBench Integration**: Seamless integration with existing evaluation pipeline
- **Database Support**: PostgreSQL backend with optimized queries
- **API Compatibility**: Support for both Ollama (local) and OpenRouter (API) models
- **CLI Interface**: Comprehensive command-line tools for experiment management
- **Python API**: Full programmatic access to all functionality

## Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up Database**
```bash
# Create PostgreSQL database
createdb scramblebench_experiments

# Set database URL
export DATABASE_URL="postgresql://user:password@localhost:5432/scramblebench_experiments"
```

3. **Initialize Database Schema**
```python
from scramblebench.experiment_tracking import DatabaseManager
import asyncio

async def init_db():
    db = DatabaseManager(os.environ['DATABASE_URL'])
    await db.initialize_database()

asyncio.run(init_db())
```

### Basic Usage

#### 1. Create an Experiment

```python
from scramblebench.experiment_tracking import ExperimentTracker
from pathlib import Path

tracker = ExperimentTracker(database_url=DATABASE_URL)

experiment_id = await tracker.create_experiment(
    name="Language Dependency Analysis",
    config=Path("configs/experiments/my_experiment.yaml"),
    description="Analysis of language dependency across models",
    research_question="How does scrambling affect model performance?",
    researcher_name="Dr. Smith"
)
```

#### 2. Queue and Run Experiments

```python
# Queue experiment
await tracker.queue_experiment(experiment_id, priority=1)

# Start experiment runner
await tracker.run_experiments(continuous=True)
```

#### 3. Monitor Progress

```python
# Get status
status = await tracker.get_experiment_status(experiment_id)
print(f"Progress: {status['progress']*100:.1f}%")

# Start monitoring dashboard
from scramblebench.experiment_tracking import ExperimentMonitor

monitor = ExperimentMonitor(db_manager)
await monitor.start_monitoring()  # Available at http://localhost:8080
```

#### 4. Generate Analysis

```python
from scramblebench.experiment_tracking import StatisticalAnalyzer, AcademicExporter

# Statistical analysis
analyzer = StatisticalAnalyzer(db_manager)
await analyzer.run_significance_tests(experiment_id, results)

# Export for publication
exporter = AcademicExporter(db_manager)
await exporter.create_publication_package(
    experiment_id=experiment_id,
    output_dir=Path("publication_package"),
    title="Language Dependency in LLMs",
    authors=["Dr. Smith", "Dr. Jones"]
)
```

### Command Line Interface

The system provides a comprehensive CLI for experiment management:

```bash
# Create experiment
scramblebench-experiment create "My Experiment" config.yaml \
  --description "Experiment description" \
  --research-question "How does X affect Y?" \
  --researcher "Dr. Smith"

# Queue experiment  
scramblebench-experiment queue <experiment_id> --priority 1

# Run experiment runner
scramblebench-experiment run --continuous --max-concurrent 3

# Monitor status
scramblebench-experiment status <experiment_id>

# List experiments
scramblebench-experiment list --status running

# Generate analysis
scramblebench-experiment analyze <experiment_id> --include-plots

# Create replication package
scramblebench-experiment replicate <experiment_id> --output-dir replication/

# Start monitoring dashboard
scramblebench-experiment monitor
```

## Configuration

### Experiment Configuration

Create comprehensive experiment configurations in YAML:

```yaml
experiment_name: "language_dependency_study"
description: "Comprehensive language dependency analysis"

# Research metadata
research_metadata:
  research_question: "How does language dependency vary across models?"
  hypothesis: "Larger models show increased dependency"
  researcher_name: "Dr. Smith"
  institution: "University Research Lab"

# Models to evaluate
models:
  - name: "llama3:8b"
    provider: "ollama"
    temperature: 0.0
    max_tokens: 200
    
  - name: "gpt-3.5-turbo"
    provider: "openrouter"
    temperature: 0.0
    max_tokens: 200

# Scrambling configuration
scrambling:
  levels: [0, 17, 33, 50, 67, 83, 100]
  methods: ["character_substitution"]
  preserve_numbers: true
  random_seed: 42

# Statistical analysis
statistical_analysis:
  calculate_language_dependency: true
  threshold_analysis: true
  confidence_level: 0.95
  correction_method: "bonferroni"

# Experiment tracking
tracking:
  enable_progress_monitoring: true
  enable_resource_monitoring: true
  auto_retry_failures: true

# Reproducibility
reproducibility:
  capture_environment_snapshot: true
  generate_replication_package: true
  validation_tolerance: "moderate"

# Export configuration  
export:
  formats: ["csv", "json", "excel"]
  generate_latex_tables: true
  create_publication_package: true
```

### Database Configuration

The system uses PostgreSQL with the following key tables:

- **experiments**: Main experiment metadata and configuration
- **models**: Model specifications and capabilities
- **questions**: Benchmark questions and their properties
- **scrambled_questions**: Scrambled versions with transformation details
- **responses**: Individual model responses and evaluation results
- **performance_metrics**: Aggregated performance statistics
- **statistical_analyses**: Results from statistical tests
- **threshold_analyses**: Language dependency threshold detection

## Academic Research Features

### Reproducibility

The system ensures complete reproducibility through:

- **Environment Capture**: Python version, packages, system info, Git state
- **Configuration Management**: Complete experiment parameter tracking
- **Data Integrity**: Checksums and validation for all data files
- **Replication Packages**: Self-contained packages with setup instructions

```python
# Validate reproducibility
validator = ReproducibilityValidator()
validation = await validator.validate_reproducibility(
    original_snapshot, tolerance="moderate"
)

if validation['is_reproducible']:
    print("✓ Experiment is reproducible")
else:
    print(f"✗ Issues: {validation['issues']}")
```

### Statistical Analysis

Comprehensive statistical analysis including:

- **Significance Testing**: t-tests, Mann-Whitney U, ANOVA, correlation analysis
- **Effect Size Calculations**: Cohen's d, Hedges' g, eta-squared
- **Multiple Comparison Correction**: Bonferroni, FDR, Holm methods
- **Power Analysis**: Statistical power calculations and sample size recommendations
- **Language Dependency Coefficients**: Custom metrics for text scrambling analysis

```python
# Run comprehensive statistical analysis
analyzer = StatisticalAnalyzer(db_manager)

# Language dependency analysis
dependency_analysis = await analyzer.analyze_thresholds(
    experiment_id, results
)

# Significance testing
significance_tests = await analyzer.run_significance_tests(
    experiment_id, results, 
    comparisons=[("model_a", "model_b"), ("model_b", "model_c")]
)

# A/B testing framework
ab_result = await analyzer.run_ab_test(
    experiment_id, "control_model", "treatment_model"
)
```

### Publication Support

Generate publication-ready materials:

- **LaTeX Tables**: Formatted tables for academic papers
- **Data Export**: CSV, JSON, Excel formats for statistical analysis
- **Methodology Documentation**: Auto-generated methodology sections
- **Results Summaries**: Structured result descriptions
- **Citation Information**: Complete bibliographic metadata

```python
# Generate publication package
exporter = AcademicExporter(db_manager)

publication_data = await exporter.create_publication_package(
    experiment_id=experiment_id,
    title="Language Dependency in Large Language Models",
    authors=["Dr. Smith", "Dr. Jones", "Dr. Brown"],
    abstract="This study investigates...",
    output_dir=Path("publication"),
    include_figures=True
)

# Generate LaTeX tables
tables = await exporter.generate_latex_tables(
    experiment_id, 
    output_dir=Path("tables"),
    table_types=['summary', 'model_comparison', 'significance_tests']
)
```

## Monitoring & Alerting

### Real-Time Dashboard

Access the web dashboard at `http://localhost:8080` when monitoring is enabled:

- **Experiment Status**: Current progress, ETA, and resource usage
- **System Metrics**: CPU, memory, disk usage, and network activity  
- **Performance Graphs**: Response times, accuracy trends, cost tracking
- **Alert Management**: View and acknowledge system alerts
- **Queue Status**: See upcoming experiments and resource allocation

### Performance Monitoring

Track comprehensive performance metrics:

```python
# Get current experiment metrics
status = await tracker.get_experiment_status(experiment_id)

print(f"Progress: {status['progress']*100:.1f}%")
print(f"ETA: {status['eta']}")
print(f"API Calls: {status['api_calls']}")
print(f"Cost: ${status['total_cost']:.4f}")
print(f"Success Rate: {status['success_rate']*100:.1f}%")
```

### Alerting System

Configure alerts for various conditions:

```yaml
monitoring:
  resource_alert_thresholds:
    cpu_percent: 85
    memory_percent: 90
    api_calls_per_minute: 200
    cost_per_hour: 50.0
    error_rate: 0.05
  
  email_alerts: true
  slack_alerts: false
```

## Integration with ScrambleBench

The experiment tracking system seamlessly integrates with existing ScrambleBench components:

### Evaluation Pipeline Integration

```python
from scramblebench.evaluation import EvaluationRunner, EvaluationConfig
from scramblebench.experiment_tracking import ExperimentTracker

# Create experiment with existing config
config = EvaluationConfig.load_from_file("config.yaml")
experiment_id = await tracker.create_experiment(
    name="My Experiment",
    config=config,  # Direct integration
    description="...",
    researcher_name="Dr. Smith"
)

# The tracker will automatically use ScrambleBench's EvaluationRunner
await tracker.run_experiments()
```

### Model Provider Support

Full support for both local and API model providers:

```yaml
models:
  # Local models via Ollama
  - name: "llama3:8b"
    provider: "ollama"
    
  # API models via OpenRouter
  - name: "gpt-3.5-turbo" 
    provider: "openrouter"
    
  # Automatic cost tracking and rate limiting
```

### Benchmark Integration

Seamless integration with ScrambleBench's benchmark system:

```yaml
benchmarks:
  mathematical_reasoning:
    paths:
      - "data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json"
      - "data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json"
    sample_size: 100
```

## Advanced Features

### Queue Management

Intelligent experiment scheduling with:

- **Priority Handling**: Higher priority experiments run first
- **Resource Constraints**: Respect API limits, memory, and cost budgets
- **Dependency Resolution**: Handle experiment dependencies automatically
- **Retry Logic**: Automatic retry with exponential backoff
- **Load Balancing**: Optimal resource utilization across concurrent experiments

```python
# Advanced queue configuration
await tracker.queue_experiment(
    experiment_id=experiment_id,
    priority=5,  # Higher priority
    depends_on=["prerequisite_experiment_id"],
    resource_requirements={
        'api_calls_per_hour': 1000,
        'memory_gb': 4.0,
        'cost_limit': 50.0
    }
)

# Monitor queue status
queue_metrics = await tracker.queue.get_queue_metrics()
print(f"Queue throughput: {queue_metrics.throughput_per_hour:.1f}/hour")
```

### A/B Testing Framework

Built-in A/B testing for model comparisons:

```python
from scramblebench.experiment_tracking import ABTestFramework

ab_framework = ABTestFramework(statistical_analyzer)

# Create A/B test
test_id = await ab_framework.create_ab_test(
    experiment_id=experiment_id,
    test_name="llama_vs_gemma",
    control_group="llama3:8b",
    treatment_group="gemma3:7b",
    primary_metric="accuracy",
    minimum_sample_size=200
)

# Get recommendations
recommendations = await ab_framework.get_ab_test_recommendations(test_id)
print(f"Recommendation: {recommendations['recommendation']}")
```

### Language Dependency Analysis

Specialized analysis for text scrambling research:

```python
# Calculate language dependency coefficients
dependency_coeff = await analyzer.calculate_language_dependency_coefficient(
    experiment_id, model_id, scrambling_method="character"
)

# Find performance thresholds
threshold_analysis = await analyzer.analyze_thresholds(
    experiment_id, results, threshold_definition="50% of baseline"
)

# Generate dependency curves
performance_curves = await analyzer.generate_performance_curves(
    experiment_id, scrambling_levels=[0, 25, 50, 75, 100]
)
```

## Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check database URL
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Initialize schema if needed
python -c "
import asyncio
from scramblebench.experiment_tracking import DatabaseManager
async def init(): 
    db = DatabaseManager('$DATABASE_URL')
    await db.initialize_database()
asyncio.run(init())
"
```

**Memory Issues**
```yaml
# Adjust configuration for lower memory usage
optimization:
  batch_size: 25  # Reduce from 50
  clear_cache_threshold: 0.5  # Clear cache more aggressively
  enable_response_caching: false  # Disable caching if needed
```

**API Rate Limiting**
```yaml
execution:
  max_concurrent_requests: 3  # Reduce concurrent requests
  timeout_per_question: 60    # Increase timeout
  retry_attempts: 5           # Increase retries

tracking:
  resource_alert_thresholds:
    api_calls_per_minute: 100  # Lower threshold
```

### Performance Optimization

For large-scale experiments:

1. **Database Optimization**
   - Use connection pooling
   - Create appropriate indexes
   - Consider read replicas for monitoring

2. **Memory Management**
   - Enable result caching for repeated queries
   - Use batch processing for large datasets
   - Monitor memory usage and adjust batch sizes

3. **API Efficiency**
   - Implement intelligent retry strategies
   - Use concurrent requests within limits
   - Cache model responses when possible

## Contributing

We welcome contributions to the experiment tracking system! Areas for improvement include:

- **New Statistical Methods**: Additional analysis techniques
- **Visualization Features**: Interactive charts and graphs  
- **Integration Extensions**: Support for additional model providers
- **Performance Optimizations**: Scalability improvements
- **Documentation**: Examples and tutorials

## License

This project is licensed under the same terms as ScrambleBench. See the main project LICENSE file for details.

## Citation

If you use the ScrambleBench Experiment Tracking System in your research, please cite:

```bibtex
@software{scramblebench_experiment_tracking,
  title={ScrambleBench Experiment Tracking System},
  author={ScrambleBench Research Team},
  year={2024},
  url={https://github.com/your-repo/scramblebench},
  note={Comprehensive experiment management for language dependency research}
}
```

## Support

For questions, issues, or contributions:

- **Documentation**: See the main ScrambleBench documentation
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join the research community discussions
- **Academic Collaboration**: Contact the research team for academic partnerships

---

The ScrambleBench Experiment Tracking System enables rigorous, reproducible research in language dependency analysis with the scale and statistical rigor required for top-tier academic publication.