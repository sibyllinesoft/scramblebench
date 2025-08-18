Batch Evaluation Tutorial
=========================

This comprehensive tutorial demonstrates how to run batch evaluations with ScrambleBench, enabling efficient testing of multiple models and datasets simultaneously. Batch evaluation is essential for large-scale robustness testing and comparative analysis across model families.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

Batch evaluation allows you to systematically evaluate multiple models against multiple benchmarks and transformation types in a single coordinated run. This approach is crucial for:

**Research Applications:**

* **Model Comparison Studies**: Compare robustness across model families (GPT, Claude, LLaMA)
* **Contamination Analysis**: Detect memorization patterns across different model sizes
* **Transformation Effectiveness**: Assess which transformations best reveal genuine capabilities
* **Publication-Ready Results**: Generate comprehensive datasets for academic papers

**Production Applications:**

* **Model Selection**: Choose the most robust model for your specific use case
* **Performance Monitoring**: Track model degradation over time
* **Cost Optimization**: Balance performance and API costs across providers
* **Quality Assurance**: Validate model updates before deployment

Understanding Batch Evaluation
------------------------------

Core Components
~~~~~~~~~~~~~~~

A batch evaluation consists of four primary components:

1. **Models Configuration**: Defines which models to evaluate and their parameters
2. **Benchmark Datasets**: Specifies the evaluation tasks and data sources
3. **Transformation Pipeline**: Sets up the contamination-resistant modifications
4. **Evaluation Protocol**: Controls sampling, concurrency, and result storage

Evaluation Modes
~~~~~~~~~~~~~~~~

ScrambleBench supports three evaluation modes:

**Quick Mode** (``mode: quick``)
  * Rapid evaluation with reduced sample sizes
  * Single transformation type per benchmark
  * Optimized for development and debugging
  * Typical runtime: 10-30 minutes

**Standard Mode** (``mode: standard``)
  * Balanced evaluation with moderate sample sizes
  * Multiple transformation types
  * Recommended for most use cases
  * Typical runtime: 2-6 hours

**Comprehensive Mode** (``mode: comprehensive``)
  * Exhaustive evaluation with full sample sizes
  * All transformation types and complexity levels
  * Used for research and publication
  * Typical runtime: 8-24 hours

Setting Up Batch Evaluation
----------------------------

Configuration File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch evaluations are configured using YAML files with the following structure:

.. code-block:: yaml

   # Basic experiment metadata
   experiment_name: my_robustness_study
   description: Comprehensive robustness evaluation across model families
   mode: standard  # quick, standard, or comprehensive
   
   # Input data sources
   benchmark_paths:
     - data/benchmarks/logic_reasoning.json
     - data/benchmarks/math_problems.json
   
   # Output configuration
   output_dir: results/batch_evaluation
   
   # Models to evaluate
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
   
   # Transformation settings
   transformations:
     enabled_types: [substitution, phonetic]
     language_complexity: 5
   
   # Evaluation controls
   max_samples: 100
   max_concurrent_requests: 3

Model Configuration
~~~~~~~~~~~~~~~~~~

Configure multiple models with provider-specific settings:

.. code-block:: yaml

   models:
     # OpenAI models via OpenRouter
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0  # requests per second
       
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
     
     # Anthropic models
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
       
     - name: anthropic/claude-3-haiku
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
     
     # Open source models
     - name: meta-llama/llama-2-70b-chat
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 120
       rate_limit: 0.5

**Model Configuration Best Practices:**

* **Rate Limiting**: Start conservatively and increase based on provider limits
* **Timeouts**: Set longer timeouts for larger models and complex tasks
* **Temperature**: Use 0.0 for reproducible evaluation results
* **Max Tokens**: Ensure sufficient token budget for complete responses

Transformation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure transformation types and parameters:

.. code-block:: yaml

   transformations:
     # Specify which transformation types to use
     enabled_types:
       - substitution      # Simple word replacement
       - phonetic         # Sound-based transformations
       - scrambled        # Character scrambling
       - constructed_agglutinative  # Complex artificial languages
     
     # Language settings
     languages:
       - constructed_agglutinative_1
       - constructed_fusional_1
       - constructed_isolating_1
     language_complexity: 6  # 1-10 scale
     
     # Transformation-specific settings
     proper_noun_strategy: random  # random, preserve, or swap
     synonym_rate: 0.4            # For synonym replacement
     preserve_function_words: true
     
     # Reproducibility
     seed: 42
     batch_size: 20

Running Batch Evaluations
-------------------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~

Run batch evaluations using the CLI:

.. code-block:: bash

   # Basic batch evaluation
   scramblebench evaluate run \
     --config configs/evaluation/my_batch_config.yaml
   
   # With custom output directory
   scramblebench evaluate run \
     --config configs/evaluation/my_batch_config.yaml \
     --output-dir results/my_experiment
   
   # Resume interrupted evaluation
   scramblebench evaluate run \
     --config configs/evaluation/my_batch_config.yaml \
     --resume
   
   # Dry run to validate configuration
   scramblebench evaluate run \
     --config configs/evaluation/my_batch_config.yaml \
     --dry-run

Python API
~~~~~~~~~~

Run batch evaluations programmatically:

.. code-block:: python

   import asyncio
   from pathlib import Path
   from scramblebench.evaluation import EvaluationRunner, EvaluationConfig
   
   async def run_batch_evaluation():
       # Load configuration
       config_path = Path("configs/evaluation/my_batch_config.yaml")
       config = EvaluationConfig.from_yaml(config_path)
       
       # Initialize runner
       runner = EvaluationRunner(
           config=config,
           data_dir=Path("data"),
           output_dir=Path("results/batch_evaluation")
       )
       
       # Run evaluation with progress monitoring
       results = await runner.run_evaluation()
       
       # Access results
       print(f"Evaluated {len(results.model_results)} models")
       print(f"Total samples processed: {results.total_samples}")
       print(f"Average accuracy: {results.overall_metrics['accuracy']:.3f}")
       
       return results
   
   # Run the evaluation
   results = asyncio.run(run_batch_evaluation())

Example Configurations
---------------------

Quick Development Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perfect for testing and development:

.. code-block:: yaml

   experiment_name: quick_dev_test
   description: Fast evaluation for development
   mode: quick
   
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
   
   output_dir: results/dev_test
   
   models:
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024
       timeout: 30
       rate_limit: 3.0
   
   transformations:
     enabled_types: [substitution]
     language_complexity: 3
     seed: 42
   
   max_samples: 20
   max_concurrent_requests: 2
   save_interval: 10

Model Comparison Study
~~~~~~~~~~~~~~~~~~~~~

Compare robustness across model families:

.. code-block:: yaml

   experiment_name: model_family_comparison
   description: Robustness comparison across GPT, Claude, and LLaMA families
   mode: standard
   
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/medium/collected_samples.json
   
   output_dir: results/model_comparison
   
   models:
     # GPT family
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
       
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
     
     # Claude family
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
       
     - name: anthropic/claude-3-haiku
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
     
     # LLaMA family
     - name: meta-llama/llama-2-70b-chat
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 120
       rate_limit: 0.5
   
   transformations:
     enabled_types: [substitution, phonetic, scrambled]
     languages:
       - constructed_agglutinative_1
       - constructed_fusional_1
     language_complexity: 5
     seed: 123
   
   max_samples: 150
   max_concurrent_requests: 4
   save_interval: 25

Research-Grade Comprehensive Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full evaluation for research publications:

.. code-block:: yaml

   experiment_name: comprehensive_robustness_analysis
   description: Publication-quality robustness evaluation with statistical analysis
   mode: comprehensive
   
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
     - data/benchmarks/collected/01_logic_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/01_logic_reasoning/hard/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/easy/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/medium/collected_samples.json
     - data/benchmarks/collected/05_reading_comprehension/easy/collected_samples.json
   
   output_dir: results/comprehensive_analysis
   
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 4096
       timeout: 120
       rate_limit: 0.8
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 4096
       timeout: 120
       rate_limit: 0.8
       
     - name: meta-llama/llama-2-70b-chat
       provider: openrouter
       temperature: 0.0
       max_tokens: 4096
       timeout: 150
       rate_limit: 0.3
   
   transformations:
     enabled_types:
       - all  # Enables all available transformation types
     
     languages:
       - constructed_agglutinative_1
       - constructed_agglutinative_2
       - constructed_fusional_1
       - constructed_isolating_1
     language_complexity: 7
     
     proper_noun_strategy: random
     synonym_rate: 0.4
     preserve_function_words: true
     seed: 456
     batch_size: 30
   
   max_samples: 500
   max_concurrent_requests: 3
   save_interval: 50
   
   # Advanced analysis options
   generate_plots: true
   calculate_significance: true
   confidence_level: 0.95
   bootstrap_samples: 1000

Monitoring and Managing Batch Jobs
----------------------------------

Progress Monitoring
~~~~~~~~~~~~~~~~~~

Monitor evaluation progress in real-time:

.. code-block:: python

   from scramblebench.evaluation import EvaluationRunner
   import asyncio
   
   async def monitor_evaluation():
       runner = EvaluationRunner.from_config("my_config.yaml")
       
       # Run with progress callback
       async def progress_callback(completed, total, current_model, current_benchmark):
           progress = completed / total * 100
           print(f"Progress: {progress:.1f}% ({completed}/{total})")
           print(f"Current: {current_model} on {current_benchmark}")
       
       results = await runner.run_evaluation(progress_callback=progress_callback)
       return results

Error Handling and Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~

Handle failures and resume interrupted evaluations:

.. code-block:: python

   from scramblebench.evaluation import EvaluationRunner
   from scramblebench.core.exceptions import APIRateLimitError, ModelTimeoutError
   
   async def robust_evaluation():
       runner = EvaluationRunner.from_config("my_config.yaml")
       
       try:
           results = await runner.run_evaluation()
       except APIRateLimitError as e:
           print(f"Rate limit hit: {e}")
           # Automatically reduce concurrent requests and retry
           runner.config.max_concurrent_requests = max(1, runner.config.max_concurrent_requests // 2)
           results = await runner.resume_evaluation()
       except ModelTimeoutError as e:
           print(f"Model timeout: {e}")
           # Increase timeout and retry
           runner.config.model_timeout += 30
           results = await runner.resume_evaluation()
       
       return results

Resource Management
~~~~~~~~~~~~~~~~~~

Optimize resource usage for long-running evaluations:

.. code-block:: python

   # Memory-efficient batch processing
   runner = EvaluationRunner(
       config=config,
       memory_limit_mb=4096,      # Limit memory usage
       checkpoint_interval=100,    # Save progress every 100 samples
       cleanup_interval=500       # Clean up temp files
   )
   
   # Custom rate limiting strategy
   rate_limits = {
       "openai/gpt-4": 0.5,      # Conservative for expensive models
       "openai/gpt-3.5-turbo": 2.0,  # Faster for cheaper models
       "anthropic/claude-3-haiku": 3.0
   }
   
   for model_config in config.models:
       if model_config.name in rate_limits:
           model_config.rate_limit = rate_limits[model_config.name]

Results Analysis
---------------

Accessing Results
~~~~~~~~~~~~~~~~

Extract and analyze results from completed evaluations:

.. code-block:: python

   from scramblebench.evaluation.results import ResultsManager
   import pandas as pd
   
   # Load results from completed evaluation
   results_manager = ResultsManager("results/my_experiment")
   results = results_manager.load_results()
   
   # Convert to pandas for analysis
   df = results.to_dataframe()
   
   # Analyze performance by model
   model_performance = df.groupby('model_name')['accuracy'].agg([
       'mean', 'std', 'min', 'max', 'count'
   ])
   print(model_performance)
   
   # Analyze robustness by transformation type
   transformation_impact = df.groupby(['model_name', 'transformation_type'])['accuracy'].mean()
   print(transformation_impact.unstack())
   
   # Calculate contamination resistance
   baseline_performance = df[df['transformation_type'] == 'original']['accuracy']
   transformed_performance = df[df['transformation_type'] != 'original']['accuracy']
   contamination_resistance = transformed_performance.mean() / baseline_performance.mean()
   print(f"Contamination resistance: {contamination_resistance:.3f}")

Statistical Analysis
~~~~~~~~~~~~~~~~~~~

Perform statistical significance testing:

.. code-block:: python

   from scramblebench.evaluation.metrics import StatisticalAnalyzer
   from scipy import stats
   
   analyzer = StatisticalAnalyzer(results)
   
   # Compare model performance
   gpt4_scores = results.get_model_scores("openai/gpt-4")
   claude_scores = results.get_model_scores("anthropic/claude-3-sonnet")
   
   # Perform t-test
   t_stat, p_value = stats.ttest_ind(gpt4_scores, claude_scores)
   print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
   
   # Calculate effect size (Cohen's d)
   effect_size = analyzer.cohens_d(gpt4_scores, claude_scores)
   print(f"Effect size (Cohen's d): {effect_size:.3f}")
   
   # Bootstrap confidence intervals
   ci_lower, ci_upper = analyzer.bootstrap_ci(gpt4_scores, confidence=0.95)
   print(f"95% CI for GPT-4: [{ci_lower:.3f}, {ci_upper:.3f}]")

Visualization
~~~~~~~~~~~~

Generate publication-quality plots:

.. code-block:: python

   from scramblebench.evaluation.plotting import PlotGenerator
   import matplotlib.pyplot as plt
   
   plotter = PlotGenerator(results)
   
   # Model comparison plot
   fig, ax = plt.subplots(figsize=(10, 6))
   plotter.plot_model_comparison(ax=ax, metric='accuracy')
   plt.title("Model Performance Comparison")
   plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
   
   # Robustness analysis
   fig, ax = plt.subplots(figsize=(12, 8))
   plotter.plot_robustness_heatmap(ax=ax)
   plt.title("Robustness Across Transformations")
   plt.savefig("robustness_heatmap.png", dpi=300, bbox_inches='tight')
   
   # Performance degradation
   fig, ax = plt.subplots(figsize=(10, 6))
   plotter.plot_degradation_analysis(ax=ax)
   plt.title("Performance Degradation by Transformation")
   plt.savefig("degradation_analysis.png", dpi=300, bbox_inches='tight')

Best Practices and Tips
----------------------

Configuration Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Start Small, Scale Up**
  * Begin with quick mode and small sample sizes
  * Gradually increase complexity and sample counts
  * Test configurations thoroughly before large runs

**Rate Limiting Strategy**
  * Start with conservative rate limits
  * Monitor API usage and adjust gradually
  * Consider cost implications of concurrent requests

**Resource Planning**
  * Estimate runtime based on: models × benchmarks × samples × transformations
  * Plan for 20-30% overhead for retries and processing
  * Monitor disk space for result storage

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Concurrent Processing**
  * Use appropriate concurrency levels (2-5 for most providers)
  * Balance speed vs. rate limit compliance
  * Monitor for timeouts and adjust accordingly

**Memory Management**
  * Enable result streaming for large evaluations
  * Use checkpoints to save progress regularly
  * Clean up intermediate files periodically

**Cost Optimization**
  * Use cheaper models for initial testing
  * Implement sampling strategies for large datasets
  * Cache transformation results for reuse

Troubleshooting Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rate Limit Errors**
  * Reduce ``max_concurrent_requests``
  * Increase ``rate_limit`` delay between requests
  * Implement exponential backoff for retries

**Memory Issues**
  * Enable result streaming with ``stream_results: true``
  * Reduce ``batch_size`` for transformations
  * Increase checkpoint frequency

**Timeout Errors**
  * Increase model timeout values
  * Simplify complex prompts or transformations
  * Consider using faster models for initial testing

**Configuration Errors**
  * Validate YAML syntax with online validators
  * Check file paths are accessible
  * Verify model names with provider documentation

Advanced Features
----------------

Custom Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement custom sampling for specialized evaluations:

.. code-block:: python

   from scramblebench.evaluation import EvaluationConfig
   
   class StratifiedSamplingConfig(EvaluationConfig):
       """Custom config with stratified sampling by difficulty."""
       
       def sample_data(self, dataset, max_samples):
           # Ensure balanced sampling across difficulty levels
           difficulties = ['easy', 'medium', 'hard']
           samples_per_difficulty = max_samples // len(difficulties)
           
           sampled_data = []
           for difficulty in difficulties:
               difficulty_samples = [
                   sample for sample in dataset 
                   if sample.get('difficulty') == difficulty
               ]
               sampled_data.extend(
                   random.sample(difficulty_samples, 
                                min(samples_per_difficulty, len(difficulty_samples)))
               )
           
           return sampled_data

Dynamic Model Selection
~~~~~~~~~~~~~~~~~~~~~~

Adapt model selection based on performance:

.. code-block:: python

   class AdaptiveEvaluationRunner(EvaluationRunner):
       """Evaluation runner that adapts based on intermediate results."""
       
       async def run_evaluation(self):
           # Start with quick evaluation
           quick_results = await self.run_quick_evaluation()
           
           # Select top-performing models for comprehensive evaluation
           top_models = self.select_top_models(quick_results, top_k=3)
           
           # Run comprehensive evaluation on selected models
           self.config.models = top_models
           comprehensive_results = await super().run_evaluation()
           
           return comprehensive_results

Integration with External Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate with MLflow for experiment tracking:

.. code-block:: python

   import mlflow
   from scramblebench.evaluation import EvaluationRunner
   
   class MLflowEvaluationRunner(EvaluationRunner):
       """Evaluation runner with MLflow integration."""
       
       async def run_evaluation(self):
           with mlflow.start_run():
               # Log configuration
               mlflow.log_params(self.config.to_dict())
               
               # Run evaluation
               results = await super().run_evaluation()
               
               # Log results
               mlflow.log_metrics({
                   'overall_accuracy': results.overall_metrics['accuracy'],
                   'total_samples': results.total_samples,
                   'total_cost': results.total_cost
               })
               
               # Log artifacts
               results.save_plots("plots/")
               mlflow.log_artifacts("plots/")
               
               return results

Related Documentation
--------------------

* :doc:`../user_guide/configuration` - Detailed configuration reference
* :doc:`../user_guide/evaluation_pipeline` - Understanding the evaluation process
* :doc:`../api/evaluation` - API reference for evaluation components
* :doc:`translation_benchmarks` - Translation-specific evaluation details
* :doc:`../examples/configuration_examples` - Additional configuration examples

For questions and support, visit the `GitHub Issues <https://github.com/sibyllinesoft/scramblebench/issues>`_ page.