Configuration Examples Tutorial
===============================

This comprehensive tutorial provides practical configuration examples for various ScrambleBench use cases. Learn how to configure evaluations for different scenarios, from quick development testing to production-ready research studies.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

ScrambleBench uses YAML configuration files to define evaluation parameters, making it easy to reproduce experiments and share configurations across teams. This tutorial covers:

**Configuration Categories:**

* **Development Configurations**: Fast iterations and debugging
* **Research Configurations**: Publication-quality evaluations
* **Production Configurations**: Robust deployment testing
* **Specialized Configurations**: Domain-specific evaluation setups

**Key Configuration Areas:**

* **Model Settings**: Provider selection, rate limiting, and parameters
* **Benchmark Data**: Dataset selection and sampling strategies
* **Transformations**: Language generation and modification settings
* **Evaluation Control**: Concurrency, timeouts, and result management
* **Analysis Options**: Statistical testing and visualization settings

Configuration Structure
-----------------------

Basic Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ScrambleBench configuration follows this basic structure:

.. code-block:: yaml

   # Experiment metadata
   experiment_name: my_experiment
   description: Brief description of the evaluation
   mode: standard  # quick, standard, or comprehensive
   
   # Data sources
   benchmark_paths:
     - path/to/benchmark1.json
     - path/to/benchmark2.jsonl
   
   # Output settings
   output_dir: results/my_experiment
   
   # Models to evaluate
   models:
     - name: model_identifier
       provider: openrouter
       # Model-specific parameters
   
   # Transformation settings
   transformations:
     enabled_types: [substitution, phonetic]
     # Transformation parameters
   
   # Evaluation controls
   max_samples: 100
   max_concurrent_requests: 3

Configuration Sections Explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Experiment Metadata**
  * ``experiment_name``: Unique identifier for the evaluation run
  * ``description``: Human-readable description for documentation
  * ``mode``: Evaluation intensity (quick/standard/comprehensive)

**Data Configuration**
  * ``benchmark_paths``: List of benchmark files to evaluate
  * ``output_dir``: Directory for storing results and artifacts

**Model Configuration**
  * ``models``: List of models with provider and parameter settings
  * Each model can have custom rate limits, timeouts, and generation parameters

**Transformation Settings**
  * ``transformations``: Controls how benchmarks are modified for contamination resistance
  * Includes language generation, complexity, and preservation settings

**Evaluation Controls**
  * ``max_samples``: Sample size limits for each benchmark
  * ``max_concurrent_requests``: Parallel processing limits
  * Progress saving and error handling settings

Development Configurations
--------------------------

Quick Development Testing
~~~~~~~~~~~~~~~~~~~~~~~~

For rapid iteration during development:

.. code-block:: yaml

   experiment_name: dev_quick_test
   description: Fast development testing with minimal samples
   mode: quick
   
   # Use small benchmark for speed
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
   
   output_dir: results/dev_test
   
   # Single fast model for development
   models:
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024
       timeout: 30
       rate_limit: 3.0  # Fast for development
   
   # Simple transformation for speed
   transformations:
     enabled_types: [substitution]
     language_complexity: 3
     seed: 42
   
   # Small sample size
   max_samples: 10
   max_concurrent_requests: 2
   save_interval: 5
   
   # Development-specific settings
   debug_mode: true
   verbose_logging: true

Local Development with Dummy Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use dummy models to avoid API costs during development:

.. code-block:: yaml

   experiment_name: local_dev_testing
   description: Local development with dummy models (no API costs)
   mode: quick
   
   benchmark_paths:
     - data/benchmarks/examples/simple_qa.json
   
   output_dir: results/local_dev
   
   # Dummy model configuration
   models:
     - name: dummy-gpt-4
       provider: dummy
       temperature: 0.0
       max_tokens: 2048
       # Dummy models don't use real API calls
       simulate_accuracy: 0.75  # Simulated performance
       simulate_latency: 2.0    # Simulated response time
   
   transformations:
     enabled_types: [substitution]
     language_complexity: 2
     seed: 123
   
   max_samples: 5
   max_concurrent_requests: 10  # No rate limits for dummy models
   
   # Development helpers
   generate_sample_outputs: true
   save_transformation_examples: true

Debugging Configuration
~~~~~~~~~~~~~~~~~~~~~~

Detailed logging and intermediate result saving for debugging:

.. code-block:: yaml

   experiment_name: debug_evaluation
   description: Detailed debugging with full logging and intermediate saves
   mode: quick
   
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
   
   output_dir: results/debug
   
   models:
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024
       timeout: 60
       rate_limit: 1.0
   
   transformations:
     enabled_types: [substitution]
     language_complexity: 4
     seed: 42
   
   max_samples: 20
   max_concurrent_requests: 1  # Sequential for easier debugging
   save_interval: 1  # Save after every sample
   
   # Extensive debugging options
   debug_mode: true
   verbose_logging: true
   save_intermediate_results: true
   save_transformation_details: true
   save_model_responses: true
   save_error_details: true
   
   # Detailed logging configuration
   logging:
     level: DEBUG
     console: true
     file: logs/debug_evaluation.log
     include_timestamps: true
     include_model_calls: true

Research Configurations
----------------------

Model Comparison Study
~~~~~~~~~~~~~~~~~~~~~

Comprehensive comparison across model families:

.. code-block:: yaml

   experiment_name: model_family_robustness_study
   description: Systematic robustness comparison across GPT, Claude, and LLaMA families
   mode: standard
   
   # Multiple benchmark categories
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/medium/collected_samples.json
     - data/benchmarks/collected/05_reading_comprehension/easy/collected_samples.json
   
   output_dir: results/model_family_study
   
   # Comprehensive model selection
   models:
     # OpenAI GPT family
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
     
     # Anthropic Claude family
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
     
     # Meta LLaMA family
     - name: meta-llama/llama-2-70b-chat
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 120
       rate_limit: 0.5
       
     - name: meta-llama/llama-2-13b-chat
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
   
   # Multiple transformation types for robustness testing
   transformations:
     enabled_types: [substitution, phonetic, scrambled]
     languages:
       - constructed_agglutinative_1
       - constructed_fusional_1
       - constructed_isolating_1
     language_complexity: 5
     
     # Proper noun handling
     proper_noun_strategy: random
     
     # Synonym replacement settings
     synonym_rate: 0.4
     preserve_function_words: true
     
     seed: 2024
     batch_size: 25
   
   # Research-appropriate sample sizes
   max_samples: 200
   max_concurrent_requests: 4
   save_interval: 50
   
   # Statistical analysis
   generate_plots: true
   calculate_significance: true
   confidence_level: 0.95
   bootstrap_samples: 1000

Publication-Quality Comprehensive Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full-scale evaluation for research publication:

.. code-block:: yaml

   experiment_name: comprehensive_contamination_analysis_2024
   description: "Publication study: Revealing Training Data Contamination Through Constructed Language Evaluation"
   mode: comprehensive
   
   # Comprehensive benchmark coverage
   benchmark_paths:
     # Logic reasoning across difficulties
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
     - data/benchmarks/collected/01_logic_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/01_logic_reasoning/hard/collected_samples.json
     
     # Mathematical reasoning
     - data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/hard/collected_samples.json
     
     # Puzzles and riddles
     - data/benchmarks/collected/03_puzzles_riddles/easy/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/medium/collected_samples.json
     - data/benchmarks/collected/03_puzzles_riddles/hard/collected_samples.json
     
     # Reading comprehension
     - data/benchmarks/collected/05_reading_comprehension/easy/collected_samples.json
     - data/benchmarks/collected/05_reading_comprehension/medium/collected_samples.json
   
   output_dir: results/comprehensive_contamination_study
   
   # State-of-the-art model selection
   models:
     # Flagship models from major providers
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
       
     - name: google/gemini-pro
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
       rate_limit: 0.5
   
   # Comprehensive transformation analysis
   transformations:
     # All transformation types
     enabled_types:
       - all  # Enables all available transformations
     
     # Multiple constructed languages
     languages:
       - constructed_agglutinative_1
       - constructed_agglutinative_2
       - constructed_fusional_1
       - constructed_fusional_2
       - constructed_isolating_1
       - constructed_synthetic_1
     
     # Multiple complexity levels
     language_complexity: [3, 5, 7]  # Test across complexity spectrum
     
     # Comprehensive transformation settings
     proper_noun_strategy: [random, preserve, swap]  # Test all strategies
     synonym_rate: [0.2, 0.4, 0.6]  # Multiple replacement rates
     preserve_function_words: true
     
     # Reproducibility
     seed: 20241201
     batch_size: 40
   
   # Large sample sizes for statistical power
   max_samples: 1000
   max_concurrent_requests: 3  # Conservative for stability
   save_interval: 100
   
   # Comprehensive analysis
   generate_plots: true
   calculate_significance: true
   confidence_level: 0.99  # High confidence for publication
   bootstrap_samples: 5000
   effect_size_calculation: true
   
   # Publication-ready outputs
   export_formats: [csv, json, latex]
   generate_summary_report: true
   include_methodology_section: true

Ablation Study Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Systematic parameter variation study:

.. code-block:: yaml

   experiment_name: transformation_ablation_study
   description: Systematic ablation study of transformation parameters
   mode: standard
   
   benchmark_paths:
     - data/benchmarks/collected/02_mathematical_reasoning/medium/collected_samples.json
   
   output_dir: results/ablation_study
   
   # Single model for controlled comparison
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
   
   # Systematic parameter variation
   transformations:
     enabled_types: [substitution, phonetic, constructed_agglutinative]
     
     # Complexity ablation (test each level)
     language_complexity: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
     
     # Proper noun strategy ablation
     proper_noun_strategy: [preserve, random, swap]
     
     # Synonym rate ablation
     synonym_rate: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
     
     # Function word preservation ablation
     preserve_function_words: [true, false]
     
     seed: 54321
     batch_size: 20
   
   max_samples: 100
   max_concurrent_requests: 2
   save_interval: 25
   
   # Detailed analysis for ablation
   generate_plots: true
   calculate_significance: true
   export_parameter_analysis: true

Production Configurations
-------------------------

Model Selection for Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate models for production deployment:

.. code-block:: yaml

   experiment_name: production_model_selection
   description: Robustness evaluation for production model selection
   mode: standard
   
   # Production-relevant benchmarks
   benchmark_paths:
     - data/benchmarks/production/customer_queries.json
     - data/benchmarks/production/edge_cases.json
     - data/benchmarks/production/domain_specific.json
   
   output_dir: results/production_selection
   
   # Candidate production models
   models:
     # High-performance options
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
       cost_per_token: 0.00006  # For cost analysis
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 2.0
       cost_per_token: 0.000015
     
     # Cost-effective options
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 45
       rate_limit: 3.0
       cost_per_token: 0.000002
       
     - name: anthropic/claude-3-haiku
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 45
       rate_limit: 3.0
       cost_per_token: 0.00000125
   
   # Production-relevant transformations
   transformations:
     enabled_types: [substitution, phonetic]  # Focus on realistic variations
     language_complexity: 4  # Moderate complexity
     proper_noun_strategy: preserve  # Maintain entity names
     synonym_rate: 0.3
     preserve_function_words: true
     seed: 98765
   
   max_samples: 300
   max_concurrent_requests: 5
   save_interval: 50
   
   # Production-focused analysis
   generate_plots: true
   calculate_cost_effectiveness: true
   performance_threshold: 0.85  # Minimum acceptable accuracy
   latency_threshold: 5.0  # Maximum acceptable latency (seconds)

Continuous Integration Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automated testing configuration for CI/CD:

.. code-block:: yaml

   experiment_name: ci_regression_test
   description: Continuous integration robustness regression testing
   mode: quick
   
   # Core benchmark suite for regression testing
   benchmark_paths:
     - data/benchmarks/ci/core_functionality.json
     - data/benchmarks/ci/regression_cases.json
   
   output_dir: results/ci_test
   
   # Primary production model
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024
       timeout: 45
       rate_limit: 2.0
   
   # Simple transformations for CI speed
   transformations:
     enabled_types: [substitution]
     language_complexity: 3
     seed: 12345  # Fixed seed for reproducible CI
   
   # Small sample size for speed
   max_samples: 50
   max_concurrent_requests: 3
   save_interval: 25
   
   # CI-specific settings
   fail_on_regression: true
   regression_threshold: 0.05  # Fail if accuracy drops >5%
   baseline_results: results/baseline/ci_baseline.json
   generate_ci_report: true
   export_junit_xml: true  # For CI integration

A/B Testing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Compare model versions or configurations:

.. code-block:: yaml

   experiment_name: model_ab_test
   description: A/B testing between model configurations
   mode: standard
   
   benchmark_paths:
     - data/benchmarks/production/user_queries.json
   
   output_dir: results/ab_test
   
   # A/B test configurations
   models:
     # Configuration A: High temperature for creativity
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.7
       max_tokens: 2048
       timeout: 60
       rate_limit: 1.5
       group: config_a
       
     # Configuration B: Low temperature for consistency
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 60
       rate_limit: 1.5
       group: config_b
   
   transformations:
     enabled_types: [substitution, phonetic]
     language_complexity: 5
     seed: 11111
   
   max_samples: 500  # Large sample for statistical power
   max_concurrent_requests: 3
   save_interval: 100
   
   # A/B testing specific analysis
   calculate_significance: true
   confidence_level: 0.95
   minimum_effect_size: 0.02  # Minimum meaningful difference
   power_analysis: true
   stratified_sampling: true  # Ensure balanced groups

Specialized Configurations
-------------------------

Domain-Specific Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration for specific domains (e.g., medical, legal, technical):

.. code-block:: yaml

   experiment_name: medical_domain_evaluation
   description: Robustness evaluation for medical domain applications
   mode: standard
   
   # Medical domain benchmarks
   benchmark_paths:
     - data/benchmarks/medical/clinical_reasoning.json
     - data/benchmarks/medical/drug_interactions.json
     - data/benchmarks/medical/diagnostic_cases.json
   
   output_dir: results/medical_domain
   
   # Models with medical training/fine-tuning
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 4096
       timeout: 120
       rate_limit: 1.0
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 4096
       timeout: 120
       rate_limit: 1.0
   
   # Domain-specific transformation settings
   transformations:
     enabled_types: [substitution, phonetic]
     
     # Preserve medical terminology
     preserve_terms:
       - medical_drugs
       - anatomical_terms
       - medical_procedures
       - units_of_measurement
     
     # Conservative complexity for safety-critical domain
     language_complexity: 4
     proper_noun_strategy: preserve  # Keep patient/doctor names
     synonym_rate: 0.2  # Lower rate to preserve medical accuracy
     preserve_function_words: true
     
     seed: 2468
   
   max_samples: 400
   max_concurrent_requests: 2  # Conservative for accuracy
   save_interval: 100
   
   # Domain-specific analysis
   calculate_safety_metrics: true
   harm_detection: true
   medical_accuracy_validation: true
   generate_domain_report: true

Multilingual Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Cross-language robustness testing:

.. code-block:: yaml

   experiment_name: multilingual_robustness
   description: Cross-language robustness evaluation with translation
   mode: standard
   
   # Multilingual benchmark suite
   benchmark_paths:
     - data/benchmarks/multilingual/english_base.json
     - data/benchmarks/multilingual/spanish_translated.json
     - data/benchmarks/multilingual/french_translated.json
     - data/benchmarks/multilingual/german_translated.json
   
   output_dir: results/multilingual
   
   # Multilingual-capable models
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
       
     - name: google/gemini-pro
       provider: openrouter
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
   
   # Multilingual transformation settings
   transformations:
     enabled_types: [substitution, phonetic, constructed_agglutinative]
     
     # Language-specific settings
     base_languages: [en, es, fr, de]
     constructed_languages:
       - constructed_romance_1  # Romance language family
       - constructed_germanic_1  # Germanic language family
       - constructed_agglutinative_1  # Different language type
     
     language_complexity: 5
     preserve_proper_nouns: true  # Important for cross-language consistency
     
     # Cross-language consistency settings
     maintain_cross_language_alignment: true
     translation_quality_threshold: 0.95
     
     seed: 13579
   
   max_samples: 250
   max_concurrent_requests: 3
   save_interval: 50
   
   # Multilingual-specific analysis
   cross_language_consistency: true
   translation_quality_analysis: true
   language_bias_detection: true
   generate_language_comparison: true

Long Context Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Specialized configuration for long context capabilities:

.. code-block:: yaml

   experiment_name: long_context_robustness
   description: Long context robustness evaluation with document transformations
   mode: standard
   
   # Long context benchmarks
   benchmark_paths:
     - data/benchmarks/longcontext/document_qa.json
     - data/benchmarks/longcontext/narrative_comprehension.json
     - data/benchmarks/longcontext/technical_manuals.json
   
   output_dir: results/long_context
   
   # Long context capable models
   models:
     - name: openai/gpt-4-32k
       provider: openrouter
       temperature: 0.0
       max_tokens: 8192
       timeout: 180  # Longer timeout for long contexts
       rate_limit: 0.5  # Slower for long context processing
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
       max_tokens: 8192
       timeout: 180
       rate_limit: 0.5
   
   # Long context transformation settings
   transformations:
     enabled_types: [document_transformation, substitution]
     
     # Document-specific settings
     document_transformation:
       chunk_size: 2048
       overlap_size: 256
       preserve_structure: true
       maintain_coherence: true
     
     # Language settings for long text
     language_complexity: 4  # Lower complexity for long context
     proper_noun_strategy: preserve
     synonym_rate: 0.3
     
     # Long context specific
     preserve_document_flow: true
     maintain_answer_alignment: true
     
     seed: 24680
   
   max_samples: 100  # Smaller samples due to processing time
   max_concurrent_requests: 2  # Conservative for long context
   save_interval: 20
   
   # Long context analysis
   context_length_analysis: true
   answer_position_bias: true
   coherence_preservation: true
   processing_time_analysis: true

Cost Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Budget-conscious evaluation with cost controls:

.. code-block:: yaml

   experiment_name: cost_optimized_evaluation
   description: Budget-conscious robustness evaluation with cost optimization
   mode: standard
   
   benchmark_paths:
     - data/benchmarks/collected/01_logic_reasoning/easy/collected_samples.json
     - data/benchmarks/collected/02_mathematical_reasoning/easy/collected_samples.json
   
   output_dir: results/cost_optimized
   
   # Cost-effective model selection
   models:
     # High-value models
     - name: openai/gpt-3.5-turbo
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024  # Limit tokens for cost
       timeout: 45
       rate_limit: 3.0
       cost_per_token: 0.000002
       max_cost_per_run: 5.00  # Budget limit
       
     - name: anthropic/claude-3-haiku
       provider: openrouter
       temperature: 0.0
       max_tokens: 1024
       timeout: 45
       rate_limit: 3.0
       cost_per_token: 0.00000125
       max_cost_per_run: 3.00
   
   # Cost-effective transformation settings
   transformations:
     enabled_types: [substitution]  # Fastest transformation
     language_complexity: 3  # Lower complexity for speed
     batch_size: 50  # Larger batches for efficiency
     seed: 97531
   
   # Optimized sampling
   max_samples: 150
   max_concurrent_requests: 5
   save_interval: 30
   
   # Cost optimization settings
   cost_optimization: true
   budget_limit: 20.00  # Total budget limit
   cost_per_sample_limit: 0.10
   auto_stop_on_budget: true
   prioritize_cost_effective_models: true
   
   # Lightweight analysis
   generate_plots: false  # Skip expensive plot generation
   calculate_significance: false
   export_minimal_results: true

Best Practices and Tips
----------------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

**Version Control**
  * Store configurations in version control (Git)
  * Use descriptive file names with dates or version numbers
  * Include configuration metadata and descriptions

**Template System**
  * Create base templates for common scenarios
  * Use YAML anchors and references for reusable sections
  * Maintain a library of proven configurations

**Documentation**
  * Document configuration choices and rationale
  * Include expected runtime and cost estimates
  * Note any special requirements or dependencies

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Model Selection**
  * Start with one model per provider for initial testing
  * Choose models appropriate for your use case and budget
  * Consider model context limits for long text evaluation

**Rate Limiting**
  * Start conservative and increase gradually
  * Monitor API usage and adjust based on provider limits
  * Consider time-of-day variations in API performance

**Sample Sizes**
  * Use power analysis to determine minimum sample sizes
  * Start small for initial testing, scale up for final runs
  * Balance statistical power with time and cost constraints

**Transformation Settings**
  * Begin with simple transformations (substitution)
  * Increase complexity gradually to find optimal resistance level
  * Consider domain-specific preservation requirements

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

**Concurrency Management**
  * Monitor API rate limits and error rates
  * Adjust concurrent requests based on provider performance
  * Use exponential backoff for error handling

**Resource Planning**
  * Estimate total runtime: (samples × models × benchmarks) / throughput
  * Plan disk space for results and intermediate files
  * Consider memory requirements for large evaluations

**Cost Management**
  * Set budget limits and monitoring
  * Use cheaper models for initial testing
  * Implement cost-per-sample tracking

Common Configuration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**YAML Anchors for Reusability**

.. code-block:: yaml

   # Define reusable model configurations
   model_configs:
     gpt4_config: &gpt4_config
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
     
     claude_config: &claude_config
       temperature: 0.0
       max_tokens: 2048
       timeout: 90
       rate_limit: 1.0
   
   models:
     - name: openai/gpt-4
       provider: openrouter
       <<: *gpt4_config
       
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       <<: *claude_config

**Environment-Specific Overrides**

.. code-block:: yaml

   # Base configuration
   base: &base
     max_samples: 100
     max_concurrent_requests: 3
     
   # Development overrides
   development:
     <<: *base
     max_samples: 10
     max_concurrent_requests: 1
     debug_mode: true
   
   # Production overrides
   production:
     <<: *base
     max_samples: 1000
     max_concurrent_requests: 5
     save_interval: 100

Related Documentation
--------------------

* :doc:`../user_guide/configuration` - Complete configuration reference
* :doc:`../user_guide/evaluation_pipeline` - Understanding evaluation flow
* :doc:`batch_evaluation` - Batch evaluation tutorial
* :doc:`../api/evaluation` - API reference for configuration classes
* :doc:`../examples/basic_usage` - Simple usage examples

For additional configuration examples and templates, visit the `examples directory <https://github.com/sibyllinesoft/scramblebench/tree/main/configs/examples>`_ in the ScrambleBench repository.