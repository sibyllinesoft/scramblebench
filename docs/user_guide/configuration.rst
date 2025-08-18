Configuration Guide
===================

ScrambleBench provides flexible configuration through YAML files, environment variables, and command-line options. This guide covers all configuration options and best practices.

Configuration Hierarchy
------------------------

Configuration is loaded in the following order (later sources override earlier ones):

1. **Default values** - Built-in defaults
2. **Configuration files** - YAML configuration files
3. **Environment variables** - OS environment variables
4. **Command-line options** - CLI flags and arguments

Configuration Files
-------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

Create a ``config.yaml`` file:

.. code-block:: yaml

   # ScrambleBench Configuration
   
   # Benchmark settings
   benchmark:
     random_seed: 42
     evaluation_mode: "exact_match"  # exact_match, semantic_similarity, custom
     evaluation_threshold: 0.8
     preserve_numbers: true
     preserve_proper_nouns: true
     
   # Model settings
   model:
     default_provider: "openrouter"
     default_model: "openai/gpt-3.5-turbo"
     timeout: 30
     rate_limit: 10.0  # requests per second
     max_retries: 3
     retry_delay: 1.0
     
   # Data settings
   data:
     benchmarks_dir: "data/benchmarks"
     languages_dir: "data/languages"
     results_dir: "data/results"
     cache_dir: "data/cache"
     max_cache_size: 1000
     
   # Logging settings
   logging:
     level: "INFO"  # DEBUG, INFO, WARNING, ERROR
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     file: "logs/scramblebench.log"

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For complex evaluations:

.. code-block:: yaml

   # Advanced ScrambleBench Configuration
   
   # Translation benchmark settings
   translation:
     language_generator:
       default_complexity: 5
       vocab_size: 1000
       phonetic_similarity_threshold: 0.7
       preserve_structure: true
       
     translator:
       batch_size: 100
       parallel_workers: 4
       cache_translations: true
       
   # Long context benchmark settings
   longcontext:
     chunk_long_documents: true
     chunk_size: 5000
     chunk_overlap: 500
     preserve_entities: true
     max_document_length: 50000
     
     transformer:
       min_sentence_length: 10
       max_transformations_per_doc: 10
       preserve_document_structure: true
       
   # Evaluation pipeline settings
   evaluation:
     batch_size: 20
     max_concurrent_requests: 5
     result_caching: true
     intermediate_saves: true
     
     metrics:
       calculate_robustness: true
       calculate_significance: true
       significance_threshold: 0.05
       
     visualization:
       generate_plots: true
       plot_format: "png"  # png, svg, pdf
       plot_dpi: 300
       include_interactive: true
       
   # Performance settings
   performance:
     memory_limit: "8GB"
     temp_directory: "/tmp/scramblebench"
     cleanup_temp_files: true
     
   # Security settings
   security:
     api_key_file: null  # Path to file containing API keys
     mask_api_keys_in_logs: true
     validate_inputs: true

Environment-Specific Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Development configuration (``config/dev.yaml``):

.. code-block:: yaml

   # Development Configuration
   
   benchmark:
     evaluation_mode: "exact_match"
     
   model:
     timeout: 10  # Faster timeouts for testing
     rate_limit: 20.0  # Higher rate limit for dev
     
   data:
     benchmarks_dir: "test_data/benchmarks"
     results_dir: "test_data/results"
     
   logging:
     level: "DEBUG"
     
   evaluation:
     batch_size: 5  # Small batches for testing
     max_concurrent_requests: 2

Production configuration (``config/prod.yaml``):

.. code-block:: yaml

   # Production Configuration
   
   benchmark:
     evaluation_mode: "semantic_similarity"
     
   model:
     timeout: 60  # Longer timeouts for reliability
     rate_limit: 5.0  # Conservative rate limiting
     max_retries: 5
     
   data:
     benchmarks_dir: "/data/benchmarks"
     results_dir: "/data/results"
     max_cache_size: 10000
     
   logging:
     level: "WARNING"
     file: "/var/log/scramblebench/scramblebench.log"
     
   evaluation:
     batch_size: 50
     max_concurrent_requests: 10
     result_caching: true
     
   performance:
     memory_limit: "32GB"

Environment Variables
---------------------

Core Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # API Keys
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   export ANTHROPIC_API_KEY="your_anthropic_api_key"  # If using direct Anthropic
   export OPENAI_API_KEY="your_openai_api_key"        # If using direct OpenAI
   
   # Core settings
   export SCRAMBLEBENCH_CONFIG_FILE="/path/to/config.yaml"
   export SCRAMBLEBENCH_DATA_DIR="/path/to/data"
   export SCRAMBLEBENCH_LOG_LEVEL="INFO"
   
   # Model settings
   export SCRAMBLEBENCH_DEFAULT_MODEL="openai/gpt-4"
   export SCRAMBLEBENCH_DEFAULT_TIMEOUT="30"
   export SCRAMBLEBENCH_RATE_LIMIT="10.0"
   
   # Evaluation settings
   export SCRAMBLEBENCH_BATCH_SIZE="20"
   export SCRAMBLEBENCH_MAX_SAMPLES="1000"

Advanced Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Performance tuning
   export SCRAMBLEBENCH_MEMORY_LIMIT="16GB"
   export SCRAMBLEBENCH_PARALLEL_WORKERS="8"
   export SCRAMBLEBENCH_CACHE_SIZE="5000"
   
   # Development settings
   export SCRAMBLEBENCH_DEBUG="true"
   export SCRAMBLEBENCH_VERBOSE="true"
   export SCRAMBLEBENCH_DISABLE_CACHE="false"
   
   # Security settings
   export SCRAMBLEBENCH_API_KEY_FILE="/secure/path/to/keys.yaml"
   export SCRAMBLEBENCH_MASK_KEYS="true"

Configuration Loading
---------------------

Using Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Specify config file via CLI
   scramblebench --config config.yaml language generate test_lang
   
   # Use environment variable
   export SCRAMBLEBENCH_CONFIG_FILE="config.yaml"
   scramblebench language generate test_lang

.. code-block:: python

   # Load in Python code
   from scramblebench.utils.config import Config
   
   # Load from file
   config = Config("config.yaml")
   
   # Load from dictionary
   config = Config({
       "benchmark": {"random_seed": 42},
       "model": {"timeout": 30}
   })
   
   # Load with environment variable overrides
   config = Config("config.yaml", use_env=True)

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

ScrambleBench validates configuration on load:

.. code-block:: python

   from scramblebench.utils.config import Config, ConfigError
   
   try:
       config = Config("config.yaml")
       if not config.validate():
           print("Configuration validation failed")
   except ConfigError as e:
       print(f"Configuration error: {e}")

Model Configuration
-------------------

OpenRouter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     provider: "openrouter"
     api_key: "${OPENROUTER_API_KEY}"  # Environment variable substitution
     base_url: "https://openrouter.ai/api/v1"
     
     # Default model settings
     default_model: "openai/gpt-3.5-turbo"
     temperature: 0.0
     max_tokens: 1000
     top_p: 1.0
     frequency_penalty: 0.0
     presence_penalty: 0.0
     
     # Rate limiting and timeouts
     rate_limit: 10.0  # requests per second
     timeout: 30
     connect_timeout: 10
     read_timeout: 60
     
     # Retry configuration
     max_retries: 3
     retry_delay: 1.0
     exponential_backoff: true
     
     # Request configuration
     user_agent: "ScrambleBench/0.1.0"
     headers:
       "HTTP-Referer": "https://github.com/nathanrice/scramblebench"

Model-Specific Settings
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Model-specific configurations
   models:
     "openai/gpt-4":
       temperature: 0.0
       max_tokens: 2000
       timeout: 60
       
     "anthropic/claude-3-sonnet":
       temperature: 0.1
       max_tokens: 4000
       timeout: 45
       
     "meta-llama/llama-2-70b-chat":
       temperature: 0.3
       max_tokens: 1000
       timeout: 30

Evaluation Configuration
------------------------

Evaluation Pipeline Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   evaluation:
     # Experiment settings
     experiment_name: "robustness_evaluation"
     description: "Testing model robustness across transformations"
     
     # Data settings
     benchmark_paths:
       - "data/benchmarks/math_problems.json"
       - "data/benchmarks/reading_comprehension.json"
     max_samples: 500
     sample_strategy: "random"  # random, stratified, first
     
     # Model settings
     models:
       - name: "openai/gpt-4"
         provider: "openrouter"
         temperature: 0.0
       - name: "anthropic/claude-3-sonnet"
         provider: "openrouter"
         temperature: 0.0
     
     # Transformation settings
     transformations:
       enabled_types:
         - "language_translation"
         - "synonym_replacement"
         - "proper_noun_swap"
       language_complexity: 5
       synonym_rate: 0.3
       proper_noun_strategy: "random"
     
     # Output settings
     output_dir: "results"
     save_intermediate: true
     generate_plots: true
     calculate_significance: true

Transformation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   transformations:
     # Language translation settings
     language_translation:
       enabled: true
       language_types: ["substitution", "phonetic"]
       complexity_range: [3, 7]
       preserve_numbers: true
       preserve_proper_nouns: true
       
     # Synonym replacement settings
     synonym_replacement:
       enabled: true
       replacement_rate: 0.3
       pos_filter: ["NOUN", "VERB", "ADJ"]
       preserve_sentiment: true
       similarity_threshold: 0.8
       
     # Proper noun replacement settings
     proper_noun_swap:
       enabled: true
       strategy: "random"  # random, systematic, cultural
       preserve_gender: true
       preserve_origin: false
       replacement_database: "data/proper_nouns.json"

Data Configuration
------------------

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data:
     # Directory settings
     benchmarks_dir: "data/benchmarks"
     languages_dir: "data/languages"
     results_dir: "data/results"
     cache_dir: "data/cache"
     temp_dir: "/tmp/scramblebench"
     
     # File format settings
     default_format: "json"
     supported_formats: ["json", "jsonl", "csv", "parquet"]
     encoding: "utf-8"
     
     # Caching settings
     enable_caching: true
     max_cache_size: 10000
     cache_ttl: 86400  # 24 hours
     cache_compression: true
     
     # Loading settings
     batch_size: 1000
     parallel_loading: true
     max_workers: 4
     chunk_size: 10000

Data Processing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   data_processing:
     # Text preprocessing
     preprocessing:
       normalize_unicode: true
       strip_whitespace: true
       remove_control_chars: true
       max_length: 10000
       
     # Validation settings
     validation:
       strict_mode: false
       required_fields: ["question", "answer"]
       validate_encoding: true
       check_duplicates: true
       
     # Filtering settings
     filtering:
       min_length: 10
       max_length: 5000
       language_filter: "en"
       quality_threshold: 0.8

Logging Configuration
---------------------

Basic Logging Setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   logging:
     # Log level
     level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
     
     # Log format
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     date_format: "%Y-%m-%d %H:%M:%S"
     
     # Log destinations
     console: true
     file: "logs/scramblebench.log"
     
     # File settings
     max_file_size: "10MB"
     backup_count: 5
     
     # Logger settings
     loggers:
       "scramblebench.core": "INFO"
       "scramblebench.translation": "DEBUG"
       "scramblebench.evaluation": "WARNING"

Advanced Logging Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   logging:
     # Multiple handlers
     handlers:
       console:
         class: "StreamHandler"
         level: "INFO"
         formatter: "standard"
         
       file:
         class: "RotatingFileHandler"
         level: "DEBUG"
         filename: "logs/scramblebench.log"
         maxBytes: 10485760  # 10MB
         backupCount: 5
         formatter: "detailed"
         
       error_file:
         class: "FileHandler"
         level: "ERROR"
         filename: "logs/errors.log"
         formatter: "detailed"
     
     # Formatters
     formatters:
       standard:
         format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
         
       detailed:
         format: "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s"

Configuration Best Practices
-----------------------------

1. **Use Environment Variables for Secrets**

   .. code-block:: yaml

      model:
        api_key: "${OPENROUTER_API_KEY}"  # Good
        # api_key: "sk-123456789..."       # Bad - hardcoded secret

2. **Separate Configs by Environment**

   .. code-block:: bash

      # Development
      scramblebench --config config/dev.yaml
      
      # Production  
      scramblebench --config config/prod.yaml

3. **Validate Configuration**

   .. code-block:: python

      config = Config("config.yaml")
      if not config.validate():
          raise ConfigError("Invalid configuration")

4. **Use Reasonable Defaults**

   .. code-block:: yaml

      model:
        timeout: 30        # Reasonable default
        rate_limit: 10.0   # Conservative default
        max_retries: 3     # Safe default

5. **Document Custom Settings**

   .. code-block:: yaml

      # Custom evaluation for domain-specific benchmarks
      benchmark:
        evaluation_mode: "semantic_similarity"  # Better for creative tasks
        evaluation_threshold: 0.75              # Slightly lower threshold

Configuration Examples
----------------------

Quick Start Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # quick_start.yaml - Minimal configuration for getting started
   
   model:
     default_model: "openai/gpt-3.5-turbo"
     timeout: 30
   
   data:
     benchmarks_dir: "data/benchmarks"
     results_dir: "data/results"
   
   logging:
     level: "INFO"

Research Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # research.yaml - Configuration for research experiments
   
   evaluation:
     max_samples: 1000
     calculate_significance: true
     generate_plots: true
     
   transformations:
     enabled_types:
       - "language_translation"
       - "synonym_replacement"
       - "proper_noun_swap"
     language_complexity: 7
     
   models:
     - name: "openai/gpt-4"
       temperature: 0.0
     - name: "anthropic/claude-3-sonnet"
       temperature: 0.0
     - name: "meta-llama/llama-2-70b-chat"
       temperature: 0.0
   
   logging:
     level: "DEBUG"
     file: "logs/research_experiment.log"

Production Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # production.yaml - Configuration for production use
   
   model:
     rate_limit: 5.0  # Conservative rate limiting
     timeout: 60
     max_retries: 5
     
   evaluation:
     batch_size: 10
     max_concurrent_requests: 3
     result_caching: true
     
   data:
     max_cache_size: 50000
     enable_caching: true
     
   logging:
     level: "WARNING"
     file: "/var/log/scramblebench/scramblebench.log"
     
   performance:
     memory_limit: "16GB"
     cleanup_temp_files: true

Configuration Validation
-------------------------

ScrambleBench includes comprehensive configuration validation:

.. code-block:: python

   from scramblebench.utils.config import Config, ConfigValidator
   
   # Load and validate configuration
   config = Config("config.yaml")
   validator = ConfigValidator()
   
   # Validate configuration
   is_valid, errors = validator.validate(config)
   
   if not is_valid:
       for error in errors:
           print(f"Configuration error: {error}")

Common validation errors and solutions:

.. code-block:: bash

   # Error: Invalid model name
   # Solution: Use correct OpenRouter model format
   model.default_model: "openai/gpt-3.5-turbo"  # Correct
   
   # Error: Invalid timeout value
   # Solution: Use positive integer/float
   model.timeout: 30  # Correct (not -30 or "30s")
   
   # Error: Missing required API key
   # Solution: Set environment variable or config value
   export OPENROUTER_API_KEY="your_key"

Troubleshooting Configuration
-----------------------------

**Configuration Not Loading:**

.. code-block:: bash

   # Check if file exists and is readable
   ls -la config.yaml
   
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   
   # Debug configuration loading
   scramblebench --verbose --config config.yaml language list

**Environment Variable Issues:**

.. code-block:: bash

   # Check if environment variables are set
   env | grep SCRAMBLEBENCH
   
   # Test variable substitution
   python -c "
   import os
   print('API Key:', os.getenv('OPENROUTER_API_KEY', 'NOT_SET'))
   "

**Permission Issues:**

.. code-block:: bash

   # Check directory permissions
   ls -la data/
   
   # Create missing directories
   mkdir -p data/{benchmarks,languages,results,cache}
   
   # Fix permissions
   chmod 755 data/
   chmod 644 config.yaml

Next Steps
----------

* :doc:`evaluation_pipeline` - Configure evaluation workflows
* :doc:`../api/utils` - Configuration API reference
* :doc:`../examples/configuration_examples` - More configuration examples