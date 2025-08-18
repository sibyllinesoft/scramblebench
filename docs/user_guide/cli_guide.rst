Command Line Interface Guide
=============================

ScrambleBench provides a comprehensive command-line interface for all major functionality. This guide covers all commands, options, and usage patterns.

Overview
--------

The CLI is organized into several command groups:

* **language** - Constructed language management
* **transform** - Text transformation operations  
* **batch** - Batch processing of datasets
* **evaluate** - Model evaluation workflows
* **util** - Utility commands

Basic Usage
-----------

.. code-block:: bash

   scramblebench [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGS]

**Global Options:**

* ``--data-dir PATH`` - Set data directory (default: ./data)
* ``--config FILE`` - Use configuration file
* ``--verbose`` - Enable verbose output
* ``--quiet`` - Suppress output
* ``--output-format FORMAT`` - Output format: text, json, yaml
* ``--help`` - Show help message

Language Commands
-----------------

Manage constructed languages for translation benchmarks.

Generate Language
~~~~~~~~~~~~~~~~~

Create a new constructed language:

.. code-block:: bash

   scramblebench language generate NAME [OPTIONS]

**Arguments:**
* ``NAME`` - Name for the generated language

**Options:**
* ``--type TYPE`` - Language type: substitution, phonetic, scrambled, synthetic
* ``--complexity LEVEL`` - Complexity level (1-10, default: 5)
* ``--vocab-size SIZE`` - Vocabulary size (default: 1000)
* ``--seed SEED`` - Random seed for reproducibility
* ``--save-rules`` - Save detailed transformation rules

**Examples:**

.. code-block:: bash

   # Basic substitution language
   scramblebench language generate simple_sub --type substitution --complexity 3

   # Complex phonetic language with custom vocabulary
   scramblebench language generate advanced_phon \
     --type phonetic \
     --complexity 8 \
     --vocab-size 2000 \
     --seed 42

   # Synthetic language with rule export
   scramblebench language generate synthetic_test \
     --type synthetic \
     --complexity 6 \
     --save-rules

List Languages
~~~~~~~~~~~~~~

Display available languages:

.. code-block:: bash

   scramblebench language list [OPTIONS]

**Options:**
* ``--format FORMAT`` - Output format: text, json, yaml
* ``--filter PATTERN`` - Filter languages by name pattern
* ``--sort-by FIELD`` - Sort by: name, type, complexity, created
* ``--verbose`` - Show detailed information

**Examples:**

.. code-block:: bash

   # List all languages
   scramblebench language list

   # JSON output for scripting
   scramblebench language list --format json

   # Filter and sort
   scramblebench language list --filter "test_*" --sort-by complexity

Show Language Details
~~~~~~~~~~~~~~~~~~~~~

Display detailed information about a language:

.. code-block:: bash

   scramblebench language show NAME [OPTIONS]

**Arguments:**
* ``NAME`` - Language name

**Options:**
* ``--show-rules`` - Display transformation rules
* ``--show-vocab`` - Display vocabulary mappings
* ``--limit N`` - Limit number of rules/vocab shown
* ``--format FORMAT`` - Output format

**Examples:**

.. code-block:: bash

   # Basic information
   scramblebench language show my_language

   # Show transformation rules
   scramblebench language show my_language --show-rules --limit 20

   # Complete details in JSON
   scramblebench language show my_language --show-rules --show-vocab --format json

Delete Language
~~~~~~~~~~~~~~~

Remove a language:

.. code-block:: bash

   scramblebench language delete NAME [OPTIONS]

**Arguments:**
* ``NAME`` - Language name

**Options:**
* ``--force`` - Skip confirmation prompt
* ``--backup`` - Create backup before deletion

**Examples:**

.. code-block:: bash

   # Interactive deletion
   scramblebench language delete old_language

   # Force deletion without prompt
   scramblebench language delete temp_language --force

Transform Commands
------------------

Apply various text transformations.

Transform Text
~~~~~~~~~~~~~~

Transform text using a constructed language:

.. code-block:: bash

   scramblebench transform text TEXT LANGUAGE [OPTIONS]

**Arguments:**
* ``TEXT`` - Text to transform
* ``LANGUAGE`` - Language name to use

**Options:**
* ``--preserve-numbers`` - Keep numbers unchanged
* ``--preserve-proper-nouns`` - Keep proper nouns unchanged  
* ``--output-format FORMAT`` - Output format
* ``--show-mapping`` - Show character/word mappings

**Examples:**

.. code-block:: bash

   # Basic transformation
   scramblebench transform text "Hello world" my_language

   # With preservation options
   scramblebench transform text "John has 5 apples" my_language \
     --preserve-numbers --preserve-proper-nouns

   # JSON output with mappings
   scramblebench transform text "Test sentence" my_language \
     --output-format json --show-mapping

Replace Proper Nouns
~~~~~~~~~~~~~~~~~~~~

Replace proper nouns with alternatives:

.. code-block:: bash

   scramblebench transform proper-nouns TEXT [OPTIONS]

**Arguments:**
* ``TEXT`` - Text containing proper nouns

**Options:**
* ``--strategy STRATEGY`` - Replacement strategy: random, systematic, cultural
* ``--preserve-gender`` - Maintain gender in name replacements
* ``--preserve-origin`` - Maintain cultural origin of names
* ``--seed SEED`` - Random seed for reproducibility

**Examples:**

.. code-block:: bash

   # Random replacement
   scramblebench transform proper-nouns "John went to Paris" --strategy random

   # Cultural preservation
   scramblebench transform proper-nouns "Maria visited Tokyo" \
     --strategy cultural --preserve-origin

Replace Synonyms
~~~~~~~~~~~~~~~~

Replace words with synonyms:

.. code-block:: bash

   scramblebench transform synonyms TEXT [OPTIONS]

**Arguments:**
* ``TEXT`` - Text to transform

**Options:**
* ``--replacement-rate RATE`` - Fraction of words to replace (0.0-1.0)
* ``--pos-filter POS`` - Only replace specific parts of speech
* ``--preserve-sentiment`` - Maintain text sentiment
* ``--seed SEED`` - Random seed

**Examples:**

.. code-block:: bash

   # Replace 30% of words
   scramblebench transform synonyms "The big dog ran fast" --replacement-rate 0.3

   # Only replace adjectives and verbs
   scramblebench transform synonyms "The quick brown fox jumps" \
     --pos-filter "ADJ,VERB" --preserve-sentiment

Batch Commands
--------------

Process datasets in batch operations.

Extract Vocabulary
~~~~~~~~~~~~~~~~~~

Extract vocabulary from benchmark datasets:

.. code-block:: bash

   scramblebench batch extract-vocab FILE [OPTIONS]

**Arguments:**
* ``FILE`` - Input dataset file

**Options:**
* ``--min-freq N`` - Minimum word frequency (default: 1)
* ``--max-words N`` - Maximum vocabulary size
* ``--output-file FILE`` - Output vocabulary file
* ``--include-pos`` - Include part-of-speech tags
* ``--format FORMAT`` - Output format

**Examples:**

.. code-block:: bash

   # Basic vocabulary extraction
   scramblebench batch extract-vocab dataset.json --min-freq 2

   # Large vocabulary with POS tags
   scramblebench batch extract-vocab large_dataset.json \
     --min-freq 5 --max-words 5000 --include-pos

Transform Dataset
~~~~~~~~~~~~~~~~~

Apply transformations to entire datasets:

.. code-block:: bash

   scramblebench batch transform FILE LANGUAGE [OPTIONS]

**Arguments:**
* ``FILE`` - Input dataset file
* ``LANGUAGE`` - Language for transformation

**Options:**
* ``--output-file FILE`` - Output file path
* ``--batch-size N`` - Processing batch size
* ``--preserve-ids`` - Keep original IDs
* ``--include-original`` - Include original text alongside transformed

**Examples:**

.. code-block:: bash

   # Transform entire dataset
   scramblebench batch transform questions.json my_language \
     --output-file transformed_questions.json

   # Large dataset with batching
   scramblebench batch transform large_dataset.json my_language \
     --batch-size 100 --preserve-ids --include-original

Evaluation Commands
-------------------

Run comprehensive model evaluations.

Run Evaluation
~~~~~~~~~~~~~~

Execute evaluation pipeline:

.. code-block:: bash

   scramblebench evaluate run [OPTIONS]

**Options:**
* ``--models MODELS`` - Comma-separated model names
* ``--benchmarks FILES`` - Comma-separated benchmark files
* ``--experiment-name NAME`` - Name for this experiment
* ``--config FILE`` - Configuration file
* ``--transformations TYPES`` - Transformation types to apply
* ``--max-samples N`` - Limit number of samples
* ``--generate-plots`` - Create visualization plots
* ``--calculate-significance`` - Run statistical significance tests
* ``--output-dir DIR`` - Results output directory

**Examples:**

.. code-block:: bash

   # Basic evaluation
   scramblebench evaluate run \
     --models "openai/gpt-3.5-turbo" \
     --benchmarks "qa_dataset.json" \
     --experiment-name "basic_test"

   # Comprehensive evaluation
   scramblebench evaluate run \
     --models "openai/gpt-4,anthropic/claude-3-sonnet,meta-llama/llama-2-70b-chat" \
     --benchmarks "math.json,reading.json,logic.json" \
     --experiment-name "robustness_study" \
     --transformations "language_translation,synonym_replacement,proper_noun_swap" \
     --max-samples 200 \
     --generate-plots \
     --calculate-significance

   # Config-based evaluation
   scramblebench evaluate run --config evaluation_config.yaml

Analyze Results
~~~~~~~~~~~~~~~

Analyze completed evaluation results:

.. code-block:: bash

   scramblebench evaluate analyze EXPERIMENT_NAME [OPTIONS]

**Arguments:**
* ``EXPERIMENT_NAME`` - Name of experiment to analyze

**Options:**
* ``--metrics METRICS`` - Specific metrics to analyze
* ``--generate-report`` - Create detailed HTML report
* ``--export-data`` - Export raw data for external analysis
* ``--format FORMAT`` - Output format

**Examples:**

.. code-block:: bash

   # Basic analysis
   scramblebench evaluate analyze my_experiment

   # Detailed report generation
   scramblebench evaluate analyze robustness_study \
     --generate-report --export-data

Compare Experiments
~~~~~~~~~~~~~~~~~~~

Compare multiple evaluation experiments:

.. code-block:: bash

   scramblebench evaluate compare EXPERIMENT1 EXPERIMENT2 [EXPERIMENT3...] [OPTIONS]

**Arguments:**
* ``EXPERIMENT1, EXPERIMENT2, ...`` - Experiment names to compare

**Options:**
* ``--metrics METRICS`` - Metrics to compare
* ``--significance-test`` - Run statistical comparison tests
* ``--generate-plots`` - Create comparison visualizations
* ``--output-file FILE`` - Save comparison report

**Examples:**

.. code-block:: bash

   # Compare two experiments
   scramblebench evaluate compare baseline_test robustness_test

   # Comprehensive comparison
   scramblebench evaluate compare exp1 exp2 exp3 \
     --significance-test --generate-plots \
     --output-file comparison_report.html

Utility Commands
----------------

Additional utility operations.

Language Statistics
~~~~~~~~~~~~~~~~~~~

Get statistics about a language:

.. code-block:: bash

   scramblebench util stats LANGUAGE [OPTIONS]

**Arguments:**
* ``LANGUAGE`` - Language name

**Options:**
* ``--detailed`` - Show detailed statistics
* ``--format FORMAT`` - Output format

**Examples:**

.. code-block:: bash

   # Basic stats
   scramblebench util stats my_language

   # Detailed analysis
   scramblebench util stats complex_language --detailed --format json

Export Rules
~~~~~~~~~~~~

Export language transformation rules:

.. code-block:: bash

   scramblebench util export-rules LANGUAGE [OPTIONS]

**Arguments:**
* ``LANGUAGE`` - Language name

**Options:**
* ``--format FORMAT`` - Export format: json, yaml, csv
* ``--output-file FILE`` - Output file path
* ``--rule-types TYPES`` - Types of rules to export

**Examples:**

.. code-block:: bash

   # Export as JSON
   scramblebench util export-rules my_language --format json

   # Export specific rule types
   scramblebench util export-rules complex_language \
     --rule-types "substitution,phonetic" --output-file rules.yaml

Validate Transformation
~~~~~~~~~~~~~~~~~~~~~~~

Validate transformation quality:

.. code-block:: bash

   scramblebench util validate LANGUAGE TEXT [OPTIONS]

**Arguments:**
* ``LANGUAGE`` - Language name
* ``TEXT`` - Text to validate transformation

**Options:**
* ``--check-reversibility`` - Test if transformation is reversible
* ``--similarity-threshold THRESHOLD`` - Similarity threshold for validation

**Examples:**

.. code-block:: bash

   # Basic validation
   scramblebench util validate my_language "Test sentence"

   # Detailed validation
   scramblebench util validate my_language "Complex text example" \
     --check-reversibility --similarity-threshold 0.8

Configuration Files
-------------------

Use YAML configuration files for complex setups:

.. code-block:: yaml

   # evaluation_config.yaml
   experiment_name: comprehensive_evaluation
   description: Multi-model robustness testing
   
   benchmark_paths:
     - data/benchmarks/math_problems.json
     - data/benchmarks/reading_comprehension.json
   
   models:
     - name: openai/gpt-4
       provider: openrouter
       temperature: 0.0
     - name: anthropic/claude-3-sonnet
       provider: openrouter
       temperature: 0.0
   
   transformations:
     enabled_types:
       - language_translation
       - synonym_replacement
       - proper_noun_swap
     synonym_rate: 0.3
   
   max_samples: 100
   generate_plots: true
   calculate_significance: true

.. code-block:: bash

   # Use configuration file
   scramblebench evaluate run --config evaluation_config.yaml

Environment Variables
---------------------

Configure ScrambleBench with environment variables:

.. code-block:: bash

   # API Keys
   export OPENROUTER_API_KEY="your_api_key"
   
   # Configuration
   export SCRAMBLEBENCH_DATA_DIR="/path/to/data"
   export SCRAMBLEBENCH_LOG_LEVEL="INFO"
   export SCRAMBLEBENCH_CONFIG_FILE="/path/to/config.yaml"
   
   # Model settings
   export SCRAMBLEBENCH_DEFAULT_MODEL="openai/gpt-3.5-turbo"
   export SCRAMBLEBENCH_DEFAULT_TIMEOUT="30"

Scripting and Automation
-------------------------

Integration with shell scripts and automation:

.. code-block:: bash

   #!/bin/bash
   # automated_evaluation.sh
   
   # Set up environment
   export OPENROUTER_API_KEY="your_key"
   DATA_DIR="./evaluation_data"
   
   # Generate languages
   for complexity in 3 5 7; do
     scramblebench language generate "test_lang_${complexity}" \
       --type substitution --complexity $complexity
   done
   
   # Run evaluations
   for lang in test_lang_3 test_lang_5 test_lang_7; do
     scramblebench evaluate run \
       --models "openai/gpt-3.5-turbo,openai/gpt-4" \
       --benchmarks "${DATA_DIR}/questions.json" \
       --experiment-name "complexity_${lang##*_}" \
       --max-samples 50
   done
   
   # Compare results
   scramblebench evaluate compare complexity_3 complexity_5 complexity_7 \
     --generate-plots --output-file complexity_comparison.html

**JSON Processing Example:**

.. code-block:: bash

   # Extract scores from evaluation results
   scramblebench evaluate analyze my_experiment --format json | \
     jq '.results[] | {model: .model_name, score: .score}'
   
   # List languages with specific criteria
   scramblebench language list --format json | \
     jq '.languages[] | select(.complexity >= 5) | .name'

Performance Tips
----------------

**For Large Datasets:**

.. code-block:: bash

   # Use batch processing
   scramblebench batch transform large_dataset.json my_language \
     --batch-size 50 --output-file processed.json

   # Limit samples for testing
   scramblebench evaluate run --max-samples 10 --models "gpt-3.5-turbo"

**For Multiple Experiments:**

.. code-block:: bash

   # Use configuration files to avoid repetition
   scramblebench evaluate run --config base_config.yaml

   # Cache language generation
   scramblebench language generate shared_lang --type substitution
   # Reuse across multiple evaluations

**Memory Management:**

.. code-block:: yaml

   # In configuration file
   evaluation:
     batch_size: 10
     max_concurrent: 2
   
   model:
     timeout: 30
     rate_limit: 5.0

Common Patterns
---------------

**Daily Testing Pipeline:**

.. code-block:: bash

   #!/bin/bash
   # daily_test.sh
   DATE=$(date +%Y%m%d)
   
   scramblebench evaluate run \
     --models "openai/gpt-3.5-turbo" \
     --benchmarks "daily_test_set.json" \
     --experiment-name "daily_${DATE}" \
     --max-samples 20
   
   scramblebench evaluate analyze "daily_${DATE}" --generate-report

**Language Development Workflow:**

.. code-block:: bash

   # Create and test new language
   scramblebench language generate new_lang --type phonetic --complexity 6
   scramblebench util validate new_lang "Test validation sentence"
   scramblebench transform text "Sample text" new_lang
   
   # If satisfied, use in evaluation
   scramblebench evaluate run --models "gpt-3.5-turbo" \
     --benchmarks "test_set.json" --experiment-name "new_lang_test"

Troubleshooting
---------------

**Common Issues:**

.. code-block:: bash

   # Check CLI installation
   which scramblebench
   scramblebench --version
   
   # Verify API key
   echo $OPENROUTER_API_KEY
   
   # Test API connectivity
   scramblebench evaluate run --models "gpt-3.5-turbo" \
     --benchmarks "simple_test.json" --max-samples 1
   
   # Check data directory permissions
   ls -la $(scramblebench --data-dir)
   
   # Enable verbose logging
   scramblebench --verbose language list

**Debug Mode:**

.. code-block:: bash

   # Maximum verbosity
   scramblebench --verbose --log-level DEBUG evaluate run ...
   
   # Check configuration loading
   scramblebench --config myconfig.yaml --verbose language list

See Also
--------

* :doc:`configuration` - Detailed configuration options
* :doc:`evaluation_pipeline` - Evaluation system overview
* :doc:`../api/cli` - Complete CLI API reference
* :doc:`../examples/basic_usage` - More usage examples