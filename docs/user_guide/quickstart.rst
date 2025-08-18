Quick Start Guide
=================

This guide will get you up and running with ScrambleBench in minutes.

Overview
--------

ScrambleBench is a contamination-resistant LLM evaluation toolkit that uses two main approaches:

1. **Translation Benchmarks**: Transform problems into constructed languages
2. **Long Context Benchmarks**: Modify documents while preserving meaning

Basic Setup
-----------

1. **Install ScrambleBench**:

   .. code-block:: bash

      git clone https://github.com/nathanrice/scramblebench.git
      cd scramblebench
      uv sync

2. **Set up API key**:

   .. code-block:: bash

      export OPENROUTER_API_KEY="your_openrouter_api_key"

3. **Verify installation**:

   .. code-block:: bash

      scramblebench --help

Your First Benchmark
---------------------

Let's create a simple translation benchmark:

Command Line Approach
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate a constructed language
   scramblebench language generate mylang --type substitution --complexity 5

   # 2. Transform some text
   scramblebench transform text "What is the capital of France?" mylang

   # 3. Create evaluation dataset
   echo '[
     {"question": "What is 2+2?", "answer": "4"},
     {"question": "What is the capital of France?", "answer": "Paris"},
     {"question": "What color is grass?", "answer": "Green"}
   ]' > simple_qa.json

   # 4. Run evaluation
   scramblebench evaluate run \
     --models "openai/gpt-3.5-turbo" \
     --benchmarks simple_qa.json \
     --experiment-name "my_first_test" \
     --max-samples 3

Python API Approach
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageType

   # Create simple test data
   test_data = [
       {"question": "What is 2+2?", "answer": "4"},
       {"question": "What is the capital of France?", "answer": "Paris"},
       {"question": "What color is grass?", "answer": "Green"}
   ]

   # Save test data
   with open("simple_qa.json", "w") as f:
       json.dump(test_data, f)

   # Create translation benchmark
   benchmark = TranslationBenchmark(
       source_dataset="simple_qa.json",
       language_type=LanguageType.SUBSTITUTION,
       language_complexity=5
   )

   # Initialize model
   model = OpenRouterClient(
       model_name="openai/gpt-3.5-turbo",
       api_key="your_openrouter_api_key"  # or from environment
   )

   # Run benchmark
   result = benchmark.run(model, num_samples=3)

   # Print results
   print(f"Benchmark: {result.benchmark_name}")
   print(f"Model: {result.model_name}")
   print(f"Score: {result.score:.2%}")
   print(f"Duration: {result.duration:.1f} seconds")

Key Concepts
------------

Constructed Languages
~~~~~~~~~~~~~~~~~~~~~~

ScrambleBench creates artificial languages that preserve logical structure:

.. code-block:: python

   from scramblebench.translation.language_generator import LanguageGenerator, LanguageType

   # Create language generator
   generator = LanguageGenerator(seed=42)

   # Generate different language types
   substitution_lang = generator.generate_language(
       name="simple_sub",
       language_type=LanguageType.SUBSTITUTION,
       complexity=3
   )

   phonetic_lang = generator.generate_language(
       name="phonetic_transform", 
       language_type=LanguageType.PHONETIC,
       complexity=5
   )

   # Transform text
   original = "The quick brown fox jumps over the lazy dog"
   translated = substitution_lang.transform(original)
   print(f"Original: {original}")
   print(f"Translated: {translated}")

**Language Types:**

* **Substitution**: Simple character/word substitutions
* **Phonetic**: Phonetically plausible transformations
* **Scrambled**: Systematic character scrambling
* **Synthetic**: Fully artificial grammar systems

Model Integration
~~~~~~~~~~~~~~~~~

ScrambleBench supports multiple LLM providers through a unified interface:

.. code-block:: python

   from scramblebench.llm import OpenRouterClient, ModelConfig

   # Basic client
   client = OpenRouterClient("openai/gpt-4")

   # With configuration
   config = ModelConfig(
       temperature=0.0,
       max_tokens=100,
       timeout=30
   )
   client = OpenRouterClient("anthropic/claude-3-sonnet", config=config)

   # Generate response
   response = client.generate("What is the capital of France?")
   print(response.text)

Configuration
~~~~~~~~~~~~~

Use YAML configuration for complex setups:

.. code-block:: yaml

   # config.yaml
   benchmark:
     random_seed: 42
     evaluation_mode: "exact_match"
     evaluation_threshold: 0.8

   model:
     default_provider: "openrouter"
     timeout: 30
     rate_limit: 10.0

   data:
     benchmarks_dir: "data/benchmarks"
     results_dir: "data/results"

.. code-block:: python

   from scramblebench.utils.config import Config

   # Load configuration
   config = Config("config.yaml")

   # Use with benchmark
   benchmark = TranslationBenchmark(
       source_dataset="qa_data.json",
       config=config
   )

Common Workflows
----------------

1. Language Exploration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate multiple languages
   for type in substitution phonetic scrambled; do
     scramblebench language generate "${type}_lang" --type $type --complexity 5
   done

   # List generated languages
   scramblebench language list --format json

   # Examine a specific language
   scramblebench language show substitution_lang --show-rules --limit 10

2. Text Transformation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Transform text with different methods
   scramblebench transform text "Hello world" substitution_lang

   # Proper noun replacement
   scramblebench transform proper-nouns "John went to New York" --strategy random

   # Synonym replacement  
   scramblebench transform synonyms "The big dog ran fast" --replacement-rate 0.5

3. Batch Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Extract vocabulary from dataset
   scramblebench batch extract-vocab my_dataset.json --min-freq 2

   # Transform entire dataset
   scramblebench batch transform my_dataset.json mylang --batch-size 50

4. Comprehensive Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Multi-model evaluation
   scramblebench evaluate run \
     --models "openai/gpt-4,anthropic/claude-3-sonnet,meta-llama/llama-2-70b-chat" \
     --benchmarks "math.json,reading.json,logic.json" \
     --experiment-name "robustness_test" \
     --transformations "language_translation,synonym_replacement" \
     --max-samples 100 \
     --generate-plots

   # Analyze results
   scramblebench evaluate analyze robustness_test

   # Compare experiments
   scramblebench evaluate compare exp1 exp2 exp3

Long Context Benchmarks
------------------------

For document-based evaluation:

.. code-block:: python

   from scramblebench import LongContextBenchmark
   from scramblebench.longcontext.document_transformer import TransformationType

   # Create long context data
   long_context_data = [{
       "id": "doc1",
       "document": """
       Artificial Intelligence (AI) is a branch of computer science 
       that aims to create intelligent machines. Machine learning is 
       a subset of AI that focuses on algorithms that can learn from data.
       """,
       "questions": ["What is AI?", "What is machine learning?"],
       "answers": [
           "A branch of computer science for intelligent machines",
           "A subset of AI focused on learning from data"
       ]
   }]

   # Create benchmark
   benchmark = LongContextBenchmark(
       dataset_name="long_context_data.json",
       transformation_type=TransformationType.HYBRID,
       language_complexity=4
   )

   # Run evaluation
   result = benchmark.run(model, num_samples=1)

Advanced Features
-----------------

Custom Evaluation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.core.evaluator import Evaluator

   def semantic_similarity_evaluator(predicted, expected, **kwargs):
       # Custom evaluation logic
       similarity = calculate_similarity(predicted, expected)
       return {
           'correct': similarity > 0.8,
           'score': similarity,
           'explanation': f"Similarity: {similarity:.3f}"
       }

   # Register custom evaluator
   evaluator = Evaluator()
   evaluator.register_custom_evaluator("semantic", semantic_similarity_evaluator)

Result Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Load and analyze results
   from scramblebench.core.reporter import Reporter

   reporter = Reporter()
   results = reporter.load_results("my_experiment")

   # Generate detailed report
   report = reporter.generate_report(
       results, 
       title="Robustness Analysis",
       include_plots=True
   )

   # Export to different formats
   reporter.export_report(report, "analysis.html", format="html")
   reporter.export_report(report, "analysis.pdf", format="pdf")

Troubleshooting
---------------

**API Key Issues:**

.. code-block:: bash

   # Verify API key
   echo $OPENROUTER_API_KEY

   # Test API access
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

**Memory Issues:**

.. code-block:: yaml

   # Reduce memory usage in config
   evaluation:
     batch_size: 5
     max_samples: 50

   model:
     rate_limit: 2.0

**Slow Evaluation:**

.. code-block:: bash

   # Use smaller models for testing
   scramblebench evaluate run --models "openai/gpt-3.5-turbo" --max-samples 10

   # Enable caching
   scramblebench --config config.yaml evaluate run ...

Next Steps
----------

Now that you're familiar with the basics:

1. **Explore the CLI**: See :doc:`cli_guide` for comprehensive command reference
2. **Configuration**: Learn advanced setup in :doc:`configuration`  
3. **Evaluation Pipeline**: Deep dive into :doc:`evaluation_pipeline`
4. **Examples**: Check the ``examples/`` directory for complete workflows
5. **API Reference**: Browse :doc:`../api/index` for detailed documentation

**Key Resources:**

* :doc:`../tutorials/translation_benchmarks` - Detailed translation benchmark tutorial
* :doc:`../tutorials/long_context_benchmarks` - Long context evaluation guide
* :doc:`../tutorials/custom_models` - Integrating custom models
* :doc:`../examples/basic_usage` - More usage examples