.. ScrambleBench documentation master file

ScrambleBench Documentation
===========================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: MIT License

.. image:: https://img.shields.io/badge/tests-pytest-orange.svg
   :target: tests/
   :alt: Tests

**ScrambleBench** is a comprehensive LLM benchmarking toolkit designed to address the critical challenge of training data contamination in language model evaluation. Through innovative use of constructed languages and document transformation techniques, ScrambleBench provides contamination-resistant evaluation methods that ensure reliable and unbiased assessment of large language models.

🌟 Key Features
===============

🌍 **Translation Benchmarks**
  Transform existing problems into constructed languages while preserving logical structure and solvability.

📚 **Long Context Benchmarks**
  Modify long documents and Q&A sets through sophisticated translation and transformation strategies.

🔧 **LLM Integration**
  Unified interface supporting 100+ models through OpenRouter API with built-in rate limiting and async support.

📊 **Evaluation Pipeline**
  Comprehensive robustness testing with multi-model support, statistical analysis, and rich visualizations.

⚙️ **Flexible Configuration**
  YAML-based configuration system with environment variable support and extensible data loaders.

🖥️ **Command Line Interface**
  Full-featured CLI for language management, batch processing, and evaluation workflows.

Quick Start
===========

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/nathanrice/scramblebench.git
   cd scramblebench

   # Install with uv (recommended)
   uv sync

   # Or with pip
   pip install -e .

   # Install development dependencies
   uv sync --group dev

   # Install documentation dependencies
   uv sync --group docs

Basic Usage
-----------

**Translation Benchmark Example:**

.. code-block:: python

   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageType

   # Create a translation benchmark
   benchmark = TranslationBenchmark(
       source_dataset="simple_qa",
       language_type=LanguageType.SUBSTITUTION,
       language_complexity=5
   )

   # Initialize your model
   model = OpenRouterClient(
       model_name="openai/gpt-4",
       api_key="your-openrouter-key"
   )

   # Run the benchmark
   result = benchmark.run(model, num_samples=50)
   print(f"Accuracy: {result.score:.2%}")

**Command Line Interface:**

.. code-block:: bash

   # Generate a constructed language
   scramblebench language generate mylang --type substitution --complexity 5

   # Transform text using the language
   scramblebench transform text "Hello world" mylang

   # Run comprehensive evaluation
   scramblebench evaluate run \
     --models "anthropic/claude-3-sonnet,openai/gpt-4" \
     --benchmarks "data/benchmarks/logic_reasoning.json" \
     --experiment-name "robustness_test"

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/cli_guide
   user_guide/configuration
   user_guide/evaluation_pipeline

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/translation_benchmarks
   tutorials/long_context_benchmarks
   tutorials/custom_models
   tutorials/batch_evaluation
   tutorials/configuration_examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/translation
   api/longcontext
   api/llm
   api/evaluation
   api/utils
   api/cli

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/advanced_usage
   examples/custom_integration
   examples/notebooks

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/testing
   development/architecture

.. toctree::
   :maxdepth: 1
   :caption: About

   about/changelog
   about/license
   about/citation

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

What Makes ScrambleBench Unique?
===============================

Training Data Contamination Problem
-----------------------------------

Modern large language models are trained on vast datasets that may inadvertently include evaluation benchmarks, leading to inflated performance scores that don't reflect true capabilities. This "contamination" problem makes it difficult to:

- Trust benchmark results
- Compare models fairly
- Assess genuine reasoning abilities
- Develop reliable evaluation metrics

ScrambleBench's Solution
-----------------------

ScrambleBench addresses contamination through two innovative approaches:

1. **Constructed Language Translation**
   - Creates artificial languages that preserve logical structure
   - Maintains problem solvability while eliminating lexical overlap
   - Supports multiple language types (substitution, phonetic, scrambled, synthetic)
   - Provides complete translation mappings for verification

2. **Document Transformation**
   - Intelligently modifies long context documents
   - Preserves semantic content while changing surface form
   - Handles multiple answer types (extractive, abstractive, multiple choice)
   - Maintains answer-document alignment through sophisticated tracking

Community and Support
====================

- **Documentation**: Comprehensive guides and API reference
- **Issues**: `GitHub Issues <https://github.com/nathanrice/scramblebench/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/nathanrice/scramblebench/discussions>`_
- **Contributing**: See :doc:`development/contributing`

License
=======

ScrambleBench is licensed under the MIT License. See :doc:`about/license` for details.

Citation
========

If you use ScrambleBench in your research, please cite:

.. code-block:: bibtex

   @software{scramblebench2024,
     title={ScrambleBench: Contamination-Resistant LLM Evaluation Through Constructed Languages},
     author={Rice, Nathan},
     year={2024},
     url={https://github.com/nathanrice/scramblebench}
   }