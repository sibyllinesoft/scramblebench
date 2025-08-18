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

**🚨 The Problem: Your LLM Evaluations Are Worthless**

Modern LLMs are trained on massive datasets that likely contain your evaluation benchmarks. When GPT-4 scores 95% on your "challenging" reasoning test, is it actually reasoning—or just reciting memorized answers? **You can't tell the difference.**

**💡 The Solution: ScrambleBench Makes Evaluation Honest Again**

**ScrambleBench** is the first toolkit that reveals what LLMs actually understand versus what they've memorized. By transforming benchmarks into contamination-resistant versions using constructed languages and document transformations, ScrambleBench exposes the gap between real intelligence and training data regurgitation.

**Why This Matters:**
- 📊 **Reliable Model Selection**: Choose models based on genuine capability, not memorization
- 🧠 **True Intelligence Assessment**: Distinguish reasoning from pattern matching  
- 🔬 **Reproducible Research**: Publish results that aren't invalidated by data contamination
- 💰 **ROI Protection**: Avoid deploying overrated models that fail in production

🚀 **Shocking Results: See How Models Really Perform**

**Real Example - Mathematical Reasoning:**
- **Original MATH Dataset**: GPT-4 scores 64% 
- **ScrambleBench Phonetic Transform**: GPT-4 drops to 31%
- **Difference**: 33 percentage points of pure memorization

**Translation keeps logic, eliminates memorization:**

.. code-block:: text

   Original Problem:
   "If x + 2y = 7 and 3x - y = 4, what is x?"

   ScrambleBench Transform (Phonetic):
   "Mf k + 2z = 7 anp 3k - z = 4, xhat ms k?"
   
   Logic preserved ✓ | Memorization eliminated ✓

**The result?** Models that seemed "intelligent" suddenly reveal they were just pattern matching training data.

🌟 **ScrambleBench Capabilities**
================================

⚡ **Instant Contamination Detection**
  Reveal memorization vs. reasoning in minutes, not months of investigation.

🧬 **Six Transformation Types**  
  From simple substitution to fully synthetic languages—find the perfect contamination resistance level.

📈 **100+ Model Support**
  Evaluate any LLM through OpenRouter integration with automatic rate limiting and error handling.

🔬 **Research-Grade Statistics**
  Statistical significance testing, confidence intervals, and publication-ready visualizations.

🏭 **Production-Ready Pipeline**
  YAML configuration, CLI automation, and batch processing for enterprise-scale evaluation.

📊 **Rich Analytics**
  Position bias analysis, coherence tracking, entity relationship preservation, and failure pattern detection.

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