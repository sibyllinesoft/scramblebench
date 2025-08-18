Translation Benchmarks Tutorial
===============================

This comprehensive tutorial demonstrates how to use ScrambleBench's translation benchmarks to evaluate LLMs in a contamination-resistant manner. Translation benchmarks transform existing problems into constructed languages while preserving logical structure and solvability.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

Translation benchmarks address the critical problem of training data contamination by creating novel versions of evaluation tasks. Instead of relying on the original benchmark text that may have been seen during training, ScrambleBench transforms the content into artificial languages that preserve the underlying logical structure while eliminating lexical overlap with training data.

**Key Benefits:**

* **Contamination Resistance**: Eliminates training data contamination issues
* **Logical Preservation**: Maintains problem solvability across transformations
* **Multi-Language Support**: Six different constructed language types
* **Scalable Complexity**: Adjustable difficulty levels (1-10)
* **Reproducible Results**: Deterministic generation with seed control

Understanding Constructed Languages
-----------------------------------

ScrambleBench supports six types of constructed languages, each designed to test different aspects of model robustness:

SUBSTITUTION Languages
~~~~~~~~~~~~~~~~~~~~~~

Simple character and word substitution ciphers that maintain readability while changing surface form.

**How it works:**
- Maps characters/words to consistent alternatives
- Preserves word boundaries and punctuation
- Maintains capitalization patterns
- Uses frequency-based mapping for realism

**Example:**

.. code-block:: text

   Original:  "What is the capital of France?"
   Scrambled: "Xhat ms the capmtal of Franne?"

**Best for:** Testing basic pattern recognition and consistency handling.

PHONETIC Languages
~~~~~~~~~~~~~~~~~~

Phonetically motivated transformations that follow realistic sound change patterns found in natural languages.

**How it works:**
- Applies systematic sound changes (e.g., /p/ → /b/, /s/ → /∫/)
- Follows phonotactic constraints
- Preserves syllable structure
- Uses linguistically plausible transformations

**Example:**

.. code-block:: text

   Original:  "The quick brown fox jumps"
   Phonetic:  "Ze qvig brown voks dhumps"

**Best for:** Testing phonological pattern learning and linguistic intuition.

SCRAMBLED Languages
~~~~~~~~~~~~~~~~~~~

Systematic character scrambling with consistent rules across the entire text.

**How it works:**
- Applies position-based character permutations
- Maintains consistency across transformations
- Preserves word length and structure
- Uses deterministic scrambling patterns

**Example:**

.. code-block:: text

   Original:  "Hello world"
   Scrambled: "Ehllo dlrow"

**Best for:** Testing pattern recognition across character-level transformations.

SYNTHETIC Languages
~~~~~~~~~~~~~~~~~~~

Fully artificial vocabulary and grammar systems with consistent internal logic.

**How it works:**
- Generates novel vocabulary with consistent phonotactics
- Creates artificial grammatical patterns
- Maintains semantic relationships
- Uses constructed morphological systems

**Example:**

.. code-block:: text

   Original:  "The cat sits on the mat"
   Synthetic: "Ko zim vuls na ko pel"

**Best for:** Testing abstract reasoning without lexical cues.

ENGLISH_LIKE Languages
~~~~~~~~~~~~~~~~~~~~~

Artificial languages that follow English phonotactic patterns while using novel vocabulary.

**How it works:**
- Maintains English sound patterns
- Uses plausible English syllable structures
- Preserves grammatical word order
- Creates recognizable but novel vocabulary

**Example:**

.. code-block:: text

   Original:     "Reading books improves knowledge"
   English-like: "Blading flooks imbroves tnowledge"

**Best for:** Testing reliance on English-specific patterns vs. general reasoning.

RANDOM_FREQUENCY Languages
~~~~~~~~~~~~~~~~~~~~~~~~~

Frequency-correlated word generation that maintains statistical properties of natural language.

**How it works:**
- Maps words based on frequency distributions
- Preserves high/medium/low frequency distinctions
- Maintains statistical language patterns
- Uses zipfian distribution matching

**Example:**

.. code-block:: text

   Original: "The quick brown fox"
   Freq-Map: "Zul blaft krone vex"
   # 'Zul' maps to high-freq 'the', 'blaft' to medium-freq 'quick', etc.

**Best for:** Testing statistical language modeling vs. logical reasoning.

Setting Up Translation Benchmarks
----------------------------------

Basic Setup
~~~~~~~~~~~

First, let's create a simple translation benchmark:

.. code-block:: python

   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageType
   import json

   # Create test dataset
   qa_data = [
       {"question": "What is 5 + 3?", "answer": "8"},
       {"question": "What color is snow?", "answer": "white"},
       {"question": "How many days are in a week?", "answer": "7"},
       {"question": "What is the capital of Japan?", "answer": "Tokyo"}
   ]

   # Save dataset
   with open("math_qa.json", "w") as f:
       json.dump(qa_data, f, indent=2)

   # Create translation benchmark
   benchmark = TranslationBenchmark(
       source_dataset="math_qa.json",
       language_type=LanguageType.SUBSTITUTION,
       language_complexity=5,
       seed=42  # For reproducible results
   )

   # Initialize model
   model = OpenRouterClient(
       model_name="openai/gpt-3.5-turbo",
       api_key="your-openrouter-key"
   )

   # Run evaluation
   results = benchmark.run(model, num_samples=10)
   print(f"Accuracy: {results.score:.2%}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated evaluations, use detailed configuration:

.. code-block:: python

   from scramblebench.core.config import BenchmarkConfig
   from scramblebench.translation.language_generator import LanguageConfig

   # Create language configuration
   lang_config = LanguageConfig(
       preserve_numbers=True,        # Keep numbers unchanged
       preserve_capitalization=True, # Maintain caps patterns
       preserve_punctuation=True,    # Keep punctuation marks
       min_word_length=2,           # Minimum word length for transformation
       transformation_probability=0.9 # 90% of words get transformed
   )

   # Create benchmark configuration
   benchmark_config = BenchmarkConfig(
       random_seed=42,
       evaluation_mode="fuzzy_match",  # Allow approximate matching
       evaluation_threshold=0.8,       # 80% similarity threshold
       max_retries=3,                  # Retry failed evaluations
       timeout=30                      # 30 second timeout per question
   )

   # Create advanced benchmark
   benchmark = TranslationBenchmark(
       source_dataset="complex_reasoning.json",
       language_type=LanguageType.PHONETIC,
       language_complexity=7,
       language_config=lang_config,
       benchmark_config=benchmark_config
   )

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The CLI provides powerful tools for benchmark creation and execution:

**Generate a Language:**

.. code-block:: bash

   # Create a substitution language
   scramblebench language generate my_substitution \
     --type substitution \
     --complexity 5 \
     --seed 42

   # Create a synthetic language with custom parameters
   scramblebench language generate my_synthetic \
     --type synthetic \
     --complexity 8 \
     --preserve-numbers \
     --preserve-capitalization

**Transform Text:**

.. code-block:: bash

   # Transform a single text string
   scramblebench transform text "What is machine learning?" my_substitution

   # Transform a file
   scramblebench transform file input.txt my_substitution --output transformed.txt

**Run Evaluation:**

.. code-block:: bash

   # Single model evaluation
   scramblebench evaluate run \
     --models "openai/gpt-4" \
     --benchmarks "data/math_questions.json" \
     --language-type substitution \
     --complexity 5 \
     --experiment-name "gpt4_math_sub"

   # Multi-model comparison
   scramblebench evaluate run \
     --models "openai/gpt-4,anthropic/claude-3-sonnet,meta-llama/llama-2-70b-chat" \
     --benchmarks "data/reasoning_tasks.json" \
     --language-types "substitution,phonetic,synthetic" \
     --complexities "3,5,7" \
     --experiment-name "multi_model_robustness" \
     --max-samples 100

Language Complexity Levels
---------------------------

ScrambleBench supports complexity levels 1-10, with increasing transformation sophistication:

**Level 1-3: Basic**
- Simple, predictable transformations
- High preservation of original structure
- Easy to reverse-engineer
- Good for initial testing

.. code-block:: python

   # Level 2 substitution example
   generator = LanguageGenerator(seed=42)
   lang = generator.generate_language("basic", LanguageType.SUBSTITUTION, complexity=2)
   
   original = "The cat sits on the mat"
   transformed = lang.transform(original)
   # Result: "The cat sits on the mat" (minimal changes)

**Level 4-6: Moderate**
- Noticeable but systematic changes
- Moderate obfuscation
- Consistent patterns emerge
- Suitable for most evaluations

.. code-block:: python

   # Level 5 phonetic example
   lang = generator.generate_language("moderate", LanguageType.PHONETIC, complexity=5)
   
   original = "Reading comprehension requires practice"
   transformed = lang.transform(original)
   # Result: "Zeading gombrehension requizes bractice"

**Level 7-8: Advanced**
- Significant transformations
- Strong pattern obfuscation
- Multiple transformation rules
- Challenging for models

.. code-block:: python

   # Level 7 synthetic example
   lang = generator.generate_language("advanced", LanguageType.SYNTHETIC, complexity=7)
   
   original = "Scientists study natural phenomena"
   transformed = lang.transform(original)
   # Result: "Glorthaks vexin nalted qomphenra"

**Level 9-10: Expert**
- Maximum transformation complexity
- Near-complete obfuscation
- Multiple overlapping rules
- Extreme stress testing

.. code-block:: python

   # Level 9 example with multiple transformations
   lang = generator.generate_language("expert", LanguageType.SYNTHETIC, complexity=9)
   
   original = "Artificial intelligence enables automation"
   transformed = lang.transform(original)
   # Result: "Qelthramic blorganeth vrixes blomathrek"

Practical Evaluation Workflows
-------------------------------

Research Paper Reproduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Reproducing benchmark results while controlling for contamination.

.. code-block:: python

   import json
   from pathlib import Path
   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.evaluation import ExperimentRunner

   # Load original benchmark (e.g., MMLU, HellaSwag, etc.)
   original_data = json.load(open("original_benchmark.json"))

   # Create contamination-resistant versions
   language_types = [LanguageType.SUBSTITUTION, LanguageType.PHONETIC, LanguageType.SYNTHETIC]
   complexity_levels = [3, 5, 7]

   results = {}
   
   for lang_type in language_types:
       for complexity in complexity_levels:
           benchmark = TranslationBenchmark(
               source_dataset="original_benchmark.json",
               language_type=lang_type,
               language_complexity=complexity,
               seed=42  # Reproducible results
           )
           
           # Test multiple models
           models = ["openai/gpt-4", "anthropic/claude-3-sonnet", "openai/gpt-3.5-turbo"]
           
           for model_name in models:
               model = OpenRouterClient(model_name)
               result = benchmark.run(model, num_samples=1000)
               
               key = f"{lang_type.name}_c{complexity}_{model_name.replace('/', '_')}"
               results[key] = {
                   'accuracy': result.score,
                   'language_type': lang_type.name,
                   'complexity': complexity,
                   'model': model_name,
                   'num_samples': len(result.predictions)
               }

   # Save comprehensive results
   with open("contamination_resistant_results.json", "w") as f:
       json.dump(results, f, indent=2)

Model Selection Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Evaluating multiple models for production deployment.

.. code-block:: bash

   #!/bin/bash
   # model_selection.sh

   MODELS=(
       "openai/gpt-4"
       "openai/gpt-3.5-turbo"
       "anthropic/claude-3-sonnet"
       "anthropic/claude-3-haiku"
       "meta-llama/llama-2-70b-chat"
       "mistralai/mistral-7b-instruct"
   )

   BENCHMARKS=(
       "data/math_reasoning.json"
       "data/reading_comprehension.json"
       "data/logical_inference.json"
   )

   # Run comprehensive evaluation
   for model in "${MODELS[@]}"; do
       for benchmark in "${BENCHMARKS[@]}"; do
           scramblebench evaluate run \
             --models "$model" \
             --benchmarks "$benchmark" \
             --language-types "substitution,phonetic,synthetic" \
             --complexities "3,5,7" \
             --experiment-name "model_selection_$(basename $benchmark .json)" \
             --max-samples 200 \
             --generate-plots \
             --save-predictions
       done
   done

   # Generate comparison report
   scramblebench evaluate compare model_selection_*

Robustness Testing
~~~~~~~~~~~~~~~~~~

**Scenario**: Testing model robustness across different transformation types.

.. code-block:: python

   from scramblebench.evaluation import RobustnessEvaluator
   from scramblebench.core.metrics import calculate_degradation

   # Set up robustness evaluation
   evaluator = RobustnessEvaluator(
       base_dataset="high_quality_qa.json",
       model_name="openai/gpt-4"
   )

   # Test across all language types
   robustness_results = {}

   for lang_type in LanguageType:
       # Test multiple complexity levels
       complexities = [1, 3, 5, 7, 9]
       type_results = []
       
       for complexity in complexities:
           result = evaluator.evaluate(
               language_type=lang_type,
               complexity=complexity,
               num_samples=100
           )
           
           type_results.append({
               'complexity': complexity,
               'accuracy': result.score,
               'avg_response_time': result.avg_response_time,
               'error_rate': result.error_rate
           })
       
       robustness_results[lang_type.name] = type_results

   # Calculate degradation metrics
   for lang_type, results in robustness_results.items():
       baseline_accuracy = results[0]['accuracy']  # Complexity 1
       
       for result in results[1:]:  # Complexity 3+
           degradation = calculate_degradation(baseline_accuracy, result['accuracy'])
           result['degradation'] = degradation
           print(f"{lang_type} Complexity {result['complexity']}: "
                 f"{degradation:.1%} degradation")

Analyzing Translation Results
-----------------------------

Understanding Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.core.reporter import Reporter
   from scramblebench.evaluation.plotting import create_robustness_plot

   # Load evaluation results
   reporter = Reporter()
   results = reporter.load_results("robustness_evaluation")

   # Analyze performance patterns
   performance_analysis = reporter.analyze_performance(results)

   print("Performance Analysis:")
   print(f"Baseline Accuracy: {performance_analysis['baseline_accuracy']:.2%}")
   print(f"Average Degradation: {performance_analysis['avg_degradation']:.2%}")
   print(f"Most Robust Language Type: {performance_analysis['most_robust_type']}")
   print(f"Least Robust Language Type: {performance_analysis['least_robust_type']}")

   # Identify failure patterns
   failure_analysis = reporter.analyze_failures(results)
   
   print("\nFailure Patterns:")
   for pattern, frequency in failure_analysis['common_patterns'].items():
       print(f"  {pattern}: {frequency:.1%} of failures")

   # Generate visualization
   fig = create_robustness_plot(
       results, 
       title="Model Robustness Across Language Types",
       include_confidence_intervals=True
   )
   fig.savefig("robustness_analysis.png", dpi=300, bbox_inches='tight')

Statistical Significance Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.core.metrics import statistical_significance_test
   from scipy import stats
   import numpy as np

   def compare_model_performance(results_a, results_b, alpha=0.05):
       """Compare two model performance distributions."""
       
       # Extract accuracy scores
       scores_a = [r.score for r in results_a.predictions]
       scores_b = [r.score for r in results_b.predictions]
       
       # Perform statistical tests
       t_stat, t_p_value = stats.ttest_ind(scores_a, scores_b)
       mannwhitney_stat, mw_p_value = stats.mannwhitneyu(scores_a, scores_b)
       
       # Effect size (Cohen's d)
       pooled_std = np.sqrt(((len(scores_a) - 1) * np.var(scores_a) + 
                            (len(scores_b) - 1) * np.var(scores_b)) / 
                           (len(scores_a) + len(scores_b) - 2))
       cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
       
       return {
           'significant_difference': t_p_value < alpha,
           't_statistic': t_stat,
           't_p_value': t_p_value,
           'mannwhitney_p': mw_p_value,
           'effect_size': cohens_d,
           'mean_difference': np.mean(scores_a) - np.mean(scores_b)
       }

   # Example usage
   baseline_results = benchmark_original.run(model, num_samples=100)
   transformed_results = benchmark_transformed.run(model, num_samples=100)

   comparison = compare_model_performance(baseline_results, transformed_results)
   print(f"Statistically significant difference: {comparison['significant_difference']}")
   print(f"Effect size (Cohen's d): {comparison['effect_size']:.3f}")

Best Practices
--------------

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

**1. Quality Control**

.. code-block:: python

   def validate_dataset(dataset_path):
       """Validate dataset format and content quality."""
       data = json.load(open(dataset_path))
       
       issues = []
       
       for i, item in enumerate(data):
           # Check required fields
           if 'question' not in item or 'answer' not in item:
               issues.append(f"Item {i}: Missing required fields")
           
           # Check content quality
           if len(item.get('question', '')) < 10:
               issues.append(f"Item {i}: Question too short")
           
           if len(item.get('answer', '')) < 1:
               issues.append(f"Item {i}: Empty answer")
       
       return issues

**2. Balanced Sampling**

.. code-block:: python

   def create_balanced_sample(dataset, categories, samples_per_category=50):
       """Create balanced samples across categories."""
       balanced_data = []
       
       for category in categories:
           category_items = [item for item in dataset if item.get('category') == category]
           if len(category_items) >= samples_per_category:
               sampled = random.sample(category_items, samples_per_category)
               balanced_data.extend(sampled)
           else:
               print(f"Warning: Only {len(category_items)} items in category '{category}'")
               balanced_data.extend(category_items)
       
       return balanced_data

Experiment Design
~~~~~~~~~~~~~~~~~

**1. Controlled Variables**

.. code-block:: python

   # Always use the same seed for reproducibility
   RANDOM_SEED = 42

   # Define experimental conditions
   EXPERIMENTAL_CONDITIONS = {
       'language_types': [LanguageType.SUBSTITUTION, LanguageType.PHONETIC],
       'complexities': [3, 5, 7],
       'sample_sizes': [50, 100, 200],
       'models': ['openai/gpt-4', 'anthropic/claude-3-sonnet']
   }

   # Run controlled experiments
   results = {}
   
   for condition in itertools.product(*EXPERIMENTAL_CONDITIONS.values()):
       lang_type, complexity, sample_size, model_name = condition
       
       # Create benchmark with controlled parameters
       benchmark = TranslationBenchmark(
           source_dataset="evaluation_set.json",
           language_type=lang_type,
           language_complexity=complexity,
           seed=RANDOM_SEED  # Consistent seed
       )
       
       model = OpenRouterClient(model_name)
       result = benchmark.run(model, num_samples=sample_size)
       
       # Store results with condition metadata
       condition_key = f"{lang_type.name}_c{complexity}_n{sample_size}_{model_name}"
       results[condition_key] = {
           'result': result,
           'conditions': {
               'language_type': lang_type.name,
               'complexity': complexity,
               'sample_size': sample_size,
               'model': model_name
           }
       }

**2. Multiple Runs for Reliability**

.. code-block:: python

   def run_multiple_experiments(benchmark, model, num_runs=5):
       """Run multiple experiments to assess reliability."""
       all_results = []
       
       for run in range(num_runs):
           # Use different seeds for each run
           benchmark.seed = RANDOM_SEED + run
           result = benchmark.run(model, num_samples=100)
           all_results.append(result.score)
       
       return {
           'mean_accuracy': np.mean(all_results),
           'std_accuracy': np.std(all_results),
           'confidence_interval': stats.t.interval(
               0.95, len(all_results)-1,
               loc=np.mean(all_results),
               scale=stats.sem(all_results)
           ),
           'individual_runs': all_results
       }

Error Handling and Debugging
----------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. API Rate Limiting**

.. code-block:: python

   from scramblebench.llm import RateLimitError
   import time

   def evaluate_with_retry(benchmark, model, max_retries=3):
       """Evaluate with automatic retry on rate limit errors."""
       
       for attempt in range(max_retries):
           try:
               return benchmark.run(model)
           except RateLimitError as e:
               if attempt < max_retries - 1:
                   wait_time = 2 ** attempt * 60  # Exponential backoff
                   print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                   time.sleep(wait_time)
               else:
                   raise e

**2. Memory Issues with Large Datasets**

.. code-block:: python

   def evaluate_in_batches(benchmark, model, total_samples, batch_size=50):
       """Evaluate large datasets in batches to manage memory."""
       
       all_results = []
       num_batches = (total_samples + batch_size - 1) // batch_size
       
       for batch_idx in range(num_batches):
           start_idx = batch_idx * batch_size
           end_idx = min(start_idx + batch_size, total_samples)
           batch_samples = end_idx - start_idx
           
           print(f"Processing batch {batch_idx + 1}/{num_batches} "
                 f"({batch_samples} samples)")
           
           batch_result = benchmark.run(model, num_samples=batch_samples)
           all_results.extend(batch_result.predictions)
           
           # Optional: garbage collection
           import gc
           gc.collect()
       
       # Combine results
       return combine_batch_results(all_results)

**3. Debugging Language Transformations**

.. code-block:: python

   def debug_transformation(language, text_samples):
       """Debug language transformation issues."""
       
       print(f"Language: {language.name}")
       print(f"Type: {language.language_type}")
       print(f"Complexity: {language.complexity}")
       print(f"Rules: {len(language.transformation_rules)}")
       
       for i, text in enumerate(text_samples):
           transformed = language.transform(text)
           print(f"\nSample {i + 1}:")
           print(f"  Original:    {text}")
           print(f"  Transformed: {transformed}")
           
           # Check for issues
           if len(transformed) == 0:
               print("  WARNING: Empty transformation")
           
           if text == transformed:
               print("  WARNING: No transformation applied")
           
           # Show character-level changes
           changes = []
           for orig_char, trans_char in zip(text, transformed):
               if orig_char != trans_char:
                   changes.append(f"'{orig_char}' → '{trans_char}'")
           
           if changes:
               print(f"  Changes: {', '.join(changes[:5])}")  # Show first 5

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Caching Transformations**

.. code-block:: python

   from functools import lru_cache
   from scramblebench.core.cache import TransformationCache

   # Use built-in caching
   cache = TransformationCache(max_size=10000)
   
   def cached_transform(language, text):
       """Transform text with caching."""
       cache_key = f"{language.name}_{hash(text)}"
       
       if cache_key in cache:
           return cache[cache_key]
       
       transformed = language.transform(text)
       cache[cache_key] = transformed
       return transformed

**2. Parallel Processing**

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor, as_completed
   from scramblebench.core.parallel import ParallelEvaluator

   def parallel_evaluation(benchmarks, models, max_workers=4):
       """Run multiple evaluations in parallel."""
       
       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           # Submit all evaluation tasks
           futures = {}
           
           for benchmark in benchmarks:
               for model in models:
                   future = executor.submit(benchmark.run, model)
                   futures[future] = (benchmark.name, model.name)
           
           # Collect results as they complete
           results = {}
           for future in as_completed(futures):
               benchmark_name, model_name = futures[future]
               try:
                   result = future.result()
                   results[f"{benchmark_name}_{model_name}"] = result
                   print(f"Completed: {benchmark_name} with {model_name}")
               except Exception as e:
                   print(f"Error in {benchmark_name} with {model_name}: {e}")
           
           return results

Advanced Topics
---------------

Custom Language Types
~~~~~~~~~~~~~~~~~~~~~~

Create your own language transformation logic:

.. code-block:: python

   from scramblebench.translation.language_generator import BaseLanguage
   from scramblebench.core.types import TransformationRule

   class PigLatinLanguage(BaseLanguage):
       """Custom Pig Latin transformation."""
       
       def __init__(self, name, complexity=1):
           super().__init__(name, "PIG_LATIN", complexity)
           self.vowels = set('aeiouAEIOU')
       
       def transform_word(self, word):
           """Transform a single word to Pig Latin."""
           if not word.isalpha():
               return word
           
           if word[0] in self.vowels:
               return word + 'way'
           else:
               # Find first vowel
               for i, char in enumerate(word):
                   if char in self.vowels:
                       return word[i:] + word[:i] + 'ay'
               return word + 'ay'  # No vowels found
       
       def transform(self, text):
           """Transform entire text."""
           words = text.split()
           transformed_words = [self.transform_word(word) for word in words]
           return ' '.join(transformed_words)

   # Use custom language
   pig_latin = PigLatinLanguage("pig_latin", complexity=2)
   benchmark = TranslationBenchmark(
       source_dataset="test_data.json",
       custom_language=pig_latin
   )

Integration with Research Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Academic Paper Pipeline**

.. code-block:: python

   def research_evaluation_pipeline(
       paper_title,
       benchmark_datasets,
       models_to_test,
       output_dir="research_results"
   ):
       """Complete pipeline for academic research evaluation."""
       
       from pathlib import Path
       import matplotlib.pyplot as plt
       
       # Create output directory
       output_path = Path(output_dir) / paper_title.replace(" ", "_")
       output_path.mkdir(parents=True, exist_ok=True)
       
       # Store all results
       all_results = {}
       
       # Run evaluations
       for dataset_name, dataset_path in benchmark_datasets.items():
           dataset_results = {}
           
           for model_name in models_to_test:
               # Create multiple benchmark variants
               variants = [
                   ("original", None),
                   ("substitution", LanguageType.SUBSTITUTION),
                   ("phonetic", LanguageType.PHONETIC),
                   ("synthetic", LanguageType.SYNTHETIC)
               ]
               
               for variant_name, lang_type in variants:
                   if lang_type is None:
                       # Original benchmark (baseline)
                       result = evaluate_original_benchmark(dataset_path, model_name)
                   else:
                       # Transformed benchmark
                       benchmark = TranslationBenchmark(
                           source_dataset=dataset_path,
                           language_type=lang_type,
                           language_complexity=5
                       )
                       model = OpenRouterClient(model_name)
                       result = benchmark.run(model, num_samples=500)
                   
                   dataset_results[f"{model_name}_{variant_name}"] = result
           
           all_results[dataset_name] = dataset_results
       
       # Generate research report
       generate_research_report(all_results, output_path)
       
       # Create publication-ready figures
       create_publication_figures(all_results, output_path)
       
       return all_results

Next Steps
----------

Now that you understand translation benchmarks:

1. **Explore Language Types**: Experiment with different language types to understand their characteristics
2. **Scale Up Evaluations**: Use the CLI for large-scale evaluations across multiple models
3. **Advanced Analysis**: Implement statistical analysis and visualization for your results  
4. **Custom Applications**: Adapt the techniques to your specific research or evaluation needs

**Related Documentation:**

* :doc:`long_context_benchmarks` - Document transformation techniques
* :doc:`custom_models` - Integrating your own models
* :doc:`../user_guide/evaluation_pipeline` - Comprehensive evaluation workflows
* :doc:`../examples/configuration_examples` - Advanced configuration patterns

**Community Resources:**

* GitHub Issues: Report bugs or request features
* GitHub Discussions: Share your evaluation results and get help
* Research Papers: See academic applications of ScrambleBench techniques