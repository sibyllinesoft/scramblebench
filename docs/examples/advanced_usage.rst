Advanced Usage Examples
=======================

This section demonstrates sophisticated ScrambleBench usage patterns for research-grade evaluation pipelines, complex transformation chains, and production-scale benchmarking workflows.

.. contents:: Table of Contents
   :local:
   :depth: 2

Complex Evaluation Pipelines
-----------------------------

Multi-Model Contamination Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive contamination analysis across multiple models with statistical significance testing:

.. code-block:: python

   import asyncio
   from typing import Dict, List, Tuple
   from pathlib import Path
   
   from scramblebench import TranslationBenchmark, LongContextBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
   from scramblebench.evaluation.runner import EvaluationRunner
   from scramblebench.evaluation.metrics import StatisticalAnalyzer
   from scramblebench.core.reporter import ContaminationReport
   
   class ContaminationAnalysisPipeline:
       """Advanced pipeline for detecting training data contamination."""
       
       def __init__(self, models: List[str], datasets: List[str]):
           self.models = models
           self.datasets = datasets
           self.language_generator = LanguageGenerator(seed=42)
           self.results: Dict[str, Dict] = {}
           
       async def run_comprehensive_analysis(
           self, 
           complexity_levels: List[int] = [3, 5, 7],
           sample_sizes: List[int] = [50, 100, 200],
           confidence_level: float = 0.95
       ) -> ContaminationReport:
           """Run contamination analysis across complexity and sample sizes."""
           
           # Generate transformation languages of varying complexity
           languages = {}
           for complexity in complexity_levels:
               for lang_type in [LanguageType.PHONETIC, LanguageType.SYNTHETIC, LanguageType.SCRAMBLED]:
                   lang_name = f"{lang_type.value}_complexity_{complexity}"
                   languages[lang_name] = self.language_generator.generate_language(
                       name=lang_name,
                       language_type=lang_type,
                       complexity=complexity,
                       vocab_size=2000
                   )
           
           # Run evaluations across all combinations
           for model_name in self.models:
               model = OpenRouterClient(model_name=model_name, api_key="your-key")
               self.results[model_name] = {}
               
               for dataset_path in self.datasets:
                   dataset_results = {}
                   
                   # Baseline evaluation (no transformation)
                   baseline_benchmark = TranslationBenchmark(
                       source_dataset=dataset_path,
                       use_transformation=False
                   )
                   baseline_result = await baseline_benchmark.run_async(
                       model, num_samples=max(sample_sizes)
                   )
                   dataset_results['baseline'] = baseline_result
                   
                   # Transformed evaluations
                   for lang_name, language in languages.items():
                       for sample_size in sample_sizes:
                           transform_benchmark = TranslationBenchmark(
                               source_dataset=dataset_path,
                               constructed_language=language,
                               preserve_structure=True,
                               preserve_entities=True
                           )
                           
                           transform_result = await transform_benchmark.run_async(
                               model, num_samples=sample_size
                           )
                           
                           key = f"{lang_name}_samples_{sample_size}"
                           dataset_results[key] = transform_result
                   
                   self.results[model_name][dataset_path] = dataset_results
           
           # Statistical analysis
           analyzer = StatisticalAnalyzer(confidence_level=confidence_level)
           contamination_scores = analyzer.compute_contamination_scores(self.results)
           significance_tests = analyzer.run_significance_tests(self.results)
           
           # Generate comprehensive report
           report = ContaminationReport(
               results=self.results,
               contamination_scores=contamination_scores,
               significance_tests=significance_tests,
               metadata={
                   'complexity_levels': complexity_levels,
                   'sample_sizes': sample_sizes,
                   'confidence_level': confidence_level,
                   'languages_used': list(languages.keys())
               }
           )
           
           return report

Hierarchical Benchmark Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organize benchmarks by cognitive difficulty and domain specificity:

.. code-block:: python

   from dataclasses import dataclass
   from enum import Enum
   
   class CognitiveDifficulty(Enum):
       BASIC = "basic"           # Simple pattern matching
       INTERMEDIATE = "intermediate"  # Multi-step reasoning
       ADVANCED = "advanced"     # Complex analytical reasoning
       EXPERT = "expert"         # Research-level problems
   
   class CognitiveDomain(Enum):
       LOGICAL = "logical"
       MATHEMATICAL = "mathematical"
       LINGUISTIC = "linguistic"
       SPATIAL = "spatial"
       TEMPORAL = "temporal"
   
   @dataclass
   class BenchmarkSpec:
       name: str
       difficulty: CognitiveDifficulty
       domain: CognitiveDomain
       dataset_path: str
       expected_baseline_score: float
       transformation_robustness_threshold: float
   
   class HierarchicalBenchmarkSuite:
       """Organize benchmarks by cognitive demands and evaluate systematically."""
       
       def __init__(self):
           self.benchmark_specs = [
               # Logical reasoning benchmarks
               BenchmarkSpec("propositional_logic", CognitiveDifficulty.BASIC, 
                           CognitiveDomain.LOGICAL, "data/logic/propositional.json", 0.85, 0.15),
               BenchmarkSpec("predicate_logic", CognitiveDifficulty.INTERMEDIATE,
                           CognitiveDomain.LOGICAL, "data/logic/predicate.json", 0.70, 0.25),
               BenchmarkSpec("modal_logic", CognitiveDifficulty.ADVANCED,
                           CognitiveDomain.LOGICAL, "data/logic/modal.json", 0.55, 0.35),
               
               # Mathematical reasoning benchmarks  
               BenchmarkSpec("arithmetic_word_problems", CognitiveDifficulty.BASIC,
                           CognitiveDomain.MATHEMATICAL, "data/math/arithmetic.json", 0.90, 0.10),
               BenchmarkSpec("algebraic_reasoning", CognitiveDifficulty.INTERMEDIATE,
                           CognitiveDomain.MATHEMATICAL, "data/math/algebra.json", 0.75, 0.20),
               BenchmarkSpec("geometric_proofs", CognitiveDifficulty.ADVANCED,
                           CognitiveDomain.MATHEMATICAL, "data/math/geometry.json", 0.45, 0.40),
               
               # Add more benchmark specifications...
           ]
           
       async def run_hierarchical_evaluation(
           self, 
           models: List[str], 
           complexity_progression: List[int] = [3, 5, 7, 9]
       ) -> Dict[str, Dict]:
           """Run evaluation with increasing transformation complexity."""
           
           results = {}
           
           for model_name in models:
               model = OpenRouterClient(model_name=model_name, api_key="your-key")
               model_results = {}
               
               for spec in self.benchmark_specs:
                   # Create progression of transformation complexity
                   spec_results = {'baseline': None, 'transformations': {}}
                   
                   # Baseline evaluation
                   baseline_benchmark = TranslationBenchmark(
                       source_dataset=spec.dataset_path,
                       use_transformation=False
                   )
                   baseline_result = await baseline_benchmark.run_async(model, num_samples=100)
                   spec_results['baseline'] = baseline_result
                   
                   # Progression of transformation complexity
                   for complexity in complexity_progression:
                       # Choose transformation type based on domain
                       if spec.domain == CognitiveDomain.LOGICAL:
                           lang_type = LanguageType.SYNTHETIC
                       elif spec.domain == CognitiveDomain.MATHEMATICAL:
                           lang_type = LanguageType.SUBSTITUTION
                       elif spec.domain == CognitiveDomain.LINGUISTIC:
                           lang_type = LanguageType.PHONETIC
                       else:
                           lang_type = LanguageType.SCRAMBLED
                       
                       language = self.language_generator.generate_language(
                           name=f"{spec.name}_complexity_{complexity}",
                           language_type=lang_type,
                           complexity=complexity,
                           vocab_size=1000
                       )
                       
                       transform_benchmark = TranslationBenchmark(
                           source_dataset=spec.dataset_path,
                           constructed_language=language,
                           preserve_structure=True
                       )
                       
                       transform_result = await transform_benchmark.run_async(
                           model, num_samples=100
                       )
                       spec_results['transformations'][complexity] = transform_result
                   
                   model_results[spec.name] = spec_results
               
               results[model_name] = model_results
           
           return results

Custom Transformation Chains
-----------------------------

Multi-Stage Document Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex document processing with preservation of semantic relationships:

.. code-block:: python

   from scramblebench.longcontext.document_transformer import DocumentTransformer
   from scramblebench.longcontext.qa_transformer import QATransformer
   from scramblebench.translation.text_transformer import TextTransformer
   
   class SemanticPreservingTransformer:
       """Multi-stage transformer preserving semantic relationships."""
       
       def __init__(self, complexity: int = 6):
           self.complexity = complexity
           self.entity_tracker = EntityRelationshipTracker()
           self.coherence_validator = CoherenceValidator()
           
       def transform_document_with_qa(
           self, 
           document: str, 
           qa_pairs: List[Dict],
           preserve_entities: bool = True,
           preserve_numerical_relationships: bool = True
       ) -> Tuple[str, List[Dict]]:
           """Transform document and QA pairs while preserving relationships."""
           
           # Stage 1: Extract and catalog entities and relationships
           if preserve_entities:
               entities = self.entity_tracker.extract_entities(document)
               relationships = self.entity_tracker.extract_relationships(document, qa_pairs)
           
           # Stage 2: Create transformation language with entity preservation
           language_generator = LanguageGenerator(seed=42)
           
           if preserve_entities:
               # Generate language with entity constraints
               preserved_words = [entity['text'] for entity in entities if entity['preserve']]
               language = language_generator.generate_language(
                   name="entity_preserving",
                   language_type=LanguageType.PHONETIC,
                   complexity=self.complexity,
                   vocab_size=2000
               )
               # Add entity preservation rules
               for word in preserved_words:
                   language.vocabulary[word] = word  # Preserve as-is
           else:
               language = language_generator.generate_language(
                   name="full_transform",
                   language_type=LanguageType.SYNTHETIC,
                   complexity=self.complexity,
                   vocab_size=2000
               )
           
           # Stage 3: Transform document with relationship tracking
           doc_transformer = DocumentTransformer(
               constructed_language=language,
               preserve_structure=True,
               track_transformations=True
           )
           
           transformed_document, transformation_map = doc_transformer.transform_with_mapping(document)
           
           # Stage 4: Transform QA pairs with answer alignment
           qa_transformer = QATransformer(
               constructed_language=language,
               transformation_map=transformation_map,
               preserve_answer_spans=True
           )
           
           transformed_qa_pairs = []
           for qa_pair in qa_pairs:
               transformed_qa = qa_transformer.transform_qa_pair(
                   qa_pair, 
                   transformed_document,
                   preserve_numerical_relationships=preserve_numerical_relationships
               )
               transformed_qa_pairs.append(transformed_qa)
           
           # Stage 5: Validate coherence and relationships
           if preserve_entities:
               coherence_score = self.coherence_validator.validate_entity_relationships(
                   original_doc=document,
                   transformed_doc=transformed_document,
                   relationships=relationships
               )
               
               if coherence_score < 0.8:
                   # Retry with higher preservation
                   return self._retry_with_higher_preservation(
                       document, qa_pairs, entities, relationships
                   )
           
           return transformed_document, transformed_qa_pairs

Adaptive Complexity Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic complexity adjustment based on model performance:

.. code-block:: python

   class AdaptiveComplexityTransformer:
       """Automatically adjust transformation complexity based on performance."""
       
       def __init__(self, target_difficulty: float = 0.7):
           self.target_difficulty = target_difficulty  # Target accuracy drop
           self.complexity_history = []
           
       async def find_optimal_complexity(
           self,
           model: ModelInterface,
           dataset_path: str,
           initial_complexity: int = 5,
           max_iterations: int = 10
       ) -> Tuple[int, ConstructedLanguage]:
           """Binary search for optimal transformation complexity."""
           
           # Get baseline performance
           baseline_benchmark = TranslationBenchmark(
               source_dataset=dataset_path,
               use_transformation=False
           )
           baseline_result = await baseline_benchmark.run_async(model, num_samples=50)
           baseline_accuracy = baseline_result.score
           target_accuracy = baseline_accuracy * self.target_difficulty
           
           # Binary search for optimal complexity
           min_complexity, max_complexity = 1, 10
           current_complexity = initial_complexity
           best_language = None
           
           for iteration in range(max_iterations):
               # Generate language at current complexity
               language = LanguageGenerator(seed=42).generate_language(
                   name=f"adaptive_complexity_{current_complexity}",
                   language_type=LanguageType.PHONETIC,
                   complexity=current_complexity,
                   vocab_size=1500
               )
               
               # Test performance
               transform_benchmark = TranslationBenchmark(
                   source_dataset=dataset_path,
                   constructed_language=language,
                   preserve_structure=True
               )
               transform_result = await transform_benchmark.run_async(model, num_samples=50)
               transform_accuracy = transform_result.score
               
               self.complexity_history.append({
                   'complexity': current_complexity,
                   'accuracy': transform_accuracy,
                   'accuracy_drop': baseline_accuracy - transform_accuracy
               })
               
               # Check if we're close to target
               accuracy_diff = abs(transform_accuracy - target_accuracy)
               if accuracy_diff < 0.05:  # Within 5% of target
                   best_language = language
                   break
               
               # Adjust complexity bounds
               if transform_accuracy > target_accuracy:
                   # Need more difficulty
                   min_complexity = current_complexity
                   current_complexity = (current_complexity + max_complexity) // 2
               else:
                   # Too difficult
                   max_complexity = current_complexity  
                   current_complexity = (min_complexity + current_complexity) // 2
               
               best_language = language
           
           return current_complexity, best_language

Advanced Model Configuration
----------------------------

Custom Model Adapters
~~~~~~~~~~~~~~~~~~~~~~

Implement custom model interfaces for proprietary or local models:

.. code-block:: python

   from scramblebench.llm.model_interface import ModelInterface
   from typing import Optional, Dict, Any
   import requests
   import time
   
   class CustomLocalModelAdapter(ModelInterface):
       """Adapter for local model servers or proprietary APIs."""
       
       def __init__(
           self,
           endpoint_url: str,
           model_name: str,
           api_key: Optional[str] = None,
           request_timeout: int = 60,
           retry_attempts: int = 3,
           custom_headers: Optional[Dict[str, str]] = None
       ):
           self.endpoint_url = endpoint_url
           self.model_name = model_name
           self.api_key = api_key
           self.request_timeout = request_timeout
           self.retry_attempts = retry_attempts
           self.custom_headers = custom_headers or {}
           
           # Performance tracking
           self.request_history = []
           self.error_count = 0
           
       async def generate_response(
           self,
           prompt: str,
           temperature: float = 0.0,
           max_tokens: Optional[int] = None,
           **kwargs
       ) -> str:
           """Generate response from custom model endpoint."""
           
           headers = {
               "Content-Type": "application/json",
               **self.custom_headers
           }
           
           if self.api_key:
               headers["Authorization"] = f"Bearer {self.api_key}"
           
           payload = {
               "model": self.model_name,
               "prompt": prompt,
               "temperature": temperature,
               "max_tokens": max_tokens or 1000,
               **kwargs
           }
           
           # Implement retry logic with exponential backoff
           for attempt in range(self.retry_attempts):
               try:
                   start_time = time.time()
                   
                   response = requests.post(
                       self.endpoint_url,
                       headers=headers,
                       json=payload,
                       timeout=self.request_timeout
                   )
                   
                   response_time = time.time() - start_time
                   
                   if response.status_code == 200:
                       result = response.json()
                       generated_text = self._extract_text_from_response(result)
                       
                       # Track performance metrics
                       self.request_history.append({
                           'timestamp': time.time(),
                           'response_time': response_time,
                           'prompt_length': len(prompt),
                           'response_length': len(generated_text),
                           'success': True
                       })
                       
                       return generated_text
                   
                   else:
                       self.error_count += 1
                       if attempt == self.retry_attempts - 1:
                           raise RuntimeError(f"Request failed: {response.status_code}")
                       
                       # Exponential backoff
                       wait_time = (2 ** attempt) * 1.0
                       await asyncio.sleep(wait_time)
                       
               except requests.exceptions.RequestException as e:
                   self.error_count += 1
                   if attempt == self.retry_attempts - 1:
                       raise RuntimeError(f"Connection error: {e}")
                   
                   wait_time = (2 ** attempt) * 1.0
                   await asyncio.sleep(wait_time)
           
           raise RuntimeError("Max retry attempts exceeded")
       
       def _extract_text_from_response(self, response_data: Dict[str, Any]) -> str:
           """Extract text from model response based on API format."""
           # Customize based on your model's response format
           if 'choices' in response_data:
               return response_data['choices'][0]['text']
           elif 'text' in response_data:
               return response_data['text']
           elif 'response' in response_data:
               return response_data['response']
           else:
               raise ValueError(f"Unexpected response format: {response_data}")
       
       def get_performance_metrics(self) -> Dict[str, float]:
           """Get performance statistics for this model adapter."""
           if not self.request_history:
               return {}
           
           response_times = [req['response_time'] for req in self.request_history]
           
           return {
               'average_response_time': sum(response_times) / len(response_times),
               'total_requests': len(self.request_history),
               'error_rate': self.error_count / (len(self.request_history) + self.error_count),
               'requests_per_minute': len(self.request_history) / 
                   ((time.time() - self.request_history[0]['timestamp']) / 60),
           }

Multi-Provider Model Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate results from multiple model providers for robust evaluation:

.. code-block:: python

   class ModelEnsemble:
       """Ensemble of models from different providers for robust evaluation."""
       
       def __init__(self, model_configs: List[Dict[str, Any]]):
           self.models = []
           for config in model_configs:
               if config['provider'] == 'openrouter':
                   model = OpenRouterClient(
                       model_name=config['model_name'],
                       api_key=config['api_key']
                   )
               elif config['provider'] == 'custom':
                   model = CustomLocalModelAdapter(
                       endpoint_url=config['endpoint'],
                       model_name=config['model_name'],
                       api_key=config.get('api_key')
                   )
               else:
                   raise ValueError(f"Unknown provider: {config['provider']}")
               
               self.models.append({
                   'model': model,
                   'name': config['name'],
                   'weight': config.get('weight', 1.0),
                   'provider': config['provider']
               })
       
       async def evaluate_ensemble(
           self,
           benchmark: TranslationBenchmark,
           num_samples: int = 100,
           aggregation_method: str = 'weighted_average'
       ) -> Dict[str, Any]:
           """Evaluate all models in ensemble and aggregate results."""
           
           individual_results = {}
           
           # Run evaluation on each model
           for model_config in self.models:
               model = model_config['model']
               model_name = model_config['name']
               
               try:
                   result = await benchmark.run_async(model, num_samples=num_samples)
                   individual_results[model_name] = {
                       'score': result.score,
                       'detailed_metrics': result.detailed_metrics,
                       'weight': model_config['weight'],
                       'provider': model_config['provider'],
                       'success': True
                   }
               except Exception as e:
                   individual_results[model_name] = {
                       'error': str(e),
                       'weight': model_config['weight'],
                       'provider': model_config['provider'],
                       'success': False
                   }
           
           # Aggregate results
           if aggregation_method == 'weighted_average':
               ensemble_score = self._weighted_average_aggregation(individual_results)
           elif aggregation_method == 'median':
               ensemble_score = self._median_aggregation(individual_results)
           elif aggregation_method == 'consensus':
               ensemble_score = self._consensus_aggregation(individual_results)
           else:
               raise ValueError(f"Unknown aggregation method: {aggregation_method}")
           
           return {
               'ensemble_score': ensemble_score,
               'individual_results': individual_results,
               'aggregation_method': aggregation_method,
               'successful_models': sum(1 for r in individual_results.values() if r['success']),
               'total_models': len(individual_results)
           }

Statistical Analysis Techniques
-------------------------------

Significance Testing for Contamination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement statistical tests to validate contamination detection:

.. code-block:: python

   import numpy as np
   from scipy import stats
   from typing import List, Tuple, Dict
   
   class ContaminationStatistics:
       """Statistical analysis for contamination detection."""
       
       def __init__(self, alpha: float = 0.05):
           self.alpha = alpha  # Significance level
           
       def paired_t_test(
           self,
           baseline_scores: List[float],
           transformed_scores: List[float]
       ) -> Dict[str, float]:
           """Paired t-test comparing baseline vs transformed performance."""
           
           if len(baseline_scores) != len(transformed_scores):
               raise ValueError("Score lists must have same length")
           
           # Calculate differences
           differences = np.array(baseline_scores) - np.array(transformed_scores)
           
           # Paired t-test
           t_statistic, p_value = stats.ttest_rel(baseline_scores, transformed_scores)
           
           # Effect size (Cohen's d for paired samples)
           mean_diff = np.mean(differences)
           std_diff = np.std(differences, ddof=1)
           cohens_d = mean_diff / std_diff if std_diff > 0 else 0
           
           # Confidence interval for mean difference
           std_error = std_diff / np.sqrt(len(differences))
           t_critical = stats.t.ppf(1 - self.alpha/2, len(differences) - 1)
           ci_lower = mean_diff - t_critical * std_error
           ci_upper = mean_diff + t_critical * std_error
           
           return {
               'mean_difference': mean_diff,
               't_statistic': t_statistic,
               'p_value': p_value,
               'cohens_d': cohens_d,
               'significant': p_value < self.alpha,
               'confidence_interval': (ci_lower, ci_upper),
               'degrees_freedom': len(differences) - 1
           }
       
       def contamination_severity_classification(
           self,
           performance_drop: float,
           effect_size: float,
           p_value: float
       ) -> Dict[str, Any]:
           """Classify contamination severity based on statistical measures."""
           
           # Classification criteria
           if p_value >= self.alpha:
               severity = "No Evidence"
               confidence = "Low"
           elif performance_drop < 0.05:  # Less than 5% drop
               severity = "Minimal"
               confidence = "Medium" if effect_size > 0.2 else "Low"
           elif performance_drop < 0.15:  # 5-15% drop
               severity = "Moderate"  
               confidence = "High" if effect_size > 0.5 else "Medium"
           elif performance_drop < 0.30:  # 15-30% drop
               severity = "Substantial"
               confidence = "High"
           else:  # > 30% drop
               severity = "Severe"
               confidence = "Very High"
           
           return {
               'severity': severity,
               'confidence': confidence,
               'performance_drop': performance_drop,
               'effect_size': effect_size,
               'p_value': p_value,
               'interpretation': self._get_interpretation(severity, confidence)
           }
       
       def _get_interpretation(self, severity: str, confidence: str) -> str:
           """Get human-readable interpretation of contamination analysis."""
           interpretations = {
               ("No Evidence", "Low"): "No statistical evidence of contamination detected.",
               ("Minimal", "Low"): "Possible minimal contamination, but evidence is weak.",
               ("Minimal", "Medium"): "Likely minimal contamination with moderate confidence.",
               ("Moderate", "Medium"): "Moderate contamination likely present.",
               ("Moderate", "High"): "Moderate contamination detected with high confidence.",
               ("Substantial", "High"): "Substantial contamination detected - model likely memorized significant portions.",
               ("Severe", "High"): "Severe contamination - model performance heavily dependent on memorization.",
               ("Severe", "Very High"): "Severe contamination with very high confidence - evaluation may be invalid."
           }
           
           return interpretations.get((severity, confidence), "Contamination assessment inconclusive.")

Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate robust confidence intervals for performance metrics:

.. code-block:: python

   class BootstrapAnalyzer:
       """Bootstrap analysis for robust confidence intervals."""
       
       def __init__(self, n_bootstrap: int = 10000, random_state: int = 42):
           self.n_bootstrap = n_bootstrap
           self.rng = np.random.RandomState(random_state)
           
       def bootstrap_confidence_interval(
           self,
           data: List[float],
           statistic_func: callable = np.mean,
           confidence_level: float = 0.95
       ) -> Tuple[float, float, float]:
           """Generate bootstrap confidence interval for any statistic."""
           
           data_array = np.array(data)
           n_samples = len(data_array)
           
           # Generate bootstrap samples
           bootstrap_statistics = []
           for _ in range(self.n_bootstrap):
               bootstrap_sample = self.rng.choice(data_array, size=n_samples, replace=True)
               bootstrap_stat = statistic_func(bootstrap_sample)
               bootstrap_statistics.append(bootstrap_stat)
           
           bootstrap_statistics = np.array(bootstrap_statistics)
           
           # Calculate confidence interval
           alpha = 1 - confidence_level
           lower_percentile = (alpha / 2) * 100
           upper_percentile = (1 - alpha / 2) * 100
           
           ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
           ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
           point_estimate = statistic_func(data_array)
           
           return point_estimate, ci_lower, ci_upper
       
       def compare_distributions_bootstrap(
           self,
           group1: List[float],
           group2: List[float],
           statistic_func: callable = np.mean,
           confidence_level: float = 0.95
       ) -> Dict[str, Any]:
           """Bootstrap comparison of two distributions."""
           
           # Bootstrap confidence intervals for each group
           stat1, ci1_lower, ci1_upper = self.bootstrap_confidence_interval(
               group1, statistic_func, confidence_level
           )
           stat2, ci2_lower, ci2_upper = self.bootstrap_confidence_interval(
               group2, statistic_func, confidence_level
           )
           
           # Bootstrap confidence interval for difference
           differences = []
           group1_array = np.array(group1)
           group2_array = np.array(group2)
           
           for _ in range(self.n_bootstrap):
               sample1 = self.rng.choice(group1_array, size=len(group1_array), replace=True)
               sample2 = self.rng.choice(group2_array, size=len(group2_array), replace=True)
               
               diff = statistic_func(sample1) - statistic_func(sample2)
               differences.append(diff)
           
           differences = np.array(differences)
           alpha = 1 - confidence_level
           diff_ci_lower = np.percentile(differences, (alpha / 2) * 100)
           diff_ci_upper = np.percentile(differences, (1 - alpha / 2) * 100)
           
           # Check if confidence intervals overlap
           overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
           significant_difference = not (diff_ci_lower <= 0 <= diff_ci_upper)
           
           return {
               'group1_statistic': stat1,
               'group1_ci': (ci1_lower, ci1_upper),
               'group2_statistic': stat2,
               'group2_ci': (ci2_lower, ci2_upper),
               'difference': stat1 - stat2,
               'difference_ci': (diff_ci_lower, diff_ci_upper),
               'confidence_intervals_overlap': overlap,
               'significant_difference': significant_difference,
               'confidence_level': confidence_level
           }

Performance Optimization Strategies
-----------------------------------

Parallel Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize evaluation throughput with parallel processing:

.. code-block:: python

   import asyncio
   from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
   from typing import List, Callable, Any
   
   class ParallelEvaluationPipeline:
       """Parallel processing pipeline for large-scale evaluations."""
       
       def __init__(
           self,
           max_concurrent_requests: int = 10,
           max_workers: int = 4,
           use_process_pool: bool = False
       ):
           self.max_concurrent_requests = max_concurrent_requests
           self.max_workers = max_workers
           self.use_process_pool = use_process_pool
           self.semaphore = asyncio.Semaphore(max_concurrent_requests)
           
       async def parallel_model_evaluation(
           self,
           models: List[ModelInterface],
           benchmarks: List[TranslationBenchmark],
           samples_per_evaluation: int = 100
       ) -> Dict[str, Dict[str, Any]]:
           """Evaluate multiple models on multiple benchmarks in parallel."""
           
           # Create all evaluation tasks
           tasks = []
           for model in models:
               for benchmark in benchmarks:
                   task = self._create_evaluation_task(
                       model, benchmark, samples_per_evaluation
                   )
                   tasks.append(task)
           
           # Execute tasks with concurrency control
           results = await asyncio.gather(*tasks, return_exceptions=True)
           
           # Organize results by model and benchmark
           organized_results = {}
           result_index = 0
           
           for model in models:
               model_name = getattr(model, 'model_name', str(model))
               organized_results[model_name] = {}
               
               for benchmark in benchmarks:
                   benchmark_name = getattr(benchmark, 'name', f'benchmark_{result_index}')
                   result = results[result_index]
                   
                   if isinstance(result, Exception):
                       organized_results[model_name][benchmark_name] = {
                           'error': str(result),
                           'success': False
                       }
                   else:
                       organized_results[model_name][benchmark_name] = result
                   
                   result_index += 1
           
           return organized_results
       
       async def _create_evaluation_task(
           self,
           model: ModelInterface,
           benchmark: TranslationBenchmark,
           num_samples: int
       ) -> Dict[str, Any]:
           """Create a single evaluation task with semaphore control."""
           
           async with self.semaphore:
               try:
                   result = await benchmark.run_async(model, num_samples=num_samples)
                   return {
                       'score': result.score,
                       'detailed_metrics': result.detailed_metrics,
                       'success': True,
                       'execution_time': getattr(result, 'execution_time', None)
                   }
               except Exception as e:
                   return {
                       'error': str(e),
                       'success': False
                   }

Caching and Memoization
~~~~~~~~~~~~~~~~~~~~~~~

Implement intelligent caching for expensive operations:

.. code-block:: python

   import hashlib
   import pickle
   from pathlib import Path
   from functools import wraps
   
   class IntelligentCache:
       """Intelligent caching system for evaluation results."""
       
       def __init__(self, cache_dir: Path, max_cache_size_gb: float = 5.0):
           self.cache_dir = Path(cache_dir)
           self.cache_dir.mkdir(parents=True, exist_ok=True)
           self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert to bytes
           
       def cached_evaluation(self, ttl_hours: int = 24):
           """Decorator for caching evaluation results."""
           
           def decorator(func: Callable) -> Callable:
               @wraps(func)
               async def wrapper(*args, **kwargs):
                   # Generate cache key
                   cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                   cache_file = self.cache_dir / f"{cache_key}.pkl"
                   
                   # Check if cached result exists and is fresh
                   if cache_file.exists():
                       cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                       if cache_age_hours < ttl_hours:
                           try:
                               with open(cache_file, 'rb') as f:
                                   return pickle.load(f)
                           except Exception:
                               # Cache file corrupted, continue to regenerate
                               cache_file.unlink(missing_ok=True)
                   
                   # Execute function and cache result
                   result = await func(*args, **kwargs)
                   
                   try:
                       with open(cache_file, 'wb') as f:
                           pickle.dump(result, f)
                       
                       # Clean up old cache files if needed
                       self._cleanup_cache()
                       
                   except Exception as e:
                       # Don't fail if caching fails
                       print(f"Warning: Failed to cache result: {e}")
                   
                   return result
               
               return wrapper
           return decorator
       
       def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
           """Generate a deterministic cache key."""
           # Create a deterministic representation
           key_data = {
               'function': func_name,
               'args': str(args),
               'kwargs': sorted(kwargs.items())
           }
           
           key_string = str(key_data)
           return hashlib.sha256(key_string.encode()).hexdigest()[:16]
       
       def _cleanup_cache(self):
           """Remove old cache files if cache size exceeds limit."""
           cache_files = list(self.cache_dir.glob("*.pkl"))
           
           # Calculate total cache size
           total_size = sum(f.stat().st_size for f in cache_files)
           
           if total_size > self.max_cache_size:
               # Sort by modification time (oldest first)
               cache_files.sort(key=lambda f: f.stat().st_mtime)
               
               # Remove files until under limit
               for cache_file in cache_files:
                   cache_file.unlink()
                   total_size -= cache_file.stat().st_size
                   
                   if total_size <= self.max_cache_size * 0.8:  # Remove to 80% of limit
                       break

Integration with Research Workflows
-----------------------------------

Academic Publication Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate publication-ready results with proper statistical reporting:

.. code-block:: python

   class AcademicResultsGenerator:
       """Generate publication-ready results with statistical rigor."""
       
       def __init__(self, output_dir: Path):
           self.output_dir = Path(output_dir)
           self.output_dir.mkdir(parents=True, exist_ok=True)
           
       def generate_publication_results(
           self,
           evaluation_results: Dict[str, Any],
           study_metadata: Dict[str, Any]
       ) -> Dict[str, Path]:
           """Generate comprehensive publication package."""
           
           outputs = {}
           
           # 1. Statistical analysis report
           stats_report = self._generate_statistical_report(evaluation_results)
           stats_file = self.output_dir / "statistical_analysis.json"
           with open(stats_file, 'w') as f:
               json.dump(stats_report, f, indent=2)
           outputs['statistical_analysis'] = stats_file
           
           # 2. LaTeX tables for paper
           latex_tables = self._generate_latex_tables(evaluation_results)
           latex_file = self.output_dir / "results_tables.tex"
           with open(latex_file, 'w') as f:
               f.write(latex_tables)
           outputs['latex_tables'] = latex_file
           
           # 3. Publication-quality plots
           plots_dir = self.output_dir / "plots"
           plots_dir.mkdir(exist_ok=True)
           plot_files = self._generate_publication_plots(evaluation_results, plots_dir)
           outputs['plots'] = plot_files
           
           # 4. Reproducibility package
           repro_file = self._generate_reproducibility_package(
               evaluation_results, study_metadata
           )
           outputs['reproducibility'] = repro_file
           
           # 5. Raw data in standard formats
           data_file = self.output_dir / "raw_results.csv"
           self._export_to_csv(evaluation_results, data_file)
           outputs['raw_data'] = data_file
           
           return outputs
       
       def _generate_latex_tables(self, results: Dict[str, Any]) -> str:
           """Generate LaTeX tables for academic papers."""
           
           latex_content = []
           
           # Main results table
           latex_content.append("\\begin{table}[htbp]")
           latex_content.append("\\centering")
           latex_content.append("\\caption{Model Performance on Contamination-Resistant Benchmarks}")
           latex_content.append("\\label{tab:contamination_results}")
           latex_content.append("\\begin{tabular}{lcccc}")
           latex_content.append("\\toprule")
           latex_content.append("Model & Baseline & Transformed & Drop & p-value \\\\")
           latex_content.append("\\midrule")
           
           for model_name, model_results in results.items():
               baseline_score = model_results.get('baseline', {}).get('score', 0)
               transformed_score = model_results.get('transformed', {}).get('score', 0)
               drop = baseline_score - transformed_score
               p_value = model_results.get('statistical_test', {}).get('p_value', 1.0)
               
               # Format with appropriate precision
               baseline_str = f"{baseline_score:.3f}"
               transformed_str = f"{transformed_score:.3f}"
               drop_str = f"{drop:.3f}"
               p_str = f"{p_value:.3f}" if p_value >= 0.001 else "$< 0.001$"
               
               latex_content.append(
                   f"{model_name} & {baseline_str} & {transformed_str} & {drop_str} & {p_str} \\\\"
               )
           
           latex_content.append("\\bottomrule")
           latex_content.append("\\end{tabular}")
           latex_content.append("\\end{table}")
           
           return "\n".join(latex_content)

This advanced usage guide provides sophisticated patterns for contamination-resistant evaluation, complex transformation chains, statistical analysis, and research-grade workflows. The examples demonstrate ScrambleBench's capability to handle enterprise-scale evaluation pipelines and academic research requirements.