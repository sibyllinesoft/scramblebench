"""
Base benchmark class for all ScrambleBench benchmarks.

This module defines the abstract base class that all specific benchmark
implementations must inherit from, providing a standardized interface
for benchmark execution, configuration, and result handling.

The :class:`BaseBenchmark` class serves as the foundation for all benchmarking
functionality in ScrambleBench, implementing common patterns for:

* Data preparation and loading
* Model evaluation workflows  
* Metrics computation and aggregation
* Result storage and management
* Configuration validation

Example:
    Creating a custom benchmark::

        class MyBenchmark(BaseBenchmark):
            def prepare_data(self):
                # Load and prepare your data
                pass
                
            def run_single_evaluation(self, model, data_item):
                # Evaluate model on single item
                response = model.generate(data_item['prompt'])
                return {'response': response, 'expected': data_item['answer']}
                
            def compute_metrics(self, results):
                # Compute aggregate metrics
                accuracy = sum(r['response'] == r['expected'] for r in results) / len(results)
                return {'score': accuracy, 'accuracy': accuracy}
                
            def get_evaluation_data(self, num_samples=None):
                # Return data items for evaluation
                return self.data[:num_samples] if num_samples else self.data

Note:
    All benchmark implementations must inherit from :class:`BaseBenchmark` and 
    implement the required abstract methods. The base class handles the common
    workflow of data preparation, evaluation, and result aggregation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import logging

from .unified_config import ScrambleBenchConfig


@dataclass
class BenchmarkResult:
    """
    Container for benchmark evaluation results.
    
    This class encapsulates all information about a completed benchmark
    evaluation, including performance metrics, timing data, and metadata.
    
    :param benchmark_name: Name of the benchmark that generated this result
    :type benchmark_name: str
    :param model_name: Name or identifier of the evaluated model
    :type model_name: str
    :param score: Primary performance score, typically in range [0.0, 1.0]
    :type score: float
    :param metrics: Dictionary containing detailed performance metrics
    :type metrics: Dict[str, Any]
    :param metadata: Additional information about the evaluation run
    :type metadata: Dict[str, Any]
    :param duration: Time taken to complete evaluation in seconds
    :type duration: float
    :param timestamp: Unix timestamp when evaluation started
    :type timestamp: float
    
    Example:
        Creating a benchmark result::
        
            result = BenchmarkResult(
                benchmark_name="math_translation",
                model_name="gpt-4",
                score=0.85,
                metrics={
                    "accuracy": 0.85,
                    "avg_response_time": 1.2,
                    "total_questions": 100
                },
                metadata={
                    "language_type": "substitution",
                    "complexity": 5
                },
                duration=120.5,
                timestamp=1703123456.0
            )
            
    Note:
        The `score` field should represent the primary performance metric
        for the benchmark. Additional detailed metrics should be stored
        in the `metrics` dictionary.
    """
    benchmark_name: str
    model_name: str
    score: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    duration: float
    timestamp: float


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks in ScrambleBench.
    
    This class defines the common interface and shared functionality
    that all benchmark implementations must provide. It implements the
    core evaluation workflow while allowing subclasses to customize
    specific aspects of data preparation, evaluation, and metrics computation.
    
    The benchmark lifecycle follows this pattern:
    
    1. **Initialization**: Set up configuration and logging
    2. **Data Preparation**: Load and transform evaluation data  
    3. **Individual Evaluation**: Process each data item with the model
    4. **Metrics Computation**: Aggregate results into summary metrics
    5. **Result Storage**: Save results for analysis and comparison
    
    :param name: Human-readable name of the benchmark
    :type name: str
    :param config: Configuration object for benchmark settings
    :type config: Optional[ScrambleBenchConfig]
    :param logger: Logger instance for this benchmark
    :type logger: Optional[logging.Logger]
    
    :ivar name: The benchmark's name
    :vartype name: str
    :ivar config: Configuration settings for this benchmark
    :vartype config: Config
    :ivar logger: Logger instance for this benchmark
    :vartype logger: logging.Logger
    :ivar _results: Internal storage for benchmark results
    :vartype _results: List[BenchmarkResult]
    
    Example:
        Implementing a simple Q&A benchmark::
        
            class SimpleQABenchmark(BaseBenchmark):
                def __init__(self, dataset_path: str, **kwargs):
                    super().__init__("simple_qa", **kwargs)
                    self.dataset_path = dataset_path
                    self.questions = []
                    
                def prepare_data(self):
                    # Load questions from file
                    with open(self.dataset_path) as f:
                        self.questions = json.load(f)
                        
                def run_single_evaluation(self, model, data_item):
                    response = model.generate(data_item['question'])
                    correct = response.strip().lower() == data_item['answer'].lower()
                    return {'correct': correct, 'response': response}
                    
                def compute_metrics(self, results):
                    accuracy = sum(r['correct'] for r in results) / len(results)
                    return {'score': accuracy, 'accuracy': accuracy}
                    
                def get_evaluation_data(self, num_samples=None):
                    data = self.questions
                    return data[:num_samples] if num_samples else data
    
    See Also:
        :class:`scramblebench.translation.benchmark.TranslationBenchmark`: Translation-based benchmarks
        :class:`scramblebench.longcontext.benchmark.LongContextBenchmark`: Long context benchmarks
    """
    
    def __init__(
        self, 
        name: str, 
        config: Optional[ScrambleBenchConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the benchmark with configuration and logging.
        
        :param name: Name of this benchmark instance
        :type name: str
        :param config: Configuration object (creates default if None)
        :type config: Optional[ScrambleBenchConfig]
        :param logger: Logger instance (creates default if None)  
        :type logger: Optional[logging.Logger]
        
        Example:
            Basic initialization::
            
                benchmark = MyBenchmark("my_benchmark")
                
            With custom configuration::
            
                config = Config({"random_seed": 42, "max_samples": 100})
                benchmark = MyBenchmark("my_benchmark", config=config)
        """
        self.name = name
        self.config = config or Config()
        self.logger = logger or logging.getLogger(f"scramblebench.{name}")
        self._results: List[BenchmarkResult] = []
    
    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare benchmark data for evaluation.
        
        This method should load, transform, or generate the data
        that will be used during benchmark execution. The implementation
        is benchmark-specific and may involve:
        
        * Loading data from files or databases
        * Transforming existing datasets 
        * Generating synthetic evaluation data
        * Applying preprocessing steps
        * Validating data integrity
        
        :raises FileNotFoundError: If required data files are missing
        :raises ValueError: If data validation fails
        
        Example:
            Loading and validating data::
            
                def prepare_data(self):
                    with open(self.dataset_path) as f:
                        self.data = json.load(f)
                    
                    # Validate data format
                    for item in self.data:
                        if 'question' not in item or 'answer' not in item:
                            raise ValueError("Invalid data format")
                            
                    self.logger.info(f"Loaded {len(self.data)} items")
        
        Note:
            This method is called automatically by :meth:`run` before 
            evaluation begins. It should populate any instance variables
            needed by :meth:`get_evaluation_data`.
        """
        pass
    
    @abstractmethod
    def run_single_evaluation(
        self, 
        model: Any, 
        data_item: Any
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single data item.
        
        This method defines how to evaluate the model on a single
        data point. It should extract the necessary information from
        the data item, query the model, and return the evaluation results.
        
        :param model: The model instance to evaluate
        :type model: Any
        :param data_item: Single item from the benchmark dataset
        :type data_item: Any
        :return: Dictionary containing evaluation results for this item
        :rtype: Dict[str, Any]
        
        :raises ModelError: If model evaluation fails
        :raises ValueError: If data_item format is invalid
        
        Example:
            Q&A evaluation::
            
                def run_single_evaluation(self, model, data_item):
                    question = data_item['question']
                    expected = data_item['answer']
                    
                    # Get model response
                    response = model.generate(question)
                    
                    # Evaluate correctness
                    correct = self.evaluate_response(response, expected)
                    
                    return {
                        'question': question,
                        'response': response,
                        'expected': expected,
                        'correct': correct,
                        'response_time': model.last_response_time
                    }
        
        Note:
            The returned dictionary should contain all information needed
            for metrics computation. Common fields include 'correct' (bool),
            'score' (float), 'response' (str), and 'expected' (str).
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate metrics from individual evaluation results.
        
        This method aggregates the results from all individual evaluations
        into summary metrics. The returned dictionary must include a 'score'
        field representing the primary benchmark performance metric.
        
        :param results: List of results from run_single_evaluation calls
        :type results: List[Dict[str, Any]]
        :return: Dictionary containing computed aggregate metrics
        :rtype: Dict[str, Any]
        
        :raises ValueError: If results list is empty or malformed
        
        Example:
            Computing accuracy metrics::
            
                def compute_metrics(self, results):
                    if not results:
                        return {'score': 0.0, 'accuracy': 0.0, 'count': 0}
                    
                    correct_count = sum(1 for r in results if r.get('correct', False))
                    total_count = len(results)
                    accuracy = correct_count / total_count
                    
                    avg_response_time = sum(r.get('response_time', 0) for r in results) / total_count
                    
                    return {
                        'score': accuracy,
                        'accuracy': accuracy, 
                        'correct_count': correct_count,
                        'total_count': total_count,
                        'avg_response_time': avg_response_time
                    }
        
        Note:
            The 'score' field is used as the primary performance metric
            and should be in the range [0.0, 1.0] where higher is better.
            Additional metrics can provide detailed performance insights.
        """
        pass
    
    def run(
        self, 
        model: Any, 
        num_samples: Optional[int] = None,
        save_results: bool = True
    ) -> BenchmarkResult:
        """
        Execute the complete benchmark evaluation workflow.
        
        This method orchestrates the entire benchmark evaluation process:
        data preparation, individual evaluations, metrics computation, and
        result storage. It provides a standardized workflow while allowing
        subclasses to customize specific evaluation steps.
        
        :param model: The model instance to evaluate
        :type model: Any
        :param num_samples: Number of samples to evaluate (None for all available)
        :type num_samples: Optional[int]
        :param save_results: Whether to save results to persistent storage
        :type save_results: bool
        :return: Complete benchmark evaluation results
        :rtype: BenchmarkResult
        
        :raises ValueError: If model is invalid or evaluation fails
        :raises RuntimeError: If benchmark is not properly configured
        
        Example:
            Running a benchmark evaluation::
            
                from scramblebench.llm import OpenRouterClient
                
                # Initialize model
                model = OpenRouterClient("gpt-4")
                
                # Run benchmark
                result = benchmark.run(
                    model=model,
                    num_samples=100,
                    save_results=True
                )
                
                print(f"Score: {result.score:.3f}")
                print(f"Duration: {result.duration:.1f}s")
        
        Note:
            This method automatically calls :meth:`prepare_data` before evaluation
            and optionally saves results via :meth:`save_result`. The evaluation
            progress is logged at INFO level.
        """
        start_time = time.time()
        
        self.logger.info(f"Starting benchmark: {self.name}")
        
        # Prepare data
        self.prepare_data()
        
        # Get evaluation data
        eval_data = self.get_evaluation_data(num_samples)
        
        # Run evaluations
        individual_results = []
        for i, data_item in enumerate(eval_data):
            self.logger.debug(f"Evaluating item {i+1}/{len(eval_data)}")
            result = self.run_single_evaluation(model, data_item)
            individual_results.append(result)
        
        # Compute metrics
        metrics = self.compute_metrics(individual_results)
        
        # Create benchmark result
        duration = time.time() - start_time
        benchmark_result = BenchmarkResult(
            benchmark_name=self.name,
            model_name=getattr(model, 'name', str(model)),
            score=metrics.get('score', 0.0),
            metrics=metrics,
            metadata={
                'num_samples': len(eval_data),
                'config': self.config.to_dict(),
            },
            duration=duration,
            timestamp=start_time
        )
        
        # Store result
        self._results.append(benchmark_result)
        
        if save_results:
            self.save_result(benchmark_result)
        
        self.logger.info(
            f"Benchmark completed: {self.name} | "
            f"Score: {benchmark_result.score:.4f} | "
            f"Duration: {duration:.2f}s"
        )
        
        return benchmark_result
    
    @abstractmethod
    def get_evaluation_data(self, num_samples: Optional[int] = None) -> List[Any]:
        """
        Get the data items for evaluation.
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            List of data items to evaluate
        """
        pass
    
    def save_result(self, result: BenchmarkResult) -> None:
        """
        Save benchmark result to storage.
        
        Args:
            result: The benchmark result to save
        """
        results_dir = Path(self.config.get('results_dir', 'data/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Implementation would save to JSON/pickle/database
        # For now, just log
        self.logger.info(f"Result saved: {result.benchmark_name}")
    
    @property
    def results(self) -> List[BenchmarkResult]:
        """Get all results from this benchmark instance."""
        return self._results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()
        
    def validate_config(self) -> bool:
        """
        Validate the benchmark configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Base validation - subclasses can override
        return self.config is not None