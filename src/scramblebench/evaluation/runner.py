"""
Main evaluation runner that orchestrates the entire evaluation pipeline.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

from .config import EvaluationConfig, ModelProvider
from .transformation_pipeline import TransformationPipeline
from .openrouter_runner import OpenRouterEvaluationRunner, EvaluationResult
from .results import ResultsManager, EvaluationResults
from .metrics import MetricsCalculator
from .plotting import PlotGenerator
from scramblebench.utils.data_loader import DataLoader


class EvaluationRunner:
    """
    Main evaluation runner that orchestrates the entire evaluation pipeline.
    
    Coordinates transformation generation, model evaluation, results storage,
    metrics calculation, and plot generation.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        data_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration
            data_dir: Directory for data storage
            logger: Logger instance
        """
        self.config = config
        self.data_dir = data_dir or Path("data")
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.transformation_pipeline = TransformationPipeline(
            config.transformations,
            data_dir=self.data_dir,
            logger=self.logger
        )
        
        self.results_manager = ResultsManager(
            config.get_output_dir(),
            logger=self.logger
        )
        
        self.data_loader = DataLoader()
        
        # Initialize model runners based on providers
        self.model_runners = self._initialize_model_runners()
        
        # Progress tracking
        self._start_time = None
        self._current_stage = "initialized"
    
    def _initialize_model_runners(self) -> Dict[str, Any]:
        """Initialize model runners based on configured providers."""
        runners = {}
        
        # Group models by provider
        provider_models = {}
        for model in self.config.models:
            provider = model.provider
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model)
        
        # Initialize runners for each provider
        for provider, models in provider_models.items():
            if provider == ModelProvider.OPENROUTER:
                # Create config with only OpenRouter models
                openrouter_config = self.config.copy()
                openrouter_config.models = models
                
                runners[provider.value] = OpenRouterEvaluationRunner(
                    openrouter_config,
                    logger=self.logger
                )
            else:
                self.logger.warning(f"Provider {provider} not yet supported")
        
        return runners
    
    async def run_evaluation(
        self,
        include_original: bool = True,
        save_intermediate: bool = True
    ) -> EvaluationResults:
        """
        Run the complete evaluation pipeline.
        
        Args:
            include_original: Whether to evaluate original (untransformed) problems
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Complete evaluation results
        """
        self._start_time = time.time()
        self.logger.info(f"Starting evaluation: {self.config.experiment_name}")
        
        try:
            # Stage 1: Load benchmark data
            self._current_stage = "loading_data"
            benchmark_problems = await self._load_benchmark_data()
            
            # Stage 2: Generate transformations
            self._current_stage = "generating_transformations"
            transformation_sets = await self._generate_transformations(benchmark_problems)
            
            if save_intermediate:
                transformations_path = self.config.get_experiment_dir() / "transformations.json"
                self.transformation_pipeline.save_transformation_sets(transformation_sets, transformations_path)
            
            # Stage 3: Run model evaluations
            self._current_stage = "running_evaluations"
            evaluation_results = await self._run_model_evaluations(transformation_sets, include_original)
            
            # Stage 4: Create results object
            self._current_stage = "processing_results"
            results = EvaluationResults(
                results=evaluation_results,
                config=self.config,
                transformation_sets=transformation_sets,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "total_problems": len(benchmark_problems),
                    "total_transformations": sum(len(ts.transformations) for ts in transformation_sets),
                    "total_evaluations": len(evaluation_results),
                    "evaluation_time": time.time() - self._start_time,
                    "stages_completed": ["loading_data", "generating_transformations", "running_evaluations", "processing_results"]
                }
            )
            
            # Stage 5: Save results
            self._current_stage = "saving_results"
            saved_files = self.results_manager.save_results(
                results,
                self.config.experiment_name,
                format="both"
            )
            
            # Stage 6: Generate metrics and plots
            if self.config.generate_plots or self.config.calculate_significance:
                self._current_stage = "generating_analysis"
                await self._generate_analysis(results, saved_files)
            
            self._current_stage = "completed"
            total_time = time.time() - self._start_time
            
            self.logger.info(f"Evaluation completed in {total_time:.2f} seconds")
            self.logger.info(f"Results saved to: {self.config.get_experiment_dir()}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed at stage '{self._current_stage}': {e}")
            raise
    
    async def _load_benchmark_data(self) -> List[Dict[str, Any]]:
        """Load benchmark data from configured paths."""
        self.logger.info("Loading benchmark data...")
        
        all_problems = []
        
        for benchmark_path in self.config.benchmark_paths:
            self.logger.info(f"Loading benchmark: {benchmark_path}")
            
            try:
                problems = self.data_loader.load_benchmark_file(benchmark_path)
                self.logger.info(f"Loaded {len(problems)} problems from {benchmark_path}")
                all_problems.extend(problems)
                
            except Exception as e:
                self.logger.error(f"Failed to load benchmark {benchmark_path}: {e}")
                continue
        
        # Sample if requested
        if self.config.max_samples and len(all_problems) > self.config.max_samples:
            import random
            if self.config.sample_seed:
                random.seed(self.config.sample_seed)
            all_problems = random.sample(all_problems, self.config.max_samples)
            self.logger.info(f"Sampled {len(all_problems)} problems")
        
        self.logger.info(f"Total problems loaded: {len(all_problems)}")
        return all_problems
    
    async def _generate_transformations(self, problems: List[Dict[str, Any]]):
        """Generate transformations for all problems."""
        self.logger.info("Generating transformations...")
        
        # Generate problem IDs
        problem_ids = [f"problem_{i}" for i in range(len(problems))]
        
        # Generate transformations
        transformation_sets = await self.transformation_pipeline.generate_transformation_sets(
            problems, problem_ids
        )
        
        # Log statistics
        stats = self.transformation_pipeline.get_transformation_stats(transformation_sets)
        self.logger.info(f"Generated {stats['total_transformations']} transformations "
                        f"({stats['successful_transformations']} successful)")
        
        return transformation_sets
    
    async def _run_model_evaluations(self, transformation_sets, include_original: bool):
        """Run evaluations using all configured model runners."""
        self.logger.info("Running model evaluations...")
        
        all_results = []
        
        for provider_name, runner in self.model_runners.items():
            self.logger.info(f"Running evaluations with {provider_name}")
            
            try:
                if hasattr(runner, 'evaluate_transformation_sets'):
                    results = await runner.evaluate_transformation_sets(
                        transformation_sets, include_original
                    )
                    all_results.extend(results)
                    
                    # Log progress
                    successful = sum(1 for r in results if r.success)
                    self.logger.info(f"{provider_name}: {successful}/{len(results)} successful evaluations")
                    
                else:
                    self.logger.warning(f"Runner {provider_name} does not support batch evaluation")
                    
            except Exception as e:
                self.logger.error(f"Error running evaluations with {provider_name}: {e}")
                continue
        
        self.logger.info(f"Total evaluations completed: {len(all_results)}")
        return all_results
    
    async def _generate_analysis(self, results: EvaluationResults, saved_files: Dict[str, Path]):
        """Generate metrics and plots."""
        experiment_dir = self.config.get_experiment_dir()
        
        # Generate metrics report
        if self.config.calculate_significance:
            self.logger.info("Calculating metrics...")
            
            metrics_calc = MetricsCalculator()
            metrics_report = metrics_calc.generate_metrics_report(results)
            
            metrics_path = experiment_dir / "metrics_report.json"
            metrics_calc.save_metrics_report(metrics_report, metrics_path)
            saved_files["metrics"] = metrics_path
            
            self.logger.info(f"Metrics report saved to {metrics_path}")
        
        # Generate plots
        if self.config.generate_plots:
            self.logger.info("Generating plots...")
            
            plots_dir = experiment_dir / "plots"
            plot_generator = PlotGenerator()
            
            plot_results = plot_generator.generate_all_plots(results, plots_dir)
            
            successful_plots = sum(1 for r in plot_results.values() if r.success)
            self.logger.info(f"Generated {successful_plots}/{len(plot_results)} plot types")
            
            # Save plot results summary
            plot_summary = {
                "generated_plots": {
                    name: {
                        "success": result.success,
                        "files": [str(p) for p in result.file_paths],
                        "error": result.error
                    }
                    for name, result in plot_results.items()
                }
            }
            
            plot_summary_path = plots_dir / "plot_summary.json"
            with open(plot_summary_path, 'w') as f:
                json.dump(plot_summary, f, indent=2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current evaluation status."""
        status = {
            "current_stage": self._current_stage,
            "experiment_name": self.config.experiment_name,
            "start_time": self._start_time,
            "elapsed_time": time.time() - self._start_time if self._start_time else 0,
            "configured_models": len(self.config.models),
            "configured_benchmarks": len(self.config.benchmark_paths)
        }
        
        return status
    
    @classmethod
    def from_config_file(
        cls,
        config_path: Union[str, Path],
        data_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ) -> 'EvaluationRunner':
        """
        Create evaluation runner from configuration file.
        
        Args:
            config_path: Path to configuration file
            data_dir: Data directory
            logger: Logger instance
            
        Returns:
            Configured evaluation runner
        """
        config = EvaluationConfig.load_from_file(config_path)
        return cls(config, data_dir, logger)
    
    def save_config(self, output_path: Optional[Path] = None) -> Path:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional output path (defaults to experiment directory)
            
        Returns:
            Path where configuration was saved
        """
        if output_path is None:
            output_path = self.config.get_experiment_dir() / "config.yaml"
        
        self.config.save_to_file(output_path)
        return output_path


# Convenience functions for common use cases

async def run_quick_evaluation(
    benchmark_paths: List[Union[str, Path]],
    models: List[str],
    experiment_name: str,
    output_dir: str = "results",
    transformations: Optional[List[str]] = None
) -> EvaluationResults:
    """
    Run a quick evaluation with minimal configuration.
    
    Args:
        benchmark_paths: Paths to benchmark files
        models: List of model names (assumes OpenRouter)
        experiment_name: Name for the experiment
        output_dir: Output directory
        transformations: List of transformation types (defaults to all)
        
    Returns:
        Evaluation results
    """
    from .config import ModelConfig, TransformationConfig, ModelProvider, TransformationType
    
    # Create model configs
    model_configs = [
        ModelConfig(name=model, provider=ModelProvider.OPENROUTER)
        for model in models
    ]
    
    # Create transformation config
    if transformations:
        transform_types = [TransformationType(t) for t in transformations]
    else:
        transform_types = [TransformationType.ALL]
    
    transformation_config = TransformationConfig(enabled_types=transform_types)
    
    # Create evaluation config
    config = EvaluationConfig(
        experiment_name=experiment_name,
        benchmark_paths=benchmark_paths,
        output_dir=output_dir,
        models=model_configs,
        transformations=transformation_config
    )
    
    # Run evaluation
    runner = EvaluationRunner(config)
    return await runner.run_evaluation()


def create_sample_config(output_path: Union[str, Path]) -> None:
    """Create a sample configuration file."""
    from .config import EvaluationConfig, ModelConfig, TransformationConfig, ModelProvider, TransformationType
    
    sample_config = EvaluationConfig(
        experiment_name="sample_evaluation",
        description="Sample evaluation configuration",
        benchmark_paths=["data/benchmarks/sample.json"],
        models=[
            ModelConfig(
                name="anthropic/claude-3-sonnet",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0,
                max_tokens=1024
            ),
            ModelConfig(
                name="openai/gpt-4",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0,
                max_tokens=1024
            )
        ],
        transformations=TransformationConfig(
            enabled_types=[TransformationType.LANGUAGE_TRANSLATION, TransformationType.SYNONYM_REPLACEMENT],
            languages=["constructed_1", "constructed_2"],
            synonym_rate=0.3
        ),
        max_samples=100,
        generate_plots=True,
        calculate_significance=True
    )
    
    sample_config.save_to_file(output_path)