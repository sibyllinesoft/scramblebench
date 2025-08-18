#!/usr/bin/env python3
"""
ScrambleBench Evaluation Pipeline Demo

This script demonstrates how to use the ScrambleBench evaluation pipeline
to conduct comprehensive LLM robustness testing.
"""

import asyncio
import json
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ScrambleBench evaluation components
from scramblebench.evaluation import (
    EvaluationConfig, ModelConfig, TransformationConfig,
    ModelProvider, TransformationType, EvaluationMode,
    EvaluationRunner, run_quick_evaluation
)


def create_sample_benchmark():
    """Create a sample benchmark for demonstration."""
    
    sample_problems = [
        {
            "question": "What is the capital of France?",
            "choices": {
                "A": "London",
                "B": "Paris", 
                "C": "Berlin",
                "D": "Madrid"
            },
            "answer": "B",
            "category": "geography"
        },
        {
            "question": "Who wrote the novel '1984'?",
            "choices": {
                "A": "George Orwell",
                "B": "Aldous Huxley",
                "C": "Ray Bradbury", 
                "D": "William Gibson"
            },
            "answer": "A",
            "category": "literature"
        },
        {
            "question": "What is 15 + 27?",
            "choices": {
                "A": "41",
                "B": "42",
                "C": "43",
                "D": "44"
            },
            "answer": "B",
            "category": "mathematics"
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": {
                "A": "Venus",
                "B": "Jupiter",
                "C": "Mars",
                "D": "Saturn"
            },
            "answer": "C",
            "category": "science"
        },
        {
            "question": "In which year did World War II end?",
            "choices": {
                "A": "1944",
                "B": "1945",
                "C": "1946",
                "D": "1947"
            },
            "answer": "B",
            "category": "history"
        }
    ]
    
    # Save to file
    benchmark_path = Path("data/benchmarks/demo_benchmark.json")
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(benchmark_path, 'w') as f:
        json.dump(sample_problems, f, indent=2)
    
    logger.info(f"Created sample benchmark with {len(sample_problems)} problems at {benchmark_path}")
    return benchmark_path


async def demo_quick_evaluation():
    """Demonstrate quick evaluation setup."""
    logger.info("=== Demo 1: Quick Evaluation ===")
    
    # Check if API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        logger.warning("OPENROUTER_API_KEY not set. Skipping API-based evaluation.")
        return
    
    # Create sample benchmark
    benchmark_path = create_sample_benchmark()
    
    try:
        # Run quick evaluation
        results = await run_quick_evaluation(
            benchmark_paths=[benchmark_path],
            models=["anthropic/claude-3-haiku"],  # Fast, inexpensive model for demo
            experiment_name="demo_quick_eval",
            output_dir="results/demo",
            transformations=["synonym_replacement"]
        )
        
        logger.info(f"Quick evaluation completed!")
        logger.info(f"Total results: {len(results.results)}")
        logger.info(f"Successful results: {sum(1 for r in results.results if r.success)}")
        logger.info(f"Success rate: {results.get_success_rate():.2%}")
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")


def demo_configuration_creation():
    """Demonstrate creating evaluation configurations."""
    logger.info("=== Demo 2: Configuration Creation ===")
    
    # Create basic configuration
    basic_config = EvaluationConfig(
        experiment_name="demo_basic_evaluation",
        description="Basic demonstration of ScrambleBench evaluation",
        mode=EvaluationMode.ROBUSTNESS,
        benchmark_paths=["data/benchmarks/demo_benchmark.json"],
        output_dir="results/demo",
        models=[
            ModelConfig(
                name="anthropic/claude-3-haiku",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0,
                max_tokens=512,
                rate_limit=2.0
            ),
            ModelConfig(
                name="openai/gpt-3.5-turbo",
                provider=ModelProvider.OPENROUTER,
                temperature=0.0,
                max_tokens=512,
                rate_limit=2.0
            )
        ],
        transformations=TransformationConfig(
            enabled_types=[
                TransformationType.LANGUAGE_TRANSLATION,
                TransformationType.SYNONYM_REPLACEMENT,
                TransformationType.PROPER_NOUN_SWAP
            ],
            languages=["constructed_agglutinative_1", "constructed_fusional_1"],
            language_complexity=5,
            synonym_rate=0.3,
            proper_noun_strategy="random",
            seed=42
        ),
        max_samples=10,  # Small number for demo
        max_concurrent_requests=2,
        generate_plots=True,
        calculate_significance=True
    )
    
    # Save configuration
    config_path = Path("configs/demo_evaluation.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    basic_config.save_to_file(config_path)
    
    logger.info(f"Created basic configuration at {config_path}")
    
    # Create comprehensive configuration
    comprehensive_config = EvaluationConfig(
        experiment_name="demo_comprehensive_evaluation", 
        description="Comprehensive demonstration with multiple models and transformations",
        mode=EvaluationMode.COMPREHENSIVE,
        benchmark_paths=["data/benchmarks/demo_benchmark.json"],
        output_dir="results/demo",
        models=[
            ModelConfig(name="anthropic/claude-3-sonnet", provider=ModelProvider.OPENROUTER),
            ModelConfig(name="anthropic/claude-3-haiku", provider=ModelProvider.OPENROUTER),
            ModelConfig(name="openai/gpt-4", provider=ModelProvider.OPENROUTER),
            ModelConfig(name="openai/gpt-3.5-turbo", provider=ModelProvider.OPENROUTER)
        ],
        transformations=TransformationConfig(
            enabled_types=[TransformationType.ALL],  # All available transformations
            languages=["constructed_agglutinative_1", "constructed_fusional_1", "constructed_isolating_1"],
            language_complexity=7,
            synonym_rate=0.4,
            seed=123
        ),
        max_samples=20,
        generate_plots=True,
        calculate_significance=True
    )
    
    config_path = Path("configs/demo_comprehensive.yaml")
    comprehensive_config.save_to_file(config_path)
    
    logger.info(f"Created comprehensive configuration at {config_path}")


async def demo_full_evaluation():
    """Demonstrate full evaluation pipeline with configuration file."""
    logger.info("=== Demo 3: Full Evaluation Pipeline ===")
    
    # Check if API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        logger.warning("OPENROUTER_API_KEY not set. Skipping full evaluation demo.")
        return
    
    # Create sample benchmark
    benchmark_path = create_sample_benchmark()
    
    # Load configuration
    config_path = Path("configs/demo_evaluation.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Run demo_configuration_creation() first.")
        return
    
    try:
        # Create evaluation runner
        runner = EvaluationRunner.from_config_file(config_path)
        
        logger.info(f"Starting evaluation: {runner.config.experiment_name}")
        logger.info(f"Models: {[m.name for m in runner.config.models]}")
        logger.info(f"Transformations: {runner.config.transformations.enabled_types}")
        
        # Run evaluation
        results = await runner.run_evaluation(
            include_original=True,
            save_intermediate=True
        )
        
        logger.info("Evaluation completed successfully!")
        
        # Display results summary
        logger.info("=== Results Summary ===")
        logger.info(f"Experiment: {results.config.experiment_name}")
        logger.info(f"Total evaluations: {len(results.results)}")
        logger.info(f"Successful evaluations: {sum(1 for r in results.results if r.success)}")
        logger.info(f"Overall success rate: {results.get_success_rate():.2%}")
        
        # Show per-model results
        for model_name in results.get_model_names():
            model_success_rate = results.get_success_rate(model_name=model_name)
            logger.info(f"  {model_name}: {model_success_rate:.2%} success rate")
        
        # Show per-transformation results
        for transform_type in results.get_transformation_types():
            transform_success_rate = results.get_success_rate(transformation_type=transform_type)
            logger.info(f"  {transform_type}: {transform_success_rate:.2%} success rate")
        
        logger.info(f"Results saved to: {runner.config.get_experiment_dir()}")
        
    except Exception as e:
        logger.error(f"Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def demo_results_analysis():
    """Demonstrate results analysis and comparison."""
    logger.info("=== Demo 4: Results Analysis ===")
    
    from scramblebench.evaluation import ResultsManager, MetricsCalculator, PlotGenerator
    
    results_dir = Path("results/demo")
    
    if not results_dir.exists():
        logger.warning("No results directory found. Run evaluations first.")
        return
    
    # List available experiments
    results_manager = ResultsManager(results_dir)
    experiments = results_manager.list_experiments()
    
    if not experiments:
        logger.warning("No completed experiments found.")
        return
    
    logger.info(f"Available experiments: {experiments}")
    
    # Analyze the first available experiment
    experiment_name = experiments[0]
    logger.info(f"Analyzing experiment: {experiment_name}")
    
    try:
        # Load results
        results = results_manager.load_results(experiment_name)
        logger.info(f"Loaded {len(results.results)} evaluation results")
        
        # Calculate metrics
        metrics_calc = MetricsCalculator()
        metrics_report = metrics_calc.generate_metrics_report(results)
        
        logger.info("=== Accuracy Metrics ===")
        if "accuracy_metrics" in metrics_report:
            for model, metrics in metrics_report["accuracy_metrics"].items():
                logger.info(f"  {model}: {metrics['exact_match']:.2%} exact match")
        
        logger.info("=== Robustness Metrics ===")
        if "robustness_metrics" in metrics_report:
            for model, metrics in metrics_report["robustness_metrics"].items():
                logger.info(f"  {model}: {metrics['avg_degradation']:.3f} avg degradation")
                if metrics['significant_degradations']:
                    logger.info(f"    Significant degradations: {metrics['significant_degradations']}")
        
        # Generate plots
        plots_dir = results_dir / experiment_name / "demo_plots"
        plot_generator = PlotGenerator()
        
        logger.info(f"Generating plots in {plots_dir}")
        plot_results = plot_generator.generate_all_plots(results, plots_dir)
        
        successful_plots = sum(1 for r in plot_results.values() if r.success)
        logger.info(f"Generated {successful_plots} plot types successfully")
        
        for plot_type, result in plot_results.items():
            if result.success:
                logger.info(f"  ✓ {plot_type}: {len(result.file_paths)} files")
            else:
                logger.info(f"  ✗ {plot_type}: {result.error}")
        
    except Exception as e:
        logger.error(f"Results analysis failed: {e}")
        import traceback
        traceback.print_exc()


def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    logger.info("=== Demo 5: CLI Usage Examples ===")
    
    logger.info("Here are some CLI commands you can try:")
    
    print("\n# Generate configuration files:")
    print("scramblebench evaluate config configs/basic.yaml --template basic")
    print("scramblebench evaluate config configs/comprehensive.yaml --template comprehensive")
    
    print("\n# Quick evaluation (requires OPENROUTER_API_KEY):")
    print("scramblebench evaluate run \\")
    print("  --models 'anthropic/claude-3-haiku' \\")
    print("  --benchmarks 'data/benchmarks/demo_benchmark.json' \\")
    print("  --experiment-name 'cli_test' \\")
    print("  --transformations 'synonym_replacement' \\")
    print("  --max-samples 5")
    
    print("\n# Run with configuration file:")
    print("scramblebench evaluate run --config configs/demo_evaluation.yaml")
    
    print("\n# List experiments:")
    print("scramblebench evaluate list")
    
    print("\n# Analyze results:")
    print("scramblebench evaluate analyze cli_test")
    
    print("\n# Compare experiments:")
    print("scramblebench evaluate compare exp1 exp2 exp3")
    
    logger.info("For more details, see EVALUATION_GUIDE.md")


async def main():
    """Run all demonstration examples."""
    logger.info("ScrambleBench Evaluation Pipeline Demo")
    logger.info("=====================================")
    
    # Demo 1: Configuration creation (always works)
    demo_configuration_creation()
    
    # Demo 2: CLI usage examples (always works)
    demo_cli_usage()
    
    # Demo 3: Quick evaluation (requires API key)
    await demo_quick_evaluation()
    
    # Demo 4: Full evaluation (requires API key)
    await demo_full_evaluation()
    
    # Demo 5: Results analysis (works if evaluations were run)
    demo_results_analysis()
    
    logger.info("Demo completed! Check the generated files:")
    logger.info("  - configs/demo_*.yaml: Sample configurations")
    logger.info("  - data/benchmarks/demo_benchmark.json: Sample benchmark")
    logger.info("  - results/demo/: Evaluation results (if API key provided)")


if __name__ == "__main__":
    # Create data directories
    Path("data/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("configs").mkdir(parents=True, exist_ok=True)
    Path("results/demo").mkdir(parents=True, exist_ok=True)
    
    # Run demonstration
    asyncio.run(main())