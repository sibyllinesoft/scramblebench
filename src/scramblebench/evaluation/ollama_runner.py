"""
Ollama-based evaluation runner for ScrambleBench.

This module provides evaluation capabilities using Ollama for local LLM inference.
It handles model initialization, sequential processing (appropriate for local inference),
error handling, and result collection for multiple models and transformations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

from tqdm import tqdm

from scramblebench.llm.ollama_client import OllamaClient, create_ollama_client
from scramblebench.llm.model_interface import ModelConfig as LLMModelConfig
from .config import ModelConfig, EvaluationConfig
from .transformation_pipeline import TransformationSet, TransformationResult
from .openrouter_runner import EvaluationResult  # Reuse the same result class


class OllamaEvaluationRunner:
    """
    Evaluation runner using Ollama for local model inference.
    
    Handles model initialization, sequential processing optimized for local inference,
    comprehensive error handling, and result collection for multiple models and transformations.
    
    Key features:
    - Local model verification and setup
    - Sequential processing to avoid resource contention
    - Comprehensive error handling for local inference scenarios
    - Progress tracking and detailed logging
    - Resource usage monitoring
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        base_url: str = "http://localhost:11434",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Ollama evaluation runner.
        
        Args:
            config: Evaluation configuration with Ollama models
            base_url: Ollama server URL
            logger: Logger instance for debugging and monitoring
        """
        self.config = config
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tracking dictionaries first
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._response_times: Dict[str, List[float]] = {}
        
        # Initialize model clients
        self.model_clients: Dict[str, OllamaClient] = {}
        self._setup_model_clients()
        
        # Resource monitoring
        self._total_inference_time = 0.0
        self._total_requests = 0
    
    def _setup_model_clients(self) -> None:
        """Setup Ollama clients for all configured models."""
        self.logger.info("Setting up Ollama model clients...")
        
        for model_config in self.config.models:
            # Handle both enum and string provider types
            provider = model_config.provider.value if hasattr(model_config.provider, 'value') else model_config.provider
            if provider != "ollama":
                self.logger.warning(f"Skipping non-Ollama model: {model_config.name}")
                continue
            
            try:
                # Convert to LLM model config
                llm_config = LLMModelConfig(
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    timeout=model_config.timeout,
                    top_p=getattr(model_config, 'top_p', 1.0),
                    frequency_penalty=getattr(model_config, 'frequency_penalty', 0.0),
                    presence_penalty=getattr(model_config, 'presence_penalty', 0.0)
                )
                
                client = create_ollama_client(
                    model_name=model_config.name,
                    base_url=self.base_url,
                    config=llm_config
                )
                
                self.model_clients[model_config.name] = client
                self._request_counts[model_config.name] = 0
                self._error_counts[model_config.name] = 0
                self._response_times[model_config.name] = []
                
                self.logger.info(f"Successfully initialized Ollama client for: {model_config.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama client for {model_config.name}: {e}")
                self.logger.error(f"Make sure model is available: ollama pull {model_config.name}")
                raise
        
        if not self.model_clients:
            raise RuntimeError("No Ollama models successfully initialized")
        
        self.logger.info(f"Initialized {len(self.model_clients)} Ollama model clients")
    
    def verify_models(self) -> bool:
        """
        Verify all models are available and ready for inference.
        
        Returns:
            True if all models are available, False otherwise
        """
        self.logger.info("Verifying Ollama model availability...")
        
        all_available = True
        for model_name, client in self.model_clients.items():
            if not client.is_model_available():
                self.logger.error(f"Model {model_name} not available")
                self.logger.error(f"Pull with: ollama pull {model_name}")
                all_available = False
            else:
                self.logger.info(f"✓ Model {model_name} is available")
        
        return all_available
    
    async def evaluate_transformations(
        self,
        transformation_sets: List[TransformationSet],
        include_original: bool = True,
        max_samples: Optional[int] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate models on all transformation sets.
        
        Args:
            transformation_sets: List of transformation sets to evaluate
            include_original: Whether to evaluate original problems
            max_samples: Maximum number of samples per transformation (None for all)
            
        Returns:
            List of evaluation results
        """
        self.logger.info("Starting Ollama model evaluation...")
        
        if not self.verify_models():
            raise RuntimeError("Some models are not available - check Ollama installation")
        
        all_results = []
        total_evaluations = self._count_total_evaluations(transformation_sets, include_original, max_samples)
        
        self.logger.info(f"Total evaluations planned: {total_evaluations}")
        
        # Sequential processing for local inference
        with tqdm(total=total_evaluations, desc="Evaluating") as pbar:
            for transformation_set in transformation_sets:
                for model_name, client in self.model_clients.items():
                    
                    # Evaluate original problems if requested
                    if include_original:
                        original_results = await self._evaluate_problems(
                            client=client,
                            model_name=model_name,
                            problems=transformation_set.original_problems,
                            transformation_type="original",
                            max_samples=max_samples,
                            pbar=pbar
                        )
                        all_results.extend(original_results)
                    
                    # Evaluate transformed problems
                    for transformation_result in transformation_set.transformations:
                        transformed_results = await self._evaluate_problems(
                            client=client,
                            model_name=model_name,
                            problems=transformation_result.problems,
                            transformation_type=transformation_result.transformation_type,
                            max_samples=max_samples,
                            pbar=pbar,
                            original_problems=transformation_set.original_problems
                        )
                        all_results.extend(transformed_results)
        
        self._log_evaluation_summary(all_results)
        return all_results
    
    def _count_total_evaluations(
        self,
        transformation_sets: List[TransformationSet],
        include_original: bool,
        max_samples: Optional[int]
    ) -> int:
        """Count total number of evaluations that will be performed."""
        total = 0
        
        for transformation_set in transformation_sets:
            for model_name in self.model_clients:
                # Count original evaluations
                if include_original:
                    original_count = len(transformation_set.original_problems)
                    if max_samples:
                        original_count = min(original_count, max_samples)
                    total += original_count
                
                # Count transformation evaluations
                for transformation_result in transformation_set.transformations:
                    transform_count = len(transformation_result.problems)
                    if max_samples:
                        transform_count = min(transform_count, max_samples)
                    total += transform_count
        
        return total
    
    async def _evaluate_problems(
        self,
        client: OllamaClient,
        model_name: str,
        problems: List[Dict[str, Any]],
        transformation_type: str,
        max_samples: Optional[int] = None,
        pbar: Optional[tqdm] = None,
        original_problems: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate a model on a set of problems.
        
        Args:
            client: Ollama client for the model
            model_name: Name of the model
            problems: List of problems to evaluate
            transformation_type: Type of transformation applied
            max_samples: Maximum number of samples to evaluate
            pbar: Progress bar to update
            original_problems: Original problems (for transformed evaluations)
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Limit samples if specified
        eval_problems = problems[:max_samples] if max_samples else problems
        
        for i, problem in enumerate(eval_problems):
            try:
                start_time = time.time()
                
                # Extract question/prompt from problem
                question = self._extract_question(problem)
                
                # Generate response
                response = client.generate(question)
                
                eval_time = time.time() - start_time
                self._total_inference_time += eval_time
                self._total_requests += 1
                
                # Track performance
                self._request_counts[model_name] += 1
                self._response_times[model_name].append(eval_time)
                
                # Create result
                if response.error:
                    self._error_counts[model_name] += 1
                    result = EvaluationResult(
                        problem_id=problem.get('id', f"{transformation_type}_{i}"),
                        transformation_type=transformation_type,
                        model_name=model_name,
                        original_problem=original_problems[i] if original_problems and i < len(original_problems) else problem,
                        transformed_problem=problem if original_problems else None,
                        model_response="",
                        response_metadata=response.metadata,
                        success=False,
                        error=response.error,
                        evaluation_time=eval_time
                    )
                else:
                    result = EvaluationResult(
                        problem_id=problem.get('id', f"{transformation_type}_{i}"),
                        transformation_type=transformation_type,
                        model_name=model_name,
                        original_problem=original_problems[i] if original_problems and i < len(original_problems) else problem,
                        transformed_problem=problem if original_problems else None,
                        model_response=response.text,
                        response_metadata=response.metadata,
                        success=True,
                        evaluation_time=eval_time
                    )
                
                results.append(result)
                
                # Update progress
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'model': model_name[:15],
                        'type': transformation_type[:10],
                        'time': f"{eval_time:.1f}s"
                    })
                
                # Small delay between requests to avoid overwhelming local inference
                if i < len(eval_problems) - 1:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {model_name} on problem {i}: {e}")
                self._error_counts[model_name] += 1
                
                result = EvaluationResult(
                    problem_id=problem.get('id', f"{transformation_type}_{i}"),
                    transformation_type=transformation_type,
                    model_name=model_name,
                    original_problem=original_problems[i] if original_problems and i < len(original_problems) else problem,
                    transformed_problem=problem if original_problems else None,
                    model_response="",
                    response_metadata={},
                    success=False,
                    error=str(e),
                    evaluation_time=0.0
                )
                
                results.append(result)
                if pbar:
                    pbar.update(1)
        
        return results
    
    def _extract_question(self, problem: Dict[str, Any]) -> str:
        """
        Extract the question/prompt from a problem dictionary.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Question text to send to the model
        """
        # Try common question field names
        for field in ['question', 'prompt', 'text', 'input', 'problem']:
            if field in problem:
                return str(problem[field])
        
        # If no standard field found, try to construct from available fields
        if 'context' in problem and 'question' in problem:
            return f"Context: {problem['context']}\n\nQuestion: {problem['question']}"
        
        # Fall back to string representation
        self.logger.warning(f"Could not extract question from problem: {list(problem.keys())}")
        return str(problem)
    
    def _log_evaluation_summary(self, results: List[EvaluationResult]) -> None:
        """Log summary statistics for the evaluation."""
        total_results = len(results)
        successful_results = len([r for r in results if r.success])
        
        self.logger.info("=== Ollama Evaluation Summary ===")
        self.logger.info(f"Total evaluations: {total_results}")
        self.logger.info(f"Successful: {successful_results} ({successful_results/total_results*100:.1f}%)")
        self.logger.info(f"Failed: {total_results - successful_results}")
        self.logger.info(f"Total inference time: {self._total_inference_time:.1f} seconds")
        self.logger.info(f"Average inference time: {self._total_inference_time/self._total_requests:.2f} seconds/request")
        
        # Model-specific statistics
        for model_name, client in self.model_clients.items():
            requests = self._request_counts[model_name]
            errors = self._error_counts[model_name]
            avg_time = sum(self._response_times[model_name]) / len(self._response_times[model_name]) if self._response_times[model_name] else 0
            
            self.logger.info(f"{model_name}: {requests} requests, {errors} errors, {avg_time:.2f}s avg")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models."""
        stats = {
            'total_requests': self._total_requests,
            'total_inference_time': self._total_inference_time,
            'average_inference_time': self._total_inference_time / self._total_requests if self._total_requests > 0 else 0,
            'models': {}
        }
        
        for model_name in self.model_clients:
            response_times = self._response_times[model_name]
            stats['models'][model_name] = {
                'requests': self._request_counts[model_name],
                'errors': self._error_counts[model_name],
                'error_rate': self._error_counts[model_name] / self._request_counts[model_name] if self._request_counts[model_name] > 0 else 0,
                'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0
            }
        
        return stats
    
    def save_performance_stats(self, output_path: Path) -> None:
        """Save performance statistics to a JSON file."""
        stats = self.get_performance_stats()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Performance stats saved to: {output_path}")