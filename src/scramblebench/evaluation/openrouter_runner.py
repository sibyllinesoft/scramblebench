"""
OpenRouter-based evaluation runner for ScrambleBench.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as atqdm

from scramblebench.llm.openrouter_client import OpenRouterClient, create_openrouter_client
from scramblebench.llm.model_interface import ModelConfig as LLMModelConfig
from .config import ModelConfig, EvaluationConfig
from .transformation_pipeline import TransformationSet, TransformationResult


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a problem."""
    problem_id: str
    transformation_type: str
    model_name: str
    original_problem: Dict[str, Any]
    transformed_problem: Optional[Dict[str, Any]]
    model_response: str
    response_metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    evaluation_time: float = 0.0


class OpenRouterEvaluationRunner:
    """
    Evaluation runner using OpenRouter API for model access.
    
    Handles concurrent API requests, rate limiting, error handling,
    and result collection for multiple models and transformations.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenRouter evaluation runner.
        
        Args:
            config: Evaluation configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize model clients
        self.model_clients: Dict[str, OpenRouterClient] = {}
        self._setup_model_clients()
        
        # Request tracking
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        
        # Rate limiting
        self._rate_limiters: Dict[str, asyncio.Semaphore] = {}
    
    def _setup_model_clients(self) -> None:
        """Setup OpenRouter clients for all configured models."""
        for model_config in self.config.models:
            if model_config.provider.value != "openrouter":
                self.logger.warning(f"Skipping non-OpenRouter model: {model_config.name}")
                continue
            
            try:
                # Convert to LLM model config
                llm_config = LLMModelConfig(
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    timeout=model_config.timeout
                )
                
                client = create_openrouter_client(
                    model_name=model_config.name,
                    api_key=model_config.api_key,
                    config=llm_config
                )
                
                # Set rate limit
                client.set_rate_limit(model_config.rate_limit)
                
                self.model_clients[model_config.name] = client
                self._request_counts[model_config.name] = 0
                self._error_counts[model_config.name] = 0
                
                # Create rate limiter semaphore
                max_concurrent = min(model_config.rate_limit * 2, self.config.max_concurrent_requests)
                self._rate_limiters[model_config.name] = asyncio.Semaphore(max_concurrent)
                
                self.logger.info(f"Initialized OpenRouter client for {model_config.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize client for {model_config.name}: {e}")
    
    async def evaluate_transformation_sets(
        self,
        transformation_sets: List[TransformationSet],
        include_original: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate all models on all transformation sets.
        
        Args:
            transformation_sets: List of transformation sets to evaluate
            include_original: Whether to also evaluate original (untransformed) problems
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting evaluation of {len(transformation_sets)} transformation sets")
        
        # Create evaluation tasks
        tasks = []
        
        for ts in transformation_sets:
            # Evaluate original problem if requested
            if include_original:
                for model_name in self.model_clients.keys():
                    task = self._evaluate_problem(
                        problem_id=ts.problem_id,
                        transformation_type="original",
                        model_name=model_name,
                        original_problem=ts.original_problem,
                        transformed_problem=None
                    )
                    tasks.append(task)
            
            # Evaluate all successful transformations
            for transformation in ts.get_successful_transformations():
                for model_name in self.model_clients.keys():
                    task = self._evaluate_problem(
                        problem_id=ts.problem_id,
                        transformation_type=transformation.transformation_type,
                        model_name=model_name,
                        original_problem=ts.original_problem,
                        transformed_problem=transformation.transformed_problem
                    )
                    tasks.append(task)
        
        self.logger.info(f"Created {len(tasks)} evaluation tasks")
        
        # Execute tasks with progress tracking
        results = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            result = await coro
            results.append(result)
            
            # Log progress periodically
            if len(results) % 100 == 0:
                success_count = sum(1 for r in results if r.success)
                self.logger.info(f"Completed {len(results)}/{len(tasks)} evaluations ({success_count} successful)")
        
        # Final statistics
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"Evaluation complete: {success_count}/{len(results)} successful")
        
        return results
    
    async def _evaluate_problem(
        self,
        problem_id: str,
        transformation_type: str,
        model_name: str,
        original_problem: Dict[str, Any],
        transformed_problem: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate a single problem with a single model."""
        
        # Use rate limiter for this model
        async with self._rate_limiters[model_name]:
            start_time = time.time()
            
            try:
                client = self.model_clients[model_name]
                
                # Use transformed problem if available, otherwise original
                problem = transformed_problem if transformed_problem is not None else original_problem
                
                # Create prompt from problem
                prompt = self._create_prompt(problem)
                
                # Generate response
                response = await self._generate_response_async(client, prompt)
                
                self._request_counts[model_name] += 1
                
                if response.error:
                    self._error_counts[model_name] += 1
                    return EvaluationResult(
                        problem_id=problem_id,
                        transformation_type=transformation_type,
                        model_name=model_name,
                        original_problem=original_problem,
                        transformed_problem=transformed_problem,
                        model_response="",
                        response_metadata=response.metadata,
                        success=False,
                        error=response.error,
                        evaluation_time=time.time() - start_time
                    )
                
                return EvaluationResult(
                    problem_id=problem_id,
                    transformation_type=transformation_type,
                    model_name=model_name,
                    original_problem=original_problem,
                    transformed_problem=transformed_problem,
                    model_response=response.text,
                    response_metadata=response.metadata,
                    success=True,
                    evaluation_time=time.time() - start_time
                )
                
            except Exception as e:
                self._error_counts[model_name] += 1
                self.logger.error(f"Error evaluating {problem_id} with {model_name}: {e}")
                
                return EvaluationResult(
                    problem_id=problem_id,
                    transformation_type=transformation_type,
                    model_name=model_name,
                    original_problem=original_problem,
                    transformed_problem=transformed_problem,
                    model_response="",
                    response_metadata={},
                    success=False,
                    error=str(e),
                    evaluation_time=time.time() - start_time
                )
    
    def _create_prompt(self, problem: Dict[str, Any]) -> str:
        """Create a prompt from a problem dictionary."""
        # This is a simple implementation - can be made more sophisticated
        if "question" in problem and "choices" in problem:
            # Multiple choice format
            prompt = f"Question: {problem['question']}\n\n"
            if isinstance(problem['choices'], list):
                for i, choice in enumerate(problem['choices']):
                    prompt += f"{chr(65+i)}. {choice}\n"
            elif isinstance(problem['choices'], dict):
                for key, choice in problem['choices'].items():
                    prompt += f"{key}. {choice}\n"
            prompt += "\nAnswer:"
            
        elif "question" in problem:
            # Open-ended question
            prompt = f"Question: {problem['question']}\nAnswer:"
            
        elif "text" in problem:
            # General text completion
            prompt = problem['text']
            
        elif "prompt" in problem:
            # Direct prompt
            prompt = problem['prompt']
            
        else:
            # Fallback: use the first string field
            for key, value in problem.items():
                if isinstance(value, str):
                    prompt = f"{key}: {value}\nResponse:"
                    break
            else:
                prompt = str(problem)
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_response_async(self, client: OpenRouterClient, prompt: str):
        """Generate response with retry logic."""
        # Convert synchronous client call to async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, client.generate, prompt)
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        total_requests = sum(self._request_counts.values())
        total_errors = sum(self._error_counts.values())
        
        model_stats = {}
        for model_name in self.model_clients.keys():
            requests = self._request_counts.get(model_name, 0)
            errors = self._error_counts.get(model_name, 0)
            
            model_stats[model_name] = {
                "requests": requests,
                "errors": errors,
                "success_rate": (requests - errors) / requests if requests > 0 else 0,
                "error_rate": errors / requests if requests > 0 else 0
            }
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_success_rate": (total_requests - total_errors) / total_requests if total_requests > 0 else 0,
            "model_stats": model_stats
        }
    
    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: Path,
        format: str = "json"
    ) -> None:
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Convert to serializable format
            data = {
                "config": self.config.dict(),
                "evaluation_stats": self.get_evaluation_stats(),
                "results": []
            }
            
            for result in results:
                result_data = {
                    "problem_id": result.problem_id,
                    "transformation_type": result.transformation_type,
                    "model_name": result.model_name,
                    "original_problem": result.original_problem,
                    "transformed_problem": result.transformed_problem,
                    "model_response": result.model_response,
                    "response_metadata": result.response_metadata,
                    "success": result.success,
                    "error": result.error,
                    "evaluation_time": result.evaluation_time
                }
                data["results"].append(result_data)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "parquet":
            # Convert to pandas DataFrame for Parquet
            import pandas as pd
            
            records = []
            for result in results:
                record = {
                    "problem_id": result.problem_id,
                    "transformation_type": result.transformation_type,
                    "model_name": result.model_name,
                    "model_response": result.model_response,
                    "success": result.success,
                    "error": result.error,
                    "evaluation_time": result.evaluation_time,
                    "response_time": result.response_metadata.get('response_time', 0),
                    "total_tokens": result.response_metadata.get('total_tokens', 0),
                    "prompt_tokens": result.response_metadata.get('prompt_tokens', 0),
                    "completion_tokens": result.response_metadata.get('completion_tokens', 0)
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            df.to_parquet(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved {len(results)} results to {output_path}")
    
    @classmethod
    def load_results(cls, input_path: Path) -> Tuple[List[EvaluationResult], Dict[str, Any]]:
        """Load evaluation results from file."""
        if input_path.suffix == ".json":
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            results = []
            for result_data in data["results"]:
                result = EvaluationResult(
                    problem_id=result_data["problem_id"],
                    transformation_type=result_data["transformation_type"],
                    model_name=result_data["model_name"],
                    original_problem=result_data["original_problem"],
                    transformed_problem=result_data["transformed_problem"],
                    model_response=result_data["model_response"],
                    response_metadata=result_data["response_metadata"],
                    success=result_data["success"],
                    error=result_data.get("error"),
                    evaluation_time=result_data["evaluation_time"]
                )
                results.append(result)
            
            return results, data.get("config", {})
        
        elif input_path.suffix == ".parquet":
            import pandas as pd
            
            df = pd.read_parquet(input_path)
            results = []
            
            for _, row in df.iterrows():
                result = EvaluationResult(
                    problem_id=row["problem_id"],
                    transformation_type=row["transformation_type"],
                    model_name=row["model_name"],
                    original_problem={},  # Not stored in parquet format
                    transformed_problem=None,
                    model_response=row["model_response"],
                    response_metadata={
                        "response_time": row.get("response_time", 0),
                        "total_tokens": row.get("total_tokens", 0),
                        "prompt_tokens": row.get("prompt_tokens", 0),
                        "completion_tokens": row.get("completion_tokens", 0)
                    },
                    success=row["success"],
                    error=row.get("error"),
                    evaluation_time=row["evaluation_time"]
                )
                results.append(result)
            
            return results, {}
        
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup clients."""
        for client in self.model_clients.values():
            if hasattr(client, '__aexit__'):
                await client.__aexit__(exc_type, exc_val, exc_tb)