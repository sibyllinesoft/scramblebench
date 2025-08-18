"""
Results management and analysis for ScrambleBench evaluations.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from .openrouter_runner import EvaluationResult
from .transformation_pipeline import TransformationSet
from .config import EvaluationConfig


@dataclass
class EvaluationResults:
    """Container for evaluation results and metadata."""
    results: List[EvaluationResult]
    config: EvaluationConfig
    transformation_sets: Optional[List[TransformationSet]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "total_results": len(self.results),
                "successful_results": sum(1 for r in self.results if r.success)
            }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        records = []
        
        for result in self.results:
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
                "completion_tokens": result.response_metadata.get('completion_tokens', 0),
                "finish_reason": result.response_metadata.get('finish_reason', ''),
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_model_names(self) -> List[str]:
        """Get list of unique model names."""
        return list(set(r.model_name for r in self.results))
    
    def get_transformation_types(self) -> List[str]:
        """Get list of unique transformation types."""
        return list(set(r.transformation_type for r in self.results))
    
    def get_problem_ids(self) -> List[str]:
        """Get list of unique problem IDs."""
        return list(set(r.problem_id for r in self.results))
    
    def filter_by_model(self, model_name: str) -> 'EvaluationResults':
        """Filter results by model name."""
        filtered_results = [r for r in self.results if r.model_name == model_name]
        return EvaluationResults(
            results=filtered_results,
            config=self.config,
            transformation_sets=self.transformation_sets,
            metadata=self.metadata
        )
    
    def filter_by_transformation(self, transformation_type: str) -> 'EvaluationResults':
        """Filter results by transformation type."""
        filtered_results = [r for r in self.results if r.transformation_type == transformation_type]
        return EvaluationResults(
            results=filtered_results,
            config=self.config,
            transformation_sets=self.transformation_sets,
            metadata=self.metadata
        )
    
    def get_success_rate(self, model_name: Optional[str] = None, transformation_type: Optional[str] = None) -> float:
        """Get success rate for specified filters."""
        filtered_results = self.results
        
        if model_name:
            filtered_results = [r for r in filtered_results if r.model_name == model_name]
        
        if transformation_type:
            filtered_results = [r for r in filtered_results if r.transformation_type == transformation_type]
        
        if not filtered_results:
            return 0.0
        
        successful = sum(1 for r in filtered_results if r.success)
        return successful / len(filtered_results)


class ResultsManager:
    """
    Manager for evaluation results with storage, loading, and analysis capabilities.
    """
    
    def __init__(
        self,
        results_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the results manager.
        
        Args:
            results_dir: Directory to store results
            logger: Logger instance
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def save_results(
        self,
        results: EvaluationResults,
        experiment_name: str,
        format: str = "both"  # "json", "parquet", or "both"
    ) -> Dict[str, Path]:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results to save
            experiment_name: Name of the experiment
            format: Output format(s)
            
        Returns:
            Dictionary mapping format names to file paths
        """
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save configuration
        config_path = experiment_dir / "config.yaml"
        results.config.save_to_file(config_path)
        saved_files["config"] = config_path
        
        # Save metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(results.metadata, f, indent=2)
        saved_files["metadata"] = metadata_path
        
        # Save results in requested formats
        if format in ["json", "both"]:
            json_path = experiment_dir / "results.json"
            self._save_results_json(results, json_path)
            saved_files["json"] = json_path
        
        if format in ["parquet", "both"]:
            parquet_path = experiment_dir / "results.parquet"
            self._save_results_parquet(results, parquet_path)
            saved_files["parquet"] = parquet_path
        
        # Save transformation sets if available
        if results.transformation_sets:
            transformations_path = experiment_dir / "transformations.json"
            self._save_transformation_sets(results.transformation_sets, transformations_path)
            saved_files["transformations"] = transformations_path
        
        # Save summary statistics
        summary_path = experiment_dir / "summary.json"
        summary = self._generate_summary(results)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files["summary"] = summary_path
        
        self.logger.info(f"Saved results for experiment '{experiment_name}' to {experiment_dir}")
        return saved_files
    
    def load_results(self, experiment_name: str) -> EvaluationResults:
        """
        Load evaluation results from experiment directory.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Loaded evaluation results
        """
        experiment_dir = self.results_dir / experiment_name
        
        if not experiment_dir.exists():
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Load configuration
        config_path = experiment_dir / "config.yaml"
        config = EvaluationConfig.load_from_file(config_path)
        
        # Load metadata
        metadata_path = experiment_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load results (prefer parquet for performance)
        results = []
        parquet_path = experiment_dir / "results.parquet"
        json_path = experiment_dir / "results.json"
        
        if parquet_path.exists():
            results = self._load_results_parquet(parquet_path)
        elif json_path.exists():
            results = self._load_results_json(json_path)
        else:
            raise ValueError(f"No results files found for experiment '{experiment_name}'")
        
        # Load transformation sets if available
        transformation_sets = None
        transformations_path = experiment_dir / "transformations.json"
        if transformations_path.exists():
            transformation_sets = self._load_transformation_sets(transformations_path)
        
        return EvaluationResults(
            results=results,
            config=config,
            transformation_sets=transformation_sets,
            metadata=metadata
        )
    
    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        if not self.results_dir.exists():
            return []
        
        experiments = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and (item / "config.yaml").exists():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get summary for an experiment without loading full results."""
        experiment_dir = self.results_dir / experiment_name
        summary_path = experiment_dir / "summary.json"
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
        else:
            # Generate summary from results
            results = self.load_results(experiment_name)
            return self._generate_summary(results)
    
    def compare_experiments(self, experiment_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for exp_name in experiment_names:
            try:
                summary = self.get_experiment_summary(exp_name)
                comparison_data.append({
                    "experiment": exp_name,
                    **summary
                })
            except Exception as e:
                self.logger.error(f"Error loading experiment {exp_name}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def _save_results_json(self, results: EvaluationResults, path: Path) -> None:
        """Save results to JSON format."""
        data = {
            "config": results.config.dict(),
            "metadata": results.metadata,
            "results": []
        }
        
        for result in results.results:
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
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_results_parquet(self, results: EvaluationResults, path: Path) -> None:
        """Save results to Parquet format."""
        df = results.to_dataframe()
        df.to_parquet(path, index=False)
    
    def _load_results_json(self, path: Path) -> List[EvaluationResult]:
        """Load results from JSON format."""
        with open(path, 'r') as f:
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
        
        return results
    
    def _load_results_parquet(self, path: Path) -> List[EvaluationResult]:
        """Load results from Parquet format."""
        df = pd.read_parquet(path)
        results = []
        
        for _, row in df.iterrows():
            result = EvaluationResult(
                problem_id=row["problem_id"],
                transformation_type=row["transformation_type"],
                model_name=row["model_name"],
                original_problem={},  # Not stored in parquet
                transformed_problem=None,
                model_response=row["model_response"],
                response_metadata={
                    "response_time": row.get("response_time", 0),
                    "total_tokens": row.get("total_tokens", 0),
                    "prompt_tokens": row.get("prompt_tokens", 0),
                    "completion_tokens": row.get("completion_tokens", 0),
                    "finish_reason": row.get("finish_reason", "")
                },
                success=row["success"],
                error=row.get("error"),
                evaluation_time=row["evaluation_time"]
            )
            results.append(result)
        
        return results
    
    def _save_transformation_sets(self, transformation_sets: List[TransformationSet], path: Path) -> None:
        """Save transformation sets to JSON."""
        # This would require importing TransformationPipeline
        # For now, create a simplified version
        data = {
            "transformation_sets": [
                {
                    "problem_id": ts.problem_id,
                    "original_problem": ts.original_problem,
                    "transformations": [
                        {
                            "transformation_type": t.transformation_type,
                            "success": t.success,
                            "error": t.error
                        }
                        for t in ts.transformations
                    ]
                }
                for ts in transformation_sets
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_transformation_sets(self, path: Path) -> List[TransformationSet]:
        """Load transformation sets from JSON."""
        # Simplified loading - would need full TransformationSet reconstruction
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Return empty list for now - full implementation would reconstruct TransformationSet objects
        return []
    
    def _generate_summary(self, results: EvaluationResults) -> Dict[str, Any]:
        """Generate summary statistics for results."""
        df = results.to_dataframe()
        
        summary = {
            "experiment_info": {
                "total_results": len(results.results),
                "successful_results": df['success'].sum(),
                "overall_success_rate": df['success'].mean(),
                "models": results.get_model_names(),
                "transformation_types": results.get_transformation_types(),
                "unique_problems": len(results.get_problem_ids())
            },
            "performance_metrics": {
                "avg_evaluation_time": df['evaluation_time'].mean(),
                "avg_response_time": df['response_time'].mean(),
                "total_tokens_used": df['total_tokens'].sum(),
                "avg_tokens_per_request": df['total_tokens'].mean()
            },
            "model_performance": {},
            "transformation_performance": {}
        }
        
        # Model-specific performance
        for model in results.get_model_names():
            model_df = df[df['model_name'] == model]
            summary["model_performance"][model] = {
                "success_rate": model_df['success'].mean(),
                "total_requests": len(model_df),
                "avg_response_time": model_df['response_time'].mean(),
                "total_tokens": model_df['total_tokens'].sum()
            }
        
        # Transformation-specific performance
        for transform_type in results.get_transformation_types():
            transform_df = df[df['transformation_type'] == transform_type]
            summary["transformation_performance"][transform_type] = {
                "success_rate": transform_df['success'].mean(),
                "total_requests": len(transform_df),
                "avg_response_time": transform_df['response_time'].mean()
            }
        
        return summary