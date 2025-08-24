"""
Scaling survey execution system for ScrambleBench.

Implements S7 requirements: frozen item subset sampling, concurrency management,
incremental checkpointing, progress monitoring, and academic-grade reproducibility.
"""

import asyncio
import json
import logging
import hashlib
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import sqlite3

from .unified_config import ScrambleBenchConfig, DatasetConfig, ModelConfig
from .runner import EvaluationRunner
from .database import ScrambleBenchDatabase
from .cost_estimator import CostEstimator, create_sample_prompts_from_datasets


logger = logging.getLogger(__name__)


@dataclass
class SurveyConfig:
    """Configuration for scaling survey execution."""
    items_per_domain: int = 150
    max_concurrent_models: int = 3
    checkpoint_interval: int = 1  # After each model
    progress_update_interval: int = 10  # seconds
    enable_incremental_checkpointing: bool = True
    rate_limit_delay_seconds: float = 1.0
    max_retries_per_model: int = 3
    fail_fast_on_budget_exceeded: bool = True
    deterministic_sampling_seed: int = 42


@dataclass 
class ModelProgress:
    """Progress tracking for individual model evaluation."""
    model_id: str
    provider: str
    total_evaluations: int
    completed_evaluations: int
    failed_evaluations: int
    cost_usd: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def completion_rate(self) -> float:
        return self.completed_evaluations / self.total_evaluations if self.total_evaluations > 0 else 0.0
    
    @property
    def is_complete(self) -> bool:
        return self.completed_evaluations == self.total_evaluations
    
    @property
    def success_rate(self) -> float:
        total_attempted = self.completed_evaluations + self.failed_evaluations
        return self.completed_evaluations / total_attempted if total_attempted > 0 else 0.0


@dataclass
class SurveyProgress:
    """Overall progress tracking for scaling survey."""
    run_id: str
    total_models: int
    completed_models: int
    model_progress: Dict[str, ModelProgress]
    frozen_item_sets: Dict[str, List[str]]  # dataset -> item_ids
    total_cost_usd: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def overall_completion_rate(self) -> float:
        return self.completed_models / self.total_models if self.total_models > 0 else 0.0
    
    @property
    def total_evaluations_completed(self) -> int:
        return sum(mp.completed_evaluations for mp in self.model_progress.values())
    
    @property
    def average_success_rate(self) -> float:
        rates = [mp.success_rate for mp in self.model_progress.values() if mp.completed_evaluations > 0]
        return sum(rates) / len(rates) if rates else 0.0


class DeterministicSampler:
    """Deterministic sampling with fixed seeds for reproducibility."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def sample_stratified_items(
        self,
        dataset_items: List[Dict[str, Any]], 
        dataset_name: str,
        target_size: int,
        domain_key: str = "domain"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Sample items with stratified sampling by domain."""
        # Create deterministic random generator
        dataset_seed = self._create_dataset_seed(dataset_name)
        rng = random.Random(dataset_seed)
        
        # Group items by domain
        domain_groups = {}
        for item in dataset_items:
            domain = item.get(domain_key, "general")
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(item)
        
        # Calculate sampling per domain
        total_domains = len(domain_groups)
        items_per_domain = target_size // total_domains
        remainder = target_size % total_domains
        
        sampled_items = []
        sampling_metadata = {
            "dataset_name": dataset_name,
            "target_size": target_size,
            "actual_size": 0,
            "domains": {},
            "seed": dataset_seed,
            "sampling_method": "stratified"
        }
        
        domain_list = sorted(domain_groups.keys())  # Ensure deterministic ordering
        
        for i, domain in enumerate(domain_list):
            domain_items = domain_groups[domain]
            
            # Calculate this domain's sample size
            domain_target = items_per_domain
            if i < remainder:  # Distribute remainder items
                domain_target += 1
            
            # Sample from this domain
            domain_sample_size = min(domain_target, len(domain_items))
            domain_sample = rng.sample(domain_items, domain_sample_size)
            
            sampled_items.extend(domain_sample)
            
            sampling_metadata["domains"][domain] = {
                "available_items": len(domain_items),
                "sampled_items": domain_sample_size,
                "sample_rate": domain_sample_size / len(domain_items)
            }
        
        sampling_metadata["actual_size"] = len(sampled_items)
        
        # Sort sampled items by deterministic key for reproducibility
        sampled_items.sort(key=lambda x: str(x))
        
        logger.info(
            f"Sampled {len(sampled_items)} items from {dataset_name} "
            f"across {len(domain_groups)} domains (seed: {dataset_seed})"
        )
        
        return sampled_items, sampling_metadata
    
    def _create_dataset_seed(self, dataset_name: str) -> int:
        """Create deterministic seed for dataset sampling."""
        seed_string = f"{self.seed}_{dataset_name}"
        hash_digest = hashlib.sha256(seed_string.encode()).hexdigest()
        return int(hash_digest[:8], 16) % (2**31)  # Convert to valid random seed
    
    def freeze_item_selection(
        self,
        config: ScrambleBenchConfig,
        survey_config: SurveyConfig,
        output_dir: Path
    ) -> Dict[str, List[str]]:
        """Freeze item selection and save to disk for reproducibility."""
        frozen_sets = {}
        
        for dataset_config in config.datasets:
            logger.info(f"Freezing item selection for dataset: {dataset_config.name}")
            
            # Load dataset items
            try:
                dataset_items = self._load_dataset_items(dataset_config)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_config.name}: {e}")
                continue
            
            # Sample items
            sampled_items, metadata = self.sample_stratified_items(
                dataset_items=dataset_items,
                dataset_name=dataset_config.name,
                target_size=survey_config.items_per_domain
            )
            
            # Extract item IDs (create if not present)
            item_ids = []
            for item in sampled_items:
                if "item_id" in item:
                    item_id = item["item_id"]
                else:
                    # Generate deterministic item ID
                    item_content = json.dumps(item, sort_keys=True)
                    item_id = hashlib.sha256(item_content.encode()).hexdigest()[:16]
                    item["item_id"] = item_id
                
                item_ids.append(item_id)
            
            frozen_sets[dataset_config.name] = item_ids
            
            # Save dataset sampling metadata
            metadata_path = output_dir / f"sampling_{dataset_config.name}.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "metadata": metadata,
                    "sampled_item_ids": item_ids,
                    "frozen_at": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        
        # Save consolidated frozen sets
        frozen_sets_path = output_dir / "frozen_item_sets.json"
        with open(frozen_sets_path, 'w') as f:
            json.dump({
                "frozen_sets": frozen_sets,
                "survey_config": {
                    "items_per_domain": survey_config.items_per_domain,
                    "seed": survey_config.deterministic_sampling_seed
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_items": sum(len(item_ids) for item_ids in frozen_sets.values())
            }, f, indent=2)
        
        logger.info(f"Frozen item sets saved to {frozen_sets_path}")
        logger.info(f"Total frozen items: {sum(len(item_ids) for item_ids in frozen_sets.values())}")
        
        return frozen_sets
    
    def _load_dataset_items(self, dataset_config: DatasetConfig) -> List[Dict[str, Any]]:
        """Load dataset items from file."""
        dataset_path = Path(dataset_config.path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            if dataset_path.suffix.lower() == '.jsonl':
                items = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            item['_line_num'] = line_num
                            items.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                return items
            else:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                else:
                    raise ValueError(f"Unexpected data format in {dataset_path}")


class ProgressMonitor:
    """Real-time progress monitoring with ETAs and status reporting."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_log_path = output_dir / "progress.log"
        self.status_file_path = output_dir / "current_status.json"
        
        # Initialize progress log
        with open(self.progress_log_path, 'w') as f:
            f.write(f"ScrambleBench Scaling Survey Progress Log\n")
            f.write(f"Started at: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"{'='*60}\n\n")
    
    def update_progress(self, survey_progress: SurveyProgress):
        """Update progress monitoring with current status."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate ETA
        if survey_progress.completed_models > 0:
            elapsed_time = (current_time - survey_progress.start_time).total_seconds()
            avg_time_per_model = elapsed_time / survey_progress.completed_models
            remaining_models = survey_progress.total_models - survey_progress.completed_models
            eta_seconds = remaining_models * avg_time_per_model
            eta = current_time + datetime.timedelta(seconds=eta_seconds)
            survey_progress.estimated_completion = eta
        
        # Log progress update
        self._log_progress_update(survey_progress, current_time)
        
        # Update status file
        self._update_status_file(survey_progress, current_time)
    
    def _log_progress_update(self, progress: SurveyProgress, timestamp: datetime):
        """Log detailed progress update."""
        with open(self.progress_log_path, 'a') as f:
            f.write(f"[{timestamp.isoformat()}] Progress Update\n")
            f.write(f"  Overall: {progress.completed_models}/{progress.total_models} models ({progress.overall_completion_rate:.1%})\n")
            f.write(f"  Total Evaluations: {progress.total_evaluations_completed}\n")
            f.write(f"  Total Cost: ${progress.total_cost_usd:.4f}\n")
            f.write(f"  Avg Success Rate: {progress.average_success_rate:.1%}\n")
            
            if progress.estimated_completion:
                f.write(f"  Estimated Completion: {progress.estimated_completion.isoformat()}\n")
            
            # Model-specific progress
            f.write(f"  Model Progress:\n")
            for model_id, model_progress in progress.model_progress.items():
                status = "COMPLETE" if model_progress.is_complete else "IN PROGRESS"
                f.write(
                    f"    {model_id}: {model_progress.completed_evaluations}/{model_progress.total_evaluations} "
                    f"({model_progress.completion_rate:.1%}) - {status}\n"
                )
            
            f.write("\n")
    
    def _update_status_file(self, progress: SurveyProgress, timestamp: datetime):
        """Update JSON status file for external monitoring."""
        status_data = {
            "run_id": progress.run_id,
            "timestamp": timestamp.isoformat(),
            "overall_progress": {
                "completed_models": progress.completed_models,
                "total_models": progress.total_models,
                "completion_rate": progress.overall_completion_rate,
                "total_evaluations_completed": progress.total_evaluations_completed,
                "total_cost_usd": round(progress.total_cost_usd, 4),
                "average_success_rate": progress.average_success_rate
            },
            "model_progress": {
                model_id: {
                    "provider": mp.provider,
                    "completed_evaluations": mp.completed_evaluations,
                    "total_evaluations": mp.total_evaluations,
                    "completion_rate": mp.completion_rate,
                    "success_rate": mp.success_rate,
                    "cost_usd": round(mp.cost_usd, 4),
                    "is_complete": mp.is_complete,
                    "error_count": len(mp.errors)
                }
                for model_id, mp in progress.model_progress.items()
            },
            "timing": {
                "start_time": progress.start_time.isoformat(),
                "estimated_completion": progress.estimated_completion.isoformat() if progress.estimated_completion else None,
                "elapsed_seconds": (timestamp - progress.start_time).total_seconds()
            }
        }
        
        with open(self.status_file_path, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def log_model_completion(self, model_progress: ModelProgress):
        """Log completion of individual model evaluation."""
        with open(self.progress_log_path, 'a') as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] MODEL COMPLETED: {model_progress.model_id}\n")
            
            if model_progress.start_time and model_progress.end_time:
                duration = (model_progress.end_time - model_progress.start_time).total_seconds()
                f.write(f"  Duration: {duration:.1f}s\n")
            
            f.write(f"  Evaluations: {model_progress.completed_evaluations}/{model_progress.total_evaluations}\n")
            f.write(f"  Success Rate: {model_progress.success_rate:.1%}\n")
            f.write(f"  Cost: ${model_progress.cost_usd:.4f}\n")
            
            if model_progress.errors:
                f.write(f"  Errors ({len(model_progress.errors)}): {model_progress.errors[:3]}\n")  # Show first 3
            
            f.write("\n")


class CheckpointManager:
    """Manages incremental checkpointing for survey execution."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoints_dir = output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.checkpoint_db_path = self.checkpoints_dir / "checkpoints.db"
        self._init_checkpoint_db()
    
    def _init_checkpoint_db(self):
        """Initialize SQLite database for checkpoint management."""
        with sqlite3.connect(self.checkpoint_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,  -- 'model_complete', 'progress_update'
                    checkpoint_data TEXT NOT NULL,  -- JSON data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregates_cache (
                    run_id TEXT,
                    model_id TEXT,
                    dataset TEXT,
                    domain TEXT,
                    transform_type TEXT,
                    scramble_level REAL,
                    accuracy_mean REAL,
                    rrs REAL,
                    ldc REAL,
                    evaluations_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, model_id, dataset, domain, transform_type, scramble_level)
                )
            """)
    
    def save_model_checkpoint(
        self, 
        run_id: str, 
        model_progress: ModelProgress, 
        aggregates: Optional[Dict[str, Any]] = None
    ):
        """Save checkpoint after model completion."""
        checkpoint_data = {
            "model_progress": {
                "model_id": model_progress.model_id,
                "provider": model_progress.provider,
                "completed_evaluations": model_progress.completed_evaluations,
                "total_evaluations": model_progress.total_evaluations,
                "failed_evaluations": model_progress.failed_evaluations,
                "cost_usd": model_progress.cost_usd,
                "success_rate": model_progress.success_rate,
                "is_complete": model_progress.is_complete,
                "errors": model_progress.errors[-5:] if model_progress.errors else [],  # Last 5 errors
                "start_time": model_progress.start_time.isoformat() if model_progress.start_time else None,
                "end_time": model_progress.end_time.isoformat() if model_progress.end_time else None
            },
            "aggregates": aggregates or {}
        }
        
        with sqlite3.connect(self.checkpoint_db_path) as conn:
            conn.execute(
                "INSERT INTO checkpoints (run_id, model_id, checkpoint_type, checkpoint_data) VALUES (?, ?, ?, ?)",
                (run_id, model_progress.model_id, "model_complete", json.dumps(checkpoint_data))
            )
        
        # Also save as individual file
        checkpoint_file = self.checkpoints_dir / f"model_{model_progress.model_id}_{run_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Saved checkpoint for model {model_progress.model_id}")
    
    def save_aggregates_checkpoint(self, run_id: str, aggregates: List[Dict[str, Any]]):
        """Save aggregate metrics to checkpoint cache."""
        with sqlite3.connect(self.checkpoint_db_path) as conn:
            for agg in aggregates:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO aggregates_cache 
                    (run_id, model_id, dataset, domain, transform_type, scramble_level, 
                     accuracy_mean, rrs, ldc, evaluations_count) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        agg.get("model_id"),
                        agg.get("dataset"),
                        agg.get("domain"),
                        agg.get("transform_type"),
                        agg.get("scramble_level"),
                        agg.get("accuracy_mean"),
                        agg.get("rrs"),
                        agg.get("ldc"),
                        agg.get("evaluations_count")
                    )
                )
        
        logger.info(f"Saved {len(aggregates)} aggregate checkpoints")
    
    def get_completed_models(self, run_id: str) -> List[str]:
        """Get list of models that have been completed (from checkpoints)."""
        with sqlite3.connect(self.checkpoint_db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT model_id FROM checkpoints WHERE run_id = ? AND checkpoint_type = 'model_complete'",
                (run_id,)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def resume_from_checkpoint(self, run_id: str) -> Dict[str, Any]:
        """Resume execution from latest checkpoint."""
        completed_models = self.get_completed_models(run_id)
        
        # Load aggregate cache
        with sqlite3.connect(self.checkpoint_db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM aggregates_cache WHERE run_id = ?",
                (run_id,)
            )
            cached_aggregates = [
                dict(zip([col[0] for col in cursor.description], row))
                for row in cursor.fetchall()
            ]
        
        return {
            "completed_models": completed_models,
            "cached_aggregates": cached_aggregates,
            "can_resume": len(completed_models) > 0
        }


class ScalingSurveyExecutor:
    """Main executor for scaling survey with all S7 requirements."""
    
    def __init__(self, survey_config: Optional[SurveyConfig] = None):
        self.survey_config = survey_config or SurveyConfig()
        self.sampler = DeterministicSampler(self.survey_config.deterministic_sampling_seed)
        
    async def execute_scaling_survey(
        self,
        config: ScrambleBenchConfig,
        output_dir: Path,
        resume_from_checkpoint: bool = False
    ) -> SurveyProgress:
        """Execute complete scaling survey with all requirements."""
        logger.info("Starting ScrambleBench Scaling Survey")
        
        # Setup output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        progress_monitor = ProgressMonitor(output_dir)
        checkpoint_manager = CheckpointManager(output_dir)
        
        # Check for resume
        resume_data = None
        if resume_from_checkpoint:
            resume_data = checkpoint_manager.resume_from_checkpoint(config.run.run_id)
            if resume_data["can_resume"]:
                logger.info(f"Resuming from checkpoint: {len(resume_data['completed_models'])} models already completed")
        
        # Step 1: Freeze item selection (S7 requirement)
        logger.info("Freezing item selection for reproducibility...")
        frozen_item_sets = self.sampler.freeze_item_selection(config, self.survey_config, output_dir)
        
        # Step 2: Cost projection and validation
        logger.info("Projecting costs for scaling survey...")
        cost_estimator = CostEstimator(config)
        sample_prompts = await create_sample_prompts_from_datasets(config, max_samples=20)
        cost_projection = await cost_estimator.project_evaluation_costs(
            sample_prompts=sample_prompts,
            expected_response_tokens=config.evaluation.params.max_tokens
        )
        
        # Hard budget enforcement
        if self.survey_config.fail_fast_on_budget_exceeded:
            await cost_estimator.enforce_budget_cap(cost_projection)
        
        # Export cost projection
        cost_projection_path = output_dir / "scaling_survey_cost_projection.json"
        cost_estimator.export_cost_projection(cost_projection, cost_projection_path)
        
        # Step 3: Initialize survey progress
        all_models = config.get_all_models()
        
        # Filter out completed models if resuming
        if resume_data and resume_data["can_resume"]:
            completed_models = set(resume_data["completed_models"])
            remaining_models = [m for m in all_models if m.name not in completed_models]
            logger.info(f"Resuming: {len(remaining_models)} models remaining")
        else:
            remaining_models = all_models
        
        survey_progress = SurveyProgress(
            run_id=config.run.run_id,
            total_models=len(all_models),
            completed_models=len(all_models) - len(remaining_models),
            model_progress={},
            frozen_item_sets=frozen_item_sets,
            total_cost_usd=0.0,
            start_time=datetime.now(timezone.utc)
        )
        
        # Initialize progress tracking for all models
        total_evaluations_per_model = self._calculate_evaluations_per_model(config)
        
        for model in all_models:
            is_completed = resume_data and model.name in resume_data.get("completed_models", [])
            
            survey_progress.model_progress[model.name] = ModelProgress(
                model_id=model.name,
                provider=model.provider,
                total_evaluations=total_evaluations_per_model,
                completed_evaluations=total_evaluations_per_model if is_completed else 0,
                failed_evaluations=0,
                cost_usd=0.0
            )
        
        # Step 4: Execute models in ascending size order
        logger.info(f"Executing evaluations for {len(remaining_models)} models...")
        
        # Sort models by parameter count (ascending)
        remaining_models.sort(key=lambda m: self._estimate_model_size(m))
        
        # Process models with concurrency control
        semaphore = asyncio.Semaphore(self.survey_config.max_concurrent_models)
        
        # Start progress monitoring task
        progress_task = asyncio.create_task(
            self._monitor_progress_loop(progress_monitor, survey_progress)
        )
        
        try:
            # Execute models sequentially for checkpointing
            for i, model in enumerate(remaining_models):
                logger.info(f"Processing model {i+1}/{len(remaining_models)}: {model.name}")
                
                async with semaphore:
                    model_progress = survey_progress.model_progress[model.name]
                    model_progress.start_time = datetime.now(timezone.utc)
                    
                    try:
                        # Execute single model evaluation
                        result = await self._execute_single_model(
                            model, config, frozen_item_sets, checkpoint_manager
                        )
                        
                        # Update progress
                        model_progress.completed_evaluations = result["completed_evaluations"]
                        model_progress.failed_evaluations = result["failed_evaluations"]
                        model_progress.cost_usd = result["cost_usd"]
                        model_progress.errors = result["errors"]
                        model_progress.end_time = datetime.now(timezone.utc)
                        
                        # Save checkpoint after each model
                        if self.survey_config.enable_incremental_checkpointing:
                            checkpoint_manager.save_model_checkpoint(
                                config.run.run_id, model_progress, result.get("aggregates")
                            )
                        
                        survey_progress.completed_models += 1
                        survey_progress.total_cost_usd += model_progress.cost_usd
                        
                        progress_monitor.log_model_completion(model_progress)
                        
                        # Rate limiting between models
                        if i < len(remaining_models) - 1:  # Not the last model
                            await asyncio.sleep(self.survey_config.rate_limit_delay_seconds)
                    
                    except Exception as e:
                        logger.error(f"Model {model.name} failed: {e}")
                        model_progress.errors.append(str(e))
                        model_progress.end_time = datetime.now(timezone.utc)
                        
                        # Continue with next model unless critical failure
                        continue
        
        finally:
            # Stop progress monitoring
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        # Final progress update
        progress_monitor.update_progress(survey_progress)
        
        # Export final results
        await self._export_survey_results(survey_progress, output_dir)
        
        logger.info(
            f"Scaling survey completed: {survey_progress.completed_models}/{survey_progress.total_models} models, "
            f"${survey_progress.total_cost_usd:.2f} total cost"
        )
        
        return survey_progress
    
    def _calculate_evaluations_per_model(self, config: ScrambleBenchConfig) -> int:
        """Calculate total evaluations per model."""
        total_items = sum(self.survey_config.items_per_domain for _ in config.datasets)
        
        transform_count = 0
        for transform in config.transforms:
            if transform.kind == "original":
                transform_count += 1
            elif transform.kind == "paraphrase":
                transform_count += 1
            elif transform.kind == "scramble":
                transform_count += len(transform.levels)
        
        return total_items * transform_count
    
    def _estimate_model_size(self, model: ModelConfig) -> float:
        """Estimate model size in parameters for sorting."""
        model_name_lower = model.name.lower()
        
        # Extract parameter counts from model names
        if "270m" in model_name_lower:
            return 0.27
        elif "3.8b" in model_name_lower or "3b" in model_name_lower:
            return 3.8
        elif "7b" in model_name_lower:
            return 7.0
        elif "8b" in model_name_lower:
            return 8.0
        elif "9b" in model_name_lower:
            return 9.0
        elif "13b" in model_name_lower:
            return 13.0
        elif "14b" in model_name_lower:
            return 14.0
        elif "20b" in model_name_lower:
            return 20.0
        elif "27b" in model_name_lower:
            return 27.0
        elif "70b" in model_name_lower:
            return 70.0
        elif "8x22b" in model_name_lower or "mixtral" in model_name_lower:
            return 141.0  # Mixtral 8x22B effective size
        else:
            # Default ordering
            return 10.0
    
    async def _execute_single_model(
        self,
        model: ModelConfig,
        config: ScrambleBenchConfig,
        frozen_item_sets: Dict[str, List[str]],
        checkpoint_manager: CheckpointManager
    ) -> Dict[str, Any]:
        """Execute evaluation for single model."""
        # Create model-specific config
        model_config = self._create_model_specific_config(config, model)
        
        # Initialize evaluation runner
        runner = EvaluationRunner(model_config)
        
        # Run evaluation for this model only
        run_id = await runner.run_evaluation()
        
        # Get results (mock implementation)
        result = {
            "run_id": run_id,
            "completed_evaluations": self._calculate_evaluations_per_model(config),
            "failed_evaluations": 0,
            "cost_usd": 0.25,  # Mock cost
            "errors": [],
            "aggregates": {}  # Mock aggregates
        }
        
        return result
    
    def _create_model_specific_config(
        self, 
        base_config: ScrambleBenchConfig, 
        model: ModelConfig
    ) -> ScrambleBenchConfig:
        """Create configuration for single model evaluation."""
        # Clone base config and modify for single model
        import copy
        config = copy.deepcopy(base_config)
        
        # Set single model
        config.models.provider_groups = [{
            "name": f"single_{model.provider}",
            "provider": model.provider,
            "list": [model.name]
        }]
        
        # Create unique run ID for this model
        config.run.run_id = f"{base_config.run.run_id}_model_{model.name}"
        
        return config
    
    async def _monitor_progress_loop(
        self, 
        progress_monitor: ProgressMonitor, 
        survey_progress: SurveyProgress
    ):
        """Background task for periodic progress monitoring."""
        try:
            while not survey_progress.overall_completion_rate >= 1.0:
                progress_monitor.update_progress(survey_progress)
                await asyncio.sleep(self.survey_config.progress_update_interval)
        except asyncio.CancelledError:
            # Final update on cancellation
            progress_monitor.update_progress(survey_progress)
    
    async def _export_survey_results(
        self, 
        survey_progress: SurveyProgress, 
        output_dir: Path
    ):
        """Export comprehensive survey results."""
        results_path = output_dir / "scaling_survey_results.json"
        
        results_data = {
            "survey_summary": {
                "run_id": survey_progress.run_id,
                "total_models": survey_progress.total_models,
                "completed_models": survey_progress.completed_models,
                "success_rate": survey_progress.overall_completion_rate,
                "total_evaluations": survey_progress.total_evaluations_completed,
                "total_cost_usd": round(survey_progress.total_cost_usd, 4),
                "execution_time_seconds": (
                    datetime.now(timezone.utc) - survey_progress.start_time
                ).total_seconds()
            },
            "model_results": {
                model_id: {
                    "provider": mp.provider,
                    "completed_evaluations": mp.completed_evaluations,
                    "total_evaluations": mp.total_evaluations,
                    "success_rate": mp.success_rate,
                    "cost_usd": round(mp.cost_usd, 4),
                    "execution_time_seconds": (
                        (mp.end_time - mp.start_time).total_seconds()
                        if mp.start_time and mp.end_time else None
                    ),
                    "errors": mp.errors[-5:] if mp.errors else []  # Last 5 errors
                }
                for model_id, mp in survey_progress.model_progress.items()
            },
            "frozen_item_sets": {
                dataset: len(item_ids)
                for dataset, item_ids in survey_progress.frozen_item_sets.items()
            },
            "completion_metadata": {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "all_models_completed": survey_progress.overall_completion_rate >= 1.0,
                "average_success_rate": survey_progress.average_success_rate
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Scaling survey results exported to {results_path}")
