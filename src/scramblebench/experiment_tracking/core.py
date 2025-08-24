"""
Core Experiment Tracking System

Central orchestrator for managing large-scale experiments with full academic 
research support including reproducibility, statistical analysis, and monitoring.
"""

import asyncio
import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from uuid import uuid4, UUID

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..core.unified_config import ScrambleBenchConfig
from ..evaluation.runner import EvaluationRunner, EvaluationResults
from .queue import ExperimentQueue, QueuedExperiment
from .monitor import ExperimentMonitor, ProgressTracker
from .statistics import StatisticalAnalyzer
from .reproducibility import ReproducibilityValidator
from .database import DatabaseManager


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PLANNED = "planned"
    QUEUED = "queued" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ExperimentMetadata:
    """Complete metadata for an experiment"""
    experiment_id: str
    name: str
    description: str
    research_question: str
    hypothesis: Optional[str]
    researcher_name: str
    institution: Optional[str]
    
    # Reproducibility
    git_commit_hash: str
    git_branch: str
    environment_snapshot: Dict[str, Any]
    random_seed: Optional[int]
    config_hash: str
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_duration: Optional[timedelta]
    
    # Status and progress
    status: ExperimentStatus
    progress: float = 0.0
    current_stage: str = "initialized"
    
    # Resource tracking
    total_api_calls: int = 0
    total_cost: float = 0.0
    compute_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Handle datetime serialization
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field] is not None:
                data[field] = data[field].isoformat()
        if data['estimated_duration'] is not None:
            data['estimated_duration'] = data['estimated_duration'].total_seconds()
        data['status'] = data['status'].value if isinstance(data['status'], ExperimentStatus) else data['status']
        return data


class ExperimentTracker:
    """
    Central experiment tracking and orchestration system
    
    Manages the complete lifecycle of research experiments from planning
    through execution, monitoring, and analysis with full reproducibility
    tracking and academic research support.
    """
    
    def __init__(
        self,
        database_url: str,
        data_dir: Optional[Path] = None,
        max_concurrent_experiments: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the experiment tracker
        
        Args:
            database_url: PostgreSQL database connection URL
            data_dir: Directory for experiment data and outputs  
            max_concurrent_experiments: Maximum concurrent experiments
            logger: Logger instance
        """
        self.database_url = database_url
        self.data_dir = data_dir or Path("experiments")
        self.max_concurrent = max_concurrent_experiments
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.db_manager = DatabaseManager(database_url, logger=self.logger)
        self.queue = ExperimentQueue(max_concurrent_experiments)
        self.monitor = ExperimentMonitor(self.db_manager)
        self.stats_analyzer = StatisticalAnalyzer(self.db_manager)
        self.reproducibility = ReproducibilityValidator()
        
        # Runtime state
        self.running_experiments: Dict[str, asyncio.Task] = {}
        self._shutdown_requested = False
        
        self.logger.info(f"ExperimentTracker initialized with database: {database_url}")
    
    async def create_experiment(
        self,
        name: str,
        config: Union[EvaluationConfig, Dict[str, Any], Path],
        description: str,
        research_question: str,
        researcher_name: str,
        hypothesis: Optional[str] = None,
        institution: Optional[str] = None,
        priority: int = 1,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new experiment with full metadata tracking
        
        Args:
            name: Experiment name
            config: Evaluation configuration  
            description: Detailed description
            research_question: Primary research question
            researcher_name: Researcher conducting the experiment
            hypothesis: Optional hypothesis being tested
            institution: Research institution
            priority: Queue priority (higher = more important)
            tags: Optional tags for organization
            
        Returns:
            Unique experiment ID
        """
        experiment_id = str(uuid4())
        
        # Load and validate configuration
        if isinstance(config, Path):
            eval_config = EvaluationConfig.load_from_file(config)
        elif isinstance(config, dict):
            eval_config = EvaluationConfig(**config)
        else:
            eval_config = config
            
        # Generate reproducibility metadata
        env_snapshot = await self.reproducibility.capture_environment()
        git_info = await self.reproducibility.get_git_info()
        config_hash = self._hash_config(eval_config)
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            research_question=research_question,
            hypothesis=hypothesis,
            researcher_name=researcher_name,
            institution=institution,
            git_commit_hash=git_info['commit_hash'],
            git_branch=git_info['branch'],
            environment_snapshot=env_snapshot,
            random_seed=eval_config.sample_seed,
            config_hash=config_hash,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            status=ExperimentStatus.PLANNED
        )
        
        # Save to database
        await self.db_manager.create_experiment(
            metadata=metadata,
            config=eval_config,
            tags=tags or []
        )
        
        # Create experiment directory
        experiment_dir = self.data_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = experiment_dir / "config.yaml"
        eval_config.save_to_file(config_path)
        
        # Save metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        self.logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id
    
    async def queue_experiment(
        self,
        experiment_id: str,
        priority: int = 1,
        depends_on: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add experiment to execution queue
        
        Args:
            experiment_id: Experiment to queue
            priority: Queue priority
            depends_on: List of experiment IDs this depends on
            resource_requirements: Required resources (API limits, compute)
        """
        # Load experiment metadata and config
        metadata = await self.db_manager.get_experiment_metadata(experiment_id)
        if not metadata:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = await self.db_manager.get_experiment_config(experiment_id)
        
        # Estimate resource requirements if not provided
        if resource_requirements is None:
            resource_requirements = await self._estimate_resource_requirements(config)
        
        # Create queued experiment
        queued_experiment = QueuedExperiment(
            experiment_id=experiment_id,
            priority=priority,
            depends_on=depends_on or [],
            resource_requirements=resource_requirements,
            estimated_duration=await self._estimate_duration(config),
            queued_at=datetime.now()
        )
        
        # Add to queue
        await self.queue.add_experiment(queued_experiment)
        
        # Update database status
        await self.db_manager.update_experiment_status(
            experiment_id, ExperimentStatus.QUEUED
        )
        
        self.logger.info(f"Queued experiment {experiment_id} with priority {priority}")
    
    async def run_experiments(
        self,
        continuous: bool = True,
        check_interval: int = 30
    ) -> None:
        """
        Main experiment runner loop
        
        Args:
            continuous: Whether to run continuously or process queue once
            check_interval: Seconds between queue checks
        """
        self.logger.info("Starting experiment runner")
        
        try:
            while not self._shutdown_requested:
                # Check for ready experiments
                ready_experiments = await self.queue.get_ready_experiments(
                    max_count=self.max_concurrent - len(self.running_experiments)
                )
                
                # Start new experiments
                for queued_exp in ready_experiments:
                    await self._start_experiment(queued_exp)
                
                # Clean up completed experiments
                await self._cleanup_completed_experiments()
                
                # Update monitoring data
                await self._update_monitoring_data()
                
                # Break if not continuous
                if not continuous:
                    break
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
        except Exception as e:
            self.logger.error(f"Error in experiment runner: {e}")
            raise
        finally:
            await self._shutdown_all_experiments()
    
    async def _start_experiment(self, queued_experiment: QueuedExperiment) -> None:
        """Start a single experiment execution"""
        experiment_id = queued_experiment.experiment_id
        
        try:
            # Load experiment data
            metadata = await self.db_manager.get_experiment_metadata(experiment_id)
            config = await self.db_manager.get_experiment_config(experiment_id)
            
            # Update status to running
            metadata.status = ExperimentStatus.RUNNING
            metadata.started_at = datetime.now()
            await self.db_manager.update_experiment_metadata(metadata)
            
            # Create progress tracker
            progress_tracker = ProgressTracker(
                experiment_id=experiment_id,
                db_manager=self.db_manager,
                logger=self.logger
            )
            
            # Create evaluation runner
            experiment_dir = self.data_dir / experiment_id
            eval_runner = EvaluationRunner(
                config=config,
                data_dir=experiment_dir,
                logger=self.logger
            )
            
            # Start experiment task
            task = asyncio.create_task(
                self._execute_experiment(
                    experiment_id=experiment_id,
                    eval_runner=eval_runner,
                    progress_tracker=progress_tracker,
                    metadata=metadata
                )
            )
            
            self.running_experiments[experiment_id] = task
            self.logger.info(f"Started experiment {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
            await self.db_manager.update_experiment_status(
                experiment_id, ExperimentStatus.FAILED
            )
    
    async def _execute_experiment(
        self,
        experiment_id: str,
        eval_runner: EvaluationRunner,
        progress_tracker: ProgressTracker,
        metadata: ExperimentMetadata
    ) -> None:
        """Execute a single experiment with monitoring"""
        start_time = time.time()
        
        try:
            # Initialize progress tracking
            await progress_tracker.initialize()
            
            # Run evaluation with progress tracking
            results = await self._run_with_progress_tracking(
                eval_runner, progress_tracker
            )
            
            # Save results to database
            await self.db_manager.save_experiment_results(
                experiment_id, results
            )
            
            # Update completion metadata
            metadata.status = ExperimentStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.progress = 1.0
            metadata.compute_hours = (time.time() - start_time) / 3600.0
            
            await self.db_manager.update_experiment_metadata(metadata)
            
            # Run post-experiment analysis
            await self._run_post_experiment_analysis(experiment_id, results)
            
            self.logger.info(f"Completed experiment {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            
            # Update failure status
            metadata.status = ExperimentStatus.FAILED
            metadata.completed_at = datetime.now()
            await self.db_manager.update_experiment_metadata(metadata)
            
            # Save error details
            await self.db_manager.save_experiment_error(experiment_id, str(e))
            
            raise
    
    async def _run_with_progress_tracking(
        self,
        eval_runner: EvaluationRunner,
        progress_tracker: ProgressTracker
    ) -> EvaluationResults:
        """Run evaluation with detailed progress tracking"""
        # This would integrate with the existing EvaluationRunner
        # to provide real-time progress updates
        
        # For now, we'll run the evaluation and update progress periodically
        eval_task = asyncio.create_task(eval_runner.run_evaluation())
        
        # Monitor progress
        while not eval_task.done():
            status = eval_runner.get_status()
            await progress_tracker.update_progress(
                stage=status['current_stage'],
                progress=self._calculate_progress(status),
                details=status
            )
            
            await asyncio.sleep(10)  # Update every 10 seconds
        
        # Get final results
        results = await eval_task
        
        # Final progress update
        await progress_tracker.update_progress(
            stage="completed",
            progress=1.0,
            details={"status": "completed"}
        )
        
        return results
    
    def _calculate_progress(self, status: Dict[str, Any]) -> float:
        """Calculate experiment progress from runner status"""
        stage = status.get('current_stage', 'initialized')
        
        # Define stage weights for progress calculation
        stage_weights = {
            'initialized': 0.0,
            'loading_data': 0.1,
            'generating_transformations': 0.2,
            'running_evaluations': 0.7,  # Most of the time
            'processing_results': 0.9,
            'saving_results': 0.95,
            'generating_analysis': 0.98,
            'completed': 1.0
        }
        
        return stage_weights.get(stage, 0.0)
    
    async def _run_post_experiment_analysis(
        self,
        experiment_id: str,
        results: EvaluationResults
    ) -> None:
        """Run statistical analysis after experiment completion"""
        try:
            # Generate performance metrics
            await self.stats_analyzer.compute_performance_metrics(
                experiment_id, results
            )
            
            # Detect language dependency thresholds
            await self.stats_analyzer.analyze_thresholds(
                experiment_id, results
            )
            
            # Run significance tests
            if len(results.results) > 30:  # Minimum sample size
                await self.stats_analyzer.run_significance_tests(
                    experiment_id, results
                )
            
            self.logger.info(f"Completed post-experiment analysis for {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Post-experiment analysis failed for {experiment_id}: {e}")
            # Don't fail the entire experiment for analysis errors
    
    async def _estimate_resource_requirements(
        self,
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """Estimate resource requirements for an experiment"""
        # Calculate expected API calls
        num_models = len(config.models)
        num_benchmarks = len(config.benchmark_paths)
        estimated_questions = config.max_samples or 1000  # Conservative estimate
        num_transformations = 7  # Default scrambling levels
        
        total_api_calls = num_models * num_benchmarks * estimated_questions * num_transformations
        
        # Estimate cost based on model types
        estimated_cost = 0.0
        for model in config.models:
            if model.provider.value == 'openrouter':
                estimated_cost += total_api_calls * 0.001  # Rough estimate
            # Local models (Ollama) have no API cost
        
        # Estimate duration
        avg_response_time = 2.0  # seconds per API call
        estimated_duration = total_api_calls * avg_response_time / 3600.0  # hours
        
        return {
            'api_calls': total_api_calls,
            'estimated_cost': estimated_cost,
            'estimated_duration_hours': estimated_duration,
            'memory_gb': 4.0,  # Conservative estimate
            'storage_gb': 1.0   # For results storage
        }
    
    async def _estimate_duration(self, config: EvaluationConfig) -> timedelta:
        """Estimate experiment duration"""
        requirements = await self._estimate_resource_requirements(config)
        hours = requirements['estimated_duration_hours']
        return timedelta(hours=hours)
    
    def _hash_config(self, config: EvaluationConfig) -> str:
        """Generate hash of configuration for reproducibility"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    async def _cleanup_completed_experiments(self) -> None:
        """Clean up completed experiment tasks"""
        completed = []
        for exp_id, task in self.running_experiments.items():
            if task.done():
                completed.append(exp_id)
        
        for exp_id in completed:
            del self.running_experiments[exp_id]
    
    async def _update_monitoring_data(self) -> None:
        """Update monitoring data for all running experiments"""
        for exp_id in self.running_experiments:
            await self.monitor.update_experiment_metrics(exp_id)
    
    async def _shutdown_all_experiments(self) -> None:
        """Gracefully shutdown all running experiments"""
        self.logger.info("Shutting down all experiments")
        
        for task in self.running_experiments.values():
            task.cancel()
        
        # Wait for all tasks to complete
        if self.running_experiments:
            await asyncio.gather(
                *self.running_experiments.values(),
                return_exceptions=True
            )
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status and progress"""
        metadata = await self.db_manager.get_experiment_metadata(experiment_id)
        if not metadata:
            return {"error": "Experiment not found"}
        
        # Get detailed progress if running
        progress_data = {}
        if metadata.status == ExperimentStatus.RUNNING:
            progress_data = await self.monitor.get_current_progress(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "name": metadata.name,
            "status": metadata.status.value,
            "progress": metadata.progress,
            "current_stage": metadata.current_stage,
            "created_at": metadata.created_at.isoformat(),
            "started_at": metadata.started_at.isoformat() if metadata.started_at else None,
            "completed_at": metadata.completed_at.isoformat() if metadata.completed_at else None,
            "total_cost": metadata.total_cost,
            "compute_hours": metadata.compute_hours,
            **progress_data
        }
    
    async def cancel_experiment(self, experiment_id: str) -> None:
        """Cancel a running or queued experiment"""
        # Cancel if running
        if experiment_id in self.running_experiments:
            self.running_experiments[experiment_id].cancel()
            del self.running_experiments[experiment_id]
        
        # Remove from queue if queued
        await self.queue.remove_experiment(experiment_id)
        
        # Update database status
        await self.db_manager.update_experiment_status(
            experiment_id, ExperimentStatus.CANCELLED
        )
        
        self.logger.info(f"Cancelled experiment {experiment_id}")
    
    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        researcher: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        return await self.db_manager.list_experiments(
            status=status,
            researcher=researcher, 
            limit=limit
        )
    
    def shutdown(self) -> None:
        """Initiate graceful shutdown"""
        self.logger.info("Shutdown requested")
        self._shutdown_requested = True