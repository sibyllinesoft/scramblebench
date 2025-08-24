"""
Experiment Queue Management System

Advanced queueing system for managing large-scale experiments with priority
handling, dependency resolution, resource constraints, and retry logic.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import heapq


class QueueStatus(Enum):
    """Status of queued experiments"""
    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    RUNNING = "running"  
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirements:
    """Resource requirements for an experiment"""
    api_calls_per_hour: int = 1000
    max_concurrent_requests: int = 10
    memory_gb: float = 4.0
    storage_gb: float = 1.0
    estimated_duration_hours: float = 2.0
    cost_limit: float = 100.0
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.estimated_duration_hours < other.estimated_duration_hours


@dataclass
class QueuedExperiment:
    """An experiment in the execution queue"""
    experiment_id: str
    priority: int  # Higher number = higher priority
    depends_on: List[str] = field(default_factory=list)
    resource_requirements: Optional[ResourceRequirements] = None
    estimated_duration: Optional[timedelta] = None
    queued_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    last_attempt_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    status: QueueStatus = QueueStatus.PENDING
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        # If same priority, FIFO
        return self.queued_at < other.queued_at
    
    def can_retry(self) -> bool:
        """Check if experiment can be retried"""
        if self.attempts >= self.max_attempts:
            return False
        
        # Exponential backoff: wait 2^attempts minutes
        if self.last_attempt_at:
            backoff_minutes = 2 ** self.attempts
            next_attempt = self.last_attempt_at + timedelta(minutes=backoff_minutes)
            return datetime.now() >= next_attempt
        
        return True


@dataclass  
class QueueMetrics:
    """Queue performance metrics"""
    total_experiments: int
    pending: int
    ready: int
    blocked: int
    running: int
    completed: int
    failed: int
    cancelled: int
    avg_wait_time: Optional[timedelta]
    avg_execution_time: Optional[timedelta]
    throughput_per_hour: float
    resource_utilization: Dict[str, float]


class ExperimentQueue:
    """
    Advanced experiment queue with priority handling, dependencies,
    resource management, and retry logic
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        resource_limits: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize experiment queue
        
        Args:
            max_concurrent: Maximum concurrent experiments
            resource_limits: Global resource limits
            logger: Logger instance
        """
        self.max_concurrent = max_concurrent
        self.resource_limits = resource_limits or {
            'total_api_calls_per_hour': 10000,
            'total_memory_gb': 32.0,
            'total_storage_gb': 100.0,
            'total_cost_per_day': 1000.0
        }
        self.logger = logger or logging.getLogger(__name__)
        
        # Queue storage
        self.experiments: Dict[str, QueuedExperiment] = {}
        self.priority_queue: List[QueuedExperiment] = []
        self.running: Dict[str, QueuedExperiment] = {}
        self.completed: Dict[str, QueuedExperiment] = {}
        self.failed: Dict[str, QueuedExperiment] = {}
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}  # exp_id -> depends_on
        self.reverse_deps: Dict[str, Set[str]] = {}      # exp_id -> dependents
        
        # Resource tracking
        self.current_resource_usage = {
            'api_calls_per_hour': 0,
            'memory_gb': 0.0,
            'storage_gb': 0.0,
            'cost_per_day': 0.0
        }
        
        # Metrics
        self.start_time = datetime.now()
        self.total_experiments_processed = 0
        self.total_wait_time = timedelta()
        self.total_execution_time = timedelta()
        
        self._lock = asyncio.Lock()
    
    async def add_experiment(self, experiment: QueuedExperiment) -> None:
        """Add experiment to queue with dependency resolution"""
        async with self._lock:
            if experiment.experiment_id in self.experiments:
                raise ValueError(f"Experiment {experiment.experiment_id} already queued")
            
            # Store experiment
            self.experiments[experiment.experiment_id] = experiment
            
            # Set up dependencies
            if experiment.depends_on:
                self.dependency_graph[experiment.experiment_id] = set(experiment.depends_on)
                
                # Update reverse dependencies
                for dep_id in experiment.depends_on:
                    if dep_id not in self.reverse_deps:
                        self.reverse_deps[dep_id] = set()
                    self.reverse_deps[dep_id].add(experiment.experiment_id)
            
            # Add to priority queue if ready
            if self._is_ready(experiment):
                experiment.status = QueueStatus.READY
                heapq.heappush(self.priority_queue, experiment)
            else:
                experiment.status = QueueStatus.BLOCKED
            
            self.logger.info(f"Added experiment {experiment.experiment_id} to queue")
    
    async def get_ready_experiments(
        self, 
        max_count: Optional[int] = None
    ) -> List[QueuedExperiment]:
        """
        Get ready experiments that can be started based on resource availability
        
        Args:
            max_count: Maximum number to return
            
        Returns:
            List of ready experiments
        """
        async with self._lock:
            if max_count is None:
                max_count = self.max_concurrent - len(self.running)
            
            ready_experiments = []
            temp_usage = self.current_resource_usage.copy()
            
            # Process priority queue
            while (self.priority_queue and 
                   len(ready_experiments) < max_count):
                
                experiment = heapq.heappop(self.priority_queue)
                
                # Verify still ready (state may have changed)
                if not self._is_ready(experiment):
                    experiment.status = QueueStatus.BLOCKED
                    continue
                
                # Check resource availability
                if self._can_allocate_resources(experiment, temp_usage):
                    ready_experiments.append(experiment)
                    experiment.status = QueueStatus.RUNNING
                    self.running[experiment.experiment_id] = experiment
                    
                    # Update temporary usage
                    if experiment.resource_requirements:
                        self._update_usage(temp_usage, experiment.resource_requirements, add=True)
                else:
                    # Put back in queue if can't allocate resources
                    heapq.heappush(self.priority_queue, experiment)
                    break
            
            # Update actual usage
            self.current_resource_usage = temp_usage
            
            return ready_experiments
    
    async def complete_experiment(
        self, 
        experiment_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Mark experiment as completed and update dependencies
        
        Args:
            experiment_id: Experiment ID
            success: Whether experiment succeeded
            error: Error message if failed
        """
        async with self._lock:
            if experiment_id not in self.running:
                self.logger.warning(f"Experiment {experiment_id} not in running state")
                return
            
            experiment = self.running.pop(experiment_id)
            
            # Free resources
            if experiment.resource_requirements:
                self._update_usage(
                    self.current_resource_usage, 
                    experiment.resource_requirements, 
                    add=False
                )
            
            # Update status and metrics
            if success:
                experiment.status = QueueStatus.COMPLETED
                self.completed[experiment_id] = experiment
                self.total_experiments_processed += 1
                
                # Calculate execution time
                if experiment.last_attempt_at:
                    execution_time = datetime.now() - experiment.last_attempt_at
                    self.total_execution_time += execution_time
                
                self.logger.info(f"Completed experiment {experiment_id}")
                
            else:
                experiment.attempts += 1
                experiment.failure_reason = error
                
                # Check if can retry
                if experiment.can_retry():
                    experiment.status = QueueStatus.PENDING
                    experiment.last_attempt_at = datetime.now()
                    
                    # Re-queue for retry
                    if self._is_ready(experiment):
                        experiment.status = QueueStatus.READY
                        heapq.heappush(self.priority_queue, experiment)
                    
                    self.logger.warning(f"Queueing retry {experiment.attempts}/{experiment.max_attempts} for {experiment_id}")
                else:
                    experiment.status = QueueStatus.FAILED
                    self.failed[experiment_id] = experiment
                    self.logger.error(f"Failed experiment {experiment_id} after {experiment.attempts} attempts")
            
            # Check and unblock dependent experiments
            await self._check_dependent_experiments(experiment_id)
    
    async def remove_experiment(self, experiment_id: str) -> bool:
        """
        Remove experiment from queue (cancel)
        
        Args:
            experiment_id: Experiment to remove
            
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            experiment = None
            
            # Find and remove from appropriate collection
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                
                # Remove from priority queue
                if experiment.status == QueueStatus.READY:
                    self.priority_queue = [e for e in self.priority_queue 
                                         if e.experiment_id != experiment_id]
                    heapq.heapify(self.priority_queue)
                
                # Free resources if running
                if experiment_id in self.running:
                    if experiment.resource_requirements:
                        self._update_usage(
                            self.current_resource_usage,
                            experiment.resource_requirements,
                            add=False
                        )
                    del self.running[experiment_id]
                
                # Update status
                experiment.status = QueueStatus.CANCELLED
                del self.experiments[experiment_id]
                
                # Clean up dependencies
                if experiment_id in self.dependency_graph:
                    del self.dependency_graph[experiment_id]
                
                if experiment_id in self.reverse_deps:
                    # Mark dependents as blocked
                    for dependent_id in self.reverse_deps[experiment_id]:
                        if dependent_id in self.experiments:
                            self.experiments[dependent_id].status = QueueStatus.BLOCKED
                    del self.reverse_deps[experiment_id]
                
                self.logger.info(f"Removed experiment {experiment_id}")
                return True
            
            return False
    
    def _is_ready(self, experiment: QueuedExperiment) -> bool:
        """Check if experiment dependencies are satisfied"""
        if not experiment.depends_on:
            return True
        
        for dep_id in experiment.depends_on:
            if dep_id not in self.completed:
                return False
        
        return True
    
    def _can_allocate_resources(
        self, 
        experiment: QueuedExperiment,
        current_usage: Dict[str, float]
    ) -> bool:
        """Check if resources are available for experiment"""
        if not experiment.resource_requirements:
            return True
        
        req = experiment.resource_requirements
        
        # Check each resource limit
        if (current_usage['api_calls_per_hour'] + req.api_calls_per_hour > 
            self.resource_limits['total_api_calls_per_hour']):
            return False
        
        if (current_usage['memory_gb'] + req.memory_gb >
            self.resource_limits['total_memory_gb']):
            return False
        
        if (current_usage['storage_gb'] + req.storage_gb >
            self.resource_limits['total_storage_gb']):
            return False
        
        return True
    
    def _update_usage(
        self,
        usage: Dict[str, float],
        requirements: ResourceRequirements,
        add: bool = True
    ) -> None:
        """Update resource usage tracking"""
        multiplier = 1 if add else -1
        
        usage['api_calls_per_hour'] += requirements.api_calls_per_hour * multiplier
        usage['memory_gb'] += requirements.memory_gb * multiplier
        usage['storage_gb'] += requirements.storage_gb * multiplier
        usage['cost_per_day'] += requirements.cost_limit * multiplier
        
        # Ensure non-negative
        for key in usage:
            usage[key] = max(0, usage[key])
    
    async def _check_dependent_experiments(self, completed_experiment_id: str) -> None:
        """Check if any experiments are now ready due to completion"""
        if completed_experiment_id not in self.reverse_deps:
            return
        
        dependents = self.reverse_deps[completed_experiment_id].copy()
        
        for dependent_id in dependents:
            if dependent_id not in self.experiments:
                continue
            
            dependent = self.experiments[dependent_id]
            
            # Check if now ready
            if (dependent.status == QueueStatus.BLOCKED and 
                self._is_ready(dependent)):
                dependent.status = QueueStatus.READY
                heapq.heappush(self.priority_queue, dependent)
                
                self.logger.info(f"Experiment {dependent_id} is now ready")
    
    async def get_queue_metrics(self) -> QueueMetrics:
        """Get queue performance metrics"""
        async with self._lock:
            # Count experiments by status
            status_counts = {status: 0 for status in QueueStatus}
            
            for experiment in self.experiments.values():
                status_counts[experiment.status] += 1
            
            for experiment in self.running.values():
                status_counts[QueueStatus.RUNNING] += 1
            
            for experiment in self.completed.values():
                status_counts[QueueStatus.COMPLETED] += 1
            
            for experiment in self.failed.values():
                status_counts[QueueStatus.FAILED] += 1
            
            # Calculate averages
            total_experiments = (len(self.experiments) + len(self.running) + 
                               len(self.completed) + len(self.failed))
            
            avg_wait_time = None
            avg_execution_time = None
            if self.total_experiments_processed > 0:
                avg_execution_time = self.total_execution_time / self.total_experiments_processed
            
            # Calculate throughput
            runtime = datetime.now() - self.start_time
            throughput = 0.0
            if runtime.total_seconds() > 0:
                throughput = self.total_experiments_processed / (runtime.total_seconds() / 3600)
            
            # Calculate resource utilization
            resource_utilization = {}
            for resource, current in self.current_resource_usage.items():
                if resource in self.resource_limits:
                    total_available = self.resource_limits[resource.replace('_per_hour', '').replace('_per_day', '')]
                    resource_utilization[resource] = current / total_available if total_available > 0 else 0.0
            
            return QueueMetrics(
                total_experiments=total_experiments,
                pending=status_counts[QueueStatus.PENDING],
                ready=status_counts[QueueStatus.READY], 
                blocked=status_counts[QueueStatus.BLOCKED],
                running=status_counts[QueueStatus.RUNNING],
                completed=status_counts[QueueStatus.COMPLETED],
                failed=status_counts[QueueStatus.FAILED],
                cancelled=status_counts[QueueStatus.CANCELLED],
                avg_wait_time=avg_wait_time,
                avg_execution_time=avg_execution_time,
                throughput_per_hour=throughput,
                resource_utilization=resource_utilization
            )
    
    async def get_experiment_position(self, experiment_id: str) -> Optional[int]:
        """Get position of experiment in queue (0-indexed)"""
        async with self._lock:
            if experiment_id in self.running:
                return 0  # Currently running
            
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            if experiment.status != QueueStatus.READY:
                return None  # Not in queue
            
            # Count experiments with higher priority ahead in queue
            position = 0
            for queued_exp in self.priority_queue:
                if queued_exp.experiment_id == experiment_id:
                    break
                if queued_exp > experiment:  # Higher priority
                    position += 1
            
            return position
    
    async def update_priority(self, experiment_id: str, new_priority: int) -> bool:
        """Update experiment priority"""
        async with self._lock:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            old_priority = experiment.priority
            experiment.priority = new_priority
            
            # If in priority queue, rebuild it
            if experiment.status == QueueStatus.READY:
                self.priority_queue = [e for e in self.priority_queue 
                                     if e.experiment_id != experiment_id]
                heapq.heappush(self.priority_queue, experiment)
                heapq.heapify(self.priority_queue)
            
            self.logger.info(f"Updated priority for {experiment_id}: {old_priority} -> {new_priority}")
            return True
    
    def get_queue_summary(self) -> Dict[str, Any]:
        """Get a summary of current queue state"""
        summary = {
            'total_experiments': len(self.experiments),
            'running': len(self.running),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'ready_in_queue': len(self.priority_queue),
            'current_resource_usage': self.current_resource_usage,
            'resource_limits': self.resource_limits,
            'uptime': str(datetime.now() - self.start_time),
            'throughput': self.total_experiments_processed
        }
        
        # Add next experiments to run
        if self.priority_queue:
            summary['next_experiments'] = [
                {
                    'experiment_id': exp.experiment_id,
                    'priority': exp.priority,
                    'estimated_duration': str(exp.estimated_duration) if exp.estimated_duration else None
                }
                for exp in sorted(self.priority_queue)[:5]  # Next 5
            ]
        
        return summary