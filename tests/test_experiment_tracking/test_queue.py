"""
Tests for experiment queue management system.

Comprehensive tests for ExperimentQueue covering priority handling,
dependency resolution, resource management, and retry logic.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from scramblebench.experiment_tracking.queue import (
    ExperimentQueue, QueuedExperiment, ResourceRequirements, QueueStatus, QueueMetrics
)


class TestResourceRequirements:
    """Test cases for ResourceRequirements class."""
    
    def test_resource_requirements_creation(self):
        """Test creating ResourceRequirements with default values."""
        req = ResourceRequirements()
        
        assert req.api_calls_per_hour == 1000
        assert req.max_concurrent_requests == 10
        assert req.memory_gb == 4.0
        assert req.storage_gb == 1.0
        assert req.estimated_duration_hours == 2.0
        assert req.cost_limit == 100.0
    
    def test_resource_requirements_custom_values(self):
        """Test creating ResourceRequirements with custom values."""
        req = ResourceRequirements(
            api_calls_per_hour=2000,
            memory_gb=8.0,
            storage_gb=5.0,
            estimated_duration_hours=3.5,
            cost_limit=250.0
        )
        
        assert req.api_calls_per_hour == 2000
        assert req.memory_gb == 8.0
        assert req.storage_gb == 5.0
        assert req.estimated_duration_hours == 3.5
        assert req.cost_limit == 250.0
    
    def test_resource_requirements_comparison(self):
        """Test ResourceRequirements comparison for priority queue."""
        req1 = ResourceRequirements(estimated_duration_hours=1.0)
        req2 = ResourceRequirements(estimated_duration_hours=2.0)
        
        assert req1 < req2  # Shorter duration should be "less"


class TestQueuedExperiment:
    """Test cases for QueuedExperiment class."""
    
    def test_queued_experiment_creation(self):
        """Test creating QueuedExperiment with required fields."""
        exp = QueuedExperiment(
            experiment_id="test-exp-1",
            priority=5
        )
        
        assert exp.experiment_id == "test-exp-1"
        assert exp.priority == 5
        assert exp.depends_on == []
        assert exp.attempts == 0
        assert exp.max_attempts == 3
        assert exp.status == QueueStatus.PENDING
    
    def test_queued_experiment_with_dependencies(self):
        """Test creating QueuedExperiment with dependencies."""
        exp = QueuedExperiment(
            experiment_id="dependent-exp",
            priority=1,
            depends_on=["exp-1", "exp-2"],
            resource_requirements=ResourceRequirements(memory_gb=8.0)
        )
        
        assert exp.depends_on == ["exp-1", "exp-2"]
        assert exp.resource_requirements.memory_gb == 8.0
    
    def test_queued_experiment_priority_comparison(self):
        """Test QueuedExperiment priority comparison for queue ordering."""
        exp1 = QueuedExperiment("exp-1", priority=1, queued_at=datetime.now())
        exp2 = QueuedExperiment("exp-2", priority=2, queued_at=datetime.now())
        exp3 = QueuedExperiment("exp-3", priority=1, queued_at=datetime.now() + timedelta(seconds=1))
        
        # Higher priority should be "less" (comes first in queue)
        assert exp2 < exp1
        
        # Same priority, earlier queued time should be "less" (FIFO)
        assert exp1 < exp3
    
    def test_queued_experiment_can_retry_initial(self):
        """Test can_retry for experiment that hasn't been attempted."""
        exp = QueuedExperiment("exp-retry", priority=1)
        
        assert exp.can_retry() is True
    
    def test_queued_experiment_can_retry_under_limit(self):
        """Test can_retry for experiment under attempt limit."""
        exp = QueuedExperiment("exp-retry", priority=1)
        exp.attempts = 2
        exp.max_attempts = 3
        exp.last_attempt_at = datetime.now() - timedelta(minutes=10)  # Long enough ago
        
        assert exp.can_retry() is True
    
    def test_queued_experiment_can_retry_max_attempts(self):
        """Test can_retry for experiment at max attempts."""
        exp = QueuedExperiment("exp-retry", priority=1)
        exp.attempts = 3
        exp.max_attempts = 3
        
        assert exp.can_retry() is False
    
    def test_queued_experiment_can_retry_backoff(self):
        """Test can_retry with exponential backoff."""
        exp = QueuedExperiment("exp-retry", priority=1)
        exp.attempts = 1
        exp.last_attempt_at = datetime.now() - timedelta(minutes=1)  # Too recent for 2^1 = 2 minute backoff
        
        assert exp.can_retry() is False
        
        # Wait long enough
        exp.last_attempt_at = datetime.now() - timedelta(minutes=3)  # Long enough for backoff
        assert exp.can_retry() is True


class TestExperimentQueue:
    """Test cases for ExperimentQueue main functionality."""
    
    def test_experiment_queue_initialization(self):
        """Test ExperimentQueue initialization."""
        queue = ExperimentQueue(
            max_concurrent=5,
            resource_limits={'total_memory_gb': 32.0}
        )
        
        assert queue.max_concurrent == 5
        assert queue.resource_limits['total_memory_gb'] == 32.0
        assert queue.experiments == {}
        assert queue.priority_queue == []
        assert queue.running == {}
        assert queue.dependency_graph == {}
    
    @pytest.mark.asyncio
    async def test_add_experiment_simple(self, experiment_queue, sample_queued_experiment):
        """Test adding simple experiment to queue."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        
        assert sample_queued_experiment.experiment_id in experiment_queue.experiments
        assert sample_queued_experiment.status == QueueStatus.READY
        assert len(experiment_queue.priority_queue) == 1
    
    @pytest.mark.asyncio
    async def test_add_experiment_duplicate(self, experiment_queue, sample_queued_experiment):
        """Test adding duplicate experiment raises error."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        
        with pytest.raises(ValueError, match="already queued"):
            await experiment_queue.add_experiment(sample_queued_experiment)
    
    @pytest.mark.asyncio
    async def test_add_experiment_with_dependencies(self, experiment_queue):
        """Test adding experiment with unmet dependencies."""
        dependent_exp = QueuedExperiment(
            experiment_id="dependent-exp",
            priority=1,
            depends_on=["non-existent-exp"]
        )
        
        await experiment_queue.add_experiment(dependent_exp)
        
        # Should be blocked, not ready
        assert dependent_exp.status == QueueStatus.BLOCKED
        assert len(experiment_queue.priority_queue) == 0
        assert "dependent-exp" in experiment_queue.dependency_graph
    
    @pytest.mark.asyncio
    async def test_get_ready_experiments_empty_queue(self, experiment_queue):
        """Test getting ready experiments from empty queue."""
        ready = await experiment_queue.get_ready_experiments()
        
        assert ready == []
    
    @pytest.mark.asyncio
    async def test_get_ready_experiments_with_limit(self, experiment_queue):
        """Test getting ready experiments with limit."""
        # Add multiple experiments
        for i in range(5):
            exp = QueuedExperiment(f"exp-{i}", priority=i)
            await experiment_queue.add_experiment(exp)
        
        ready = await experiment_queue.get_ready_experiments(max_count=2)
        
        assert len(ready) == 2
        # Should be ordered by priority (highest first)
        assert ready[0].priority >= ready[1].priority
    
    @pytest.mark.asyncio
    async def test_get_ready_experiments_resource_constraints(self, experiment_queue):
        """Test getting experiments with resource constraints."""
        # Add experiment that would exceed memory limit
        high_memory_exp = QueuedExperiment(
            "memory-intensive",
            priority=1,
            resource_requirements=ResourceRequirements(memory_gb=20.0)  # Exceeds default limit
        )
        
        low_memory_exp = QueuedExperiment(
            "memory-light",
            priority=1,
            resource_requirements=ResourceRequirements(memory_gb=2.0)
        )
        
        await experiment_queue.add_experiment(high_memory_exp)
        await experiment_queue.add_experiment(low_memory_exp)
        
        ready = await experiment_queue.get_ready_experiments()
        
        # Only low memory experiment should be ready
        assert len(ready) == 1
        assert ready[0].experiment_id == "memory-light"
    
    @pytest.mark.asyncio
    async def test_complete_experiment_success(self, experiment_queue, sample_queued_experiment):
        """Test completing experiment successfully."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        ready = await experiment_queue.get_ready_experiments()
        
        # Complete the experiment
        await experiment_queue.complete_experiment(
            sample_queued_experiment.experiment_id,
            success=True
        )
        
        assert sample_queued_experiment.experiment_id in experiment_queue.completed
        assert sample_queued_experiment.status == QueueStatus.COMPLETED
        assert sample_queued_experiment.experiment_id not in experiment_queue.running
    
    @pytest.mark.asyncio
    async def test_complete_experiment_failure_with_retry(self, experiment_queue, sample_queued_experiment):
        """Test completing experiment with failure and retry."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        ready = await experiment_queue.get_ready_experiments()
        
        # Fail the experiment (should retry)
        await experiment_queue.complete_experiment(
            sample_queued_experiment.experiment_id,
            success=False,
            error="Test failure"
        )
        
        assert sample_queued_experiment.attempts == 1
        assert sample_queued_experiment.failure_reason == "Test failure"
        assert sample_queued_experiment.status == QueueStatus.READY  # Should be re-queued
        assert len(experiment_queue.priority_queue) == 1
    
    @pytest.mark.asyncio
    async def test_complete_experiment_failure_max_attempts(self, experiment_queue, sample_queued_experiment):
        """Test completing experiment that reaches max attempts."""
        sample_queued_experiment.attempts = 3
        sample_queued_experiment.max_attempts = 3
        
        await experiment_queue.add_experiment(sample_queued_experiment)
        ready = await experiment_queue.get_ready_experiments()
        
        # Fail the experiment (should not retry)
        await experiment_queue.complete_experiment(
            sample_queued_experiment.experiment_id,
            success=False,
            error="Final failure"
        )
        
        assert sample_queued_experiment.experiment_id in experiment_queue.failed
        assert sample_queued_experiment.status == QueueStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_complete_experiment_not_running(self, experiment_queue):
        """Test completing experiment that's not running."""
        # Should handle gracefully without error
        await experiment_queue.complete_experiment("non-existent", success=True)
        
        # No assertions needed - just verify no exception raised
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, experiment_queue):
        """Test dependency resolution when parent completes."""
        # Add parent experiment
        parent_exp = QueuedExperiment("parent-exp", priority=1)
        await experiment_queue.add_experiment(parent_exp)
        
        # Add dependent experiment
        child_exp = QueuedExperiment(
            "child-exp",
            priority=1,
            depends_on=["parent-exp"]
        )
        await experiment_queue.add_experiment(child_exp)
        
        # Child should be blocked
        assert child_exp.status == QueueStatus.BLOCKED
        assert len(experiment_queue.priority_queue) == 1  # Only parent
        
        # Start and complete parent
        ready = await experiment_queue.get_ready_experiments()
        await experiment_queue.complete_experiment("parent-exp", success=True)
        
        # Child should now be ready
        assert child_exp.status == QueueStatus.READY
        assert len(experiment_queue.priority_queue) == 1  # Now child
    
    @pytest.mark.asyncio
    async def test_remove_experiment_queued(self, experiment_queue, sample_queued_experiment):
        """Test removing queued experiment."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        
        removed = await experiment_queue.remove_experiment(sample_queued_experiment.experiment_id)
        
        assert removed is True
        assert sample_queued_experiment.experiment_id not in experiment_queue.experiments
        assert sample_queued_experiment.status == QueueStatus.CANCELLED
        assert len(experiment_queue.priority_queue) == 0
    
    @pytest.mark.asyncio
    async def test_remove_experiment_running(self, experiment_queue, sample_queued_experiment):
        """Test removing running experiment."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        ready = await experiment_queue.get_ready_experiments()  # Moves to running
        
        removed = await experiment_queue.remove_experiment(sample_queued_experiment.experiment_id)
        
        assert removed is True
        assert sample_queued_experiment.experiment_id not in experiment_queue.running
        assert sample_queued_experiment.status == QueueStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_remove_experiment_with_dependents(self, experiment_queue):
        """Test removing experiment that has dependents."""
        # Add parent and child
        parent_exp = QueuedExperiment("parent-exp", priority=1)
        child_exp = QueuedExperiment("child-exp", priority=1, depends_on=["parent-exp"])
        
        await experiment_queue.add_experiment(parent_exp)
        await experiment_queue.add_experiment(child_exp)
        
        # Remove parent
        await experiment_queue.remove_experiment("parent-exp")
        
        # Child should be marked as blocked (dependency removed)
        assert child_exp.status == QueueStatus.BLOCKED
    
    @pytest.mark.asyncio
    async def test_remove_experiment_not_found(self, experiment_queue):
        """Test removing non-existent experiment."""
        removed = await experiment_queue.remove_experiment("non-existent")
        
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_get_experiment_position(self, experiment_queue):
        """Test getting experiment position in queue."""
        # Add experiments with different priorities
        exp1 = QueuedExperiment("exp-1", priority=1, queued_at=datetime.now())
        exp2 = QueuedExperiment("exp-2", priority=3, queued_at=datetime.now())  # Higher priority
        exp3 = QueuedExperiment("exp-3", priority=1, queued_at=datetime.now() + timedelta(seconds=1))
        
        await experiment_queue.add_experiment(exp1)
        await experiment_queue.add_experiment(exp2)
        await experiment_queue.add_experiment(exp3)
        
        # exp2 should be first (highest priority)
        pos2 = await experiment_queue.get_experiment_position("exp-2")
        assert pos2 == 0
        
        # exp1 should be second (same priority as exp3 but queued earlier)
        pos1 = await experiment_queue.get_experiment_position("exp-1")
        assert pos1 == 1
    
    @pytest.mark.asyncio
    async def test_get_experiment_position_running(self, experiment_queue, sample_queued_experiment):
        """Test position for running experiment."""
        await experiment_queue.add_experiment(sample_queued_experiment)
        ready = await experiment_queue.get_ready_experiments()  # Moves to running
        
        position = await experiment_queue.get_experiment_position(sample_queued_experiment.experiment_id)
        
        assert position == 0  # Running experiments have position 0
    
    @pytest.mark.asyncio
    async def test_get_experiment_position_not_found(self, experiment_queue):
        """Test position for non-existent experiment."""
        position = await experiment_queue.get_experiment_position("non-existent")
        
        assert position is None
    
    @pytest.mark.asyncio
    async def test_update_priority(self, experiment_queue):
        """Test updating experiment priority."""
        exp = QueuedExperiment("priority-exp", priority=1)
        await experiment_queue.add_experiment(exp)
        
        updated = await experiment_queue.update_priority("priority-exp", 5)
        
        assert updated is True
        assert exp.priority == 5
    
    @pytest.mark.asyncio
    async def test_update_priority_not_found(self, experiment_queue):
        """Test updating priority for non-existent experiment."""
        updated = await experiment_queue.update_priority("non-existent", 5)
        
        assert updated is False
    
    @pytest.mark.asyncio
    async def test_get_queue_metrics(self, experiment_queue):
        """Test getting queue performance metrics."""
        # Add various experiments in different states
        ready_exp = QueuedExperiment("ready-exp", priority=1)
        await experiment_queue.add_experiment(ready_exp)
        
        blocked_exp = QueuedExperiment("blocked-exp", priority=1, depends_on=["non-existent"])
        await experiment_queue.add_experiment(blocked_exp)
        
        # Start one experiment
        ready = await experiment_queue.get_ready_experiments()
        
        metrics = await experiment_queue.get_queue_metrics()
        
        assert isinstance(metrics, QueueMetrics)
        assert metrics.total_experiments == 2
        assert metrics.ready == 0  # ready_exp moved to running
        assert metrics.blocked == 1
        assert metrics.running == 1
        assert metrics.completed == 0
        assert metrics.failed == 0
    
    def test_get_queue_summary(self, experiment_queue):
        """Test getting queue summary."""
        summary = experiment_queue.get_queue_summary()
        
        assert isinstance(summary, dict)
        assert "total_experiments" in summary
        assert "running" in summary
        assert "completed" in summary
        assert "failed" in summary
        assert "current_resource_usage" in summary
        assert "resource_limits" in summary
        assert "uptime" in summary


class TestExperimentQueueResourceManagement:
    """Test cases for resource management in experiment queue."""
    
    @pytest.mark.asyncio
    async def test_resource_allocation_tracking(self, experiment_queue):
        """Test resource allocation tracking."""
        # Add experiment with specific resources
        exp = QueuedExperiment(
            "resource-exp",
            priority=1,
            resource_requirements=ResourceRequirements(
                api_calls_per_hour=1000,
                memory_gb=4.0,
                storage_gb=2.0
            )
        )
        
        await experiment_queue.add_experiment(exp)
        ready = await experiment_queue.get_ready_experiments()
        
        # Resources should be allocated
        assert experiment_queue.current_resource_usage['api_calls_per_hour'] == 1000
        assert experiment_queue.current_resource_usage['memory_gb'] == 4.0
        assert experiment_queue.current_resource_usage['storage_gb'] == 2.0
    
    @pytest.mark.asyncio
    async def test_resource_deallocation_on_completion(self, experiment_queue):
        """Test resource deallocation when experiment completes."""
        exp = QueuedExperiment(
            "resource-exp",
            priority=1,
            resource_requirements=ResourceRequirements(memory_gb=4.0)
        )
        
        await experiment_queue.add_experiment(exp)
        ready = await experiment_queue.get_ready_experiments()
        
        # Complete experiment
        await experiment_queue.complete_experiment("resource-exp", success=True)
        
        # Resources should be freed
        assert experiment_queue.current_resource_usage['memory_gb'] == 0.0
    
    @pytest.mark.asyncio
    async def test_resource_constraint_blocking(self, experiment_queue):
        """Test that resource constraints block experiments."""
        # Set tight resource limits
        experiment_queue.resource_limits = {
            'total_api_calls_per_hour': 1500,
            'total_memory_gb': 8.0,
            'total_storage_gb': 10.0,
            'total_cost_per_day': 100.0
        }
        
        # Add experiment that fits
        exp1 = QueuedExperiment(
            "exp-1",
            priority=1,
            resource_requirements=ResourceRequirements(memory_gb=4.0)
        )
        await experiment_queue.add_experiment(exp1)
        
        # Add experiment that would exceed limits
        exp2 = QueuedExperiment(
            "exp-2",
            priority=2,  # Higher priority but resource constrained
            resource_requirements=ResourceRequirements(memory_gb=6.0)  # 4 + 6 > 8 limit
        )
        await experiment_queue.add_experiment(exp2)
        
        ready = await experiment_queue.get_ready_experiments()
        
        # Only first experiment should be ready
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-1"
    
    @pytest.mark.asyncio 
    async def test_resource_availability_after_completion(self, experiment_queue):
        """Test that resources become available after experiment completion."""
        # Set tight limits
        experiment_queue.resource_limits = {
            'total_memory_gb': 6.0,
            'total_api_calls_per_hour': 2000,
            'total_storage_gb': 10.0,
            'total_cost_per_day': 100.0
        }
        
        # Add experiments that together exceed limits
        exp1 = QueuedExperiment("exp-1", priority=1, resource_requirements=ResourceRequirements(memory_gb=4.0))
        exp2 = QueuedExperiment("exp-2", priority=1, resource_requirements=ResourceRequirements(memory_gb=4.0))
        
        await experiment_queue.add_experiment(exp1)
        await experiment_queue.add_experiment(exp2)
        
        # Only first should be ready
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-1"
        
        # Complete first experiment
        await experiment_queue.complete_experiment("exp-1", success=True)
        
        # Now second should be ready
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-2"


class TestExperimentQueueComplexScenarios:
    """Test cases for complex queue scenarios."""
    
    @pytest.mark.asyncio
    async def test_complex_dependency_chain(self, experiment_queue, complex_dependency_graph):
        """Test complex dependency resolution."""
        # Add all experiments from dependency graph
        experiments = {}
        for exp_data in complex_dependency_graph:
            exp = QueuedExperiment(
                experiment_id=exp_data["id"],
                priority=exp_data["priority"],
                depends_on=exp_data["depends_on"]
            )
            experiments[exp_data["id"]] = exp
            await experiment_queue.add_experiment(exp)
        
        # Only experiments without dependencies should be ready initially
        ready = await experiment_queue.get_ready_experiments()
        ready_ids = [exp.experiment_id for exp in ready]
        
        # exp-a and exp-e have no dependencies
        assert "exp-a" in ready_ids
        assert "exp-e" in ready_ids
        assert len(ready_ids) == 2
    
    @pytest.mark.asyncio
    async def test_cascading_dependency_resolution(self, experiment_queue):
        """Test cascading dependency resolution as experiments complete."""
        # Create dependency chain: A -> B -> C
        exp_a = QueuedExperiment("exp-a", priority=1)
        exp_b = QueuedExperiment("exp-b", priority=1, depends_on=["exp-a"])
        exp_c = QueuedExperiment("exp-c", priority=1, depends_on=["exp-b"])
        
        await experiment_queue.add_experiment(exp_a)
        await experiment_queue.add_experiment(exp_b)
        await experiment_queue.add_experiment(exp_c)
        
        # Only A should be ready
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-a"
        
        # Complete A, B should become ready
        await experiment_queue.complete_experiment("exp-a", success=True)
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-b"
        
        # Complete B, C should become ready
        await experiment_queue.complete_experiment("exp-b", success=True)
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-c"
    
    @pytest.mark.asyncio
    async def test_priority_with_dependencies(self, experiment_queue):
        """Test priority ordering with dependency constraints."""
        # Create experiments with different priorities but dependencies
        exp_low = QueuedExperiment("exp-low", priority=1)  # Low priority, no deps
        exp_high = QueuedExperiment("exp-high", priority=10, depends_on=["exp-low"])  # High priority, has deps
        
        await experiment_queue.add_experiment(exp_low)
        await experiment_queue.add_experiment(exp_high)
        
        # Low priority experiment should be ready first due to dependency
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-low"
        
        # After completion, high priority should be ready
        await experiment_queue.complete_experiment("exp-low", success=True)
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) == 1
        assert ready[0].experiment_id == "exp-high"
    
    @pytest.mark.asyncio
    async def test_failed_dependency_handling(self, experiment_queue):
        """Test handling of failed dependency experiments."""
        # Create dependency where parent fails
        exp_parent = QueuedExperiment("exp-parent", priority=1, max_attempts=1)
        exp_child = QueuedExperiment("exp-child", priority=1, depends_on=["exp-parent"])
        
        await experiment_queue.add_experiment(exp_parent)
        await experiment_queue.add_experiment(exp_child)
        
        # Start and fail parent
        ready = await experiment_queue.get_ready_experiments()
        await experiment_queue.complete_experiment("exp-parent", success=False, error="Parent failed")
        
        # Child should remain blocked (parent failed)
        assert exp_child.status == QueueStatus.BLOCKED
        ready = await experiment_queue.get_ready_experiments()
        child_ready = [exp for exp in ready if exp.experiment_id == "exp-child"]
        assert len(child_ready) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_experiment_limits(self, experiment_queue):
        """Test concurrent experiment limits are enforced."""
        experiment_queue.max_concurrent = 2  # Set low limit
        
        # Add 5 experiments
        for i in range(5):
            exp = QueuedExperiment(f"exp-{i}", priority=i)
            await experiment_queue.add_experiment(exp)
        
        # Only 2 should be ready (max_concurrent limit)
        ready = await experiment_queue.get_ready_experiments()
        assert len(ready) <= 2
    
    @pytest.mark.asyncio
    async def test_experiment_retry_with_backoff(self, experiment_queue):
        """Test experiment retry with exponential backoff."""
        exp = QueuedExperiment("retry-exp", priority=1, max_attempts=3)
        await experiment_queue.add_experiment(exp)
        
        # First attempt
        ready = await experiment_queue.get_ready_experiments()
        await experiment_queue.complete_experiment("retry-exp", success=False, error="First failure")
        
        assert exp.attempts == 1
        assert exp.can_retry() is False  # Too soon due to backoff
        
        # Simulate time passing
        exp.last_attempt_at = datetime.now() - timedelta(minutes=5)  # Long enough for backoff
        assert exp.can_retry() is True
    
    @pytest.mark.asyncio
    async def test_queue_metrics_comprehensive(self, experiment_queue):
        """Test comprehensive queue metrics calculation."""
        # Create experiments in various states
        completed_exp = QueuedExperiment("completed", priority=1)
        running_exp = QueuedExperiment("running", priority=1)
        failed_exp = QueuedExperiment("failed", priority=1, max_attempts=1)
        blocked_exp = QueuedExperiment("blocked", priority=1, depends_on=["non-existent"])
        
        # Add and process experiments
        await experiment_queue.add_experiment(completed_exp)
        await experiment_queue.add_experiment(running_exp)
        await experiment_queue.add_experiment(failed_exp)
        await experiment_queue.add_experiment(blocked_exp)
        
        # Process some experiments
        ready = await experiment_queue.get_ready_experiments()  # Should get completed and running
        
        await experiment_queue.complete_experiment("completed", success=True)
        await experiment_queue.complete_experiment("running", success=False, error="Failed")  # Will exceed max_attempts
        
        # Get metrics
        metrics = await experiment_queue.get_queue_metrics()
        
        assert metrics.completed == 1
        assert metrics.failed == 1
        assert metrics.blocked == 1
        assert metrics.total_experiments == 4
        
        # Check that throughput is calculated
        assert metrics.throughput_per_hour >= 0