"""
Tests for core experiment tracking functionality.

Comprehensive tests for ExperimentTracker class covering experiment
lifecycle, coordination, monitoring, and error handling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path

from scramblebench.experiment_tracking.core import (
    ExperimentTracker, ExperimentMetadata, ExperimentStatus
)
from scramblebench.experiment_tracking.queue import QueuedExperiment, ResourceRequirements


class TestExperimentMetadata:
    """Test cases for ExperimentMetadata class."""
    
    def test_experiment_metadata_creation(self, sample_experiment_metadata):
        """Test creating ExperimentMetadata instance."""
        metadata = sample_experiment_metadata
        
        assert metadata.experiment_id == "exp-test-123"
        assert metadata.name == "Test Experiment"
        assert metadata.status == ExperimentStatus.PLANNED
        assert metadata.progress == 0.0
        assert metadata.total_api_calls == 0
        assert metadata.total_cost == 0.0
    
    def test_experiment_metadata_to_dict(self, sample_experiment_metadata):
        """Test converting ExperimentMetadata to dictionary."""
        metadata = sample_experiment_metadata
        data_dict = metadata.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["experiment_id"] == metadata.experiment_id
        assert data_dict["name"] == metadata.name
        assert data_dict["status"] == metadata.status.value
        
        # Check datetime serialization
        assert isinstance(data_dict["created_at"], str)
        assert "T" in data_dict["created_at"]  # ISO format
    
    def test_experiment_metadata_with_optional_fields(self):
        """Test ExperimentMetadata with optional fields set to None."""
        metadata = ExperimentMetadata(
            experiment_id="minimal-exp",
            name="Minimal Experiment",
            description="Basic test",
            research_question="Question?",
            hypothesis=None,  # Optional
            researcher_name="Tester",
            institution=None,  # Optional
            git_commit_hash="abc123",
            git_branch="main",
            environment_snapshot={},
            random_seed=None,  # Optional
            config_hash="hash123",
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            status=ExperimentStatus.PLANNED
        )
        
        assert metadata.hypothesis is None
        assert metadata.institution is None
        assert metadata.random_seed is None
        
        # to_dict should handle None values
        data_dict = metadata.to_dict()
        assert data_dict["hypothesis"] is None
        assert data_dict["institution"] is None


class TestExperimentTracker:
    """Test cases for ExperimentTracker main functionality."""
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_initialization(self, temp_experiment_dir, mock_database_url):
        """Test ExperimentTracker initialization."""
        with patch('scramblebench.experiment_tracking.core.DatabaseManager') as mock_db_class, \
             patch('scramblebench.experiment_tracking.core.ExperimentQueue') as mock_queue_class, \
             patch('scramblebench.experiment_tracking.core.ExperimentMonitor') as mock_monitor_class, \
             patch('scramblebench.experiment_tracking.core.StatisticalAnalyzer') as mock_stats_class, \
             patch('scramblebench.experiment_tracking.core.ReproducibilityValidator') as mock_repro_class:
            
            tracker = ExperimentTracker(
                database_url=mock_database_url,
                data_dir=temp_experiment_dir,
                max_concurrent_experiments=3
            )
            
            assert tracker.database_url == mock_database_url
            assert tracker.data_dir == temp_experiment_dir
            assert tracker.max_concurrent == 3
            assert tracker.running_experiments == {}
            assert tracker._shutdown_requested is False
            
            # Verify components were initialized
            assert mock_db_class.called
            assert mock_queue_class.called
            assert mock_monitor_class.called
            assert mock_stats_class.called
            assert mock_repro_class.called
    
    @pytest.mark.asyncio
    async def test_create_experiment_success(self, experiment_tracker, sample_experiment_config, temp_experiment_dir):
        """Test successful experiment creation."""
        with patch('scramblebench.experiment_tracking.core.EvaluationConfig') as mock_config_class:
            mock_eval_config = Mock()
            mock_eval_config.sample_seed = 42
            mock_eval_config.save_to_file = Mock()
            mock_config_class.return_value = mock_eval_config
            
            experiment_id = await experiment_tracker.create_experiment(
                name="Test Experiment",
                config=sample_experiment_config,
                description="Test description",
                research_question="Does this work?",
                researcher_name="Test Researcher"
            )
            
            assert experiment_id is not None
            assert len(experiment_id) > 0
            
            # Verify experiment directory was created
            exp_dir = temp_experiment_dir / experiment_id
            assert exp_dir.exists()
            
            # Verify database was called
            experiment_tracker.db_manager.create_experiment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_experiment_with_path_config(self, experiment_tracker, temp_experiment_dir):
        """Test experiment creation with config file path."""
        # Create a mock config file
        config_file = temp_experiment_dir / "test_config.yaml"
        config_file.write_text("models: []\nbenchmarks: []")
        
        with patch('scramblebench.experiment_tracking.core.EvaluationConfig') as mock_config_class:
            mock_eval_config = Mock()
            mock_eval_config.sample_seed = 42
            mock_eval_config.save_to_file = Mock()
            mock_config_class.load_from_file.return_value = mock_eval_config
            
            experiment_id = await experiment_tracker.create_experiment(
                name="Path Config Test",
                config=config_file,
                description="Test with path config",
                research_question="Path config question?",
                researcher_name="Test Researcher"
            )
            
            assert experiment_id is not None
            mock_config_class.load_from_file.assert_called_once_with(config_file)
    
    @pytest.mark.asyncio
    async def test_queue_experiment_success(self, experiment_tracker, sample_experiment_metadata):
        """Test successful experiment queuing."""
        # Mock database responses
        experiment_tracker.db_manager.get_experiment_metadata.return_value = sample_experiment_metadata
        experiment_tracker.db_manager.get_experiment_config.return_value = Mock()
        
        await experiment_tracker.queue_experiment(
            experiment_id=sample_experiment_metadata.experiment_id,
            priority=2,
            depends_on=["exp-dep-1"],
            resource_requirements={"api_calls": 1000}
        )
        
        # Verify database calls
        experiment_tracker.db_manager.get_experiment_metadata.assert_called_once_with(sample_experiment_metadata.experiment_id)
        experiment_tracker.db_manager.update_experiment_status.assert_called_once()
        experiment_tracker.queue.add_experiment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_queue_experiment_not_found(self, experiment_tracker):
        """Test queuing non-existent experiment raises error."""
        experiment_tracker.db_manager.get_experiment_metadata.return_value = None
        
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await experiment_tracker.queue_experiment(
                experiment_id="non-existent-exp",
                priority=1
            )
    
    @pytest.mark.asyncio
    async def test_run_experiments_single_iteration(self, experiment_tracker, sample_queued_experiment):
        """Test running experiments for single iteration (non-continuous)."""
        # Mock queue to return ready experiment
        experiment_tracker.queue.get_ready_experiments.return_value = [sample_queued_experiment]
        
        with patch.object(experiment_tracker, '_start_experiment') as mock_start, \
             patch.object(experiment_tracker, '_cleanup_completed_experiments') as mock_cleanup, \
             patch.object(experiment_tracker, '_update_monitoring_data') as mock_update:
            
            await experiment_tracker.run_experiments(continuous=False)
            
            mock_start.assert_called_once_with(sample_queued_experiment)
            mock_cleanup.assert_called_once()
            mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_experiments_continuous_with_shutdown(self, experiment_tracker):
        """Test continuous experiment running with shutdown."""
        # Mock empty queue
        experiment_tracker.queue.get_ready_experiments.return_value = []
        
        with patch.object(experiment_tracker, '_cleanup_completed_experiments'), \
             patch.object(experiment_tracker, '_update_monitoring_data'):
            
            # Start continuous running
            run_task = asyncio.create_task(
                experiment_tracker.run_experiments(continuous=True, check_interval=0.1)
            )
            
            # Wait a bit then request shutdown
            await asyncio.sleep(0.05)
            experiment_tracker.shutdown()
            
            # Wait for shutdown
            await run_task
            
            assert experiment_tracker._shutdown_requested is True
    
    @pytest.mark.asyncio
    async def test_start_experiment_success(self, experiment_tracker, sample_queued_experiment, sample_experiment_metadata, mock_evaluation_runner):
        """Test successful experiment start."""
        # Mock database responses
        experiment_tracker.db_manager.get_experiment_metadata.return_value = sample_experiment_metadata
        experiment_tracker.db_manager.get_experiment_config.return_value = Mock()
        
        with patch('scramblebench.experiment_tracking.core.ProgressTracker') as mock_tracker_class, \
             patch('scramblebench.experiment_tracking.core.EvaluationRunner', return_value=mock_evaluation_runner), \
             patch.object(experiment_tracker, '_execute_experiment') as mock_execute:
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            await experiment_tracker._start_experiment(sample_queued_experiment)
            
            # Verify experiment was added to running experiments
            assert sample_queued_experiment.experiment_id in experiment_tracker.running_experiments
            
            # Verify task was created
            task = experiment_tracker.running_experiments[sample_queued_experiment.experiment_id]
            assert isinstance(task, asyncio.Task)
    
    @pytest.mark.asyncio
    async def test_start_experiment_failure(self, experiment_tracker, sample_queued_experiment):
        """Test experiment start failure handling."""
        # Mock database to raise error
        experiment_tracker.db_manager.get_experiment_metadata.side_effect = Exception("Database error")
        
        await experiment_tracker._start_experiment(sample_queued_experiment)
        
        # Verify failure status was set
        experiment_tracker.db_manager.update_experiment_status.assert_called_with(
            sample_queued_experiment.experiment_id, ExperimentStatus.FAILED
        )
    
    @pytest.mark.asyncio
    async def test_execute_experiment_success(self, experiment_tracker, sample_experiment_metadata, mock_evaluation_runner, mock_progress_tracker, sample_evaluation_results):
        """Test successful experiment execution."""
        with patch.object(experiment_tracker, '_run_with_progress_tracking', return_value=sample_evaluation_results) as mock_run, \
             patch.object(experiment_tracker, '_run_post_experiment_analysis') as mock_analysis:
            
            await experiment_tracker._execute_experiment(
                experiment_id=sample_experiment_metadata.experiment_id,
                eval_runner=mock_evaluation_runner,
                progress_tracker=mock_progress_tracker,
                metadata=sample_experiment_metadata
            )
            
            # Verify progress was initialized
            mock_progress_tracker.initialize.assert_called_once()
            
            # Verify evaluation was run
            mock_run.assert_called_once()
            
            # Verify results were saved
            experiment_tracker.db_manager.save_experiment_results.assert_called_once()
            
            # Verify metadata was updated
            assert sample_experiment_metadata.status == ExperimentStatus.COMPLETED
            assert sample_experiment_metadata.progress == 1.0
            assert sample_experiment_metadata.completed_at is not None
            
            # Verify post-analysis was run
            mock_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_experiment_failure(self, experiment_tracker, sample_experiment_metadata, mock_evaluation_runner, mock_progress_tracker):
        """Test experiment execution failure handling."""
        with patch.object(experiment_tracker, '_run_with_progress_tracking', side_effect=Exception("Execution failed")):
            
            await experiment_tracker._execute_experiment(
                experiment_id=sample_experiment_metadata.experiment_id,
                eval_runner=mock_evaluation_runner,
                progress_tracker=mock_progress_tracker,
                metadata=sample_experiment_metadata
            )
            
            # Verify failure status was set
            assert sample_experiment_metadata.status == ExperimentStatus.FAILED
            assert sample_experiment_metadata.completed_at is not None
            
            # Verify error was saved
            experiment_tracker.db_manager.save_experiment_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_with_progress_tracking(self, experiment_tracker, mock_evaluation_runner, mock_progress_tracker):
        """Test evaluation running with progress tracking."""
        # Mock evaluation task
        async def mock_eval():
            await asyncio.sleep(0.1)
            return Mock(results=[])
        
        mock_evaluation_runner.run_evaluation = mock_eval
        mock_evaluation_runner.get_status.return_value = {
            "current_stage": "running_evaluations",
            "progress": 0.5
        }
        
        result = await experiment_tracker._run_with_progress_tracking(
            mock_evaluation_runner, mock_progress_tracker
        )
        
        assert result is not None
        mock_progress_tracker.update_progress.assert_called()
    
    def test_calculate_progress(self, experiment_tracker):
        """Test progress calculation from runner status."""
        test_cases = [
            ({"current_stage": "initialized"}, 0.0),
            ({"current_stage": "loading_data"}, 0.1),
            ({"current_stage": "running_evaluations"}, 0.7),
            ({"current_stage": "completed"}, 1.0),
            ({"current_stage": "unknown_stage"}, 0.0)
        ]
        
        for status, expected_progress in test_cases:
            progress = experiment_tracker._calculate_progress(status)
            assert progress == expected_progress
    
    @pytest.mark.asyncio
    async def test_run_post_experiment_analysis(self, experiment_tracker, sample_evaluation_results):
        """Test post-experiment analysis execution."""
        experiment_id = "test-exp-analysis"
        
        await experiment_tracker._run_post_experiment_analysis(experiment_id, sample_evaluation_results)
        
        # Verify statistical analysis was called
        experiment_tracker.stats_analyzer.compute_performance_metrics.assert_called_once_with(
            experiment_id, sample_evaluation_results
        )
        experiment_tracker.stats_analyzer.analyze_thresholds.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_post_experiment_analysis_with_significance_tests(self, experiment_tracker, sample_evaluation_results):
        """Test post-experiment analysis with significance tests for large samples."""
        experiment_id = "test-exp-large"
        
        # Mock large sample size
        sample_evaluation_results.results = [Mock() for _ in range(50)]  # > 30 samples
        
        await experiment_tracker._run_post_experiment_analysis(experiment_id, sample_evaluation_results)
        
        # Verify significance tests were run
        experiment_tracker.stats_analyzer.run_significance_tests.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_post_experiment_analysis_error_handling(self, experiment_tracker, sample_evaluation_results):
        """Test post-experiment analysis error handling."""
        experiment_id = "test-exp-error"
        
        # Mock analysis to raise error
        experiment_tracker.stats_analyzer.compute_performance_metrics.side_effect = Exception("Analysis failed")
        
        # Should not raise exception (errors should be caught and logged)
        await experiment_tracker._run_post_experiment_analysis(experiment_id, sample_evaluation_results)
        
        # Verify error was handled gracefully
        experiment_tracker.stats_analyzer.compute_performance_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_estimate_resource_requirements(self, experiment_tracker, sample_experiment_config):
        """Test resource requirement estimation."""
        with patch('scramblebench.experiment_tracking.core.EvaluationConfig', return_value=Mock(
            models=[Mock(provider=Mock(value='openrouter')), Mock(provider=Mock(value='ollama'))],
            benchmark_paths=["bench1.json", "bench2.json"],
            max_samples=500
        )) as mock_config:
            
            requirements = await experiment_tracker._estimate_resource_requirements(mock_config.return_value)
            
            assert "api_calls" in requirements
            assert "estimated_cost" in requirements
            assert "estimated_duration_hours" in requirements
            assert "memory_gb" in requirements
            assert "storage_gb" in requirements
            
            # Verify reasonable estimates
            assert requirements["api_calls"] > 0
            assert requirements["estimated_duration_hours"] > 0
            assert requirements["memory_gb"] > 0
    
    @pytest.mark.asyncio
    async def test_estimate_duration(self, experiment_tracker, sample_experiment_config):
        """Test experiment duration estimation."""
        with patch('scramblebench.experiment_tracking.core.EvaluationConfig', return_value=Mock()) as mock_config, \
             patch.object(experiment_tracker, '_estimate_resource_requirements', return_value={"estimated_duration_hours": 2.5}):
            
            duration = await experiment_tracker._estimate_duration(mock_config.return_value)
            
            assert isinstance(duration, timedelta)
            assert duration.total_seconds() == 2.5 * 3600  # 2.5 hours in seconds
    
    def test_hash_config(self, experiment_tracker, sample_experiment_config):
        """Test configuration hashing for reproducibility."""
        with patch('scramblebench.experiment_tracking.core.EvaluationConfig') as mock_config_class:
            mock_config = Mock()
            mock_config.to_dict.return_value = sample_experiment_config
            
            hash1 = experiment_tracker._hash_config(mock_config)
            hash2 = experiment_tracker._hash_config(mock_config)
            
            assert hash1 == hash2  # Same config should produce same hash
            assert len(hash1) == 16  # Hash should be 16 characters (truncated SHA256)
    
    @pytest.mark.asyncio
    async def test_cleanup_completed_experiments(self, experiment_tracker):
        """Test cleanup of completed experiment tasks."""
        # Add some mock tasks
        completed_task = Mock()
        completed_task.done.return_value = True
        
        running_task = Mock()
        running_task.done.return_value = False
        
        experiment_tracker.running_experiments = {
            "completed-exp": completed_task,
            "running-exp": running_task
        }
        
        await experiment_tracker._cleanup_completed_experiments()
        
        # Verify only running experiment remains
        assert "completed-exp" not in experiment_tracker.running_experiments
        assert "running-exp" in experiment_tracker.running_experiments
    
    @pytest.mark.asyncio
    async def test_update_monitoring_data(self, experiment_tracker):
        """Test monitoring data update for all running experiments."""
        experiment_tracker.running_experiments = {
            "exp-1": Mock(),
            "exp-2": Mock()
        }
        
        await experiment_tracker._update_monitoring_data()
        
        # Verify monitor was called for each running experiment
        assert experiment_tracker.monitor.update_experiment_metrics.call_count == 2
    
    @pytest.mark.asyncio
    async def test_shutdown_all_experiments(self, experiment_tracker):
        """Test graceful shutdown of all running experiments."""
        # Create mock tasks
        task1 = Mock()
        task2 = Mock()
        
        experiment_tracker.running_experiments = {
            "exp-1": task1,
            "exp-2": task2
        }
        
        with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
            await experiment_tracker._shutdown_all_experiments()
            
            # Verify all tasks were cancelled
            task1.cancel.assert_called_once()
            task2.cancel.assert_called_once()
            
            # Verify gather was called to wait for completion
            mock_gather.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_experiment_status_running(self, experiment_tracker, sample_experiment_metadata):
        """Test getting status for running experiment."""
        sample_experiment_metadata.status = ExperimentStatus.RUNNING
        experiment_tracker.db_manager.get_experiment_metadata.return_value = sample_experiment_metadata
        
        progress_data = {"stage": "evaluating", "progress": 0.6}
        experiment_tracker.monitor.get_current_progress.return_value = progress_data
        
        status = await experiment_tracker.get_experiment_status(sample_experiment_metadata.experiment_id)
        
        assert status["experiment_id"] == sample_experiment_metadata.experiment_id
        assert status["status"] == "running"
        assert status["progress"] == sample_experiment_metadata.progress
        assert "stage" in status  # Progress data should be merged
    
    @pytest.mark.asyncio
    async def test_get_experiment_status_not_found(self, experiment_tracker):
        """Test getting status for non-existent experiment."""
        experiment_tracker.db_manager.get_experiment_metadata.return_value = None
        
        status = await experiment_tracker.get_experiment_status("non-existent")
        
        assert "error" in status
        assert status["error"] == "Experiment not found"
    
    @pytest.mark.asyncio
    async def test_cancel_experiment_running(self, experiment_tracker):
        """Test cancelling running experiment."""
        experiment_id = "running-exp"
        
        # Mock running task
        mock_task = Mock()
        experiment_tracker.running_experiments[experiment_id] = mock_task
        
        await experiment_tracker.cancel_experiment(experiment_id)
        
        # Verify task was cancelled
        mock_task.cancel.assert_called_once()
        assert experiment_id not in experiment_tracker.running_experiments
        
        # Verify queue removal and status update
        experiment_tracker.queue.remove_experiment.assert_called_once_with(experiment_id)
        experiment_tracker.db_manager.update_experiment_status.assert_called_once_with(
            experiment_id, ExperimentStatus.CANCELLED
        )
    
    @pytest.mark.asyncio
    async def test_cancel_experiment_queued(self, experiment_tracker):
        """Test cancelling queued experiment."""
        experiment_id = "queued-exp"
        
        await experiment_tracker.cancel_experiment(experiment_id)
        
        # Verify queue removal and status update (no running task to cancel)
        experiment_tracker.queue.remove_experiment.assert_called_once_with(experiment_id)
        experiment_tracker.db_manager.update_experiment_status.assert_called_once_with(
            experiment_id, ExperimentStatus.CANCELLED
        )
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, experiment_tracker):
        """Test listing experiments with filters."""
        mock_experiments = [
            {"experiment_id": "exp-1", "status": "completed"},
            {"experiment_id": "exp-2", "status": "running"}
        ]
        experiment_tracker.db_manager.list_experiments.return_value = mock_experiments
        
        result = await experiment_tracker.list_experiments(
            status=ExperimentStatus.COMPLETED,
            researcher="test@example.com",
            limit=50
        )
        
        assert result == mock_experiments
        experiment_tracker.db_manager.list_experiments.assert_called_once_with(
            status=ExperimentStatus.COMPLETED,
            researcher="test@example.com",
            limit=50
        )
    
    def test_shutdown(self, experiment_tracker):
        """Test shutdown request."""
        assert experiment_tracker._shutdown_requested is False
        
        experiment_tracker.shutdown()
        
        assert experiment_tracker._shutdown_requested is True


class TestExperimentTrackerIntegration:
    """Integration tests for ExperimentTracker with real components."""
    
    @pytest.mark.asyncio
    async def test_full_experiment_lifecycle_simulation(self, temp_experiment_dir, mock_database_url):
        """Test complete experiment lifecycle from creation to completion."""
        with patch('scramblebench.experiment_tracking.core.DatabaseManager') as mock_db_class, \
             patch('scramblebench.experiment_tracking.core.EvaluationRunner') as mock_runner_class:
            
            # Set up mocks
            mock_db = AsyncMock()
            mock_db_class.return_value = mock_db
            
            mock_runner = AsyncMock()
            mock_runner.get_status.return_value = {"current_stage": "completed"}
            mock_runner.run_evaluation.return_value = Mock(results=[])
            mock_runner_class.return_value = mock_runner
            
            tracker = ExperimentTracker(
                database_url=mock_database_url,
                data_dir=temp_experiment_dir
            )
            
            # Create experiment
            experiment_id = await tracker.create_experiment(
                name="Integration Test",
                config={"models": [], "benchmarks": []},
                description="Full lifecycle test",
                research_question="Integration testing?",
                researcher_name="Test Suite"
            )
            
            # Queue experiment
            mock_db.get_experiment_metadata.return_value = Mock(
                experiment_id=experiment_id,
                status=ExperimentStatus.PLANNED
            )
            mock_db.get_experiment_config.return_value = Mock()
            
            await tracker.queue_experiment(experiment_id, priority=1)
            
            # Verify lifecycle steps were called
            mock_db.create_experiment.assert_called_once()
            mock_db.get_experiment_metadata.assert_called()
            mock_db.update_experiment_status.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_experiment_management(self, experiment_tracker):
        """Test managing multiple concurrent experiments."""
        # Create multiple queued experiments
        queued_experiments = [
            QueuedExperiment(
                experiment_id=f"concurrent-exp-{i}",
                priority=i,
                resource_requirements=ResourceRequirements(memory_gb=1.0)
            )
            for i in range(3)
        ]
        
        # Mock queue to return experiments
        experiment_tracker.queue.get_ready_experiments.return_value = queued_experiments
        
        # Mock database responses
        for exp in queued_experiments:
            experiment_tracker.db_manager.get_experiment_metadata.return_value = Mock(
                experiment_id=exp.experiment_id,
                status=ExperimentStatus.QUEUED
            )
        
        with patch.object(experiment_tracker, '_execute_experiment', new_callable=AsyncMock):
            # Start experiments
            await experiment_tracker.run_experiments(continuous=False)
            
            # Verify all experiments were started
            assert len(experiment_tracker.running_experiments) == len(queued_experiments)
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self, experiment_tracker, sample_queued_experiment):
        """Test error recovery and experiment retry logic."""
        # Mock database to raise error on first call, succeed on second
        experiment_tracker.db_manager.get_experiment_metadata.side_effect = [
            Exception("Temporary database error"),
            Mock(experiment_id=sample_queued_experiment.experiment_id, status=ExperimentStatus.QUEUED)
        ]
        
        # First attempt should fail gracefully
        await experiment_tracker._start_experiment(sample_queued_experiment)
        
        # Verify failure was recorded
        experiment_tracker.db_manager.update_experiment_status.assert_called_with(
            sample_queued_experiment.experiment_id, ExperimentStatus.FAILED
        )
        
        # Reset mock for retry
        experiment_tracker.db_manager.update_experiment_status.reset_mock()
        
        # Second attempt should succeed
        await experiment_tracker._start_experiment(sample_queued_experiment)
        
        # Verify experiment was started (metadata update called)
        experiment_tracker.db_manager.update_experiment_metadata.assert_called()