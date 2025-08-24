"""
Tests for repository pattern implementations.

Comprehensive tests for BaseRepository and all specific repository classes,
covering CRUD operations, complex queries, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from scramblebench.db.repository import (
    BaseRepository, RunRepository, ItemRepository, EvaluationRepository,
    AggregateRepository, ParaphraseCacheRepository, RepositoryFactory
)
from scramblebench.db.models import Run, Item, Evaluation, Aggregate, ParaphraseCache


class TestBaseRepository:
    """Test cases for BaseRepository generic functionality."""
    
    def test_create_success(self, run_repository, sample_run_data):
        """Test successful entity creation."""
        run = run_repository.create(**sample_run_data)
        
        assert run is not None
        assert run.run_id == sample_run_data["run_id"]
        assert run.name == sample_run_data["name"]
        assert hasattr(run, 'id')  # Should have generated ID
    
    def test_create_with_invalid_data(self, run_repository):
        """Test creation with invalid data raises appropriate error."""
        with pytest.raises(Exception):  # Could be TypeError, IntegrityError, etc.
            run_repository.create(invalid_field="invalid_value")
    
    def test_get_by_id_success(self, run_repository, sample_run_data):
        """Test successful entity retrieval by ID."""
        # Create entity first
        created_run = run_repository.create(**sample_run_data)
        
        # Retrieve by ID
        retrieved_run = run_repository.get_by_id(created_run.id)
        
        assert retrieved_run is not None
        assert retrieved_run.id == created_run.id
        assert retrieved_run.run_id == sample_run_data["run_id"]
    
    def test_get_by_id_not_found(self, run_repository):
        """Test get_by_id returns None for non-existent ID."""
        result = run_repository.get_by_id(99999)
        assert result is None
    
    def test_get_all_empty_table(self, run_repository):
        """Test get_all with empty table."""
        results = run_repository.get_all()
        assert results == []
    
    def test_get_all_with_data(self, run_repository, multiple_runs_data):
        """Test get_all with populated table."""
        # Create multiple runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        results = run_repository.get_all()
        assert len(results) == len(multiple_runs_data)
    
    def test_get_all_with_limit(self, run_repository, multiple_runs_data):
        """Test get_all with limit parameter."""
        # Create multiple runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        results = run_repository.get_all(limit=2)
        assert len(results) == 2
    
    def test_get_all_with_offset(self, run_repository, multiple_runs_data):
        """Test get_all with offset parameter."""
        # Create multiple runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        all_results = run_repository.get_all()
        offset_results = run_repository.get_all(offset=1)
        
        assert len(offset_results) == len(all_results) - 1
    
    def test_get_all_with_limit_and_offset(self, run_repository, multiple_runs_data):
        """Test get_all with both limit and offset."""
        # Create multiple runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        results = run_repository.get_all(limit=1, offset=1)
        assert len(results) == 1
    
    def test_update_success(self, run_repository, sample_run_data):
        """Test successful entity update."""
        # Create entity
        run = run_repository.create(**sample_run_data)
        
        # Update entity
        updated_run = run_repository.update(run.id, name="Updated Name", status="completed")
        
        assert updated_run is not None
        assert updated_run.id == run.id
        assert updated_run.name == "Updated Name"
        assert updated_run.status == "completed"
    
    def test_update_not_found(self, run_repository):
        """Test update returns None for non-existent entity."""
        result = run_repository.update(99999, name="Updated Name")
        assert result is None
    
    def test_update_with_invalid_field(self, run_repository, sample_run_data):
        """Test update ignores invalid fields gracefully."""
        # Create entity
        run = run_repository.create(**sample_run_data)
        
        # Update with invalid field (should be ignored)
        updated_run = run_repository.update(
            run.id, 
            name="Updated Name",
            invalid_field="should_be_ignored"
        )
        
        assert updated_run is not None
        assert updated_run.name == "Updated Name"
        # Invalid field should be ignored, not cause error
    
    def test_delete_success(self, run_repository, sample_run_data):
        """Test successful entity deletion."""
        # Create entity
        run = run_repository.create(**sample_run_data)
        entity_id = run.id
        
        # Delete entity
        deleted = run_repository.delete(entity_id)
        assert deleted is True
        
        # Verify deletion
        retrieved = run_repository.get_by_id(entity_id)
        assert retrieved is None
    
    def test_delete_not_found(self, run_repository):
        """Test delete returns False for non-existent entity."""
        result = run_repository.delete(99999)
        assert result is False
    
    def test_count_empty_table(self, run_repository):
        """Test count with empty table."""
        count = run_repository.count()
        assert count == 0
    
    def test_count_with_data(self, run_repository, multiple_runs_data):
        """Test count with populated table."""
        # Create multiple runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        count = run_repository.count()
        assert count == len(multiple_runs_data)
    
    def test_exists_true(self, run_repository, sample_run_data):
        """Test exists returns True for existing entity."""
        # Create entity
        run_repository.create(**sample_run_data)
        
        # Check existence
        exists = run_repository.exists(run_id=sample_run_data["run_id"])
        assert exists is True
    
    def test_exists_false(self, run_repository):
        """Test exists returns False for non-existent entity."""
        exists = run_repository.exists(run_id="non-existent-id")
        assert exists is False
    
    def test_exists_with_multiple_filters(self, run_repository, sample_run_data):
        """Test exists with multiple filter conditions."""
        # Create entity
        run_repository.create(**sample_run_data)
        
        # Check existence with multiple filters
        exists = run_repository.exists(
            run_id=sample_run_data["run_id"],
            status=sample_run_data["status"]
        )
        assert exists is True
        
        # Check with mismatched filters
        exists = run_repository.exists(
            run_id=sample_run_data["run_id"],
            status="different_status"
        )
        assert exists is False


class TestRunRepository:
    """Test cases for RunRepository specific functionality."""
    
    def test_get_by_run_id_success(self, run_repository, sample_run_data):
        """Test get run by run_id."""
        # Create run
        run_repository.create(**sample_run_data)
        
        # Retrieve by run_id
        retrieved = run_repository.get_by_run_id(sample_run_data["run_id"])
        
        assert retrieved is not None
        assert retrieved.run_id == sample_run_data["run_id"]
        assert retrieved.name == sample_run_data["name"]
    
    def test_get_by_run_id_not_found(self, run_repository):
        """Test get_by_run_id returns None for non-existent run_id."""
        result = run_repository.get_by_run_id("non-existent-run-id")
        assert result is None
    
    def test_get_active_runs(self, run_repository, multiple_runs_data):
        """Test getting active (running) runs."""
        # Create runs with different statuses
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        active_runs = run_repository.get_active_runs()
        
        # Should only return runs with status='running'
        assert len(active_runs) == 1
        assert all(run.status == "running" for run in active_runs)
    
    def test_get_completed_runs(self, run_repository, multiple_runs_data):
        """Test getting completed runs."""
        # Create runs with different statuses
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        completed_runs = run_repository.get_completed_runs()
        
        # Should only return completed runs, ordered by completion date
        assert len(completed_runs) == 1
        assert all(run.status == "completed" for run in completed_runs)
        assert completed_runs[0].completed_at is not None
    
    def test_get_completed_runs_with_limit(self, run_repository):
        """Test getting completed runs with limit."""
        # Create multiple completed runs
        base_time = datetime.utcnow()
        for i in range(3):
            run_data = {
                "run_id": f"completed-run-{i}",
                "name": f"Completed Run {i}",
                "config_hash": f"hash{i}",
                "benchmark": "test",
                "model_ids": ["test-model"],
                "total_evaluations": 100,
                "completed_evaluations": 100,
                "status": "completed",
                "created_at": base_time - timedelta(days=i+1),
                "completed_at": base_time - timedelta(hours=i)
            }
            run_repository.create(**run_data)
        
        completed_runs = run_repository.get_completed_runs(limit=2)
        
        assert len(completed_runs) == 2
        # Should be ordered by most recent completion first
        assert completed_runs[0].completed_at >= completed_runs[1].completed_at
    
    def test_get_runs_by_config_hash(self, run_repository, multiple_runs_data):
        """Test getting runs by config hash."""
        # Create runs
        for run_data in multiple_runs_data:
            run_repository.create(**run_data)
        
        # Get runs with specific config hash
        target_hash = multiple_runs_data[0]["config_hash"]
        matching_runs = run_repository.get_runs_by_config_hash(target_hash)
        
        assert len(matching_runs) == 1
        assert all(run.config_hash == target_hash for run in matching_runs)
    
    def test_update_progress(self, run_repository, sample_run_data):
        """Test updating run progress."""
        # Create run
        run = run_repository.create(**sample_run_data)
        
        # Update progress
        updated_run = run_repository.update_progress(run.run_id, 750)
        
        assert updated_run is not None
        assert updated_run.completed_evaluations == 750
    
    def test_mark_completed_default_timestamp(self, run_repository, sample_run_data):
        """Test marking run as completed with default timestamp."""
        # Create run
        run = run_repository.create(**sample_run_data)
        
        # Mark completed
        completed_run = run_repository.mark_completed(run.run_id)
        
        assert completed_run is not None
        assert completed_run.status == "completed"
        assert completed_run.completed_at is not None
        assert abs((completed_run.completed_at - datetime.utcnow()).total_seconds()) < 10
    
    def test_mark_completed_custom_timestamp(self, run_repository, sample_run_data):
        """Test marking run as completed with custom timestamp."""
        # Create run
        run = run_repository.create(**sample_run_data)
        
        # Mark completed with custom timestamp
        custom_time = datetime.utcnow() - timedelta(hours=1)
        completed_run = run_repository.mark_completed(run.run_id, custom_time)
        
        assert completed_run is not None
        assert completed_run.status == "completed"
        assert completed_run.completed_at == custom_time
    
    def test_mark_failed_without_error(self, run_repository, sample_run_data):
        """Test marking run as failed without error message."""
        # Create run
        run = run_repository.create(**sample_run_data)
        
        # Mark failed
        failed_run = run_repository.mark_failed(run.run_id)
        
        assert failed_run is not None
        assert failed_run.status == "failed"
    
    def test_mark_failed_with_error_message(self, run_repository, sample_run_data):
        """Test marking run as failed with error message."""
        # Create run
        run = run_repository.create(**sample_run_data)
        
        # Mark failed with error
        error_msg = "API timeout occurred"
        failed_run = run_repository.mark_failed(run.run_id, error_msg)
        
        assert failed_run is not None
        assert failed_run.status == "failed"
        assert failed_run.meta_data == error_msg


class TestItemRepository:
    """Test cases for ItemRepository specific functionality."""
    
    def test_get_by_item_id_success(self, item_repository, sample_item_data):
        """Test get item by item_id."""
        # Create item
        item_repository.create(**sample_item_data)
        
        # Retrieve by item_id
        retrieved = item_repository.get_by_item_id(sample_item_data["item_id"])
        
        assert retrieved is not None
        assert retrieved.item_id == sample_item_data["item_id"]
        assert retrieved.question == sample_item_data["question"]
    
    def test_get_by_dataset(self, item_repository, multiple_items_data):
        """Test getting items by dataset."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        # Get MMLU items
        mmlu_items = item_repository.get_by_dataset("mmlu")
        assert len(mmlu_items) == 2
        assert all(item.dataset == "mmlu" for item in mmlu_items)
        
        # Get HellaSwag items
        hellaswag_items = item_repository.get_by_dataset("hellaswag")
        assert len(hellaswag_items) == 1
        assert all(item.dataset == "hellaswag" for item in hellaswag_items)
    
    def test_get_by_dataset_and_domain(self, item_repository, multiple_items_data):
        """Test getting items by dataset and domain."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        # Get MMLU math items
        math_items = item_repository.get_by_dataset("mmlu", domain="math")
        assert len(math_items) == 1
        assert math_items[0].domain == "math"
        
        # Get MMLU science items
        science_items = item_repository.get_by_dataset("mmlu", domain="science")
        assert len(science_items) == 1
        assert science_items[0].domain == "science"
    
    def test_get_datasets(self, item_repository, multiple_items_data):
        """Test getting unique datasets."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        datasets = item_repository.get_datasets()
        
        assert len(datasets) == 2
        assert "mmlu" in datasets
        assert "hellaswag" in datasets
    
    def test_get_domains_all(self, item_repository, multiple_items_data):
        """Test getting all unique domains."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        domains = item_repository.get_domains()
        
        # Should exclude None values
        assert len(domains) == 2
        assert "math" in domains
        assert "science" in domains
        assert None not in domains
    
    def test_get_domains_by_dataset(self, item_repository, multiple_items_data):
        """Test getting domains for specific dataset."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        mmlu_domains = item_repository.get_domains("mmlu")
        assert len(mmlu_domains) == 2
        assert "math" in mmlu_domains
        assert "science" in mmlu_domains
        
        hellaswag_domains = item_repository.get_domains("hellaswag")
        assert len(hellaswag_domains) == 0  # HellaSwag has None domain
    
    def test_search_by_question(self, item_repository, multiple_items_data):
        """Test searching items by question text."""
        # Create items
        for item_data in multiple_items_data:
            item_repository.create(**item_data)
        
        # Search for math-related questions
        results = item_repository.search_by_question("5 * 6")
        assert len(results) == 1
        assert "5 * 6" in results[0].question
        
        # Search for water-related questions
        results = item_repository.search_by_question("H2O")
        assert len(results) == 1
        assert "H2O" in results[0].question
    
    def test_search_by_question_with_limit(self, item_repository):
        """Test search with limit parameter."""
        # Create many items with similar questions
        for i in range(5):
            item_data = {
                "item_id": f"test-item-{i}",
                "dataset": "test",
                "question": f"Test question {i} about testing",
                "answer": f"Answer {i}"
            }
            item_repository.create(**item_data)
        
        results = item_repository.search_by_question("test", limit=3)
        assert len(results) == 3


class TestEvaluationRepository:
    """Test cases for EvaluationRepository specific functionality."""
    
    def test_get_by_eval_id(self, evaluation_repository, sample_evaluation_data):
        """Test get evaluation by eval_id."""
        # Create evaluation
        evaluation_repository.create(**sample_evaluation_data)
        
        # Retrieve by eval_id
        retrieved = evaluation_repository.get_by_eval_id(sample_evaluation_data["eval_id"])
        
        assert retrieved is not None
        assert retrieved.eval_id == sample_evaluation_data["eval_id"]
        assert retrieved.model_id == sample_evaluation_data["model_id"]
    
    def test_get_by_run(self, evaluation_repository, performance_test_data):
        """Test getting evaluations by run."""
        # Create run and evaluations
        for eval_data in performance_test_data["evaluations"][:10]:  # Just use 10 for this test
            evaluation_repository.create(**eval_data)
        
        # Get evaluations for run
        run_evals = evaluation_repository.get_by_run("performance-run")
        
        assert len(run_evals) == 10
        assert all(eval.run_id == "performance-run" for eval in run_evals)
        
        # Should be ordered by timestamp
        timestamps = [eval.timestamp for eval in run_evals]
        assert timestamps == sorted(timestamps)
    
    def test_get_by_run_with_limit(self, evaluation_repository, performance_test_data):
        """Test getting evaluations by run with limit."""
        # Create evaluations
        for eval_data in performance_test_data["evaluations"][:10]:
            evaluation_repository.create(**eval_data)
        
        # Get limited evaluations
        run_evals = evaluation_repository.get_by_run("performance-run", limit=5)
        
        assert len(run_evals) == 5
    
    def test_get_by_model(self, evaluation_repository, performance_test_data):
        """Test getting evaluations by model."""
        # Create evaluations
        for eval_data in performance_test_data["evaluations"][:20]:  # Mix of models
            evaluation_repository.create(**eval_data)
        
        # Get evaluations for specific model
        model_evals = evaluation_repository.get_by_model("gpt-4")
        
        assert len(model_evals) > 0
        assert all(eval.model_id == "gpt-4" for eval in model_evals)
    
    def test_get_by_model_and_run(self, evaluation_repository, performance_test_data):
        """Test getting evaluations by model and run."""
        # Create evaluations
        for eval_data in performance_test_data["evaluations"][:20]:
            evaluation_repository.create(**eval_data)
        
        # Get evaluations for specific model and run
        model_run_evals = evaluation_repository.get_by_model("gpt-4", run_id="performance-run")
        
        assert len(model_run_evals) > 0
        assert all(eval.model_id == "gpt-4" and eval.run_id == "performance-run" for eval in model_run_evals)
    
    def test_get_by_transform(self, evaluation_repository, performance_test_data):
        """Test getting evaluations by transform."""
        # Create evaluations
        for eval_data in performance_test_data["evaluations"][:20]:
            evaluation_repository.create(**eval_data)
        
        # Get evaluations for original transform
        original_evals = evaluation_repository.get_by_transform("original")
        
        assert len(original_evals) > 0
        assert all(eval.transform == "original" for eval in original_evals)
    
    def test_get_correct_evaluations(self, evaluation_repository, performance_test_data):
        """Test getting only correct evaluations."""
        # Create evaluations
        for eval_data in performance_test_data["evaluations"][:20]:
            evaluation_repository.create(**eval_data)
        
        # Get correct evaluations
        correct_evals = evaluation_repository.get_correct_evaluations()
        
        assert len(correct_evals) > 0
        assert all(eval.is_correct is True for eval in correct_evals)
    
    def test_get_evaluation_stats(self, evaluation_repository, performance_test_data):
        """Test getting evaluation statistics for a run."""
        # Create evaluations
        evaluations_data = performance_test_data["evaluations"][:30]  # Use subset
        for eval_data in evaluations_data:
            evaluation_repository.create(**eval_data)
        
        # Get stats
        stats = evaluation_repository.get_evaluation_stats("performance-run")
        
        assert stats["total_evaluations"] == 30
        assert stats["completed_evaluations"] <= stats["total_evaluations"]
        assert 0 <= stats["accuracy"] <= 1
        assert stats["avg_latency_ms"] > 0
        assert stats["total_cost_usd"] >= 0
    
    def test_get_model_performance(self, evaluation_repository, performance_test_data):
        """Test getting performance metrics by model."""
        # Create evaluations
        evaluations_data = performance_test_data["evaluations"][:30]
        for eval_data in evaluations_data:
            evaluation_repository.create(**eval_data)
        
        # Get model performance
        performance = evaluation_repository.get_model_performance("performance-run")
        
        assert len(performance) > 0
        
        for model_stats in performance:
            assert "model_id" in model_stats
            assert "total_evaluations" in model_stats
            assert "correct_evaluations" in model_stats
            assert "accuracy" in model_stats
            assert 0 <= model_stats["accuracy"] <= 1
            assert model_stats["avg_latency_ms"] >= 0
            assert model_stats["total_cost_usd"] >= 0


class TestAggregateRepository:
    """Test cases for AggregateRepository specific functionality."""
    
    def test_get_by_run(self, aggregate_repository, sample_aggregate_data):
        """Test getting aggregates by run."""
        # Create aggregate
        aggregate_repository.create(**sample_aggregate_data)
        
        # Get by run
        run_aggregates = aggregate_repository.get_by_run(sample_aggregate_data["run_id"])
        
        assert len(run_aggregates) == 1
        assert run_aggregates[0].run_id == sample_aggregate_data["run_id"]
    
    def test_get_by_model_and_transform(self, aggregate_repository):
        """Test getting aggregates by model and transform."""
        # Create multiple aggregates
        base_data = {
            "run_id": "test-run",
            "domain": "math",
            "acc_mean": 0.8,
            "n_samples": 100
        }
        
        # Create aggregates for different model/transform combinations
        combinations = [
            ("gpt-4", "original"),
            ("gpt-4", "scrambled"),
            ("claude-3", "original")
        ]
        
        for model_id, transform in combinations:
            data = base_data.copy()
            data.update({"model_id": model_id, "transform": transform})
            aggregate_repository.create(**data)
        
        # Get specific combination
        results = aggregate_repository.get_by_model_and_transform("gpt-4", "original")
        
        assert len(results) == 1
        assert results[0].model_id == "gpt-4"
        assert results[0].transform == "original"
    
    def test_get_model_comparison(self, aggregate_repository):
        """Test getting model comparison data."""
        # Create aggregates for comparison
        base_data = {
            "run_id": "comparison-run",
            "domain": "math",
            "n_samples": 100
        }
        
        test_data = [
            {"model_id": "gpt-4", "transform": "original", "acc_mean": 0.9, "rrs": 0.8, "ldc": 0.7},
            {"model_id": "gpt-4", "transform": "scrambled", "acc_mean": 0.7, "rrs": 0.6, "ldc": 0.5},
            {"model_id": "claude-3", "transform": "original", "acc_mean": 0.85, "rrs": 0.75, "ldc": 0.65}
        ]
        
        for data in test_data:
            full_data = {**base_data, **data}
            aggregate_repository.create(**full_data)
        
        # Get comparison data
        comparison = aggregate_repository.get_model_comparison("comparison-run")
        
        assert len(comparison) == 3
        
        # Check data structure
        for entry in comparison:
            assert "model_id" in entry
            assert "transform" in entry
            assert "avg_accuracy" in entry
            assert entry["avg_accuracy"] > 0
    
    def test_get_best_performing_models(self, aggregate_repository):
        """Test getting best performing models."""
        # Create aggregates with different performance levels
        base_data = {
            "run_id": "best-models-run",
            "transform": "original",
            "domain": "math",
            "n_samples": 100
        }
        
        models_performance = [
            {"model_id": "excellent-model", "acc_mean": 0.95},
            {"model_id": "good-model", "acc_mean": 0.85},
            {"model_id": "average-model", "acc_mean": 0.75},
            {"model_id": "poor-model", "acc_mean": 0.65}
        ]
        
        for model_data in models_performance:
            full_data = {**base_data, **model_data}
            aggregate_repository.create(**full_data)
        
        # Get best performing models
        best_models = aggregate_repository.get_best_performing_models("best-models-run", limit=2)
        
        assert len(best_models) == 2
        assert best_models[0].acc_mean >= best_models[1].acc_mean  # Should be sorted by performance
        assert best_models[0].model_id == "excellent-model"


class TestParaphraseCacheRepository:
    """Test cases for ParaphraseCacheRepository specific functionality."""
    
    def test_get_by_item_all_paraphrases(self, paraphrase_repository):
        """Test getting all paraphrases for an item."""
        # Create multiple paraphrases for same item
        base_data = {
            "item_id": "test-item-paraphrases",
            "provider": "openai",
            "paraphrase": "Paraphrase text",
            "cos_sim": 0.9
        }
        
        paraphrases_data = [
            {**base_data, "candidate_id": 1, "accepted": True},
            {**base_data, "candidate_id": 2, "accepted": False},
            {**base_data, "candidate_id": 3, "accepted": True}
        ]
        
        for data in paraphrases_data:
            paraphrase_repository.create(**data)
        
        # Get all paraphrases (should only return accepted by default)
        accepted_paraphrases = paraphrase_repository.get_by_item("test-item-paraphrases")
        assert len(accepted_paraphrases) == 2
        assert all(p.accepted is True for p in accepted_paraphrases)
        
        # Get all paraphrases including rejected
        all_paraphrases = paraphrase_repository.get_by_item("test-item-paraphrases", accepted_only=False)
        assert len(all_paraphrases) == 3
    
    def test_get_by_provider(self, paraphrase_repository):
        """Test getting paraphrases by provider."""
        # Create paraphrases from different providers
        providers_data = [
            {"item_id": "item-1", "candidate_id": 1, "provider": "openai", "accepted": True},
            {"item_id": "item-2", "candidate_id": 1, "provider": "anthropic", "accepted": True},
            {"item_id": "item-3", "candidate_id": 1, "provider": "openai", "accepted": False}
        ]
        
        base_data = {"paraphrase": "Test paraphrase", "cos_sim": 0.9}
        
        for data in providers_data:
            full_data = {**base_data, **data}
            paraphrase_repository.create(**full_data)
        
        # Get OpenAI paraphrases (accepted only)
        openai_paraphrases = paraphrase_repository.get_by_provider("openai")
        assert len(openai_paraphrases) == 1
        assert openai_paraphrases[0].accepted is True
        
        # Get all OpenAI paraphrases
        all_openai = paraphrase_repository.get_by_provider("openai", accepted_only=False)
        assert len(all_openai) == 2
    
    def test_get_accepted_paraphrase(self, paraphrase_repository):
        """Test getting the accepted paraphrase for an item."""
        # Create multiple paraphrases with one accepted
        base_data = {
            "item_id": "accepted-test-item",
            "provider": "openai",
            "paraphrase": "Test paraphrase",
            "cos_sim": 0.9
        }
        
        paraphrases_data = [
            {**base_data, "candidate_id": 1, "accepted": False},
            {**base_data, "candidate_id": 2, "accepted": True},
            {**base_data, "candidate_id": 3, "accepted": False}
        ]
        
        for data in paraphrases_data:
            paraphrase_repository.create(**data)
        
        # Get accepted paraphrase
        accepted = paraphrase_repository.get_accepted_paraphrase("accepted-test-item")
        
        assert accepted is not None
        assert accepted.candidate_id == 2
        assert accepted.accepted is True
    
    def test_get_accepted_paraphrase_none_accepted(self, paraphrase_repository):
        """Test getting accepted paraphrase when none accepted."""
        # Create paraphrases with none accepted
        base_data = {
            "item_id": "no-accepted-item",
            "candidate_id": 1,
            "provider": "openai",
            "paraphrase": "Test paraphrase",
            "cos_sim": 0.9,
            "accepted": False
        }
        
        paraphrase_repository.create(**base_data)
        
        # Should return None
        accepted = paraphrase_repository.get_accepted_paraphrase("no-accepted-item")
        assert accepted is None
    
    def test_accept_paraphrase(self, paraphrase_repository, sample_paraphrase_data):
        """Test accepting a paraphrase."""
        # Create paraphrase
        paraphrase = paraphrase_repository.create(**sample_paraphrase_data)
        
        # Accept it
        updated = paraphrase_repository.accept_paraphrase(paraphrase.id)
        
        assert updated is not None
        assert updated.accepted is True
    
    def test_get_quality_stats(self, paraphrase_repository):
        """Test getting paraphrase quality statistics."""
        # Create paraphrases with various quality metrics
        paraphrases_data = [
            {
                "item_id": "item-1", "candidate_id": 1, "provider": "openai",
                "paraphrase": "High quality", "cos_sim": 0.95, "edit_ratio": 0.8,
                "bleu_score": 0.9, "accepted": True
            },
            {
                "item_id": "item-2", "candidate_id": 1, "provider": "openai",
                "paraphrase": "Medium quality", "cos_sim": 0.75, "edit_ratio": 0.6,
                "bleu_score": 0.7, "accepted": False
            },
            {
                "item_id": "item-3", "candidate_id": 1, "provider": "openai",
                "paraphrase": "Low quality", "cos_sim": 0.55, "edit_ratio": 0.4,
                "bleu_score": 0.5, "accepted": None  # Not evaluated yet
            }
        ]
        
        for data in paraphrases_data:
            paraphrase_repository.create(**data)
        
        # Get quality stats
        stats = paraphrase_repository.get_quality_stats()
        
        assert stats["total_paraphrases"] == 3
        assert stats["evaluated_paraphrases"] == 2  # Only those with non-None accepted
        assert stats["accepted_paraphrases"] == 1
        assert stats["acceptance_rate"] == 0.5  # 1/2
        assert 0 < stats["avg_cos_sim"] < 1
        assert 0 < stats["avg_edit_ratio"] < 1
        assert 0 < stats["avg_bleu_score"] < 1
    
    def test_find_high_quality_candidates(self, paraphrase_repository):
        """Test finding high-quality paraphrase candidates."""
        # Create paraphrases with different quality levels
        candidates_data = [
            {
                "item_id": "item-1", "candidate_id": 1, "provider": "openai",
                "paraphrase": "High quality candidate", "cos_sim": 0.9, 
                "edit_ratio": 0.3, "accepted": False
            },
            {
                "item_id": "item-2", "candidate_id": 1, "provider": "openai",
                "paraphrase": "Low similarity", "cos_sim": 0.6, 
                "edit_ratio": 0.4, "accepted": False
            },
            {
                "item_id": "item-3", "candidate_id": 1, "provider": "openai",
                "paraphrase": "Low edit ratio", "cos_sim": 0.9, 
                "edit_ratio": 0.1, "accepted": False
            },
            {
                "item_id": "item-4", "candidate_id": 1, "provider": "openai",
                "paraphrase": "Already accepted", "cos_sim": 0.9, 
                "edit_ratio": 0.4, "accepted": True
            }
        ]
        
        for data in candidates_data:
            paraphrase_repository.create(**data)
        
        # Find high quality candidates
        candidates = paraphrase_repository.find_high_quality_candidates(
            min_cos_sim=0.85,
            min_edit_ratio=0.25
        )
        
        # Should only return item-1 (high quality, not accepted, meets thresholds)
        assert len(candidates) == 1
        assert candidates[0].item_id == "item-1"
        assert candidates[0].accepted is False


class TestRepositoryFactory:
    """Test cases for RepositoryFactory."""
    
    def test_factory_initialization(self, db_manager):
        """Test RepositoryFactory initialization."""
        factory = RepositoryFactory(db_manager)
        
        assert factory.db_manager is db_manager
        assert factory._repositories == {}
    
    def test_get_run_repository(self, repository_factory):
        """Test getting RunRepository from factory."""
        repo = repository_factory.get_run_repository()
        
        assert isinstance(repo, RunRepository)
        assert repo.model_class is Run
        
        # Should return same instance on subsequent calls
        repo2 = repository_factory.get_run_repository()
        assert repo is repo2
    
    def test_get_item_repository(self, repository_factory):
        """Test getting ItemRepository from factory."""
        repo = repository_factory.get_item_repository()
        
        assert isinstance(repo, ItemRepository)
        assert repo.model_class is Item
    
    def test_get_evaluation_repository(self, repository_factory):
        """Test getting EvaluationRepository from factory."""
        repo = repository_factory.get_evaluation_repository()
        
        assert isinstance(repo, EvaluationRepository)
        assert repo.model_class is Evaluation
    
    def test_get_aggregate_repository(self, repository_factory):
        """Test getting AggregateRepository from factory."""
        repo = repository_factory.get_aggregate_repository()
        
        assert isinstance(repo, AggregateRepository)
        assert repo.model_class is Aggregate
    
    def test_get_paraphrase_cache_repository(self, repository_factory):
        """Test getting ParaphraseCacheRepository from factory."""
        repo = repository_factory.get_paraphrase_cache_repository()
        
        assert isinstance(repo, ParaphraseCacheRepository)
        assert repo.model_class is ParaphraseCache
    
    def test_repository_caching(self, repository_factory):
        """Test that repositories are cached in factory."""
        # Get different repository types
        run_repo1 = repository_factory.get_run_repository()
        item_repo1 = repository_factory.get_item_repository()
        
        # Get same repositories again
        run_repo2 = repository_factory.get_run_repository()
        item_repo2 = repository_factory.get_item_repository()
        
        # Should be same instances
        assert run_repo1 is run_repo2
        assert item_repo1 is item_repo2
        
        # But different repository types should be different instances
        assert run_repo1 is not item_repo1


class TestRepositoryErrorHandling:
    """Test error handling scenarios for repositories."""
    
    def test_repository_database_error_handling(self, run_repository):
        """Test repository error handling with database errors."""
        with patch.object(run_repository.db_manager, 'session_scope') as mock_scope:
            mock_scope.side_effect = SQLAlchemyError("Database connection failed")
            
            with pytest.raises(SQLAlchemyError, match="Database connection failed"):
                run_repository.get_all()
    
    def test_repository_transaction_rollback(self, run_repository, sample_run_data, transaction_test_helper):
        """Test repository transaction rollback on errors."""
        with transaction_test_helper.expect_rollback():
            with patch.object(run_repository.db_manager, 'session_scope') as mock_scope:
                # Mock session that raises error during commit
                mock_session = Mock()
                mock_scope.return_value.__enter__.return_value = mock_session
                mock_scope.return_value.__exit__.side_effect = SQLAlchemyError("Commit failed")
                
                run_repository.create(**sample_run_data)
    
    def test_repository_with_invalid_model_class(self, db_manager):
        """Test repository with invalid model class."""
        class InvalidModel:
            pass
        
        # This should work during initialization
        repo = BaseRepository(db_manager, InvalidModel)
        
        # But should fail when trying to use it
        with pytest.raises(Exception):
            repo.create(test_field="test_value")
    
    @patch('scramblebench.db.repository.logger')
    def test_repository_logging(self, mock_logger, run_repository, sample_run_data):
        """Test that repository operations are properly logged."""
        # Create entity (should log creation)
        run = run_repository.create(**sample_run_data)
        
        # Verify logging calls were made
        assert mock_logger.debug.called
        
        # Update entity (should log update)
        run_repository.update(run.id, name="Updated Name")
        
        # Delete entity (should log deletion)
        run_repository.delete(run.id)
        
        # Should have multiple debug calls
        assert mock_logger.debug.call_count >= 3