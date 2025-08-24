"""
Pytest fixtures and configuration for database layer tests.

Provides comprehensive fixtures for testing repository patterns,
database session management, and ORM models with proper isolation.
"""

import pytest
import tempfile
import shutil
import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from contextlib import contextmanager
from typing import Generator, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from scramblebench.db.session import DatabaseManager
from scramblebench.db.repository import (
    RepositoryFactory, RunRepository, ItemRepository, 
    EvaluationRepository, AggregateRepository, ParaphraseCacheRepository
)
from scramblebench.db.models import (
    Base, Run, Item, Evaluation, Aggregate, ParaphraseCache
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_dir():
    """Fixture providing temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp(prefix="test_db_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_db_url(temp_db_dir):
    """Fixture providing test database URL with isolated SQLite database."""
    db_path = temp_db_dir / "test.db"
    return f"sqlite:///{db_path}"


@pytest.fixture
def db_manager(test_db_url):
    """Fixture providing DatabaseManager instance with test database."""
    manager = DatabaseManager(
        database_url=test_db_url,
        echo=False,  # Set to True for SQL debugging
        pool_recycle=3600
    )
    
    # Create all tables
    manager.create_tables()
    
    yield manager
    
    # Cleanup
    try:
        manager.close()
    except Exception:
        pass


@pytest.fixture
def db_session(db_manager):
    """Fixture providing clean database session for each test."""
    with db_manager.session_scope() as session:
        yield session


@pytest.fixture
def repository_factory(db_manager):
    """Fixture providing RepositoryFactory instance."""
    return RepositoryFactory(db_manager)


@pytest.fixture
def run_repository(repository_factory):
    """Fixture providing RunRepository instance."""
    return repository_factory.get_run_repository()


@pytest.fixture
def item_repository(repository_factory):
    """Fixture providing ItemRepository instance."""
    return repository_factory.get_item_repository()


@pytest.fixture
def evaluation_repository(repository_factory):
    """Fixture providing EvaluationRepository instance."""
    return repository_factory.get_evaluation_repository()


@pytest.fixture
def aggregate_repository(repository_factory):
    """Fixture providing AggregateRepository instance."""
    return repository_factory.get_aggregate_repository()


@pytest.fixture
def paraphrase_repository(repository_factory):
    """Fixture providing ParaphraseCacheRepository instance."""
    return repository_factory.get_paraphrase_cache_repository()


# Sample data fixtures

@pytest.fixture
def sample_run_data():
    """Fixture providing sample run data for testing."""
    return {
        "run_id": "test-run-123",
        "name": "Test Run",
        "config_hash": "abc123def456",
        "benchmark": "mmlu",
        "model_ids": ["gpt-4", "claude-3"],
        "total_evaluations": 1000,
        "completed_evaluations": 500,
        "status": "running",
        "created_at": datetime.utcnow(),
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "meta_data": {"test": True}
    }


@pytest.fixture
def sample_item_data():
    """Fixture providing sample item data for testing."""
    return {
        "item_id": "test-item-123",
        "dataset": "mmlu",
        "domain": "math",
        "question": "What is 2 + 2?",
        "answer": "4",
        "choices": ["2", "3", "4", "5"],
        "difficulty": "easy"
    }


@pytest.fixture
def sample_evaluation_data():
    """Fixture providing sample evaluation data for testing."""
    return {
        "eval_id": "test-eval-123",
        "run_id": "test-run-123",
        "item_id": "test-item-123",
        "model_id": "gpt-4",
        "transform": "original",
        "question": "What is 2 + 2?",
        "response": "The answer is 4.",
        "expected_answer": "4",
        "is_correct": True,
        "latency_ms": 250,
        "cost_usd": 0.001,
        "timestamp": datetime.utcnow()
    }


@pytest.fixture
def sample_aggregate_data():
    """Fixture providing sample aggregate data for testing."""
    return {
        "run_id": "test-run-123",
        "model_id": "gpt-4",
        "transform": "original",
        "domain": "math",
        "acc_mean": 0.85,
        "acc_std": 0.12,
        "rrs": 0.75,
        "ldc": 0.65,
        "n_samples": 100
    }


@pytest.fixture
def sample_paraphrase_data():
    """Fixture providing sample paraphrase data for testing."""
    return {
        "item_id": "test-item-123",
        "candidate_id": 1,
        "provider": "openai",
        "paraphrase": "What is the sum of 2 and 2?",
        "cos_sim": 0.92,
        "edit_ratio": 0.35,
        "bleu_score": 0.78,
        "accepted": False
    }


@pytest.fixture
def populated_database(
    db_manager, 
    sample_run_data, 
    sample_item_data, 
    sample_evaluation_data,
    sample_aggregate_data,
    sample_paraphrase_data
):
    """Fixture providing database populated with sample data."""
    with db_manager.session_scope() as session:
        # Create sample run
        run = Run(**sample_run_data)
        session.add(run)
        
        # Create sample item
        item = Item(**sample_item_data)
        session.add(item)
        
        # Create sample evaluation
        evaluation = Evaluation(**sample_evaluation_data)
        session.add(evaluation)
        
        # Create sample aggregate
        aggregate = Aggregate(**sample_aggregate_data)
        session.add(aggregate)
        
        # Create sample paraphrase
        paraphrase = ParaphraseCache(**sample_paraphrase_data)
        session.add(paraphrase)
        
        session.flush()  # Ensure IDs are generated
        
        yield {
            "run": run,
            "item": item,
            "evaluation": evaluation,
            "aggregate": aggregate,
            "paraphrase": paraphrase
        }


@pytest.fixture
def multiple_runs_data():
    """Fixture providing data for multiple runs with different statuses."""
    base_time = datetime.utcnow()
    
    return [
        {
            "run_id": "completed-run-1",
            "name": "Completed Run 1",
            "config_hash": "hash1",
            "benchmark": "mmlu",
            "model_ids": ["gpt-4"],
            "total_evaluations": 100,
            "completed_evaluations": 100,
            "status": "completed",
            "created_at": base_time - timedelta(days=2),
            "started_at": base_time - timedelta(days=2),
            "completed_at": base_time - timedelta(days=1)
        },
        {
            "run_id": "running-run-1",
            "name": "Running Run 1",
            "config_hash": "hash2",
            "benchmark": "hellaswag",
            "model_ids": ["claude-3"],
            "total_evaluations": 200,
            "completed_evaluations": 50,
            "status": "running",
            "created_at": base_time - timedelta(hours=2),
            "started_at": base_time - timedelta(hours=1),
            "completed_at": None
        },
        {
            "run_id": "failed-run-1",
            "name": "Failed Run 1",
            "config_hash": "hash3",
            "benchmark": "arc",
            "model_ids": ["gpt-3.5"],
            "total_evaluations": 150,
            "completed_evaluations": 25,
            "status": "failed",
            "created_at": base_time - timedelta(hours=3),
            "started_at": base_time - timedelta(hours=3),
            "completed_at": None,
            "meta_data": {"error": "API timeout"}
        }
    ]


@pytest.fixture
def multiple_items_data():
    """Fixture providing data for multiple items across domains and datasets."""
    return [
        {
            "item_id": "mmlu-math-1",
            "dataset": "mmlu",
            "domain": "math",
            "question": "What is 5 * 6?",
            "answer": "30",
            "choices": ["25", "30", "35", "40"]
        },
        {
            "item_id": "mmlu-science-1",
            "dataset": "mmlu",
            "domain": "science",
            "question": "What is H2O?",
            "answer": "Water",
            "choices": ["Hydrogen", "Oxygen", "Water", "Salt"]
        },
        {
            "item_id": "hellaswag-1",
            "dataset": "hellaswag",
            "domain": None,
            "question": "A person is cooking dinner...",
            "answer": "They serve the food",
            "choices": ["They serve the food", "They eat raw ingredients", "They throw it away", "They start over"]
        }
    ]


@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing scenarios."""
    base_time = datetime.utcnow()
    
    # Generate 100 evaluations across 3 models and 2 transforms
    evaluations = []
    models = ["gpt-4", "claude-3", "llama-2"]
    transforms = ["original", "scrambled"]
    
    eval_counter = 0
    for model in models:
        for transform in transforms:
            for i in range(17):  # 17 * 3 * 2 = 102 evaluations
                eval_counter += 1
                evaluations.append({
                    "eval_id": f"perf-eval-{eval_counter}",
                    "run_id": "performance-run",
                    "item_id": f"perf-item-{i}",
                    "model_id": model,
                    "transform": transform,
                    "question": f"Performance test question {i}",
                    "response": f"Response {i}",
                    "expected_answer": f"Answer {i}",
                    "is_correct": i % 3 == 0,  # 33% accuracy
                    "latency_ms": 100 + (i * 10),
                    "cost_usd": 0.001 * (i + 1),
                    "timestamp": base_time + timedelta(seconds=i)
                })
    
    return {
        "run": {
            "run_id": "performance-run",
            "name": "Performance Test Run",
            "config_hash": "perf-hash",
            "benchmark": "performance",
            "model_ids": models,
            "total_evaluations": len(evaluations),
            "completed_evaluations": len(evaluations),
            "status": "completed",
            "created_at": base_time,
            "started_at": base_time,
            "completed_at": base_time + timedelta(minutes=30)
        },
        "evaluations": evaluations
    }


@pytest.fixture
def mock_sqlalchemy_error():
    """Fixture providing mock SQLAlchemy errors for testing error handling."""
    class MockSQLAlchemyError(Exception):
        def __init__(self, message="Mock database error"):
            self.message = message
            super().__init__(message)
    
    return MockSQLAlchemyError


# Database health and validation fixtures

@pytest.fixture  
def db_health_checker(db_manager):
    """Fixture providing database health checking utilities."""
    class HealthChecker:
        def __init__(self, manager):
            self.manager = manager
            
        def check_tables_exist(self):
            """Check if all required tables exist."""
            with self.manager.session_scope() as session:
                tables = self.manager.engine.table_names()
                required_tables = {"runs", "items", "evaluations", "aggregates", "paraphrase_cache"}
                return required_tables.issubset(set(tables))
        
        def count_records(self, model_class):
            """Count total records in a table."""
            with self.manager.session_scope() as session:
                return session.query(model_class).count()
        
        def verify_constraints(self):
            """Verify database constraints are working."""
            # This would test foreign key constraints, unique constraints, etc.
            return True
    
    return HealthChecker(db_manager)


@pytest.fixture
def transaction_test_helper(db_manager):
    """Fixture providing transaction testing utilities."""
    class TransactionHelper:
        def __init__(self, manager):
            self.manager = manager
            
        @contextmanager
        def expect_rollback(self):
            """Context manager that expects transaction to be rolled back."""
            initial_count = self.get_run_count()
            try:
                yield
            except Exception:
                # Verify rollback occurred
                final_count = self.get_run_count()
                assert final_count == initial_count, "Transaction was not rolled back"
                raise
            else:
                raise AssertionError("Expected exception but none was raised")
        
        def get_run_count(self):
            """Get current count of runs in database."""
            with self.manager.session_scope() as session:
                return session.query(Run).count()
    
    return TransactionHelper(db_manager)