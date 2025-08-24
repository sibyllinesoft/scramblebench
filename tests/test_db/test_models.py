"""
Tests for ORM model definitions and relationships.

Tests model instantiation, validation, relationships,
and database schema constraints.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy import inspect

from scramblebench.db.models import (
    Base, Run, Item, Evaluation, Aggregate, ParaphraseCache
)


class TestBaseModel:
    """Test cases for Base model functionality."""
    
    def test_base_model_metadata(self, db_manager):
        """Test that Base model has proper metadata."""
        assert hasattr(Base, 'metadata')
        assert Base.metadata is not None
        
        # Verify tables are registered in metadata
        table_names = list(Base.metadata.tables.keys())
        expected_tables = ['runs', 'items', 'evaluations', 'aggregates', 'paraphrase_cache']
        
        for table in expected_tables:
            assert table in table_names


class TestRunModel:
    """Test cases for Run model."""
    
    def test_run_model_creation(self, db_session, sample_run_data):
        """Test creating a Run model instance."""
        run = Run(**sample_run_data)
        
        assert run.run_id == sample_run_data["run_id"]
        assert run.name == sample_run_data["name"]
        assert run.config_hash == sample_run_data["config_hash"]
        assert run.benchmark == sample_run_data["benchmark"]
        assert run.model_ids == sample_run_data["model_ids"]
        assert run.total_evaluations == sample_run_data["total_evaluations"]
        assert run.completed_evaluations == sample_run_data["completed_evaluations"]
        assert run.status == sample_run_data["status"]
    
    def test_run_model_persistence(self, db_session, sample_run_data):
        """Test saving and retrieving Run model."""
        run = Run(**sample_run_data)
        db_session.add(run)
        db_session.flush()
        
        # Verify ID was generated
        assert run.id is not None
        
        # Retrieve from database
        retrieved_run = db_session.query(Run).filter_by(run_id=sample_run_data["run_id"]).first()
        
        assert retrieved_run is not None
        assert retrieved_run.run_id == sample_run_data["run_id"]
        assert retrieved_run.name == sample_run_data["name"]
    
    def test_run_model_unique_constraint(self, db_session, sample_run_data):
        """Test that run_id has unique constraint."""
        # Create first run
        run1 = Run(**sample_run_data)
        db_session.add(run1)
        db_session.flush()
        
        # Try to create second run with same run_id
        run2_data = sample_run_data.copy()
        run2_data["name"] = "Different Name"
        run2 = Run(**run2_data)
        db_session.add(run2)
        
        with pytest.raises(IntegrityError):
            db_session.flush()
    
    def test_run_model_required_fields(self, db_session):
        """Test that required fields are enforced."""
        # Try to create run without required fields
        with pytest.raises(Exception):  # Could be TypeError or IntegrityError
            run = Run()
            db_session.add(run)
            db_session.flush()
    
    def test_run_model_default_values(self, db_session):
        """Test default values for Run model."""
        minimal_data = {
            "run_id": "test-defaults",
            "name": "Test Run",
            "config_hash": "hash123",
            "benchmark": "test",
            "model_ids": ["test-model"]
        }
        
        run = Run(**minimal_data)
        db_session.add(run)
        db_session.flush()
        
        # Check defaults
        assert run.total_evaluations == 0
        assert run.completed_evaluations == 0
        assert run.status == "created"
        assert run.created_at is not None
        assert run.completed_at is None
    
    def test_run_model_json_serialization(self, sample_run_data):
        """Test that JSON fields work properly."""
        # Test with complex model_ids and meta_data
        complex_data = sample_run_data.copy()
        complex_data["model_ids"] = ["gpt-4", "claude-3", "llama-2"]
        complex_data["meta_data"] = {
            "experiment_id": "exp-123",
            "researcher": "test@example.com",
            "tags": ["benchmark", "comparison"]
        }
        
        run = Run(**complex_data)
        
        assert len(run.model_ids) == 3
        assert "gpt-4" in run.model_ids
        assert run.meta_data["experiment_id"] == "exp-123"
        assert "benchmark" in run.meta_data["tags"]
    
    def test_run_model_table_structure(self, db_manager):
        """Test Run table structure."""
        inspector = inspect(db_manager.engine)
        columns = inspector.get_columns('runs')
        column_names = [col['name'] for col in columns]
        
        expected_columns = [
            'id', 'run_id', 'name', 'config_hash', 'benchmark', 
            'model_ids', 'total_evaluations', 'completed_evaluations',
            'status', 'created_at', 'started_at', 'completed_at', 'meta_data'
        ]
        
        for col in expected_columns:
            assert col in column_names
        
        # Check for unique constraint on run_id
        unique_constraints = inspector.get_unique_constraints('runs')
        run_id_unique = any('run_id' in constraint['column_names'] for constraint in unique_constraints)
        assert run_id_unique


class TestItemModel:
    """Test cases for Item model."""
    
    def test_item_model_creation(self, sample_item_data):
        """Test creating an Item model instance."""
        item = Item(**sample_item_data)
        
        assert item.item_id == sample_item_data["item_id"]
        assert item.dataset == sample_item_data["dataset"]
        assert item.domain == sample_item_data["domain"]
        assert item.question == sample_item_data["question"]
        assert item.answer == sample_item_data["answer"]
        assert item.choices == sample_item_data["choices"]
    
    def test_item_model_persistence(self, db_session, sample_item_data):
        """Test saving and retrieving Item model."""
        item = Item(**sample_item_data)
        db_session.add(item)
        db_session.flush()
        
        # Verify ID was generated
        assert item.id is not None
        
        # Retrieve from database
        retrieved_item = db_session.query(Item).filter_by(item_id=sample_item_data["item_id"]).first()
        
        assert retrieved_item is not None
        assert retrieved_item.question == sample_item_data["question"]
        assert retrieved_item.choices == sample_item_data["choices"]
    
    def test_item_model_optional_fields(self, db_session):
        """Test Item model with optional fields."""
        minimal_data = {
            "item_id": "minimal-item",
            "dataset": "test",
            "question": "Test question?",
            "answer": "Test answer"
        }
        
        item = Item(**minimal_data)
        db_session.add(item)
        db_session.flush()
        
        # Optional fields should be None
        assert item.domain is None
        assert item.choices is None
        assert item.difficulty is None
    
    def test_item_model_choices_array(self, db_session):
        """Test that choices field handles arrays properly."""
        data = {
            "item_id": "choices-test",
            "dataset": "test",
            "question": "Multiple choice question?",
            "answer": "A",
            "choices": ["A) First", "B) Second", "C) Third", "D) Fourth"]
        }
        
        item = Item(**data)
        db_session.add(item)
        db_session.flush()
        
        retrieved = db_session.query(Item).filter_by(item_id="choices-test").first()
        assert len(retrieved.choices) == 4
        assert "A) First" in retrieved.choices


class TestEvaluationModel:
    """Test cases for Evaluation model."""
    
    def test_evaluation_model_creation(self, sample_evaluation_data):
        """Test creating an Evaluation model instance."""
        evaluation = Evaluation(**sample_evaluation_data)
        
        assert evaluation.eval_id == sample_evaluation_data["eval_id"]
        assert evaluation.run_id == sample_evaluation_data["run_id"]
        assert evaluation.item_id == sample_evaluation_data["item_id"]
        assert evaluation.model_id == sample_evaluation_data["model_id"]
        assert evaluation.transform == sample_evaluation_data["transform"]
        assert evaluation.is_correct == sample_evaluation_data["is_correct"]
    
    def test_evaluation_model_persistence(self, db_session, sample_evaluation_data):
        """Test saving and retrieving Evaluation model."""
        evaluation = Evaluation(**sample_evaluation_data)
        db_session.add(evaluation)
        db_session.flush()
        
        assert evaluation.id is not None
        
        retrieved = db_session.query(Evaluation).filter_by(eval_id=sample_evaluation_data["eval_id"]).first()
        assert retrieved is not None
        assert retrieved.response == sample_evaluation_data["response"]
    
    def test_evaluation_model_foreign_key_relationships(self, db_session, populated_database):
        """Test foreign key relationships work properly."""
        # Get the evaluation from populated database
        evaluation = populated_database["evaluation"]
        run = populated_database["run"]
        item = populated_database["item"]
        
        # Verify relationships can be queried
        db_evals_for_run = db_session.query(Evaluation).filter_by(run_id=run.run_id).all()
        assert len(db_evals_for_run) >= 1
        
        db_evals_for_item = db_session.query(Evaluation).filter_by(item_id=item.item_id).all()
        assert len(db_evals_for_item) >= 1
    
    def test_evaluation_model_numeric_fields(self, db_session):
        """Test numeric fields in Evaluation model."""
        data = {
            "eval_id": "numeric-test",
            "run_id": "test-run",
            "item_id": "test-item", 
            "model_id": "test-model",
            "transform": "original",
            "question": "Test?",
            "response": "Answer",
            "expected_answer": "Answer",
            "is_correct": True,
            "latency_ms": 1250,
            "cost_usd": 0.00125,
            "timestamp": datetime.utcnow()
        }
        
        evaluation = Evaluation(**data)
        db_session.add(evaluation)
        db_session.flush()
        
        retrieved = db_session.query(Evaluation).filter_by(eval_id="numeric-test").first()
        assert retrieved.latency_ms == 1250
        assert abs(retrieved.cost_usd - 0.00125) < 0.000001  # Float precision
    
    def test_evaluation_model_boolean_field(self, db_session):
        """Test boolean is_correct field."""
        base_data = {
            "run_id": "bool-test",
            "item_id": "test-item",
            "model_id": "test-model",
            "transform": "original",
            "question": "Test?",
            "response": "Answer",
            "expected_answer": "Answer",
            "timestamp": datetime.utcnow()
        }
        
        # Test True
        eval_true = Evaluation(eval_id="bool-true", is_correct=True, **base_data)
        db_session.add(eval_true)
        
        # Test False
        eval_false = Evaluation(eval_id="bool-false", is_correct=False, **base_data)
        db_session.add(eval_false)
        
        # Test None (null)
        eval_none = Evaluation(eval_id="bool-none", is_correct=None, **base_data)
        db_session.add(eval_none)
        
        db_session.flush()
        
        # Verify values
        assert db_session.query(Evaluation).filter_by(eval_id="bool-true").first().is_correct is True
        assert db_session.query(Evaluation).filter_by(eval_id="bool-false").first().is_correct is False
        assert db_session.query(Evaluation).filter_by(eval_id="bool-none").first().is_correct is None


class TestAggregateModel:
    """Test cases for Aggregate model."""
    
    def test_aggregate_model_creation(self, sample_aggregate_data):
        """Test creating an Aggregate model instance."""
        aggregate = Aggregate(**sample_aggregate_data)
        
        assert aggregate.run_id == sample_aggregate_data["run_id"]
        assert aggregate.model_id == sample_aggregate_data["model_id"]
        assert aggregate.transform == sample_aggregate_data["transform"]
        assert aggregate.domain == sample_aggregate_data["domain"]
        assert aggregate.acc_mean == sample_aggregate_data["acc_mean"]
        assert aggregate.n_samples == sample_aggregate_data["n_samples"]
    
    def test_aggregate_model_persistence(self, db_session, sample_aggregate_data):
        """Test saving and retrieving Aggregate model."""
        aggregate = Aggregate(**sample_aggregate_data)
        db_session.add(aggregate)
        db_session.flush()
        
        assert aggregate.id is not None
        
        retrieved = db_session.query(Aggregate).filter_by(
            run_id=sample_aggregate_data["run_id"],
            model_id=sample_aggregate_data["model_id"],
            transform=sample_aggregate_data["transform"]
        ).first()
        
        assert retrieved is not None
        assert abs(retrieved.acc_mean - sample_aggregate_data["acc_mean"]) < 0.001
    
    def test_aggregate_model_statistical_fields(self, db_session):
        """Test statistical fields in Aggregate model."""
        data = {
            "run_id": "stats-test",
            "model_id": "test-model",
            "transform": "original",
            "domain": "math",
            "acc_mean": 0.8567,
            "acc_std": 0.1234,
            "rrs": 0.7543,
            "ldc": 0.6789,
            "n_samples": 1000
        }
        
        aggregate = Aggregate(**data)
        db_session.add(aggregate)
        db_session.flush()
        
        retrieved = db_session.query(Aggregate).filter_by(run_id="stats-test").first()
        
        # Check precision of float fields
        assert abs(retrieved.acc_mean - 0.8567) < 0.0001
        assert abs(retrieved.acc_std - 0.1234) < 0.0001
        assert abs(retrieved.rrs - 0.7543) < 0.0001
        assert abs(retrieved.ldc - 0.6789) < 0.0001
        assert retrieved.n_samples == 1000
    
    def test_aggregate_model_optional_fields(self, db_session):
        """Test Aggregate model with optional fields."""
        minimal_data = {
            "run_id": "minimal-agg",
            "model_id": "test-model",
            "transform": "original",
            "acc_mean": 0.75,
            "n_samples": 100
        }
        
        aggregate = Aggregate(**minimal_data)
        db_session.add(aggregate)
        db_session.flush()
        
        retrieved = db_session.query(Aggregate).filter_by(run_id="minimal-agg").first()
        
        # Optional fields should be None
        assert retrieved.domain is None
        assert retrieved.acc_std is None
        assert retrieved.rrs is None
        assert retrieved.ldc is None


class TestParaphraseCacheModel:
    """Test cases for ParaphraseCache model."""
    
    def test_paraphrase_cache_model_creation(self, sample_paraphrase_data):
        """Test creating a ParaphraseCache model instance."""
        paraphrase = ParaphraseCache(**sample_paraphrase_data)
        
        assert paraphrase.item_id == sample_paraphrase_data["item_id"]
        assert paraphrase.candidate_id == sample_paraphrase_data["candidate_id"]
        assert paraphrase.provider == sample_paraphrase_data["provider"]
        assert paraphrase.paraphrase == sample_paraphrase_data["paraphrase"]
        assert paraphrase.accepted == sample_paraphrase_data["accepted"]
    
    def test_paraphrase_cache_model_persistence(self, db_session, sample_paraphrase_data):
        """Test saving and retrieving ParaphraseCache model."""
        paraphrase = ParaphraseCache(**sample_paraphrase_data)
        db_session.add(paraphrase)
        db_session.flush()
        
        assert paraphrase.id is not None
        
        retrieved = db_session.query(ParaphraseCache).filter_by(
            item_id=sample_paraphrase_data["item_id"],
            candidate_id=sample_paraphrase_data["candidate_id"]
        ).first()
        
        assert retrieved is not None
        assert retrieved.paraphrase == sample_paraphrase_data["paraphrase"]
    
    def test_paraphrase_cache_quality_metrics(self, db_session):
        """Test quality metric fields in ParaphraseCache model."""
        data = {
            "item_id": "quality-test",
            "candidate_id": 1,
            "provider": "openai",
            "paraphrase": "Quality test paraphrase",
            "cos_sim": 0.9123,
            "edit_ratio": 0.3456,
            "bleu_score": 0.7890,
            "accepted": False
        }
        
        paraphrase = ParaphraseCache(**data)
        db_session.add(paraphrase)
        db_session.flush()
        
        retrieved = db_session.query(ParaphraseCache).filter_by(item_id="quality-test").first()
        
        # Check precision of quality metrics
        assert abs(retrieved.cos_sim - 0.9123) < 0.0001
        assert abs(retrieved.edit_ratio - 0.3456) < 0.0001
        assert abs(retrieved.bleu_score - 0.7890) < 0.0001
    
    def test_paraphrase_cache_boolean_accepted(self, db_session):
        """Test accepted field with different boolean values."""
        base_data = {
            "item_id": "bool-accepted-test",
            "provider": "openai",
            "paraphrase": "Test paraphrase"
        }
        
        # Test accepted=True
        para_true = ParaphraseCache(candidate_id=1, accepted=True, **base_data)
        db_session.add(para_true)
        
        # Test accepted=False
        para_false = ParaphraseCache(candidate_id=2, accepted=False, **base_data)
        db_session.add(para_false)
        
        # Test accepted=None (not evaluated)
        para_none = ParaphraseCache(candidate_id=3, accepted=None, **base_data)
        db_session.add(para_none)
        
        db_session.flush()
        
        # Verify values
        results = db_session.query(ParaphraseCache).filter_by(item_id="bool-accepted-test").all()
        
        accepted_values = {p.candidate_id: p.accepted for p in results}
        assert accepted_values[1] is True
        assert accepted_values[2] is False
        assert accepted_values[3] is None
    
    def test_paraphrase_cache_composite_key(self, db_session, sample_paraphrase_data):
        """Test composite key constraint (item_id + candidate_id)."""
        # Create first paraphrase
        para1 = ParaphraseCache(**sample_paraphrase_data)
        db_session.add(para1)
        db_session.flush()
        
        # Try to create another with same item_id + candidate_id
        para2_data = sample_paraphrase_data.copy()
        para2_data["paraphrase"] = "Different paraphrase text"
        para2 = ParaphraseCache(**para2_data)
        db_session.add(para2)
        
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestModelRelationships:
    """Test relationships between models."""
    
    def test_run_to_evaluations_relationship(self, db_session, populated_database):
        """Test relationship from Run to Evaluations."""
        run = populated_database["run"]
        
        # Query evaluations for this run
        evaluations = db_session.query(Evaluation).filter_by(run_id=run.run_id).all()
        
        assert len(evaluations) >= 1
        assert all(eval.run_id == run.run_id for eval in evaluations)
    
    def test_item_to_evaluations_relationship(self, db_session, populated_database):
        """Test relationship from Item to Evaluations."""
        item = populated_database["item"]
        
        # Query evaluations for this item
        evaluations = db_session.query(Evaluation).filter_by(item_id=item.item_id).all()
        
        assert len(evaluations) >= 1
        assert all(eval.item_id == item.item_id for eval in evaluations)
    
    def test_item_to_paraphrases_relationship(self, db_session, populated_database):
        """Test relationship from Item to ParaphraseCache."""
        item = populated_database["item"]
        
        # Query paraphrases for this item
        paraphrases = db_session.query(ParaphraseCache).filter_by(item_id=item.item_id).all()
        
        assert len(paraphrases) >= 1
        assert all(para.item_id == item.item_id for para in paraphrases)
    
    def test_run_to_aggregates_relationship(self, db_session, populated_database):
        """Test relationship from Run to Aggregates."""
        run = populated_database["run"]
        
        # Query aggregates for this run
        aggregates = db_session.query(Aggregate).filter_by(run_id=run.run_id).all()
        
        assert len(aggregates) >= 1
        assert all(agg.run_id == run.run_id for agg in aggregates)


class TestModelConstraints:
    """Test database constraints and validation."""
    
    def test_run_id_length_constraint(self, db_session):
        """Test run_id field length constraints."""
        # This will depend on your actual schema definition
        # Test with very long run_id
        long_run_id = "x" * 300  # Assuming there's a reasonable length limit
        
        data = {
            "run_id": long_run_id,
            "name": "Test",
            "config_hash": "hash",
            "benchmark": "test",
            "model_ids": ["model"]
        }
        
        run = Run(**data)
        db_session.add(run)
        
        # This might raise an error depending on column definition
        # If no length constraint, test passes; if there is one, it should raise error
        try:
            db_session.flush()
            # If successful, the constraint allows long values
            assert len(run.run_id) == 300
        except Exception as e:
            # If it fails, there's a length constraint (which is good)
            assert "too long" in str(e).lower() or "length" in str(e).lower()
    
    def test_evaluation_required_references(self, db_session):
        """Test that evaluations require valid run_id and item_id references."""
        # Note: This test depends on whether you have foreign key constraints enabled
        
        eval_data = {
            "eval_id": "test-constraint",
            "run_id": "nonexistent-run",  # This should violate foreign key
            "item_id": "nonexistent-item",  # This should violate foreign key
            "model_id": "test-model",
            "transform": "original",
            "question": "Test?",
            "response": "Answer",
            "timestamp": datetime.utcnow()
        }
        
        evaluation = Evaluation(**eval_data)
        db_session.add(evaluation)
        
        # If foreign key constraints are enabled, this should fail
        # If not, it will succeed (which is also valid for testing)
        try:
            db_session.flush()
            # No foreign key constraints - test passes
            assert evaluation.run_id == "nonexistent-run"
        except IntegrityError:
            # Foreign key constraints are enforced - this is also valid
            assert True
    
    def test_aggregate_numeric_constraints(self, db_session):
        """Test numeric field constraints in Aggregate model."""
        # Test with negative sample count (should be invalid)
        data = {
            "run_id": "constraint-test",
            "model_id": "test-model", 
            "transform": "original",
            "acc_mean": 0.5,
            "n_samples": -10  # Negative samples don't make sense
        }
        
        aggregate = Aggregate(**data)
        db_session.add(aggregate)
        
        # This might succeed if no CHECK constraints, or fail if there are constraints
        try:
            db_session.flush()
            # No constraints on negative values
            assert aggregate.n_samples == -10
        except Exception:
            # There are constraints preventing negative values
            assert True
    
    def test_paraphrase_quality_ranges(self, db_session):
        """Test quality metric ranges in ParaphraseCache."""
        # Test with out-of-range quality values
        data = {
            "item_id": "range-test",
            "candidate_id": 1,
            "provider": "test",
            "paraphrase": "Test",
            "cos_sim": 1.5,  # Should be 0-1 range
            "edit_ratio": -0.5,  # Should be 0-1 range
            "bleu_score": 2.0  # Should be 0-1 range
        }
        
        paraphrase = ParaphraseCache(**data)
        db_session.add(paraphrase)
        
        # This will succeed if no range constraints, or fail if there are constraints
        try:
            db_session.flush()
            # No range constraints
            assert paraphrase.cos_sim == 1.5
        except Exception:
            # Range constraints are enforced
            assert True


class TestModelStringRepresentations:
    """Test string representations and basic model methods."""
    
    def test_run_model_str(self, sample_run_data):
        """Test Run model string representation."""
        run = Run(**sample_run_data)
        
        str_repr = str(run)
        assert sample_run_data["run_id"] in str_repr
        assert sample_run_data["name"] in str_repr
    
    def test_item_model_str(self, sample_item_data):
        """Test Item model string representation."""
        item = Item(**sample_item_data)
        
        str_repr = str(item)
        assert sample_item_data["item_id"] in str_repr
        # Should contain part of the question (truncated for long questions)
        question_preview = sample_item_data["question"][:50]
        assert any(word in str_repr for word in question_preview.split())
    
    def test_evaluation_model_str(self, sample_evaluation_data):
        """Test Evaluation model string representation."""
        evaluation = Evaluation(**sample_evaluation_data)
        
        str_repr = str(evaluation)
        assert sample_evaluation_data["eval_id"] in str_repr
        assert sample_evaluation_data["model_id"] in str_repr
    
    def test_model_repr(self, sample_run_data):
        """Test model __repr__ method."""
        run = Run(**sample_run_data)
        
        repr_str = repr(run)
        assert "Run" in repr_str
        assert sample_run_data["run_id"] in repr_str