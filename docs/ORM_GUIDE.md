# ScrambleBench ORM Layer Guide

The ScrambleBench ORM layer provides a modern, SQLAlchemy-based interface for database operations. This guide explains how to use the ORM layer according to the TODO.md specifications.

## Overview

The ORM layer consists of:
- **Models**: SQLAlchemy ORM models in `src/scramblebench/core/models.py`
- **DatabaseManager**: Session and connection management in `src/scramblebench/db/session.py`  
- **Repositories**: Data access layer in `src/scramblebench/db/repository.py`
- **Migrations**: Alembic migrations in `alembic/`

## Key Principles

1. **Single Source of Truth**: The ORM models are the canonical representation of the database schema
2. **Repository Pattern**: All database operations go through repository classes
3. **Session Management**: All database interactions use the DatabaseManager for proper session lifecycle
4. **Type Safety**: Full SQLAlchemy type annotations and validation

## Quick Start

### 1. Initialize Database

```python
from scramblebench.db import DatabaseManager, initialize_database

# Initialize with default SQLite database
db_manager = initialize_database()

# Or specify custom database URL
db_manager = initialize_database(
    database_url="sqlite:///custom/path/database.db",
    echo=True  # Show SQL statements for debugging
)
```

### 2. Use Repositories

```python
from scramblebench.db import RunRepository, ItemRepository, EvaluationRepository

# Create repositories
run_repo = RunRepository(db_manager)
item_repo = ItemRepository(db_manager)
eval_repo = EvaluationRepository(db_manager)

# Create a run
run = run_repo.create(
    run_id="my_experiment_001",
    config_yaml="model: gpt-4\ndataset: gsm8k",
    config_hash="abc123def456",
    seed=42,
    total_evaluations=1000
)

# Query runs
active_runs = run_repo.get_active_runs()
completed_runs = run_repo.get_completed_runs(limit=10)
```

### 3. Session Management

```python
# Recommended: Use session context manager
with db_manager.session_scope() as session:
    run = session.query(Run).filter_by(run_id="my_experiment").first()
    if run:
        run.status = "completed"
    # Session automatically committed/rolled back

# For complex transactions
with db_manager.transaction_scope() as session:
    # Multiple operations in single transaction
    session.add(run)
    session.add(evaluation)
    # Transaction committed automatically
```

## Models

### Core Models

The following models represent the core ScrambleBench entities:

#### Run
Represents a complete evaluation run:
```python
run = Run(
    run_id="experiment_001",
    config_yaml="...",
    config_hash="abc123",
    seed=42,
    total_evaluations=1000,
    git_sha="1234567890abcdef"
)

# Properties
print(f"Progress: {run.progress_percentage:.1f}%")
print(f"Completed: {run.is_completed}")
print(f"Duration: {run.duration} seconds")
```

#### Item
Represents benchmark items (questions/problems):
```python
item = Item(
    item_id="gsm8k_001",
    dataset="gsm8k",
    domain="arithmetic",
    question="If John has 5 apples and gives away 2, how many does he have left?",
    answer="3"
)
```

#### Evaluation  
Represents individual model evaluations:
```python
evaluation = Evaluation(
    eval_id="eval_001",
    run_id="experiment_001", 
    item_id="gsm8k_001",
    model_id="gpt-4",
    provider="openai",
    transform="original",
    prompt="Question: ...",
    completion="Answer: 3",
    is_correct=True,
    acc=1.0,
    prompt_tokens=50,
    completion_tokens=10,
    latency_ms=1200,
    cost_usd=0.002
)

# Properties
print(f"Total tokens: {evaluation.total_tokens}")
print(f"Speed: {evaluation.tokens_per_second:.1f} tok/s")
```

#### Aggregate
Precomputed aggregate metrics:
```python
aggregate = Aggregate(
    run_id="experiment_001",
    model_id="gpt-4", 
    dataset="gsm8k",
    transform="scramble",
    acc_mean=0.87,
    acc_ci_low=0.84,
    acc_ci_high=0.90,
    rrs=0.92,  # Reasoning Robustness Score
    ldc=0.23,  # Language Dependency Coefficient
    n_items=1000
)

# Properties  
print(f"Performance: {aggregate.performance_category}")
print(f"Language dependency: {aggregate.language_dependency_category}")
```

#### ParaphraseCache
Cached paraphrased versions of items:
```python
paraphrase = ParaphraseCache(
    item_id="gsm8k_001",
    provider="anthropic",
    candidate_id=0,
    text="John starts with 5 apples. He gives 2 away. How many remain?",
    cos_sim=0.92,
    edit_ratio=0.35,
    bleu_score=0.78,
    accepted=True
)

# Properties
print(f"Quality score: {paraphrase.quality_score:.2f}")
print(f"Meets threshold: {paraphrase.meets_quality_threshold}")
```

## Repositories

### RunRepository

```python
run_repo = RunRepository(db_manager)

# Basic operations
run = run_repo.create(run_id="exp1", config_yaml="...", ...)
run = run_repo.get_by_run_id("exp1")
run_repo.update_progress("exp1", completed_evaluations=500)
run_repo.mark_completed("exp1")

# Query operations
active_runs = run_repo.get_active_runs()
completed_runs = run_repo.get_completed_runs(limit=5)
similar_runs = run_repo.get_runs_by_config_hash("abc123")
```

### ItemRepository

```python
item_repo = ItemRepository(db_manager)

# Basic operations
item = item_repo.create(item_id="test1", dataset="gsm8k", ...)
item = item_repo.get_by_item_id("test1")

# Query operations
gsm8k_items = item_repo.get_by_dataset("gsm8k")
math_items = item_repo.get_by_dataset("gsm8k", domain="arithmetic")
datasets = item_repo.get_datasets()
domains = item_repo.get_domains("gsm8k")
results = item_repo.search_by_question("apple")
```

### EvaluationRepository

```python
eval_repo = EvaluationRepository(db_manager)

# Basic operations
eval = eval_repo.create(eval_id="e1", run_id="exp1", ...)
eval = eval_repo.get_by_eval_id("e1")

# Query operations
run_evals = eval_repo.get_by_run("exp1")
model_evals = eval_repo.get_by_model("gpt-4", "exp1")
correct_evals = eval_repo.get_correct_evaluations("exp1")

# Analytics
stats = eval_repo.get_evaluation_stats("exp1")
performance = eval_repo.get_model_performance("exp1")
```

### AggregateRepository

```python
agg_repo = AggregateRepository(db_manager)

# Basic operations
agg = agg_repo.create(run_id="exp1", model_id="gpt-4", ...)

# Query operations
run_aggregates = agg_repo.get_by_run("exp1")
model_transform = agg_repo.get_by_model_and_transform("gpt-4", "scramble")
comparison = agg_repo.get_model_comparison("exp1")
top_models = agg_repo.get_best_performing_models("exp1", "original", limit=5)
```

### ParaphraseCacheRepository

```python
para_repo = ParaphraseCacheRepository(db_manager)

# Basic operations
para = para_repo.create(item_id="test1", provider="anthropic", ...)

# Query operations
item_paraphrases = para_repo.get_by_item("test1")
provider_paraphrases = para_repo.get_by_provider("anthropic")
accepted = para_repo.get_accepted_paraphrase("test1")
para_repo.accept_paraphrase(para_id)

# Analytics
stats = para_repo.get_quality_stats()
candidates = para_repo.find_high_quality_candidates(min_cos_sim=0.9)
```

## Database Migrations with Alembic

### Setup
The Alembic configuration is already set up and points to the ORM models.

### Common Operations

```bash
# Check current migration status
alembic current

# Create a new migration after model changes
alembic revision --autogenerate -m "Add new column to runs table"

# Apply migrations
alembic upgrade head

# Downgrade to previous version  
alembic downgrade -1

# Show migration history
alembic history --verbose
```

### Migration Best Practices

1. **Always review auto-generated migrations** before applying
2. **Test migrations on a copy** of production data
3. **Add data migrations** when schema changes require data transformation
4. **Keep migrations small** and focused on single changes

## Advanced Usage

### Custom Queries

```python
with db_manager.session_scope() as session:
    # Complex join query
    results = session.query(
        Run.run_id,
        Evaluation.model_id,
        func.avg(Evaluation.acc).label('avg_accuracy'),
        func.count(Evaluation.eval_id).label('eval_count')
    ).join(
        Evaluation, Run.run_id == Evaluation.run_id
    ).filter(
        Run.status == 'completed',
        Evaluation.transform == 'original'
    ).group_by(
        Run.run_id, Evaluation.model_id
    ).having(
        func.count(Evaluation.eval_id) > 100
    ).all()
    
    for result in results:
        print(f"{result.run_id} {result.model_id}: {result.avg_accuracy:.3f} ({result.eval_count} evals)")
```

### Batch Operations

```python
# Efficient batch creation
evaluations = []
for i in range(1000):
    eval_data = {
        'eval_id': f'eval_{i:06d}',
        'run_id': 'exp1',
        'item_id': f'item_{i}',
        # ... other fields
    }
    evaluations.append(Evaluation(**eval_data))

with db_manager.session_scope() as session:
    session.add_all(evaluations)
    # Bulk insert committed automatically
```

### Performance Considerations

1. **Use bulk operations** for large datasets
2. **Leverage database indexes** (already defined in models)
3. **Use select/joinload** for eager loading relationships
4. **Monitor query performance** with `echo=True` during development

## Error Handling

```python
from sqlalchemy.exc import SQLAlchemyError

try:
    with db_manager.session_scope() as session:
        # Database operations here
        pass
except SQLAlchemyError as e:
    logger.error(f"Database operation failed: {e}")
    # Handle error appropriately
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Integration with Existing ScrambleBench Code

### Migrating from Raw SQL

```python
# Old way: Raw SQL
cursor.execute("SELECT * FROM runs WHERE status = 'completed'")
results = cursor.fetchall()

# New way: ORM with Repository
run_repo = RunRepository(db_manager)
completed_runs = run_repo.get_completed_runs()
```

### Using in CLI Commands

```python
# In CLI command handlers
@click.command()
def analyze_run(run_id: str):
    db_manager = get_database_manager()
    run_repo = RunRepository(db_manager)
    eval_repo = EvaluationRepository(db_manager)
    
    run = run_repo.get_by_run_id(run_id)
    if not run:
        click.echo(f"Run {run_id} not found")
        return
    
    stats = eval_repo.get_evaluation_stats(run_id)
    click.echo(f"Run {run_id}: {stats['accuracy']:.2%} accuracy")
```

### Using in Analysis Modules

```python
# In analysis modules
def compute_model_comparison(run_id: str) -> Dict[str, Any]:
    db_manager = get_database_manager()
    agg_repo = AggregateRepository(db_manager)
    
    comparison_data = agg_repo.get_model_comparison(run_id)
    
    # Process and return analysis results
    return {
        'models': len(comparison_data),
        'best_model': max(comparison_data, key=lambda x: x['avg_accuracy']),
        'comparison_matrix': comparison_data
    }
```

## Configuration

### Database URL Format

```python
# SQLite (default)
database_url = "sqlite:///path/to/database.db"

# PostgreSQL
database_url = "postgresql://user:password@localhost:5432/scramblebench"

# MySQL
database_url = "mysql+pymysql://user:password@localhost:3306/scramblebench"
```

### Environment Variables

```bash
# Set database URL via environment
export SCRAMBLEBENCH_DATABASE_URL="sqlite:///data/scramblebench.db"

# Enable SQL debugging
export SCRAMBLEBENCH_DATABASE_ECHO=true
```

## Testing

### Unit Tests

```python
import pytest
from scramblebench.db import DatabaseManager, RunRepository

@pytest.fixture
def db_manager():
    """Create in-memory database for testing."""
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    manager.create_tables()
    return manager

@pytest.fixture  
def run_repo(db_manager):
    return RunRepository(db_manager)

def test_create_run(run_repo):
    run = run_repo.create(
        run_id="test_run",
        config_yaml="test: config",
        config_hash="test123",
        seed=42
    )
    
    assert run.run_id == "test_run"
    assert run.seed == 42
    
    # Test retrieval
    retrieved = run_repo.get_by_run_id("test_run")
    assert retrieved is not None
    assert retrieved.run_id == run.run_id
```

### Integration Tests

```python
def test_full_evaluation_workflow(db_manager):
    """Test complete workflow from run creation to analysis."""
    
    # Create repositories
    run_repo = RunRepository(db_manager)
    item_repo = ItemRepository(db_manager)
    eval_repo = EvaluationRepository(db_manager)
    
    # Create test data
    run = run_repo.create(run_id="integration_test", ...)
    item = item_repo.create(item_id="test_item", ...)
    evaluation = eval_repo.create(eval_id="test_eval", ...)
    
    # Test relationships and queries
    assert evaluation.run.run_id == run.run_id
    assert evaluation.item.item_id == item.item_id
    
    stats = eval_repo.get_evaluation_stats(run.run_id)
    assert stats['total_evaluations'] == 1
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'sqlalchemy'
   ```
   Solution: Install dependencies with `pip install -e .`

2. **Migration Conflicts**
   ```
   alembic.util.exc.CommandError: Target database is not up to date.
   ```
   Solution: Run `alembic upgrade head` to apply pending migrations

3. **Connection Issues**
   ```
   sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked
   ```
   Solution: Ensure no other processes are using the database file

### Debugging

1. **Enable SQL logging**: Set `echo=True` in DatabaseManager
2. **Check table creation**: Verify tables exist with `db_manager.create_tables()`
3. **Validate migrations**: Use `alembic current` and `alembic history`

## Best Practices

1. **Always use repositories** for data access instead of direct session queries
2. **Use session context managers** for automatic cleanup
3. **Test database operations** with in-memory SQLite databases
4. **Keep migrations small** and reviewable
5. **Document complex queries** with comments
6. **Handle errors gracefully** with proper exception handling
7. **Use type hints** for better code documentation and IDE support

## Examples

See `examples/orm_usage_example.py` for a complete working example that demonstrates all major ORM functionality.

---

This ORM layer provides a solid foundation for all database operations in ScrambleBench, following the specifications in TODO.md for architectural unification and enhanced research workflows.