#!/usr/bin/env python3
"""
Example usage of the ScrambleBench ORM layer.

This example demonstrates:
1. Database initialization
2. Creating and querying runs
3. Adding evaluations 
4. Using repositories for data access
5. Computing aggregates

Run with: python examples/orm_usage_example.py
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ORM imports  
from scramblebench.db import (
    DatabaseManager, 
    Run, Item, Evaluation, Aggregate,
    RunRepository, ItemRepository, EvaluationRepository, AggregateRepository
)

def main():
    """Demonstrate ORM usage."""
    print("ScrambleBench ORM Usage Example")
    print("=" * 40)
    
    # 1. Initialize database
    print("1. Initializing database...")
    db_manager = DatabaseManager(
        database_url="sqlite:///examples/demo.db",
        echo=True  # Show SQL statements
    )
    db_manager.create_tables()
    print("✓ Database initialized\n")
    
    # 2. Create repositories
    print("2. Creating repositories...")
    run_repo = RunRepository(db_manager)
    item_repo = ItemRepository(db_manager)
    eval_repo = EvaluationRepository(db_manager) 
    agg_repo = AggregateRepository(db_manager)
    print("✓ Repositories created\n")
    
    # 3. Create a sample run
    print("3. Creating sample run...")
    sample_run = run_repo.create(
        run_id="demo_run_001",
        config_yaml="model: gpt-3.5-turbo\ndataset: demo",
        config_hash="abc123",
        seed=42,
        total_evaluations=5,
        git_sha="1234567890abcdef"
    )
    print(f"✓ Created run: {sample_run.run_id}\n")
    
    # 4. Create sample items
    print("4. Creating sample items...")
    sample_items = []
    for i in range(3):
        item = item_repo.create(
            item_id=f"demo_item_{i:03d}",
            dataset="demo",
            domain="math",
            question=f"What is {i+1} + {i+1}?",
            answer=str((i+1) * 2)
        )
        sample_items.append(item)
    print(f"✓ Created {len(sample_items)} items\n")
    
    # 5. Create sample evaluations
    print("5. Creating sample evaluations...")
    models = ["gpt-3.5-turbo", "gpt-4"] 
    transforms = ["original", "scramble"]
    
    eval_count = 0
    for item in sample_items:
        for model in models:
            for transform in transforms:
                # Simulate some variation in correctness
                is_correct = (eval_count % 3) != 0  # 66% accuracy
                
                evaluation = eval_repo.create(
                    eval_id=f"eval_{eval_count:06d}",
                    run_id=sample_run.run_id,
                    item_id=item.item_id,
                    model_id=model,
                    provider="openai",
                    transform=transform,
                    prompt=f"Question: {item.question}",
                    completion=item.answer if is_correct else "Wrong answer",
                    is_correct=is_correct,
                    acc=1.0 if is_correct else 0.0,
                    prompt_tokens=50,
                    completion_tokens=10,
                    latency_ms=1200 + (eval_count * 100),
                    cost_usd=0.001 * (eval_count + 1),
                    seed=42
                )
                eval_count += 1
    
    print(f"✓ Created {eval_count} evaluations\n")
    
    # 6. Query data using repositories
    print("6. Querying data with repositories...")
    
    # Get run details
    run = run_repo.get_by_run_id("demo_run_001")
    print(f"✓ Run {run.run_id}: {run.progress_percentage:.1f}% complete")
    
    # Get evaluation stats
    stats = eval_repo.get_evaluation_stats(sample_run.run_id)
    print(f"✓ Evaluation stats: {stats['accuracy']:.2%} accuracy, ${stats['total_cost_usd']:.3f} cost")
    
    # Get model performance
    performance = eval_repo.get_model_performance(sample_run.run_id)
    for perf in performance:
        print(f"✓ {perf['model_id']}: {perf['accuracy']:.2%} accuracy")
    
    # Get items by dataset
    demo_items = item_repo.get_by_dataset("demo")
    print(f"✓ Found {len(demo_items)} items in demo dataset")
    
    print()
    
    # 7. Demonstrate session management
    print("7. Demonstrating session management...")
    with db_manager.session_scope() as session:
        # Complex query across tables
        results = session.query(
            Evaluation.model_id,
            Evaluation.transform,
            Item.domain,
        ).join(Item).filter(
            Evaluation.run_id == sample_run.run_id,
            Evaluation.is_correct == True
        ).all()
        
        print(f"✓ Found {len(results)} correct evaluations")
        for result in results[:3]:  # Show first 3
            print(f"  - {result.model_id} ({result.transform}) in {result.domain}")
    
    print()
    
    # 8. Create sample aggregates
    print("8. Creating sample aggregates...")
    for model in models:
        for transform in transforms:
            # Calculate accuracy for this model/transform combination
            model_evals = eval_repo.get_by_model(model, sample_run.run_id)
            transform_evals = [e for e in model_evals if e.transform == transform]
            
            if transform_evals:
                accuracy = sum(e.is_correct for e in transform_evals) / len(transform_evals)
                
                aggregate = agg_repo.create(
                    run_id=sample_run.run_id,
                    model_id=model,
                    dataset="demo",
                    domain="math",
                    transform=transform,
                    acc_mean=accuracy,
                    acc_ci_low=max(0, accuracy - 0.1),
                    acc_ci_high=min(1, accuracy + 0.1),
                    rrs=0.8 if transform == "scramble" else None,
                    ldc=0.3 if transform == "scramble" else None,
                    n_items=len(transform_evals)
                )
                print(f"✓ Created aggregate for {model} {transform}: {accuracy:.2%}")
    
    print()
    
    # 9. Health check
    print("9. Database health check...")
    health = db_manager.health_check()
    if health['status'] == 'healthy':
        print("✓ Database is healthy")
    else:
        print(f"✗ Database health check failed: {health.get('error', 'Unknown error')}")
    
    print()
    
    # 10. Summary
    print("10. Summary")
    print(f"✓ Created 1 run with {eval_count} evaluations")
    print(f"✓ Database file: {db_manager.database_url}")
    print(f"✓ All ORM operations completed successfully")
    
    print("\n" + "=" * 40)
    print("🎉 ORM usage example completed!")
    print("\nThe ORM layer provides:")
    print("- Clean separation between business logic and data access")
    print("- Type-safe database operations") 
    print("- Automatic session and transaction management")
    print("- Repository pattern for organized data access")
    print("- Full integration with existing ScrambleBench models")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    main()