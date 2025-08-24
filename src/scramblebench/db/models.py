"""
Database ORM models for ScrambleBench.

This module re-exports all SQLAlchemy ORM models from core.models,
providing them through the db layer as the single source of truth
for database schema as specified in TODO.md.

All database interactions should use these models through the
repository pattern or DatabaseManager.
"""

# Re-export all models from the core module
from scramblebench.core.models import (
    Base,
    Run,
    Item, 
    Evaluation,
    Aggregate,
    ParaphraseCache,
    DatabaseStats,
    MigrationHistory
)

__all__ = [
    "Base",
    "Run",
    "Item",
    "Evaluation", 
    "Aggregate",
    "ParaphraseCache",
    "DatabaseStats",
    "MigrationHistory"
]