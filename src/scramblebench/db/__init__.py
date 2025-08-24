"""
Database layer for ScrambleBench.

This module provides the SQLAlchemy-based database layer including:
- ORM models (re-exported from core.models)
- DatabaseManager for session management
- Repository classes for data access
- Database configuration and connection management

This serves as the single source of truth for all database operations
as specified in TODO.md requirements.
"""

from .models import (
    Base,
    Run,
    Item,
    Evaluation,
    Aggregate,
    ParaphraseCache,
    DatabaseStats,
    MigrationHistory
)
from .session import DatabaseManager
from .repository import (
    BaseRepository,
    RunRepository,
    ItemRepository,
    EvaluationRepository,
    AggregateRepository,
    ParaphraseCacheRepository
)

__all__ = [
    # ORM Models
    "Base",
    "Run", 
    "Item",
    "Evaluation",
    "Aggregate",
    "ParaphraseCache",
    "DatabaseStats",
    "MigrationHistory",
    # Session Management
    "DatabaseManager",
    # Repository Pattern
    "BaseRepository",
    "RunRepository",
    "ItemRepository", 
    "EvaluationRepository",
    "AggregateRepository",
    "ParaphraseCacheRepository"
]