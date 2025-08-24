"""
Database configuration and session management for ScrambleBench SQLAlchemy ORM.

This module provides database connection management, session factories,
and configuration for the SQLAlchemy ORM layer that works alongside
the existing DuckDB implementation.
"""

import os
from typing import Optional, Generator
from pathlib import Path

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base
from .unified_config import ScrambleBenchConfig

# Global session maker - will be configured by setup_database
SessionLocal: Optional[sessionmaker] = None
engine: Optional[Engine] = None


def get_database_url(config: Optional[ScrambleBenchConfig] = None) -> str:
    """
    Get database URL from configuration or environment.
    
    Args:
        config: ScrambleBench configuration object
        
    Returns:
        Database URL string
    """
    # Check environment variable first
    if db_url := os.getenv("SCRAMBLEBENCH_DATABASE_URL"):
        return db_url
    
    # Use config if provided
    if config and hasattr(config, 'db') and hasattr(config.db, 'uri'):
        db_path = Path(config.db.uri)
        # Convert DuckDB path to SQLite for SQLAlchemy compatibility
        if str(db_path).endswith('.duckdb'):
            sqlite_path = str(db_path).replace('.duckdb', '_sqlalchemy.db')
        else:
            sqlite_path = str(db_path) + '_sqlalchemy.db'
        return f"sqlite:///{sqlite_path}"
    
    # Default to SQLite database in db directory
    default_path = Path("db/scramblebench_sqlalchemy.db")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{default_path}"


def setup_database(
    database_url: Optional[str] = None,
    config: Optional[ScrambleBenchConfig] = None,
    echo: bool = False
) -> Engine:
    """
    Set up database engine and session factory.
    
    Args:
        database_url: Database URL (if None, will be determined from config/env)
        config: ScrambleBench configuration
        echo: Whether to echo SQL queries (for debugging)
        
    Returns:
        SQLAlchemy Engine instance
    """
    global engine, SessionLocal
    
    if database_url is None:
        database_url = get_database_url(config)
    
    # Configure engine based on database type
    if database_url.startswith("sqlite://"):
        # SQLite-specific configuration
        engine = create_engine(
            database_url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        # PostgreSQL or other database configuration
        engine = create_engine(database_url, echo=echo)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine


def create_tables(engine_instance: Optional[Engine] = None) -> None:
    """
    Create all tables in the database.
    
    Args:
        engine_instance: SQLAlchemy engine (uses global engine if None)
    """
    if engine_instance is None:
        engine_instance = engine
    
    if engine_instance is None:
        raise RuntimeError("Database not initialized. Call setup_database() first.")
    
    Base.metadata.create_all(bind=engine_instance)


def drop_tables(engine_instance: Optional[Engine] = None) -> None:
    """
    Drop all tables in the database.
    
    WARNING: This will delete all data!
    
    Args:
        engine_instance: SQLAlchemy engine (uses global engine if None)
    """
    if engine_instance is None:
        engine_instance = engine
    
    if engine_instance is None:
        raise RuntimeError("Database not initialized. Call setup_database() first.")
    
    Base.metadata.drop_all(bind=engine_instance)


def get_session() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.
    
    This is a generator function that yields a database session
    and ensures it's properly closed after use.
    
    Yields:
        SQLAlchemy Session instance
        
    Example:
        ```python
        with next(get_session()) as session:
            runs = session.query(Run).all()
        ```
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call setup_database() first.")
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_session_sync() -> Session:
    """
    Get database session for synchronous usage.
    
    Note: Remember to close the session when done!
    
    Returns:
        SQLAlchemy Session instance
        
    Example:
        ```python
        session = get_session_sync()
        try:
            runs = session.query(Run).all()
        finally:
            session.close()
        ```
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call setup_database() first.")
    
    return SessionLocal()


def init_database(
    config: Optional[ScrambleBenchConfig] = None,
    create_schema: bool = True,
    echo: bool = False
) -> Engine:
    """
    Initialize database with proper configuration.
    
    This is a convenience function that sets up the database,
    creates tables if requested, and returns the engine.
    
    Args:
        config: ScrambleBench configuration
        create_schema: Whether to create database schema
        echo: Whether to echo SQL queries
        
    Returns:
        SQLAlchemy Engine instance
    """
    engine_instance = setup_database(config=config, echo=echo)
    
    if create_schema:
        create_tables(engine_instance)
    
    return engine_instance


# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions."""
    
    def __init__(self):
        if SessionLocal is None:
            raise RuntimeError("Database not initialized. Call setup_database() first.")
        self.session = SessionLocal()
    
    def __enter__(self) -> Session:
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()


def get_database_info() -> dict:
    """
    Get information about the current database configuration.
    
    Returns:
        Dictionary with database information
    """
    if engine is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "url": str(engine.url).replace(engine.url.password or "", "***") if engine.url.password else str(engine.url),
        "dialect": engine.dialect.name,
        "driver": engine.dialect.driver,
        "pool_size": getattr(engine.pool, 'size', 'N/A'),
        "pool_checked_out": getattr(engine.pool, 'checked_out', 'N/A'),
    }