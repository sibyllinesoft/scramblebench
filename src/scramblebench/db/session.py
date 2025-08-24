"""
Database session management for ScrambleBench.

This module provides the DatabaseManager class that wraps SQLAlchemy
session management and provides a clean interface for database operations.
As specified in TODO.md, all database interactions must go through this
DatabaseManager class.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict
from urllib.parse import urlparse

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database session manager that wraps SQLAlchemy session lifecycle.
    
    Provides:
    - Connection pooling and configuration
    - Session management and lifecycle
    - Transaction management
    - Error handling and logging
    - Schema creation and migrations
    
    All database interactions in ScrambleBench must go through this class
    as specified in TODO.md requirements.
    """
    
    def __init__(
        self, 
        database_url: Optional[str] = None,
        echo: bool = False,
        pool_recycle: int = 3600,
        **engine_kwargs: Any
    ):
        """
        Initialize DatabaseManager.
        
        Args:
            database_url: SQLAlchemy database URL. If None, uses default SQLite
            echo: Whether to echo SQL statements for debugging
            pool_recycle: Connection pool recycle time in seconds
            **engine_kwargs: Additional engine configuration
        """
        if database_url is None:
            # Default to SQLite database in project db/ directory
            db_dir = os.path.join(os.getcwd(), "db")
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{db_dir}/scramblebench_sqlalchemy.db"
        
        self.database_url = database_url
        self.echo = echo
        
        # Configure engine based on database type
        engine_config = {
            "echo": echo,
            "pool_recycle": pool_recycle,
            **engine_kwargs
        }
        
        # SQLite-specific configuration
        if database_url.startswith("sqlite"):
            engine_config.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20
                }
            })
        
        self._engine = create_engine(database_url, **engine_config)
        self._session_factory = sessionmaker(bind=self._engine)
        
        # Configure SQLite for better performance and integrity
        if database_url.startswith("sqlite"):
            self._configure_sqlite()
        
        logger.info(f"DatabaseManager initialized with URL: {self._mask_credentials(database_url)}")
    
    def _configure_sqlite(self) -> None:
        """Configure SQLite-specific settings for performance and integrity."""
        
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for optimal performance and reliability."""
            cursor = dbapi_connection.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Use WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Synchronous mode for data safety
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Cache size (negative value = KB)
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            # Timeout for busy database
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            cursor.close()
    
    def _mask_credentials(self, url: str) -> str:
        """Mask credentials in database URL for logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                masked_url = url.replace(parsed.password, "***")
                return masked_url
            return url
        except Exception:
            return "***"
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        return self._engine
    
    def create_tables(self) -> None:
        """
        Create all database tables based on ORM models.
        
        This will create tables for all models defined in the Base metadata.
        Should be called during application initialization.
        """
        try:
            Base.metadata.create_all(self._engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """
        Drop all database tables.
        
        WARNING: This will destroy all data. Use with extreme caution.
        """
        try:
            Base.metadata.drop_all(self._engine)
            logger.warning("All database tables dropped")
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy Session instance
            
        Note:
            Caller is responsible for closing the session.
            Consider using session_scope() context manager instead.
        """
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic cleanup.
        
        Provides:
        - Automatic session creation and cleanup
        - Transaction management with automatic rollback on error
        - Exception handling and logging
        
        Example:
            with db_manager.session_scope() as session:
                run = session.query(Run).filter_by(run_id="test").first()
                # ... do work with session
            # Session is automatically closed and committed/rolled back
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back due to error: {e}")
            raise
        finally:
            session.close()
            logger.debug("Database session closed")
    
    @contextmanager 
    def transaction_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for explicit transaction management.
        
        Similar to session_scope() but with explicit transaction control.
        Useful when you need fine-grained transaction management.
        
        Example:
            with db_manager.transaction_scope() as session:
                # Multiple operations in single transaction
                session.add(run)
                session.add(evaluation)
                # Transaction committed automatically on success
        """
        session = self.get_session()
        try:
            session.begin()
            yield session
            session.commit()
            logger.debug("Explicit transaction committed successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Explicit transaction rolled back due to error: {e}")
            raise
        finally:
            session.close()
            logger.debug("Transaction session closed")
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute raw SQL statement.
        
        Args:
            sql: Raw SQL statement
            params: Optional parameters for the SQL statement
            
        Returns:
            Result of the SQL execution
            
        Note:
            Use with caution. Prefer ORM operations when possible.
            This is provided for complex queries that are difficult in ORM.
        """
        with self.session_scope() as session:
            try:
                result = session.execute(sql, params or {})
                logger.debug(f"Executed raw SQL: {sql[:100]}...")
                return result
            except SQLAlchemyError as e:
                logger.error(f"Raw SQL execution failed: {e}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dictionary containing health check results
        """
        try:
            with self.session_scope() as session:
                # Simple query to test connectivity
                result = session.execute("SELECT 1").scalar()
                
                # Get some basic stats
                stats = {
                    "status": "healthy",
                    "database_url": self._mask_credentials(self.database_url),
                    "connection_test": result == 1,
                    "engine_pool_size": getattr(self._engine.pool, 'size', None),
                    "engine_pool_checked_out": getattr(self._engine.pool, 'checked_out', None)
                }
                
                logger.info("Database health check passed")
                return stats
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self._mask_credentials(self.database_url)
            }
    
    def close(self) -> None:
        """
        Close all database connections and clean up resources.
        
        Should be called during application shutdown.
        """
        try:
            self._engine.dispose()
            logger.info("Database connections closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")


# Global database manager instance
# This can be configured and used throughout the application
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(
    database_url: Optional[str] = None,
    **kwargs: Any
) -> DatabaseManager:
    """
    Get or create the global DatabaseManager instance.
    
    Args:
        database_url: Database URL (only used on first call)
        **kwargs: Additional DatabaseManager configuration
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url=database_url, **kwargs)
    
    return _db_manager


def initialize_database(
    database_url: Optional[str] = None,
    create_tables: bool = True,
    **kwargs: Any
) -> DatabaseManager:
    """
    Initialize the database system.
    
    Args:
        database_url: Database URL
        create_tables: Whether to create tables automatically
        **kwargs: Additional DatabaseManager configuration
        
    Returns:
        Configured DatabaseManager instance
    """
    db_manager = get_database_manager(database_url=database_url, **kwargs)
    
    if create_tables:
        db_manager.create_tables()
    
    logger.info("Database system initialized successfully")
    return db_manager