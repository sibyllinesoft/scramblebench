"""
Tests for DatabaseManager session management.

Comprehensive tests for database connection, session lifecycle,
transaction management, and error handling.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from contextlib import contextmanager

from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy import text

from scramblebench.db.session import DatabaseManager, get_database_manager, initialize_database
from scramblebench.db.models import Run, Item


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    def test_initialization_default_sqlite(self):
        """Test DatabaseManager initialization with default SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory for default database creation
                import os
                os.chdir(temp_dir)
                
                manager = DatabaseManager()
                
                assert manager.database_url.startswith("sqlite:///")
                assert "scramblebench_sqlalchemy.db" in manager.database_url
                assert manager.echo is False
                assert manager._engine is not None
                assert manager._session_factory is not None
                
                # Verify db directory was created
                assert Path("db").exists()
                
            finally:
                os.chdir(original_cwd)
    
    def test_initialization_custom_database_url(self, test_db_url):
        """Test DatabaseManager initialization with custom URL."""
        manager = DatabaseManager(
            database_url=test_db_url,
            echo=True,
            pool_recycle=1800
        )
        
        assert manager.database_url == test_db_url
        assert manager.echo is True
        assert manager._engine is not None
        assert manager._session_factory is not None
    
    def test_sqlite_configuration(self, test_db_url):
        """Test SQLite-specific configuration settings."""
        manager = DatabaseManager(database_url=test_db_url)
        
        # Verify SQLite-specific engine configuration
        connect_args = manager._engine.pool._creator.keywords.get('connect_args', {})
        assert connect_args.get('check_same_thread') is False
        assert connect_args.get('timeout') == 20
    
    def test_create_tables_success(self, db_manager):
        """Test successful table creation."""
        # Tables should already be created by fixture, verify they exist
        with db_manager.session_scope() as session:
            # Try to query each table to verify it exists
            session.query(Run).count()  # Should not raise exception
            session.query(Item).count()
    
    def test_create_tables_error_handling(self, test_db_url):
        """Test error handling during table creation."""
        manager = DatabaseManager(database_url=test_db_url)
        
        with patch.object(manager._engine, 'execute') as mock_execute:
            mock_execute.side_effect = SQLAlchemyError("Table creation failed")
            
            with pytest.raises(SQLAlchemyError, match="Table creation failed"):
                manager.create_tables()
    
    def test_drop_tables_success(self, db_manager):
        """Test successful table dropping."""
        # First ensure tables exist
        with db_manager.session_scope() as session:
            session.query(Run).count()  # Should work
        
        # Drop tables
        db_manager.drop_tables()
        
        # Verify tables are gone
        with pytest.raises(Exception):  # Should raise OperationalError or similar
            with db_manager.session_scope() as session:
                session.query(Run).count()
    
    def test_drop_tables_error_handling(self, db_manager):
        """Test error handling during table dropping."""
        with patch.object(db_manager._engine, 'execute') as mock_execute:
            mock_execute.side_effect = SQLAlchemyError("Drop failed")
            
            with pytest.raises(SQLAlchemyError, match="Drop failed"):
                db_manager.drop_tables()
    
    def test_get_session(self, db_manager):
        """Test getting a new session."""
        session = db_manager.get_session()
        
        assert session is not None
        assert hasattr(session, 'query')
        assert hasattr(session, 'commit')
        assert hasattr(session, 'rollback')
        
        # Clean up
        session.close()
    
    def test_session_scope_success(self, db_manager, sample_run_data):
        """Test successful session scope context manager."""
        with db_manager.session_scope() as session:
            run = Run(**sample_run_data)
            session.add(run)
            # Transaction should be committed automatically
        
        # Verify data was committed
        with db_manager.session_scope() as session:
            saved_run = session.query(Run).filter_by(run_id=sample_run_data["run_id"]).first()
            assert saved_run is not None
            assert saved_run.name == sample_run_data["name"]
    
    def test_session_scope_rollback_on_exception(self, db_manager, sample_run_data):
        """Test session rollback when exception occurs."""
        initial_count = 0
        with db_manager.session_scope() as session:
            initial_count = session.query(Run).count()
        
        with pytest.raises(ValueError, match="Intentional error"):
            with db_manager.session_scope() as session:
                run = Run(**sample_run_data)
                session.add(run)
                session.flush()  # Force the add operation
                raise ValueError("Intentional error for testing")
        
        # Verify rollback occurred
        with db_manager.session_scope() as session:
            final_count = session.query(Run).count()
            assert final_count == initial_count
    
    def test_transaction_scope_success(self, db_manager, sample_run_data):
        """Test successful transaction scope context manager."""
        with db_manager.transaction_scope() as session:
            run = Run(**sample_run_data)
            session.add(run)
        
        # Verify data was committed
        with db_manager.session_scope() as session:
            saved_run = session.query(Run).filter_by(run_id=sample_run_data["run_id"]).first()
            assert saved_run is not None
    
    def test_transaction_scope_rollback(self, db_manager, sample_run_data):
        """Test transaction rollback on exception."""
        initial_count = 0
        with db_manager.session_scope() as session:
            initial_count = session.query(Run).count()
        
        with pytest.raises(ValueError, match="Transaction test error"):
            with db_manager.transaction_scope() as session:
                run = Run(**sample_run_data)
                session.add(run)
                session.flush()
                raise ValueError("Transaction test error")
        
        # Verify rollback
        with db_manager.session_scope() as session:
            final_count = session.query(Run).count()
            assert final_count == initial_count
    
    def test_execute_raw_sql_success(self, db_manager):
        """Test successful raw SQL execution."""
        result = db_manager.execute_raw_sql("SELECT 1 as test_value")
        
        assert result is not None
        row = result.fetchone()
        assert row[0] == 1
    
    def test_execute_raw_sql_with_parameters(self, db_manager, sample_run_data):
        """Test raw SQL execution with parameters."""
        # First insert a run
        with db_manager.session_scope() as session:
            run = Run(**sample_run_data)
            session.add(run)
        
        # Query with parameters
        result = db_manager.execute_raw_sql(
            "SELECT name FROM runs WHERE run_id = :run_id",
            {"run_id": sample_run_data["run_id"]}
        )
        
        row = result.fetchone()
        assert row[0] == sample_run_data["name"]
    
    def test_execute_raw_sql_error_handling(self, db_manager):
        """Test error handling in raw SQL execution."""
        with pytest.raises(SQLAlchemyError):
            db_manager.execute_raw_sql("SELECT * FROM nonexistent_table")
    
    def test_health_check_success(self, db_manager):
        """Test successful database health check."""
        health_status = db_manager.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["connection_test"] is True
        assert "database_url" in health_status
        assert "***" in health_status["database_url"]  # Credentials should be masked
    
    def test_health_check_failure(self, test_db_url):
        """Test health check failure handling."""
        # Create manager with invalid database
        invalid_url = "sqlite:///nonexistent/path/to/database.db"
        manager = DatabaseManager(database_url=invalid_url)
        
        # Health check should handle the error gracefully
        health_status = manager.health_check()
        
        assert health_status["status"] == "unhealthy"
        assert "error" in health_status
        assert "database_url" in health_status
    
    def test_credentials_masking(self, test_db_url):
        """Test that credentials are properly masked in logs."""
        # Test with URL containing password
        url_with_creds = "postgresql://user:secret123@localhost/db"
        manager = DatabaseManager(database_url=url_with_creds)
        
        masked_url = manager._mask_credentials(url_with_creds)
        assert "secret123" not in masked_url
        assert "***" in masked_url
        assert "user" in masked_url  # Username should remain
    
    def test_credentials_masking_no_password(self, test_db_url):
        """Test credentials masking when no password present."""
        url_no_creds = "sqlite:///test.db"
        manager = DatabaseManager()
        
        masked_url = manager._mask_credentials(url_no_creds)
        assert masked_url == url_no_creds  # Should be unchanged
    
    def test_credentials_masking_invalid_url(self):
        """Test credentials masking with invalid URL."""
        manager = DatabaseManager()
        
        # Should handle invalid URLs gracefully
        masked_url = manager._mask_credentials("not-a-valid-url")
        assert masked_url == "***"  # Fallback for parsing errors
    
    def test_close_connections(self, db_manager):
        """Test closing database connections and cleanup."""
        # Get a session to establish connection
        with db_manager.session_scope() as session:
            session.query(Run).count()
        
        # Close should not raise exception
        db_manager.close()
        
        # Engine should be disposed
        assert db_manager._engine.pool.size() == 0
    
    def test_close_connections_with_error(self, db_manager):
        """Test connection cleanup with error handling."""
        with patch.object(db_manager._engine, 'dispose') as mock_dispose:
            mock_dispose.side_effect = Exception("Cleanup error")
            
            # Should handle cleanup errors gracefully
            db_manager.close()
            assert mock_dispose.called
    
    def test_concurrent_session_access(self, db_manager):
        """Test concurrent access to database sessions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def database_worker(worker_id):
            try:
                with db_manager.session_scope() as session:
                    # Simulate some work
                    count = session.query(Run).count()
                    time.sleep(0.01)  # Small delay to increase concurrency chance
                    results.append(f"Worker {worker_id}: {count}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=database_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5
    
    def test_sqlite_pragma_settings(self, test_db_url):
        """Test that SQLite pragma settings are applied correctly."""
        manager = DatabaseManager(database_url=test_db_url)
        manager.create_tables()  # This should trigger the pragma settings
        
        # Check pragma settings directly via SQLite connection
        with manager.session_scope() as session:
            # Test foreign key constraints
            result = session.execute(text("PRAGMA foreign_keys")).fetchone()
            assert result[0] == 1  # Should be enabled
            
            # Test journal mode
            result = session.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0].upper() == "WAL"
            
            # Test synchronous mode
            result = session.execute(text("PRAGMA synchronous")).fetchone()
            assert result[0] == 1  # NORMAL mode


class TestGlobalFunctions:
    """Test cases for global database management functions."""
    
    def test_get_database_manager_first_call(self, test_db_url):
        """Test getting database manager on first call."""
        # Clear any existing global manager
        import scramblebench.db.session
        scramblebench.db.session._db_manager = None
        
        manager = get_database_manager(database_url=test_db_url)
        
        assert manager is not None
        assert manager.database_url == test_db_url
    
    def test_get_database_manager_subsequent_calls(self, test_db_url):
        """Test getting database manager on subsequent calls."""
        # Clear any existing global manager
        import scramblebench.db.session
        scramblebench.db.session._db_manager = None
        
        # First call
        manager1 = get_database_manager(database_url=test_db_url)
        
        # Second call (should return same instance, ignoring new URL)
        manager2 = get_database_manager(database_url="different_url")
        
        assert manager1 is manager2
        assert manager2.database_url == test_db_url  # Original URL preserved
    
    def test_initialize_database_with_table_creation(self, test_db_url):
        """Test database initialization with table creation."""
        # Clear any existing global manager
        import scramblebench.db.session
        scramblebench.db.session._db_manager = None
        
        manager = initialize_database(
            database_url=test_db_url,
            create_tables=True
        )
        
        assert manager is not None
        
        # Verify tables were created
        with manager.session_scope() as session:
            # Should not raise exception
            session.query(Run).count()
            session.query(Item).count()
    
    def test_initialize_database_without_table_creation(self, test_db_url):
        """Test database initialization without table creation."""
        # Clear any existing global manager
        import scramblebench.db.session
        scramblebench.db.session._db_manager = None
        
        manager = initialize_database(
            database_url=test_db_url,
            create_tables=False
        )
        
        assert manager is not None
        
        # Tables should not exist yet
        with pytest.raises(Exception):  # OperationalError or similar
            with manager.session_scope() as session:
                session.query(Run).count()


class TestDatabaseManagerEdgeCases:
    """Test edge cases and error scenarios for DatabaseManager."""
    
    def test_multiple_simultaneous_transactions(self, db_manager, sample_run_data):
        """Test handling multiple simultaneous transactions."""
        # This test verifies that the database can handle overlapping transactions
        # without deadlocks or corruption
        
        def create_run_with_unique_id(run_id):
            data = sample_run_data.copy()
            data["run_id"] = run_id
            with db_manager.session_scope() as session:
                run = Run(**data)
                session.add(run)
                return run_id
        
        import concurrent.futures
        
        # Create multiple runs simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(create_run_with_unique_id, f"concurrent-run-{i}")
                for i in range(3)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all runs were created
        assert len(results) == 3
        
        with db_manager.session_scope() as session:
            saved_runs = session.query(Run).filter(
                Run.run_id.like("concurrent-run-%")
            ).all()
            assert len(saved_runs) == 3
    
    def test_session_scope_nested_exceptions(self, db_manager, sample_run_data):
        """Test nested exception handling in session scope."""
        initial_count = 0
        with db_manager.session_scope() as session:
            initial_count = session.query(Run).count()
        
        with pytest.raises(RuntimeError, match="Outer error"):
            with db_manager.session_scope() as session:
                run = Run(**sample_run_data)
                session.add(run)
                
                try:
                    raise ValueError("Inner error")
                except ValueError:
                    # Transform into different exception
                    raise RuntimeError("Outer error")
        
        # Verify rollback occurred
        with db_manager.session_scope() as session:
            final_count = session.query(Run).count()
            assert final_count == initial_count
    
    def test_raw_sql_with_complex_query(self, db_manager, populated_database):
        """Test raw SQL execution with complex queries."""
        # Test complex query with joins, aggregations, etc.
        complex_query = """
        SELECT 
            r.run_id,
            r.name,
            COUNT(e.eval_id) as eval_count,
            AVG(e.latency_ms) as avg_latency
        FROM runs r
        LEFT JOIN evaluations e ON r.run_id = e.run_id
        WHERE r.status = :status
        GROUP BY r.run_id, r.name
        ORDER BY eval_count DESC
        """
        
        result = db_manager.execute_raw_sql(
            complex_query,
            {"status": "running"}
        )
        
        rows = result.fetchall()
        assert len(rows) >= 0  # May be 0 or more depending on test data
        
        # If there are results, verify structure
        if rows:
            row = rows[0]
            assert len(row) == 4  # run_id, name, eval_count, avg_latency
            assert row[0] is not None  # run_id should not be null
    
    def test_database_manager_properties(self, db_manager):
        """Test DatabaseManager properties and attributes."""
        # Test engine property
        engine = db_manager.engine
        assert engine is not None
        assert hasattr(engine, 'execute')
        assert hasattr(engine, 'dispose')
        
        # Test that engine property returns same instance
        assert db_manager.engine is engine
        
        # Test database URL property
        assert db_manager.database_url.startswith("sqlite:///")
        
        # Test echo property
        assert isinstance(db_manager.echo, bool)