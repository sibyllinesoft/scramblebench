"""
Database Management System for Experiment Tracking

Advanced database interface for experiment tracking with full academic research
support, optimized queries, and data integrity validation.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID
from pathlib import Path
import pandas as pd

from sqlalchemy import create_engine, text, MetaData, Table, select, insert, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import asyncpg

from .core import ExperimentMetadata, ExperimentStatus
from .monitor import ProgressSnapshot
from .statistics import SignificanceTest, ABTestResult, LanguageDependencyAnalysis
from ..core.unified_config import ScrambleBenchConfig
from ..evaluation.results import EvaluationResults


class DatabaseManager:
    """
    Advanced database manager for experiment tracking with optimized queries,
    connection pooling, and comprehensive data management capabilities.
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL database URL
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            logger: Logger instance
        """
        self.database_url = database_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Create async engine with connection pooling
        self.async_engine = create_async_engine(
            database_url.replace('postgresql://', 'postgresql+asyncpg://'),
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Database metadata
        self.metadata = MetaData()
        
        self.logger.info(f"Database manager initialized for {database_url}")
    
    async def initialize_database(self) -> None:
        """Initialize database schema if needed"""
        try:
            # Load schema from SQL file
            schema_path = Path(__file__).parent.parent.parent.parent / "schema" / "language_dependency_atlas.sql"
            
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                # Execute schema creation
                async with self.async_engine.begin() as conn:
                    await conn.execute(text(schema_sql))
                
                self.logger.info("Database schema initialized successfully")
            else:
                self.logger.warning("Schema file not found - assuming database is already initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    async def create_experiment(
        self,
        metadata: ExperimentMetadata,
        config: EvaluationConfig,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Create new experiment in database
        
        Args:
            metadata: Experiment metadata
            config: Evaluation configuration
            tags: Optional tags for organization
        """
        async with self.async_session_factory() as session:
            try:
                # Insert into experiments table
                experiment_data = {
                    'experiment_id': metadata.experiment_id,
                    'experiment_name': metadata.name,
                    'description': metadata.description,
                    'research_question': metadata.research_question,
                    'hypothesis': metadata.hypothesis,
                    'git_commit_hash': metadata.git_commit_hash,
                    'git_branch': metadata.git_branch,
                    'environment_snapshot': metadata.environment_snapshot,
                    'random_seed': metadata.random_seed,
                    'created_at': metadata.created_at,
                    'status': metadata.status.value,
                    'researcher_name': metadata.researcher_name,
                    'institution': metadata.institution,
                    'configuration': config.to_dict()
                }
                
                result = await session.execute(
                    text("""
                        INSERT INTO experiments (
                            experiment_id, experiment_name, description, research_question,
                            hypothesis, git_commit_hash, git_branch, environment_snapshot,
                            random_seed, created_at, status, researcher_name, institution,
                            configuration
                        ) VALUES (
                            :experiment_id, :experiment_name, :description, :research_question,
                            :hypothesis, :git_commit_hash, :git_branch, :environment_snapshot,
                            :random_seed, :created_at, :status, :researcher_name, :institution,
                            :configuration
                        )
                    """),
                    experiment_data
                )
                
                # Add tags if provided
                if tags:
                    for tag in tags:
                        await session.execute(
                            text("""
                                INSERT INTO experiment_tags (experiment_id, tag)
                                VALUES (:experiment_id, :tag)
                                ON CONFLICT DO NOTHING
                            """),
                            {'experiment_id': metadata.experiment_id, 'tag': tag}
                        )
                
                await session.commit()
                self.logger.info(f"Created experiment {metadata.experiment_id} in database")
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to create experiment in database: {e}")
                raise
    
    async def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update experiment status
        
        Args:
            experiment_id: Experiment ID
            status: New status
            additional_data: Additional fields to update
        """
        async with self.async_session_factory() as session:
            try:
                update_data = {'status': status.value}
                
                # Add timestamp based on status
                if status == ExperimentStatus.RUNNING:
                    update_data['started_at'] = datetime.now()
                elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                    update_data['completed_at'] = datetime.now()
                
                if additional_data:
                    update_data.update(additional_data)
                
                # Build dynamic UPDATE query
                set_clauses = [f"{key} = :{key}" for key in update_data.keys()]
                sql = f"""
                    UPDATE experiments 
                    SET {', '.join(set_clauses)}
                    WHERE experiment_id = :experiment_id
                """
                
                update_data['experiment_id'] = experiment_id
                
                await session.execute(text(sql), update_data)
                await session.commit()
                
                self.logger.info(f"Updated experiment {experiment_id} status to {status.value}")
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to update experiment status: {e}")
                raise
    
    async def update_experiment_metadata(self, metadata: ExperimentMetadata) -> None:
        """Update complete experiment metadata"""
        async with self.async_session_factory() as session:
            try:
                update_data = {
                    'experiment_id': metadata.experiment_id,
                    'status': metadata.status.value,
                    'started_at': metadata.started_at,
                    'completed_at': metadata.completed_at,
                    'progress': metadata.progress,
                    'current_stage': metadata.current_stage,
                    'total_api_calls': metadata.total_api_calls,
                    'total_cost': metadata.total_cost,
                    'compute_hours': metadata.compute_hours
                }
                
                await session.execute(
                    text("""
                        UPDATE experiments SET
                            status = :status,
                            started_at = :started_at,
                            completed_at = :completed_at
                        WHERE experiment_id = :experiment_id
                    """),
                    update_data
                )
                
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to update experiment metadata: {e}")
                raise
    
    async def get_experiment_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata by ID"""
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT experiment_id, experiment_name, description, research_question,
                               hypothesis, git_commit_hash, git_branch, environment_snapshot,
                               random_seed, created_at, started_at, completed_at, status,
                               researcher_name, institution
                        FROM experiments
                        WHERE experiment_id = :experiment_id
                    """),
                    {'experiment_id': experiment_id}
                )
                
                row = result.fetchone()
                if not row:
                    return None
                
                return ExperimentMetadata(
                    experiment_id=row[0],
                    name=row[1],
                    description=row[2],
                    research_question=row[3],
                    hypothesis=row[4],
                    git_commit_hash=row[5],
                    git_branch=row[6],
                    environment_snapshot=row[7],
                    random_seed=row[8],
                    created_at=row[9],
                    started_at=row[10],
                    completed_at=row[11],
                    status=ExperimentStatus(row[12]),
                    researcher_name=row[13],
                    institution=row[14],
                    config_hash=""  # Would need to calculate from stored config
                )
                
            except Exception as e:
                self.logger.error(f"Failed to get experiment metadata: {e}")
                raise
    
    async def get_experiment_config(self, experiment_id: str) -> Optional[EvaluationConfig]:
        """Get experiment configuration by ID"""
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(
                    text("SELECT configuration FROM experiments WHERE experiment_id = :experiment_id"),
                    {'experiment_id': experiment_id}
                )
                
                row = result.fetchone()
                if not row or not row[0]:
                    return None
                
                config_dict = row[0]
                return EvaluationConfig(**config_dict)
                
            except Exception as e:
                self.logger.error(f"Failed to get experiment config: {e}")
                raise
    
    async def save_experiment_results(
        self,
        experiment_id: str,
        results: EvaluationResults
    ) -> None:
        """
        Save experiment results to database
        
        Args:
            experiment_id: Experiment ID
            results: Evaluation results to save
        """
        async with self.async_session_factory() as session:
            try:
                # First ensure models exist
                for result in results.results:
                    if hasattr(result, 'model_id'):
                        await self._ensure_model_exists(session, result.model_id)
                
                # Save individual responses
                for result in results.results:
                    response_data = {
                        'experiment_id': experiment_id,
                        'model_id': getattr(result, 'model_id', 'unknown'),
                        'raw_response': getattr(result, 'response', ''),
                        'is_correct': result.success,
                        'response_time_ms': getattr(result, 'response_time_ms', 0),
                        'cost': getattr(result, 'cost', 0.0),
                        'created_at': datetime.now()
                    }
                    
                    await session.execute(
                        text("""
                            INSERT INTO responses (
                                experiment_id, model_id, raw_response, is_correct,
                                response_time_ms, cost, created_at
                            ) VALUES (
                                :experiment_id, :model_id, :raw_response, :is_correct,
                                :response_time_ms, :cost, :created_at
                            )
                        """),
                        response_data
                    )
                
                await session.commit()
                self.logger.info(f"Saved {len(results.results)} results for experiment {experiment_id}")
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to save experiment results: {e}")
                raise
    
    async def save_progress_snapshot(
        self,
        experiment_id: str,
        snapshot: ProgressSnapshot
    ) -> None:
        """Save progress snapshot to database"""
        async with self.async_session_factory() as session:
            try:
                snapshot_data = {
                    'experiment_id': experiment_id,
                    'timestamp': snapshot.timestamp,
                    'stage': snapshot.stage,
                    'progress': snapshot.progress,
                    'details': snapshot.details,
                    'cpu_percent': snapshot.cpu_percent,
                    'memory_mb': snapshot.memory_mb,
                    'api_calls_count': snapshot.api_calls_count,
                    'cost_incurred': snapshot.cost_incurred,
                    'error_rate': snapshot.error_rate,
                    'success_rate': snapshot.success_rate
                }
                
                await session.execute(
                    text("""
                        INSERT INTO progress_snapshots (
                            experiment_id, timestamp, stage, progress, details,
                            cpu_percent, memory_mb, api_calls_count, cost_incurred,
                            error_rate, success_rate
                        ) VALUES (
                            :experiment_id, :timestamp, :stage, :progress, :details,
                            :cpu_percent, :memory_mb, :api_calls_count, :cost_incurred,
                            :error_rate, :success_rate
                        )
                    """),
                    snapshot_data
                )
                
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                # Don't raise for progress snapshots - they're not critical
                self.logger.warning(f"Failed to save progress snapshot: {e}")
    
    async def get_experiment_data(
        self,
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get experiment data with optional filters
        
        Args:
            experiment_id: Experiment ID
            filters: Optional filters to apply
            
        Returns:
            List of experiment data records
        """
        async with self.async_session_factory() as session:
            try:
                base_query = """
                    SELECT r.*, sq.scrambling_method, sq.scrambling_intensity,
                           q.domain, q.question_type, q.difficulty_rating
                    FROM responses r
                    LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
                    LEFT JOIN questions q ON COALESCE(r.original_question_id, sq.original_question_id) = q.question_id
                    WHERE r.experiment_id = :experiment_id
                """
                
                params = {'experiment_id': experiment_id}
                
                # Add filters if provided
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if key == 'model_id':
                            conditions.append("r.model_id = :model_id")
                            params['model_id'] = value
                        elif key == 'scrambling_method':
                            conditions.append("sq.scrambling_method = :scrambling_method")
                            params['scrambling_method'] = value
                        elif key == 'domain':
                            conditions.append("q.domain = :domain")
                            params['domain'] = value
                    
                    if conditions:
                        base_query += " AND " + " AND ".join(conditions)
                
                result = await session.execute(text(base_query), params)
                
                # Convert to list of dictionaries
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to get experiment data: {e}")
                raise
    
    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        researcher: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering"""
        async with self.async_session_factory() as session:
            try:
                query = """
                    SELECT experiment_id, experiment_name, description, status,
                           researcher_name, created_at, started_at, completed_at
                    FROM experiments
                """
                
                conditions = []
                params = {}
                
                if status:
                    conditions.append("status = :status")
                    params['status'] = status.value
                
                if researcher:
                    conditions.append("researcher_name = :researcher")
                    params['researcher'] = researcher
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC LIMIT :limit"
                params['limit'] = limit
                
                result = await session.execute(text(query), params)
                
                columns = result.keys()
                experiments = []
                
                for row in result.fetchall():
                    experiment = dict(zip(columns, row))
                    # Convert datetime objects to ISO strings for JSON serialization
                    for date_field in ['created_at', 'started_at', 'completed_at']:
                        if experiment[date_field]:
                            experiment[date_field] = experiment[date_field].isoformat()
                    experiments.append(experiment)
                
                return experiments
                
            except Exception as e:
                self.logger.error(f"Failed to list experiments: {e}")
                raise
    
    async def save_experiment_error(
        self,
        experiment_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save experiment error information"""
        async with self.async_session_factory() as session:
            try:
                error_data = {
                    'experiment_id': experiment_id,
                    'error_message': error_message,
                    'error_details': error_details or {},
                    'occurred_at': datetime.now()
                }
                
                await session.execute(
                    text("""
                        INSERT INTO experiment_errors (
                            experiment_id, error_message, error_details, occurred_at
                        ) VALUES (
                            :experiment_id, :error_message, :error_details, :occurred_at
                        )
                    """),
                    error_data
                )
                
                await session.commit()
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to save experiment error: {e}")
    
    async def get_performance_summary(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Get performance summary for experiment"""
        async with self.async_session_factory() as session:
            try:
                # Get basic statistics
                result = await session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_responses,
                            COUNT(CASE WHEN is_correct THEN 1 END) as correct_responses,
                            AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy,
                            AVG(response_time_ms) as avg_response_time,
                            SUM(cost) as total_cost,
                            COUNT(DISTINCT model_id) as models_tested
                        FROM responses
                        WHERE experiment_id = :experiment_id
                    """),
                    {'experiment_id': experiment_id}
                )
                
                row = result.fetchone()
                
                if not row or row[0] == 0:
                    return {'error': 'No data found for experiment'}
                
                return {
                    'total_responses': row[0],
                    'correct_responses': row[1],
                    'accuracy': round(row[2], 4) if row[2] else 0.0,
                    'avg_response_time_ms': round(row[3], 2) if row[3] else 0.0,
                    'total_cost': round(row[4], 6) if row[4] else 0.0,
                    'models_tested': row[5]
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get performance summary: {e}")
                raise
    
    async def _ensure_model_exists(self, session: AsyncSession, model_id: str) -> None:
        """Ensure model exists in database, create if not"""
        try:
            # Check if model exists
            result = await session.execute(
                text("SELECT model_id FROM models WHERE model_id = :model_id"),
                {'model_id': model_id}
            )
            
            if not result.fetchone():
                # Create basic model entry
                await session.execute(
                    text("""
                        INSERT INTO models (model_id, model_name, model_family, model_version, access_type)
                        VALUES (:model_id, :model_name, :model_family, :model_version, :access_type)
                    """),
                    {
                        'model_id': model_id,
                        'model_name': model_id,
                        'model_family': model_id.split(':')[0] if ':' in model_id else 'unknown',
                        'model_version': model_id.split(':')[1] if ':' in model_id else '1.0',
                        'access_type': 'api'
                    }
                )
                
        except IntegrityError:
            # Model already exists (race condition)
            pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            async with self.async_session_factory() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database statistics
                stats_result = await session.execute(
                    text("""
                        SELECT 
                            (SELECT COUNT(*) FROM experiments) as total_experiments,
                            (SELECT COUNT(*) FROM experiments WHERE status = 'completed') as completed_experiments,
                            (SELECT COUNT(*) FROM experiments WHERE status = 'running') as running_experiments,
                            (SELECT COUNT(*) FROM responses) as total_responses
                    """)
                )
                
                stats = stats_result.fetchone()
                
                return {
                    'status': 'healthy',
                    'database_responsive': True,
                    'total_experiments': stats[0] if stats else 0,
                    'completed_experiments': stats[1] if stats else 0,
                    'running_experiments': stats[2] if stats else 0,
                    'total_responses': stats[3] if stats else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'database_responsive': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_old_data(
        self,
        retention_days: int = 90,
        dry_run: bool = True
    ) -> Dict[str, int]:
        """Clean up old experiment data"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        async with self.async_session_factory() as session:
            try:
                # Find old experiments
                result = await session.execute(
                    text("""
                        SELECT experiment_id FROM experiments 
                        WHERE completed_at < :cutoff_date 
                        AND status IN ('completed', 'failed', 'cancelled')
                    """),
                    {'cutoff_date': cutoff_date}
                )
                
                old_experiment_ids = [row[0] for row in result.fetchall()]
                
                if dry_run:
                    return {
                        'experiments_to_cleanup': len(old_experiment_ids),
                        'dry_run': True
                    }
                
                # Delete old data (CASCADE should handle related records)
                deleted_count = 0
                for exp_id in old_experiment_ids:
                    await session.execute(
                        text("DELETE FROM experiments WHERE experiment_id = :exp_id"),
                        {'exp_id': exp_id}
                    )
                    deleted_count += 1
                
                await session.commit()
                
                return {
                    'experiments_cleaned_up': deleted_count,
                    'dry_run': False
                }
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Failed to cleanup old data: {e}")
                raise
    
    async def close(self) -> None:
        """Close database connections"""
        await self.async_engine.dispose()
        self.logger.info("Database connections closed")