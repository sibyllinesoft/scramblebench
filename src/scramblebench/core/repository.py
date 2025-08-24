"""
Repository pattern implementation for ScrambleBench ORM models.

This module provides high-level repository classes that encapsulate
database operations and provide a clean interface for working with
ScrambleBench data through the SQLAlchemy ORM.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.orm import Session, Query

from .models import Run, Item, Evaluation, Aggregate, ParaphraseCache
from .db_config import get_session_sync, DatabaseSession


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy session (creates new one if None)
        """
        self._session = session
        self._owns_session = session is None
    
    @property
    def session(self) -> Session:
        """Get database session."""
        if self._session is None:
            self._session = get_session_sync()
        return self._session
    
    def close(self) -> None:
        """Close session if owned by this repository."""
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None


class RunRepository(BaseRepository):
    """Repository for Run model operations."""
    
    def create(self, run_data: Dict[str, Any]) -> Run:
        """
        Create a new run.
        
        Args:
            run_data: Dictionary with run data
            
        Returns:
            Created Run instance
        """
        run = Run(**run_data)
        self.session.add(run)
        self.session.commit()
        return run
    
    def get_by_id(self, run_id: str) -> Optional[Run]:
        """Get run by ID."""
        return self.session.query(Run).filter(Run.run_id == run_id).first()
    
    def get_all(self, limit: Optional[int] = None) -> List[Run]:
        """Get all runs, optionally limited."""
        query = self.session.query(Run).order_by(desc(Run.started_at))
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_active_runs(self) -> List[Run]:
        """Get all active (running) runs."""
        return self.session.query(Run).filter(Run.status == 'running').all()
    
    def get_completed_runs(self) -> List[Run]:
        """Get all completed runs."""
        return self.session.query(Run).filter(Run.status == 'completed').all()
    
    def update_status(self, run_id: str, status: str, completed_evaluations: Optional[int] = None) -> bool:
        """
        Update run status and progress.
        
        Args:
            run_id: Run ID
            status: New status
            completed_evaluations: Number of completed evaluations
            
        Returns:
            True if run was updated, False if not found
        """
        run = self.get_by_id(run_id)
        if run is None:
            return False
        
        run.status = status
        if completed_evaluations is not None:
            run.completed_evaluations = completed_evaluations
        if status == 'completed':
            run.completed_at = datetime.now()
        
        self.session.commit()
        return True
    
    def delete(self, run_id: str) -> bool:
        """
        Delete a run and all its related data.
        
        Args:
            run_id: Run ID to delete
            
        Returns:
            True if run was deleted, False if not found
        """
        run = self.get_by_id(run_id)
        if run is None:
            return False
        
        self.session.delete(run)
        self.session.commit()
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get run statistics."""
        total_runs = self.session.query(Run).count()
        completed_runs = self.session.query(Run).filter(Run.status == 'completed').count()
        active_runs = self.session.query(Run).filter(Run.status == 'running').count()
        failed_runs = self.session.query(Run).filter(Run.status == 'failed').count()
        
        return {
            'total_runs': total_runs,
            'completed_runs': completed_runs,
            'active_runs': active_runs,
            'failed_runs': failed_runs,
            'completion_rate': completed_runs / total_runs if total_runs > 0 else 0.0
        }


class EvaluationRepository(BaseRepository):
    """Repository for Evaluation model operations."""
    
    def create(self, evaluation_data: Dict[str, Any]) -> Evaluation:
        """Create a new evaluation."""
        evaluation = Evaluation(**evaluation_data)
        self.session.add(evaluation)
        self.session.commit()
        return evaluation
    
    def get_by_run_id(self, run_id: str, limit: Optional[int] = None) -> List[Evaluation]:
        """Get evaluations for a specific run."""
        query = self.session.query(Evaluation).filter(Evaluation.run_id == run_id)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_by_model(self, model_id: str, limit: Optional[int] = None) -> List[Evaluation]:
        """Get evaluations for a specific model."""
        query = self.session.query(Evaluation).filter(Evaluation.model_id == model_id)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_by_transform(self, transform: str, scramble_level: Optional[float] = None) -> List[Evaluation]:
        """Get evaluations for a specific transform type."""
        query = self.session.query(Evaluation).filter(Evaluation.transform == transform)
        if scramble_level is not None:
            query = query.filter(Evaluation.scramble_level == scramble_level)
        return query.all()
    
    def get_model_performance(self, model_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            run_id: Optional run ID to filter by
            
        Returns:
            Dictionary with performance metrics
        """
        query = self.session.query(Evaluation).filter(Evaluation.model_id == model_id)
        if run_id:
            query = query.filter(Evaluation.run_id == run_id)
        
        evaluations = query.all()
        if not evaluations:
            return {}
        
        correct_count = sum(1 for e in evaluations if e.is_correct)
        total_count = len(evaluations)
        
        return {
            'model_id': model_id,
            'total_evaluations': total_count,
            'correct_evaluations': correct_count,
            'accuracy': correct_count / total_count if total_count > 0 else 0.0,
            'avg_latency_ms': sum(e.latency_ms for e in evaluations if e.latency_ms) / total_count,
            'total_cost_usd': sum(e.cost_usd for e in evaluations if e.cost_usd),
            'total_tokens': sum(e.total_tokens for e in evaluations if e.total_tokens),
        }
    
    def get_transform_comparison(self, run_id: str, model_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across different transforms for a model in a run.
        
        Args:
            run_id: Run ID
            model_id: Model ID
            
        Returns:
            Dictionary with transform comparison data
        """
        evaluations = self.session.query(Evaluation).filter(
            and_(Evaluation.run_id == run_id, Evaluation.model_id == model_id)
        ).all()
        
        # Group by transform
        transform_groups = {}
        for eval in evaluations:
            key = eval.transform
            if eval.scramble_level is not None:
                key = f"{eval.transform}_{eval.scramble_level}"
            
            if key not in transform_groups:
                transform_groups[key] = []
            transform_groups[key].append(eval)
        
        # Calculate metrics for each group
        results = {}
        for transform, evals in transform_groups.items():
            correct = sum(1 for e in evals if e.is_correct)
            total = len(evals)
            
            results[transform] = {
                'total_evaluations': total,
                'correct_evaluations': correct,
                'accuracy': correct / total if total > 0 else 0.0,
                'avg_latency_ms': sum(e.latency_ms for e in evals if e.latency_ms) / total,
                'total_cost_usd': sum(e.cost_usd for e in evals if e.cost_usd),
            }
        
        return results


class AggregateRepository(BaseRepository):
    """Repository for Aggregate model operations."""
    
    def create(self, aggregate_data: Dict[str, Any]) -> Aggregate:
        """Create a new aggregate."""
        aggregate = Aggregate(**aggregate_data)
        self.session.add(aggregate)
        self.session.commit()
        return aggregate
    
    def get_by_run_id(self, run_id: str) -> List[Aggregate]:
        """Get aggregates for a specific run."""
        return self.session.query(Aggregate).filter(Aggregate.run_id == run_id).all()
    
    def get_model_summary(self, model_id: str) -> List[Aggregate]:
        """Get aggregate summary for a model across all runs."""
        return self.session.query(Aggregate).filter(Aggregate.model_id == model_id).all()
    
    def get_language_dependency_ranking(self, limit: int = 10) -> List[Aggregate]:
        """
        Get models ranked by language dependency (lowest LDC first).
        
        Args:
            limit: Number of results to return
            
        Returns:
            List of aggregates ordered by language dependency
        """
        return (
            self.session.query(Aggregate)
            .filter(Aggregate.ldc.isnot(None))
            .order_by(asc(Aggregate.ldc))
            .limit(limit)
            .all()
        )
    
    def get_performance_ranking(self, transform: str = 'original', limit: int = 10) -> List[Aggregate]:
        """
        Get models ranked by performance on a specific transform.
        
        Args:
            transform: Transform type to rank by
            limit: Number of results to return
            
        Returns:
            List of aggregates ordered by performance
        """
        return (
            self.session.query(Aggregate)
            .filter(Aggregate.transform == transform)
            .order_by(desc(Aggregate.acc_mean))
            .limit(limit)
            .all()
        )


class ItemRepository(BaseRepository):
    """Repository for Item model operations."""
    
    def create(self, item_data: Dict[str, Any]) -> Item:
        """Create a new item."""
        item = Item(**item_data)
        self.session.add(item)
        self.session.commit()
        return item
    
    def get_by_id(self, item_id: str) -> Optional[Item]:
        """Get item by ID."""
        return self.session.query(Item).filter(Item.item_id == item_id).first()
    
    def get_by_dataset(self, dataset: str, domain: Optional[str] = None) -> List[Item]:
        """Get items by dataset and optional domain."""
        query = self.session.query(Item).filter(Item.dataset == dataset)
        if domain:
            query = query.filter(Item.domain == domain)
        return query.all()
    
    def get_datasets(self) -> List[str]:
        """Get list of all datasets."""
        return [row[0] for row in self.session.query(Item.dataset).distinct().all()]
    
    def get_domains(self, dataset: Optional[str] = None) -> List[str]:
        """Get list of all domains, optionally filtered by dataset."""
        query = self.session.query(Item.domain).distinct()
        if dataset:
            query = query.filter(Item.dataset == dataset)
        return [row[0] for row in query.all() if row[0] is not None]


class ParaphraseCacheRepository(BaseRepository):
    """Repository for ParaphraseCache model operations."""
    
    def create(self, cache_data: Dict[str, Any]) -> ParaphraseCache:
        """Create a new paraphrase cache entry."""
        cache_entry = ParaphraseCache(**cache_data)
        self.session.add(cache_entry)
        self.session.commit()
        return cache_entry
    
    def get_by_item_id(self, item_id: str, provider: Optional[str] = None) -> List[ParaphraseCache]:
        """Get paraphrase candidates for an item."""
        query = self.session.query(ParaphraseCache).filter(ParaphraseCache.item_id == item_id)
        if provider:
            query = query.filter(ParaphraseCache.provider == provider)
        return query.all()
    
    def get_accepted_paraphrase(self, item_id: str, provider: str) -> Optional[ParaphraseCache]:
        """Get the accepted paraphrase for an item and provider."""
        return (
            self.session.query(ParaphraseCache)
            .filter(
                and_(
                    ParaphraseCache.item_id == item_id,
                    ParaphraseCache.provider == provider,
                    ParaphraseCache.accepted == True
                )
            )
            .first()
        )
    
    def get_coverage_stats(self, provider: str) -> Dict[str, Any]:
        """Get paraphrase coverage statistics for a provider."""
        total_items = self.session.query(ParaphraseCache.item_id).distinct().count()
        cached_items = (
            self.session.query(ParaphraseCache.item_id)
            .filter(ParaphraseCache.provider == provider)
            .distinct()
            .count()
        )
        accepted_items = (
            self.session.query(ParaphraseCache.item_id)
            .filter(
                and_(
                    ParaphraseCache.provider == provider,
                    ParaphraseCache.accepted == True
                )
            )
            .distinct()
            .count()
        )
        
        return {
            'provider': provider,
            'total_items': total_items,
            'cached_items': cached_items,
            'accepted_items': accepted_items,
            'cache_coverage': cached_items / total_items if total_items > 0 else 0.0,
            'acceptance_rate': accepted_items / cached_items if cached_items > 0 else 0.0,
        }


# Convenience class for managing all repositories
class RepositoryManager:
    """Manager class that provides access to all repositories."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize repository manager with optional shared session."""
        self._session = session
        self._runs = None
        self._evaluations = None
        self._aggregates = None
        self._items = None
        self._paraphrase_cache = None
    
    @property
    def runs(self) -> RunRepository:
        """Get runs repository."""
        if self._runs is None:
            self._runs = RunRepository(self._session)
        return self._runs
    
    @property
    def evaluations(self) -> EvaluationRepository:
        """Get evaluations repository."""
        if self._evaluations is None:
            self._evaluations = EvaluationRepository(self._session)
        return self._evaluations
    
    @property
    def aggregates(self) -> AggregateRepository:
        """Get aggregates repository."""
        if self._aggregates is None:
            self._aggregates = AggregateRepository(self._session)
        return self._aggregates
    
    @property
    def items(self) -> ItemRepository:
        """Get items repository."""
        if self._items is None:
            self._items = ItemRepository(self._session)
        return self._items
    
    @property
    def paraphrase_cache(self) -> ParaphraseCacheRepository:
        """Get paraphrase cache repository."""
        if self._paraphrase_cache is None:
            self._paraphrase_cache = ParaphraseCacheRepository(self._session)
        return self._paraphrase_cache
    
    def close_all(self) -> None:
        """Close all repositories."""
        for repo in [self._runs, self._evaluations, self._aggregates, self._items, self._paraphrase_cache]:
            if repo is not None:
                repo.close()


# Context manager for repository usage
def get_repositories() -> RepositoryManager:
    """
    Get repository manager with shared database session.
    
    Example:
        ```python
        repos = get_repositories()
        try:
            runs = repos.runs.get_all()
            for run in runs:
                evals = repos.evaluations.get_by_run_id(run.run_id)
        finally:
            repos.close_all()
        ```
    """
    return RepositoryManager()