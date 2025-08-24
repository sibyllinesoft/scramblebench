"""
Repository pattern implementation for ScrambleBench.

Provides clean data access layer with repository classes for each major entity.
These repositories encapsulate database operations and provide a clean API
for the application layer to interact with the database.

Following the repository pattern helps:
- Separate business logic from data access logic
- Provide consistent interfaces for database operations
- Enable easier testing with mock repositories
- Centralize query logic and optimizations
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type, TypeVar, Generic
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, and_, or_, desc, asc

from .models import Base, Run, Item, Evaluation, Aggregate, ParaphraseCache
from .session import DatabaseManager

logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T], ABC):
    """
    Base repository class with common CRUD operations.
    
    Provides standard database operations that can be inherited by
    specific repository classes. Handles session management and
    error handling consistently.
    """
    
    def __init__(self, db_manager: DatabaseManager, model_class: Type[T]):
        """
        Initialize base repository.
        
        Args:
            db_manager: DatabaseManager instance for session management
            model_class: SQLAlchemy model class for this repository
        """
        self.db_manager = db_manager
        self.model_class = model_class
    
    def create(self, **kwargs: Any) -> T:
        """
        Create a new entity.
        
        Args:
            **kwargs: Entity attributes
            
        Returns:
            Created entity instance
        """
        with self.db_manager.session_scope() as session:
            entity = self.model_class(**kwargs)
            session.add(entity)
            session.flush()  # Get ID without committing
            session.refresh(entity)
            logger.debug(f"Created {self.model_class.__name__} with ID: {getattr(entity, 'id', 'N/A')}")
            return entity
    
    def get_by_id(self, entity_id: Any) -> Optional[T]:
        """
        Get entity by primary key.
        
        Args:
            entity_id: Primary key value
            
        Returns:
            Entity instance or None if not found
        """
        with self.db_manager.session_scope() as session:
            return session.query(self.model_class).get(entity_id)
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Get all entities with optional pagination.
        
        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            
        Returns:
            List of entities
        """
        with self.db_manager.session_scope() as session:
            query = session.query(self.model_class)
            if offset > 0:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
    
    def update(self, entity_id: Any, **kwargs: Any) -> Optional[T]:
        """
        Update entity by ID.
        
        Args:
            entity_id: Primary key value
            **kwargs: Attributes to update
            
        Returns:
            Updated entity instance or None if not found
        """
        with self.db_manager.session_scope() as session:
            entity = session.query(self.model_class).get(entity_id)
            if entity:
                for key, value in kwargs.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)
                session.flush()
                session.refresh(entity)
                logger.debug(f"Updated {self.model_class.__name__} with ID: {entity_id}")
            return entity
    
    def delete(self, entity_id: Any) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Primary key value
            
        Returns:
            True if deleted, False if not found
        """
        with self.db_manager.session_scope() as session:
            entity = session.query(self.model_class).get(entity_id)
            if entity:
                session.delete(entity)
                logger.debug(f"Deleted {self.model_class.__name__} with ID: {entity_id}")
                return True
            return False
    
    def count(self) -> int:
        """
        Get total count of entities.
        
        Returns:
            Total number of entities
        """
        with self.db_manager.session_scope() as session:
            return session.query(self.model_class).count()
    
    def exists(self, **filters: Any) -> bool:
        """
        Check if entity exists with given filters.
        
        Args:
            **filters: Filter conditions
            
        Returns:
            True if entity exists, False otherwise
        """
        with self.db_manager.session_scope() as session:
            query = session.query(self.model_class)
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            return query.first() is not None


class RunRepository(BaseRepository[Run]):
    """Repository for Run entities."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Run)
    
    def get_by_run_id(self, run_id: str) -> Optional[Run]:
        """Get run by run_id."""
        with self.db_manager.session_scope() as session:
            return session.query(Run).filter_by(run_id=run_id).first()
    
    def get_active_runs(self) -> List[Run]:
        """Get all currently active (running) runs."""
        with self.db_manager.session_scope() as session:
            return session.query(Run).filter_by(status='running').all()
    
    def get_completed_runs(self, limit: Optional[int] = None) -> List[Run]:
        """Get completed runs, most recent first."""
        with self.db_manager.session_scope() as session:
            query = session.query(Run).filter_by(status='completed').order_by(desc(Run.completed_at))
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def get_runs_by_config_hash(self, config_hash: str) -> List[Run]:
        """Get runs with the same configuration hash."""
        with self.db_manager.session_scope() as session:
            return session.query(Run).filter_by(config_hash=config_hash).all()
    
    def update_progress(self, run_id: str, completed_evaluations: int) -> Optional[Run]:
        """Update run progress."""
        return self.update(run_id, completed_evaluations=completed_evaluations)
    
    def mark_completed(self, run_id: str, completed_at: Optional[datetime] = None) -> Optional[Run]:
        """Mark run as completed."""
        if completed_at is None:
            completed_at = datetime.utcnow()
        return self.update(run_id, status='completed', completed_at=completed_at)
    
    def mark_failed(self, run_id: str, error_message: Optional[str] = None) -> Optional[Run]:
        """Mark run as failed."""
        update_data = {'status': 'failed'}
        if error_message:
            update_data['meta_data'] = error_message
        return self.update(run_id, **update_data)


class ItemRepository(BaseRepository[Item]):
    """Repository for Item entities."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Item)
    
    def get_by_item_id(self, item_id: str) -> Optional[Item]:
        """Get item by item_id."""
        with self.db_manager.session_scope() as session:
            return session.query(Item).filter_by(item_id=item_id).first()
    
    def get_by_dataset(self, dataset: str, domain: Optional[str] = None) -> List[Item]:
        """Get items by dataset and optional domain."""
        with self.db_manager.session_scope() as session:
            query = session.query(Item).filter_by(dataset=dataset)
            if domain:
                query = query.filter_by(domain=domain)
            return query.all()
    
    def get_datasets(self) -> List[str]:
        """Get list of unique datasets."""
        with self.db_manager.session_scope() as session:
            return [row[0] for row in session.query(Item.dataset).distinct().all()]
    
    def get_domains(self, dataset: Optional[str] = None) -> List[str]:
        """Get list of unique domains, optionally filtered by dataset."""
        with self.db_manager.session_scope() as session:
            query = session.query(Item.domain).distinct()
            if dataset:
                query = query.filter_by(dataset=dataset)
            return [row[0] for row in query.all() if row[0] is not None]
    
    def search_by_question(self, search_term: str, limit: int = 100) -> List[Item]:
        """Search items by question text."""
        with self.db_manager.session_scope() as session:
            return session.query(Item).filter(
                Item.question.like(f'%{search_term}%')
            ).limit(limit).all()


class EvaluationRepository(BaseRepository[Evaluation]):
    """Repository for Evaluation entities."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Evaluation)
    
    def get_by_eval_id(self, eval_id: str) -> Optional[Evaluation]:
        """Get evaluation by eval_id."""
        with self.db_manager.session_scope() as session:
            return session.query(Evaluation).filter_by(eval_id=eval_id).first()
    
    def get_by_run(self, run_id: str, limit: Optional[int] = None) -> List[Evaluation]:
        """Get evaluations for a specific run."""
        with self.db_manager.session_scope() as session:
            query = session.query(Evaluation).filter_by(run_id=run_id).order_by(Evaluation.timestamp)
            if limit:
                query = query.limit(limit)
            return query.all()
    
    def get_by_model(self, model_id: str, run_id: Optional[str] = None) -> List[Evaluation]:
        """Get evaluations for a specific model."""
        with self.db_manager.session_scope() as session:
            query = session.query(Evaluation).filter_by(model_id=model_id)
            if run_id:
                query = query.filter_by(run_id=run_id)
            return query.order_by(Evaluation.timestamp).all()
    
    def get_by_transform(self, transform: str, run_id: Optional[str] = None) -> List[Evaluation]:
        """Get evaluations for a specific transform."""
        with self.db_manager.session_scope() as session:
            query = session.query(Evaluation).filter_by(transform=transform)
            if run_id:
                query = query.filter_by(run_id=run_id)
            return query.order_by(Evaluation.timestamp).all()
    
    def get_correct_evaluations(self, run_id: Optional[str] = None) -> List[Evaluation]:
        """Get evaluations where is_correct is True."""
        with self.db_manager.session_scope() as session:
            query = session.query(Evaluation).filter_by(is_correct=True)
            if run_id:
                query = query.filter_by(run_id=run_id)
            return query.all()
    
    def get_evaluation_stats(self, run_id: str) -> Dict[str, Any]:
        """Get evaluation statistics for a run."""
        with self.db_manager.session_scope() as session:
            stats = session.query(
                func.count(Evaluation.eval_id).label('total_count'),
                func.count(Evaluation.is_correct).label('completed_count'),
                func.sum(Evaluation.is_correct.cast(int)).label('correct_count'),
                func.avg(Evaluation.latency_ms).label('avg_latency'),
                func.sum(Evaluation.cost_usd).label('total_cost')
            ).filter_by(run_id=run_id).first()
            
            return {
                'total_evaluations': stats.total_count or 0,
                'completed_evaluations': stats.completed_count or 0,
                'correct_evaluations': stats.correct_count or 0,
                'accuracy': (stats.correct_count / stats.completed_count) if stats.completed_count else 0,
                'avg_latency_ms': stats.avg_latency or 0,
                'total_cost_usd': stats.total_cost or 0
            }
    
    def get_model_performance(self, run_id: str) -> List[Dict[str, Any]]:
        """Get performance metrics grouped by model."""
        with self.db_manager.session_scope() as session:
            results = session.query(
                Evaluation.model_id,
                func.count(Evaluation.eval_id).label('total'),
                func.sum(Evaluation.is_correct.cast(int)).label('correct'),
                func.avg(Evaluation.latency_ms).label('avg_latency'),
                func.sum(Evaluation.cost_usd).label('total_cost')
            ).filter_by(run_id=run_id).group_by(Evaluation.model_id).all()
            
            return [
                {
                    'model_id': result.model_id,
                    'total_evaluations': result.total,
                    'correct_evaluations': result.correct or 0,
                    'accuracy': (result.correct / result.total) if result.total else 0,
                    'avg_latency_ms': result.avg_latency or 0,
                    'total_cost_usd': result.total_cost or 0
                }
                for result in results
            ]


class AggregateRepository(BaseRepository[Aggregate]):
    """Repository for Aggregate entities."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, Aggregate)
    
    def get_by_run(self, run_id: str) -> List[Aggregate]:
        """Get aggregates for a specific run."""
        with self.db_manager.session_scope() as session:
            return session.query(Aggregate).filter_by(run_id=run_id).all()
    
    def get_by_model_and_transform(
        self, 
        model_id: str, 
        transform: str, 
        run_id: Optional[str] = None
    ) -> List[Aggregate]:
        """Get aggregates for specific model and transform."""
        with self.db_manager.session_scope() as session:
            query = session.query(Aggregate).filter_by(
                model_id=model_id, 
                transform=transform
            )
            if run_id:
                query = query.filter_by(run_id=run_id)
            return query.all()
    
    def get_model_comparison(self, run_id: str) -> List[Dict[str, Any]]:
        """Get model comparison data for a run."""
        with self.db_manager.session_scope() as session:
            results = session.query(
                Aggregate.model_id,
                Aggregate.transform,
                func.avg(Aggregate.acc_mean).label('avg_accuracy'),
                func.avg(Aggregate.rrs).label('avg_rrs'),
                func.avg(Aggregate.ldc).label('avg_ldc')
            ).filter_by(run_id=run_id).group_by(
                Aggregate.model_id, 
                Aggregate.transform
            ).all()
            
            return [
                {
                    'model_id': result.model_id,
                    'transform': result.transform,
                    'avg_accuracy': result.avg_accuracy or 0,
                    'avg_rrs': result.avg_rrs,
                    'avg_ldc': result.avg_ldc
                }
                for result in results
            ]
    
    def get_best_performing_models(
        self, 
        run_id: str, 
        transform: str = 'original',
        limit: int = 10
    ) -> List[Aggregate]:
        """Get best performing models for a transform."""
        with self.db_manager.session_scope() as session:
            return session.query(Aggregate).filter_by(
                run_id=run_id,
                transform=transform
            ).order_by(desc(Aggregate.acc_mean)).limit(limit).all()


class ParaphraseCacheRepository(BaseRepository[ParaphraseCache]):
    """Repository for ParaphraseCache entities."""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(db_manager, ParaphraseCache)
    
    def get_by_item(self, item_id: str, accepted_only: bool = True) -> List[ParaphraseCache]:
        """Get paraphrases for a specific item."""
        with self.db_manager.session_scope() as session:
            query = session.query(ParaphraseCache).filter_by(item_id=item_id)
            if accepted_only:
                query = query.filter_by(accepted=True)
            return query.order_by(ParaphraseCache.candidate_id).all()
    
    def get_by_provider(self, provider: str, accepted_only: bool = True) -> List[ParaphraseCache]:
        """Get paraphrases by provider."""
        with self.db_manager.session_scope() as session:
            query = session.query(ParaphraseCache).filter_by(provider=provider)
            if accepted_only:
                query = query.filter_by(accepted=True)
            return query.all()
    
    def get_accepted_paraphrase(self, item_id: str) -> Optional[ParaphraseCache]:
        """Get the accepted paraphrase for an item."""
        with self.db_manager.session_scope() as session:
            return session.query(ParaphraseCache).filter_by(
                item_id=item_id,
                accepted=True
            ).first()
    
    def accept_paraphrase(self, paraphrase_id: int) -> Optional[ParaphraseCache]:
        """Mark a paraphrase as accepted."""
        return self.update(paraphrase_id, accepted=True)
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get paraphrase quality statistics."""
        with self.db_manager.session_scope() as session:
            stats = session.query(
                func.count(ParaphraseCache.id).label('total'),
                func.count(ParaphraseCache.accepted).label('evaluated'),
                func.sum(ParaphraseCache.accepted.cast(int)).label('accepted'),
                func.avg(ParaphraseCache.cos_sim).label('avg_cos_sim'),
                func.avg(ParaphraseCache.edit_ratio).label('avg_edit_ratio'),
                func.avg(ParaphraseCache.bleu_score).label('avg_bleu_score')
            ).first()
            
            return {
                'total_paraphrases': stats.total or 0,
                'evaluated_paraphrases': stats.evaluated or 0,
                'accepted_paraphrases': stats.accepted or 0,
                'acceptance_rate': (stats.accepted / stats.evaluated) if stats.evaluated else 0,
                'avg_cos_sim': stats.avg_cos_sim or 0,
                'avg_edit_ratio': stats.avg_edit_ratio or 0,
                'avg_bleu_score': stats.avg_bleu_score or 0
            }
    
    def find_high_quality_candidates(
        self, 
        min_cos_sim: float = 0.85,
        min_edit_ratio: float = 0.25
    ) -> List[ParaphraseCache]:
        """Find high-quality paraphrase candidates."""
        with self.db_manager.session_scope() as session:
            return session.query(ParaphraseCache).filter(
                and_(
                    ParaphraseCache.cos_sim >= min_cos_sim,
                    ParaphraseCache.edit_ratio >= min_edit_ratio,
                    ParaphraseCache.accepted == False
                )
            ).order_by(desc(ParaphraseCache.cos_sim)).all()


# Repository factory for easy access
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._repositories: Dict[str, BaseRepository] = {}
    
    def get_run_repository(self) -> RunRepository:
        """Get or create RunRepository instance."""
        if 'run' not in self._repositories:
            self._repositories['run'] = RunRepository(self.db_manager)
        return self._repositories['run']
    
    def get_item_repository(self) -> ItemRepository:
        """Get or create ItemRepository instance."""
        if 'item' not in self._repositories:
            self._repositories['item'] = ItemRepository(self.db_manager)
        return self._repositories['item']
    
    def get_evaluation_repository(self) -> EvaluationRepository:
        """Get or create EvaluationRepository instance."""
        if 'evaluation' not in self._repositories:
            self._repositories['evaluation'] = EvaluationRepository(self.db_manager)
        return self._repositories['evaluation']
    
    def get_aggregate_repository(self) -> AggregateRepository:
        """Get or create AggregateRepository instance."""
        if 'aggregate' not in self._repositories:
            self._repositories['aggregate'] = AggregateRepository(self.db_manager)
        return self._repositories['aggregate']
    
    def get_paraphrase_cache_repository(self) -> ParaphraseCacheRepository:
        """Get or create ParaphraseCacheRepository instance."""
        if 'paraphrase_cache' not in self._repositories:
            self._repositories['paraphrase_cache'] = ParaphraseCacheRepository(self.db_manager)
        return self._repositories['paraphrase_cache']