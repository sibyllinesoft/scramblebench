"""
SQLAlchemy ORM models for ScrambleBench.

This module provides SQLAlchemy ORM models that correspond to the database schema
defined in TODO.md and implemented in database.py. These models provide a modern
ORM interface for interacting with the ScrambleBench database.

The models are designed to work alongside the existing DuckDB implementation,
providing optional ORM capabilities for applications that need them.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, 
    Column, 
    DateTime, 
    Float, 
    Integer, 
    String, 
    Text,
    ForeignKey,
    Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Run(Base):
    """
    ORM model for the runs table.
    
    Represents a complete evaluation run with configuration, progress tracking,
    and metadata storage.
    """
    
    __tablename__ = 'runs'
    
    run_id = Column(String, primary_key=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    git_sha = Column(String, nullable=True)
    config_yaml = Column(Text, nullable=False)
    config_hash = Column(String, nullable=False)
    seed = Column(Integer, nullable=False)
    status = Column(String, default='running')  # running, completed, failed, cancelled
    total_evaluations = Column(Integer, default=0)
    completed_evaluations = Column(Integer, default=0)
    meta_data = Column('metadata', Text, nullable=True)  # JSON metadata
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="run", cascade="all, delete-orphan")
    aggregates = relationship("Aggregate", back_populates="run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Run(run_id='{self.run_id}', status='{self.status}', progress={self.completed_evaluations}/{self.total_evaluations})>"
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_evaluations == 0:
            return 0.0
        return (self.completed_evaluations / self.total_evaluations) * 100.0
    
    @property
    def is_completed(self) -> bool:
        """Check if run is completed."""
        return self.status == 'completed'
    
    @property
    def duration(self) -> Optional[float]:
        """Get run duration in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class Item(Base):
    """
    ORM model for the items table.
    
    Represents individual benchmark items (questions/problems) that are
    evaluated across different models and transformations.
    """
    
    __tablename__ = 'items'
    
    item_id = Column(String, primary_key=True)
    dataset = Column(String, nullable=False)
    domain = Column(String, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    meta_data = Column('metadata', Text, nullable=True)  # JSON metadata
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="item")
    paraphrase_cache = relationship("ParaphraseCache", back_populates="item")
    
    def __repr__(self):
        return f"<Item(item_id='{self.item_id}', dataset='{self.dataset}', domain='{self.domain}')>"


class Evaluation(Base):
    """
    ORM model for the evals table.
    
    Represents individual model evaluations on specific items with complete
    performance metrics, tokenization analysis, and cost tracking.
    """
    
    __tablename__ = 'evals'
    
    eval_id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey('runs.run_id'), nullable=False)
    item_id = Column(String, ForeignKey('items.item_id'), nullable=False)
    model_id = Column(String, nullable=False)
    model_family = Column(String, nullable=True)
    n_params = Column(Float, nullable=True)  # Model parameters in billions
    provider = Column(String, nullable=False)  # ollama, openrouter, anthropic, etc.
    transform = Column(String, nullable=False)  # original, paraphrase, scramble
    scramble_level = Column(Float, nullable=True)  # 0.0-1.0 for scramble intensity
    prompt = Column(Text, nullable=True)
    completion = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    acc = Column(Float, nullable=True)  # Individual accuracy score (0.0 or 1.0)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    tok_kl = Column(Float, nullable=True)  # Token KL divergence
    tok_frag = Column(Float, nullable=True)  # Token fragmentation ratio
    latency_ms = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    seed = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    run = relationship("Run", back_populates="evaluations")
    item = relationship("Item", back_populates="evaluations")
    
    # Indexes
    __table_args__ = (
        Index('idx_evals_run_model', 'run_id', 'model_id'),
        Index('idx_evals_transform', 'transform', 'scramble_level'),
        Index('idx_evals_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Evaluation(eval_id='{self.eval_id}', model='{self.model_id}', transform='{self.transform}', correct={self.is_correct})>"
    
    @property
    def total_tokens(self) -> Optional[int]:
        """Get total token count."""
        if self.prompt_tokens and self.completion_tokens:
            return self.prompt_tokens + self.completion_tokens
        return None
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate generation speed in tokens per second."""
        if self.completion_tokens and self.latency_ms:
            return (self.completion_tokens / self.latency_ms) * 1000
        return None


class Aggregate(Base):
    """
    ORM model for the aggregates table.
    
    Stores precomputed aggregate metrics for efficient analysis and visualization.
    Includes reasoning robustness scores and language dependency coefficients.
    """
    
    __tablename__ = 'aggregates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Surrogate key
    run_id = Column(String, ForeignKey('runs.run_id'), nullable=False)
    model_id = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    domain = Column(String, nullable=True)
    transform = Column(String, nullable=False)
    scramble_level = Column(Float, nullable=True)
    acc_mean = Column(Float, nullable=False)  # Mean accuracy
    acc_ci_low = Column(Float, nullable=True)  # Confidence interval lower bound
    acc_ci_high = Column(Float, nullable=True)  # Confidence interval upper bound
    rrs = Column(Float, nullable=True)  # Reasoning Robustness Score
    ldc = Column(Float, nullable=True)  # Language Dependency Coefficient
    n_items = Column(Integer, nullable=False)  # Number of items in aggregate
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    run = relationship("Run", back_populates="aggregates")
    
    # Composite unique constraint (matches the TODO.md spec)
    __table_args__ = (
        Index('idx_aggregates_lookup', 'run_id', 'model_id', 'transform'),
        Index('idx_aggregates_model_performance', 'model_id', 'acc_mean'),
    )
    
    def __repr__(self):
        return f"<Aggregate(model='{self.model_id}', transform='{self.transform}', acc={self.acc_mean:.3f}, n={self.n_items})>"
    
    @property
    def performance_category(self) -> str:
        """Categorize performance level."""
        if self.acc_mean >= 0.9:
            return "excellent"
        elif self.acc_mean >= 0.7:
            return "good"
        elif self.acc_mean >= 0.5:
            return "fair"
        else:
            return "poor"
    
    @property
    def language_dependency_category(self) -> Optional[str]:
        """Categorize language dependency."""
        if self.ldc is None:
            return None
        
        if self.ldc <= 0.2:
            return "highly_robust"
        elif self.ldc <= 0.5:
            return "moderately_robust"
        elif self.ldc <= 0.8:
            return "somewhat_dependent"
        else:
            return "highly_dependent"


class ParaphraseCache(Base):
    """
    ORM model for the paraphrase_cache table.
    
    Caches paraphrased versions of benchmark items with quality metrics
    and validation scores for contamination detection.
    """
    
    __tablename__ = 'paraphrase_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Surrogate key
    item_id = Column(String, ForeignKey('items.item_id'), nullable=False)
    provider = Column(String, nullable=False)  # Provider used for paraphrasing
    candidate_id = Column(Integer, nullable=False)  # Candidate number (0, 1, 2, ...)
    text = Column(Text, nullable=False)  # Paraphrased text
    cos_sim = Column(Float, nullable=True)  # Semantic similarity score
    edit_ratio = Column(Float, nullable=True)  # Surface divergence ratio
    bleu_score = Column(Float, nullable=True)  # BLEU score with original
    accepted = Column(Boolean, default=False)  # Whether this candidate was accepted
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    item = relationship("Item", back_populates="paraphrase_cache")
    
    # Composite unique constraint and indexes
    __table_args__ = (
        Index('idx_paraphrase_lookup', 'item_id', 'accepted'),
        Index('idx_paraphrase_provider', 'provider', 'accepted'),
        Index('idx_paraphrase_quality', 'cos_sim', 'edit_ratio'),
    )
    
    def __repr__(self):
        return f"<ParaphraseCache(item='{self.item_id}', provider='{self.provider}', accepted={self.accepted})>"
    
    @property
    def quality_score(self) -> Optional[float]:
        """Calculate composite quality score."""
        if self.cos_sim and self.edit_ratio:
            # Balance semantic similarity with surface divergence
            return (self.cos_sim * 0.7) + ((1 - self.edit_ratio) * 0.3)
        return None
    
    @property
    def meets_quality_threshold(self) -> bool:
        """Check if paraphrase meets quality thresholds."""
        if self.cos_sim and self.edit_ratio:
            return self.cos_sim >= 0.85 and self.edit_ratio >= 0.25
        return False


# Additional utility classes for database management
class DatabaseStats(Base):
    """
    ORM model for tracking database statistics and metadata.
    """
    
    __tablename__ = 'database_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_name = Column(String, nullable=False, unique=True)
    stat_value = Column(String, nullable=True)  # JSON or string value
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<DatabaseStats(name='{self.stat_name}', value='{self.stat_value}')>"


class MigrationHistory(Base):
    """
    ORM model for tracking database migrations and schema changes.
    """
    
    __tablename__ = 'migration_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    migration_id = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    applied_at = Column(DateTime, default=func.now())
    schema_version = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<MigrationHistory(id='{self.migration_id}', applied_at='{self.applied_at}')>"