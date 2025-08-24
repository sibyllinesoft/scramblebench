"""
Simplified models import for Alembic migrations.
This avoids circular dependencies from the main package.
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import required modules for SQLAlchemy models
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
    """ORM model for the runs table."""
    
    __tablename__ = 'runs'
    
    run_id = Column(String, primary_key=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    git_sha = Column(String, nullable=True)
    config_yaml = Column(Text, nullable=False)
    config_hash = Column(String, nullable=False)
    seed = Column(Integer, nullable=False)
    status = Column(String, default='running')
    total_evaluations = Column(Integer, default=0)
    completed_evaluations = Column(Integer, default=0)
    meta_data = Column('metadata', Text, nullable=True)
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="run", cascade="all, delete-orphan")
    aggregates = relationship("Aggregate", back_populates="run", cascade="all, delete-orphan")


class Item(Base):
    """ORM model for the items table."""
    
    __tablename__ = 'items'
    
    item_id = Column(String, primary_key=True)
    dataset = Column(String, nullable=False)
    domain = Column(String, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    meta_data = Column('metadata', Text, nullable=True)
    
    # Relationships
    evaluations = relationship("Evaluation", back_populates="item")
    paraphrase_cache = relationship("ParaphraseCache", back_populates="item")


class Evaluation(Base):
    """ORM model for the evals table."""
    
    __tablename__ = 'evals'
    
    eval_id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey('runs.run_id'), nullable=False)
    item_id = Column(String, ForeignKey('items.item_id'), nullable=False)
    model_id = Column(String, nullable=False)
    model_family = Column(String, nullable=True)
    n_params = Column(Float, nullable=True)
    provider = Column(String, nullable=False)
    transform = Column(String, nullable=False)
    scramble_level = Column(Float, nullable=True)
    prompt = Column(Text, nullable=True)
    completion = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    acc = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    tok_kl = Column(Float, nullable=True)
    tok_frag = Column(Float, nullable=True)
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


class Aggregate(Base):
    """ORM model for the aggregates table."""
    
    __tablename__ = 'aggregates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey('runs.run_id'), nullable=False)
    model_id = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    domain = Column(String, nullable=True)
    transform = Column(String, nullable=False)
    scramble_level = Column(Float, nullable=True)
    acc_mean = Column(Float, nullable=False)
    acc_ci_low = Column(Float, nullable=True)
    acc_ci_high = Column(Float, nullable=True)
    rrs = Column(Float, nullable=True)
    ldc = Column(Float, nullable=True)
    n_items = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    run = relationship("Run", back_populates="aggregates")
    
    # Indexes
    __table_args__ = (
        Index('idx_aggregates_lookup', 'run_id', 'model_id', 'transform'),
        Index('idx_aggregates_model_performance', 'model_id', 'acc_mean'),
    )


class ParaphraseCache(Base):
    """ORM model for the paraphrase_cache table."""
    
    __tablename__ = 'paraphrase_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String, ForeignKey('items.item_id'), nullable=False)
    provider = Column(String, nullable=False)
    candidate_id = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    cos_sim = Column(Float, nullable=True)
    edit_ratio = Column(Float, nullable=True)
    bleu_score = Column(Float, nullable=True)
    accepted = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    item = relationship("Item", back_populates="paraphrase_cache")
    
    # Indexes
    __table_args__ = (
        Index('idx_paraphrase_lookup', 'item_id', 'accepted'),
        Index('idx_paraphrase_provider', 'provider', 'accepted'),
        Index('idx_paraphrase_quality', 'cos_sim', 'edit_ratio'),
    )


class DatabaseStats(Base):
    """ORM model for tracking database statistics and metadata."""
    
    __tablename__ = 'database_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stat_name = Column(String, nullable=False, unique=True)
    stat_value = Column(String, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class MigrationHistory(Base):
    """ORM model for tracking database migrations and schema changes."""
    
    __tablename__ = 'migration_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    migration_id = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    applied_at = Column(DateTime, default=func.now())
    schema_version = Column(String, nullable=True)