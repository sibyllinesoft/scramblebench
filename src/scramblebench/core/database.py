"""
Database operations for ScrambleBench.

This module implements the DuckDB schema and operations specified in TODO.md,
with support for runs, items, evals, aggregates, and paraphrase cache tables.
"""

import duckdb
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class Database:
    """DuckDB database operations for ScrambleBench."""
    
    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure database file exists and has correct schema
        self._connection = None
        self._ensure_schema()
    
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection, creating if necessary."""
        if self._connection is None:
            self._connection = duckdb.connect(str(self.db_path))
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def _ensure_schema(self):
        """Ensure database schema matches TODO.md specification."""
        conn = self.get_connection()
        
        # Create tables following the exact schema from TODO.md
        
        # Runs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                git_sha TEXT,
                config_yaml TEXT,
                config_hash TEXT,
                seed INTEGER,
                status TEXT DEFAULT 'running',
                total_evaluations INTEGER,
                completed_evaluations INTEGER DEFAULT 0,
                metadata TEXT  -- JSON metadata
            )
        """)
        
        # Items table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                domain TEXT,
                question TEXT NOT NULL,
                answer TEXT,
                metadata TEXT  -- JSON metadata
            )
        """)
        
        # Evals table - core evaluation results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evals (
                eval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_family TEXT,
                n_params FLOAT,
                provider TEXT,
                transform TEXT NOT NULL,
                scramble_level FLOAT,
                prompt TEXT,
                completion TEXT,
                is_correct BOOLEAN,
                acc FLOAT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                tok_kl FLOAT,  -- Token KL divergence
                tok_frag FLOAT,  -- Token fragmentation ratio
                latency_ms INTEGER,
                cost_usd FLOAT,
                seed INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                FOREIGN KEY (item_id) REFERENCES items(item_id)
            )
        """)
        
        # Aggregates table - precomputed metrics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aggregates (
                run_id TEXT,
                model_id TEXT,
                dataset TEXT,
                domain TEXT,
                transform TEXT,
                scramble_level FLOAT,
                acc_mean FLOAT,
                acc_ci_low FLOAT,
                acc_ci_high FLOAT,
                RRS FLOAT,  -- Reasoning Robustness Score
                LDC FLOAT,  -- Language Dependency Coefficient
                n_items INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, model_id, dataset, domain, transform, scramble_level),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Paraphrase cache table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paraphrase_cache (
                item_id TEXT,
                provider TEXT,
                candidate_id INTEGER,
                text TEXT,
                cos_sim FLOAT,  -- Semantic similarity score
                edit_ratio FLOAT,  -- Surface divergence ratio
                bleu_score FLOAT,  -- BLEU score with original
                accepted BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, provider, candidate_id),
                FOREIGN KEY (item_id) REFERENCES items(item_id)
            )
        """)
        
        # Create useful indices
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evals_run_model ON evals(run_id, model_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evals_transform ON evals(transform, scramble_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_aggregates_lookup ON aggregates(run_id, model_id, transform)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paraphrase_lookup ON paraphrase_cache(item_id, accepted)")
        
        logger.info(f"Database schema initialized at {self.db_path}")
    
    # ========================================================================
    # Run Management
    # ========================================================================
    
    def create_run(self, run_id: str, config_yaml: str, config_hash: str, 
                   git_sha: str, seed: int, total_evaluations: int = 0) -> None:
        """Create a new run record."""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT INTO runs (
                run_id, started_at, git_sha, config_yaml, config_hash, 
                seed, total_evaluations, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
        """, (
            run_id, datetime.now(), git_sha, config_yaml, 
            config_hash, seed, total_evaluations
        ))
        
        logger.info(f"Created run: {run_id}")
    
    def update_run_status(self, run_id: str, status: str, completed_evaluations: int = None) -> None:
        """Update run status and progress."""
        conn = self.get_connection()
        
        if status == 'completed':
            conn.execute("""
                UPDATE runs 
                SET status = ?, completed_at = ?, completed_evaluations = ?
                WHERE run_id = ?
            """, (status, datetime.now(), completed_evaluations, run_id))
        else:
            if completed_evaluations is not None:
                conn.execute("""
                    UPDATE runs 
                    SET status = ?, completed_evaluations = ?
                    WHERE run_id = ?
                """, (status, completed_evaluations, run_id))
            else:
                conn.execute("""
                    UPDATE runs SET status = ? WHERE run_id = ?
                """, (status, run_id))
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed run status."""
        conn = self.get_connection()
        
        result = conn.execute("""
            SELECT run_id, status, started_at, completed_at, 
                   total_evaluations, completed_evaluations,
                   config_hash, seed
            FROM runs WHERE run_id = ?
        """, (run_id,)).fetchone()
        
        if not result:
            return None
        
        columns = ['run_id', 'status', 'started_at', 'completed_at',
                  'total_evaluations', 'completed_evaluations', 
                  'config_hash', 'seed']
        
        return dict(zip(columns, result))
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with status."""
        conn = self.get_connection()
        
        results = conn.execute("""
            SELECT run_id, status, started_at, completed_at,
                   total_evaluations, completed_evaluations
            FROM runs
            ORDER BY started_at DESC
        """).fetchall()
        
        columns = ['run_id', 'status', 'started_at', 'completed_at',
                  'total_evaluations', 'completed_evaluations']
        
        return [dict(zip(columns, row)) for row in results]
    
    # ========================================================================
    # Item Management
    # ========================================================================
    
    def upsert_item(self, item_id: str, dataset: str, domain: str, 
                    question: str, answer: str, metadata: Dict[str, Any] = None) -> None:
        """Insert or update an item."""
        conn = self.get_connection()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        conn.execute("""
            INSERT INTO items (item_id, dataset, domain, question, answer, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (item_id) DO UPDATE SET
                dataset = EXCLUDED.dataset,
                domain = EXCLUDED.domain,
                question = EXCLUDED.question,
                answer = EXCLUDED.answer,
                metadata = EXCLUDED.metadata
        """, (item_id, dataset, domain, question, answer, metadata_json))
    
    def get_items_for_dataset(self, dataset: str, domain: str = None) -> List[Dict[str, Any]]:
        """Get all items for a dataset, optionally filtered by domain."""
        conn = self.get_connection()
        
        if domain:
            results = conn.execute("""
                SELECT item_id, dataset, domain, question, answer, metadata
                FROM items
                WHERE dataset = ? AND domain = ?
            """, (dataset, domain)).fetchall()
        else:
            results = conn.execute("""
                SELECT item_id, dataset, domain, question, answer, metadata
                FROM items
                WHERE dataset = ?
            """, (dataset,)).fetchall()
        
        columns = ['item_id', 'dataset', 'domain', 'question', 'answer', 'metadata']
        
        items = []
        for row in results:
            item = dict(zip(columns, row))
            if item['metadata']:
                item['metadata'] = json.loads(item['metadata'])
            items.append(item)
        
        return items
    
    # ========================================================================
    # Evaluation Results
    # ========================================================================
    
    def insert_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Insert an evaluation result."""
        conn = self.get_connection()
        
        eval_id = eval_data.get('eval_id') or str(uuid.uuid4())
        
        conn.execute("""
            INSERT INTO evals (
                eval_id, run_id, item_id, model_id, model_family, n_params,
                provider, transform, scramble_level, prompt, completion,
                is_correct, acc, prompt_tokens, completion_tokens,
                tok_kl, tok_frag, latency_ms, cost_usd, seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval_id, eval_data['run_id'], eval_data['item_id'],
            eval_data['model_id'], eval_data.get('model_family'),
            eval_data.get('n_params'), eval_data['provider'],
            eval_data['transform'], eval_data.get('scramble_level'),
            eval_data['prompt'], eval_data['completion'],
            eval_data['is_correct'], eval_data['acc'],
            eval_data.get('prompt_tokens'), eval_data.get('completion_tokens'),
            eval_data.get('tok_kl'), eval_data.get('tok_frag'),
            eval_data.get('latency_ms'), eval_data.get('cost_usd'),
            eval_data['seed']
        ))
    
    def get_evaluations(self, run_id: str, model_id: str = None, 
                       transform: str = None) -> List[Dict[str, Any]]:
        """Get evaluation results with optional filtering."""
        conn = self.get_connection()
        
        query = "SELECT * FROM evals WHERE run_id = ?"
        params = [run_id]
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if transform:
            query += " AND transform = ?"
            params.append(transform)
        
        results = conn.execute(query, params).fetchall()
        
        # Get column names
        columns = [desc[0] for desc in conn.description]
        
        return [dict(zip(columns, row)) for row in results]
    
    # ========================================================================
    # Aggregated Metrics
    # ========================================================================
    
    def upsert_aggregate(self, agg_data: Dict[str, Any]) -> None:
        """Insert or update aggregate metrics."""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT INTO aggregates (
                run_id, model_id, dataset, domain, transform, scramble_level,
                acc_mean, acc_ci_low, acc_ci_high, RRS, LDC, n_items
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, model_id, dataset, domain, transform, scramble_level)
            DO UPDATE SET
                acc_mean = EXCLUDED.acc_mean,
                acc_ci_low = EXCLUDED.acc_ci_low,
                acc_ci_high = EXCLUDED.acc_ci_high,
                RRS = EXCLUDED.RRS,
                LDC = EXCLUDED.LDC,
                n_items = EXCLUDED.n_items,
                timestamp = CURRENT_TIMESTAMP
        """, (
            agg_data['run_id'], agg_data['model_id'], agg_data['dataset'],
            agg_data['domain'], agg_data['transform'], agg_data.get('scramble_level'),
            agg_data['acc_mean'], agg_data.get('acc_ci_low'), agg_data.get('acc_ci_high'),
            agg_data.get('RRS'), agg_data.get('LDC'), agg_data['n_items']
        ))
    
    def get_aggregates(self, run_id: str, model_id: str = None) -> List[Dict[str, Any]]:
        """Get aggregate results."""
        conn = self.get_connection()
        
        if model_id:
            results = conn.execute("""
                SELECT * FROM aggregates 
                WHERE run_id = ? AND model_id = ?
                ORDER BY dataset, domain, transform, scramble_level
            """, (run_id, model_id)).fetchall()
        else:
            results = conn.execute("""
                SELECT * FROM aggregates 
                WHERE run_id = ?
                ORDER BY model_id, dataset, domain, transform, scramble_level
            """, (run_id,)).fetchall()
        
        columns = [desc[0] for desc in conn.description]
        return [dict(zip(columns, row)) for row in results]
    
    # ========================================================================
    # Paraphrase Cache
    # ========================================================================
    
    def cache_paraphrase(self, item_id: str, provider: str, candidate_id: int,
                        text: str, cos_sim: float, edit_ratio: float,
                        bleu_score: float, accepted: bool) -> None:
        """Cache a paraphrase candidate."""
        conn = self.get_connection()
        
        conn.execute("""
            INSERT INTO paraphrase_cache (
                item_id, provider, candidate_id, text, cos_sim,
                edit_ratio, bleu_score, accepted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (item_id, provider, candidate_id) DO UPDATE SET
                text = EXCLUDED.text,
                cos_sim = EXCLUDED.cos_sim,
                edit_ratio = EXCLUDED.edit_ratio,
                bleu_score = EXCLUDED.bleu_score,
                accepted = EXCLUDED.accepted,
                timestamp = CURRENT_TIMESTAMP
        """, (item_id, provider, candidate_id, text, cos_sim, edit_ratio, bleu_score, accepted))
    
    def get_cached_paraphrase(self, item_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get cached paraphrase for an item."""
        conn = self.get_connection()
        
        result = conn.execute("""
            SELECT item_id, provider, candidate_id, text, cos_sim,
                   edit_ratio, bleu_score, accepted
            FROM paraphrase_cache
            WHERE item_id = ? AND provider = ? AND accepted = true
            ORDER BY cos_sim DESC
            LIMIT 1
        """, (item_id, provider)).fetchone()
        
        if not result:
            return None
        
        columns = ['item_id', 'provider', 'candidate_id', 'text', 'cos_sim',
                  'edit_ratio', 'bleu_score', 'accepted']
        return dict(zip(columns, result))
    
    def get_paraphrase_coverage(self, provider: str) -> Dict[str, int]:
        """Get paraphrase coverage statistics."""
        conn = self.get_connection()
        
        total_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        
        cached_items = conn.execute("""
            SELECT COUNT(DISTINCT item_id) 
            FROM paraphrase_cache 
            WHERE provider = ?
        """, (provider,)).fetchone()[0]
        
        accepted_items = conn.execute("""
            SELECT COUNT(DISTINCT item_id)
            FROM paraphrase_cache
            WHERE provider = ? AND accepted = true
        """, (provider,)).fetchone()[0]
        
        return {
            'total_items': total_items,
            'cached_items': cached_items,
            'accepted_items': accepted_items,
            'acceptance_rate': accepted_items / cached_items if cached_items > 0 else 0.0,
            'coverage_rate': accepted_items / total_items if total_items > 0 else 0.0
        }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def compute_canonical_metrics(self, run_id: str, model_id: str, dataset: str, domain: str) -> Dict[str, float]:
        """Compute canonical metrics (RRS, LDC) for a model on a dataset/domain."""
        conn = self.get_connection()
        
        # Get accuracy for original (baseline)
        acc0_result = conn.execute("""
            SELECT AVG(CAST(is_correct AS FLOAT)) as acc0
            FROM evals
            WHERE run_id = ? AND model_id = ? AND dataset = ? AND domain = ?
              AND transform = 'original'
        """, (run_id, model_id, dataset, domain)).fetchone()
        
        if not acc0_result or acc0_result[0] is None:
            return {'acc0': 0.0, 'RRS': 0.0, 'LDC': 0.0}
        
        acc0 = acc0_result[0]
        
        # Get accuracy for other transforms
        other_accs = conn.execute("""
            SELECT transform, scramble_level, AVG(CAST(is_correct AS FLOAT)) as acc
            FROM evals
            WHERE run_id = ? AND model_id = ? AND dataset = ? AND domain = ?
              AND transform != 'original'
            GROUP BY transform, scramble_level
        """, (run_id, model_id, dataset, domain)).fetchall()
        
        results = {'acc0': acc0}
        
        for transform, scramble_level, acc in other_accs:
            # Compute RRS = Acc_scram / Acc0
            rrs = acc / acc0 if acc0 > 0 else 0.0
            
            # Compute LDC = 1 - RRS
            ldc = 1.0 - rrs
            
            key_suffix = f"_{scramble_level}" if scramble_level is not None else ""
            results[f'acc_{transform}{key_suffix}'] = acc
            results[f'RRS_{transform}{key_suffix}'] = rrs
            results[f'LDC_{transform}{key_suffix}'] = ldc
        
        return results
    
    def backup_database(self, backup_path: Path) -> None:
        """Create a backup of the database."""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self.get_connection()
        
        stats = {}
        
        # Table row counts
        for table in ['runs', 'items', 'evals', 'aggregates', 'paraphrase_cache']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[f'{table}_count'] = count
        
        # Database size
        stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
        
        # Latest run info
        latest_run = conn.execute("""
            SELECT run_id, status, started_at 
            FROM runs 
            ORDER BY started_at DESC 
            LIMIT 1
        """).fetchone()
        
        if latest_run:
            stats['latest_run'] = {
                'run_id': latest_run[0],
                'status': latest_run[1],
                'started_at': latest_run[2]
            }
        
        return stats