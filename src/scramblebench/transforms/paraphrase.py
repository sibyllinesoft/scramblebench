"""
Paraphrase transform strategy with semantic equivalence and surface divergence checks.

Implements paraphrase generation with safety checks to ensure semantic equivalence
while maintaining surface-level differences to test contamination vs. brittleness.
"""

import asyncio
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import editdistance

from .base import BaseTransform, TransformResult
from ..core.database import Database

logger = logging.getLogger(__name__)


class ParaphraseValidator:
    """Validates paraphrases for semantic equivalence and surface divergence."""
    
    def __init__(self):
        """Initialize validator with semantic similarity model."""
        try:
            # Use a lightweight but effective sentence embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            # Fallback to simpler validation if sentence-transformers not available
            self.embedding_model = None
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def validate_paraphrase(self, original: str, paraphrase: str, 
                          semantic_threshold: float = 0.85,
                          surface_threshold: float = 0.25) -> Dict[str, Any]:
        """Validate paraphrase for semantic equivalence and surface divergence."""
        
        # Compute semantic similarity
        semantic_score = self._compute_semantic_similarity(original, paraphrase)
        semantic_valid = semantic_score >= semantic_threshold
        
        # Compute surface divergence
        edit_ratio = self._compute_edit_distance_ratio(original, paraphrase)
        bleu_score = self._compute_bleu_score(original, paraphrase)
        
        # Surface divergence: either high edit distance OR low BLEU
        # Use explicit BLEU threshold (≤0.6) as specified in TODO.md
        surface_divergent = edit_ratio >= surface_threshold or bleu_score <= 0.6
        
        return {
            'semantic_score': semantic_score,
            'semantic_valid': semantic_valid,
            'edit_ratio': edit_ratio,
            'bleu_score': bleu_score,
            'surface_divergent': surface_divergent,
            'overall_valid': semantic_valid and surface_divergent,
            'thresholds': {
                'semantic_threshold': semantic_threshold,
                'surface_threshold': surface_threshold
            }
        }
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if self.embedding_model is None:
            # Fallback: simple token overlap similarity
            return self._compute_token_overlap_similarity(text1, text2)
        
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception:
            # Fallback if embedding computation fails
            return self._compute_token_overlap_similarity(text1, text2)
    
    def _compute_token_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback semantic similarity based on token overlap."""
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))
        
        if not tokens1 and not tokens2:
            return 1.0
        elif not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_edit_distance_ratio(self, text1: str, text2: str) -> float:
        """Compute edit distance ratio (normalized by length)."""
        distance = editdistance.eval(text1, text2)
        max_length = max(len(text1), len(text2))
        
        return distance / max_length if max_length > 0 else 0.0
    
    def _compute_bleu_score(self, reference: str, candidate: str) -> float:
        """Compute BLEU score between reference and candidate."""
        try:
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # Use smoothing to handle short sentences
            bleu = sentence_bleu([ref_tokens], cand_tokens, 
                               smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
            return bleu
        except Exception:
            return 0.0


class ParaphraseTransform(BaseTransform):
    """Paraphrase transform with semantic equivalence validation and database integration."""
    
    def _initialize(self):
        """Initialize paraphrase transform components."""
        self.provider = self.config.get('provider', 'hosted_heldout')
        self.n_candidates = self.config.get('n_candidates', 2)
        self.semantic_threshold = self.config.get('semantic_sim_threshold', 0.85)
        self.surface_threshold = self.config.get('surface_divergence_min', 0.25)
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/paraphrase'))
        self.temperature = self.config.get('temperature', 0.3)  # Only for generation
        self.bleu_threshold = self.config.get('bleu_threshold', 0.6)  # BLEU ≤ 0.6 for surface divergence
        
        # Database integration (optional - can fallback to file cache)
        self.database = self.config.get('database')  # Database instance
        self.use_database_cache = self.database is not None
        
        # Initialize validator
        self.validator = ParaphraseValidator()
        
        # Ensure cache directory exists for file-based fallback
        if not self.use_database_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model adapter for paraphrase generation
        self._model_adapter = None
        
        logger.info(f"ParaphraseTransform initialized: provider={self.provider}, "
                   f"database_cache={self.use_database_cache}")
    
    def set_model_adapter(self, adapter):
        """Set the model adapter for paraphrase generation."""
        self._model_adapter = adapter
    
    async def transform_async(self, text: str, item_id: str = None, **kwargs) -> TransformResult:
        """Apply paraphrase transformation asynchronously."""
        
        # Generate item ID if not provided
        if item_id is None:
            item_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Check cache first
        cached_paraphrase = self._get_cached_paraphrase(item_id)
        if cached_paraphrase:
            return TransformResult(
                original_text=text,
                transformed_text=cached_paraphrase['text'],
                transform_type="paraphrase",
                transform_metadata={
                    "provider": self.provider,
                    "semantic_score": cached_paraphrase['semantic_score'],
                    "surface_divergent": cached_paraphrase['surface_divergent'],
                    "cached": True,
                    "item_id": item_id
                },
                success=True
            )
        
        # Generate paraphrase candidates
        candidates = await self._generate_paraphrase_candidates(text)
        
        if not candidates:
            return TransformResult(
                original_text=text,
                transformed_text=text,  # Fallback to original
                transform_type="paraphrase",
                transform_metadata={"error": "No candidates generated"},
                success=False,
                error_message="Failed to generate paraphrase candidates"
            )
        
        # Validate candidates and select best one
        best_candidate = self._select_best_candidate(text, candidates)
        
        if best_candidate is None:
            return TransformResult(
                original_text=text,
                transformed_text=text,  # Fallback to original
                transform_type="paraphrase",
                transform_metadata={
                    "error": "No valid candidates",
                    "candidates_generated": len(candidates)
                },
                success=False,
                error_message="No paraphrase candidates passed validation"
            )
        
        # Cache the successful paraphrase
        self._cache_paraphrase(item_id, text, best_candidate)
        
        return TransformResult(
            original_text=text,
            transformed_text=best_candidate['text'],
            transform_type="paraphrase",
            transform_metadata={
                "provider": self.provider,
                "semantic_score": best_candidate['validation']['semantic_score'],
                "surface_divergent": best_candidate['validation']['surface_divergent'],
                "edit_ratio": best_candidate['validation']['edit_ratio'],
                "bleu_score": best_candidate['validation']['bleu_score'],
                "cached": False,
                "item_id": item_id,
                "candidates_generated": len(candidates)
            },
            success=True
        )
    
    def transform(self, text: str, item_id: str = None, **kwargs) -> TransformResult:
        """Synchronous wrapper for async transform."""
        try:
            return asyncio.run(self.transform_async(text, item_id, **kwargs))
        except Exception as e:
            return TransformResult(
                original_text=text,
                transformed_text=text,
                transform_type="paraphrase",
                transform_metadata={"error": str(e)},
                success=False,
                error_message=f"Paraphrase transform failed: {e}"
            )
    
    async def _generate_paraphrase_candidates(self, text: str) -> List[Dict[str, Any]]:
        """Generate paraphrase candidates using the model adapter."""
        if self._model_adapter is None:
            raise ValueError("Model adapter not set. Use set_model_adapter() first.")
        
        candidates = []
        
        # Create paraphrase prompt
        prompt = self._create_paraphrase_prompt(text)
        
        for i in range(self.n_candidates):
            try:
                result = await self._model_adapter.generate(
                    prompt=prompt,
                    temperature=self.temperature,  # Non-zero for generation diversity
                    max_tokens=len(text) * 2,  # Allow longer paraphrases
                    seed=self.seed + i  # Different seed per candidate
                )
                
                if result.success and result.text.strip():
                    paraphrase_text = self._extract_paraphrase_from_response(result.text)
                    
                    if paraphrase_text and paraphrase_text.strip() != text.strip():
                        candidates.append({
                            'text': paraphrase_text,
                            'generation_metadata': result.metadata,
                            'candidate_id': i
                        })
            
            except Exception as e:
                print(f"Failed to generate paraphrase candidate {i}: {e}")
                continue
        
        return candidates
    
    def _create_paraphrase_prompt(self, text: str) -> str:
        """Create prompt for paraphrase generation."""
        return f"""Please rewrite the following text to have the same meaning but use different words and sentence structure. Keep the core meaning identical.

Original: {text}

Rewritten:"""
    
    def _extract_paraphrase_from_response(self, response: str) -> Optional[str]:
        """Extract paraphrase text from model response."""
        # Simple extraction - take the first non-empty line after cleaning
        lines = response.strip().split('\n')
        
        for line in lines:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith(('Original:', 'Rewritten:', 'Please')):
                return cleaned
        
        return None
    
    def _select_best_candidate(self, original: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best paraphrase candidate based on validation criteria."""
        validated_candidates = []
        
        for candidate in candidates:
            validation = self.validator.validate_paraphrase(
                original, candidate['text'],
                semantic_threshold=self.semantic_threshold,
                surface_threshold=self.surface_threshold
            )
            
            candidate['validation'] = validation
            
            if validation['overall_valid']:
                validated_candidates.append(candidate)
        
        if not validated_candidates:
            return None
        
        # Select candidate with highest semantic score among valid ones
        best_candidate = max(validated_candidates, 
                           key=lambda c: c['validation']['semantic_score'])
        
        return best_candidate
    
    def _get_cached_paraphrase(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get cached paraphrase for item using database or file cache."""
        if self.use_database_cache:
            try:
                cached_data = self.database.get_cached_paraphrase(item_id, self.provider)
                if cached_data and cached_data.get('accepted', False):
                    return {
                        'text': cached_data['text'],
                        'semantic_score': cached_data['cos_sim'],
                        'surface_divergent': True,  # Cached items were already validated
                        'accepted': True
                    }
            except Exception as e:
                logger.warning(f"Database cache lookup failed for {item_id}: {e}")
        
        # Fallback to file cache
        cache_file = self.cache_dir / f"{item_id}_{self.provider}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # Validate cache format and acceptance
                if cached_data.get('accepted', False):
                    return cached_data
            except Exception as e:
                logger.warning(f"File cache lookup failed for {item_id}: {e}")
        
        return None
    
    def _cache_paraphrase(self, item_id: str, original: str, candidate: Dict[str, Any]):
        """Cache successful paraphrase in database or file cache."""
        validation = candidate['validation']
        
        if self.use_database_cache:
            try:
                self.database.cache_paraphrase(
                    item_id=item_id,
                    provider=self.provider,
                    candidate_id=candidate.get('candidate_id', 0),
                    text=candidate['text'],
                    cos_sim=validation['semantic_score'],
                    edit_ratio=validation['edit_ratio'],
                    bleu_score=validation['bleu_score'],
                    accepted=validation['overall_valid']
                )
                logger.debug(f"Cached paraphrase in database: {item_id}")
                return
            except Exception as e:
                logger.warning(f"Database cache write failed for {item_id}: {e}")
        
        # Fallback to file cache
        cache_data = {
            'item_id': item_id,
            'provider': self.provider,
            'original_text': original,
            'text': candidate['text'],
            'semantic_score': validation['semantic_score'],
            'edit_ratio': validation['edit_ratio'],
            'bleu_score': validation['bleu_score'],
            'surface_divergent': validation['surface_divergent'],
            'accepted': validation['overall_valid'],
            'generation_metadata': candidate.get('generation_metadata', {}),
            'cached_at': str(asyncio.get_event_loop().time()),
            'config_hash': self.get_config_hash()
        }
        
        cache_file = self.cache_dir / f"{item_id}_{self.provider}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.debug(f"Cached paraphrase in file: {cache_file}")
    
    def get_transform_type(self) -> str:
        """Get transform type identifier."""
        return "paraphrase"
    
    def is_deterministic(self) -> bool:
        """Paraphrase transform is not deterministic due to model generation."""
        return False
    
    def validate_cache_directory(self, cache_dir: Path = None) -> Dict[str, Any]:
        """Validate paraphrase cache directory."""
        target_dir = cache_dir or self.cache_dir
        
        if not target_dir.exists():
            return {
                'total_paraphrases': 0,
                'valid_count': 0,
                'invalid_count': 0,
                'error': f'Cache directory not found: {target_dir}'
            }
        
        cache_files = list(target_dir.glob(f"*_{self.provider}.json"))
        
        total_count = 0
        valid_count = 0
        semantic_fail_count = 0
        surface_fail_count = 0
        both_fail_count = 0
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                total_count += 1
                
                # Re-validate using current thresholds
                if 'original_text' in data and 'text' in data:
                    validation = self.validator.validate_paraphrase(
                        data['original_text'], data['text'],
                        self.semantic_threshold, self.surface_threshold
                    )
                    
                    if validation['overall_valid']:
                        valid_count += 1
                    elif not validation['semantic_valid'] and not validation['surface_divergent']:
                        both_fail_count += 1
                    elif not validation['semantic_valid']:
                        semantic_fail_count += 1
                    elif not validation['surface_divergent']:
                        surface_fail_count += 1
            
            except Exception:
                total_count += 1
                both_fail_count += 1
        
        return {
            'total_paraphrases': total_count,
            'valid_count': valid_count,
            'semantic_fail_count': semantic_fail_count,
            'surface_fail_count': surface_fail_count,
            'both_fail_count': both_fail_count,
            'invalid_count': semantic_fail_count + surface_fail_count + both_fail_count,
            'cache_directory': str(target_dir),
            'provider': self.provider
        }