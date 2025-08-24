"""
Base transform strategy for ScrambleBench.

Defines the interface that all transform strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TransformResult:
    """Result of applying a transformation."""
    original_text: str
    transformed_text: str
    transform_type: str
    transform_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class BaseTransform(ABC):
    """Base class for all transform strategies."""
    
    def __init__(self, config: Dict[str, Any], seed: int = 1337):
        """Initialize transform with configuration and seed."""
        self.config = config
        self.seed = seed
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize transform-specific components."""
        pass
    
    @abstractmethod
    def transform(self, text: str, **kwargs) -> TransformResult:
        """Apply transformation to text."""
        pass
    
    @abstractmethod
    def get_transform_type(self) -> str:
        """Get the transform type identifier."""
        pass
    
    def batch_transform(self, texts: List[str], **kwargs) -> List[TransformResult]:
        """Apply transformation to a batch of texts."""
        results = []
        for text in texts:
            result = self.transform(text, **kwargs)
            results.append(result)
        return results
    
    def is_deterministic(self) -> bool:
        """Whether this transform produces deterministic results."""
        return True  # Most transforms should be deterministic
    
    def get_config_hash(self) -> str:
        """Get hash of configuration for reproducibility tracking."""
        import hashlib
        import json
        
        config_str = json.dumps(self.config, sort_keys=True)
        config_with_seed = f"{config_str}_{self.seed}"
        return hashlib.sha256(config_with_seed.encode()).hexdigest()[:12]