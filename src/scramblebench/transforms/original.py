"""
Original transform strategy - pass-through with no transformation.

This serves as the baseline for all evaluation comparisons.
"""

from typing import Dict, Any
from .base import BaseTransform, TransformResult


class OriginalTransform(BaseTransform):
    """Original transform that passes text through unchanged."""
    
    def _initialize(self):
        """Initialize original transform (no-op)."""
        pass
    
    def transform(self, text: str, **kwargs) -> TransformResult:
        """Return text unchanged."""
        return TransformResult(
            original_text=text,
            transformed_text=text,
            transform_type="original",
            transform_metadata={
                "seed": self.seed,
                "config_hash": self.get_config_hash()
            },
            success=True
        )
    
    def get_transform_type(self) -> str:
        """Get transform type identifier."""
        return "original"
    
    def is_deterministic(self) -> bool:
        """Original transform is always deterministic."""
        return True