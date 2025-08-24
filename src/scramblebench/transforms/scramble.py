"""
Scramble transform strategy with deterministic symbol substitution.

Implements symbol substitution with configurable intensity levels and 
deterministic mapping based on seed for reproducible scrambling.
"""

import re
import random
from typing import Dict, Any, List, Set
from .base import BaseTransform, TransformResult


class ScrambleTransform(BaseTransform):
    """Deterministic scramble transform with symbol substitution."""
    
    def _initialize(self):
        """Initialize scramble transform with symbol mappings."""
        self.scheme_type = self.config.get('scheme', {}).get('type', 'symbol_substitution')
        self.alphabet = self.config.get('scheme', {}).get('alphabet', '@#$%&*+=?')
        
        # Create deterministic random generator with seed
        self._rng = random.Random(self.seed)
        
        # Pre-compute symbol mappings for different intensity levels
        self._symbol_mappings = {}
        
        if self.scheme_type == 'symbol_substitution':
            self._initialize_symbol_mappings()
    
    def _initialize_symbol_mappings(self):
        """Initialize deterministic symbol mappings."""
        # Characters that can be replaced (letters, numbers, some punctuation)
        replaceable_chars = set(
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
        )
        
        # Create shuffled replacement alphabet for each character
        replacement_symbols = list(self.alphabet)
        
        for char in replaceable_chars:
            # Use character-specific seed for consistent mapping
            char_seed = self.seed + ord(char)
            char_rng = random.Random(char_seed)
            
            # Shuffle symbols for this character
            char_symbols = replacement_symbols.copy()
            char_rng.shuffle(char_symbols)
            
            self._symbol_mappings[char] = char_symbols
    
    def transform(self, text: str, scramble_level: float = 0.3, **kwargs) -> TransformResult:
        """Apply scramble transformation with specified intensity level."""
        if not (0.0 <= scramble_level <= 1.0):
            return TransformResult(
                original_text=text,
                transformed_text=text,
                transform_type="scramble",
                transform_metadata={},
                success=False,
                error_message=f"Invalid scramble level: {scramble_level}. Must be between 0.0 and 1.0"
            )
        
        if scramble_level == 0.0:
            # No scrambling
            transformed_text = text
        else:
            transformed_text = self._apply_scrambling(text, scramble_level)
        
        # Compute perturbation metrics
        original_chars = len(text)
        transformed_chars = len(transformed_text)
        
        # Count character differences
        char_differences = sum(
            1 for orig, trans in zip(text, transformed_text) 
            if orig != trans
        )
        
        char_change_ratio = char_differences / original_chars if original_chars > 0 else 0.0
        
        return TransformResult(
            original_text=text,
            transformed_text=transformed_text,
            transform_type="scramble",
            transform_metadata={
                "scramble_level": scramble_level,
                "scheme_type": self.scheme_type,
                "alphabet": self.alphabet,
                "seed": self.seed,
                "config_hash": self.get_config_hash(),
                "char_change_ratio": char_change_ratio,
                "char_differences": char_differences,
                "original_length": original_chars,
                "transformed_length": transformed_chars
            },
            success=True
        )
    
    def _apply_scrambling(self, text: str, level: float) -> str:
        """Apply symbol substitution scrambling at specified level."""
        if self.scheme_type != 'symbol_substitution':
            # Future: implement other scrambling schemes
            return text
        
        # Create deterministic selection of characters to replace
        replaceable_positions = []
        
        for i, char in enumerate(text):
            if char in self._symbol_mappings:
                replaceable_positions.append(i)
        
        if not replaceable_positions:
            return text  # No replaceable characters
        
        # Deterministically select positions to scramble based on level
        num_to_replace = int(len(replaceable_positions) * level)
        
        # Use text-specific seed for position selection
        text_seed = self.seed + hash(text) % (2**31)
        position_rng = random.Random(text_seed)
        
        positions_to_replace = position_rng.sample(
            replaceable_positions, 
            min(num_to_replace, len(replaceable_positions))
        )
        
        # Apply replacements
        text_chars = list(text)
        
        for pos in positions_to_replace:
            original_char = text_chars[pos]
            
            if original_char in self._symbol_mappings:
                # Get replacement symbol (first available for this character)
                replacement_symbols = self._symbol_mappings[original_char]
                
                # Use position-specific selection for symbol choice
                symbol_index = (pos + self.seed) % len(replacement_symbols)
                replacement_symbol = replacement_symbols[symbol_index]
                
                text_chars[pos] = replacement_symbol
        
        return ''.join(text_chars)
    
    def get_transform_type(self) -> str:
        """Get transform type identifier."""
        return "scramble"
    
    def is_deterministic(self) -> bool:
        """Scramble transform is deterministic given same seed and level."""
        return True
    
    def get_available_levels(self) -> List[float]:
        """Get list of available scramble levels from config."""
        return self.config.get('levels', [0.1, 0.2, 0.3, 0.4, 0.5])
    
    def validate_text_scrambling(self, text: str, scrambled: str, expected_level: float) -> Dict[str, Any]:
        """Validate that scrambling meets expected level and properties."""
        if len(text) != len(scrambled):
            return {
                "valid": False,
                "reason": "Length mismatch",
                "expected_length": len(text),
                "actual_length": len(scrambled)
            }
        
        # Count character differences
        differences = sum(1 for orig, scram in zip(text, scrambled) if orig != scram)
        actual_ratio = differences / len(text) if len(text) > 0 else 0.0
        
        # Allow some tolerance for actual vs expected level
        level_tolerance = 0.05
        
        if abs(actual_ratio - expected_level) > level_tolerance:
            return {
                "valid": False,
                "reason": "Scramble level mismatch",
                "expected_level": expected_level,
                "actual_level": actual_ratio,
                "tolerance": level_tolerance
            }
        
        # Check that scrambled characters use only symbols from alphabet
        scrambled_chars = set(scrambled) - set(text)
        alphabet_chars = set(self.alphabet)
        
        if scrambled_chars - alphabet_chars:
            return {
                "valid": False,
                "reason": "Invalid scramble symbols used",
                "invalid_symbols": list(scrambled_chars - alphabet_chars)
            }
        
        return {
            "valid": True,
            "actual_level": actual_ratio,
            "differences": differences,
            "symbols_used": list(scrambled_chars)
        }