"""
Problem translation system for constructed language benchmarks.

This module handles the translation of benchmark problems into
constructed languages while preserving their logical structure,
difficulty, and solution correctness.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
import logging
from pathlib import Path

from scramblebench.translation.language_generator import ConstructedLanguage, LanguageRule


@dataclass
class TranslationUnit:
    """
    A single unit of translation (word, phrase, or pattern).
    
    Attributes:
        original: Original text
        translated: Translated text
        unit_type: Type of unit (word, phrase, pattern, etc.)
        confidence: Confidence score for the translation
        metadata: Additional translation metadata
    """
    original: str
    translated: str
    unit_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TranslatedProblem:
    """
    A benchmark problem that has been translated to a constructed language.
    
    Attributes:
        original_problem: Original problem data
        translated_problem: Translated problem data
        translation_key: Mapping from translated to original terms
        language_name: Name of the language used for translation
        translation_units: List of individual translation units
        metadata: Additional translation metadata
    """
    original_problem: Dict[str, Any]
    translated_problem: Dict[str, Any]
    translation_key: Dict[str, str]
    language_name: str
    translation_units: List[TranslationUnit]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProblemTranslator:
    """
    Translates benchmark problems into constructed languages.
    
    Handles the systematic transformation of problems while maintaining
    their logical structure and ensuring solution correctness is preserved.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the problem translator.
        
        Args:
            logger: Logger instance (creates default if None)
        """
        self.logger = logger or logging.getLogger("scramblebench.translator")
        self._translation_cache: Dict[str, TranslatedProblem] = {}
    
    def translate_problem(
        self,
        problem: Dict[str, Any],
        language: ConstructedLanguage,
        preserve_numbers: bool = True,
        preserve_proper_nouns: bool = True
    ) -> TranslatedProblem:
        """
        Translate a benchmark problem into the specified constructed language.
        
        Args:
            problem: Original problem data
            language: Constructed language to translate into
            preserve_numbers: Whether to preserve numeric values
            preserve_proper_nouns: Whether to preserve proper nouns
            
        Returns:
            TranslatedProblem containing the translated version
        """
        self.logger.info(f"Translating problem to {language.name}")
        
        # Create cache key
        cache_key = self._create_cache_key(problem, language.name)
        if cache_key in self._translation_cache:
            self.logger.debug("Using cached translation")
            return self._translation_cache[cache_key]
        
        # Initialize translation tracking
        translation_units = []
        translation_key = {}
        translated_problem = {}
        
        # Translate each field in the problem
        for field_name, field_value in problem.items():
            if isinstance(field_value, str):
                translated_value, units = self._translate_text(
                    field_value,
                    language,
                    preserve_numbers=preserve_numbers,
                    preserve_proper_nouns=preserve_proper_nouns
                )
                translated_problem[field_name] = translated_value
                translation_units.extend(units)
                
                # Update translation key
                for unit in units:
                    translation_key[unit.translated] = unit.original
            
            elif isinstance(field_value, list):
                translated_list = []
                for item in field_value:
                    if isinstance(item, str):
                        translated_item, units = self._translate_text(
                            item,
                            language,
                            preserve_numbers=preserve_numbers,
                            preserve_proper_nouns=preserve_proper_nouns
                        )
                        translated_list.append(translated_item)
                        translation_units.extend(units)
                        
                        for unit in units:
                            translation_key[unit.translated] = unit.original
                    else:
                        translated_list.append(item)
                
                translated_problem[field_name] = translated_list
            
            elif isinstance(field_value, dict):
                # Recursively translate nested dictionaries
                translated_dict = self._translate_dict(
                    field_value,
                    language,
                    preserve_numbers=preserve_numbers,
                    preserve_proper_nouns=preserve_proper_nouns
                )
                translated_problem[field_name] = translated_dict['data']
                translation_units.extend(translated_dict['units'])
                translation_key.update(translated_dict['key'])
            
            else:
                # Keep non-string values as-is
                translated_problem[field_name] = field_value
        
        # Create translated problem object
        result = TranslatedProblem(
            original_problem=problem,
            translated_problem=translated_problem,
            translation_key=translation_key,
            language_name=language.name,
            translation_units=translation_units,
            metadata={
                'preserve_numbers': preserve_numbers,
                'preserve_proper_nouns': preserve_proper_nouns,
                'total_translations': len(translation_units),
                'unique_translations': len(set(unit.original for unit in translation_units))
            }
        )
        
        # Cache the result
        self._translation_cache[cache_key] = result
        
        self.logger.info(
            f"Translation completed: {len(translation_units)} units, "
            f"{len(translation_key)} unique mappings"
        )
        
        return result
    
    def _translate_text(
        self,
        text: str,
        language: ConstructedLanguage,
        preserve_numbers: bool = True,
        preserve_proper_nouns: bool = True
    ) -> Tuple[str, List[TranslationUnit]]:
        """
        Translate a single text string using the language rules.
        
        Args:
            text: Text to translate
            language: Language to use for translation
            preserve_numbers: Whether to preserve numeric values
            preserve_proper_nouns: Whether to preserve proper nouns
            
        Returns:
            Tuple of (translated_text, translation_units)
        """
        translated = text
        units = []
        
        # Track what has been preserved
        preserved_spans = []
        
        # Preserve numbers if requested
        if preserve_numbers:
            number_pattern = r'\b\d+(?:\.\d+)?\b'
            for match in re.finditer(number_pattern, text):
                preserved_spans.append((match.start(), match.end()))
        
        # Preserve proper nouns if requested
        if preserve_proper_nouns:
            # Simple heuristic: words that start with capital letters
            proper_noun_pattern = r'\b[A-Z][a-z]+\b'
            for match in re.finditer(proper_noun_pattern, text):
                # Check if it's not at the start of a sentence
                if match.start() == 0:
                    continue
                if match.start() > 0 and text[match.start() - 1] in '.!?':
                    continue
                preserved_spans.append((match.start(), match.end()))
        
        # Sort spans by start position
        preserved_spans.sort()
        
        # Apply language rules in priority order
        for rule in sorted(language.rules, key=lambda r: r.priority, reverse=True):
            if rule.rule_type == "word":
                # Word-level replacements (highest priority)
                old_translated = translated
                translated = self._apply_word_rule(
                    translated, rule, preserved_spans, text
                )
                if translated != old_translated:
                    units.append(TranslationUnit(
                        original=rule.source,
                        translated=rule.target,
                        unit_type="word",
                        confidence=1.0,
                        metadata={'rule_priority': rule.priority}
                    ))
        
        # Apply character-level rules
        for rule in sorted(language.rules, key=lambda r: r.priority, reverse=True):
            if rule.rule_type in ["character", "phonetic", "rotation", "consonant", "vowel"]:
                old_translated = translated
                translated = self._apply_character_rule(
                    translated, rule, preserved_spans, text
                )
                # Note: Character rules generate many small units, 
                # we might want to aggregate them
        
        # Apply vocabulary mappings
        for original_word, translated_word in language.vocabulary.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(original_word) + r'\b'
            if re.search(pattern, translated, re.IGNORECASE):
                old_translated = translated
                translated = re.sub(
                    pattern, translated_word, translated, flags=re.IGNORECASE
                )
                if translated != old_translated:
                    units.append(TranslationUnit(
                        original=original_word,
                        translated=translated_word,
                        unit_type="vocabulary",
                        confidence=1.0,
                        metadata={'source': 'vocabulary_mapping'}
                    ))
        
        return translated, units
    
    def _apply_word_rule(
        self,
        text: str,
        rule: LanguageRule,
        preserved_spans: List[Tuple[int, int]],
        original_text: str
    ) -> str:
        """Apply a word-level transformation rule."""
        # Use regex for word boundary matching
        pattern = rule.source
        if not pattern.startswith('\\b') and not pattern.endswith('\\b'):
            # Ensure word boundaries for exact word matching
            if rule.rule_type == "word":
                pattern = r'\b' + re.escape(rule.source) + r'\b'
        
        # Find all matches
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        # Filter out matches that overlap with preserved spans
        valid_matches = []
        for match in matches:
            overlap = False
            for start, end in preserved_spans:
                if (match.start() < end and match.end() > start):
                    overlap = True
                    break
            if not overlap:
                valid_matches.append(match)
        
        # Apply replacements from right to left to preserve positions
        for match in reversed(valid_matches):
            text = text[:match.start()] + rule.target + text[match.end():]
        
        return text
    
    def _apply_character_rule(
        self,
        text: str,
        rule: LanguageRule,
        preserved_spans: List[Tuple[int, int]],
        original_text: str
    ) -> str:
        """Apply a character-level transformation rule."""
        result = ""
        i = 0
        
        while i < len(text):
            # Check if current position is in a preserved span
            in_preserved = False
            for start, end in preserved_spans:
                # Map position back to original text (approximate)
                if start <= i < end:
                    in_preserved = True
                    break
            
            if in_preserved:
                result += text[i]
                i += 1
            else:
                # Check if rule matches at current position
                if text[i:].startswith(rule.source):
                    result += rule.target
                    i += len(rule.source)
                else:
                    result += text[i]
                    i += 1
        
        return result
    
    def _translate_dict(
        self,
        data: Dict[str, Any],
        language: ConstructedLanguage,
        preserve_numbers: bool = True,
        preserve_proper_nouns: bool = True
    ) -> Dict[str, Any]:
        """Recursively translate a dictionary."""
        translated_data = {}
        all_units = []
        all_keys = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                translated_value, units = self._translate_text(
                    value,
                    language,
                    preserve_numbers=preserve_numbers,
                    preserve_proper_nouns=preserve_proper_nouns
                )
                translated_data[key] = translated_value
                all_units.extend(units)
                
                for unit in units:
                    all_keys[unit.translated] = unit.original
            
            elif isinstance(value, dict):
                nested_result = self._translate_dict(
                    value,
                    language,
                    preserve_numbers=preserve_numbers,
                    preserve_proper_nouns=preserve_proper_nouns
                )
                translated_data[key] = nested_result['data']
                all_units.extend(nested_result['units'])
                all_keys.update(nested_result['key'])
            
            elif isinstance(value, list):
                translated_list = []
                for item in value:
                    if isinstance(item, str):
                        translated_item, units = self._translate_text(
                            item,
                            language,
                            preserve_numbers=preserve_numbers,
                            preserve_proper_nouns=preserve_proper_nouns
                        )
                        translated_list.append(translated_item)
                        all_units.extend(units)
                        
                        for unit in units:
                            all_keys[unit.translated] = unit.original
                    else:
                        translated_list.append(item)
                
                translated_data[key] = translated_list
            
            else:
                translated_data[key] = value
        
        return {
            'data': translated_data,
            'units': all_units,
            'key': all_keys
        }
    
    def _create_cache_key(self, problem: Dict[str, Any], language_name: str) -> str:
        """Create a cache key for the problem-language combination."""
        # Simple hash based on problem content and language
        problem_str = str(sorted(problem.items()))
        return f"{language_name}_{hash(problem_str)}"
    
    def translate_answer(
        self,
        answer: str,
        translation_key: Dict[str, str],
        reverse: bool = False
    ) -> str:
        """
        Translate an answer using the translation key.
        
        Args:
            answer: Answer to translate
            translation_key: Translation mapping
            reverse: If True, translate from constructed language back to English
            
        Returns:
            Translated answer
        """
        if reverse:
            # Translate from constructed language back to English
            mapping = translation_key  # Already maps constructed -> original
        else:
            # Translate from English to constructed language
            mapping = {v: k for k, v in translation_key.items()}
        
        translated = answer
        
        # Apply translations (word boundaries for accuracy)
        for original, target in mapping.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            translated = re.sub(pattern, target, translated, flags=re.IGNORECASE)
        
        return translated
    
    def verify_translation_consistency(
        self,
        translated_problem: TranslatedProblem
    ) -> Dict[str, Any]:
        """
        Verify that the translation maintains problem consistency.
        
        Args:
            translated_problem: Problem to verify
            
        Returns:
            Dictionary containing verification results
        """
        verification = {
            'consistent': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check translation coverage
        original_words = set()
        translated_words = set()
        
        for unit in translated_problem.translation_units:
            original_words.add(unit.original.lower())
            translated_words.add(unit.translated.lower())
        
        # Check for untranslated common words
        # (This would need a more sophisticated implementation)
        
        # Check for translation consistency (same original -> same translated)
        translation_map = {}
        for unit in translated_problem.translation_units:
            key = unit.original.lower()
            if key in translation_map:
                if translation_map[key] != unit.translated.lower():
                    verification['consistent'] = False
                    verification['issues'].append(
                        f"Inconsistent translation: '{unit.original}' -> "
                        f"'{translation_map[key]}' and '{unit.translated}'"
                    )
            else:
                translation_map[key] = unit.translated.lower()
        
        verification['statistics'] = {
            'unique_original_words': len(original_words),
            'unique_translated_words': len(translated_words),
            'total_translation_units': len(translated_problem.translation_units),
            'translation_key_size': len(translated_problem.translation_key)
        }
        
        return verification
    
    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._translation_cache.clear()
        self.logger.info("Translation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the translation cache."""
        return {
            'cached_translations': len(self._translation_cache)
        }