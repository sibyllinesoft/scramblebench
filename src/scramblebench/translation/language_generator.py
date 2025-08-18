"""
Constructed language generation for translation benchmarks.

This module provides sophisticated artificial language generation capabilities
for creating contamination-resistant benchmark transformations. It implements
multiple language types with varying complexity levels to transform problems
while preserving their logical structure and solvability.

The module includes:

* **Language Types**: Six different artificial language paradigms
* **Phonotactic Constraints**: Linguistically plausible sound patterns
* **Morphological Rules**: Systematic word formation patterns
* **Frequency Modeling**: Realistic word length and frequency distributions
* **Complexity Scaling**: Adjustable transformation complexity (1-10 scale)

Example:
    Basic language generation::

        from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
        
        # Create generator with reproducible seed
        generator = LanguageGenerator(seed=42)
        
        # Generate different language types
        substitution_lang = generator.generate_language(
            name="simple_cipher",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3,
            vocab_size=1000
        )
        
        phonetic_lang = generator.generate_language(
            name="phonetic_transform", 
            language_type=LanguageType.PHONETIC,
            complexity=7,
            vocab_size=2000
        )
        
        # Save and reload languages
        generator.save_language(substitution_lang, Path("simple_cipher.json"))
        loaded_lang = generator.load_language(Path("simple_cipher.json"))

Language Types:
    * :attr:`LanguageType.SUBSTITUTION` - Simple character/word substitution ciphers
    * :attr:`LanguageType.PHONETIC` - Phonetically motivated sound changes  
    * :attr:`LanguageType.SCRAMBLED` - Systematic character scrambling patterns
    * :attr:`LanguageType.SYNTHETIC` - Fully artificial vocabulary with grammar
    * :attr:`LanguageType.ENGLISH_LIKE` - Plausible English phonotactic patterns
    * :attr:`LanguageType.RANDOM_FREQUENCY` - Frequency-correlated word generation

Note:
    Generated languages maintain systematic transformation rules that preserve
    the logical structure of problems while eliminating lexical similarity to
    training data. All transformations are reversible for verification.

See Also:
    :class:`scramblebench.translation.translator.ProblemTranslator`: Problem translation using generated languages
    :class:`scramblebench.translation.benchmark.TranslationBenchmark`: Benchmark implementation
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import string
import json
import logging
import math
import re
from pathlib import Path
from collections import Counter


class LanguageType(Enum):
    """
    Enumeration of constructed language types for benchmark transformation.
    
    Each language type implements a different approach to creating artificial
    languages that preserve logical structure while eliminating lexical overlap
    with potential training data.
    
    :cvar SUBSTITUTION: Simple character and word substitution ciphers
    :vartype SUBSTITUTION: str
    :cvar PHONETIC: Phonetically motivated sound changes following linguistic patterns
    :vartype PHONETIC: str
    :cvar SCRAMBLED: Systematic character scrambling with consistent rules
    :vartype SCRAMBLED: str
    :cvar SYNTHETIC: Fully artificial vocabulary with constructed grammar rules
    :vartype SYNTHETIC: str
    :cvar ENGLISH_LIKE: Generated words following English phonotactic constraints
    :vartype ENGLISH_LIKE: str
    :cvar RANDOM_FREQUENCY: Words generated to match realistic frequency distributions
    :vartype RANDOM_FREQUENCY: str
    
    Example:
        Using different language types::
        
            # Simple substitution for basic transformations
            lang1 = generator.generate_language("cipher", LanguageType.SUBSTITUTION, complexity=3)
            
            # Phonetic for linguistically plausible changes
            lang2 = generator.generate_language("phonetic", LanguageType.PHONETIC, complexity=6)
            
            # Synthetic for completely artificial vocabulary
            lang3 = generator.generate_language("alien", LanguageType.SYNTHETIC, complexity=8)
    
    Note:
        Higher complexity levels (7-10) produce more sophisticated transformations
        but may require longer generation time. Lower levels (1-3) are suitable
        for rapid prototyping and testing.
    """
    SUBSTITUTION = "substitution"  # Simple character/word substitution
    PHONETIC = "phonetic"  # Phonetically plausible transformations
    SCRAMBLED = "scrambled"  # Character scrambling with rules
    SYNTHETIC = "synthetic"  # Generated vocabulary with grammar
    ENGLISH_LIKE = "english_like"  # Plausible English-looking words
    RANDOM_FREQUENCY = "random_frequency"  # Frequency-correlated random generation


@dataclass
class LanguageRule:
    """
    A single transformation rule for systematic language transformation.
    
    Language rules define how text elements are transformed during benchmark
    translation. Rules can operate at character, morpheme, word, or phrase levels
    with configurable priority and conditional application.
    
    :param source: Source pattern to match (supports regex for complex patterns)
    :type source: str
    :param target: Target pattern to replace with (can be string or callable)
    :type target: str
    :param rule_type: Classification of rule type for application order
    :type rule_type: str
    :param priority: Application priority (higher numbers applied first)
    :type priority: int
    :param conditions: Optional conditions for conditional rule application
    :type conditions: Dict[str, str]
    
    Example:
        Creating transformation rules::
        
            # Character-level substitution
            char_rule = LanguageRule(
                source="ch",
                target="kh", 
                rule_type="phonetic",
                priority=5
            )
            
            # Word-level replacement with regex
            word_rule = LanguageRule(
                source=r"\\bthe\\b",
                target="dha",
                rule_type="word",
                priority=10
            )
            
            # Conditional morphological rule
            morph_rule = LanguageRule(
                source="ing$",
                target="an",
                rule_type="suffix",
                priority=8,
                conditions={"pos": "verb"}
            )
    
    Note:
        Rules with higher priority values are applied before lower priority rules.
        This allows fine-grained control over transformation order to prevent
        rule conflicts and ensure consistent results.
    
    See Also:
        :class:`ConstructedLanguage`: Container for complete rule sets
        :meth:`LanguageGenerator._apply_transformation_rules`: Rule application logic
    """
    source: str
    target: str
    rule_type: str
    priority: int = 1
    conditions: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConstructedLanguage:
    """
    A complete constructed language definition.
    
    Attributes:
        name: Name of the language
        language_type: Type of language transformation
        rules: List of transformation rules
        vocabulary: Core vocabulary mappings
        metadata: Additional language information
    """
    name: str
    language_type: LanguageType
    rules: List[LanguageRule]
    vocabulary: Dict[str, str]
    metadata: Dict[str, str] = field(default_factory=dict)


class LanguageGenerator:
    """
    Advanced generator for constructed languages in contamination-resistant benchmarks.
    
    This class implements sophisticated artificial language generation using
    linguistic principles to create systematic transformation systems. Generated
    languages maintain logical problem structure while eliminating lexical
    overlap with potential training data.
    
    The generator supports six distinct language paradigms with configurable
    complexity levels (1-10) and incorporates realistic linguistic features:
    
    * **Phonotactic Constraints**: Following natural language sound patterns
    * **Morphological Systems**: Prefix/suffix transformation rules  
    * **Frequency Distributions**: Realistic word length and usage patterns
    * **Semantic Preservation**: Maintaining logical problem relationships
    * **Reversible Transformations**: Full bidirectional mapping capability
    
    :param seed: Random seed for reproducible language generation
    :type seed: Optional[int]
    :param logger: Logger instance for generation process tracking
    :type logger: Optional[logging.Logger]
    
    :ivar rng: Random number generator for consistent results
    :vartype rng: random.Random
    :ivar _generated_languages: Cache of previously generated languages
    :vartype _generated_languages: Dict[str, ConstructedLanguage]
    :ivar english_onsets: Valid English syllable-initial consonant clusters
    :vartype english_onsets: Dict[str, List[str]]
    :ivar english_codas: Valid English syllable-final consonant clusters  
    :vartype english_codas: Dict[str, List[str]]
    :ivar english_vowels: English vowel patterns and diphthongs
    :vartype english_vowels: Dict[str, List[str]]
    
    Example:
        Comprehensive language generation workflow::
        
            # Initialize with reproducible seed
            generator = LanguageGenerator(seed=42)
            
            # Generate languages of varying complexity
            simple_lang = generator.generate_language(
                name="basic_cipher",
                language_type=LanguageType.SUBSTITUTION,
                complexity=3,
                vocab_size=500
            )
            
            complex_lang = generator.generate_language(
                name="advanced_phonetic",
                language_type=LanguageType.PHONETIC, 
                complexity=8,
                vocab_size=2000
            )
            
            # Batch vocabulary generation
            word_list = ["question", "answer", "problem", "solution"]
            translations = generator.generate_vocabulary_batch(
                simple_lang, word_list
            )
            
            # Persistence operations
            generator.save_language(simple_lang, Path("basic_cipher.json"))
            loaded_lang = generator.load_language(Path("basic_cipher.json"))
            
            # Language management
            available_languages = generator.list_languages()
            specific_lang = generator.get_language("basic_cipher")
    
    Note:
        All generated languages are deterministic given the same seed, ensuring
        reproducible benchmark transformations. The generator caches languages
        to avoid regeneration and maintains linguistic consistency across
        vocabulary expansions.
        
    See Also:
        :class:`LanguageType`: Available language transformation paradigms
        :class:`ConstructedLanguage`: Generated language data structure
        :class:`scramblebench.translation.translator.ProblemTranslator`: Language application
    """
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the language generator.
        
        Args:
            seed: Random seed for reproducible language generation
            logger: Logger instance (creates default if None)
        """
        self.logger = logger or logging.getLogger("scramblebench.language_generator")
        self.rng = random.Random(seed)
        self._generated_languages: Dict[str, ConstructedLanguage] = {}
        
        # Initialize English phonotactics data
        self._init_english_phonotactics()
        
        # Initialize frequency data for word length correlation
        self._init_frequency_data()
    
    def _init_english_phonotactics(self) -> None:
        """Initialize English phonotactic constraints and patterns."""
        # Valid English onset clusters (beginning of syllables)
        self.english_onsets = {
            # Single consonants
            'single': ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'y', 'z'],
            # Two-consonant clusters
            'two': ['bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw', 'tr', 'tw', 'th', 'sh', 'ch', 'wh'],
            # Three-consonant clusters
            'three': ['scr', 'spl', 'spr', 'str', 'squ', 'thr', 'shr']
        }
        
        # Valid English coda clusters (end of syllables)
        self.english_codas = {
            'single': ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x', 'z'],
            'two': ['ct', 'ft', 'ld', 'lf', 'lk', 'lm', 'lp', 'lt', 'mp', 'nd', 'ng', 'nk', 'nt', 'pt', 'rd', 'rf', 'rk', 'rl', 'rm', 'rn', 'rp', 'rt', 'sk', 'sp', 'st', 'ts'],
            'three': ['mpt', 'nct', 'nst', 'rst', 'rts']
        }
        
        # English vowel patterns
        self.english_vowels = {
            'short': ['a', 'e', 'i', 'o', 'u'],
            'long': ['ay', 'ee', 'ie', 'oa', 'ue'],
            'diphthongs': ['ai', 'au', 'aw', 'ei', 'ew', 'oi', 'ou', 'ow']
        }
        
        # Common English morphological patterns
        self.english_morphology = {
            'prefixes': ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out', 'up', 'sub', 'inter', 'super'],
            'suffixes': ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment', 'ness', 'ful', 'less', 'able', 'ible'],
            'compound_connectors': ['', 'y', 'i']
        }
        
        # Frequency weights for phoneme selection (based on English phoneme frequency)
        self.consonant_weights = {
            't': 16, 'n': 15, 's': 14, 'r': 13, 'l': 11, 'd': 9, 'k': 8, 'm': 7,
            'g': 6, 'f': 6, 'p': 5, 'b': 4, 'v': 4, 'h': 4, 'w': 3, 'y': 3,
            'z': 2, 'j': 2, 'q': 1, 'x': 1
        }
        
        self.vowel_weights = {
            'e': 27, 'a': 18, 'i': 15, 'o': 13, 'u': 7
        }
    
    def _init_frequency_data(self) -> None:
        """Initialize word frequency and length correlation data."""
        # Zipf distribution parameters for English word lengths
        # Based on corpus analysis: shorter words are much more frequent
        self.length_frequency_params = {
            1: 0.05,   # Very rare (I, a)
            2: 0.15,   # Common (to, of, in, is)
            3: 0.25,   # Very common (the, and, for)
            4: 0.20,   # Common (that, with, have)
            5: 0.15,   # Moderate (about, would, there)
            6: 0.10,   # Less common (should, before)
            7: 0.05,   # Uncommon (through, because)
            8: 0.03,   # Rare (although, together)
            9: 0.015,  # Very rare (something, different)
            10: 0.005  # Extremely rare (everything, understand)
        }
        
        # Normalize to ensure sum = 1
        total = sum(self.length_frequency_params.values())
        self.length_frequency_params = {k: v/total for k, v in self.length_frequency_params.items()}
        
        # Generate cumulative distribution for sampling
        self.length_cumulative = []
        cumsum = 0
        for length in sorted(self.length_frequency_params.keys()):
            cumsum += self.length_frequency_params[length]
            self.length_cumulative.append((length, cumsum))
    
    def generate_language(
        self,
        name: str,
        language_type: LanguageType,
        complexity: int = 5,
        vocab_size: int = 1000
    ) -> ConstructedLanguage:
        """
        Generate a comprehensive constructed language for benchmark transformation.
        
        This method creates a complete artificial language system including
        transformation rules, core vocabulary, and metadata. The generated
        language maintains systematic patterns while providing sufficient
        complexity to avoid training data contamination.
        
        :param name: Unique identifier for the generated language
        :type name: str
        :param language_type: Paradigm for language generation strategy
        :type language_type: LanguageType
        :param complexity: Transformation complexity level (1=simple, 10=highly complex)
        :type complexity: int
        :param vocab_size: Number of core vocabulary items to generate
        :type vocab_size: int
        :return: Complete constructed language with rules and vocabulary
        :rtype: ConstructedLanguage
        
        :raises ValueError: If language_type is not supported
        :raises RuntimeError: If language generation fails
        
        Example:
            Generate languages for different use cases::
            
                # Simple substitution for basic testing
                simple = generator.generate_language(
                    "test_cipher", LanguageType.SUBSTITUTION, complexity=2, vocab_size=100
                )
                
                # Complex phonetic for production evaluation
                phonetic = generator.generate_language(
                    "eval_phonetic", LanguageType.PHONETIC, complexity=7, vocab_size=5000
                )
                
                # Synthetic alien language for creative tasks
                synthetic = generator.generate_language(
                    "alien_lang", LanguageType.SYNTHETIC, complexity=9, vocab_size=2000
                )
        
        Complexity Levels:
            * **1-3**: Basic transformations, suitable for prototyping
            * **4-6**: Moderate complexity, good for standard evaluation  
            * **7-8**: High complexity, comprehensive linguistic features
            * **9-10**: Maximum complexity, research-grade transformations
        
        Note:
            Generated languages are cached automatically and can be retrieved
            using :meth:`get_language`. All transformations are deterministic
            given the same seed value.
            
        See Also:
            :meth:`generate_vocabulary_batch`: Extend vocabulary for specific words
            :meth:`save_language`: Persist language for reuse
            :class:`LanguageType`: Available generation strategies
        """
        self.logger.info(f"Generating {language_type.value} language: {name}")
        
        if language_type == LanguageType.SUBSTITUTION:
            language = self._generate_substitution_language(name, complexity, vocab_size)
        elif language_type == LanguageType.PHONETIC:
            language = self._generate_phonetic_language(name, complexity, vocab_size)
        elif language_type == LanguageType.SCRAMBLED:
            language = self._generate_scrambled_language(name, complexity, vocab_size)
        elif language_type == LanguageType.SYNTHETIC:
            language = self._generate_synthetic_language(name, complexity, vocab_size)
        elif language_type == LanguageType.ENGLISH_LIKE:
            language = self._generate_english_like_language(name, complexity, vocab_size)
        elif language_type == LanguageType.RANDOM_FREQUENCY:
            language = self._generate_random_frequency_language(name, complexity, vocab_size)
        else:
            raise ValueError(f"Unsupported language type: {language_type}")
        
        self._generated_languages[name] = language
        return language
    
    def _generate_substitution_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate a simple substitution cipher language."""
        rules = []
        vocabulary = {}
        
        # Character-level substitutions
        char_map = {}
        for char in string.ascii_lowercase:
            # Create a consistent but scrambled mapping
            new_char = chr((ord(char) - ord('a') + 13) % 26 + ord('a'))
            char_map[char] = new_char
            rules.append(LanguageRule(
                source=char,
                target=new_char,
                rule_type="character",
                priority=1
            ))
        
        # Word-level substitutions for common words
        common_words = [
            "the", "and", "or", "not", "is", "are", "was", "were",
            "if", "then", "else", "when", "where", "how", "what",
            "number", "word", "time", "way", "people", "water",
            "first", "last", "long", "little", "right", "left"
        ]
        
        for word in common_words[:min(len(common_words), vocab_size // 10)]:
            # Generate pseudo-word
            new_word = self._generate_pseudo_word(len(word))
            vocabulary[word] = new_word
            rules.append(LanguageRule(
                source=f"\\b{word}\\b",
                target=new_word,
                rule_type="word",
                priority=10  # Higher priority than character rules
            ))
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.SUBSTITUTION,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'character_mapping': json.dumps(char_map),
                'description': 'Simple substitution cipher with word replacements'
            }
        )
    
    def _generate_phonetic_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate a phonetically plausible language."""
        rules = []
        vocabulary = {}
        
        # Phonetic transformation rules
        phonetic_rules = [
            ("ch", "kh"),
            ("sh", "sh"),
            ("th", "dh"),
            ("ph", "f"),
            ("tion", "sion"),
            ("ck", "k"),
            ("qu", "kw"),
        ]
        
        for i, (source, target) in enumerate(phonetic_rules):
            rules.append(LanguageRule(
                source=source,
                target=target,
                rule_type="phonetic",
                priority=5 + i
            ))
        
        # Vowel shifts
        vowel_shifts = [
            ("a", "æ"),
            ("e", "ə"),
            ("i", "ɪ"),
            ("o", "ɔ"),
            ("u", "ʊ"),
        ]
        
        for i, (source, target) in enumerate(vowel_shifts):
            if complexity > 3:  # Only apply in higher complexity
                rules.append(LanguageRule(
                    source=source,
                    target=target,
                    rule_type="vowel",
                    priority=2
                ))
        
        # Generate vocabulary with phonetic rules
        base_words = self._get_common_vocabulary(vocab_size)
        for word in base_words:
            transformed = self._apply_phonetic_rules(word, phonetic_rules[:complexity])
            vocabulary[word] = transformed
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.PHONETIC,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'description': 'Phonetically motivated transformations',
                'rule_count': str(len(rules))
            }
        )
    
    def _generate_scrambled_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate a language with systematic character scrambling."""
        rules = []
        vocabulary = {}
        
        # Create scrambling patterns based on complexity
        if complexity <= 3:
            # Simple rotation
            shift = complexity
            for char in string.ascii_lowercase:
                new_char = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                rules.append(LanguageRule(
                    source=char,
                    target=new_char,
                    rule_type="rotation",
                    priority=1
                ))
        else:
            # More complex scrambling patterns
            # Consonant clusters
            consonants = "bcdfghjklmnpqrstvwxyz"
            consonant_map = list(consonants)
            self.rng.shuffle(consonant_map)
            
            for i, char in enumerate(consonants):
                rules.append(LanguageRule(
                    source=char,
                    target=consonant_map[i],
                    rule_type="consonant",
                    priority=3
                ))
            
            # Vowel scrambling
            vowels = "aeiou"
            vowel_map = list(vowels)
            self.rng.shuffle(vowel_map)
            
            for i, char in enumerate(vowels):
                rules.append(LanguageRule(
                    source=char,
                    target=vowel_map[i],
                    rule_type="vowel",
                    priority=2
                ))
        
        # Apply scrambling to vocabulary
        base_words = self._get_common_vocabulary(vocab_size)
        for word in base_words:
            scrambled = self._apply_scrambling_rules(word, rules)
            vocabulary[word] = scrambled
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.SCRAMBLED,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'description': 'Systematic character scrambling',
                'scrambling_pattern': f'complexity_{complexity}'
            }
        )
    
    def _generate_synthetic_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate a synthetic language with artificial vocabulary."""
        rules = []
        vocabulary = {}
        
        # Define synthetic phonemes
        consonants = ["p", "t", "k", "m", "n", "s", "l", "r", "w", "j"]
        vowels = ["a", "e", "i", "o", "u"]
        
        if complexity > 5:
            # Add more complex phonemes
            consonants.extend(["th", "sh", "ch", "ng"])
            vowels.extend(["ai", "ou", "ei"])
        
        # Grammar rules for synthetic language
        grammar_rules = [
            ("ing", "an"),  # Present participle
            ("ed", "et"),   # Past tense
            ("s", "os"),    # Plural
            ("ly", "al"),   # Adverb suffix
        ]
        
        for source, target in grammar_rules:
            rules.append(LanguageRule(
                source=f"{source}$",  # End of word
                target=target,
                rule_type="grammar",
                priority=8
            ))
        
        # Generate synthetic vocabulary
        base_words = self._get_common_vocabulary(vocab_size)
        for word in base_words:
            synthetic_word = self._generate_synthetic_word(
                len(word), consonants, vowels
            )
            vocabulary[word] = synthetic_word
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.SYNTHETIC,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'description': 'Fully synthetic language with artificial grammar',
                'phoneme_inventory': json.dumps({
                    'consonants': consonants,
                    'vowels': vowels
                })
            }
        )
    
    def _generate_english_like_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate an English-like language with plausible phonotactics."""
        rules = []
        vocabulary = {}
        
        # Morphological transformation rules based on complexity
        morphological_rules = []
        
        if complexity >= 3:
            # Add prefix transformations
            for prefix in self.english_morphology['prefixes'][:complexity]:
                alt_prefix = self._generate_morpheme_variant(prefix)
                morphological_rules.append(LanguageRule(
                    source=f"^{prefix}",
                    target=alt_prefix,
                    rule_type="prefix",
                    priority=9
                ))
        
        if complexity >= 4:
            # Add suffix transformations
            for suffix in self.english_morphology['suffixes'][:complexity]:
                alt_suffix = self._generate_morpheme_variant(suffix)
                morphological_rules.append(LanguageRule(
                    source=f"{suffix}$",
                    target=alt_suffix,
                    rule_type="suffix",
                    priority=8
                ))
        
        rules.extend(morphological_rules)
        
        # Generate vocabulary with English-like phonotactics
        base_words = self._get_common_vocabulary(vocab_size)
        
        for word in base_words:
            english_like_word = self._generate_english_like_word(
                word, complexity
            )
            vocabulary[word] = english_like_word
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.ENGLISH_LIKE,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'description': 'English-like language with plausible phonotactics',
                'morphological_patterns': len(morphological_rules),
                'phonotactic_constraints': 'english_based'
            }
        )
    
    def _generate_random_frequency_language(
        self,
        name: str,
        complexity: int,
        vocab_size: int
    ) -> ConstructedLanguage:
        """Generate language with frequency-correlated word lengths."""
        rules = []
        vocabulary = {}
        
        # Generate words with length distribution matching frequency patterns
        generated_words = []
        word_lengths = []
        
        # Sample word lengths according to frequency distribution
        for _ in range(vocab_size):
            length = self._sample_word_length_by_frequency()
            word_lengths.append(length)
            
            # Generate word with frequency-appropriate characteristics
            if length <= 3:
                # Short words: high frequency, simple phonotactics
                word = self._generate_high_frequency_word(length)
            elif length <= 6:
                # Medium words: moderate complexity
                word = self._generate_medium_frequency_word(length, complexity)
            else:
                # Long words: low frequency, complex morphology
                word = self._generate_low_frequency_word(length, complexity)
            
            generated_words.append(word)
        
        # Create vocabulary mapping from common English words to generated words
        base_words = self._get_common_vocabulary(vocab_size)
        
        # Sort generated words by length (shorter = more frequent)
        word_freq_pairs = list(zip(base_words, generated_words, word_lengths))
        word_freq_pairs.sort(key=lambda x: x[2])  # Sort by length
        
        for i, (base_word, generated_word, length) in enumerate(word_freq_pairs):
            vocabulary[base_word] = generated_word
        
        # Add frequency-based transformation rules
        if complexity >= 5:
            # High frequency words get simpler transformations
            freq_rules = [
                LanguageRule(
                    source=r"\\b(a|an|the|is|are|was|were)\\b",
                    target=lambda m: self._generate_high_frequency_word(len(m.group(1))),
                    rule_type="high_frequency",
                    priority=10
                ),
                LanguageRule(
                    source=r"\\b\\w{7,}\\b",  # Long words
                    target=lambda m: self._generate_low_frequency_word(len(m.group(0)), complexity),
                    rule_type="low_frequency",
                    priority=5
                )
            ]
            rules.extend(freq_rules)
        
        return ConstructedLanguage(
            name=name,
            language_type=LanguageType.RANDOM_FREQUENCY,
            rules=sorted(rules, key=lambda r: r.priority, reverse=True),
            vocabulary=vocabulary,
            metadata={
                'complexity': str(complexity),
                'description': 'Frequency-correlated random word generation',
                'length_distribution': json.dumps(dict(Counter(word_lengths))),
                'frequency_correlation': 'zipf_based'
            }
        )
    
    def _generate_pseudo_word(self, length: int) -> str:
        """Generate a pseudo-word of specified length."""
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"
        
        word = ""
        for i in range(length):
            if i % 2 == 0:  # Alternate consonants and vowels
                word += self.rng.choice(consonants)
            else:
                word += self.rng.choice(vowels)
        
        return word[:length]
    
    def _generate_synthetic_word(
        self,
        length: int,
        consonants: List[str],
        vowels: List[str]
    ) -> str:
        """Generate a synthetic word using given phoneme inventory."""
        if length <= 0:
            return ""
        
        # Start with consonant
        word = self.rng.choice(consonants)
        
        for i in range(1, length):
            if word[-1] in vowels:
                # Last was vowel, add consonant
                word += self.rng.choice(consonants)
            else:
                # Last was consonant, add vowel
                word += self.rng.choice(vowels)
        
        return word[:length]
    
    def _apply_phonetic_rules(
        self,
        word: str,
        rules: List[Tuple[str, str]]
    ) -> str:
        """Apply phonetic transformation rules to a word."""
        transformed = word
        for source, target in rules:
            transformed = transformed.replace(source, target)
        return transformed
    
    def _apply_scrambling_rules(
        self,
        word: str,
        rules: List[LanguageRule]
    ) -> str:
        """Apply scrambling rules to a word."""
        transformed = word.lower()
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            if rule.rule_type in ["rotation", "consonant", "vowel"]:
                transformed = transformed.replace(rule.source, rule.target)
        return transformed
    
    def _get_common_vocabulary(self, size: int) -> List[str]:
        """Get a list of common English words for vocabulary generation."""
        # In practice, this would load from a frequency list
        # For now, return a basic set
        common_words = [
            "the", "and", "or", "not", "is", "are", "was", "were", "have", "has",
            "do", "does", "did", "will", "would", "could", "should", "can", "may",
            "if", "then", "else", "when", "where", "how", "what", "who", "why",
            "time", "way", "day", "man", "woman", "child", "people", "world",
            "number", "word", "water", "oil", "food", "money", "house", "car",
            "first", "last", "long", "little", "right", "left", "new", "old",
            "good", "bad", "big", "small", "high", "low", "fast", "slow",
            "make", "get", "give", "take", "go", "come", "see", "know", "think",
            "say", "tell", "ask", "answer", "work", "play", "run", "walk"
        ]
        
        # Extend with generated words if needed
        while len(common_words) < size:
            common_words.append(self._generate_pseudo_word(self.rng.randint(3, 8)))
        
        return common_words[:size]
    
    def _generate_morpheme_variant(self, morpheme: str) -> str:
        """Generate a plausible variant of a morpheme."""
        if len(morpheme) <= 2:
            # Short morphemes: simple consonant/vowel substitution
            result = ""
            for char in morpheme:
                if char in 'aeiou':
                    # Vowel substitution
                    vowel_options = [v for v in 'aeiou' if v != char]
                    result += self.rng.choice(vowel_options)
                else:
                    # Consonant substitution with similar sounds
                    similar_consonants = {
                        'p': ['b', 'f'], 'b': ['p', 'v'], 'f': ['p', 'v'], 'v': ['b', 'f'],
                        't': ['d', 's'], 'd': ['t', 'z'], 's': ['t', 'z'], 'z': ['d', 's'],
                        'k': ['g'], 'g': ['k'], 'm': ['n'], 'n': ['m'], 'l': ['r'], 'r': ['l']
                    }
                    options = similar_consonants.get(char, [char])
                    result += self.rng.choice(options + [char])
            return result
        else:
            # Longer morphemes: more complex transformations
            return self._apply_phonetic_shift(morpheme)
    
    def _apply_phonetic_shift(self, word: str) -> str:
        """Apply systematic phonetic shifts to a word."""
        transformations = [
            ('sh', 'zh'), ('ch', 'j'), ('th', 'f'), ('ng', 'nk'),
            ('ck', 'k'), ('ph', 'f'), ('gh', ''), ('tion', 'shun'),
            ('sion', 'zhun'), ('ough', 'uf')
        ]
        
        result = word
        for source, target in transformations:
            if source in result:
                result = result.replace(source, target)
                break  # Apply only one transformation
        
        return result
    
    def _generate_english_like_word(self, original_word: str, complexity: int) -> str:
        """Generate an English-like word based on the original."""
        target_length = len(original_word)
        
        # Determine syllable count (roughly)
        syllable_count = max(1, target_length // 3)
        
        word = ""
        remaining_length = target_length
        
        for i in range(syllable_count):
            if i == syllable_count - 1:
                # Last syllable: use all remaining length
                syllable_length = remaining_length
            else:
                # Distribute length roughly evenly
                syllable_length = max(1, remaining_length // (syllable_count - i))
            
            syllable = self._generate_english_syllable(syllable_length, complexity)
            word += syllable
            remaining_length -= len(syllable)
            
            if remaining_length <= 0:
                break
        
        # Truncate or extend to match target length
        if len(word) > target_length:
            word = word[:target_length]
        elif len(word) < target_length and word:
            # Extend with vowels or consonants as appropriate
            while len(word) < target_length:
                if word[-1] in 'aeiou':
                    word += self._weighted_choice(self.consonant_weights)
                else:
                    word += self._weighted_choice(self.vowel_weights)
        
        return word or self._generate_english_syllable(target_length, complexity)
    
    def _generate_english_syllable(self, target_length: int, complexity: int) -> str:
        """Generate a single English-like syllable."""
        if target_length <= 1:
            return self._weighted_choice(self.vowel_weights)
        
        syllable = ""
        
        # Onset (beginning consonants)
        if self.rng.random() < 0.8:  # 80% chance of onset
            if complexity >= 5 and target_length >= 4 and self.rng.random() < 0.3:
                # Complex onset
                onset = self.rng.choice(self.english_onsets['two'] + self.english_onsets['three'])
            elif target_length >= 3 and self.rng.random() < 0.5:
                # Two-consonant onset
                onset = self.rng.choice(self.english_onsets['two'])
            else:
                # Single consonant onset
                onset = self._weighted_choice(self.consonant_weights)
            syllable += onset
        
        # Nucleus (vowel)
        if complexity >= 4 and self.rng.random() < 0.2:
            # Diphthong or long vowel
            nucleus = self.rng.choice(self.english_vowels['long'] + self.english_vowels['diphthongs'])
        else:
            # Simple vowel
            nucleus = self._weighted_choice(self.vowel_weights)
        syllable += nucleus
        
        # Coda (ending consonants)
        remaining_length = target_length - len(syllable)
        if remaining_length > 0 and self.rng.random() < 0.6:  # 60% chance of coda
            if complexity >= 6 and remaining_length >= 2 and self.rng.random() < 0.3:
                # Complex coda
                coda_options = [c for c in self.english_codas['two'] + self.english_codas['three'] 
                              if len(c) <= remaining_length]
                if coda_options:
                    coda = self.rng.choice(coda_options)
                else:
                    coda = self._weighted_choice(self.consonant_weights)
            else:
                # Simple coda
                coda = self._weighted_choice(self.consonant_weights)
            syllable += coda
        
        return syllable[:target_length]  # Ensure we don't exceed target length
    
    def _weighted_choice(self, weights: Dict[str, int]) -> str:
        """Choose an item based on weighted probabilities."""
        items = list(weights.keys())
        weights_list = list(weights.values())
        # Convert weights to probabilities for manual weighted selection
        total_weight = sum(weights_list)
        cumulative = 0
        rand_val = self.rng.random() * total_weight
        
        for i, weight in enumerate(weights_list):
            cumulative += weight
            if rand_val <= cumulative:
                return items[i]
        
        # Fallback
        return self.rng.choice(items)
    
    def _sample_word_length_by_frequency(self) -> int:
        """Sample a word length based on frequency distribution."""
        rand_val = self.rng.random()
        
        for length, cumulative_prob in self.length_cumulative:
            if rand_val <= cumulative_prob:
                return length
        
        # Fallback
        return 5
    
    def _generate_high_frequency_word(self, length: int) -> str:
        """Generate a high-frequency word (short, simple)."""
        if length == 1:
            return self.rng.choice(['a', 'i'])
        elif length == 2:
            # Simple CV or VC pattern
            patterns = ['CV', 'VC']
            pattern = self.rng.choice(patterns)
            
            if pattern == 'CV':
                consonant = self._weighted_choice({'t': 5, 'n': 4, 's': 4, 'd': 3, 'f': 2})
                vowel = self._weighted_choice({'o': 3, 'e': 2, 'a': 2})
                return consonant + vowel
            else:  # VC
                vowel = self._weighted_choice({'a': 3, 'i': 2, 'o': 2})
                consonant = self._weighted_choice({'n': 3, 't': 2, 's': 2})
                return vowel + consonant
        else:
            # CVC pattern for 3+ characters
            word = self._weighted_choice({'t': 4, 'f': 3, 'w': 3, 'b': 2})
            word += self._weighted_choice({'e': 4, 'a': 3, 'i': 2})
            
            for i in range(2, length):
                if i % 2 == 0:  # Even positions: prefer consonants
                    word += self._weighted_choice({'t': 3, 'n': 3, 's': 2, 'r': 2})
                else:  # Odd positions: prefer vowels
                    word += self._weighted_choice({'e': 3, 'a': 2, 'i': 2})
            
            return word[:length]
    
    def _generate_medium_frequency_word(self, length: int, complexity: int) -> str:
        """Generate a medium-frequency word with moderate complexity."""
        # Use standard English syllable patterns
        return self._generate_english_like_word('x' * length, min(complexity, 4))
    
    def _generate_low_frequency_word(self, length: int, complexity: int) -> str:
        """Generate a low-frequency word (long, complex morphology)."""
        # Start with a base word
        base_length = max(3, length // 2)
        base_word = self._generate_english_like_word('x' * base_length, complexity)
        
        remaining_length = length - len(base_word)
        
        # Add morphological complexity
        if remaining_length >= 2 and complexity >= 6:
            # Add prefix
            if self.rng.random() < 0.4:
                prefix_length = min(3, remaining_length // 2)
                prefix = self._generate_morpheme_variant(
                    self.rng.choice(self.english_morphology['prefixes'][:prefix_length])
                )
                base_word = prefix + base_word
                remaining_length -= len(prefix)
            
            # Add suffix
            if remaining_length >= 2:
                suffix_length = min(4, remaining_length)
                suffix = self._generate_morpheme_variant(
                    self.rng.choice(self.english_morphology['suffixes'][:suffix_length])
                )
                base_word = base_word + suffix
        
        # Pad or truncate to exact length
        if len(base_word) > length:
            return base_word[:length]
        elif len(base_word) < length:
            # Add filler phonemes
            while len(base_word) < length:
                if base_word[-1] in 'aeiou':
                    base_word += self._weighted_choice({'n': 2, 't': 2, 's': 1, 'r': 1})
                else:
                    base_word += self._weighted_choice({'e': 2, 'a': 1, 'i': 1})
        
        return base_word[:length]
    
    def generate_vocabulary_batch(
        self,
        language: ConstructedLanguage,
        word_list: List[str]
    ) -> Dict[str, str]:
        """
        Generate vocabulary translations for a specific list of words.
        
        Args:
            language: The constructed language to use
            word_list: List of words to translate
            
        Returns:
            Dictionary mapping original words to translated words
        """
        batch_vocabulary = {}
        
        for word in word_list:
            if word in language.vocabulary:
                # Use existing vocabulary mapping
                batch_vocabulary[word] = language.vocabulary[word]
            else:
                # Generate new translation based on language type
                if language.language_type == LanguageType.ENGLISH_LIKE:
                    translation = self._generate_english_like_word(word, 
                        int(language.metadata.get('complexity', '5')))
                elif language.language_type == LanguageType.RANDOM_FREQUENCY:
                    length = len(word)
                    complexity = int(language.metadata.get('complexity', '5'))
                    if length <= 3:
                        translation = self._generate_high_frequency_word(length)
                    elif length <= 6:
                        translation = self._generate_medium_frequency_word(length, complexity)
                    else:
                        translation = self._generate_low_frequency_word(length, complexity)
                elif language.language_type == LanguageType.SYNTHETIC:
                    phoneme_data = json.loads(language.metadata.get('phoneme_inventory', '{}'))
                    consonants = phoneme_data.get('consonants', ['p', 't', 'k', 'm', 'n', 's', 'l', 'r'])
                    vowels = phoneme_data.get('vowels', ['a', 'e', 'i', 'o', 'u'])
                    translation = self._generate_synthetic_word(len(word), consonants, vowels)
                else:
                    # Fallback to pseudo-word generation
                    translation = self._generate_pseudo_word(len(word))
                
                batch_vocabulary[word] = translation
                # Add to language vocabulary for consistency
                language.vocabulary[word] = translation
        
        return batch_vocabulary
    
    def save_language(self, language: ConstructedLanguage, path: Path) -> None:
        """
        Save a constructed language to disk.
        
        Args:
            language: Language to save
            path: Path to save the language file
        """
        language_data = {
            'name': language.name,
            'language_type': language.language_type.value,
            'rules': [
                {
                    'source': rule.source,
                    'target': rule.target,
                    'rule_type': rule.rule_type,
                    'priority': rule.priority,
                    'conditions': rule.conditions
                }
                for rule in language.rules
            ],
            'vocabulary': language.vocabulary,
            'metadata': language.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(language_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Language saved: {path}")
    
    def load_language(self, path: Path) -> ConstructedLanguage:
        """
        Load a constructed language from disk.
        
        Args:
            path: Path to the language file
            
        Returns:
            Loaded ConstructedLanguage object
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        rules = [
            LanguageRule(
                source=rule['source'],
                target=rule['target'],
                rule_type=rule['rule_type'],
                priority=rule['priority'],
                conditions=rule.get('conditions', {})
            )
            for rule in data['rules']
        ]
        
        language = ConstructedLanguage(
            name=data['name'],
            language_type=LanguageType(data['language_type']),
            rules=rules,
            vocabulary=data['vocabulary'],
            metadata=data.get('metadata', {})
        )
        
        self._generated_languages[language.name] = language
        self.logger.info(f"Language loaded: {path}")
        
        return language
    
    def get_language(self, name: str) -> Optional[ConstructedLanguage]:
        """Get a previously generated language by name."""
        return self._generated_languages.get(name)
    
    def list_languages(self) -> List[str]:
        """List names of all generated languages."""
        return list(self._generated_languages.keys())