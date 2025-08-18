"""
Text transformation utilities for benchmark data manipulation.

This module provides various text transformation strategies including:
- Proper noun replacement
- Synonym substitution  
- Vocabulary extraction
- Advanced text manipulation techniques
"""

import re
import random
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
from collections import Counter, defaultdict

# Try to import NLTK components, fall back to simple implementations
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class TransformationResult:
    """Result of a text transformation operation."""
    original_text: str
    transformed_text: str
    replacements: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextTransformer(ABC):
    """Abstract base class for text transformers."""
    
    @abstractmethod
    def transform_text(self, text: str) -> TransformationResult:
        """Transform input text according to the transformer's strategy."""
        pass


class ProperNounSwapper(TextTransformer):
    """
    Swaps proper nouns in text with alternatives using various strategies.
    """
    
    PERSON_NAMES = {
        'male': [
            'Alexander', 'Benjamin', 'Christopher', 'Daniel', 'Edward',
            'Frederick', 'Gabriel', 'Harrison', 'Isaac', 'Jacob',
            'Kenneth', 'Leonardo', 'Michael', 'Nicholas', 'Oliver',
            'Patrick', 'Quinton', 'Robert', 'Samuel', 'Theodore'
        ],
        'female': [
            'Alexandra', 'Beatrice', 'Catherine', 'Diana', 'Elizabeth',
            'Francesca', 'Gabrielle', 'Helena', 'Isabella', 'Josephine',
            'Katherine', 'Lillian', 'Margaret', 'Natalie', 'Olivia',
            'Penelope', 'Quinn', 'Rebecca', 'Stephanie', 'Victoria'
        ],
        'neutral': [
            'Alex', 'Blake', 'Cameron', 'Drew', 'Emery', 'Finley',
            'Gray', 'Harper', 'Indigo', 'Jordan', 'Kai', 'Logan',
            'Morgan', 'Nova', 'Ocean', 'Phoenix', 'Quinn', 'River',
            'Sage', 'Taylor'
        ]
    }
    
    PLACE_NAMES = {
        'cities': [
            'Aetherton', 'Bellwater', 'Crystalport', 'Dragonspire', 'Emberfall',
            'Frostholm', 'Goldenvale', 'Havenbrook', 'Ironbridge', 'Jadecliff',
            'Kingsford', 'Lightspear', 'Moonhaven', 'Northwind', 'Oceanview',
            'Pinerest', 'Quietbrook', 'Ravenshire', 'Silverdale', 'Thornfield'
        ],
        'countries': [
            'Avalonia', 'Byzantia', 'Cordelia', 'Draconia', 'Eldoria',
            'Florentia', 'Galaxia', 'Hibernia', 'Isolde', 'Jormunland',
            'Kyrenia', 'Lumina', 'Mystara', 'Nordica', 'Oceania',
            'Pandora', 'Questra', 'Ruritania', 'Solaria', 'Thalassia'
        ],
        'regions': [
            'Amberlands', 'Blackmoor', 'Crimsonfields', 'Darkwood', 'Evergreen',
            'Frostlands', 'Goldplains', 'Highlands', 'Ironhills', 'Jadevalley',
            'Kingsreach', 'Lowlands', 'Midlands', 'Northlands', 'Oldwood',
            'Pridelands', 'Quietlands', 'Redwood', 'Southlands', 'Westlands'
        ]
    }
    
    ORGANIZATION_NAMES = [
        'Acme Corporation', 'Beacon Industries', 'Crimson Enterprises',
        'Delta Solutions', 'Echo Systems', 'Falcon Technologies',
        'Genesis Group', 'Horizon Holdings', 'Infinity Inc', 'Jupiter Corp',
        'Keystone Company', 'Lunar Dynamics', 'Meridian Labs', 'Nova Corp',
        'Omega Industries', 'Phoenix Systems', 'Quantum Solutions',
        'Radiant Technologies', 'Stellar Enterprises', 'Titan Group'
    ]
    
    def __init__(
        self,
        strategy: str = 'random',
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the proper noun swapper.
        
        Args:
            strategy: Replacement strategy ('random', 'thematic', 'phonetic')
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.logger = logger or logging.getLogger("scramblebench.proper_noun_swapper")
        
        # Build replacement cache
        self._person_replacements = {}
        self._place_replacements = {}
        self._org_replacements = {}
    
    def transform_text(self, text: str) -> TransformationResult:
        """Transform text by replacing proper nouns."""
        # Find proper nouns using regex patterns
        proper_nouns = self._find_proper_nouns(text)
        
        transformed = text
        replacements = {}
        
        for noun in proper_nouns:
            replacement = self._get_replacement(noun)
            if replacement != noun:
                # Use word boundaries for precise replacement
                pattern = r'\b' + re.escape(noun) + r'\b'
                transformed = re.sub(pattern, replacement, transformed)
                replacements[noun] = replacement
        
        return TransformationResult(
            original_text=text,
            transformed_text=transformed,
            replacements=replacements,
            metadata={
                'strategy': self.strategy,
                'proper_nouns_found': len(proper_nouns),
                'replacements_made': len(replacements)
            }
        )
    
    def _find_proper_nouns(self, text: str) -> Set[str]:
        """Find proper nouns in text using various heuristics."""
        proper_nouns = set()
        
        # Simple regex-based approach
        # Look for capitalized words (excluding sentence starts)
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        
        # Filter out sentence-starting words
        sentences = re.split(r'[.!?]+', text)
        sentence_starters = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                first_word = re.match(r'\b[A-Z][a-zA-Z]+\b', sentence)
                if first_word:
                    sentence_starters.add(first_word.group())
        
        # Common function words that might be capitalized
        function_words = {
            'The', 'A', 'An', 'And', 'Or', 'But', 'For', 'Nor', 'So', 'Yet',
            'In', 'On', 'At', 'By', 'With', 'From', 'To', 'Of', 'About',
            'This', 'That', 'These', 'Those', 'His', 'Her', 'Its', 'Their',
            'My', 'Your', 'Our', 'He', 'She', 'It', 'They', 'We', 'You', 'I'
        }
        
        for word in words:
            # Skip if it's a sentence starter or common function word
            if word not in sentence_starters and word not in function_words:
                # Additional check: word should be reasonably long
                if len(word) >= 3:
                    proper_nouns.add(word)
        
        # Look for multi-word proper nouns (simple approach)
        # Match patterns like "New York" or "John Smith"
        multi_word_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        multi_word_matches = re.findall(multi_word_pattern, text)
        for match in multi_word_matches:
            proper_nouns.add(match)
        
        return proper_nouns
    
    def _get_replacement(self, noun: str) -> str:
        """Get replacement for a proper noun based on strategy."""
        if self.strategy == 'random':
            return self._get_random_replacement(noun)
        elif self.strategy == 'thematic':
            return self._get_thematic_replacement(noun)
        elif self.strategy == 'phonetic':
            return self._get_phonetic_replacement(noun)
        else:
            return noun
    
    def _get_random_replacement(self, noun: str) -> str:
        """Get a random replacement from appropriate category."""
        # Simple heuristics to categorize
        if self._looks_like_person_name(noun):
            if noun not in self._person_replacements:
                all_names = (
                    self.PERSON_NAMES['male'] + 
                    self.PERSON_NAMES['female'] + 
                    self.PERSON_NAMES['neutral']
                )
                self._person_replacements[noun] = self.rng.choice(all_names)
            return self._person_replacements[noun]
        
        elif self._looks_like_place_name(noun):
            if noun not in self._place_replacements:
                all_places = (
                    self.PLACE_NAMES['cities'] + 
                    self.PLACE_NAMES['countries'] + 
                    self.PLACE_NAMES['regions']
                )
                self._place_replacements[noun] = self.rng.choice(all_places)
            return self._place_replacements[noun]
        
        else:
            # Default to organization or generic replacement
            if noun not in self._org_replacements:
                self._org_replacements[noun] = self.rng.choice(self.ORGANIZATION_NAMES)
            return self._org_replacements[noun]
    
    def _get_thematic_replacement(self, noun: str) -> str:
        """Get thematically appropriate replacement."""
        # For thematic replacement, we could maintain consistency
        # within domains (e.g., all fantasy names, all sci-fi names)
        # For now, use random but cache for consistency
        return self._get_random_replacement(noun)
    
    def _get_phonetic_replacement(self, noun: str) -> str:
        """Get phonetically similar replacement."""
        # Simple phonetic replacement: preserve length and some sound patterns
        if noun not in self._person_replacements:
            length = len(noun)
            
            if self._looks_like_person_name(noun):
                # Find names of similar length
                all_names = (
                    self.PERSON_NAMES['male'] + 
                    self.PERSON_NAMES['female'] + 
                    self.PERSON_NAMES['neutral']
                )
                similar_length = [name for name in all_names 
                                if abs(len(name) - length) <= 2]
                if similar_length:
                    self._person_replacements[noun] = self.rng.choice(similar_length)
                else:
                    self._person_replacements[noun] = self.rng.choice(all_names)
            
            elif self._looks_like_place_name(noun):
                all_places = (
                    self.PLACE_NAMES['cities'] + 
                    self.PLACE_NAMES['countries'] + 
                    self.PLACE_NAMES['regions']
                )
                similar_length = [place for place in all_places 
                                if abs(len(place) - length) <= 2]
                if similar_length:
                    self._place_replacements[noun] = self.rng.choice(similar_length)
                else:
                    self._place_replacements[noun] = self.rng.choice(all_places)
            
            else:
                similar_length = [org for org in self.ORGANIZATION_NAMES 
                                if abs(len(org) - length) <= 3]
                if similar_length:
                    self._org_replacements[noun] = self.rng.choice(similar_length)
                else:
                    self._org_replacements[noun] = self.rng.choice(self.ORGANIZATION_NAMES)
        
        if noun in self._person_replacements:
            return self._person_replacements[noun]
        elif noun in self._place_replacements:
            return self._place_replacements[noun]
        else:
            return self._org_replacements.get(noun, noun)
    
    def _looks_like_person_name(self, noun: str) -> bool:
        """Heuristic to determine if a noun looks like a person name."""
        # Simple heuristics
        if len(noun.split()) == 1:
            # Single word - could be first name
            return len(noun) >= 3 and len(noun) <= 12
        elif len(noun.split()) == 2:
            # Two words - likely "First Last"
            parts = noun.split()
            return all(len(part) >= 2 and len(part) <= 12 for part in parts)
        return False
    
    def _looks_like_place_name(self, noun: str) -> bool:
        """Heuristic to determine if a noun looks like a place name."""
        # Look for common place name patterns
        place_indicators = ['burg', 'ville', 'ton', 'ford', 'field', 'wood', 'land', 'shire']
        return any(noun.lower().endswith(indicator) for indicator in place_indicators)


class SynonymReplacer(TextTransformer):
    """
    Replaces words with synonyms to create variation while preserving meaning.
    """
    
    # Function words that typically shouldn't be replaced
    FUNCTION_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'they',
        'have', 'had', 'what', 'were', 'been', 'their', 'said', 'each',
        'which', 'she', 'do', 'how', 'if', 'up', 'out', 'many', 'then',
        'them', 'can', 'would', 'my', 'no', 'come', 'could', 'now',
        'than', 'like', 'other', 'into', 'him', 'time', 'two', 'more',
        'these', 'go', 'see', 'only', 'so', 'his', 'when', 'here',
        'who', 'did', 'get', 'may', 'way', 'new', 'day', 'use', 'her',
        'man', 'one', 'our', 'all', 'any', 'me', 'us', 'am', 'does'
    }
    
    def __init__(
        self,
        replacement_rate: float = 0.3,
        preserve_function_words: bool = True,
        preserve_proper_nouns: bool = True,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the synonym replacer.
        
        Args:
            replacement_rate: Proportion of eligible words to replace (0.0-1.0)
            preserve_function_words: Whether to preserve function words
            preserve_proper_nouns: Whether to preserve proper nouns
            seed: Random seed for reproducibility
            logger: Logger instance
        """
        if not (0.0 <= replacement_rate <= 1.0):
            raise ValueError("Replacement rate must be between 0.0 and 1.0")
        
        self.replacement_rate = replacement_rate
        self.preserve_function_words = preserve_function_words
        self.preserve_proper_nouns = preserve_proper_nouns
        self.rng = random.Random(seed)
        self.logger = logger or logging.getLogger("scramblebench.synonym_replacer")
        
        # Cache for synonyms to ensure consistency
        self._synonym_cache = {}
    
    def transform_text(self, text: str) -> TransformationResult:
        """Transform text by replacing words with synonyms."""
        if NLTK_AVAILABLE:
            return self._transform_with_nltk(text)
        else:
            return self._transform_simple(text)
    
    def _transform_with_nltk(self, text: str) -> TransformationResult:
        """Transform text using NLTK for proper linguistic analysis."""
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            transformed_tokens = []
            replacements = {}
            
            for token, pos in pos_tags:
                if self._should_replace_word(token, pos):
                    synonym = self._get_synonym_nltk(token, pos)
                    if synonym and synonym != token.lower():
                        transformed_tokens.append(synonym)
                        replacements[token] = synonym
                    else:
                        transformed_tokens.append(token)
                else:
                    transformed_tokens.append(token)
            
            # Reconstruct text (simple approach - may not preserve exact spacing)
            transformed_text = ' '.join(transformed_tokens)
            
            # Clean up punctuation spacing
            transformed_text = re.sub(r'\s+([,.!?;:])', r'\1', transformed_text)
            transformed_text = re.sub(r'\(\s+', '(', transformed_text)
            transformed_text = re.sub(r'\s+\)', ')', transformed_text)
            
            return TransformationResult(
                original_text=text,
                transformed_text=transformed_text,
                replacements=replacements,
                metadata={
                    'replacement_rate': self.replacement_rate,
                    'method': 'nltk',
                    'words_processed': len(tokens),
                    'words_replaced': len(replacements)
                }
            )
            
        except Exception as e:
            self.logger.warning(f"NLTK processing failed: {e}, falling back to simple method")
            return self._transform_simple(text)
    
    def _transform_simple(self, text: str) -> TransformationResult:
        """Simple transformation without NLTK dependencies."""
        # Basic word tokenization
        words = re.findall(r'\b\w+\b', text)
        replacements = {}
        
        # Simple synonym dictionary (limited)
        simple_synonyms = {
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'mini', 'petite', 'minuscule'],
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'dejected'],
            'walk': ['stroll', 'wander', 'march', 'hike', 'pace'],
            'run': ['jog', 'sprint', 'dash', 'race', 'hurry'],
            'look': ['see', 'view', 'observe', 'watch', 'gaze'],
            'say': ['speak', 'tell', 'state', 'declare', 'mention'],
            'think': ['believe', 'consider', 'ponder', 'contemplate'],
            'make': ['create', 'build', 'construct', 'produce', 'craft'],
            'take': ['grab', 'seize', 'acquire', 'obtain', 'get'],
            'give': ['provide', 'offer', 'present', 'donate', 'grant'],
            'find': ['discover', 'locate', 'uncover', 'detect', 'spot'],
            'know': ['understand', 'realize', 'recognize', 'comprehend'],
            'want': ['desire', 'wish', 'crave', 'need', 'require'],
            'help': ['assist', 'aid', 'support', 'serve', 'back']
        }
        
        transformed = text
        
        for word in words:
            if self._should_replace_word_simple(word):
                word_lower = word.lower()
                if word_lower in simple_synonyms:
                    if word_lower not in self._synonym_cache:
                        self._synonym_cache[word_lower] = self.rng.choice(simple_synonyms[word_lower])
                    
                    synonym = self._synonym_cache[word_lower]
                    
                    # Preserve original case
                    if word.isupper():
                        synonym = synonym.upper()
                    elif word.istitle():
                        synonym = synonym.title()
                    
                    # Replace in text using word boundaries
                    pattern = r'\b' + re.escape(word) + r'\b'
                    transformed = re.sub(pattern, synonym, transformed)
                    replacements[word] = synonym
        
        return TransformationResult(
            original_text=text,
            transformed_text=transformed,
            replacements=replacements,
            metadata={
                'replacement_rate': self.replacement_rate,
                'method': 'simple',
                'words_processed': len(words),
                'words_replaced': len(replacements)
            }
        )
    
    def _should_replace_word(self, word: str, pos: str) -> bool:
        """Determine if a word should be replaced based on POS tag."""
        # Skip if random chance says no
        if self.rng.random() > self.replacement_rate:
            return False
        
        # Skip function words if preserving them
        if self.preserve_function_words and word.lower() in self.FUNCTION_WORDS:
            return False
        
        # Skip proper nouns if preserving them
        if self.preserve_proper_nouns and pos in ['NNP', 'NNPS']:
            return False
        
        # Skip very short words
        if len(word) <= 2:
            return False
        
        # Only replace certain parts of speech
        replaceable_pos = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']
        return any(pos.startswith(p) for p in replaceable_pos)
    
    def _should_replace_word_simple(self, word: str) -> bool:
        """Simple version of word replacement decision."""
        # Skip if random chance says no
        if self.rng.random() > self.replacement_rate:
            return False
        
        # Skip function words if preserving them
        if self.preserve_function_words and word.lower() in self.FUNCTION_WORDS:
            return False
        
        # Skip proper nouns (simple heuristic)
        if self.preserve_proper_nouns and word[0].isupper():
            return False
        
        # Skip very short words
        if len(word) <= 2:
            return False
        
        return True
    
    def _get_synonym_nltk(self, word: str, pos: str) -> Optional[str]:
        """Get synonym using NLTK wordnet."""
        if word.lower() in self._synonym_cache:
            return self._synonym_cache[word.lower()]
        
        try:
            # Map POS tags to wordnet POS
            pos_map = {
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'J': wordnet.ADJ,
                'R': wordnet.ADV
            }
            
            wn_pos = pos_map.get(pos[0], wordnet.NOUN)
            
            # Get synsets for the word
            synsets = wordnet.synsets(word.lower(), pos=wn_pos)
            
            if not synsets:
                return None
            
            # Get all synonyms from all synsets
            synonyms = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word.lower() and len(synonym) > 2:
                        synonyms.add(synonym)
            
            if synonyms:
                synonym = self.rng.choice(list(synonyms))
                self._synonym_cache[word.lower()] = synonym
                
                # Preserve case of original word
                if word.isupper():
                    return synonym.upper()
                elif word.istitle():
                    return synonym.title()
                else:
                    return synonym
            
        except Exception as e:
            self.logger.warning(f"Failed to get synonym for '{word}': {e}")
        
        return None


class VocabularyExtractor:
    """
    Extracts vocabulary from benchmark problems for language generation.
    """
    
    def __init__(
        self,
        min_frequency: int = 2,
        max_vocabulary_size: int = 5000,
        exclude_function_words: bool = True,
        min_word_length: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize vocabulary extractor.
        
        Args:
            min_frequency: Minimum frequency for words to be included
            max_vocabulary_size: Maximum number of words to extract
            exclude_function_words: Whether to exclude common function words
            min_word_length: Minimum length for words to be included
            logger: Logger instance
        """
        self.min_frequency = min_frequency
        self.max_vocabulary_size = max_vocabulary_size
        self.exclude_function_words = exclude_function_words
        self.min_word_length = min_word_length
        self.logger = logger or logging.getLogger("scramblebench.vocabulary_extractor")
        
        # Common function words to exclude
        self.function_words = SynonymReplacer.FUNCTION_WORDS
    
    def extract_from_problems(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract vocabulary from a list of benchmark problems.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            Dictionary containing extracted vocabulary and statistics
        """
        self.logger.info(f"Extracting vocabulary from {len(problems)} problems")
        
        # Collect all text from problems
        all_text = []
        for problem in problems:
            text = self._extract_text_from_problem(problem)
            all_text.append(text)
        
        # Tokenize and count words
        word_counts = Counter()
        total_words = 0
        
        for text in all_text:
            words = self._tokenize_text(text)
            total_words += len(words)
            
            for word in words:
                if self._should_include_word(word):
                    word_counts[word.lower()] += 1
        
        # Filter by frequency and limit size
        filtered_words = {
            word: count for word, count in word_counts.items()
            if count >= self.min_frequency
        }
        
        # Sort by frequency and take top words
        sorted_words = sorted(
            filtered_words.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_vocabulary_size]
        
        # Prepare result
        vocabulary = {
            'words': [word for word, count in sorted_words],
            'word_frequencies': dict(sorted_words),
            'statistics': {
                'total_problems': len(problems),
                'total_words_processed': total_words,
                'unique_words_found': len(word_counts),
                'words_after_filtering': len(filtered_words),
                'final_vocabulary_size': len(sorted_words),
                'min_frequency_threshold': self.min_frequency,
                'most_common_word': sorted_words[0][0] if sorted_words else None,
                'highest_frequency': sorted_words[0][1] if sorted_words else 0
            }
        }
        
        self.logger.info(f"Extracted {len(sorted_words)} words from vocabulary")
        
        return vocabulary
    
    def _extract_text_from_problem(self, problem: Dict[str, Any]) -> str:
        """Extract all text content from a problem dictionary."""
        text_parts = []
        
        def extract_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(problem)
        return ' '.join(text_parts)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback to regex tokenization
        return re.findall(r'\b\w+\b', text)
    
    def _should_include_word(self, word: str) -> bool:
        """Determine if a word should be included in vocabulary."""
        # Check length
        if len(word) < self.min_word_length:
            return False
        
        # Check if it's a function word (if excluding)
        if self.exclude_function_words and word.lower() in self.function_words:
            return False
        
        # Check if it's all digits
        if word.isdigit():
            return False
        
        # Check if it contains only alphabetic characters
        if not word.isalpha():
            return False
        
        return True
    
    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract vocabulary from a benchmark file.
        
        Args:
            file_path: Path to the benchmark file
            
        Returns:
            Dictionary containing extracted vocabulary and statistics
        """
        from scramblebench.utils.data_loader import DataLoader
        
        data_loader = DataLoader()
        problems = data_loader.load_benchmark_file(file_path)
        
        return self.extract_from_problems(problems)
    
    def save_vocabulary(self, vocabulary: Dict[str, Any], output_path: Path) -> None:
        """Save vocabulary to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        self.logger.info(f"Vocabulary saved to {output_path}")
    
    def load_vocabulary(self, vocab_path: Path) -> Dict[str, Any]:
        """Load vocabulary from a JSON file."""
        with open(vocab_path, 'r') as f:
            vocabulary = json.load(f)
        
        self.logger.info(f"Vocabulary loaded from {vocab_path}")
        return vocabulary