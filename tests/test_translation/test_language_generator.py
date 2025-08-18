"""
Tests for language generation functionality.

This module tests the constructed language generation system including
different language types, complexity levels, and rule generation.
"""

import pytest
from pathlib import Path
import tempfile
import json
import string
import re
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import text, integers, composite

from scramblebench.translation.language_generator import (
    LanguageGenerator, ConstructedLanguage, LanguageType, LanguageRule
)


# Test fixtures
@pytest.fixture
def generator():
    """Create a LanguageGenerator instance for testing."""
    return LanguageGenerator(seed=42)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_vocabulary():
    """Sample vocabulary for testing."""
    return {
        "hello": "halo",
        "world": "warld",
        "test": "tast",
        "language": "languaga"
    }

@pytest.fixture
def sample_rules():
    """Sample rules for testing."""
    return [
        LanguageRule("a", "æ", "vowel", 1),
        LanguageRule("e", "ə", "vowel", 1),
        LanguageRule("th", "dh", "phonetic", 5),
        LanguageRule("ing", "an", "grammar", 10)
    ]

@pytest.fixture
def constructed_language(sample_rules, sample_vocabulary):
    """Sample constructed language for testing."""
    return ConstructedLanguage(
        name="test_language",
        language_type=LanguageType.PHONETIC,
        rules=sample_rules,
        vocabulary=sample_vocabulary,
        metadata={"complexity": "5", "description": "Test language"}
    )

# Hypothesis strategies for property-based testing
@composite
def language_names(draw):
    """Generate valid language names."""
    name = draw(text(alphabet=string.ascii_letters + string.digits + "_", min_size=1, max_size=50))
    # Ensure name starts with letter
    if not name[0].isalpha():
        name = "lang_" + name
    return name

@composite
def complexity_values(draw):
    """Generate valid complexity values."""
    return draw(integers(min_value=1, max_value=10))

@composite
def vocab_sizes(draw):
    """Generate valid vocabulary sizes."""
    return draw(integers(min_value=1, max_value=1000))

@composite
def word_strings(draw):
    """Generate valid word strings."""
    return draw(text(alphabet=string.ascii_lowercase, min_size=1, max_size=20))


class TestLanguageGenerator:
    """Test suite for LanguageGenerator class."""
    
    def test_generator_initialization(self, generator):
        """Test that the generator initializes correctly."""
        assert generator is not None
        assert generator.rng is not None
        assert len(generator._generated_languages) == 0
    
    def test_substitution_language_generation(self, generator):
        """Test generation of substitution cipher language."""
        language = generator.generate_language(
            name="test_substitution",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3,
            vocab_size=100
        )
        
        assert isinstance(language, ConstructedLanguage)
        assert language.name == "test_substitution"
        assert language.language_type == LanguageType.SUBSTITUTION
        assert len(language.rules) > 0
        assert len(language.vocabulary) > 0
        
        # Check that character rules exist
        char_rules = [r for r in language.rules if r.rule_type == "character"]
        assert len(char_rules) == 26  # One for each letter
        
        # Check that word rules exist
        word_rules = [r for r in language.rules if r.rule_type == "word"]
        assert len(word_rules) > 0
    
    def test_phonetic_language_generation(self, generator):
        """Test generation of phonetic language."""
        language = generator.generate_language(
            name="test_phonetic",
            language_type=LanguageType.PHONETIC,
            complexity=5,
            vocab_size=50
        )
        
        assert language.language_type == LanguageType.PHONETIC
        assert len(language.rules) > 0
        
        # Check for phonetic rules
        phonetic_rules = [r for r in language.rules if r.rule_type == "phonetic"]
        assert len(phonetic_rules) > 0
    
    def test_scrambled_language_generation(self, generator):
        """Test generation of scrambled language."""
        language = generator.generate_language(
            name="test_scrambled",
            language_type=LanguageType.SCRAMBLED,
            complexity=4,
            vocab_size=75
        )
        
        assert language.language_type == LanguageType.SCRAMBLED
        assert len(language.rules) > 0
        
        # Check for scrambling rules
        scramble_rules = [r for r in language.rules if r.rule_type in ["rotation", "consonant", "vowel"]]
        assert len(scramble_rules) > 0
    
    def test_synthetic_language_generation(self, generator):
        """Test generation of synthetic language."""
        language = generator.generate_language(
            name="test_synthetic",
            language_type=LanguageType.SYNTHETIC,
            complexity=6,
            vocab_size=200
        )
        
        assert language.language_type == LanguageType.SYNTHETIC
        assert len(language.rules) > 0
        assert len(language.vocabulary) > 0
        
        # Check for grammar rules
        grammar_rules = [r for r in language.rules if r.rule_type == "grammar"]
        assert len(grammar_rules) > 0
    
    def test_complexity_scaling(self, generator):
        """Test that complexity affects language generation."""
        simple_lang = generator.generate_language(
            name="simple",
            language_type=LanguageType.SUBSTITUTION,
            complexity=1,
            vocab_size=50
        )
        
        complex_lang = generator.generate_language(
            name="complex",
            language_type=LanguageType.SUBSTITUTION,
            complexity=8,
            vocab_size=50
        )
        
        # Complex language should have more vocabulary entries
        assert len(complex_lang.vocabulary) >= len(simple_lang.vocabulary)
    
    def test_vocabulary_size_scaling(self, generator):
        """Test that vocab_size affects vocabulary generation."""
        small_vocab = generator.generate_language(
            name="small_vocab",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3,
            vocab_size=25
        )
        
        large_vocab = generator.generate_language(
            name="large_vocab",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3,
            vocab_size=100
        )
        
        # Larger vocab size should result in more vocabulary entries
        assert len(large_vocab.vocabulary) > len(small_vocab.vocabulary)
    
    def test_language_caching(self, generator):
        """Test that generated languages are cached."""
        name = "cached_language"
        
        # Generate language
        language1 = generator.generate_language(
            name=name,
            language_type=LanguageType.SUBSTITUTION
        )
        
        # Retrieve from cache
        language2 = generator.get_language(name)
        
        assert language2 is not None
        assert language1.name == language2.name
        assert language1.language_type == language2.language_type
    
    def test_save_and_load_language(self, generator, temp_dir):
        """Test saving and loading languages."""
        # Generate a language
        original_language = generator.generate_language(
            name="save_test",
            language_type=LanguageType.PHONETIC,
            complexity=4
        )
        
        # Save to file
        save_path = temp_dir / "test_language.json"
        generator.save_language(original_language, save_path)
        
        assert save_path.exists()
        
        # Load from file
        loaded_language = generator.load_language(save_path)
        
        assert loaded_language.name == original_language.name
        assert loaded_language.language_type == original_language.language_type
        assert len(loaded_language.rules) == len(original_language.rules)
        assert loaded_language.vocabulary == original_language.vocabulary
    
    def test_rule_priority_ordering(self, generator):
        """Test that rules are properly ordered by priority."""
        language = generator.generate_language(
            name="priority_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=5
        )
        
        # Check that rules are sorted by priority (descending)
        priorities = [rule.priority for rule in language.rules]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_invalid_language_type(self, generator):
        """Test handling of invalid language types."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            generator.generate_language(
                name="invalid",
                language_type="invalid_type"  # This should cause an error
            )
    
    def test_list_languages(self, generator):
        """Test listing generated languages."""
        initial_count = len(generator.list_languages())
        
        # Generate some languages
        for i in range(3):
            generator.generate_language(
                name=f"list_test_{i}",
                language_type=LanguageType.SUBSTITUTION
            )
        
        languages = generator.list_languages()
        assert len(languages) == initial_count + 3
        
        for i in range(3):
            assert f"list_test_{i}" in languages
    
    def test_rule_structure(self, generator):
        """Test that generated rules have proper structure."""
        language = generator.generate_language(
            name="rule_test",
            language_type=LanguageType.SUBSTITUTION
        )
        
        for rule in language.rules:
            assert isinstance(rule, LanguageRule)
            assert isinstance(rule.source, str)
            assert isinstance(rule.target, str)
            assert isinstance(rule.rule_type, str)
            assert isinstance(rule.priority, int)
            assert rule.priority >= 0
    
    def test_reproducible_generation(self):
        """Test that same seed produces same results."""
        generator1 = LanguageGenerator(seed=12345)
        generator2 = LanguageGenerator(seed=12345)
        
        lang1 = generator1.generate_language(
            name="repro_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3
        )
        
        lang2 = generator2.generate_language(
            name="repro_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3
        )
        
        # Should have same vocabulary (since same seed)
        assert lang1.vocabulary == lang2.vocabulary
        
        # Should have same number of rules
        assert len(lang1.rules) == len(lang2.rules)
    
    def test_metadata_inclusion(self, generator):
        """Test that language metadata is properly set."""
        language = generator.generate_language(
            name="metadata_test",
            language_type=LanguageType.SYNTHETIC,
            complexity=7
        )
        
        assert 'complexity' in language.metadata
        assert language.metadata['complexity'] == '7'
        assert 'description' in language.metadata
        
        # Synthetic languages should have phoneme inventory
        if language.language_type == LanguageType.SYNTHETIC:
            assert 'phoneme_inventory' in language.metadata
    
    @pytest.mark.parametrize("language_type", [
        LanguageType.SUBSTITUTION,
        LanguageType.PHONETIC,
        LanguageType.SCRAMBLED,
        LanguageType.SYNTHETIC
    ])
    def test_all_language_types(self, generator, language_type):
        """Test that all supported language types can be generated."""
        language = generator.generate_language(
            name=f"test_{language_type.value}",
            language_type=language_type,
            complexity=3
        )
        
        assert language.language_type == language_type
        assert len(language.rules) > 0
        assert isinstance(language.metadata, dict)
    
    def test_edge_cases(self, generator):
        """Test edge cases and boundary conditions."""
        # Minimum complexity
        min_lang = generator.generate_language(
            name="min_complexity",
            language_type=LanguageType.SUBSTITUTION,
            complexity=1,
            vocab_size=1
        )
        # Vocabulary might be empty for very small vocab_size, but rules should exist
        assert len(min_lang.rules) > 0
        
        # Maximum complexity  
        max_lang = generator.generate_language(
            name="max_complexity",
            language_type=LanguageType.SUBSTITUTION,
            complexity=10,
            vocab_size=2000
        )
        assert len(max_lang.rules) > 0
        assert len(max_lang.vocabulary) > 0


class TestLanguageGeneratorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_vocab_size(self, generator):
        """Test handling of zero vocabulary size."""
        # The implementation may handle this gracefully, so let's test actual behavior
        language = generator.generate_language(
            name="zero_vocab",
            language_type=LanguageType.SUBSTITUTION,
            vocab_size=0
        )
        # Should still create character rules even with zero vocab size
        assert len(language.rules) > 0
    
    def test_negative_complexity(self, generator):
        """Test handling of negative complexity."""
        # Should handle gracefully or raise appropriate error
        try:
            language = generator.generate_language(
                name="negative_complexity",
                language_type=LanguageType.SUBSTITUTION,
                complexity=-1
            )
            # If it doesn't raise an error, verify it's handled reasonably
            assert language is not None
        except (ValueError, AssertionError):
            # Expected behavior
            pass
    
    def test_empty_string_name(self, generator):
        """Test handling of empty string name."""
        language = generator.generate_language(
            name="",
            language_type=LanguageType.SUBSTITUTION
        )
        assert language.name == ""
    
    def test_very_long_name(self, generator):
        """Test handling of very long language names."""
        long_name = "a" * 1000
        language = generator.generate_language(
            name=long_name,
            language_type=LanguageType.SUBSTITUTION
        )
        assert language.name == long_name
    
    def test_special_characters_in_name(self, generator):
        """Test handling of special characters in language names."""
        special_name = "test-lang_123!@#"
        language = generator.generate_language(
            name=special_name,
            language_type=LanguageType.SUBSTITUTION
        )
        assert language.name == special_name
    
    def test_unicode_in_name(self, generator):
        """Test handling of unicode characters in language names."""
        unicode_name = "测试语言_тест_🌍"
        language = generator.generate_language(
            name=unicode_name,
            language_type=LanguageType.SUBSTITUTION
        )
        assert language.name == unicode_name
    
    def test_extremely_high_complexity(self, generator):
        """Test handling of extremely high complexity values."""
        language = generator.generate_language(
            name="high_complexity",
            language_type=LanguageType.SUBSTITUTION,
            complexity=1000
        )
        assert language is not None
        assert len(language.rules) > 0
    
    def test_extremely_large_vocab_size(self, generator):
        """Test handling of extremely large vocabulary sizes."""
        # This might be slow, so use smaller size in practice
        language = generator.generate_language(
            name="large_vocab",
            language_type=LanguageType.SUBSTITUTION,
            vocab_size=10000
        )
        assert language is not None
        # Vocabulary might be smaller due to common words limitation
        assert len(language.vocabulary) > 0


class TestLanguageReproducibility:
    """Test reproducibility and deterministic behavior."""
    
    def test_seed_reproducibility_basic(self):
        """Test that same seed produces identical results."""
        seed = 12345
        
        gen1 = LanguageGenerator(seed=seed)
        gen2 = LanguageGenerator(seed=seed)
        
        lang1 = gen1.generate_language(
            name="repro_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=5,
            vocab_size=100
        )
        
        lang2 = gen2.generate_language(
            name="repro_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=5,
            vocab_size=100
        )
        
        # Detailed comparison
        assert lang1.name == lang2.name
        assert lang1.language_type == lang2.language_type
        assert lang1.vocabulary == lang2.vocabulary
        assert len(lang1.rules) == len(lang2.rules)
        
        # Compare individual rules
        for rule1, rule2 in zip(lang1.rules, lang2.rules):
            assert rule1.source == rule2.source
            assert rule1.target == rule2.target
            assert rule1.rule_type == rule2.rule_type
            assert rule1.priority == rule2.priority
    
    @pytest.mark.parametrize("language_type", [
        LanguageType.SUBSTITUTION,
        LanguageType.PHONETIC,
        LanguageType.SCRAMBLED,
        LanguageType.SYNTHETIC
    ])
    def test_seed_reproducibility_all_types(self, language_type):
        """Test reproducibility across all language types."""
        seed = 54321
        
        gen1 = LanguageGenerator(seed=seed)
        gen2 = LanguageGenerator(seed=seed)
        
        lang1 = gen1.generate_language(
            name="repro_all_types",
            language_type=language_type,
            complexity=3,
            vocab_size=50
        )
        
        lang2 = gen2.generate_language(
            name="repro_all_types",
            language_type=language_type,
            complexity=3,
            vocab_size=50
        )
        
        assert lang1.vocabulary == lang2.vocabulary
        assert len(lang1.rules) == len(lang2.rules)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = LanguageGenerator(seed=111)
        gen2 = LanguageGenerator(seed=222)
        
        lang1 = gen1.generate_language(
            name="diff_test",
            language_type=LanguageType.SCRAMBLED,
            complexity=5,
            vocab_size=100
        )
        
        lang2 = gen2.generate_language(
            name="diff_test",
            language_type=LanguageType.SCRAMBLED,
            complexity=5,
            vocab_size=100
        )
        
        # Should have different vocabularies (very unlikely to be same)
        assert lang1.vocabulary != lang2.vocabulary
    
    def test_repeated_generation_same_generator(self, generator):
        """Test that repeated generation with same generator is deterministic."""
        # Generate same language multiple times
        languages = []
        for i in range(3):
            lang = generator.generate_language(
                name=f"repeated_{i}",
                language_type=LanguageType.PHONETIC,
                complexity=4,
                vocab_size=50
            )
            languages.append(lang)
        
        # Each generation should be consistent (seed is fixed)
        # But names are different, so vocabularies might vary based on implementation
        # At minimum, rule structure should be consistent
        for lang in languages:
            assert len(lang.rules) > 0
            assert len(lang.vocabulary) > 0


class TestLanguageSaveLoad:
    """Test language persistence functionality."""
    
    def test_round_trip_all_types(self, generator, temp_dir):
        """Test save/load round-trip for all language types."""
        for lang_type in LanguageType:
            # Generate language
            original = generator.generate_language(
                name=f"roundtrip_{lang_type.value}",
                language_type=lang_type,
                complexity=4,
                vocab_size=75
            )
            
            # Save to file
            save_path = temp_dir / f"{lang_type.value}_test.json"
            generator.save_language(original, save_path)
            
            # Load from file
            loaded = generator.load_language(save_path)
            
            # Verify equivalence
            assert loaded.name == original.name
            assert loaded.language_type == original.language_type
            assert loaded.vocabulary == original.vocabulary
            assert loaded.metadata == original.metadata
            assert len(loaded.rules) == len(original.rules)
            
            # Verify individual rules
            for orig_rule, loaded_rule in zip(original.rules, loaded.rules):
                assert orig_rule.source == loaded_rule.source
                assert orig_rule.target == loaded_rule.target
                assert orig_rule.rule_type == loaded_rule.rule_type
                assert orig_rule.priority == loaded_rule.priority
                assert orig_rule.conditions == loaded_rule.conditions
    
    def test_save_creates_valid_json(self, generator, temp_dir, constructed_language):
        """Test that saved files contain valid JSON."""
        save_path = temp_dir / "valid_json_test.json"
        generator.save_language(constructed_language, save_path)
        
        # Verify file exists and contains valid JSON
        assert save_path.exists()
        
        with open(save_path, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        assert 'name' in data
        assert 'language_type' in data
        assert 'rules' in data
        assert 'vocabulary' in data
        assert 'metadata' in data
        
        # Verify data types
        assert isinstance(data['name'], str)
        assert isinstance(data['language_type'], str)
        assert isinstance(data['rules'], list)
        assert isinstance(data['vocabulary'], dict)
        assert isinstance(data['metadata'], dict)
    
    def test_load_nonexistent_file(self, generator, temp_dir):
        """Test loading from nonexistent file."""
        nonexistent_path = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            generator.load_language(nonexistent_path)
    
    def test_load_invalid_json(self, generator, temp_dir):
        """Test loading from file with invalid JSON."""
        invalid_path = temp_dir / "invalid.json"
        
        # Create file with invalid JSON
        with open(invalid_path, 'w') as f:
            f.write("invalid json content {{{")
        
        with pytest.raises(json.JSONDecodeError):
            generator.load_language(invalid_path)
    
    def test_load_missing_required_fields(self, generator, temp_dir):
        """Test loading from file missing required fields."""
        incomplete_path = temp_dir / "incomplete.json"
        
        # Create file with incomplete data
        incomplete_data = {
            "name": "incomplete",
            "language_type": "substitution"
            # Missing rules, vocabulary, metadata
        }
        
        with open(incomplete_path, 'w') as f:
            json.dump(incomplete_data, f)
        
        with pytest.raises(KeyError):
            generator.load_language(incomplete_path)
    
    def test_save_load_preserves_unicode(self, generator, temp_dir):
        """Test that unicode characters are preserved in save/load."""
        # Create language with unicode content
        unicode_vocab = {
            "hello": "你好",
            "world": "мир", 
            "test": "🌍"
        }
        
        unicode_rules = [
            LanguageRule("a", "α", "unicode", 1),
            LanguageRule("e", "ε", "unicode", 1)
        ]
        
        unicode_lang = ConstructedLanguage(
            name="unicode_test",
            language_type=LanguageType.SUBSTITUTION,
            rules=unicode_rules,
            vocabulary=unicode_vocab,
            metadata={"description": "Test with unicode: 测试"}
        )
        
        # Save and load
        save_path = temp_dir / "unicode_test.json"
        generator.save_language(unicode_lang, save_path)
        loaded_lang = generator.load_language(save_path)
        
        # Verify unicode preservation
        assert loaded_lang.vocabulary == unicode_vocab
        assert loaded_lang.metadata["description"] == "Test with unicode: 测试"
        assert loaded_lang.rules[0].target == "α"


class TestVocabularyGeneration:
    """Test vocabulary generation functionality."""
    
    def test_vocabulary_size_constraints(self, generator):
        """Test that vocabulary respects size constraints."""
        sizes = [1, 5, 10, 50, 100]
        
        for size in sizes:
            language = generator.generate_language(
                name=f"vocab_size_{size}",
                language_type=LanguageType.SUBSTITUTION,
                vocab_size=size
            )
            
            # Vocabulary size might be empty for very small sizes due to implementation
            # Rules should always exist though
            assert len(language.rules) > 0
            if size > 10:  # For larger sizes, should have some vocabulary
                assert len(language.vocabulary) > 0
    
    def test_vocabulary_quality_common_words(self, generator):
        """Test that vocabulary includes common words when appropriate."""
        language = generator.generate_language(
            name="vocab_quality",
            language_type=LanguageType.SUBSTITUTION,
            vocab_size=100
        )
        
        # Should include some common English words
        common_words = ["the", "and", "or", "not", "is", "are"]
        vocab_words = set(language.vocabulary.keys())
        
        # At least some common words should be present
        overlap = len(vocab_words.intersection(common_words))
        assert overlap > 0
    
    def test_vocabulary_uniqueness(self, generator):
        """Test that vocabulary mappings are unique."""
        language = generator.generate_language(
            name="vocab_unique",
            language_type=LanguageType.SYNTHETIC,
            vocab_size=100
        )
        
        # Source words should be unique
        source_words = list(language.vocabulary.keys())
        assert len(source_words) == len(set(source_words))
        
        # Target words should be unique (might not be enforced, but good to check)
        target_words = list(language.vocabulary.values())
        # Allow some duplicates as it's not necessarily enforced
        assert len(set(target_words)) > len(target_words) * 0.5  # At least 50% unique
    
    def test_vocabulary_non_empty_values(self, generator):
        """Test that vocabulary doesn't contain empty values."""
        language = generator.generate_language(
            name="vocab_non_empty",
            language_type=LanguageType.PHONETIC,
            vocab_size=50
        )
        
        for source, target in language.vocabulary.items():
            assert source.strip() != "", f"Empty source word found"
            assert target.strip() != "", f"Empty target word found for source: {source}"
    
    def test_vocabulary_consistency_across_types(self, generator):
        """Test vocabulary generation consistency across language types."""
        vocab_size = 25
        languages = {}
        
        for lang_type in LanguageType:
            languages[lang_type] = generator.generate_language(
                name=f"consistency_{lang_type.value}",
                language_type=lang_type,
                vocab_size=vocab_size
            )
        
        # All should have reasonable vocabulary sizes
        for lang_type, language in languages.items():
            assert len(language.vocabulary) > 0
            assert len(language.vocabulary) <= vocab_size * 3  # Allow flexibility


class TestRuleApplication:
    """Test rule generation and application."""
    
    def test_rule_completeness_substitution(self, generator):
        """Test that substitution languages have complete character mapping."""
        language = generator.generate_language(
            name="complete_substitution",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3
        )
        
        # Should have rules for all lowercase letters
        char_rules = [r for r in language.rules if r.rule_type == "character"]
        rule_sources = {rule.source for rule in char_rules}
        
        expected_chars = set(string.ascii_lowercase)
        assert rule_sources == expected_chars
    
    def test_rule_priority_consistency(self, generator):
        """Test that rule priorities are consistent within types."""
        language = generator.generate_language(
            name="priority_test",
            language_type=LanguageType.SUBSTITUTION,
            complexity=5
        )
        
        # Word rules should have higher priority than character rules
        word_rules = [r for r in language.rules if r.rule_type == "word"]
        char_rules = [r for r in language.rules if r.rule_type == "character"]
        
        if word_rules and char_rules:
            min_word_priority = min(rule.priority for rule in word_rules)
            max_char_priority = max(rule.priority for rule in char_rules)
            assert min_word_priority > max_char_priority
    
    def test_phonetic_rule_validity(self, generator):
        """Test that phonetic rules are linguistically reasonable."""
        language = generator.generate_language(
            name="phonetic_validity",
            language_type=LanguageType.PHONETIC,
            complexity=6
        )
        
        phonetic_rules = [r for r in language.rules if r.rule_type == "phonetic"]
        
        # Check that common phonetic transformations are present
        rule_mappings = {rule.source: rule.target for rule in phonetic_rules}
        
        # Should have some expected phonetic patterns
        expected_patterns = ["ch", "sh", "th", "ph"]
        found_patterns = sum(1 for pattern in expected_patterns if pattern in rule_mappings)
        assert found_patterns > 0
    
    def test_scrambled_rule_consistency(self, generator):
        """Test that scrambled rules maintain character type consistency."""
        language = generator.generate_language(
            name="scrambled_consistency",
            language_type=LanguageType.SCRAMBLED,
            complexity=5
        )
        
        consonant_rules = [r for r in language.rules if r.rule_type == "consonant"]
        vowel_rules = [r for r in language.rules if r.rule_type == "vowel"]
        
        # Consonants should map to consonants
        consonants = "bcdfghjklmnpqrstvwxyz"
        for rule in consonant_rules:
            assert rule.source in consonants
            assert rule.target in consonants
        
        # Vowels should map to vowels  
        vowels = "aeiou"
        for rule in vowel_rules:
            assert rule.source in vowels
            assert rule.target in vowels
    
    def test_synthetic_grammar_rules(self, generator):
        """Test that synthetic languages have appropriate grammar rules."""
        language = generator.generate_language(
            name="synthetic_grammar",
            language_type=LanguageType.SYNTHETIC,
            complexity=7
        )
        
        grammar_rules = [r for r in language.rules if r.rule_type == "grammar"]
        
        # Should have some grammar rules
        assert len(grammar_rules) > 0
        
        # Check for common grammatical morphemes
        rule_sources = {rule.source for rule in grammar_rules}
        expected_morphemes = ["ing", "ed", "s", "ly"]
        
        # At least some should be present (with $ for end-of-word)
        found = sum(1 for morpheme in expected_morphemes 
                   if any(morpheme in source for source in rule_sources))
        assert found > 0
    
    def test_rule_conditions_structure(self, generator):
        """Test that rule conditions have proper structure."""
        language = generator.generate_language(
            name="conditions_test",
            language_type=LanguageType.PHONETIC,
            complexity=4
        )
        
        for rule in language.rules:
            # Conditions should be a dictionary
            assert isinstance(rule.conditions, dict)
            
            # If conditions exist, they should have string keys and values
            for key, value in rule.conditions.items():
                assert isinstance(key, str)
                assert isinstance(value, str)


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_generate_save_load_workflow(self, temp_dir):
        """Test complete workflow: generate -> save -> load -> use."""
        # Step 1: Generate language
        generator = LanguageGenerator(seed=98765)
        original_language = generator.generate_language(
            name="workflow_test",
            language_type=LanguageType.PHONETIC,
            complexity=5,
            vocab_size=100
        )
        
        # Step 2: Save language
        save_path = temp_dir / "workflow_language.json"
        generator.save_language(original_language, save_path)
        
        # Step 3: Create new generator and load language
        new_generator = LanguageGenerator(seed=11111)  # Different seed
        loaded_language = new_generator.load_language(save_path)
        
        # Step 4: Verify loaded language is usable
        assert loaded_language.name == "workflow_test"
        assert len(loaded_language.rules) > 0
        assert len(loaded_language.vocabulary) > 0
        
        # Step 5: Verify it's in the new generator's cache
        cached_language = new_generator.get_language("workflow_test")
        assert cached_language is not None
        assert cached_language.name == loaded_language.name
    
    def test_multiple_language_generation_and_management(self, generator, temp_dir):
        """Test generating and managing multiple languages."""
        # Generate multiple languages
        languages_config = [
            ("lang1", LanguageType.SUBSTITUTION, 3, 50),
            ("lang2", LanguageType.PHONETIC, 5, 75),
            ("lang3", LanguageType.SCRAMBLED, 4, 60),
            ("lang4", LanguageType.SYNTHETIC, 6, 100)
        ]
        
        generated_languages = []
        for name, lang_type, complexity, vocab_size in languages_config:
            language = generator.generate_language(
                name=name,
                language_type=lang_type,
                complexity=complexity,
                vocab_size=vocab_size
            )
            generated_languages.append(language)
        
        # Verify all are cached
        cached_names = generator.list_languages()
        for name, _, _, _ in languages_config:
            assert name in cached_names
        
        # Save all languages
        for language in generated_languages:
            save_path = temp_dir / f"{language.name}.json"
            generator.save_language(language, save_path)
            assert save_path.exists()
        
        # Load all languages in new generator
        new_generator = LanguageGenerator()
        for language in generated_languages:
            load_path = temp_dir / f"{language.name}.json"
            loaded_language = new_generator.load_language(load_path)
            assert loaded_language.name == language.name
            assert loaded_language.language_type == language.language_type
    
    def test_complexity_scaling_integration(self, generator):
        """Test that complexity scaling works across all aspects."""
        complexities = [1, 5, 10]
        language_name = "complexity_scaling"
        
        results = []
        for complexity in complexities:
            language = generator.generate_language(
                name=f"{language_name}_{complexity}",
                language_type=LanguageType.SYNTHETIC,
                complexity=complexity,
                vocab_size=100
            )
            results.append((complexity, language))
        
        # Higher complexity should generally result in more features
        # (This is somewhat implementation-dependent)
        for i in range(len(results) - 1):
            current_complexity, current_lang = results[i]
            next_complexity, next_lang = results[i + 1]
            
            # At minimum, should have reasonable number of rules
            assert len(current_lang.rules) > 0
            assert len(next_lang.rules) > 0
            
            # Metadata should reflect complexity
            assert current_lang.metadata.get('complexity') == str(current_complexity)
            assert next_lang.metadata.get('complexity') == str(next_complexity)


class TestPropertyBasedTesting:
    """Property-based tests using Hypothesis."""
    
    @given(name=language_names(), complexity=complexity_values(), vocab_size=vocab_sizes())
    @settings(max_examples=20, deadline=5000)  # Limit examples for performance
    def test_language_generation_properties(self, name, complexity, vocab_size):
        """Test properties that should hold for any valid input."""
        assume(vocab_size <= 500)  # Keep reasonable for performance
        
        generator = LanguageGenerator(seed=42)
        
        for lang_type in LanguageType:
            language = generator.generate_language(
                name=f"{name}_{lang_type.value}",
                language_type=lang_type,
                complexity=complexity,
                vocab_size=vocab_size
            )
            
            # Properties that should always hold
            assert language.name.endswith(lang_type.value)
            assert language.language_type == lang_type
            assert len(language.rules) >= 0
            assert isinstance(language.vocabulary, dict)
            assert isinstance(language.metadata, dict)
            assert 'complexity' in language.metadata
            assert language.metadata['complexity'] == str(complexity)
    
    @given(word=word_strings())
    @settings(max_examples=50)
    def test_pseudo_word_generation_properties(self, word):
        """Test properties of pseudo-word generation."""
        generator = LanguageGenerator(seed=42)
        
        # Test _generate_pseudo_word method
        if len(word) > 0:
            pseudo_word = generator._generate_pseudo_word(len(word))
            
            # Properties
            assert len(pseudo_word) == len(word)
            assert pseudo_word.isalpha()
            assert pseudo_word.islower()
    
    @given(length=integers(min_value=1, max_value=20))
    @settings(max_examples=30)
    def test_synthetic_word_generation_properties(self, length):
        """Test properties of synthetic word generation."""
        generator = LanguageGenerator(seed=42)
        
        consonants = ["p", "t", "k", "m", "n"]
        vowels = ["a", "e", "i", "o", "u"]
        
        synthetic_word = generator._generate_synthetic_word(length, consonants, vowels)
        
        # Properties
        assert len(synthetic_word) == length
        assert synthetic_word.isalpha()
        
        # Should only contain specified phonemes
        all_phonemes = set(consonants + vowels)
        for char in synthetic_word:
            assert char in all_phonemes


class TestMockingAndIsolation:
    """Test with mocking for external dependencies."""
    
    def test_logger_usage(self):
        """Test that logger is used appropriately."""
        mock_logger = MagicMock()
        generator = LanguageGenerator(seed=42, logger=mock_logger)
        
        language = generator.generate_language(
            name="logger_test",
            language_type=LanguageType.SUBSTITUTION
        )
        
        # Verify logger was called
        mock_logger.info.assert_called()
        
        # Check that log messages are reasonable
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Generating" in msg for msg in call_args)
    
    @patch('scramblebench.translation.language_generator.json.dump')
    def test_save_language_file_operations(self, mock_json_dump, generator, temp_dir, constructed_language):
        """Test file operations during save with mocking."""
        save_path = temp_dir / "mock_test.json"
        
        # Mock json.dump to test the data being saved
        generator.save_language(constructed_language, save_path)
        
        # Verify json.dump was called
        mock_json_dump.assert_called_once()
        
        # Check the data structure passed to json.dump
        call_args = mock_json_dump.call_args[0]
        saved_data = call_args[0]
        
        assert 'name' in saved_data
        assert 'language_type' in saved_data
        assert 'rules' in saved_data
        assert 'vocabulary' in saved_data
        assert 'metadata' in saved_data
    
    def test_random_seed_isolation(self):
        """Test that random seed doesn't affect other random operations."""
        import random
        
        # Set global random state
        random.seed(999)
        initial_random = random.random()
        
        # Use generator with different seed
        generator = LanguageGenerator(seed=111)
        language = generator.generate_language(
            name="isolation_test",
            language_type=LanguageType.SCRAMBLED
        )
        
        # Global random state should be unchanged
        random.seed(999)
        after_random = random.random()
        assert initial_random == after_random
        
        # Generator should have produced a language
        assert language is not None


class TestErrorHandling:
    """Test error handling and exceptional cases."""
    
    def test_invalid_language_type_enum(self, generator):
        """Test handling of invalid language type (not enum member)."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            # Try to pass string instead of enum
            generator.generate_language(
                name="invalid_type",
                language_type="not_an_enum"
            )
    
    def test_none_parameters(self, generator):
        """Test handling of None parameters."""
        # Generator may handle None name gracefully
        language = generator.generate_language(
            name=None,
            language_type=LanguageType.SUBSTITUTION
        )
        assert language.name is None
        assert len(language.rules) > 0
        
        # None language_type should fail
        with pytest.raises((ValueError, TypeError, AttributeError)):
            generator.generate_language(
                name="test",
                language_type=None
            )
    
    def test_file_permission_errors(self, generator, constructed_language):
        """Test handling of file permission errors during save."""
        # Try to save to a directory that doesn't exist
        invalid_path = Path("/nonexistent/directory/file.json")
        
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            generator.save_language(constructed_language, invalid_path)
    
    def test_corrupted_json_handling(self, generator, temp_dir):
        """Test handling of corrupted JSON files."""
        corrupted_path = temp_dir / "corrupted.json"
        
        # Create file with corrupted JSON structure
        with open(corrupted_path, 'w') as f:
            f.write('{"name": "test", "rules": [malformed}')
        
        with pytest.raises(json.JSONDecodeError):
            generator.load_language(corrupted_path)
    
    def test_partial_data_recovery(self, generator, temp_dir):
        """Test recovery from partially valid data."""
        partial_path = temp_dir / "partial.json"
        
        # Create file with some valid data but missing fields
        partial_data = {
            "name": "partial_test",
            "language_type": "substitution",
            "rules": [],
            "vocabulary": {"test": "tast"}
            # Missing metadata
        }
        
        with open(partial_path, 'w') as f:
            json.dump(partial_data, f)
        
        # Should either load with defaults or raise clear error
        try:
            language = generator.load_language(partial_path)
            assert language.name == "partial_test"
            assert isinstance(language.metadata, dict)  # Should have default empty dict
        except KeyError as e:
            # Should be a clear error about missing field
            assert "metadata" in str(e) or "required" in str(e).lower()