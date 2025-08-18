"""
Tests for document transformation functionality.

This module tests the document transformation system including
different transformation types, section parsing, and preservation
of document structure and information.
"""

import pytest
from pathlib import Path
import tempfile

from scramblebench.longcontext.document_transformer import (
    DocumentTransformer, TransformedDocument, TransformationType, DocumentSection
)
from scramblebench.translation.language_generator import (
    LanguageGenerator, LanguageType
)
from scramblebench.translation.translator import ProblemTranslator


class TestDocumentTransformer:
    """Test suite for DocumentTransformer class."""
    
    @pytest.fixture
    def transformer(self):
        """Create a DocumentTransformer instance for testing."""
        return DocumentTransformer()
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return """
# Introduction

This is a sample document for testing the document transformation system.
It contains multiple paragraphs and different types of content.

## First Section

The first section discusses important concepts. It mentions specific numbers
like 42 and proper nouns like Python and OpenAI. This paragraph contains
detailed information about the topic.

## Second Section

The second section provides additional context. It includes:

- First item in the list
- Second item with details
- Third item for completeness

### Subsection

This is a subsection with more specific information. The content here
is technical and contains references to various entities and concepts.

## Conclusion

In conclusion, this document serves as a comprehensive example for testing
document transformation capabilities across different scenarios.
        """.strip()
    
    @pytest.fixture
    def constructed_language(self):
        """Create a constructed language for translation tests."""
        generator = LanguageGenerator(seed=42)
        return generator.generate_language(
            name="test_language",
            language_type=LanguageType.SUBSTITUTION,
            complexity=3
        )
    
    def test_transformer_initialization(self, transformer):
        """Test that the transformer initializes correctly."""
        assert transformer is not None
        assert transformer.translator is not None
        assert len(transformer._transformation_cache) == 0
    
    def test_document_section_parsing(self, transformer, sample_document):
        """Test parsing document into logical sections."""
        sections = transformer._parse_document_sections(sample_document)
        
        assert len(sections) > 0
        
        # Should have headings and paragraphs
        section_types = [s.section_type for s in sections]
        assert "heading" in section_types
        assert "paragraph" in section_types
        
        # Check section positions
        positions = [s.position for s in sections]
        assert positions == sorted(positions)  # Should be in order
        
        # Check importance scores
        for section in sections:
            assert 0 <= section.importance <= 1
    
    def test_section_type_classification(self, transformer):
        """Test classification of different section types."""
        test_cases = [
            ("# Main Heading", "heading"),
            ("## Subheading", "heading"),
            ("UPPERCASE HEADING", "heading"),
            ("- First item\n- Second item", "list"),
            ("1. First item\n2. Second item", "numbered_list"),
            ('"This is a quote"', "quote"),
            ("Regular paragraph text", "paragraph")
        ]
        
        for text, expected_type in test_cases:
            section_type = transformer._classify_section_type(text)
            assert section_type == expected_type
    
    def test_translation_transformation(self, transformer, sample_document, constructed_language):
        """Test document transformation using translation."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.TRANSLATION,
            language=constructed_language,
            preserve_structure=True,
            preserve_entities=True
        )
        
        assert isinstance(transformed_doc, TransformedDocument)
        assert transformed_doc.transformation_type == TransformationType.TRANSLATION
        assert len(transformed_doc.sections) > 0
        assert len(transformed_doc.transformation_map) > 0
        
        # Original and transformed should be different
        assert transformed_doc.original_document != transformed_doc.transformed_document
        
        # Should preserve structure (roughly same length)
        original_lines = len(transformed_doc.original_document.split('\n'))
        transformed_lines = len(transformed_doc.transformed_document.split('\n'))
        assert abs(original_lines - transformed_lines) <= 2
    
    def test_paraphrase_transformation(self, transformer, sample_document):
        """Test document transformation using paraphrasing."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.PARAPHRASE,
            preserve_structure=True,
            preserve_entities=True
        )
        
        assert transformed_doc.transformation_type == TransformationType.PARAPHRASE
        assert len(transformed_doc.transformation_map) >= 0  # May have substitutions
        
        # Document should be somewhat different
        if len(transformed_doc.transformation_map) > 0:
            assert transformed_doc.original_document != transformed_doc.transformed_document
    
    def test_structural_transformation(self, transformer, sample_document):
        """Test document transformation using structural changes."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.STRUCTURAL,
            preserve_structure=False,  # Allow structural changes
            preserve_entities=True
        )
        
        assert transformed_doc.transformation_type == TransformationType.STRUCTURAL
        
        # Should have reordered sections
        original_sections = transformer._parse_document_sections(sample_document)
        transformed_sections = transformed_doc.sections
        
        assert len(transformed_sections) == len(original_sections)
    
    def test_hybrid_transformation(self, transformer, sample_document, constructed_language):
        """Test hybrid transformation combining multiple methods."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.HYBRID,
            language=constructed_language,
            preserve_structure=True,
            preserve_entities=True
        )
        
        assert transformed_doc.transformation_type == TransformationType.HYBRID
        assert len(transformed_doc.transformation_map) > 0
        
        # Should have metadata about transformations applied
        assert 'transformations' in transformed_doc.metadata
        assert isinstance(transformed_doc.metadata['transformations'], list)
    
    def test_transformation_without_language(self, transformer, sample_document):
        """Test transformation that doesn't require a language."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.STRUCTURAL,
            preserve_structure=True,
            preserve_entities=True
        )
        
        assert transformed_doc is not None
        assert transformed_doc.transformation_type == TransformationType.STRUCTURAL
    
    def test_translation_requires_language(self, transformer, sample_document):
        """Test that translation transformation requires a language."""
        with pytest.raises(ValueError, match="Language is required"):
            transformer.transform_document(
                document=sample_document,
                transformation_type=TransformationType.TRANSLATION,
                language=None  # Should cause error
            )
    
    def test_importance_calculation(self, transformer):
        """Test section importance calculation."""
        test_sections = [
            ("# Important Heading", "heading"),
            ("This is a very long paragraph with lots of detailed information about the topic.", "paragraph"),
            ("Short text", "paragraph"),
            ("Section with numbers like 123 and names like John", "paragraph")
        ]
        
        for text, section_type in test_sections:
            importance = transformer._calculate_section_importance(text, section_type)
            assert 0 <= importance <= 1
    
    def test_document_reconstruction(self, transformer, sample_document):
        """Test reconstruction of documents from sections."""
        sections = transformer._parse_document_sections(sample_document)
        reconstructed = transformer._reconstruct_document(sections)
        
        # Should be able to reconstruct something meaningful
        assert len(reconstructed) > 0
        assert isinstance(reconstructed, str)
        
        # Should preserve most of the content
        original_words = set(sample_document.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        
        # Most words should be preserved
        preserved_ratio = len(original_words & reconstructed_words) / len(original_words)
        assert preserved_ratio > 0.8
    
    def test_transformation_caching(self, transformer, sample_document):
        """Test that transformations are cached properly."""
        # First transformation
        result1 = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.PARAPHRASE
        )
        
        # Second identical transformation should use cache
        result2 = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.PARAPHRASE
        )
        
        # Results should be identical (from cache)
        assert result1.transformed_document == result2.transformed_document
        assert result1.transformation_map == result2.transformation_map
    
    def test_transformation_stats(self, transformer, sample_document, constructed_language):
        """Test transformation statistics generation."""
        transformed_doc = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.TRANSLATION,
            language=constructed_language
        )
        
        stats = transformer.get_transformation_stats(transformed_doc)
        
        assert 'transformation_type' in stats
        assert 'original_length' in stats
        assert 'transformed_length' in stats
        assert 'length_ratio' in stats
        assert 'section_count' in stats
        
        assert stats['transformation_type'] == TransformationType.TRANSLATION.value
        assert stats['original_length'] > 0
        assert stats['transformed_length'] > 0
        assert stats['length_ratio'] > 0
    
    def test_preserve_entities_flag(self, transformer, sample_document, constructed_language):
        """Test the preserve_entities flag functionality."""
        # With entity preservation
        preserved = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.TRANSLATION,
            language=constructed_language,
            preserve_entities=True
        )
        
        # Without entity preservation
        not_preserved = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.TRANSLATION,
            language=constructed_language,
            preserve_entities=False
        )
        
        # Results should be different
        assert preserved.transformed_document != not_preserved.transformed_document
        
        # Metadata should reflect the setting
        assert preserved.metadata['preserve_entities'] == True
        assert not_preserved.metadata['preserve_entities'] == False
    
    def test_preserve_structure_flag(self, transformer, sample_document):
        """Test the preserve_structure flag functionality."""
        # With structure preservation
        preserved = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.STRUCTURAL,
            preserve_structure=True
        )
        
        # Without structure preservation
        not_preserved = transformer.transform_document(
            document=sample_document,
            transformation_type=TransformationType.STRUCTURAL,
            preserve_structure=False
        )
        
        # Metadata should reflect the setting
        assert preserved.metadata['preserve_structure'] == True
        assert not_preserved.metadata['preserve_structure'] == False
    
    def test_cache_management(self, transformer):
        """Test cache management functionality."""
        initial_stats = transformer.get_cache_stats()
        assert 'cached_transformations' in initial_stats
        
        # Clear cache
        transformer.clear_cache()
        
        # Stats should show empty cache
        after_clear_stats = transformer.get_cache_stats()
        assert after_clear_stats['cached_transformations'] == 0
    
    def test_invalid_transformation_type(self, transformer, sample_document):
        """Test handling of invalid transformation types."""
        with pytest.raises(ValueError):
            transformer.transform_document(
                document=sample_document,
                transformation_type="invalid_type"  # Should cause error
            )
    
    def test_empty_document(self, transformer):
        """Test handling of empty documents."""
        sections = transformer._parse_document_sections("")
        assert len(sections) == 0
        
        # Transform empty document
        transformed = transformer.transform_document(
            document="",
            transformation_type=TransformationType.PARAPHRASE
        )
        
        assert transformed.transformed_document == ""
        assert len(transformed.sections) == 0
    
    def test_section_metadata(self, transformer, sample_document):
        """Test that section metadata is properly populated."""
        sections = transformer._parse_document_sections(sample_document)
        
        for section in sections:
            assert isinstance(section.metadata, dict)
            assert 'word_count' in section.metadata
            assert 'char_count' in section.metadata
            assert section.metadata['word_count'] >= 0
            assert section.metadata['char_count'] >= 0