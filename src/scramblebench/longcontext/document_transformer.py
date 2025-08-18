"""
Document transformation for long context benchmarks.

This module provides methods to transform long documents through
translation, paraphrasing, or structural modifications while
preserving the essential information needed for Q&A tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
import logging
from pathlib import Path
import hashlib

from scramblebench.translation.language_generator import ConstructedLanguage
from scramblebench.translation.translator import ProblemTranslator


class TransformationType(Enum):
    """Types of document transformations."""
    TRANSLATION = "translation"  # Translate to constructed language
    PARAPHRASE = "paraphrase"  # Rephrase content while preserving meaning
    STRUCTURAL = "structural"  # Reorder paragraphs/sections
    HYBRID = "hybrid"  # Combination of multiple transformations


@dataclass
class DocumentSection:
    """
    A section of a document with metadata.
    
    Attributes:
        content: Text content of the section
        section_type: Type of section (paragraph, heading, list, etc.)
        position: Original position in the document
        importance: Importance score for Q&A tasks (0-1)
        metadata: Additional section metadata
    """
    content: str
    section_type: str
    position: int
    importance: float = 0.5
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TransformedDocument:
    """
    A document that has been transformed for benchmarking.
    
    Attributes:
        original_document: Original document content
        transformed_document: Transformed document content
        transformation_type: Type of transformation applied
        sections: List of document sections
        transformation_map: Mapping between original and transformed elements
        metadata: Additional transformation metadata
    """
    original_document: str
    transformed_document: str
    transformation_type: TransformationType
    sections: List[DocumentSection]
    transformation_map: Dict[str, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentTransformer:
    """
    Transforms long documents for contamination-free evaluation.
    
    Applies various transformation strategies to documents while preserving
    the information content necessary for accurate Q&A evaluation.
    """
    
    def __init__(
        self,
        translator: Optional[ProblemTranslator] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the document transformer.
        
        Args:
            translator: Problem translator for language transformations
            logger: Logger instance (creates default if None)
        """
        self.translator = translator or ProblemTranslator()
        self.logger = logger or logging.getLogger("scramblebench.document_transformer")
        self._transformation_cache: Dict[str, TransformedDocument] = {}
    
    def transform_document(
        self,
        document: str,
        transformation_type: TransformationType,
        language: Optional[ConstructedLanguage] = None,
        preserve_structure: bool = True,
        preserve_entities: bool = True
    ) -> TransformedDocument:
        """
        Transform a document using the specified transformation type.
        
        Args:
            document: Original document text
            transformation_type: Type of transformation to apply
            language: Constructed language (required for translation)
            preserve_structure: Whether to preserve document structure
            preserve_entities: Whether to preserve named entities
            
        Returns:
            TransformedDocument containing the transformation result
        """
        # Create cache key
        cache_key = self._create_cache_key(
            document, transformation_type, language, preserve_structure, preserve_entities
        )
        
        if cache_key in self._transformation_cache:
            self.logger.debug("Using cached transformation")
            return self._transformation_cache[cache_key]
        
        self.logger.info(f"Transforming document using {transformation_type.value}")
        
        # Parse document into sections
        sections = self._parse_document_sections(document)
        
        # Apply transformation based on type
        if transformation_type == TransformationType.TRANSLATION:
            if language is None:
                raise ValueError("Language is required for translation transformation")
            result = self._transform_by_translation(
                document, sections, language, preserve_structure, preserve_entities
            )
        elif transformation_type == TransformationType.PARAPHRASE:
            result = self._transform_by_paraphrase(
                document, sections, preserve_structure, preserve_entities
            )
        elif transformation_type == TransformationType.STRUCTURAL:
            result = self._transform_by_structure(
                document, sections, preserve_structure, preserve_entities
            )
        elif transformation_type == TransformationType.HYBRID:
            result = self._transform_hybrid(
                document, sections, language, preserve_structure, preserve_entities
            )
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")
        
        # Cache the result
        self._transformation_cache[cache_key] = result
        
        self.logger.info(
            f"Document transformation completed: "
            f"{len(sections)} sections, "
            f"{len(result.transformation_map)} mappings"
        )
        
        return result
    
    def _parse_document_sections(self, document: str) -> List[DocumentSection]:
        """
        Parse a document into logical sections.
        
        Args:
            document: Document text to parse
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', document.strip())
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Determine section type
            section_type = self._classify_section_type(paragraph)
            
            # Calculate importance (simple heuristic)
            importance = self._calculate_section_importance(paragraph, section_type)
            
            section = DocumentSection(
                content=paragraph.strip(),
                section_type=section_type,
                position=i,
                importance=importance,
                metadata={
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph)
                }
            )
            
            sections.append(section)
        
        return sections
    
    def _classify_section_type(self, text: str) -> str:
        """Classify the type of a text section."""
        text = text.strip()
        
        # Heading detection
        if len(text) < 100 and (
            text.isupper() or
            re.match(r'^#+\s+', text) or  # Markdown heading
            re.match(r'^\d+\.?\s+[A-Z]', text)  # Numbered section
        ):
            return "heading"
        
        # List detection
        if re.match(r'^[\s]*[-*•]\s+', text, re.MULTILINE):
            return "list"
        
        # Numbered list detection
        if re.match(r'^[\s]*\d+\.?\s+', text, re.MULTILINE):
            return "numbered_list"
        
        # Quote detection
        if text.startswith('"') and text.endswith('"'):
            return "quote"
        
        # Default to paragraph
        return "paragraph"
    
    def _calculate_section_importance(self, text: str, section_type: str) -> float:
        """
        Calculate the importance of a section for Q&A tasks.
        
        Args:
            text: Section text
            section_type: Type of the section
            
        Returns:
            Importance score between 0 and 1
        """
        base_importance = {
            "heading": 0.8,
            "paragraph": 0.6,
            "list": 0.5,
            "numbered_list": 0.7,
            "quote": 0.4
        }.get(section_type, 0.5)
        
        # Adjust based on content characteristics
        word_count = len(text.split())
        
        # Longer sections often contain more information
        length_factor = min(1.0, word_count / 50)
        
        # Sections with numbers or proper nouns are often important
        has_numbers = bool(re.search(r'\d+', text))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', text))
        
        content_bonus = 0.1 * (has_numbers + has_proper_nouns)
        
        final_importance = min(1.0, base_importance + length_factor * 0.2 + content_bonus)
        
        return final_importance
    
    def _transform_by_translation(
        self,
        document: str,
        sections: List[DocumentSection],
        language: ConstructedLanguage,
        preserve_structure: bool,
        preserve_entities: bool
    ) -> TransformedDocument:
        """Transform document by translating to constructed language."""
        transformation_map = {}
        transformed_sections = []
        
        for section in sections:
            # Create a pseudo-problem for translation
            pseudo_problem = {"text": section.content}
            
            translated = self.translator.translate_problem(
                problem=pseudo_problem,
                language=language,
                preserve_numbers=True,
                preserve_proper_nouns=preserve_entities
            )
            
            transformed_content = translated.translated_problem["text"]
            
            # Update transformation map
            transformation_map.update(translated.translation_key)
            
            # Create transformed section
            transformed_section = DocumentSection(
                content=transformed_content,
                section_type=section.section_type,
                position=section.position,
                importance=section.importance,
                metadata={
                    **section.metadata,
                    'translation_units': len(translated.translation_units)
                }
            )
            
            transformed_sections.append(transformed_section)
        
        # Reconstruct document
        if preserve_structure:
            transformed_document = self._reconstruct_document(transformed_sections)
        else:
            # Simple concatenation
            transformed_document = '\n\n'.join(
                section.content for section in transformed_sections
            )
        
        return TransformedDocument(
            original_document=document,
            transformed_document=transformed_document,
            transformation_type=TransformationType.TRANSLATION,
            sections=transformed_sections,
            transformation_map=transformation_map,
            metadata={
                'language_name': language.name,
                'language_type': language.language_type.value,
                'preserve_structure': preserve_structure,
                'preserve_entities': preserve_entities,
                'section_count': len(sections)
            }
        )
    
    def _transform_by_paraphrase(
        self,
        document: str,
        sections: List[DocumentSection],
        preserve_structure: bool,
        preserve_entities: bool
    ) -> TransformedDocument:
        """
        Transform document by paraphrasing content.
        
        Note: This is a placeholder implementation. In practice,
        you would use advanced NLP models for paraphrasing.
        """
        self.logger.warning(
            "Paraphrase transformation not fully implemented. "
            "Using simple word substitutions as placeholder."
        )
        
        # Simple substitution patterns for demonstration
        substitution_patterns = [
            (r'\bsaid\b', 'stated'),
            (r'\bshowed\b', 'demonstrated'),
            (r'\bbecause\b', 'due to the fact that'),
            (r'\balso\b', 'additionally'),
            (r'\bhowever\b', 'nevertheless'),
            (r'\btherefore\b', 'consequently'),
            (r'\bimportant\b', 'significant'),
            (r'\blarge\b', 'substantial'),
            (r'\bsmall\b', 'minimal'),
        ]
        
        transformation_map = {}
        transformed_sections = []
        
        for section in sections:
            transformed_content = section.content
            
            # Apply substitution patterns
            for pattern, replacement in substitution_patterns:
                old_content = transformed_content
                transformed_content = re.sub(
                    pattern, replacement, transformed_content, flags=re.IGNORECASE
                )
                
                # Track changes
                if old_content != transformed_content:
                    # Find the actual substitutions made
                    for match in re.finditer(pattern, old_content, re.IGNORECASE):
                        transformation_map[replacement] = match.group()
            
            transformed_section = DocumentSection(
                content=transformed_content,
                section_type=section.section_type,
                position=section.position,
                importance=section.importance,
                metadata={
                    **section.metadata,
                    'paraphrase_changes': len([
                        1 for p, r in substitution_patterns
                        if re.search(p, section.content, re.IGNORECASE)
                    ])
                }
            )
            
            transformed_sections.append(transformed_section)
        
        # Reconstruct document
        if preserve_structure:
            transformed_document = self._reconstruct_document(transformed_sections)
        else:
            transformed_document = '\n\n'.join(
                section.content for section in transformed_sections
            )
        
        return TransformedDocument(
            original_document=document,
            transformed_document=transformed_document,
            transformation_type=TransformationType.PARAPHRASE,
            sections=transformed_sections,
            transformation_map=transformation_map,
            metadata={
                'preserve_structure': preserve_structure,
                'preserve_entities': preserve_entities,
                'substitution_patterns_used': len(substitution_patterns),
                'section_count': len(sections)
            }
        )
    
    def _transform_by_structure(
        self,
        document: str,
        sections: List[DocumentSection],
        preserve_structure: bool,
        preserve_entities: bool
    ) -> TransformedDocument:
        """Transform document by reordering sections."""
        transformation_map = {}
        
        # Keep headings in place, shuffle other sections
        headings = [s for s in sections if s.section_type == "heading"]
        other_sections = [s for s in sections if s.section_type != "heading"]
        
        # Simple reordering: reverse order of non-heading sections
        reordered_other = list(reversed(other_sections))
        
        # Recombine sections
        if preserve_structure and headings:
            # Try to maintain heading-section relationships
            transformed_sections = []
            heading_idx = 0
            other_idx = 0
            
            for original_section in sections:
                if original_section.section_type == "heading":
                    if heading_idx < len(headings):
                        transformed_sections.append(headings[heading_idx])
                        heading_idx += 1
                else:
                    if other_idx < len(reordered_other):
                        # Update position to reflect new order
                        reordered_section = reordered_other[other_idx]
                        reordered_section.metadata['original_position'] = reordered_section.position
                        reordered_section.position = len(transformed_sections)
                        
                        transformed_sections.append(reordered_section)
                        other_idx += 1
            
            # Add any remaining sections
            while heading_idx < len(headings):
                transformed_sections.append(headings[heading_idx])
                heading_idx += 1
            while other_idx < len(reordered_other):
                transformed_sections.append(reordered_other[other_idx])
                other_idx += 1
        else:
            # Simple concatenation of reordered sections
            transformed_sections = headings + reordered_other
            
            # Update positions
            for i, section in enumerate(transformed_sections):
                section.metadata['original_position'] = section.position
                section.position = i
        
        # Create transformation map (section position changes)
        for section in transformed_sections:
            if 'original_position' in section.metadata:
                original_pos = section.metadata['original_position']
                new_pos = section.position
                transformation_map[f"section_{new_pos}"] = f"section_{original_pos}"
        
        # Reconstruct document
        transformed_document = self._reconstruct_document(transformed_sections)
        
        return TransformedDocument(
            original_document=document,
            transformed_document=transformed_document,
            transformation_type=TransformationType.STRUCTURAL,
            sections=transformed_sections,
            transformation_map=transformation_map,
            metadata={
                'preserve_structure': preserve_structure,
                'preserve_entities': preserve_entities,
                'headings_count': len(headings),
                'reordered_sections': len(other_sections),
                'section_count': len(sections)
            }
        )
    
    def _transform_hybrid(
        self,
        document: str,
        sections: List[DocumentSection],
        language: Optional[ConstructedLanguage],
        preserve_structure: bool,
        preserve_entities: bool
    ) -> TransformedDocument:
        """Transform document using multiple transformation strategies."""
        # First apply structural transformation
        structural_result = self._transform_by_structure(
            document, sections, preserve_structure, preserve_entities
        )
        
        # Then apply translation if language is provided
        if language is not None:
            translation_result = self._transform_by_translation(
                structural_result.transformed_document,
                structural_result.sections,
                language,
                preserve_structure,
                preserve_entities
            )
            
            # Combine transformation maps
            combined_map = {**structural_result.transformation_map, **translation_result.transformation_map}
            
            return TransformedDocument(
                original_document=document,
                transformed_document=translation_result.transformed_document,
                transformation_type=TransformationType.HYBRID,
                sections=translation_result.sections,
                transformation_map=combined_map,
                metadata={
                    'transformations': ['structural', 'translation'],
                    'language_name': language.name if language else None,
                    'preserve_structure': preserve_structure,
                    'preserve_entities': preserve_entities,
                    'section_count': len(sections)
                }
            )
        else:
            # Just structural + paraphrase
            paraphrase_result = self._transform_by_paraphrase(
                structural_result.transformed_document,
                structural_result.sections,
                preserve_structure,
                preserve_entities
            )
            
            combined_map = {**structural_result.transformation_map, **paraphrase_result.transformation_map}
            
            return TransformedDocument(
                original_document=document,
                transformed_document=paraphrase_result.transformed_document,
                transformation_type=TransformationType.HYBRID,
                sections=paraphrase_result.sections,
                transformation_map=combined_map,
                metadata={
                    'transformations': ['structural', 'paraphrase'],
                    'preserve_structure': preserve_structure,
                    'preserve_entities': preserve_entities,
                    'section_count': len(sections)
                }
            )
    
    def _reconstruct_document(self, sections: List[DocumentSection]) -> str:
        """Reconstruct a document from sections while preserving structure."""
        document_parts = []
        
        for section in sorted(sections, key=lambda s: s.position):
            content = section.content
            
            # Add appropriate spacing based on section type
            if section.section_type == "heading":
                if document_parts:  # Not the first section
                    document_parts.append('\n\n')
                document_parts.append(content)
                document_parts.append('\n\n')
            elif section.section_type in ["list", "numbered_list"]:
                document_parts.append(content)
                document_parts.append('\n\n')
            else:
                document_parts.append(content)
                document_parts.append('\n\n')
        
        return ''.join(document_parts).strip()
    
    def _create_cache_key(
        self,
        document: str,
        transformation_type: TransformationType,
        language: Optional[ConstructedLanguage],
        preserve_structure: bool,
        preserve_entities: bool
    ) -> str:
        """Create a cache key for the transformation."""
        # Create hash of document content
        doc_hash = hashlib.md5(document.encode()).hexdigest()
        
        # Create key components
        key_parts = [
            doc_hash,
            transformation_type.value,
            language.name if language else "no_language",
            str(preserve_structure),
            str(preserve_entities)
        ]
        
        return "_".join(key_parts)
    
    def get_transformation_stats(self, transformed_doc: TransformedDocument) -> Dict[str, Any]:
        """
        Get statistics about a document transformation.
        
        Args:
            transformed_doc: Transformed document to analyze
            
        Returns:
            Dictionary containing transformation statistics
        """
        original_words = len(transformed_doc.original_document.split())
        transformed_words = len(transformed_doc.transformed_document.split())
        
        # Calculate section-level changes
        section_changes = 0
        for section in transformed_doc.sections:
            if 'original_position' in section.metadata:
                if section.metadata['original_position'] != section.position:
                    section_changes += 1
        
        stats = {
            'transformation_type': transformed_doc.transformation_type.value,
            'original_length': len(transformed_doc.original_document),
            'transformed_length': len(transformed_doc.transformed_document),
            'original_words': original_words,
            'transformed_words': transformed_words,
            'length_ratio': len(transformed_doc.transformed_document) / len(transformed_doc.original_document),
            'word_ratio': transformed_words / original_words,
            'section_count': len(transformed_doc.sections),
            'transformation_mappings': len(transformed_doc.transformation_map),
            'section_reorderings': section_changes,
            'avg_section_importance': sum(s.importance for s in transformed_doc.sections) / len(transformed_doc.sections)
        }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the transformation cache."""
        self._transformation_cache.clear()
        self.logger.info("Document transformation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the transformation cache."""
        return {
            'cached_transformations': len(self._transformation_cache)
        }