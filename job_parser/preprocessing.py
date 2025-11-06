"""
Text preprocessing module for job description parsing.

This module provides text cleaning, sanitization, and tokenization functionality
to prepare job descriptions for NLP processing.
"""

import re
import html
from typing import List, Optional
import spacy
from spacy.lang.en import English

from .interfaces import TextPreprocessorInterface
from .exceptions import InputValidationError, ModelLoadError
from .logging_config import get_logger


class TextPreprocessor(TextPreprocessorInterface):
    """Text preprocessing component for job descriptions.
    
    This class handles text cleaning, HTML/tag removal, sanitization,
    and sentence tokenization using spaCy for downstream NLP processing.
    
    Attributes:
        nlp: spaCy language model for tokenization
        logger: Logger instance for this preprocessor
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the text preprocessor.
        
        Args:
            model_name: Name of the spaCy model to use for tokenization
            
        Raises:
            ModelLoadError: If spaCy model loading fails
        """
        self.logger = get_logger(__name__)
        
        try:
            # Load spaCy model for tokenization
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded spaCy model: {model_name}")
            
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model '{model_name}': {e}")
            try:
                # Fallback to basic English tokenizer
                self.nlp = English()
                self.nlp.add_pipe('sentencizer')
                self.logger.warning("Using fallback English tokenizer")
            except Exception as fallback_error:
                raise ModelLoadError(
                    f"Failed to load spaCy model '{model_name}' and fallback failed: {fallback_error}",
                    cause=e
                )
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading spaCy model: {e}", cause=e)
    
    def preprocess(self, text: str) -> List[str]:
        """Preprocess and tokenize input text.
        
        Performs complete text preprocessing including sanitization, cleaning,
        and sentence tokenization.
        
        Args:
            text: Raw job description text
            
        Returns:
            List of preprocessed sentences ready for NLP processing
            
        Raises:
            TypeError: If input is not a string or is empty
            InputValidationError: If text preprocessing fails or exceeds length limit
        """
        # Validate input type and content first (outside try-catch to preserve exception types)
        self._validate_input(text)
        
        try:
            self.logger.debug("Starting text preprocessing")
            
            # Step 1: Sanitize input to prevent injection risks
            sanitized_text = self.sanitize(text)
            
            # Step 2: Clean HTML and formatting
            cleaned_text = self._clean_html_and_tags(sanitized_text)
            
            # Step 3: Normalize whitespace and formatting
            normalized_text = self._normalize_text(cleaned_text)
            
            # Step 4: Tokenize into sentences
            sentences = self._tokenize_sentences(normalized_text)
            
            # Step 5: Filter and clean sentences
            processed_sentences = self._filter_sentences(sentences)
            
            self.logger.debug(f"Preprocessed {len(processed_sentences)} sentences from input text")
            return processed_sentences
            
        except InputValidationError:
            # Re-raise input validation errors as-is
            raise
        except Exception as e:
            self.logger.error(f"Text preprocessing failed: {e}")
            raise InputValidationError(f"Failed to preprocess text: {e}", cause=e)
    
    def sanitize(self, text: str) -> str:
        """Sanitize input text to prevent injection risks.
        
        Removes potentially dangerous content while preserving legitimate
        job description formatting and content.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text safe for processing
        """
        if not isinstance(text, str):
            return ""
        
        # HTML decode first to handle encoded entities
        sanitized = html.unescape(text)
        
        # Remove script and style tags completely
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r'<style[^>]*>.*?</style>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove potentially dangerous attributes and protocols
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'data:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        # Remove null bytes and control characters (except newlines and tabs)
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized.strip()
    
    def _clean_html_and_tags(self, text: str) -> str:
        """Remove HTML tags and clean formatting.
        
        Args:
            text: Text potentially containing HTML tags
            
        Returns:
            Text with HTML tags removed and formatting cleaned
        """
        # Remove HTML tags but preserve content
        cleaned = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up common HTML entities that might remain
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&hellip;': '...',
            '&mdash;': '—',
            '&ndash;': '–'
        }
        
        for entity, replacement in html_entities.items():
            cleaned = cleaned.replace(entity, replacement)
        
        return cleaned
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and text formatting.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text with consistent formatting
        """
        # Normalize different types of whitespace
        normalized = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        normalized = re.sub(r'\s+([,.!?;:])', r'\1', normalized)
        normalized = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', normalized)
        
        # Normalize quotes
        normalized = re.sub(r'["""]', '"', normalized)
        normalized = re.sub(r"[''']", "'", normalized)
        
        # Clean up bullet points and list markers
        normalized = re.sub(r'^\s*[•·▪▫◦‣⁃]\s*', '• ', normalized, flags=re.MULTILINE)
        normalized = re.sub(r'^\s*[-*+]\s+', '• ', normalized, flags=re.MULTILINE)
        
        return normalized.strip()
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences using spaCy.
        
        Args:
            text: Normalized text to tokenize
            
        Returns:
            List of sentence strings
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            return sentences
            
        except Exception as e:
            self.logger.warning(f"spaCy tokenization failed, using fallback: {e}")
            # Fallback to simple sentence splitting
            return self._fallback_sentence_split(text)
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting when spaCy fails.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentence strings using simple rules
        """
        # Simple sentence splitting on common sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _validate_input(self, text: str) -> None:
        """Validate input text according to requirements.
        
        Validates string type, non-empty content, and 2000-word length limit.
        
        Args:
            text: Input text to validate
            
        Raises:
            TypeError: If input is not a string or is empty
            InputValidationError: If text exceeds 2000-word limit
        """
        # Validate input type (Requirement 4.1)
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")
        
        # Validate non-empty input (Requirement 4.1)
        if not text.strip():
            raise TypeError("Input text cannot be empty")
        
        # Validate 2000-word limit (Requirement 1.2)
        word_count = len(text.split())
        if word_count > 2000:
            raise InputValidationError(
                f"Input text exceeds maximum length of 2000 words, got {word_count} words"
            )
    
    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Filter and clean sentence list.
        
        Args:
            sentences: List of raw sentences
            
        Returns:
            List of filtered and cleaned sentences
        """
        filtered = []
        
        for sentence in sentences:
            # Skip very short sentences (likely fragments)
            if len(sentence.split()) < 3:
                continue
            
            # Skip sentences that are mostly punctuation or numbers
            word_chars = re.sub(r'[^\w\s]', '', sentence)
            if len(word_chars.strip()) < len(sentence) * 0.5:
                continue
            
            # Skip sentences that are too long (likely formatting issues)
            if len(sentence.split()) > 100:
                continue
            
            filtered.append(sentence.strip())
        
        return filtered