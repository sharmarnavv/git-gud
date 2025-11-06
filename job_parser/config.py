"""
Configuration management system for the Job Description Parser.

This module handles all configuration settings including environment variables,
default values, and runtime configuration options.
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ParserConfig:
    """Configuration class for the Job Description Parser.
    
    This class centralizes all configuration options and provides defaults
    that can be overridden through environment variables or direct assignment.
    
    Attributes:
        ontology_path: Path to the skills ontology CSV file
        similarity_threshold: Minimum cosine similarity for skill matches (0-1)
        max_text_length: Maximum allowed input text length in words
        enable_ner: Whether to enable Named Entity Recognition
        enable_semantic: Whether to enable semantic matching
        confidence_weighting: Weights for combining different confidence sources
        model_name: Name of the Sentence-BERT model to use
        log_level: Logging level for the parser
    """
    
    ontology_path: str = field(default_factory=lambda: os.getenv('ONTOLOGY_PATH', 'skills_ontology.csv'))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv('SIMILARITY_THRESHOLD', '0.7')))
    max_text_length: int = field(default_factory=lambda: int(os.getenv('MAX_TEXT_LENGTH', '2000')))
    enable_ner: bool = field(default_factory=lambda: os.getenv('ENABLE_NER', 'true').lower() == 'true')
    enable_semantic: bool = field(default_factory=lambda: os.getenv('ENABLE_SEMANTIC', 'true').lower() == 'true')
    model_name: str = field(default_factory=lambda: os.getenv('SENTENCE_BERT_MODEL', './trained_model/trained_model'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    confidence_weighting: Dict[str, float] = field(default_factory=lambda: {
        'semantic': float(os.getenv('SEMANTIC_WEIGHT', '0.6')),
        'ner': float(os.getenv('NER_WEIGHT', '0.4'))
    })
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        
        if self.max_text_length <= 0:
            raise ValueError(f"max_text_length must be positive, got {self.max_text_length}")
        
        if not self.ontology_path:
            raise ValueError("ontology_path cannot be empty")
        
        # Validate confidence weights sum to 1.0
        total_weight = sum(self.confidence_weighting.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"confidence_weighting must sum to 1.0, got {total_weight}")


# Global configuration instance
CONFIG = ParserConfig()