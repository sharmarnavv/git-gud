"""
Base interfaces and abstract classes for the Job Description Parser.

This module defines the core interfaces that all parser components must implement,
ensuring consistent behavior and enabling dependency injection for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SkillMatch:
    """Represents a matched skill with metadata.
    
    Attributes:
        skill: The matched skill name
        category: Skill category (technical, soft, tools)
        confidence: Confidence score (0-1)
        source: Source of the match ('semantic' or 'ner')
        context: Text context where the skill was found
    """
    skill: str
    category: str
    confidence: float
    source: str
    context: str = ""


@dataclass
class JobDescriptionInput:
    """Input model for job description parsing.
    
    Attributes:
        text: The job description text to parse
        max_length: Maximum allowed text length in words
    """
    text: str
    max_length: int = 2000
    
    def validate(self) -> None:
        """Validate input constraints.
        
        Raises:
            InputValidationError: If input validation fails
        """
        from .exceptions import InputValidationError
        
        if not isinstance(self.text, str):
            raise InputValidationError(f"Input text must be a string, got {type(self.text)}")
        
        if not self.text.strip():
            raise InputValidationError("Input text cannot be empty")
        
        word_count = len(self.text.split())
        if word_count > self.max_length:
            raise InputValidationError(
                f"Input text exceeds maximum length of {self.max_length} words, got {word_count}"
            )


@dataclass
class ParsedJobDescription:
    """Output model for parsed job description data.
    
    Attributes:
        skills_required: List of extracted required skills
        experience_level: Inferred experience level (entry, mid, senior)
        tools_mentioned: List of tools and technologies mentioned
        confidence_scores: Confidence scores for each extracted skill
        categories: Skills organized by category
        metadata: Additional parsing metadata
    """
    skills_required: List[str]
    experience_level: str
    tools_mentioned: List[str]
    confidence_scores: Dict[str, float]
    categories: Dict[str, List[str]]
    metadata: Dict[str, Any]
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'skills_required': self.skills_required,
            'experience_level': self.experience_level,
            'tools_mentioned': self.tools_mentioned,
            'confidence_scores': self.confidence_scores,
            'categories': self.categories,
            'metadata': self.metadata
        }


class OntologyLoaderInterface(ABC):
    """Abstract interface for ontology loading components."""
    
    @abstractmethod
    def load_ontology(self, csv_path: str) -> Dict[str, List[str]]:
        """Load skills ontology from CSV file.
        
        Args:
            csv_path: Path to the CSV ontology file
            
        Returns:
            Dictionary with categories as keys and skill lists as values
            
        Raises:
            OntologyLoadError: If ontology loading fails
        """
        pass


class TextPreprocessorInterface(ABC):
    """Abstract interface for text preprocessing components."""
    
    @abstractmethod
    def preprocess(self, text: str) -> List[str]:
        """Preprocess and tokenize input text.
        
        Args:
            text: Raw job description text
            
        Returns:
            List of preprocessed sentences
            
        Raises:
            InputValidationError: If text preprocessing fails
        """
        pass
    
    @abstractmethod
    def sanitize(self, text: str) -> str:
        """Sanitize input text to prevent injection risks.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text safe for processing
        """
        pass


class SemanticMatcherInterface(ABC):
    """Abstract interface for semantic matching components."""
    
    @abstractmethod
    def find_skill_matches(self, 
                          sentences: List[str], 
                          ontology: Dict[str, List[str]], 
                          threshold: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        """Find semantic matches between job text and ontology skills.
        
        Args:
            sentences: Preprocessed job description sentences
            ontology: Skills ontology dictionary
            threshold: Minimum similarity threshold for matches
            
        Returns:
            Dictionary mapping categories to lists of (skill, confidence) tuples
            
        Raises:
            ModelLoadError: If semantic model loading fails
        """
        pass


class NERExtractorInterface(ABC):
    """Abstract interface for Named Entity Recognition components."""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[SkillMatch]:
        """Extract named entities with confidence scores.
        
        Args:
            text: Job description text
            
        Returns:
            List of SkillMatch objects for extracted entities
            
        Raises:
            ModelLoadError: If NER model loading fails
        """
        pass


class SkillCategorizerInterface(ABC):
    """Abstract interface for skill categorization components."""
    
    @abstractmethod
    def categorize_skills(self, 
                         skill_matches: List[SkillMatch], 
                         ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Categorize and rank extracted skills.
        
        Args:
            skill_matches: List of matched skills
            ontology: Skills ontology for categorization
            
        Returns:
            Dictionary mapping categories to ranked skill lists
        """
        pass
    
    @abstractmethod
    def compute_confidence_scores(self, skill_matches: List[SkillMatch]) -> Dict[str, float]:
        """Compute final confidence scores for skills.
        
        Args:
            skill_matches: List of matched skills with individual confidences
            
        Returns:
            Dictionary mapping skills to final confidence scores
        """
        pass
    
    @abstractmethod
    def infer_experience_level(self, text: str, skill_matches: List[SkillMatch]) -> str:
        """Infer experience level from job description text and extracted skills.
        
        Args:
            text: Original job description text
            skill_matches: List of extracted skill matches
            
        Returns:
            Experience level string: 'entry-level', 'mid-level', or 'senior-level'
        """
        pass


class JobDescriptionParserInterface(ABC):
    """Abstract interface for the main job description parser."""
    
    @abstractmethod
    def parse_job_description(self, job_desc: str) -> ParsedJobDescription:
        """Parse job description into structured data.
        
        Args:
            job_desc: Raw job description text
            
        Returns:
            ParsedJobDescription object with extracted information
            
        Raises:
            JobParserError: If parsing fails
        """
        pass