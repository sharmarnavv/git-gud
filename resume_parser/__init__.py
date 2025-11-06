"""
Resume Parser Package

This package provides functionality for parsing and extracting information from resumes,
including skills extraction, contact information, work experience, and education.
"""

from .resume_interfaces import (
    ContactInfo,
    WorkExperience,
    Education,
    Certification,
    ParsedResume,
    DocumentLoaderInterface,
    ContactExtractorInterface,
    ResumeSkillsExtractorInterface,
    ExperienceExtractorInterface,
    EducationExtractorInterface,
    ResumeParserInterface
)

from .resume_skills_extractor import ResumeSkillsExtractor
from .resume_ontology_enhancer import ResumeOntologyEnhancer
from .contact_extractor import ContactExtractor
from .experience_extractor import ExperienceExtractor
from .education_extractor import EducationExtractor
from .resume_parser import ResumeParser
from .resume_exceptions import (
    ResumeParserError,
    DocumentLoadError,
    SkillsExtractionError,
    ContactExtractionError,
    ExperienceExtractionError,
    EducationExtractionError
)

__version__ = "1.0.0"
__author__ = "sharmarnav"

__all__ = [
    # Data models
    "ContactInfo",
    "WorkExperience", 
    "Education",
    "Certification",
    "ParsedResume",
    
    # Interfaces
    "DocumentLoaderInterface",
    "ContactExtractorInterface",
    "ResumeSkillsExtractorInterface",
    "ExperienceExtractorInterface",
    "EducationExtractorInterface",
    "ResumeParserInterface",
    
    # Main classes
    "ResumeSkillsExtractor",
    "ResumeOntologyEnhancer",
    "ContactExtractor",
    "ExperienceExtractor",
    "EducationExtractor",
    "ResumeParser",
    
    # Exceptions
    "ResumeParserError",
    "DocumentLoadError",
    "SkillsExtractionError",
    "ContactExtractionError",
    "ExperienceExtractionError",
    "EducationExtractionError"
]