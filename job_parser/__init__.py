"""
Job Description Parser - NLP-powered system for extracting skills from job descriptions.

This package provides tools for parsing unstructured job descriptions into structured
JSON data with confidence scores, targeting job search automation applications.
"""

from .parser import JobDescriptionParser
from .exceptions import JobParserError, OntologyLoadError, ModelLoadError, InputValidationError
from .config import ParserConfig

__version__ = "1.0.0"
__all__ = [
    "JobDescriptionParser",
    "JobParserError",
    "OntologyLoadError", 
    "ModelLoadError",
    "InputValidationError",
    "ParserConfig"
]