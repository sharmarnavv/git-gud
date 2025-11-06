"""
Custom exception hierarchy for the Job Description Parser.

This module defines all custom exceptions used throughout the parser system,
providing clear error messaging and proper exception chaining.
"""

from typing import Optional


class JobParserError(Exception):
    """Base exception for all job parser related errors.
    
    This is the root exception class that all other parser exceptions inherit from.
    It provides a consistent interface for error handling throughout the system.
    """
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """Initialize the base parser error.
        
        Args:
            message: Human-readable error description
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause


class OntologyLoadError(JobParserError):
    """Raised when skills ontology loading fails.
    
    This exception is raised when the CSV ontology file cannot be loaded,
    is malformed, or contains invalid data structure.
    """
    pass


class ModelLoadError(JobParserError):
    """Raised when NLP models fail to load.
    
    This exception is raised when Sentence-BERT or spaCy models cannot be
    loaded due to missing files, network issues, or incompatible versions.
    """
    pass


class InputValidationError(JobParserError):
    """Raised for invalid input parameters.
    
    This exception is raised when input validation fails, such as non-string
    inputs, empty text, or text exceeding maximum length limits.
    """
    pass