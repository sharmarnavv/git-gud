"""
Resume parsing exceptions for the Resume-Job Matcher.

This module defines custom exceptions for resume parsing operations,
following the same patterns as the job description parser.
"""

from job_parser.exceptions import JobParserError


class ResumeParserError(JobParserError):
    """Base exception for resume parsing errors."""
    pass


class DocumentLoadError(ResumeParserError):
    """Exception raised when document loading fails."""
    pass


class ContactExtractionError(ResumeParserError):
    """Exception raised when contact information extraction fails."""
    pass


class ExperienceExtractionError(ResumeParserError):
    """Exception raised when work experience extraction fails."""
    pass


class EducationExtractionError(ResumeParserError):
    """Exception raised when education extraction fails."""
    pass


class SkillsExtractionError(ResumeParserError):
    """Exception raised when skills extraction fails."""
    pass


class FileFormatError(DocumentLoadError):
    """Exception raised for unsupported file formats."""
    pass


class TextExtractionError(DocumentLoadError):
    """Exception raised when text extraction from document fails."""
    pass


class ValidationError(ResumeParserError):
    """Exception raised when resume data validation fails."""
    pass