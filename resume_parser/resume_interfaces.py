"""
Resume parsing interfaces and data models for the Resume-Job Matcher.

This module defines the core interfaces and data structures for resume parsing,
following the same patterns as the job description parser.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


@dataclass
class ContactInfo:
    """Contact information extracted from resume.
    
    Attributes:
        name: Full name of the candidate
        email: Email address
        phone: Phone number
        address: Physical address
        linkedin: LinkedIn profile URL
        github: GitHub profile URL
    """
    name: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    linkedin: Optional[str] = None
    github: Optional[str] = None
    
    def validate(self) -> None:
        """Validate contact information.
        
        Raises:
            ValueError: If validation fails
        """
        import re
        
        if self.email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', self.email):
            raise ValueError(f"Invalid email format: {self.email}")
        
        if self.phone and not re.match(r'^[\+]?[1-9][\d\s\-\(\)\.]{7,15}$', self.phone.replace(' ', '').replace('-', '')):
            raise ValueError(f"Invalid phone format: {self.phone}")


@dataclass
class WorkExperience:
    """Work experience entry from resume.
    
    Attributes:
        job_title: Job title or position
        company: Company name
        start_date: Start date (string format)
        end_date: End date (None for current position)
        description: Job description text
        skills_used: List of skills mentioned in this experience
        duration_months: Duration in months
    """
    job_title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: Optional[str] = None
    description: str = ""
    skills_used: List[str] = field(default_factory=list)
    duration_months: int = 0


@dataclass
class Education:
    """Education entry from resume.
    
    Attributes:
        degree: Degree type (Bachelor's, Master's, etc.)
        major: Field of study or major
        institution: School or university name
        graduation_date: Graduation date
        gpa: GPA if available
    """
    degree: str = ""
    major: str = ""
    institution: str = ""
    graduation_date: str = ""
    gpa: Optional[str] = None


@dataclass
class Certification:
    """Certification or professional qualification.
    
    Attributes:
        name: Certification name
        issuer: Issuing organization
        issue_date: Date issued
        expiry_date: Expiration date if applicable
        credential_id: Credential ID if available
    """
    name: str = ""
    issuer: str = ""
    issue_date: str = ""
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None


@dataclass
class ParsedResume:
    """Complete parsed resume data structure.
    
    Attributes:
        contact_info: Contact information
        skills: List of extracted skills
        experience: List of work experiences
        education: List of education entries
        certifications: List of certifications
        metadata: Additional parsing metadata
    """
    contact_info: ContactInfo = field(default_factory=ContactInfo)
    skills: List[str] = field(default_factory=list)
    experience: List[WorkExperience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    certifications: List[Certification] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'contact_info': {
                'name': self.contact_info.name,
                'email': self.contact_info.email,
                'phone': self.contact_info.phone,
                'address': self.contact_info.address,
                'linkedin': self.contact_info.linkedin,
                'github': self.contact_info.github
            },
            'skills': self.skills,
            'experience': [
                {
                    'job_title': exp.job_title,
                    'company': exp.company,
                    'start_date': exp.start_date,
                    'end_date': exp.end_date,
                    'description': exp.description,
                    'skills_used': exp.skills_used,
                    'duration_months': exp.duration_months
                }
                for exp in self.experience
            ],
            'education': [
                {
                    'degree': edu.degree,
                    'major': edu.major,
                    'institution': edu.institution,
                    'graduation_date': edu.graduation_date,
                    'gpa': edu.gpa
                }
                for edu in self.education
            ],
            'certifications': [
                {
                    'name': cert.name,
                    'issuer': cert.issuer,
                    'issue_date': cert.issue_date,
                    'expiry_date': cert.expiry_date,
                    'credential_id': cert.credential_id
                }
                for cert in self.certifications
            ],
            'metadata': self.metadata
        }


class DocumentLoaderInterface(ABC):
    """Abstract interface for document loading components."""
    
    @abstractmethod
    def load_document(self, file_path: str) -> str:
        """Load and extract text from document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentLoadError: If document loading fails
        """
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file can be processed, False otherwise
        """
        pass


class ContactExtractorInterface(ABC):
    """Abstract interface for contact information extraction."""
    
    @abstractmethod
    def extract_contact_info(self, text: str) -> ContactInfo:
        """Extract contact information from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            ContactInfo object with extracted information
        """
        pass


class ResumeSkillsExtractorInterface(ABC):
    """Abstract interface for resume skills extraction."""
    
    @abstractmethod
    def extract_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[str]:
        """Extract skills from resume text.
        
        Args:
            text: Resume text content
            ontology: Skills ontology for matching
            
        Returns:
            List of extracted skills
        """
        pass
    
    @abstractmethod
    def analyze_skill_context(self, text: str, skill: str) -> Dict[str, Any]:
        """Analyze context where skill is mentioned.
        
        Args:
            text: Resume text content
            skill: Skill to analyze
            
        Returns:
            Dictionary with context analysis
        """
        pass


class ExperienceExtractorInterface(ABC):
    """Abstract interface for work experience extraction."""
    
    @abstractmethod
    def extract_experience(self, text: str) -> List[WorkExperience]:
        """Extract work experience from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            List of WorkExperience objects
        """
        pass


class EducationExtractorInterface(ABC):
    """Abstract interface for education extraction."""
    
    @abstractmethod
    def extract_education(self, text: str) -> List[Education]:
        """Extract education information from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            List of Education objects
        """
        pass


class ResumeParserInterface(ABC):
    """Abstract interface for the main resume parser."""
    
    @abstractmethod
    def parse_resume(self, file_path: str) -> ParsedResume:
        """Parse resume file into structured data.
        
        Args:
            file_path: Path to resume file
            
        Returns:
            ParsedResume object with extracted information
            
        Raises:
            ResumeParserError: If parsing fails
        """
        pass
    
    @abstractmethod
    def parse_resume_text(self, text: str) -> ParsedResume:
        """Parse resume text into structured data.
        
        Args:
            text: Resume text content
            
        Returns:
            ParsedResume object with extracted information
        """
        pass