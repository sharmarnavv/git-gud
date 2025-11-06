"""
Education and certification extractor for resume parsing.

This module provides functionality to extract education information
and certifications from resume text using pattern matching and NER techniques.
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from .resume_interfaces import EducationExtractorInterface, Education, Certification
from .resume_exceptions import EducationExtractionError
from job_parser.logging_config import get_logger


class EducationExtractor(EducationExtractorInterface):
    """Education and certification extractor using pattern matching."""
    
    def __init__(self):
        """Initialize the education extractor."""
        self.logger = get_logger(__name__)
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for education extraction."""
        
        # Degree patterns
        self.degree_patterns = [
            re.compile(r'\b(Ph\.?D\.?|Doctor of Philosophy|Doctorate)\b', re.IGNORECASE),
            re.compile(r'\b(M\.?S\.?|Master of Science|Masters? of Science)\b', re.IGNORECASE),
            re.compile(r'\b(M\.?A\.?|Master of Arts|Masters? of Arts)\b', re.IGNORECASE),
            re.compile(r'\b(MBA|Master of Business Administration)\b', re.IGNORECASE),
            re.compile(r'\b(M\.?Eng\.?|Master of Engineering)\b', re.IGNORECASE),
            re.compile(r'\b(B\.?S\.?|Bachelor of Science|Bachelors? of Science)\b', re.IGNORECASE),
            re.compile(r'\b(B\.?A\.?|Bachelor of Arts|Bachelors? of Arts)\b', re.IGNORECASE),
            re.compile(r'\b(B\.?Eng\.?|Bachelor of Engineering)\b', re.IGNORECASE),
            re.compile(r'\b(Associate|A\.?A\.?|A\.?S\.?)\b', re.IGNORECASE),
            re.compile(r'\b(Certificate|Diploma)\b', re.IGNORECASE),
        ]
        
        # Education section headers
        self.education_headers = [
            re.compile(r'education', re.IGNORECASE),
            re.compile(r'academic\s+background', re.IGNORECASE),
            re.compile(r'qualifications', re.IGNORECASE),
            re.compile(r'degrees?', re.IGNORECASE),
        ]
        
        # Certification section headers
        self.certification_headers = [
            re.compile(r'certifications?', re.IGNORECASE),
            re.compile(r'licenses?', re.IGNORECASE),
            re.compile(r'professional\s+certifications?', re.IGNORECASE),
            re.compile(r'credentials?', re.IGNORECASE),
        ] 
       
        # Date patterns for graduation/certification dates
        self.date_patterns = [
            re.compile(r'(\d{1,2})/(\d{4})', re.IGNORECASE),            # MM/YYYY
            re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})', re.IGNORECASE),  # Month YYYY
            re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', re.IGNORECASE),  # Full Month YYYY
            re.compile(r'(\d{4})', re.IGNORECASE),                      # YYYY only
        ]
        
        # GPA patterns
        self.gpa_patterns = [
            re.compile(r'GPA:?\s*(\d+\.?\d*)\s*/?\s*(\d+\.?\d*)?', re.IGNORECASE),
            re.compile(r'Grade:?\s*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*GPA', re.IGNORECASE),
        ]
        
        # Common majors/fields of study
        self.major_keywords = [
            'computer science', 'engineering', 'mathematics', 'physics', 'chemistry',
            'biology', 'business', 'economics', 'finance', 'accounting', 'marketing',
            'psychology', 'sociology', 'history', 'english', 'literature',
            'information technology', 'data science', 'artificial intelligence'
        ]
        
        # Common certification providers
        self.certification_providers = [
            'microsoft', 'amazon', 'google', 'cisco', 'oracle', 'ibm', 'salesforce',
            'pmp', 'comptia', 'certified', 'professional', 'associate'
        ]
    
    def extract_education(self, text: str) -> List[Education]:
        """Extract education information from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            List of Education objects
        """
        try:
            self.logger.info("Extracting education information from resume")
            
            # Find education section
            education_section = self._find_education_section(text)
            
            if not education_section:
                self.logger.warning("No education section found")
                return []
            
            # Extract individual education entries
            educations = self._parse_education_entries(education_section)
            
            self.logger.info(f"Extracted {len(educations)} education entries")
            return educations
            
        except Exception as e:
            self.logger.error(f"Education extraction failed: {e}")
            raise EducationExtractionError(f"Failed to extract education: {e}", cause=e)
    
    def extract_certifications(self, text: str) -> List[Certification]:
        """Extract certifications from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            List of Certification objects
        """
        try:
            self.logger.info("Extracting certifications from resume")
            
            # Find certification section
            cert_section = self._find_certification_section(text)
            
            if not cert_section:
                self.logger.warning("No certification section found")
                return []
            
            # Extract individual certification entries
            certifications = self._parse_certification_entries(cert_section)
            
            self.logger.info(f"Extracted {len(certifications)} certifications")
            return certifications
            
        except Exception as e:
            self.logger.error(f"Certification extraction failed: {e}")
            raise EducationExtractionError(f"Failed to extract certifications: {e}", cause=e)    

    def _find_education_section(self, text: str) -> str:
        """Find and extract the education section from resume text."""
        return self._find_section(text, self.education_headers)
    
    def _find_certification_section(self, text: str) -> str:
        """Find and extract the certification section from resume text."""
        return self._find_section(text, self.certification_headers)
    
    def _find_section(self, text: str, header_patterns: List[re.Pattern]) -> str:
        """Find and extract a section based on header patterns."""
        lines = text.split('\n')
        
        in_section = False
        section_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check if this line starts the target section
            if any(pattern.search(line) for pattern in header_patterns):
                in_section = True
                continue
            
            # If we're in the section, collect lines until we hit another section
            if in_section:
                if self._is_new_section_header(line):
                    break
                else:
                    section_lines.append(line)
        
        return '\n'.join(section_lines)
    
    def _is_new_section_header(self, line: str) -> bool:
        """Check if line is a new section header."""
        common_headers = [
            'experience', 'work', 'employment', 'skills', 'projects', 
            'achievements', 'awards', 'publications', 'references'
        ]
        
        line_lower = line.lower().strip()
        
        # Check for common section headers
        for header in common_headers:
            if line_lower.startswith(header) and ':' in line:
                return True
        
        # Check for all caps headers
        if line.isupper() and len(line.split()) <= 3:
            return True
        
        return False
    
    def _parse_education_entries(self, education_text: str) -> List[Education]:
        """Parse individual education entries from education section text."""
        educations = []
        
        # Split by potential education entries
        entries = self._split_into_entries(education_text)
        
        for entry in entries:
            education = self._parse_single_education(entry)
            if education and education.degree:  # Only add if we found a degree
                educations.append(education)
        
        return educations
    
    def _parse_certification_entries(self, cert_text: str) -> List[Certification]:
        """Parse individual certification entries from certification section text."""
        certifications = []
        
        # Split by potential certification entries
        entries = self._split_into_entries(cert_text)
        
        for entry in entries:
            certification = self._parse_single_certification(entry)
            if certification and certification.name:  # Only add if we found a certification name
                certifications.append(certification)
        
        return certifications    

    def _split_into_entries(self, text: str) -> List[str]:
        """Split section text into individual entries."""
        lines = text.split('\n')
        entries = []
        current_entry = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line looks like a new entry
            if self._looks_like_new_entry(line):
                # Save previous entry
                if current_entry:
                    entries.append('\n'.join(current_entry))
                    current_entry = []
            
            current_entry.append(line)
        
        # Add final entry
        if current_entry:
            entries.append('\n'.join(current_entry))
        
        return entries
    
    def _looks_like_new_entry(self, line: str) -> bool:
        """Check if line looks like a new education/certification entry."""
        line_lower = line.lower()
        
        # Check for degree indicators
        has_degree = any(pattern.search(line) for pattern in self.degree_patterns)
        
        # Check for institution indicators
        institution_indicators = ['university', 'college', 'institute', 'school']
        has_institution = any(indicator in line_lower for indicator in institution_indicators)
        
        # Check for certification providers
        has_cert_provider = any(provider in line_lower for provider in self.certification_providers)
        
        # Check for dates
        has_dates = any(pattern.search(line) for pattern in self.date_patterns)
        
        return has_degree or has_institution or has_cert_provider or has_dates
    
    def _parse_single_education(self, entry_text: str) -> Optional[Education]:
        """Parse a single education entry."""
        try:
            education = Education()
            
            # Extract degree
            degree = self._extract_degree(entry_text)
            education.degree = degree
            
            # Extract major/field of study
            major = self._extract_major(entry_text, degree)
            education.major = major
            
            # Extract institution
            institution = self._extract_institution(entry_text)
            education.institution = institution
            
            # Extract graduation date
            graduation_date = self._extract_graduation_date(entry_text)
            education.graduation_date = graduation_date
            
            # Extract GPA
            gpa = self._extract_gpa(entry_text)
            education.gpa = gpa
            
            return education
            
        except Exception as e:
            self.logger.warning(f"Failed to parse education entry: {e}")
            return None
    
    def _parse_single_certification(self, entry_text: str) -> Optional[Certification]:
        """Parse a single certification entry."""
        try:
            certification = Certification()
            
            # Extract certification name
            name = self._extract_certification_name(entry_text)
            certification.name = name
            
            # Extract issuer
            issuer = self._extract_certification_issuer(entry_text)
            certification.issuer = issuer
            
            # Extract issue date
            issue_date = self._extract_certification_date(entry_text)
            certification.issue_date = issue_date
            
            # Extract expiry date (if mentioned)
            expiry_date = self._extract_expiry_date(entry_text)
            certification.expiry_date = expiry_date
            
            # Extract credential ID (if mentioned)
            credential_id = self._extract_credential_id(entry_text)
            certification.credential_id = credential_id
            
            return certification
            
        except Exception as e:
            self.logger.warning(f"Failed to parse certification entry: {e}")
            return None    

    def _extract_degree(self, text: str) -> str:
        """Extract degree type from text."""
        for pattern in self.degree_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()
        return ""
    
    def _extract_major(self, text: str, degree: str) -> str:
        """Extract major/field of study from text."""
        # Remove the degree from text to avoid confusion
        text_without_degree = text.replace(degree, '').strip()
        
        # Look for "in" keyword
        in_match = re.search(r'\bin\s+([^,\n]+)', text_without_degree, re.IGNORECASE)
        if in_match:
            return in_match.group(1).strip()
        
        # Look for common major keywords
        for major in self.major_keywords:
            if major.lower() in text_without_degree.lower():
                return major
        
        # Try to extract from context
        lines = text_without_degree.split('\n')
        for line in lines:
            line = line.strip()
            if len(line.split()) <= 5 and not any(pattern.search(line) for pattern in self.date_patterns):
                # Might be a major
                return line
        
        return ""
    
    def _extract_institution(self, text: str) -> str:
        """Extract institution name from text."""
        institution_indicators = ['university', 'college', 'institute', 'school']
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in institution_indicators):
                # Clean up the line
                institution = line.strip()
                # Remove dates
                institution = self._remove_dates_from_text(institution)
                return institution.strip()
        
        return ""
    
    def _extract_graduation_date(self, text: str) -> str:
        """Extract graduation date from text."""
        dates = []
        
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        dates.append(f"{match[0]} {match[1]}")
                else:
                    dates.append(match)
        
        # Return the most recent date (likely graduation date)
        if dates:
            return dates[-1]
        
        return ""
    
    def _extract_gpa(self, text: str) -> Optional[str]:
        """Extract GPA from text."""
        for pattern in self.gpa_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) >= 2 and match.group(2):
                    # Format: GPA/Total
                    return f"{match.group(1)}/{match.group(2)}"
                else:
                    # Just GPA value
                    return match.group(1)
        
        return None   
 
    def _extract_certification_name(self, text: str) -> str:
        """Extract certification name from text."""
        lines = text.split('\n')
        
        # First line is usually the certification name
        if lines:
            name = lines[0].strip()
            # Remove dates and issuer info
            name = self._remove_dates_from_text(name)
            # Remove common prefixes
            name = re.sub(r'^(certified|certificate|certification)\s+', '', name, flags=re.IGNORECASE)
            return name.strip()
        
        return ""
    
    def _extract_certification_issuer(self, text: str) -> str:
        """Extract certification issuer from text."""
        # Look for issuer patterns
        issuer_patterns = [
            re.compile(r'issued\s+by\s+([^,\n]+)', re.IGNORECASE),
            re.compile(r'from\s+([^,\n]+)', re.IGNORECASE),
            re.compile(r'by\s+([^,\n]+)', re.IGNORECASE),
        ]
        
        for pattern in issuer_patterns:
            match = pattern.search(text)
            if match:
                issuer = match.group(1).strip()
                # Remove dates
                issuer = self._remove_dates_from_text(issuer)
                return issuer.strip()
        
        # Look for known certification providers
        for provider in self.certification_providers:
            if provider.lower() in text.lower():
                return provider.title()
        
        return ""
    
    def _extract_certification_date(self, text: str) -> str:
        """Extract certification issue date from text."""
        return self._extract_graduation_date(text)  # Same logic as graduation date
    
    def _extract_expiry_date(self, text: str) -> Optional[str]:
        """Extract certification expiry date from text."""
        expiry_patterns = [
            re.compile(r'expires?\s+([^,\n]+)', re.IGNORECASE),
            re.compile(r'valid\s+until\s+([^,\n]+)', re.IGNORECASE),
            re.compile(r'expiry:?\s+([^,\n]+)', re.IGNORECASE),
        ]
        
        for pattern in expiry_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_credential_id(self, text: str) -> Optional[str]:
        """Extract credential ID from text."""
        id_patterns = [
            re.compile(r'credential\s+id:?\s+([a-zA-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'id:?\s+([a-zA-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'certificate\s+#:?\s+([a-zA-Z0-9\-]+)', re.IGNORECASE),
        ]
        
        for pattern in id_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _remove_dates_from_text(self, text: str) -> str:
        """Remove date patterns from text."""
        for pattern in self.date_patterns:
            text = pattern.sub('', text)
        
        # Remove common date-related words
        date_words = ['graduated', 'completed', 'earned', '-', '–', '|', '(', ')']
        for word in date_words:
            text = text.replace(word, '')
        
        return text.strip()
    
    def classify_education_level(self, educations: List[Education]) -> str:
        """Classify the highest education level achieved."""
        if not educations:
            return "Not Specified"
        
        levels = {
            'doctorate': 5,
            'phd': 5,
            'doctor': 5,
            'master': 4,
            'mba': 4,
            'bachelor': 3,
            'associate': 2,
            'certificate': 1,
            'diploma': 1
        }
        
        highest_level = 0
        highest_degree = "Not Specified"
        
        for education in educations:
            degree_lower = education.degree.lower()
            for level_name, level_value in levels.items():
                if level_name in degree_lower and level_value > highest_level:
                    highest_level = level_value
                    highest_degree = education.degree
        
        return highest_degree