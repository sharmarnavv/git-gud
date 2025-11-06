"""
Contact information extractor for resume parsing.

This module provides functionality to extract contact information from resume text
using regex patterns and NER-based extraction.
"""

import re
from typing import Dict, List, Optional, Tuple

from .resume_interfaces import ContactExtractorInterface, ContactInfo
from .resume_exceptions import ContactExtractionError
from job_parser.logging_config import get_logger


class ContactExtractor(ContactExtractorInterface):
    """Contact information extractor using regex and NER."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Initialize spaCy model if available
        self._nlp = None
        self._spacy_available = False
        
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
            self.logger.info("spaCy model loaded for contact extraction")
        except (ImportError, OSError) as e:
            self.logger.warning(f"spaCy not available for contact extraction: {e}")
        
        # Compile regex patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for contact information extraction."""
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Unified phone pattern with priority for Indian formats
        # Indian: +91-XXXXX-XXXXX, +91 XXXXX XXXXX, 91XXXXXXXXXX, XXXXXXXXXX (10 digits starting with 6-9)
        # US/International: Various formats
        self.phone_pattern = re.compile(
            r'(?:'
            r'\+91[-.\s]?[6-9]\d{9}|'  # +91 followed by Indian mobile
            r'91[6-9]\d{9}|'  # 91 followed by Indian mobile (without +)
            r'\b[6-9]\d{9}\b|'  # Indian mobile without country code
            r'\(?[2-9]\d{2}\)?[-.\s]?[2-9]\d{2}[-.\s]?\d{4}|'  # US format: (XXX) XXX-XXXX
            r'[2-9]\d{2}[-.\s][2-9]\d{2}[-.\s]\d{4}|'  # US format: XXX-XXX-XXXX
            r'\+?[1-9]\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4}'  # International format
            r')',
            re.IGNORECASE
        )
        
        # LinkedIn URL pattern
        self.linkedin_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9-]+)/?',
            re.IGNORECASE
        )
        
        # GitHub URL pattern
        self.github_pattern = re.compile(
            r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9-_]+)/?',
            re.IGNORECASE
        )
        
        # Pin code/ZIP code pattern (6 digits for India, 5 digits for US)
        self.pincode_pattern = re.compile(r'\b\d{6}\b|\b\d{5}(?:-\d{4})?\b', re.IGNORECASE)
    
    def extract_contact_info(self, text: str) -> ContactInfo:
        """Extract contact information from resume text.
        
        Args:
            text: Resume text content
            
        Returns:
            ContactInfo object with extracted information
        """
        try:
            contact_info = ContactInfo()
            
            # Extract name using NER if available, otherwise use heuristics
            contact_info.name = self._extract_name(text)
            
            # Extract email
            contact_info.email = self._extract_email(text)
            
            # Extract phone
            contact_info.phone = self._extract_phone(text)
            
            # Extract address
            contact_info.address = self._extract_address(text)
            
            # Extract LinkedIn
            contact_info.linkedin = self._extract_linkedin(text)
            
            # Extract GitHub
            contact_info.github = self._extract_github(text)
            
            # Validate extracted information
            contact_info.validate()
            
            self.logger.debug(f"Extracted contact info: name={bool(contact_info.name)}, "
                            f"email={bool(contact_info.email)}, phone={bool(contact_info.phone)}")
            
            return contact_info
            
        except Exception as e:
            self.logger.error(f"Contact extraction failed: {e}")
            raise ContactExtractionError(f"Failed to extract contact information: {e}", cause=e)
    
    def _extract_name(self, text: str) -> str:
        """Extract name using NER or heuristics."""
        try:
            # Try NER-based extraction first
            if self._spacy_available and self._nlp:
                name = self._extract_name_with_ner(text)
                if name:
                    return name
            
            # Fallback to heuristic extraction
            return self._extract_name_heuristic(text)
            
        except Exception as e:
            self.logger.warning(f"Name extraction failed: {e}")
            return ""
    
    def _extract_name_with_ner(self, text: str) -> str:
        """Extract name using spaCy NER."""
        # Process first few lines where name is likely to appear
        lines = text.split('\n')[:5]
        text_sample = '\n'.join(lines)
        
        doc = self._nlp(text_sample)
        
        # Look for PERSON entities
        person_names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Filter out common non-name words
                name_text = ent.text.strip()
                if self._is_likely_name(name_text):
                    person_names.append(name_text)
        
        # Return the first valid name found
        if person_names:
            return person_names[0]
        
        return ""
    
    def _extract_name_heuristic(self, text: str) -> str:
        """Extract name using heuristic patterns."""
        lines = text.split('\n')
        
        # Look in first few lines
        for line in lines[:5]:
            line = line.strip()
            
            # Skip empty lines and lines with email/phone
            if not line or '@' in line or self.phone_pattern.search(line):
                continue
            
            # Look for name-like patterns
            words = line.split()
            if 2 <= len(words) <= 4:  # Typical name length
                # Check if words look like names (capitalized, no numbers)
                if all(word[0].isupper() and word.isalpha() for word in words):
                    return line
        
        return ""
    
    def _is_likely_name(self, text: str) -> bool:
        """Check if text is likely to be a person's name."""
        # Filter out common non-name entities
        exclude_words = {
            'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary',
            'experience', 'education', 'skills', 'contact', 'information'
        }
        
        words = text.lower().split()
        if any(word in exclude_words for word in words):
            return False
        
        # Check for reasonable name characteristics
        if len(words) < 1 or len(words) > 4:
            return False
        
        # All words should be alphabetic and reasonably short
        return all(word.isalpha() and len(word) > 1 and len(word) < 20 for word in words)
    
    def _extract_email(self, text: str) -> str:
        """Extract email address from text."""
        matches = self.email_pattern.findall(text)
        
        if matches:
            # Return the first email found
            return matches[0]
        
        return ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text with priority for Indian formats."""
        match = self.phone_pattern.search(text)
        if match:
            phone = match.group(0)
            # Clean up the phone number
            phone = re.sub(r'[^\d+]', '', phone)  # Remove all non-digit characters except +
            
            # Format based on length and pattern
            if phone.startswith('+91') and len(phone) == 13:
                # Indian format: +91XXXXXXXXXX -> +91-XXXXX-XXXXX
                return f"+91-{phone[3:8]}-{phone[8:]}"
            elif phone.startswith('91') and len(phone) == 12:
                # Indian without +: 91XXXXXXXXXX -> +91-XXXXX-XXXXX
                # Check if the number after 91 starts with valid Indian mobile prefix
                mobile_part = phone[2:]
                if mobile_part[0] in '6789':
                    return f"+91-{mobile_part[:5]}-{mobile_part[5:]}"
                else:
                    return phone  # Not a valid Indian mobile number
            elif len(phone) == 10 and phone[0] in '6789':
                # Indian mobile without country code: XXXXXXXXXX -> +91-XXXXX-XXXXX
                return f"+91-{phone[:5]}-{phone[5:]}"
            elif len(phone) == 10 and phone[0] not in '6789':
                # 10-digit number not starting with 6-9, likely not Indian mobile
                # Check if it matches US format pattern (area code 2-9, exchange 2-9)
                if re.match(r'^[2-9]\d{2}[2-9]\d{2}\d{4}$', phone):
                    return phone  # Valid US format
                else:
                    return ""  # Invalid format
            else:
                # Return as-is for other formats
                return phone
        
        return ""
    
    def _extract_address(self, text: str) -> str:
        """Extract pin code/ZIP code from text."""
        match = self.pincode_pattern.search(text)
        if match:
            return match.group(0)
        
        return ""
    
    def _extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL from text."""
        match = self.linkedin_pattern.search(text)
        if match:
            # Return full URL
            username = match.group(1)
            return f"https://linkedin.com/in/{username}"
        
        return None
    
    def _extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL from text."""
        match = self.github_pattern.search(text)
        if match:
            # Return full URL
            username = match.group(1)
            return f"https://github.com/{username}"
        
        return None