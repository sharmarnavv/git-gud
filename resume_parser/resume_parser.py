"""
Main resume parser orchestrator for the Resume-Job Matcher.

This module provides the main ResumeParser class that orchestrates all
extraction components and provides comprehensive resume parsing functionality.
"""

import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from .resume_interfaces import (
    ResumeParserInterface, 
    ParsedResume, 
    ContactInfo, 
    WorkExperience, 
    Education, 
    Certification
)
from .resume_exceptions import (
    ResumeParserError, 
    DocumentLoadError, 
    ValidationError
)
from .document_loaders import DocumentLoaderFactory
from .contact_extractor import ContactExtractor
from .resume_skills_extractor import ResumeSkillsExtractor
from .experience_extractor import ExperienceExtractor
from .education_extractor import EducationExtractor
from job_parser.logging_config import get_logger
from job_parser.ontology import OntologyLoader


class ResumeParser(ResumeParserInterface):
    """Main resume parser that orchestrates all extraction components."""
    
    def __init__(self, 
                 ontology_loader: Optional[OntologyLoader] = None,
                 ontology_path: Optional[str] = None,
                 enable_semantic_matching: bool = True,
                 enable_ner: bool = True):
        """Initialize the resume parser with all components.
        
        Args:
            ontology_loader: Optional ontology loader instance
            ontology_path: Optional path to ontology CSV file
            enable_semantic_matching: Whether to enable semantic matching for skills
            enable_ner: Whether to enable NER extraction
        """
        self.logger = get_logger(__name__)
        
        # Initialize ontology loader and load ontology
        self.ontology_loader = ontology_loader
        self.ontology = None
        
        if self.ontology_loader is None:
            self.ontology_loader = OntologyLoader()
        
        # Try to load ontology
        try:
            if ontology_path and os.path.exists(ontology_path):
                self.ontology = self.ontology_loader.load_ontology(ontology_path)
            else:
                # Try default paths
                default_paths = ['skills_ontology.csv', 'comprehensive_skills_ontology.csv']
                for path in default_paths:
                    if os.path.exists(path):
                        self.ontology = self.ontology_loader.load_ontology(path)
                        break
                
                # Fallback to built-in ontology
                if self.ontology is None:
                    self.ontology = self.ontology_loader.FALLBACK_ONTOLOGY
                    self.logger.info("Using fallback ontology")
        except Exception as e:
            self.logger.warning(f"Failed to load ontology: {e}")
            self.ontology = self.ontology_loader.FALLBACK_ONTOLOGY
        
        # Initialize document loader factory
        self.document_loader_factory = DocumentLoaderFactory()
        
        # Initialize extraction components
        self._initialize_extractors(enable_semantic_matching, enable_ner)
        
        # Parser configuration
        self.config = {
            'enable_semantic_matching': enable_semantic_matching,
            'enable_ner': enable_ner,
            'min_confidence_threshold': 0.3,
            'max_file_size_mb': 10,
            'supported_formats': ['.pdf', '.docx', '.txt'],
            'quality_scoring_enabled': True,
            'metadata_generation_enabled': True
        }
        
        self.logger.info("ResumeParser initialized successfully")
    
    def _initialize_extractors(self, enable_semantic_matching: bool, enable_ner: bool):
        """Initialize all extraction components."""
        try:
            # Initialize contact extractor
            self.contact_extractor = ContactExtractor()
            
            # Initialize skills extractor with optional components
            semantic_matcher = None
            ner_extractor = None
            
            if enable_semantic_matching:
                try:
                    from job_parser.semantic_matching import SemanticMatcher
                    semantic_matcher = SemanticMatcher(
                        model_name='./trained_model/trained_model',
                        threshold=0.5,
                        enable_caching=True
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize semantic matcher: {e}")
            
            if enable_ner:
                try:
                    from job_parser.ner_extraction import NERExtractor
                    ner_extractor = NERExtractor()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize NER extractor: {e}")
            
            self.skills_extractor = ResumeSkillsExtractor(
                semantic_matcher=semantic_matcher,
                ner_extractor=ner_extractor
            )
            
            # Initialize experience extractor
            self.experience_extractor = ExperienceExtractor()
            
            # Initialize education extractor
            self.education_extractor = EducationExtractor()
            
            self.logger.info("All extraction components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize extractors: {e}")
            raise ResumeParserError(f"Extractor initialization failed: {e}", cause=e)
    
    def parse_resume(self, file_path: str) -> ParsedResume:
        """Parse resume file into structured data.
        
        Args:
            file_path: Path to resume file
            
        Returns:
            ParsedResume object with extracted information
            
        Raises:
            ResumeParserError: If parsing fails
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting resume parsing for: {file_path}")
            
            # Validate file
            self._validate_file(file_path)
            
            # Load and extract text from document
            text = self._load_document_text(file_path)
            
            # Parse the extracted text
            parsed_resume = self.parse_resume_text(text)
            
            # Add file metadata
            file_metadata = self._generate_file_metadata(file_path)
            parsed_resume.metadata.update(file_metadata)
            
            # Calculate parsing time
            parsing_time = time.time() - start_time
            parsed_resume.metadata['parsing_time_seconds'] = round(parsing_time, 2)
            
            self.logger.info(f"Resume parsing completed in {parsing_time:.2f} seconds")
            return parsed_resume
            
        except Exception as e:
            self.logger.error(f"Resume parsing failed for {file_path}: {e}")
            if isinstance(e, ResumeParserError):
                raise
            else:
                raise ResumeParserError(f"Resume parsing failed: {e}", cause=e)
    
    def parse_resume_text(self, text: str) -> ParsedResume:
        """Parse resume text into structured data.
        
        Args:
            text: Resume text content
            
        Returns:
            ParsedResume object with extracted information
        """
        try:
            start_time = time.time()
            self.logger.info("Starting resume text parsing")
            
            # Initialize parsed resume object
            parsed_resume = ParsedResume()
            
            # Track extraction results for metadata
            extraction_results = {
                'contact_extraction': {'success': False, 'error': None},
                'skills_extraction': {'success': False, 'error': None, 'count': 0},
                'experience_extraction': {'success': False, 'error': None, 'count': 0},
                'education_extraction': {'success': False, 'error': None, 'count': 0}
            }
            
            # Extract contact information
            try:
                parsed_resume.contact_info = self._extract_contact_info_with_fallback(text)
                extraction_results['contact_extraction']['success'] = True
                self.logger.debug("Contact information extracted successfully")
            except Exception as e:
                self.logger.warning(f"Contact extraction failed: {e}")
                extraction_results['contact_extraction']['error'] = str(e)
                parsed_resume.contact_info = ContactInfo()  # Empty contact info
            
            # Extract skills
            try:
                parsed_resume.skills = self._extract_skills_with_fallback(text)
                extraction_results['skills_extraction']['success'] = True
                extraction_results['skills_extraction']['count'] = len(parsed_resume.skills)
                self.logger.debug(f"Extracted {len(parsed_resume.skills)} skills")
            except Exception as e:
                self.logger.warning(f"Skills extraction failed: {e}")
                extraction_results['skills_extraction']['error'] = str(e)
                parsed_resume.skills = []
            
            # Extract work experience
            try:
                parsed_resume.experience = self._extract_experience_with_fallback(text)
                extraction_results['experience_extraction']['success'] = True
                extraction_results['experience_extraction']['count'] = len(parsed_resume.experience)
                self.logger.debug(f"Extracted {len(parsed_resume.experience)} work experiences")
            except Exception as e:
                self.logger.warning(f"Experience extraction failed: {e}")
                extraction_results['experience_extraction']['error'] = str(e)
                parsed_resume.experience = []
            
            # Extract education
            try:
                education_list = self._extract_education_with_fallback(text)
                parsed_resume.education = education_list
                extraction_results['education_extraction']['success'] = True
                extraction_results['education_extraction']['count'] = len(education_list)
                self.logger.debug(f"Extracted {len(education_list)} education entries")
            except Exception as e:
                self.logger.warning(f"Education extraction failed: {e}")
                extraction_results['education_extraction']['error'] = str(e)
                parsed_resume.education = []
            
            # Extract certifications (part of education extractor)
            try:
                parsed_resume.certifications = self.education_extractor.extract_certifications(text)
                self.logger.debug(f"Extracted {len(parsed_resume.certifications)} certifications")
            except Exception as e:
                self.logger.warning(f"Certification extraction failed: {e}")
                parsed_resume.certifications = []
            
            # Generate metadata
            if self.config['metadata_generation_enabled']:
                metadata = self._generate_parsing_metadata(text, extraction_results, start_time)
                parsed_resume.metadata.update(metadata)
            
            # Calculate quality score
            if self.config['quality_scoring_enabled']:
                quality_score = self._calculate_resume_quality_score(parsed_resume)
                parsed_resume.metadata['quality_score'] = quality_score
            
            # Validate parsed resume
            self._validate_parsed_resume(parsed_resume)
            
            parsing_time = time.time() - start_time
            self.logger.info(f"Resume text parsing completed in {parsing_time:.2f} seconds")
            
            return parsed_resume
            
        except Exception as e:
            self.logger.error(f"Resume text parsing failed: {e}")
            if isinstance(e, ResumeParserError):
                raise
            else:
                raise ResumeParserError(f"Resume text parsing failed: {e}", cause=e)
    
    def _validate_file(self, file_path: str):
        """Validate resume file before processing."""
        if not os.path.exists(file_path):
            raise DocumentLoadError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config['max_file_size_mb']:
            raise DocumentLoadError(f"File too large: {file_size_mb:.1f}MB (max: {self.config['max_file_size_mb']}MB)")
        
        # Check file format
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.config['supported_formats']:
            raise DocumentLoadError(f"Unsupported file format: {file_ext}")
        
        self.logger.debug(f"File validation passed: {file_path}")
    
    def _load_document_text(self, file_path: str) -> str:
        """Load and extract text from document file."""
        try:
            # Get appropriate loader
            loader = self.document_loader_factory.get_loader(file_path)
            
            # Load document text
            text = loader.load_document(file_path)
            
            if not text or len(text.strip()) < 50:
                raise DocumentLoadError("Document appears to be empty or too short")
            
            self.logger.debug(f"Document text loaded: {len(text)} characters")
            return text
            
        except Exception as e:
            self.logger.error(f"Document loading failed: {e}")
            if isinstance(e, DocumentLoadError):
                raise
            else:
                raise DocumentLoadError(f"Failed to load document: {e}", cause=e)
    
    def _extract_contact_info_with_fallback(self, text: str) -> ContactInfo:
        """Extract contact information with error handling and fallback."""
        try:
            return self.contact_extractor.extract_contact_info(text)
        except Exception as e:
            self.logger.warning(f"Primary contact extraction failed, using fallback: {e}")
            # Fallback: create minimal contact info
            return ContactInfo()
    
    def _extract_skills_with_fallback(self, text: str) -> List[str]:
        """Extract skills with error handling and fallback."""
        try:
            if self.ontology:
                return self.skills_extractor.extract_skills(text, self.ontology)
            else:
                # Fallback: use basic skill extraction without ontology
                return self._extract_skills_basic(text)
        except Exception as e:
            self.logger.warning(f"Primary skills extraction failed, using fallback: {e}")
            return self._extract_skills_basic(text)
    
    def _extract_skills_basic(self, text: str) -> List[str]:
        """Basic skill extraction fallback method."""
        # Simple keyword-based skill extraction
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jenkins',
            'machine learning', 'data analysis', 'project management', 'agile'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_experience_with_fallback(self, text: str) -> List[WorkExperience]:
        """Extract work experience with error handling and fallback."""
        try:
            return self.experience_extractor.extract_experience(text)
        except Exception as e:
            self.logger.warning(f"Primary experience extraction failed, using fallback: {e}")
            # Fallback: return empty list
            return []
    
    def _extract_education_with_fallback(self, text: str) -> List[Education]:
        """Extract education with error handling and fallback."""
        try:
            return self.education_extractor.extract_education(text)
        except Exception as e:
            self.logger.warning(f"Primary education extraction failed, using fallback: {e}")
            # Fallback: return empty list
            return []
    
    def _generate_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Generate metadata about the processed file."""
        try:
            file_stats = os.stat(file_path)
            file_info = self.document_loader_factory.detect_file_format(file_path)
            
            return {
                'source_file': {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size_bytes': file_stats.st_size,
                    'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                    'extension': file_info.get('extension', ''),
                    'mime_type': file_info.get('mime_type', ''),
                    'modified_time': file_stats.st_mtime
                }
            }
        except Exception as e:
            self.logger.warning(f"Failed to generate file metadata: {e}")
            return {'source_file': {'path': file_path, 'error': str(e)}}
    
    def _generate_parsing_metadata(self, text: str, extraction_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Generate comprehensive parsing metadata."""
        parsing_time = time.time() - start_time
        
        # Calculate text statistics
        text_stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
        }
        
        # Calculate extraction success rates
        total_extractions = len(extraction_results)
        successful_extractions = sum(1 for result in extraction_results.values() if result['success'])
        success_rate = successful_extractions / total_extractions if total_extractions > 0 else 0
        
        return {
            'parsing_metadata': {
                'parser_version': '1.0.0',
                'parsing_time_seconds': round(parsing_time, 3),
                'extraction_success_rate': round(success_rate, 2),
                'extraction_results': extraction_results,
                'text_statistics': text_stats,
                'configuration': self.config.copy()
            }
        }
    
    def _calculate_resume_quality_score(self, parsed_resume: ParsedResume) -> float:
        """Calculate overall quality score for the parsed resume (0.0-1.0)."""
        try:
            score_components = {
                'contact_completeness': 0.0,
                'skills_presence': 0.0,
                'experience_presence': 0.0,
                'education_presence': 0.0,
                'content_richness': 0.0
            }
            
            # Contact information completeness (0-0.2)
            contact = parsed_resume.contact_info
            contact_fields = [contact.name, contact.email, contact.phone]
            filled_contact_fields = sum(1 for field in contact_fields if field and field.strip())
            score_components['contact_completeness'] = (filled_contact_fields / len(contact_fields)) * 0.2
            
            # Skills presence (0-0.3)
            if parsed_resume.skills:
                skills_score = min(len(parsed_resume.skills) / 10, 1.0)  # Max score at 10+ skills
                score_components['skills_presence'] = skills_score * 0.3
            
            # Experience presence (0-0.3)
            if parsed_resume.experience:
                # Score based on number of experiences and their completeness
                exp_score = 0
                for exp in parsed_resume.experience:
                    exp_completeness = 0
                    if exp.job_title: exp_completeness += 0.25
                    if exp.company: exp_completeness += 0.25
                    if exp.start_date: exp_completeness += 0.25
                    if exp.description: exp_completeness += 0.25
                    exp_score += exp_completeness
                
                avg_exp_score = exp_score / len(parsed_resume.experience)
                num_exp_score = min(len(parsed_resume.experience) / 3, 1.0)  # Max score at 3+ experiences
                score_components['experience_presence'] = (avg_exp_score * num_exp_score) * 0.3
            
            # Education presence (0-0.1)
            if parsed_resume.education:
                edu_score = min(len(parsed_resume.education) / 2, 1.0)  # Max score at 2+ education entries
                score_components['education_presence'] = edu_score * 0.1
            
            # Content richness (0-0.1)
            total_content_length = 0
            if parsed_resume.experience:
                total_content_length += sum(len(exp.description) for exp in parsed_resume.experience)
            
            if total_content_length > 500:  # Reasonable amount of content
                score_components['content_richness'] = 0.1
            elif total_content_length > 200:
                score_components['content_richness'] = 0.05
            
            # Calculate final score
            total_score = sum(score_components.values())
            
            # Add to metadata for debugging
            parsed_resume.metadata['quality_score_breakdown'] = score_components
            
            return round(total_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _validate_parsed_resume(self, parsed_resume: ParsedResume):
        """Validate the parsed resume data."""
        try:
            # Validate contact info
            if parsed_resume.contact_info:
                parsed_resume.contact_info.validate()
            
            # Basic validation - ensure we have some meaningful content
            has_content = (
                (parsed_resume.contact_info and parsed_resume.contact_info.name) or
                parsed_resume.skills or
                parsed_resume.experience or
                parsed_resume.education
            )
            
            if not has_content:
                raise ValidationError("Resume appears to have no extractable content")
            
            self.logger.debug("Resume validation passed")
            
        except Exception as e:
            self.logger.error(f"Resume validation failed: {e}")
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"Resume validation failed: {e}", cause=e)
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the parser configuration and capabilities."""
        return {
            'version': '1.0.0',
            'configuration': self.config.copy(),
            'supported_formats': self.document_loader_factory.get_supported_formats(),
            'components': {
                'document_loader': True,
                'contact_extractor': True,
                'skills_extractor': True,
                'experience_extractor': True,
                'education_extractor': True,
                'ontology_loader': self.ontology_loader is not None,
                'semantic_matching': self.config['enable_semantic_matching'],
                'ner_extraction': self.config['enable_ner']
            }
        }
    
    def analyze_resume_completeness(self, parsed_resume: ParsedResume) -> Dict[str, Any]:
        """Analyze completeness of parsed resume and suggest improvements."""
        analysis = {
            'completeness_score': 0.0,
            'missing_sections': [],
            'suggestions': [],
            'strengths': []
        }
        
        try:
            # Check contact information
            contact = parsed_resume.contact_info
            if not contact.name:
                analysis['missing_sections'].append('name')
                analysis['suggestions'].append('Add your full name to the resume')
            else:
                analysis['strengths'].append('Name is present')
            
            if not contact.email:
                analysis['missing_sections'].append('email')
                analysis['suggestions'].append('Add a professional email address')
            else:
                analysis['strengths'].append('Email is present')
            
            if not contact.phone:
                analysis['missing_sections'].append('phone')
                analysis['suggestions'].append('Add a phone number for contact')
            
            # Check skills
            if not parsed_resume.skills:
                analysis['missing_sections'].append('skills')
                analysis['suggestions'].append('Add a skills section with relevant technical and soft skills')
            elif len(parsed_resume.skills) < 5:
                analysis['suggestions'].append('Consider adding more skills to strengthen your profile')
            else:
                analysis['strengths'].append(f'{len(parsed_resume.skills)} skills identified')
            
            # Check experience
            if not parsed_resume.experience:
                analysis['missing_sections'].append('experience')
                analysis['suggestions'].append('Add work experience section with job details')
            else:
                analysis['strengths'].append(f'{len(parsed_resume.experience)} work experiences found')
                
                # Check experience completeness
                incomplete_experiences = []
                for i, exp in enumerate(parsed_resume.experience):
                    if not exp.job_title or not exp.company or not exp.description:
                        incomplete_experiences.append(i + 1)
                
                if incomplete_experiences:
                    analysis['suggestions'].append(f'Complete missing details in experience entries: {incomplete_experiences}')
            
            # Check education
            if not parsed_resume.education:
                analysis['missing_sections'].append('education')
                analysis['suggestions'].append('Add education section with degree and institution details')
            else:
                analysis['strengths'].append(f'{len(parsed_resume.education)} education entries found')
            
            # Calculate completeness score
            total_sections = 4  # contact, skills, experience, education
            complete_sections = total_sections - len(analysis['missing_sections'])
            analysis['completeness_score'] = round(complete_sections / total_sections, 2)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Resume completeness analysis failed: {e}")
            analysis['error'] = str(e)
            return analysis