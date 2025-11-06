"""
Education requirements matching and scoring for resume-job analysis.

This module implements education matching including degree levels, fields of study,
institution analysis, certification matching, and gap analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import re

from .resume_interfaces import ParsedResume, Education, Certification
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EducationMatch:
    """Represents an education match between resume and job requirements.
    
    Attributes:
        education_type: Type of education (degree, certification, etc.)
        match_level: Level of match (exact, equivalent, partial)
        confidence: Confidence score (0-1)
        resume_education: Matching education from resume
        job_requirement: Job requirement that was matched
        relevance_score: Relevance score for the field of study
    """
    education_type: str
    match_level: str
    confidence: float
    resume_education: str
    job_requirement: str
    relevance_score: float


@dataclass
class EducationAnalysisResult:
    """Result of education requirements analysis.
    
    Attributes:
        overall_score: Overall education match score (0-100)
        degree_match_score: Degree level match score (0-100)
        field_match_score: Field of study match score (0-100)
        institution_score: Institution prestige/relevance score (0-100)
        certification_score: Professional certification match score (0-100)
        matched_requirements: List of matched education requirements
        missing_requirements: List of missing education requirements
        alternative_paths: Suggested alternative qualification paths
        confidence_score: Overall confidence in the analysis
        metadata: Additional analysis metadata
    """
    overall_score: float
    degree_match_score: float
    field_match_score: float
    institution_score: float
    certification_score: float
    matched_requirements: List[EducationMatch]
    missing_requirements: List[str]
    alternative_paths: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class EducationRequirementsScorer:
    """Education requirements scorer for resume-job comparison.
    
    This class analyzes education requirements including degree levels,
    fields of study, institution prestige, and professional certifications.
    """
    
    def __init__(self,
                 degree_hierarchy: Optional[Dict[str, int]] = None,
                 field_mappings: Optional[Dict[str, List[str]]] = None,
                 institution_tiers: Optional[Dict[str, List[str]]] = None):
        """Initialize education requirements scorer.
        
        Args:
            degree_hierarchy: Hierarchy of degree levels with numeric values
            field_mappings: Mappings between related fields of study
            institution_tiers: Institution prestige tiers
        """
        # Degree hierarchy (higher number = higher level)
        self.degree_hierarchy = degree_hierarchy or {
            'high school': 1,
            'diploma': 1,
            'certificate': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'mba': 5,
            'doctorate': 6,
            'phd': 6,
            'md': 6,
            'jd': 6
        }
        
        # Field of study mappings
        self.field_mappings = field_mappings or self._load_field_mappings()
        
        # Institution tiers (for prestige scoring)
        self.institution_tiers = institution_tiers or self._load_institution_tiers()
        
        # Common degree variations and synonyms
        self.degree_synonyms = self._load_degree_synonyms()
        
        # Professional certification categories
        self.certification_categories = self._load_certification_categories()
        
        # Performance tracking
        self._education_stats = {
            'total_analyses': 0,
            'degree_matches': 0,
            'field_matches': 0,
            'certification_matches': 0,
            'institution_matches': 0
        }
        
        logger.info("Education requirements scorer initialized")
    
    def calculate_education_similarity(self,
                                     resume: ParsedResume,
                                     job_description: ParsedJobDescription,
                                     job_text: str = "") -> EducationAnalysisResult:
        """Calculate comprehensive education requirements similarity.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description data
            job_text: Original job description text for analysis
            
        Returns:
            EducationAnalysisResult with detailed analysis
        """
        try:
            logger.debug("Calculating education similarity")
            
            # Extract education requirements from job
            job_requirements = self._extract_education_requirements(job_description, job_text)
            
            # Analyze degree level matching
            degree_match_score, degree_matches = self._analyze_degree_matching(
                resume.education, job_requirements
            )
            
            # Analyze field of study matching
            field_match_score, field_matches = self._analyze_field_matching(
                resume.education, job_requirements
            )
            
            # Analyze institution prestige/relevance
            institution_score = self._analyze_institution_quality(
                resume.education, job_requirements
            )
            
            # Analyze professional certifications
            certification_score, cert_matches = self._analyze_certification_matching(
                resume.certifications, job_requirements, job_text
            )
            
            # Combine all matches
            all_matches = degree_matches + field_matches + cert_matches
            
            # Identify missing requirements
            missing_requirements = self._identify_missing_requirements(
                job_requirements, all_matches
            )
            
            # Generate alternative qualification paths
            alternative_paths = self._generate_alternative_paths(
                resume, missing_requirements, job_requirements
            )
            
            # Calculate overall education score
            overall_score = self._calculate_overall_education_score(
                degree_match_score, field_match_score, 
                institution_score, certification_score
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_education_confidence(
                resume.education, resume.certifications, len(job_requirements)
            )
            
            # Create metadata
            metadata = {
                'total_job_requirements': len(job_requirements),
                'resume_education_count': len(resume.education),
                'resume_certification_count': len(resume.certifications),
                'matched_requirements_count': len(all_matches),
                'missing_requirements_count': len(missing_requirements),
                'highest_resume_degree': self._get_highest_degree(resume.education),
                'required_degree_level': self._extract_required_degree_level(job_requirements),
                'education_stats': self._education_stats.copy()
            }
            
            # Update performance stats
            self._education_stats['total_analyses'] += 1
            if degree_match_score > 70:
                self._education_stats['degree_matches'] += 1
            if field_match_score > 70:
                self._education_stats['field_matches'] += 1
            if certification_score > 70:
                self._education_stats['certification_matches'] += 1
            if institution_score > 70:
                self._education_stats['institution_matches'] += 1
            
            result = EducationAnalysisResult(
                overall_score=overall_score,
                degree_match_score=degree_match_score,
                field_match_score=field_match_score,
                institution_score=institution_score,
                certification_score=certification_score,
                matched_requirements=all_matches,
                missing_requirements=missing_requirements,
                alternative_paths=alternative_paths,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
            logger.debug(f"Education similarity calculated: {overall_score:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Education similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate education similarity: {e}")
    
    def _extract_education_requirements(self,
                                      job_description: ParsedJobDescription,
                                      job_text: str) -> List[Dict[str, Any]]:
        """Extract education requirements from job description.
        
        Args:
            job_description: Parsed job description
            job_text: Original job text
            
        Returns:
            List of education requirements
        """
        requirements = []
        
        # Extract from job text using patterns
        if job_text:
            degree_requirements = self._extract_degree_requirements(job_text)
            field_requirements = self._extract_field_requirements(job_text)
            certification_requirements = self._extract_certification_requirements(job_text)
            
            requirements.extend(degree_requirements)
            requirements.extend(field_requirements)
            requirements.extend(certification_requirements)
        
        # If no requirements found, infer from job level and skills
        if not requirements:
            inferred_requirements = self._infer_education_requirements(job_description)
            requirements.extend(inferred_requirements)
        
        return requirements
    
    def _extract_degree_requirements(self, job_text: str) -> List[Dict[str, Any]]:
        """Extract degree requirements from job text.
        
        Args:
            job_text: Job description text
            
        Returns:
            List of degree requirement dictionaries
        """
        requirements = []
        text_lower = job_text.lower()
        
        # Degree requirement patterns
        degree_patterns = [
            r'(bachelor[\'s]*|ba|bs|b\.a\.|b\.s\.)\s*(?:degree)?\s*(?:in\s+([^,\.\n]+))?',
            r'(master[\'s]*|ma|ms|m\.a\.|m\.s\.)\s*(?:degree)?\s*(?:in\s+([^,\.\n]+))?',
            r'(doctorate|phd|ph\.d\.)\s*(?:degree)?\s*(?:in\s+([^,\.\n]+))?',
            r'(mba)\s*(?:degree)?',
            r'(associate[\'s]*|aa|as)\s*(?:degree)?\s*(?:in\s+([^,\.\n]+))?',
            r'(high school|diploma)\s*(?:degree|diploma)?'
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                degree_type = match.group(1)
                field = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                # Normalize degree type
                normalized_degree = self._normalize_degree_type(degree_type)
                
                requirement = {
                    'type': 'degree',
                    'level': normalized_degree,
                    'field': field.strip() if field else None,
                    'required': True,
                    'source': 'explicit'
                }
                requirements.append(requirement)
        
        return requirements
    
    def _extract_field_requirements(self, job_text: str) -> List[Dict[str, Any]]:
        """Extract field of study requirements from job text.
        
        Args:
            job_text: Job description text
            
        Returns:
            List of field requirement dictionaries
        """
        requirements = []
        text_lower = job_text.lower()
        
        # Field requirement patterns
        field_patterns = [
            r'degree\s+in\s+([^,\.\n]+)',
            r'background\s+in\s+([^,\.\n]+)',
            r'education\s+in\s+([^,\.\n]+)',
            r'study\s+in\s+([^,\.\n]+)',
            r'major\s+in\s+([^,\.\n]+)'
        ]
        
        for pattern in field_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                field = match.group(1).strip()
                
                # Clean up field name
                field = re.sub(r'\s+or\s+related\s+field.*', '', field)
                field = re.sub(r'\s+or\s+equivalent.*', '', field)
                
                if field and len(field) > 2:
                    requirement = {
                        'type': 'field',
                        'field': field,
                        'required': True,
                        'source': 'explicit'
                    }
                    requirements.append(requirement)
        
        return requirements
    
    def _extract_certification_requirements(self, job_text: str) -> List[Dict[str, Any]]:
        """Extract certification requirements from job text.
        
        Args:
            job_text: Job description text
            
        Returns:
            List of certification requirement dictionaries
        """
        requirements = []
        text_lower = job_text.lower()
        
        # Common certification patterns
        cert_patterns = [
            r'(pmp|project management professional)',
            r'(cissp|certified information systems security professional)',
            r'(cpa|certified public accountant)',
            r'(cfa|chartered financial analyst)',
            r'(aws|amazon web services)\s*certified',
            r'(azure|microsoft azure)\s*certified',
            r'(google cloud|gcp)\s*certified',
            r'(scrum master|csm|certified scrum master)',
            r'(agile|certified agile)',
            r'(six sigma|lean six sigma)',
            r'certification\s+in\s+([^,\.\n]+)',
            r'certified\s+([^,\.\n]+)',
            r'license\s+in\s+([^,\.\n]+)',
            r'licensed\s+([^,\.\n]+)'
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                cert_name = match.group(1) if match.group(1) else match.group(0)
                
                requirement = {
                    'type': 'certification',
                    'name': cert_name.strip(),
                    'required': 'required' in text_lower or 'must have' in text_lower,
                    'preferred': 'preferred' in text_lower or 'plus' in text_lower,
                    'source': 'explicit'
                }
                requirements.append(requirement)
        
        return requirements
    
    def _infer_education_requirements(self, job_description: ParsedJobDescription) -> List[Dict[str, Any]]:
        """Infer education requirements from job level and skills.
        
        Args:
            job_description: Parsed job description
            
        Returns:
            List of inferred education requirements
        """
        requirements = []
        
        # Infer based on experience level
        experience_level = job_description.experience_level.lower()
        
        if 'senior' in experience_level or 'lead' in experience_level:
            requirements.append({
                'type': 'degree',
                'level': 'bachelor',
                'field': None,
                'required': True,
                'source': 'inferred'
            })
        elif 'entry' in experience_level or 'junior' in experience_level:
            requirements.append({
                'type': 'degree',
                'level': 'bachelor',
                'field': None,
                'required': False,
                'source': 'inferred'
            })
        
        # Infer field based on skills
        technical_skills = job_description.skills_required
        inferred_field = self._infer_field_from_skills(technical_skills)
        
        if inferred_field:
            requirements.append({
                'type': 'field',
                'field': inferred_field,
                'required': False,
                'source': 'inferred'
            })
        
        return requirements
    
    def _infer_field_from_skills(self, skills: List[str]) -> Optional[str]:
        """Infer field of study from technical skills.
        
        Args:
            skills: List of technical skills
            
        Returns:
            Inferred field of study or None
        """
        skills_lower = [skill.lower() for skill in skills]
        
        # Field inference rules
        field_indicators = {
            'computer science': ['python', 'java', 'javascript', 'programming', 'software', 'algorithms'],
            'data science': ['machine learning', 'data analysis', 'statistics', 'python', 'r'],
            'engineering': ['matlab', 'autocad', 'solidworks', 'engineering'],
            'business': ['project management', 'business analysis', 'strategy'],
            'finance': ['financial analysis', 'accounting', 'excel', 'financial modeling'],
            'marketing': ['digital marketing', 'seo', 'social media', 'marketing'],
            'design': ['photoshop', 'illustrator', 'ui/ux', 'design']
        }
        
        field_scores = {}
        for field, indicators in field_indicators.items():
            score = sum(1 for indicator in indicators if any(indicator in skill for skill in skills_lower))
            if score > 0:
                field_scores[field] = score
        
        if field_scores:
            return max(field_scores, key=field_scores.get)
        
        return None
    
    def _analyze_degree_matching(self,
                               resume_education: List[Education],
                               job_requirements: List[Dict[str, Any]]) -> Tuple[float, List[EducationMatch]]:
        """Analyze degree level matching between resume and job requirements.
        
        Args:
            resume_education: List of education from resume
            job_requirements: List of job education requirements
            
        Returns:
            Tuple of (degree_match_score, list_of_matches)
        """
        matches = []
        degree_requirements = [req for req in job_requirements if req['type'] == 'degree']
        
        if not degree_requirements:
            return 100.0, matches  # No degree requirements
        
        # Get highest degree from resume
        highest_resume_degree = self._get_highest_degree(resume_education)
        
        total_score = 0.0
        for req in degree_requirements:
            required_level = req['level']
            
            # Check if resume meets requirement
            match_score = self._compare_degree_levels(highest_resume_degree, required_level)
            
            if match_score > 0:
                match = EducationMatch(
                    education_type='degree',
                    match_level='exact' if match_score >= 100 else 'equivalent',
                    confidence=match_score / 100.0,
                    resume_education=highest_resume_degree,
                    job_requirement=required_level,
                    relevance_score=match_score
                )
                matches.append(match)
            
            total_score += match_score
        
        # Average score across all degree requirements
        average_score = total_score / len(degree_requirements)
        
        return average_score, matches
    
    def _analyze_field_matching(self,
                              resume_education: List[Education],
                              job_requirements: List[Dict[str, Any]]) -> Tuple[float, List[EducationMatch]]:
        """Analyze field of study matching between resume and job requirements.
        
        Args:
            resume_education: List of education from resume
            job_requirements: List of job education requirements
            
        Returns:
            Tuple of (field_match_score, list_of_matches)
        """
        matches = []
        field_requirements = [req for req in job_requirements if req['type'] == 'field' and req.get('field')]
        
        if not field_requirements:
            return 100.0, matches  # No field requirements
        
        # Get all fields from resume
        resume_fields = [edu.major.lower() for edu in resume_education if edu.major]
        
        total_score = 0.0
        for req in field_requirements:
            required_field = req['field'].lower()
            
            # Find best matching field
            best_match_score = 0.0
            best_match_field = None
            
            for resume_field in resume_fields:
                match_score = self._compare_fields(resume_field, required_field)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_field = resume_field
            
            if best_match_score > 0:
                match = EducationMatch(
                    education_type='field',
                    match_level='exact' if best_match_score >= 90 else 'related',
                    confidence=best_match_score / 100.0,
                    resume_education=best_match_field,
                    job_requirement=required_field,
                    relevance_score=best_match_score
                )
                matches.append(match)
            
            total_score += best_match_score
        
        # Average score across all field requirements
        average_score = total_score / len(field_requirements)
        
        return average_score, matches
    
    def _analyze_institution_quality(self,
                                   resume_education: List[Education],
                                   job_requirements: List[Dict[str, Any]]) -> float:
        """Analyze institution quality and prestige.
        
        Args:
            resume_education: List of education from resume
            job_requirements: List of job education requirements
            
        Returns:
            Institution quality score (0-100)
        """
        if not resume_education:
            return 0.0
        
        institution_scores = []
        
        for edu in resume_education:
            if edu.institution:
                tier_score = self._get_institution_tier_score(edu.institution)
                institution_scores.append(tier_score)
        
        if not institution_scores:
            return 50.0  # Neutral score if no institutions
        
        # Return highest institution score
        return max(institution_scores)
    
    def _analyze_certification_matching(self,
                                      resume_certifications: List[Certification],
                                      job_requirements: List[Dict[str, Any]],
                                      job_text: str) -> Tuple[float, List[EducationMatch]]:
        """Analyze professional certification matching.
        
        Args:
            resume_certifications: List of certifications from resume
            job_requirements: List of job education requirements
            job_text: Original job text for additional analysis
            
        Returns:
            Tuple of (certification_score, list_of_matches)
        """
        matches = []
        cert_requirements = [req for req in job_requirements if req['type'] == 'certification']
        
        if not cert_requirements:
            return 100.0, matches  # No certification requirements
        
        # Get all certifications from resume
        resume_certs = [cert.name.lower() for cert in resume_certifications if cert.name]
        
        total_score = 0.0
        for req in cert_requirements:
            required_cert = req['name'].lower()
            
            # Find best matching certification
            best_match_score = 0.0
            best_match_cert = None
            
            for resume_cert in resume_certs:
                match_score = self._compare_certifications(resume_cert, required_cert)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_cert = resume_cert
            
            if best_match_score > 0:
                match = EducationMatch(
                    education_type='certification',
                    match_level='exact' if best_match_score >= 90 else 'equivalent',
                    confidence=best_match_score / 100.0,
                    resume_education=best_match_cert,
                    job_requirement=required_cert,
                    relevance_score=best_match_score
                )
                matches.append(match)
            
            # Weight by requirement importance
            weight = 1.0 if req.get('required', False) else 0.5
            total_score += best_match_score * weight
        
        # Calculate weighted average
        total_weight = sum(1.0 if req.get('required', False) else 0.5 for req in cert_requirements)
        average_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return average_score, matches
    
    def _get_highest_degree(self, education_list: List[Education]) -> str:
        """Get the highest degree level from education list.
        
        Args:
            education_list: List of education entries
            
        Returns:
            Highest degree level
        """
        if not education_list:
            return 'none'
        
        highest_level = 0
        highest_degree = 'none'
        
        for edu in education_list:
            if edu.degree:
                normalized_degree = self._normalize_degree_type(edu.degree)
                level = self.degree_hierarchy.get(normalized_degree, 0)
                
                if level > highest_level:
                    highest_level = level
                    highest_degree = normalized_degree
        
        return highest_degree
    
    def _normalize_degree_type(self, degree: str) -> str:
        """Normalize degree type to standard format.
        
        Args:
            degree: Raw degree string
            
        Returns:
            Normalized degree type
        """
        degree_lower = degree.lower().strip()
        
        # Check synonyms first
        for standard_degree, synonyms in self.degree_synonyms.items():
            if degree_lower in synonyms or degree_lower == standard_degree:
                return standard_degree
        
        # Fallback pattern matching
        if any(term in degree_lower for term in ['phd', 'ph.d', 'doctorate']):
            return 'phd'
        elif any(term in degree_lower for term in ['master', 'ms', 'm.s', 'ma', 'm.a']):
            return 'master'
        elif 'mba' in degree_lower:
            return 'mba'
        elif any(term in degree_lower for term in ['bachelor', 'bs', 'b.s', 'ba', 'b.a']):
            return 'bachelor'
        elif any(term in degree_lower for term in ['associate', 'aa', 'as']):
            return 'associate'
        elif any(term in degree_lower for term in ['certificate', 'cert']):
            return 'certificate'
        elif any(term in degree_lower for term in ['diploma', 'high school']):
            return 'diploma'
        
        return 'unknown'
    
    def _compare_degree_levels(self, resume_degree: str, required_degree: str) -> float:
        """Compare degree levels and return match score.
        
        Args:
            resume_degree: Degree from resume
            required_degree: Required degree from job
            
        Returns:
            Match score (0-100)
        """
        resume_level = self.degree_hierarchy.get(resume_degree, 0)
        required_level = self.degree_hierarchy.get(required_degree, 0)
        
        if resume_level >= required_level:
            # Meets or exceeds requirement
            if resume_level == required_level:
                return 100.0  # Perfect match
            else:
                return 110.0  # Exceeds requirement (bonus)
        else:
            # Below requirement
            if resume_level == 0:
                return 0.0  # No degree
            else:
                # Partial credit based on how close
                ratio = resume_level / required_level
                return ratio * 70.0  # Max 70% for below requirement
    
    def _compare_fields(self, resume_field: str, required_field: str) -> float:
        """Compare fields of study and return match score.
        
        Args:
            resume_field: Field from resume
            required_field: Required field from job
            
        Returns:
            Match score (0-100)
        """
        # Exact match
        if resume_field == required_field:
            return 100.0
        
        # Check field mappings for related fields
        for main_field, related_fields in self.field_mappings.items():
            if required_field in [main_field] + related_fields:
                if resume_field in [main_field] + related_fields:
                    if resume_field == main_field or required_field == main_field:
                        return 90.0  # Close match
                    else:
                        return 75.0  # Related field
        
        # Substring matching for partial matches
        if resume_field in required_field or required_field in resume_field:
            return 60.0
        
        # Fuzzy matching for similar terms
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, resume_field, required_field).ratio()
        if similarity > 0.7:
            return similarity * 80.0
        
        return 0.0
    
    def _compare_certifications(self, resume_cert: str, required_cert: str) -> float:
        """Compare certifications and return match score.
        
        Args:
            resume_cert: Certification from resume
            required_cert: Required certification from job
            
        Returns:
            Match score (0-100)
        """
        # Exact match
        if resume_cert == required_cert:
            return 100.0
        
        # Check for common abbreviations and variations
        cert_variations = {
            'pmp': ['project management professional', 'pmp certification'],
            'cissp': ['certified information systems security professional'],
            'cpa': ['certified public accountant'],
            'cfa': ['chartered financial analyst'],
            'aws certified': ['amazon web services certified', 'aws certification'],
            'azure certified': ['microsoft azure certified', 'azure certification'],
            'scrum master': ['certified scrum master', 'csm'],
            'agile': ['certified agile practitioner', 'agile certification']
        }
        
        for main_cert, variations in cert_variations.items():
            if (required_cert in [main_cert] + variations and 
                resume_cert in [main_cert] + variations):
                return 95.0
        
        # Substring matching
        if resume_cert in required_cert or required_cert in resume_cert:
            return 80.0
        
        # Category matching (e.g., both are cloud certifications)
        resume_category = self._get_certification_category(resume_cert)
        required_category = self._get_certification_category(required_cert)
        
        if resume_category and resume_category == required_category:
            return 60.0
        
        return 0.0
    
    def _get_certification_category(self, cert_name: str) -> Optional[str]:
        """Get the category of a certification.
        
        Args:
            cert_name: Certification name
            
        Returns:
            Certification category or None
        """
        cert_lower = cert_name.lower()
        
        for category, keywords in self.certification_categories.items():
            if any(keyword in cert_lower for keyword in keywords):
                return category
        
        return None
    
    def _get_institution_tier_score(self, institution: str) -> float:
        """Get institution tier score based on prestige.
        
        Args:
            institution: Institution name
            
        Returns:
            Tier score (0-100)
        """
        institution_lower = institution.lower()
        
        # Check against institution tiers
        for tier, institutions in self.institution_tiers.items():
            if any(inst in institution_lower for inst in institutions):
                tier_scores = {
                    'tier1': 100.0,
                    'tier2': 85.0,
                    'tier3': 70.0,
                    'tier4': 55.0
                }
                return tier_scores.get(tier, 50.0)
        
        # Default score for unknown institutions
        return 50.0
    
    def _identify_missing_requirements(self,
                                     job_requirements: List[Dict[str, Any]],
                                     matches: List[EducationMatch]) -> List[str]:
        """Identify missing education requirements.
        
        Args:
            job_requirements: List of job requirements
            matches: List of matched requirements
            
        Returns:
            List of missing requirement descriptions
        """
        missing = []
        matched_requirements = set()
        
        # Track what was matched
        for match in matches:
            matched_requirements.add(match.job_requirement)
        
        # Find unmatched requirements
        for req in job_requirements:
            if req.get('required', True):  # Only consider required items
                if req['type'] == 'degree':
                    req_key = req['level']
                elif req['type'] == 'field':
                    req_key = req.get('field', '')
                elif req['type'] == 'certification':
                    req_key = req.get('name', '')
                else:
                    continue
                
                if req_key and req_key not in matched_requirements:
                    missing.append(f"{req['type']}: {req_key}")
        
        return missing
    
    def _generate_alternative_paths(self,
                                  resume: ParsedResume,
                                  missing_requirements: List[str],
                                  job_requirements: List[Dict[str, Any]]) -> List[str]:
        """Generate alternative qualification paths for missing requirements.
        
        Args:
            resume: Parsed resume data
            missing_requirements: List of missing requirements
            job_requirements: List of job requirements
            
        Returns:
            List of alternative path suggestions
        """
        alternatives = []
        
        # Analyze what's missing and suggest alternatives
        for missing in missing_requirements:
            if missing.startswith('degree:'):
                degree_type = missing.split(':', 1)[1].strip()
                alternatives.extend(self._suggest_degree_alternatives(resume, degree_type))
            elif missing.startswith('field:'):
                field = missing.split(':', 1)[1].strip()
                alternatives.extend(self._suggest_field_alternatives(resume, field))
            elif missing.startswith('certification:'):
                cert = missing.split(':', 1)[1].strip()
                alternatives.extend(self._suggest_certification_alternatives(resume, cert))
        
        return list(set(alternatives))  # Remove duplicates
    
    def _suggest_degree_alternatives(self, resume: ParsedResume, required_degree: str) -> List[str]:
        """Suggest alternatives for missing degree requirements.
        
        Args:
            resume: Parsed resume data
            required_degree: Required degree type
            
        Returns:
            List of alternative suggestions
        """
        alternatives = []
        current_degree = self._get_highest_degree(resume.education)
        
        if current_degree == 'none':
            alternatives.append(f"Consider pursuing a {required_degree}'s degree")
            alternatives.append("Highlight relevant coursework, bootcamps, or self-study")
        elif self.degree_hierarchy.get(current_degree, 0) < self.degree_hierarchy.get(required_degree, 0):
            alternatives.append(f"Consider upgrading to a {required_degree}'s degree")
            alternatives.append("Emphasize professional experience as equivalent qualification")
        
        # Experience-based alternatives
        total_experience = sum(exp.duration_months for exp in resume.experience) / 12.0
        if total_experience >= 5:
            alternatives.append("Highlight extensive professional experience as degree equivalent")
        
        return alternatives
    
    def _suggest_field_alternatives(self, resume: ParsedResume, required_field: str) -> List[str]:
        """Suggest alternatives for missing field requirements.
        
        Args:
            resume: Parsed resume data
            required_field: Required field of study
            
        Returns:
            List of alternative suggestions
        """
        alternatives = []
        
        # Check if current field is related
        current_fields = [edu.major for edu in resume.education if edu.major]
        
        if current_fields:
            alternatives.append(f"Emphasize transferable skills from {current_fields[0]} to {required_field}")
        
        # Skill-based alternatives
        relevant_skills = self._find_relevant_skills(resume.skills, required_field)
        if relevant_skills:
            alternatives.append(f"Highlight relevant skills: {', '.join(relevant_skills[:3])}")
        
        # Additional education suggestions
        alternatives.append(f"Consider additional coursework or certification in {required_field}")
        
        return alternatives
    
    def _suggest_certification_alternatives(self, resume: ParsedResume, required_cert: str) -> List[str]:
        """Suggest alternatives for missing certification requirements.
        
        Args:
            resume: Parsed resume data
            required_cert: Required certification
            
        Returns:
            List of alternative suggestions
        """
        alternatives = []
        
        # Direct suggestion
        alternatives.append(f"Consider obtaining {required_cert} certification")
        
        # Related certifications
        cert_category = self._get_certification_category(required_cert)
        if cert_category:
            alternatives.append(f"Highlight any {cert_category} experience or related certifications")
        
        # Experience-based alternatives
        relevant_experience = self._find_relevant_experience(resume.experience, required_cert)
        if relevant_experience:
            alternatives.append("Emphasize relevant professional experience in lieu of certification")
        
        return alternatives
    
    def _find_relevant_skills(self, skills: List[str], field: str) -> List[str]:
        """Find skills relevant to a field of study.
        
        Args:
            skills: List of skills from resume
            field: Field of study
            
        Returns:
            List of relevant skills
        """
        field_lower = field.lower()
        relevant = []
        
        # Field-specific skill mappings
        field_skill_mappings = {
            'computer science': ['programming', 'python', 'java', 'javascript', 'algorithms'],
            'data science': ['machine learning', 'statistics', 'python', 'r', 'sql'],
            'business': ['project management', 'analysis', 'strategy', 'leadership'],
            'engineering': ['matlab', 'autocad', 'design', 'analysis'],
            'finance': ['financial analysis', 'excel', 'modeling', 'accounting']
        }
        
        # Find matching field
        for mapped_field, mapped_skills in field_skill_mappings.items():
            if mapped_field in field_lower:
                for skill in skills:
                    if any(mapped_skill in skill.lower() for mapped_skill in mapped_skills):
                        relevant.append(skill)
                break
        
        return relevant[:5]  # Return top 5
    
    def _find_relevant_experience(self, experiences: List, cert_name: str) -> bool:
        """Check if experience is relevant to certification.
        
        Args:
            experiences: List of work experiences
            cert_name: Certification name
            
        Returns:
            True if relevant experience found
        """
        cert_lower = cert_name.lower()
        
        for exp in experiences:
            exp_text = f"{exp.job_title} {exp.description}".lower()
            
            # Check for relevant keywords
            if 'project management' in cert_lower and 'project' in exp_text:
                return True
            elif 'security' in cert_lower and 'security' in exp_text:
                return True
            elif 'aws' in cert_lower and ('aws' in exp_text or 'cloud' in exp_text):
                return True
            elif 'scrum' in cert_lower and ('agile' in exp_text or 'scrum' in exp_text):
                return True
        
        return False
    
    def _extract_required_degree_level(self, job_requirements: List[Dict[str, Any]]) -> str:
        """Extract the required degree level from job requirements.
        
        Args:
            job_requirements: List of job requirements
            
        Returns:
            Required degree level or 'none'
        """
        degree_requirements = [req for req in job_requirements if req['type'] == 'degree']
        
        if not degree_requirements:
            return 'none'
        
        # Return highest required degree
        highest_level = 0
        highest_degree = 'none'
        
        for req in degree_requirements:
            level = self.degree_hierarchy.get(req['level'], 0)
            if level > highest_level:
                highest_level = level
                highest_degree = req['level']
        
        return highest_degree
    
    def _calculate_overall_education_score(self,
                                         degree_score: float,
                                         field_score: float,
                                         institution_score: float,
                                         certification_score: float) -> float:
        """Calculate overall education match score.
        
        Args:
            degree_score: Degree match score
            field_score: Field match score
            institution_score: Institution score
            certification_score: Certification score
            
        Returns:
            Overall education score (0-100)
        """
        # Weighted combination
        weights = {
            'degree': 0.4,
            'field': 0.3,
            'certification': 0.2,
            'institution': 0.1
        }
        
        overall_score = (
            degree_score * weights['degree'] +
            field_score * weights['field'] +
            certification_score * weights['certification'] +
            institution_score * weights['institution']
        )
        
        return min(100.0, max(0.0, overall_score))
    
    def _calculate_education_confidence(self,
                                      education: List[Education],
                                      certifications: List[Certification],
                                      num_requirements: int) -> float:
        """Calculate confidence score for education analysis.
        
        Args:
            education: List of education entries
            certifications: List of certifications
            num_requirements: Number of job requirements
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Amount of education data
        edu_data_factor = min(1.0, len(education) / 2.0)  # Optimal around 2+ entries
        confidence_factors.append(edu_data_factor)
        
        # Factor 2: Quality of education data
        quality_scores = []
        for edu in education:
            quality = 0.0
            if edu.degree: quality += 0.3
            if edu.major: quality += 0.3
            if edu.institution: quality += 0.2
            if edu.graduation_date: quality += 0.2
            quality_scores.append(quality)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        confidence_factors.append(avg_quality)
        
        # Factor 3: Certification data
        cert_factor = min(1.0, len(certifications) / 3.0)  # Bonus for certifications
        confidence_factors.append(cert_factor * 0.5)  # Lower weight
        
        # Factor 4: Number of requirements to match
        req_factor = min(1.0, num_requirements / 5.0) if num_requirements > 0 else 1.0
        confidence_factors.append(req_factor)
        
        # Combine factors
        overall_confidence = np.mean(confidence_factors)
        return min(1.0, max(0.0, overall_confidence))
    
    def _load_field_mappings(self) -> Dict[str, List[str]]:
        """Load field of study mappings.
        
        Returns:
            Dictionary mapping main fields to related fields
        """
        return {
            'computer science': [
                'software engineering', 'information technology', 'computer engineering',
                'information systems', 'software development'
            ],
            'data science': [
                'statistics', 'mathematics', 'data analytics', 'machine learning',
                'artificial intelligence', 'data engineering'
            ],
            'business administration': [
                'business', 'management', 'business management', 'administration',
                'organizational management'
            ],
            'engineering': [
                'mechanical engineering', 'electrical engineering', 'civil engineering',
                'chemical engineering', 'industrial engineering'
            ],
            'finance': [
                'accounting', 'economics', 'financial management', 'business finance',
                'financial economics'
            ],
            'marketing': [
                'digital marketing', 'business marketing', 'communications',
                'advertising', 'public relations'
            ],
            'psychology': [
                'organizational psychology', 'human resources', 'behavioral science',
                'social psychology'
            ]
        }
    
    def _load_institution_tiers(self) -> Dict[str, List[str]]:
        """Load institution prestige tiers.
        
        Returns:
            Dictionary mapping tiers to institution keywords
        """
        return {
            'tier1': [
                'harvard', 'stanford', 'mit', 'caltech', 'princeton', 'yale',
                'columbia', 'university of chicago', 'penn', 'northwestern'
            ],
            'tier2': [
                'berkeley', 'ucla', 'michigan', 'virginia', 'carnegie mellon',
                'duke', 'dartmouth', 'brown', 'cornell', 'johns hopkins'
            ],
            'tier3': [
                'state university', 'university of', 'college of', 'institute of technology'
            ],
            'tier4': [
                'community college', 'technical college', 'vocational school'
            ]
        }
    
    def _load_degree_synonyms(self) -> Dict[str, List[str]]:
        """Load degree synonyms and variations.
        
        Returns:
            Dictionary mapping standard degrees to synonyms
        """
        return {
            'bachelor': ['ba', 'bs', 'b.a.', 'b.s.', 'bachelor\'s', 'bachelors', 'undergraduate'],
            'master': ['ma', 'ms', 'm.a.', 'm.s.', 'master\'s', 'masters', 'graduate'],
            'mba': ['master of business administration', 'masters in business'],
            'phd': ['ph.d.', 'ph.d', 'doctorate', 'doctoral', 'doctor of philosophy'],
            'associate': ['aa', 'as', 'a.a.', 'a.s.', 'associate\'s'],
            'certificate': ['cert', 'certification', 'professional certificate'],
            'diploma': ['high school diploma', 'hs diploma', 'secondary education']
        }
    
    def _load_certification_categories(self) -> Dict[str, List[str]]:
        """Load certification categories.
        
        Returns:
            Dictionary mapping categories to keywords
        """
        return {
            'project_management': ['pmp', 'project management', 'scrum', 'agile', 'prince2'],
            'cloud': ['aws', 'azure', 'google cloud', 'gcp', 'cloud'],
            'security': ['cissp', 'security', 'cybersecurity', 'information security'],
            'finance': ['cpa', 'cfa', 'frm', 'financial', 'accounting'],
            'it': ['comptia', 'cisco', 'microsoft', 'oracle', 'vmware'],
            'quality': ['six sigma', 'lean', 'quality management', 'iso'],
            'data': ['data science', 'analytics', 'big data', 'machine learning']
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for education scoring.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._education_stats.copy()
        
        # Calculate success rates
        total = stats['total_analyses']
        if total > 0:
            stats['degree_match_rate'] = stats['degree_matches'] / total
            stats['field_match_rate'] = stats['field_matches'] / total
            stats['certification_match_rate'] = stats['certification_matches'] / total
            stats['institution_match_rate'] = stats['institution_matches'] / total
        else:
            stats['degree_match_rate'] = 0.0
            stats['field_match_rate'] = 0.0
            stats['certification_match_rate'] = 0.0
            stats['institution_match_rate'] = 0.0
        
        # Add configuration info
        stats['configuration'] = {
            'degree_levels': len(self.degree_hierarchy),
            'field_mappings': len(self.field_mappings),
            'institution_tiers': len(self.institution_tiers),
            'certification_categories': len(self.certification_categories)
        }
        
        return stats