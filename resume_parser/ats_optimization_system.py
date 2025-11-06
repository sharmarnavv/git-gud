"""
ATS (Applicant Tracking System) optimization system for resume improvement.

This module provides functionality to analyze resumes for ATS compatibility
and generate specific recommendations for improving keyword matching,
formatting, and structure to pass through ATS filters effectively.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import Counter
from enum import Enum

from .resume_interfaces import ParsedResume
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


class ATSIssueType(Enum):
    """Types of ATS compatibility issues."""
    MISSING_KEYWORDS = "missing_keywords"
    KEYWORD_DENSITY = "keyword_density"
    SECTION_HEADERS = "section_headers"
    FILE_FORMAT = "file_format"
    FORMATTING = "formatting"
    STRUCTURE = "structure"


class KeywordType(Enum):
    """Types of keywords for ATS optimization."""
    TECHNICAL_SKILLS = "technical_skills"
    SOFT_SKILLS = "soft_skills"
    JOB_TITLES = "job_titles"
    INDUSTRY_TERMS = "industry_terms"
    CERTIFICATIONS = "certifications"
    TOOLS_TECHNOLOGIES = "tools_technologies"


@dataclass
class KeywordAnalysis:
    """Analysis of keyword usage in resume vs job description.
    
    Attributes:
        keyword: The keyword being analyzed
        keyword_type: Type of keyword
        job_frequency: Frequency in job description
        resume_frequency: Frequency in resume
        importance_score: Importance score (0-1)
        variations: List of keyword variations found
        context_locations: Where the keyword appears in resume
    """
    keyword: str
    keyword_type: KeywordType
    job_frequency: int
    resume_frequency: int
    importance_score: float
    variations: List[str] = field(default_factory=list)
    context_locations: List[str] = field(default_factory=list)


@dataclass
class ATSSuggestion:
    """ATS optimization suggestion.
    
    Attributes:
        suggestion_type: Type of ATS suggestion
        title: Brief title of the suggestion
        description: Detailed description
        priority: Priority level (high, medium, low)
        impact_score: Expected impact on ATS compatibility (0-1)
        implementation_effort: Effort required (easy, medium, hard)
        specific_actions: List of specific actions to take
        keywords_to_add: Keywords that should be added
        integration_examples: Examples of how to integrate keywords naturally
    """
    suggestion_type: str
    title: str
    description: str
    priority: str
    impact_score: float
    implementation_effort: str
    specific_actions: List[str] = field(default_factory=list)
    keywords_to_add: List[str] = field(default_factory=list)
    integration_examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeywordExtractor:
    """Extracts and analyzes keywords from job descriptions and resumes."""
    
    def __init__(self):
        """Initialize the keyword extractor."""
        # Common ATS-friendly section headers
        self.ats_section_headers = {
            'contact': ['Contact Information', 'Contact', 'Personal Information'],
            'summary': ['Professional Summary', 'Summary', 'Profile', 'Objective'],
            'skills': ['Skills', 'Technical Skills', 'Core Competencies', 'Qualifications'],
            'experience': ['Work Experience', 'Professional Experience', 'Experience', 'Employment History'],
            'education': ['Education', 'Academic Background', 'Educational Qualifications'],
            'certifications': ['Certifications', 'Professional Certifications', 'Licenses']
        }
        
        # Common soft skills keywords
        self.soft_skills_keywords = {
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'creative', 'detail-oriented', 'organized', 'adaptable', 'collaborative',
            'initiative', 'time-management', 'critical-thinking', 'interpersonal',
            'presentation', 'negotiation', 'mentoring', 'coaching', 'strategic'
        }
        
        # Common technical skill patterns
        self.technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (SQL, API, etc.)
            r'\b\w+\.\w+\b',   # Technologies with dots (Node.js, etc.)
            r'\b\w+\+\+?\b',   # Programming languages (C++, etc.)
            r'\b\w+#\b'        # Languages with # (C#, F#, etc.)
        ]
    
    def extract_job_keywords(self, job_description: ParsedJobDescription) -> Dict[KeywordType, List[str]]:
        """Extract keywords from job description categorized by type.
        
        Args:
            job_description: Parsed job description
            
        Returns:
            Dictionary mapping keyword types to lists of keywords
        """
        try:
            logger.debug("Extracting keywords from job description")
            
            keywords = {
                KeywordType.TECHNICAL_SKILLS: [],
                KeywordType.SOFT_SKILLS: [],
                KeywordType.JOB_TITLES: [],
                KeywordType.INDUSTRY_TERMS: [],
                KeywordType.CERTIFICATIONS: [],
                KeywordType.TOOLS_TECHNOLOGIES: []
            }
            
            # Extract from skills_required
            if job_description.skills_required:
                for skill in job_description.skills_required:
                    skill_lower = skill.lower()
                    
                    # Categorize skill
                    if skill_lower in self.soft_skills_keywords:
                        keywords[KeywordType.SOFT_SKILLS].append(skill)
                    elif self._is_technical_skill(skill):
                        keywords[KeywordType.TECHNICAL_SKILLS].append(skill)
                    else:
                        keywords[KeywordType.TOOLS_TECHNOLOGIES].append(skill)
            
            # Extract from job title
            job_title = job_description.metadata.get('job_title', '') if job_description.metadata else ''
            if job_title:
                # Extract job title keywords
                title_keywords = self._extract_title_keywords(job_title)
                keywords[KeywordType.JOB_TITLES].extend(title_keywords)
            
            # Extract from description text
            description_text = ""
            if hasattr(job_description, 'description') and job_description.description:
                description_text = job_description.description
            elif job_description.metadata and 'description' in job_description.metadata:
                description_text = job_description.metadata['description']
            
            if description_text:
                desc_keywords = self._extract_description_keywords(description_text)
                for keyword_type, keyword_list in desc_keywords.items():
                    keywords[keyword_type].extend(keyword_list)
            
            # Remove duplicates and clean up
            for keyword_type in keywords:
                keywords[keyword_type] = list(set(keywords[keyword_type]))
                keywords[keyword_type] = [kw for kw in keywords[keyword_type] if len(kw.strip()) > 2]
            
            logger.debug(f"Extracted {sum(len(kws) for kws in keywords.values())} keywords from job description")
            return keywords
            
        except Exception as e:
            logger.error(f"Job keyword extraction failed: {e}")
            return {kt: [] for kt in KeywordType}
    
    def extract_resume_keywords(self, resume: ParsedResume) -> Dict[KeywordType, List[str]]:
        """Extract keywords from resume categorized by type.
        
        Args:
            resume: Parsed resume data
            
        Returns:
            Dictionary mapping keyword types to lists of keywords
        """
        try:
            logger.debug("Extracting keywords from resume")
            
            keywords = {
                KeywordType.TECHNICAL_SKILLS: [],
                KeywordType.SOFT_SKILLS: [],
                KeywordType.JOB_TITLES: [],
                KeywordType.INDUSTRY_TERMS: [],
                KeywordType.CERTIFICATIONS: [],
                KeywordType.TOOLS_TECHNOLOGIES: []
            }
            
            # Extract from skills
            if resume.skills:
                for skill in resume.skills:
                    skill_lower = skill.lower()
                    
                    if skill_lower in self.soft_skills_keywords:
                        keywords[KeywordType.SOFT_SKILLS].append(skill)
                    elif self._is_technical_skill(skill):
                        keywords[KeywordType.TECHNICAL_SKILLS].append(skill)
                    else:
                        keywords[KeywordType.TOOLS_TECHNOLOGIES].append(skill)
            
            # Extract from job titles in experience
            if resume.experience:
                for exp in resume.experience:
                    if exp.job_title:
                        title_keywords = self._extract_title_keywords(exp.job_title)
                        keywords[KeywordType.JOB_TITLES].extend(title_keywords)
                    
                    # Extract from experience descriptions
                    if exp.description:
                        desc_keywords = self._extract_description_keywords(exp.description)
                        for keyword_type, keyword_list in desc_keywords.items():
                            keywords[keyword_type].extend(keyword_list)
            
            # Extract from certifications
            if resume.certifications:
                for cert in resume.certifications:
                    if cert.name:
                        keywords[KeywordType.CERTIFICATIONS].append(cert.name)
            
            # Remove duplicates
            for keyword_type in keywords:
                keywords[keyword_type] = list(set(keywords[keyword_type]))
            
            logger.debug(f"Extracted {sum(len(kws) for kws in keywords.values())} keywords from resume")
            return keywords
            
        except Exception as e:
            logger.error(f"Resume keyword extraction failed: {e}")
            return {kt: [] for kt in KeywordType}
    
    def _is_technical_skill(self, skill: str) -> bool:
        """Determine if a skill is technical based on patterns."""
        skill_lower = skill.lower()
        
        # Check against technical patterns
        for pattern in self.technical_patterns:
            if re.search(pattern, skill):
                return True
        
        # Common technical keywords
        technical_indicators = [
            'programming', 'development', 'software', 'database', 'framework',
            'library', 'api', 'cloud', 'devops', 'machine learning', 'ai',
            'data', 'analytics', 'web', 'mobile', 'frontend', 'backend'
        ]
        
        return any(indicator in skill_lower for indicator in technical_indicators)
    
    def _extract_title_keywords(self, title: str) -> List[str]:
        """Extract keywords from job titles."""
        # Common job title keywords
        title_keywords = []
        
        # Split title and extract meaningful words
        words = re.findall(r'\b[A-Za-z]+\b', title)
        
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'with', 'by'}
        meaningful_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        title_keywords.extend(meaningful_words)
        
        # Add full title if it's a common job title pattern
        if any(keyword in title.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist']):
            title_keywords.append(title)
        
        return title_keywords
    
    def _extract_description_keywords(self, text: str) -> Dict[KeywordType, List[str]]:
        """Extract keywords from description text."""
        keywords = {
            KeywordType.TECHNICAL_SKILLS: [],
            KeywordType.SOFT_SKILLS: [],
            KeywordType.INDUSTRY_TERMS: [],
            KeywordType.TOOLS_TECHNOLOGIES: []
        }
        
        text_lower = text.lower()
        
        # Extract technical skills using patterns
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            keywords[KeywordType.TECHNICAL_SKILLS].extend(matches)
        
        # Extract soft skills
        for soft_skill in self.soft_skills_keywords:
            if soft_skill in text_lower:
                keywords[KeywordType.SOFT_SKILLS].append(soft_skill)
        
        # Extract industry terms (simple approach - can be enhanced)
        industry_patterns = [
            r'\b(?:agile|scrum|kanban|devops|ci/cd)\b',
            r'\b(?:saas|paas|iaas|cloud|aws|azure|gcp)\b',
            r'\b(?:api|rest|graphql|microservices)\b'
        ]
        
        for pattern in industry_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords[KeywordType.INDUSTRY_TERMS].extend(matches)
        
        return keywords


class ATSCompatibilityAnalyzer:
    """Analyzes resume for ATS compatibility issues."""
    
    def __init__(self):
        """Initialize the ATS compatibility analyzer."""
        self.keyword_extractor = KeywordExtractor()
        
        # ATS-unfriendly formatting patterns
        self.ats_unfriendly_patterns = {
            'special_characters': r'[★☆♦♠♣♥◆◇○●◎◉]',
            'complex_formatting': r'[│┌┐└┘├┤┬┴┼]',
            'unusual_bullets': r'[►▶▷▸▹▻▼▽△▲]'
        }
        
        # Recommended keyword density ranges
        self.keyword_density_ranges = {
            'technical_skills': (0.02, 0.08),  # 2-8% of total words
            'soft_skills': (0.01, 0.04),      # 1-4% of total words
            'job_titles': (0.005, 0.02)       # 0.5-2% of total words
        }
    
    def analyze_ats_compatibility(self, 
                                resume: ParsedResume, 
                                job_description: ParsedJobDescription) -> Dict[str, Any]:
        """Perform comprehensive ATS compatibility analysis.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description
            
        Returns:
            Dictionary with ATS compatibility analysis results
        """
        try:
            logger.info("Analyzing ATS compatibility")
            
            # Extract keywords from both resume and job description
            job_keywords = self.keyword_extractor.extract_job_keywords(job_description)
            resume_keywords = self.keyword_extractor.extract_resume_keywords(resume)
            
            # Perform keyword matching analysis
            keyword_analysis = self._analyze_keyword_matching(job_keywords, resume_keywords)
            
            # Analyze section headers
            section_analysis = self._analyze_section_headers(resume)
            
            # Analyze formatting compatibility
            formatting_analysis = self._analyze_formatting_compatibility(resume)
            
            # Calculate overall ATS score
            ats_score = self._calculate_ats_score(keyword_analysis, section_analysis, formatting_analysis)
            
            analysis_result = {
                'ats_compatibility_score': ats_score,
                'keyword_analysis': keyword_analysis,
                'section_analysis': section_analysis,
                'formatting_analysis': formatting_analysis,
                'missing_keywords': self._identify_missing_keywords(job_keywords, resume_keywords),
                'keyword_density_issues': self._analyze_keyword_density(resume, resume_keywords),
                'recommendations': []
            }
            
            logger.info(f"ATS compatibility analysis completed with score: {ats_score:.2f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"ATS compatibility analysis failed: {e}")
            return {
                'ats_compatibility_score': 0.0,
                'keyword_analysis': {},
                'section_analysis': {},
                'formatting_analysis': {},
                'missing_keywords': {},
                'keyword_density_issues': [],
                'recommendations': []
            }
    
    def _analyze_keyword_matching(self, 
                                job_keywords: Dict[KeywordType, List[str]], 
                                resume_keywords: Dict[KeywordType, List[str]]) -> Dict[str, Any]:
        """Analyze keyword matching between job and resume."""
        try:
            matching_analysis = {
                'total_job_keywords': 0,
                'total_resume_keywords': 0,
                'matching_keywords': 0,
                'match_percentage': 0.0,
                'matches_by_type': {},
                'missing_by_type': {}
            }
            
            total_job_kw = sum(len(keywords) for keywords in job_keywords.values())
            total_resume_kw = sum(len(keywords) for keywords in resume_keywords.values())
            total_matches = 0
            
            matching_analysis['total_job_keywords'] = total_job_kw
            matching_analysis['total_resume_keywords'] = total_resume_kw
            
            # Analyze matches by keyword type
            for keyword_type in KeywordType:
                job_kw_set = set(kw.lower() for kw in job_keywords.get(keyword_type, []))
                resume_kw_set = set(kw.lower() for kw in resume_keywords.get(keyword_type, []))
                
                matches = job_kw_set.intersection(resume_kw_set)
                missing = job_kw_set - resume_kw_set
                
                matching_analysis['matches_by_type'][keyword_type.value] = {
                    'job_keywords': len(job_kw_set),
                    'resume_keywords': len(resume_kw_set),
                    'matches': len(matches),
                    'missing': len(missing),
                    'match_rate': len(matches) / len(job_kw_set) if job_kw_set else 0.0,
                    'matched_keywords': list(matches),
                    'missing_keywords': list(missing)
                }
                
                total_matches += len(matches)
            
            matching_analysis['matching_keywords'] = total_matches
            matching_analysis['match_percentage'] = (total_matches / total_job_kw) if total_job_kw > 0 else 0.0
            
            return matching_analysis
            
        except Exception as e:
            logger.error(f"Keyword matching analysis failed: {e}")
            return {}
    
    def _analyze_section_headers(self, resume: ParsedResume) -> Dict[str, Any]:
        """Analyze section headers for ATS compatibility."""
        try:
            section_analysis = {
                'ats_friendly_headers': [],
                'problematic_headers': [],
                'missing_sections': [],
                'header_score': 0.0
            }
            
            # Check for presence of standard sections
            standard_sections = ['contact', 'skills', 'experience', 'education']
            present_sections = []
            
            if resume.contact_info and (resume.contact_info.name or resume.contact_info.email):
                present_sections.append('contact')
            if resume.skills:
                present_sections.append('skills')
            if resume.experience:
                present_sections.append('experience')
            if resume.education:
                present_sections.append('education')
            
            # Identify missing sections
            missing_sections = [section for section in standard_sections if section not in present_sections]
            section_analysis['missing_sections'] = missing_sections
            
            # Calculate header score
            section_score = len(present_sections) / len(standard_sections)
            section_analysis['header_score'] = section_score
            
            # Add recommendations for ATS-friendly headers
            section_analysis['ats_friendly_headers'] = present_sections
            
            return section_analysis
            
        except Exception as e:
            logger.error(f"Section header analysis failed: {e}")
            return {}
    
    def _analyze_formatting_compatibility(self, resume: ParsedResume) -> Dict[str, Any]:
        """Analyze formatting for ATS compatibility."""
        try:
            formatting_analysis = {
                'compatibility_issues': [],
                'formatting_score': 1.0,
                'recommendations': []
            }
            
            # Check for ATS-unfriendly characters in text content
            all_text = ""
            
            # Collect all text from resume
            if resume.experience:
                for exp in resume.experience:
                    if exp.description:
                        all_text += exp.description + " "
                    if exp.job_title:
                        all_text += exp.job_title + " "
            
            if resume.skills:
                all_text += " ".join(resume.skills) + " "
            
            # Check for problematic patterns
            issues_found = []
            for pattern_name, pattern in self.ats_unfriendly_patterns.items():
                if re.search(pattern, all_text):
                    issues_found.append(pattern_name)
                    formatting_analysis['compatibility_issues'].append({
                        'type': pattern_name,
                        'description': f"Found ATS-unfriendly {pattern_name.replace('_', ' ')}"
                    })
            
            # Calculate formatting score
            if issues_found:
                penalty = len(issues_found) * 0.2
                formatting_analysis['formatting_score'] = max(0.0, 1.0 - penalty)
            
            return formatting_analysis
            
        except Exception as e:
            logger.error(f"Formatting compatibility analysis failed: {e}")
            return {}
    
    def _identify_missing_keywords(self, 
                                 job_keywords: Dict[KeywordType, List[str]], 
                                 resume_keywords: Dict[KeywordType, List[str]]) -> Dict[str, List[str]]:
        """Identify missing keywords by type."""
        missing_keywords = {}
        
        for keyword_type in KeywordType:
            job_kw_set = set(kw.lower() for kw in job_keywords.get(keyword_type, []))
            resume_kw_set = set(kw.lower() for kw in resume_keywords.get(keyword_type, []))
            
            missing = job_kw_set - resume_kw_set
            if missing:
                missing_keywords[keyword_type.value] = list(missing)
        
        return missing_keywords
    
    def _analyze_keyword_density(self, 
                               resume: ParsedResume, 
                               resume_keywords: Dict[KeywordType, List[str]]) -> List[Dict[str, Any]]:
        """Analyze keyword density issues."""
        density_issues = []
        
        try:
            # Calculate total word count in resume
            total_words = 0
            
            if resume.experience:
                for exp in resume.experience:
                    if exp.description:
                        total_words += len(exp.description.split())
            
            if total_words == 0:
                return density_issues
            
            # Check density for each keyword type
            for keyword_type, keywords in resume_keywords.items():
                if not keywords:
                    continue
                
                keyword_count = len(keywords)
                density = keyword_count / total_words
                
                # Get recommended range
                recommended_range = self.keyword_density_ranges.get(
                    keyword_type.value.replace('_skills', '_skills'), 
                    (0.01, 0.05)
                )
                
                min_density, max_density = recommended_range
                
                if density < min_density:
                    density_issues.append({
                        'keyword_type': keyword_type.value,
                        'issue': 'low_density',
                        'current_density': density,
                        'recommended_range': recommended_range,
                        'suggestion': f"Consider adding more {keyword_type.value.replace('_', ' ')} keywords"
                    })
                elif density > max_density:
                    density_issues.append({
                        'keyword_type': keyword_type.value,
                        'issue': 'high_density',
                        'current_density': density,
                        'recommended_range': recommended_range,
                        'suggestion': f"Consider reducing {keyword_type.value.replace('_', ' ')} keyword density"
                    })
            
            return density_issues
            
        except Exception as e:
            logger.error(f"Keyword density analysis failed: {e}")
            return []
    
    def _calculate_ats_score(self, 
                           keyword_analysis: Dict[str, Any], 
                           section_analysis: Dict[str, Any], 
                           formatting_analysis: Dict[str, Any]) -> float:
        """Calculate overall ATS compatibility score."""
        try:
            # Keyword matching score (50% weight)
            keyword_score = keyword_analysis.get('match_percentage', 0.0)
            
            # Section structure score (30% weight)
            section_score = section_analysis.get('header_score', 0.0)
            
            # Formatting compatibility score (20% weight)
            formatting_score = formatting_analysis.get('formatting_score', 1.0)
            
            # Calculate weighted average
            ats_score = (keyword_score * 0.5) + (section_score * 0.3) + (formatting_score * 0.2)
            
            return round(ats_score, 2)
            
        except Exception as e:
            logger.error(f"ATS score calculation failed: {e}")
            return 0.0


class ATSOptimizationSystem:
    """Main system for generating ATS optimization suggestions."""
    
    def __init__(self):
        """Initialize the ATS optimization system."""
        self.compatibility_analyzer = ATSCompatibilityAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        
        # Natural integration templates for keywords
        self.integration_templates = {
            KeywordType.TECHNICAL_SKILLS: [
                "Utilized {keyword} to develop and implement solutions",
                "Proficient in {keyword} with {years}+ years of experience",
                "Applied {keyword} expertise to optimize system performance",
                "Leveraged {keyword} for data analysis and reporting"
            ],
            KeywordType.SOFT_SKILLS: [
                "Demonstrated strong {keyword} skills while leading team projects",
                "Applied {keyword} abilities to resolve complex challenges",
                "Utilized {keyword} skills to improve team collaboration",
                "Showcased {keyword} through successful project delivery"
            ],
            KeywordType.TOOLS_TECHNOLOGIES: [
                "Experienced with {keyword} for project development",
                "Implemented solutions using {keyword} and related technologies",
                "Proficient in {keyword} for data processing and analysis",
                "Used {keyword} to streamline workflow processes"
            ]
        }
        
        logger.info("ATS optimization system initialized")
    
    def generate_ats_suggestions(self, 
                               resume: ParsedResume, 
                               job_description: ParsedJobDescription) -> List[ATSSuggestion]:
        """Generate comprehensive ATS optimization suggestions.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description
            
        Returns:
            List of ATS optimization suggestions
        """
        try:
            logger.info("Generating ATS optimization suggestions")
            
            # Perform ATS compatibility analysis
            ats_analysis = self.compatibility_analyzer.analyze_ats_compatibility(resume, job_description)
            
            suggestions = []
            
            # Generate keyword optimization suggestions
            keyword_suggestions = self._generate_keyword_suggestions(resume, job_description, ats_analysis)
            suggestions.extend(keyword_suggestions)
            
            # Generate section header suggestions
            header_suggestions = self._generate_header_suggestions(resume, ats_analysis)
            suggestions.extend(header_suggestions)
            
            # Generate formatting suggestions
            formatting_suggestions = self._generate_formatting_suggestions(resume, ats_analysis)
            suggestions.extend(formatting_suggestions)
            
            # Generate file format suggestions
            file_format_suggestions = self._generate_file_format_suggestions()
            suggestions.extend(file_format_suggestions)
            
            # Rank and filter suggestions
            ranked_suggestions = self._rank_ats_suggestions(suggestions)
            
            logger.info(f"Generated {len(ranked_suggestions)} ATS optimization suggestions")
            return ranked_suggestions
            
        except Exception as e:
            logger.error(f"ATS suggestion generation failed: {e}")
            return []
    
    def _generate_keyword_suggestions(self, 
                                    resume: ParsedResume, 
                                    job_description: ParsedJobDescription,
                                    ats_analysis: Dict[str, Any]) -> List[ATSSuggestion]:
        """Generate keyword optimization suggestions."""
        suggestions = []
        
        try:
            missing_keywords = ats_analysis.get('missing_keywords', {})
            keyword_analysis = ats_analysis.get('keyword_analysis', {})
            
            # High-priority missing technical skills
            technical_missing = missing_keywords.get('technical_skills', [])
            if technical_missing:
                suggestions.append(ATSSuggestion(
                    suggestion_type="keyword_optimization",
                    title="Add Missing Technical Skills",
                    description=f"Your resume is missing {len(technical_missing)} important technical skills mentioned in the job description.",
                    priority="high",
                    impact_score=0.8,
                    implementation_effort="medium",
                    specific_actions=[
                        "Review the missing technical skills and identify which ones you have experience with",
                        "Add relevant technical skills to your Skills section",
                        "Integrate technical skills naturally into your experience descriptions",
                        "Consider mentioning related technologies or frameworks you've used"
                    ],
                    keywords_to_add=technical_missing[:8],  # Limit to top 8
                    integration_examples=self._generate_integration_examples(
                        technical_missing[:3], KeywordType.TECHNICAL_SKILLS
                    )
                ))
            
            # Missing soft skills
            soft_missing = missing_keywords.get('soft_skills', [])
            if soft_missing:
                suggestions.append(ATSSuggestion(
                    suggestion_type="keyword_optimization",
                    title="Incorporate Soft Skills Keywords",
                    description=f"Add {len(soft_missing)} soft skills that are valued for this position.",
                    priority="medium",
                    impact_score=0.6,
                    implementation_effort="easy",
                    specific_actions=[
                        "Identify soft skills you possess from the missing list",
                        "Integrate soft skills naturally into experience descriptions",
                        "Provide specific examples of how you've demonstrated these skills",
                        "Avoid simply listing soft skills without context"
                    ],
                    keywords_to_add=soft_missing[:5],
                    integration_examples=self._generate_integration_examples(
                        soft_missing[:3], KeywordType.SOFT_SKILLS
                    )
                ))
            
            # Missing job titles or industry terms
            job_title_missing = missing_keywords.get('job_titles', [])
            if job_title_missing:
                suggestions.append(ATSSuggestion(
                    suggestion_type="keyword_optimization",
                    title="Align Job Title Keywords",
                    description="Include job title keywords that match the position you're applying for.",
                    priority="medium",
                    impact_score=0.7,
                    implementation_effort="easy",
                    specific_actions=[
                        "Review your job titles and consider adding relevant keywords",
                        "Use industry-standard job title terminology",
                        "Include target job title keywords in your professional summary",
                        "Mention relevant job functions and responsibilities"
                    ],
                    keywords_to_add=job_title_missing[:3],
                    integration_examples=[
                        f"Professional Summary: Experienced {job_title_missing[0] if job_title_missing else 'professional'} with expertise in...",
                        f"Seeking opportunities as a {job_title_missing[0] if job_title_missing else 'professional'} to leverage..."
                    ]
                ))
            
            # Keyword density optimization
            density_issues = ats_analysis.get('keyword_density_issues', [])
            for issue in density_issues:
                if issue['issue'] == 'low_density':
                    suggestions.append(ATSSuggestion(
                        suggestion_type="keyword_density",
                        title=f"Increase {issue['keyword_type'].replace('_', ' ').title()} Keyword Density",
                        description=f"Your resume has low density of {issue['keyword_type'].replace('_', ' ')} keywords.",
                        priority="medium",
                        impact_score=0.5,
                        implementation_effort="medium",
                        specific_actions=[
                            f"Add more {issue['keyword_type'].replace('_', ' ')} throughout your resume",
                            "Integrate keywords naturally into experience descriptions",
                            "Ensure keywords appear in multiple sections when relevant",
                            "Use keyword variations and synonyms"
                        ]
                    ))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Keyword suggestion generation failed: {e}")
            return []
    
    def _generate_header_suggestions(self, 
                                   resume: ParsedResume, 
                                   ats_analysis: Dict[str, Any]) -> List[ATSSuggestion]:
        """Generate section header optimization suggestions."""
        suggestions = []
        
        try:
            section_analysis = ats_analysis.get('section_analysis', {})
            missing_sections = section_analysis.get('missing_sections', [])
            
            if missing_sections:
                suggestions.append(ATSSuggestion(
                    suggestion_type="section_headers",
                    title="Add Missing Resume Sections",
                    description=f"Your resume is missing {len(missing_sections)} important sections that ATS systems expect.",
                    priority="high",
                    impact_score=0.8,
                    implementation_effort="medium",
                    specific_actions=[
                        f"Add the following sections: {', '.join(missing_sections)}",
                        "Use standard, ATS-friendly section headers",
                        "Organize sections in conventional order",
                        "Ensure each section has relevant content"
                    ],
                    metadata={'missing_sections': missing_sections}
                ))
            
            # Suggest ATS-friendly header formats
            suggestions.append(ATSSuggestion(
                suggestion_type="section_headers",
                title="Use ATS-Friendly Section Headers",
                description="Optimize section headers for better ATS parsing and recognition.",
                priority="medium",
                impact_score=0.6,
                implementation_effort="easy",
                specific_actions=[
                    "Use standard section headers: 'EXPERIENCE', 'SKILLS', 'EDUCATION'",
                    "Avoid creative or unusual section names",
                    "Use consistent header formatting throughout",
                    "Place headers on separate lines with clear spacing"
                ],
                integration_examples=[
                    "Use 'WORK EXPERIENCE' instead of 'Career Journey'",
                    "Use 'TECHNICAL SKILLS' instead of 'My Toolkit'",
                    "Use 'EDUCATION' instead of 'Academic Background'"
                ]
            ))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Header suggestion generation failed: {e}")
            return []
    
    def _generate_formatting_suggestions(self, 
                                       resume: ParsedResume, 
                                       ats_analysis: Dict[str, Any]) -> List[ATSSuggestion]:
        """Generate formatting optimization suggestions."""
        suggestions = []
        
        try:
            formatting_analysis = ats_analysis.get('formatting_analysis', {})
            compatibility_issues = formatting_analysis.get('compatibility_issues', [])
            
            if compatibility_issues:
                suggestions.append(ATSSuggestion(
                    suggestion_type="formatting",
                    title="Remove ATS-Unfriendly Formatting",
                    description="Your resume contains formatting elements that may cause ATS parsing issues.",
                    priority="high",
                    impact_score=0.7,
                    implementation_effort="easy",
                    specific_actions=[
                        "Remove special characters and symbols (★, ♦, etc.)",
                        "Use simple bullet points (• or -)",
                        "Avoid complex tables and graphics",
                        "Use standard fonts (Arial, Calibri, Times New Roman)"
                    ],
                    metadata={'issues_found': compatibility_issues}
                ))
            
            # General ATS formatting best practices
            suggestions.append(ATSSuggestion(
                suggestion_type="formatting",
                title="Optimize Resume Structure for ATS",
                description="Ensure your resume structure is ATS-compatible for better parsing.",
                priority="medium",
                impact_score=0.6,
                implementation_effort="easy",
                specific_actions=[
                    "Use a simple, single-column layout",
                    "Avoid headers and footers",
                    "Use standard bullet points for lists",
                    "Ensure consistent spacing and alignment",
                    "Use clear section breaks"
                ],
                integration_examples=[
                    "Use simple bullet points: • Achievement 1",
                    "Clear section headers: WORK EXPERIENCE",
                    "Standard date format: 01/2020 - 12/2022"
                ]
            ))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Formatting suggestion generation failed: {e}")
            return []
    
    def _generate_file_format_suggestions(self) -> List[ATSSuggestion]:
        """Generate file format optimization suggestions."""
        suggestions = []
        
        suggestions.append(ATSSuggestion(
            suggestion_type="file_format",
            title="Use ATS-Compatible File Formats",
            description="Ensure your resume file format is compatible with ATS systems.",
            priority="medium",
            impact_score=0.5,
            implementation_effort="easy",
            specific_actions=[
                "Save resume as .docx (Microsoft Word) format for best compatibility",
                "PDF is acceptable but may have parsing issues with some ATS",
                "Avoid .txt files as they lose formatting",
                "Test your resume with online ATS scanners",
                "Keep file name simple: 'FirstName_LastName_Resume.docx'"
            ],
            integration_examples=[
                "Recommended: John_Smith_Resume.docx",
                "Acceptable: John_Smith_Resume.pdf",
                "Avoid: My Amazing Resume v2.1 FINAL.docx"
            ]
        ))
        
        return suggestions
    
    def _generate_integration_examples(self, 
                                     keywords: List[str], 
                                     keyword_type: KeywordType) -> List[str]:
        """Generate examples of how to integrate keywords naturally."""
        examples = []
        
        try:
            templates = self.integration_templates.get(keyword_type, [])
            
            for keyword in keywords[:3]:  # Limit to 3 examples
                if templates:
                    template = templates[0]  # Use first template for simplicity
                    example = template.format(keyword=keyword, years="2")
                    examples.append(f"• {example}")
            
            return examples
            
        except Exception as e:
            logger.error(f"Integration example generation failed: {e}")
            return []
    
    def _rank_ats_suggestions(self, suggestions: List[ATSSuggestion]) -> List[ATSSuggestion]:
        """Rank ATS suggestions by priority and impact score."""
        try:
            # Define priority weights
            priority_weights = {'high': 3, 'medium': 2, 'low': 1}
            
            # Sort by priority weight and impact score
            ranked = sorted(
                suggestions,
                key=lambda s: (priority_weights.get(s.priority, 2), s.impact_score),
                reverse=True
            )
            
            # Limit to top suggestions to avoid overwhelming the user
            return ranked[:10]
            
        except Exception as e:
            logger.error(f"ATS suggestion ranking failed: {e}")
            return suggestions
    
    def analyze_ats_optimization_potential(self, 
                                         resume: ParsedResume, 
                                         job_description: ParsedJobDescription) -> Dict[str, Any]:
        """Analyze the potential for ATS optimization improvements.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description
            
        Returns:
            Dictionary with ATS optimization analysis
        """
        try:
            logger.info("Analyzing ATS optimization potential")
            
            # Perform compatibility analysis
            ats_analysis = self.compatibility_analyzer.analyze_ats_compatibility(resume, job_description)
            
            # Generate suggestions
            suggestions = self.generate_ats_suggestions(resume, job_description)
            
            # Categorize suggestions by type
            suggestions_by_type = {}
            for suggestion in suggestions:
                suggestion_type = suggestion.suggestion_type
                if suggestion_type not in suggestions_by_type:
                    suggestions_by_type[suggestion_type] = []
                suggestions_by_type[suggestion_type].append(suggestion)
            
            # Calculate improvement potential
            current_score = ats_analysis.get('ats_compatibility_score', 0.0)
            potential_improvement = self._calculate_improvement_potential(suggestions)
            
            optimization_analysis = {
                'current_ats_score': current_score,
                'potential_improvement': potential_improvement,
                'projected_score': min(1.0, current_score + potential_improvement),
                'ats_analysis': ats_analysis,
                'suggestions': suggestions,
                'suggestions_by_type': suggestions_by_type,
                'summary': {
                    'total_suggestions': len(suggestions),
                    'high_priority_suggestions': len([s for s in suggestions if s.priority == 'high']),
                    'quick_wins': len([s for s in suggestions if s.implementation_effort == 'easy']),
                    'missing_keywords_count': sum(len(keywords) for keywords in ats_analysis.get('missing_keywords', {}).values()),
                    'main_optimization_areas': list(suggestions_by_type.keys())[:3]
                }
            }
            
            logger.info(f"ATS optimization analysis completed. Current score: {current_score:.2f}, Potential improvement: {potential_improvement:.2f}")
            return optimization_analysis
            
        except Exception as e:
            logger.error(f"ATS optimization analysis failed: {e}")
            return {
                'current_ats_score': 0.0,
                'potential_improvement': 0.0,
                'projected_score': 0.0,
                'ats_analysis': {},
                'suggestions': [],
                'suggestions_by_type': {},
                'summary': {
                    'total_suggestions': 0,
                    'high_priority_suggestions': 0,
                    'quick_wins': 0,
                    'missing_keywords_count': 0,
                    'main_optimization_areas': []
                }
            }
    
    def _calculate_improvement_potential(self, suggestions: List[ATSSuggestion]) -> float:
        """Calculate potential improvement from implementing suggestions."""
        try:
            total_potential = 0.0
            
            for suggestion in suggestions:
                # Weight improvement by priority and implementation effort
                priority_weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(suggestion.priority, 0.5)
                effort_weight = {'easy': 1.0, 'medium': 0.8, 'hard': 0.6}.get(suggestion.implementation_effort, 0.7)
                
                suggestion_potential = suggestion.impact_score * priority_weight * effort_weight
                total_potential += suggestion_potential
            
            # Normalize to reasonable improvement range (0-0.4)
            normalized_potential = min(0.4, total_potential * 0.1)
            
            return round(normalized_potential, 2)
            
        except Exception as e:
            logger.error(f"Improvement potential calculation failed: {e}")
            return 0.0