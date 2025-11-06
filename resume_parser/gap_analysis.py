"""
Gap analysis system for resume-job matching.

This module implements detailed gap analysis to identify missing skills,
experience shortfalls, and education requirements between resumes and job descriptions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import numpy as np

from .resume_interfaces import ParsedResume, WorkExperience, Education
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SkillGap:
    """Represents a missing skill with analysis details.
    
    Attributes:
        skill: The missing skill name
        category: Skill category (technical, soft, tools)
        priority: Priority level (high, medium, low)
        confidence: Confidence in the gap identification (0-1)
        alternatives: List of alternative/similar skills the candidate has
        learning_resources: Suggested learning resources or paths
        impact_score: Impact on overall match score (0-1)
    """
    skill: str
    category: str
    priority: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    learning_resources: List[str] = field(default_factory=list)
    impact_score: float = 0.0


@dataclass
class ExperienceGap:
    """Represents experience-related gaps and analysis.
    
    Attributes:
        years_required: Required years of experience
        years_candidate: Candidate's years of experience
        shortfall_years: Years of experience shortfall (negative if overqualified)
        relevance_score: How relevant candidate's experience is (0-1)
        industry_match: Whether candidate has industry experience
        seniority_gap: Gap in seniority level
        transferable_skills: Skills that transfer from other industries
        progression_analysis: Career progression analysis
    """
    years_required: float
    years_candidate: float
    shortfall_years: float
    relevance_score: float
    industry_match: bool
    seniority_gap: str
    transferable_skills: List[str] = field(default_factory=list)
    progression_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EducationGap:
    """Represents education and certification gaps.
    
    Attributes:
        degree_required: Required degree level
        degree_candidate: Candidate's degree level
        degree_gap: Gap in degree level
        field_match: Whether field of study matches
        certification_gaps: Missing certifications
        alternative_paths: Alternative qualification paths
        roi_analysis: Return on investment for additional education
    """
    degree_required: str
    degree_candidate: str
    degree_gap: str
    field_match: bool
    certification_gaps: List[str] = field(default_factory=list)
    alternative_paths: List[str] = field(default_factory=list)
    roi_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapAnalysisResult:
    """Complete gap analysis result.
    
    Attributes:
        skills_gaps: List of identified skill gaps
        experience_gap: Experience gap analysis
        education_gap: Education gap analysis
        overall_gap_score: Overall gap score (0-1, lower is better)
        improvement_potential: Potential score improvement from addressing gaps
        priority_recommendations: Top priority recommendations
        metadata: Additional analysis metadata
    """
    skills_gaps: List[SkillGap]
    experience_gap: ExperienceGap
    education_gap: EducationGap
    overall_gap_score: float
    improvement_potential: float
    priority_recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillsGapAnalyzer:
    """Analyzer for identifying and categorizing skill gaps between resume and job requirements.
    
    This class implements comprehensive skill gap analysis including missing technical skills,
    soft skills gaps, skill level mismatches, and synonym/alternative detection.
    """
    
    def __init__(self, 
                 skills_ontology: Optional[Dict[str, List[str]]] = None,
                 synonym_mapping: Optional[Dict[str, List[str]]] = None):
        """Initialize skills gap analyzer.
        
        Args:
            skills_ontology: Skills ontology for categorization
            synonym_mapping: Mapping of skills to their synonyms/alternatives
        """
        self.skills_ontology = skills_ontology or {}
        self.synonym_mapping = synonym_mapping or {}
        
        # Default skill categories and priorities
        self._skill_categories = {
            'programming_languages': {
                'priority': 'high',
                'keywords': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'scala']
            },
            'frameworks': {
                'priority': 'high', 
                'keywords': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express']
            },
            'databases': {
                'priority': 'high',
                'keywords': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']
            },
            'cloud_platforms': {
                'priority': 'high',
                'keywords': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform']
            },
            'tools': {
                'priority': 'medium',
                'keywords': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'figma']
            },
            'soft_skills': {
                'priority': 'medium',
                'keywords': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical']
            },
            'methodologies': {
                'priority': 'medium',
                'keywords': ['agile', 'scrum', 'devops', 'ci/cd', 'tdd', 'microservices']
            }
        }
        
        # Skill level indicators
        self._skill_levels = {
            'beginner': ['basic', 'beginner', 'entry', 'junior', 'learning', 'familiar'],
            'intermediate': ['intermediate', 'mid', 'working', 'proficient', 'experienced'],
            'advanced': ['advanced', 'senior', 'expert', 'lead', 'architect', 'principal']
        }
        
        # Build reverse synonym mapping for faster lookup
        self._reverse_synonyms = {}
        for skill, synonyms in self.synonym_mapping.items():
            for synonym in synonyms:
                self._reverse_synonyms[synonym.lower()] = skill.lower()
        
        logger.info("Skills gap analyzer initialized")
    
    def analyze_skills_gaps(self, 
                          resume: ParsedResume, 
                          job: ParsedJobDescription) -> List[SkillGap]:
        """Analyze skill gaps between resume and job requirements.
        
        Args:
            resume: Parsed resume data
            job: Parsed job description data
            
        Returns:
            List of identified skill gaps with analysis
        """
        try:
            logger.debug("Analyzing skills gaps")
            
            # Normalize skills for comparison
            resume_skills = self._normalize_skills(resume.skills)
            job_skills = self._normalize_skills(job.skills_required)
            
            # Find missing skills
            missing_skills = self._find_missing_skills(resume_skills, job_skills)
            
            # Analyze each missing skill
            skill_gaps = []
            for skill in missing_skills:
                gap = self._analyze_single_skill_gap(skill, resume_skills, job_skills, resume, job)
                skill_gaps.append(gap)
            
            # Sort by priority and impact
            skill_gaps.sort(key=lambda x: (self._priority_order(x.priority), -x.impact_score))
            
            logger.debug(f"Identified {len(skill_gaps)} skill gaps")
            return skill_gaps
            
        except Exception as e:
            logger.error(f"Skills gap analysis failed: {e}")
            return []
    
    def _normalize_skills(self, skills: List[str]) -> Set[str]:
        """Normalize skills for consistent comparison.
        
        Args:
            skills: List of skill names
            
        Returns:
            Set of normalized skill names
        """
        normalized = set()
        
        for skill in skills:
            # Convert to lowercase and clean
            clean_skill = re.sub(r'[^\w\s+#\.]', '', skill.lower().strip())
            
            # Handle common technical term variations
            clean_skill = self._normalize_technical_terms(clean_skill)
            
            # Check for synonyms
            if clean_skill in self._reverse_synonyms:
                clean_skill = self._reverse_synonyms[clean_skill]
            
            if clean_skill:
                normalized.add(clean_skill)
        
        return normalized
    
    def _normalize_technical_terms(self, skill: str) -> str:
        """Normalize technical terms for consistent matching.
        
        Args:
            skill: Skill name to normalize
            
        Returns:
            Normalized skill name
        """
        # Common normalizations
        normalizations = {
            'c++': 'cplusplus',
            'c#': 'csharp',
            'f#': 'fsharp',
            'node.js': 'nodejs',
            'react.js': 'reactjs',
            'vue.js': 'vuejs',
            'angular.js': 'angularjs',
            'machine learning': 'machinelearning',
            'data science': 'datascience',
            'artificial intelligence': 'ai',
            'natural language processing': 'nlp'
        }
        
        return normalizations.get(skill, skill)
    
    def _find_missing_skills(self, resume_skills: Set[str], job_skills: Set[str]) -> List[str]:
        """Find skills required by job but missing from resume.
        
        Args:
            resume_skills: Set of normalized resume skills
            job_skills: Set of normalized job skills
            
        Returns:
            List of missing skills
        """
        # Direct missing skills
        missing = job_skills - resume_skills
        
        # Check for partial matches and alternatives
        filtered_missing = []
        for skill in missing:
            if not self._has_alternative_skill(skill, resume_skills):
                filtered_missing.append(skill)
        
        return list(filtered_missing)
    
    def _has_alternative_skill(self, missing_skill: str, resume_skills: Set[str]) -> bool:
        """Check if resume has alternative/similar skills.
        
        Args:
            missing_skill: The missing skill to check
            resume_skills: Set of resume skills
            
        Returns:
            True if alternative skill found
        """
        # Check synonyms
        if missing_skill in self.synonym_mapping:
            synonyms = set(syn.lower() for syn in self.synonym_mapping[missing_skill])
            if synonyms.intersection(resume_skills):
                return True
        
        # Check partial matches for compound skills
        if ' ' in missing_skill or '.' in missing_skill:
            parts = re.split(r'[\s\.]', missing_skill)
            if len(parts) > 1:
                for part in parts:
                    if len(part) > 2 and part in resume_skills:
                        return True
        
        return False
    
    def _analyze_single_skill_gap(self, 
                                skill: str, 
                                resume_skills: Set[str], 
                                job_skills: Set[str],
                                resume: ParsedResume,
                                job: ParsedJobDescription) -> SkillGap:
        """Analyze a single skill gap in detail.
        
        Args:
            skill: The missing skill
            resume_skills: Set of resume skills
            job_skills: Set of job skills
            resume: Full resume data
            job: Full job data
            
        Returns:
            SkillGap object with detailed analysis
        """
        # Determine category and priority
        category = self._categorize_skill(skill)
        priority = self._determine_priority(skill, category, job)
        
        # Calculate confidence in gap identification
        confidence = self._calculate_gap_confidence(skill, resume_skills, job_skills)
        
        # Find alternative skills
        alternatives = self._find_alternative_skills(skill, resume_skills)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(skill, category, priority, job)
        
        # Generate learning resources
        learning_resources = self._suggest_learning_resources(skill, category)
        
        return SkillGap(
            skill=skill,
            category=category,
            priority=priority,
            confidence=confidence,
            alternatives=alternatives,
            learning_resources=learning_resources,
            impact_score=impact_score
        )
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill based on predefined categories.
        
        Args:
            skill: Skill name
            
        Returns:
            Category name
        """
        skill_lower = skill.lower()
        
        # Check against predefined categories
        for category, info in self._skill_categories.items():
            if any(keyword in skill_lower for keyword in info['keywords']):
                return category
        
        # Check ontology if available
        if self.skills_ontology:
            for category, skills in self.skills_ontology.items():
                if skill in skills or skill_lower in [s.lower() for s in skills]:
                    return category
        
        return 'other'
    
    def _determine_priority(self, skill: str, category: str, job: ParsedJobDescription) -> str:
        """Determine priority level for a missing skill.
        
        Args:
            skill: Skill name
            category: Skill category
            job: Job description data
            
        Returns:
            Priority level (high, medium, low)
        """
        # Base priority from category
        base_priority = self._skill_categories.get(category, {}).get('priority', 'medium')
        
        # Adjust based on job context
        job_text = ' '.join([
            ' '.join(job.skills_required),
            ' '.join(job.tools_mentioned),
            str(job.metadata.get('description', ''))
        ]).lower()
        
        # Count mentions in job description
        skill_mentions = job_text.count(skill.lower())
        
        # Adjust priority based on frequency
        if skill_mentions >= 3:
            return 'high'
        elif skill_mentions >= 2 and base_priority != 'low':
            return 'high' if base_priority == 'high' else 'medium'
        
        return base_priority
    
    def _calculate_gap_confidence(self, 
                                skill: str, 
                                resume_skills: Set[str], 
                                job_skills: Set[str]) -> float:
        """Calculate confidence in gap identification.
        
        Args:
            skill: Missing skill
            resume_skills: Resume skills
            job_skills: Job skills
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.8  # Base confidence
        
        # Reduce confidence if similar skills exist
        similar_count = 0
        for resume_skill in resume_skills:
            if self._skills_are_similar(skill, resume_skill):
                similar_count += 1
        
        if similar_count > 0:
            confidence -= min(0.3, similar_count * 0.1)
        
        # Increase confidence for exact matches in job requirements
        if skill in job_skills:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _skills_are_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are similar.
        
        Args:
            skill1: First skill
            skill2: Second skill
            
        Returns:
            True if skills are similar
        """
        # Simple similarity check based on common words
        words1 = set(skill1.lower().split())
        words2 = set(skill2.lower().split())
        
        # Check for common words (excluding very common ones)
        common_words = words1.intersection(words2)
        common_words = {w for w in common_words if len(w) > 2}
        
        return len(common_words) > 0
    
    def _find_alternative_skills(self, missing_skill: str, resume_skills: Set[str]) -> List[str]:
        """Find alternative skills the candidate has.
        
        Args:
            missing_skill: The missing skill
            resume_skills: Set of resume skills
            
        Returns:
            List of alternative skills
        """
        alternatives = []
        
        # Check synonyms
        if missing_skill in self.synonym_mapping:
            for synonym in self.synonym_mapping[missing_skill]:
                if synonym.lower() in resume_skills:
                    alternatives.append(synonym)
        
        # Check similar skills
        for resume_skill in resume_skills:
            if self._skills_are_similar(missing_skill, resume_skill):
                alternatives.append(resume_skill)
        
        return alternatives[:3]  # Limit to top 3
    
    def _calculate_impact_score(self, 
                              skill: str, 
                              category: str, 
                              priority: str, 
                              job: ParsedJobDescription) -> float:
        """Calculate impact score for missing skill.
        
        Args:
            skill: Missing skill
            category: Skill category
            priority: Priority level
            job: Job description
            
        Returns:
            Impact score (0-1)
        """
        # Base impact from priority
        priority_impact = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        base_impact = priority_impact.get(priority, 0.5)
        
        # Adjust based on category importance
        category_multiplier = {
            'programming_languages': 1.2,
            'frameworks': 1.1,
            'databases': 1.1,
            'cloud_platforms': 1.0,
            'tools': 0.8,
            'soft_skills': 0.7,
            'methodologies': 0.9
        }
        
        multiplier = category_multiplier.get(category, 1.0)
        
        return min(1.0, base_impact * multiplier)
    
    def _suggest_learning_resources(self, skill: str, category: str) -> List[str]:
        """Suggest learning resources for a skill.
        
        Args:
            skill: Skill name
            category: Skill category
            
        Returns:
            List of learning resource suggestions
        """
        resources = []
        
        # Category-specific resources
        if category == 'programming_languages':
            resources.extend([
                f"Online courses (Coursera, Udemy) for {skill}",
                f"Official {skill} documentation and tutorials",
                f"Practice coding challenges on LeetCode/HackerRank"
            ])
        elif category == 'frameworks':
            resources.extend([
                f"Official {skill} documentation",
                f"Build projects using {skill}",
                f"Follow {skill} tutorials on YouTube"
            ])
        elif category == 'cloud_platforms':
            resources.extend([
                f"{skill} certification courses",
                f"Hands-on labs and free tier practice",
                f"Cloud architecture courses"
            ])
        elif category == 'soft_skills':
            resources.extend([
                "Leadership and communication workshops",
                "Professional development courses",
                "Mentoring and coaching programs"
            ])
        else:
            resources.extend([
                f"Online tutorials for {skill}",
                f"Professional courses and certifications",
                f"Hands-on practice and projects"
            ])
        
        return resources[:3]  # Limit to top 3
    
    def _priority_order(self, priority: str) -> int:
        """Get numeric order for priority sorting.
        
        Args:
            priority: Priority level
            
        Returns:
            Numeric order (lower is higher priority)
        """
        order = {'high': 0, 'medium': 1, 'low': 2}
        return order.get(priority, 1)
    
    def detect_skill_level_mismatches(self, 
                                    resume: ParsedResume, 
                                    job: ParsedJobDescription) -> List[Dict[str, Any]]:
        """Detect skill level mismatches between resume and job requirements.
        
        Args:
            resume: Parsed resume data
            job: Parsed job description data
            
        Returns:
            List of skill level mismatch analyses
        """
        try:
            logger.debug("Detecting skill level mismatches")
            
            mismatches = []
            
            # Extract skill levels from job description
            job_text = str(job.metadata.get('description', ''))
            job_skill_levels = self._extract_skill_levels(job_text)
            
            # Extract skill levels from resume
            resume_text = ' '.join([
                exp.description for exp in resume.experience
            ])
            resume_skill_levels = self._extract_skill_levels(resume_text)
            
            # Compare skill levels
            for skill, job_level in job_skill_levels.items():
                resume_level = resume_skill_levels.get(skill, 'beginner')
                
                if self._level_order(job_level) > self._level_order(resume_level):
                    mismatch = {
                        'skill': skill,
                        'required_level': job_level,
                        'candidate_level': resume_level,
                        'gap_severity': self._level_order(job_level) - self._level_order(resume_level),
                        'recommendations': self._get_level_improvement_recommendations(skill, resume_level, job_level)
                    }
                    mismatches.append(mismatch)
            
            logger.debug(f"Found {len(mismatches)} skill level mismatches")
            return mismatches
            
        except Exception as e:
            logger.error(f"Skill level mismatch detection failed: {e}")
            return []
    
    def _extract_skill_levels(self, text: str) -> Dict[str, str]:
        """Extract skill levels from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping skills to levels
        """
        skill_levels = {}
        text_lower = text.lower()
        
        # Look for level indicators near skill mentions
        for category_info in self._skill_categories.values():
            for skill in category_info['keywords']:
                if skill in text_lower:
                    # Find level indicators near the skill
                    level = self._find_level_near_skill(text_lower, skill)
                    if level:
                        skill_levels[skill] = level
        
        return skill_levels
    
    def _find_level_near_skill(self, text: str, skill: str) -> Optional[str]:
        """Find level indicator near a skill mention.
        
        Args:
            text: Text to search
            skill: Skill to find level for
            
        Returns:
            Level if found, None otherwise
        """
        # Find skill position
        skill_pos = text.find(skill)
        if skill_pos == -1:
            return None
        
        # Look in surrounding context (50 characters before and after)
        start = max(0, skill_pos - 50)
        end = min(len(text), skill_pos + len(skill) + 50)
        context = text[start:end]
        
        # Check for level indicators
        for level, indicators in self._skill_levels.items():
            if any(indicator in context for indicator in indicators):
                return level
        
        return 'intermediate'  # Default level
    
    def _level_order(self, level: str) -> int:
        """Get numeric order for skill level.
        
        Args:
            level: Skill level
            
        Returns:
            Numeric order (higher is more advanced)
        """
        order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        return order.get(level, 2)
    
    def _get_level_improvement_recommendations(self, 
                                            skill: str, 
                                            current_level: str, 
                                            target_level: str) -> List[str]:
        """Get recommendations for improving skill level.
        
        Args:
            skill: Skill name
            current_level: Current skill level
            target_level: Target skill level
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        if target_level == 'advanced':
            recommendations.extend([
                f"Lead projects using {skill}",
                f"Mentor others in {skill}",
                f"Contribute to open source {skill} projects",
                f"Obtain advanced {skill} certifications"
            ])
        elif target_level == 'intermediate':
            recommendations.extend([
                f"Complete intermediate {skill} courses",
                f"Build complex projects with {skill}",
                f"Practice {skill} best practices",
                f"Join {skill} communities and forums"
            ])
        
        return recommendations[:3]


class ExperienceGapAnalyzer:
    """Analyzer for experience gaps, overqualification scenarios, and career progression analysis.
    
    This class implements comprehensive experience analysis including years of experience,
    industry relevance, seniority levels, and transferable skills identification.
    """
    
    def __init__(self):
        """Initialize experience gap analyzer."""
        # Experience level mappings
        self._experience_levels = {
            'entry': {'min_years': 0, 'max_years': 2, 'keywords': ['entry', 'junior', 'associate', 'trainee']},
            'mid': {'min_years': 2, 'max_years': 5, 'keywords': ['mid', 'intermediate', 'regular', 'specialist']},
            'senior': {'min_years': 5, 'max_years': 10, 'keywords': ['senior', 'lead', 'principal', 'staff']},
            'executive': {'min_years': 10, 'max_years': float('inf'), 'keywords': ['director', 'manager', 'vp', 'cto', 'ceo']}
        }
        
        # Industry categories for relevance analysis
        self._industry_categories = {
            'technology': ['software', 'tech', 'it', 'computer', 'digital', 'startup', 'saas'],
            'finance': ['bank', 'financial', 'investment', 'insurance', 'fintech', 'trading'],
            'healthcare': ['health', 'medical', 'hospital', 'pharmaceutical', 'biotech'],
            'retail': ['retail', 'ecommerce', 'commerce', 'shopping', 'consumer'],
            'manufacturing': ['manufacturing', 'industrial', 'automotive', 'aerospace'],
            'consulting': ['consulting', 'advisory', 'professional services'],
            'education': ['education', 'university', 'school', 'academic', 'research']
        }
        
        # Transferable skills mapping
        self._transferable_skills = {
            'leadership': ['team management', 'project management', 'mentoring', 'coaching'],
            'communication': ['presentation', 'writing', 'negotiation', 'client relations'],
            'analytical': ['data analysis', 'problem solving', 'research', 'strategic thinking'],
            'technical': ['programming', 'system design', 'architecture', 'troubleshooting'],
            'business': ['strategy', 'operations', 'process improvement', 'stakeholder management']
        }
        
        logger.info("Experience gap analyzer initialized")
    
    def analyze_experience_gap(self, 
                             resume: ParsedResume, 
                             job: ParsedJobDescription) -> ExperienceGap:
        """Analyze experience gaps between resume and job requirements.
        
        Args:
            resume: Parsed resume data
            job: Parsed job description data
            
        Returns:
            ExperienceGap object with detailed analysis
        """
        try:
            logger.debug("Analyzing experience gaps")
            
            # Extract experience requirements from job
            years_required = self._extract_required_experience(job)
            required_seniority = self._extract_required_seniority(job)
            
            # Calculate candidate experience
            years_candidate = self._calculate_total_experience(resume)
            candidate_seniority = self._determine_candidate_seniority(resume)
            
            # Calculate shortfall
            shortfall_years = years_required - years_candidate
            
            # Analyze relevance
            relevance_score = self._calculate_experience_relevance(resume, job)
            
            # Check industry match
            industry_match = self._check_industry_match(resume, job)
            
            # Analyze seniority gap
            seniority_gap = self._analyze_seniority_gap(candidate_seniority, required_seniority)
            
            # Identify transferable skills
            transferable_skills = self._identify_transferable_skills(resume, job)
            
            # Career progression analysis
            progression_analysis = self._analyze_career_progression(resume)
            
            return ExperienceGap(
                years_required=years_required,
                years_candidate=years_candidate,
                shortfall_years=shortfall_years,
                relevance_score=relevance_score,
                industry_match=industry_match,
                seniority_gap=seniority_gap,
                transferable_skills=transferable_skills,
                progression_analysis=progression_analysis
            )
            
        except Exception as e:
            logger.error(f"Experience gap analysis failed: {e}")
            return ExperienceGap(
                years_required=0.0,
                years_candidate=0.0,
                shortfall_years=0.0,
                relevance_score=0.0,
                industry_match=False,
                seniority_gap="unknown",
                transferable_skills=[],
                progression_analysis={}
            )
    
    def _extract_required_experience(self, job: ParsedJobDescription) -> float:
        """Extract required years of experience from job description.
        
        Args:
            job: Job description data
            
        Returns:
            Required years of experience
        """
        job_text = str(job.metadata.get('description', '')).lower()
        
        # Look for explicit year mentions
        year_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
            r'(\d+)\+\s*years?'
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, job_text)
            years_found.extend([int(match) for match in matches])
        
        if years_found:
            return float(max(years_found))  # Take the highest requirement
        
        # Infer from experience level keywords
        if any(keyword in job_text for keyword in self._experience_levels['senior']['keywords']):
            return 5.0
        elif any(keyword in job_text for keyword in self._experience_levels['mid']['keywords']):
            return 2.0
        elif any(keyword in job_text for keyword in self._experience_levels['entry']['keywords']):
            return 0.0
        
        return 2.0  # Default assumption
    
    def _extract_required_seniority(self, job: ParsedJobDescription) -> str:
        """Extract required seniority level from job description.
        
        Args:
            job: Job description data
            
        Returns:
            Required seniority level
        """
        job_text = str(job.metadata.get('description', '')).lower()
        
        # Check for seniority keywords
        for level, info in self._experience_levels.items():
            if any(keyword in job_text for keyword in info['keywords']):
                return level
        
        return 'mid'  # Default
    
    def _calculate_total_experience(self, resume: ParsedResume) -> float:
        """Calculate total years of experience from resume.
        
        Args:
            resume: Resume data
            
        Returns:
            Total years of experience
        """
        total_months = 0
        
        for exp in resume.experience:
            if exp.duration_months > 0:
                total_months += exp.duration_months
            else:
                # Try to calculate from dates if duration not available
                months = self._calculate_duration_from_dates(exp.start_date, exp.end_date)
                total_months += months
        
        return total_months / 12.0  # Convert to years
    
    def _calculate_duration_from_dates(self, start_date: str, end_date: Optional[str]) -> int:
        """Calculate duration in months from date strings.
        
        Args:
            start_date: Start date string
            end_date: End date string (None for current)
            
        Returns:
            Duration in months
        """
        try:
            from datetime import datetime
            
            # Simple date parsing (can be enhanced)
            if not start_date:
                return 0
            
            # Extract year from date string
            start_year = self._extract_year_from_date(start_date)
            if end_date:
                end_year = self._extract_year_from_date(end_date)
            else:
                end_year = datetime.now().year
            
            if start_year and end_year:
                return max(0, (end_year - start_year) * 12)
            
        except Exception as e:
            logger.warning(f"Date calculation failed: {e}")
        
        return 12  # Default 1 year
    
    def _extract_year_from_date(self, date_str: str) -> Optional[int]:
        """Extract year from date string.
        
        Args:
            date_str: Date string
            
        Returns:
            Year if found, None otherwise
        """
        if not date_str:
            return None
        
        # Look for 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return int(year_match.group())
        
        return None
    
    def _determine_candidate_seniority(self, resume: ParsedResume) -> str:
        """Determine candidate's seniority level.
        
        Args:
            resume: Resume data
            
        Returns:
            Seniority level
        """
        total_years = self._calculate_total_experience(resume)
        
        # Check job titles for seniority indicators
        job_titles = [exp.job_title.lower() for exp in resume.experience]
        title_text = ' '.join(job_titles)
        
        # Check for executive level
        if any(keyword in title_text for keyword in self._experience_levels['executive']['keywords']):
            return 'executive'
        
        # Check for senior level
        if any(keyword in title_text for keyword in self._experience_levels['senior']['keywords']):
            return 'senior'
        
        # Use years of experience as fallback
        for level, info in self._experience_levels.items():
            if info['min_years'] <= total_years <= info['max_years']:
                return level
        
        return 'mid'  # Default
    
    def _calculate_experience_relevance(self, resume: ParsedResume, job: ParsedJobDescription) -> float:
        """Calculate how relevant candidate's experience is to the job.
        
        Args:
            resume: Resume data
            job: Job description data
            
        Returns:
            Relevance score (0-1)
        """
        job_skills = set(skill.lower() for skill in job.skills_required)
        job_tools = set(tool.lower() for tool in job.tools_mentioned)
        job_requirements = job_skills.union(job_tools)
        
        if not job_requirements:
            return 0.5  # Default if no clear requirements
        
        # Analyze each work experience
        relevance_scores = []
        
        for exp in resume.experience:
            exp_text = f"{exp.job_title} {exp.description}".lower()
            exp_skills = set(skill.lower() for skill in exp.skills_used)
            
            # Calculate skill overlap
            skill_overlap = len(exp_skills.intersection(job_requirements))
            skill_relevance = skill_overlap / len(job_requirements) if job_requirements else 0
            
            # Calculate text similarity (simple keyword matching)
            text_matches = sum(1 for req in job_requirements if req in exp_text)
            text_relevance = text_matches / len(job_requirements) if job_requirements else 0
            
            # Weight by experience duration
            duration_weight = min(1.0, exp.duration_months / 24.0)  # Cap at 2 years
            
            exp_relevance = (skill_relevance * 0.6 + text_relevance * 0.4) * duration_weight
            relevance_scores.append(exp_relevance)
        
        # Return weighted average (more recent experience weighted higher)
        if not relevance_scores:
            return 0.0
        
        # Simple average for now (can be enhanced with recency weighting)
        return sum(relevance_scores) / len(relevance_scores)
    
    def _check_industry_match(self, resume: ParsedResume, job: ParsedJobDescription) -> bool:
        """Check if candidate has relevant industry experience.
        
        Args:
            resume: Resume data
            job: Job description data
            
        Returns:
            True if industry match found
        """
        # Extract job industry
        job_text = str(job.metadata.get('description', '')).lower()
        job_industry = self._identify_industry(job_text)
        
        if not job_industry:
            return True  # Assume match if industry unclear
        
        # Check resume companies and descriptions
        for exp in resume.experience:
            exp_text = f"{exp.company} {exp.description}".lower()
            exp_industry = self._identify_industry(exp_text)
            
            if exp_industry == job_industry:
                return True
        
        return False
    
    def _identify_industry(self, text: str) -> Optional[str]:
        """Identify industry from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Industry name if identified
        """
        text_lower = text.lower()
        
        for industry, keywords in self._industry_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return industry
        
        return None
    
    def _analyze_seniority_gap(self, candidate_level: str, required_level: str) -> str:
        """Analyze gap between candidate and required seniority.
        
        Args:
            candidate_level: Candidate's seniority level
            required_level: Required seniority level
            
        Returns:
            Gap description
        """
        level_order = {'entry': 0, 'mid': 1, 'senior': 2, 'executive': 3}
        
        candidate_order = level_order.get(candidate_level, 1)
        required_order = level_order.get(required_level, 1)
        
        gap = required_order - candidate_order
        
        if gap > 0:
            return f"underqualified_by_{gap}_levels"
        elif gap < 0:
            return f"overqualified_by_{abs(gap)}_levels"
        else:
            return "appropriate_level"
    
    def _identify_transferable_skills(self, resume: ParsedResume, job: ParsedJobDescription) -> List[str]:
        """Identify transferable skills from candidate's experience.
        
        Args:
            resume: Resume data
            job: Job description data
            
        Returns:
            List of transferable skills
        """
        transferable = []
        
        # Analyze experience descriptions
        for exp in resume.experience:
            exp_text = exp.description.lower()
            
            for skill_category, indicators in self._transferable_skills.items():
                if any(indicator in exp_text for indicator in indicators):
                    if skill_category not in transferable:
                        transferable.append(skill_category)
        
        return transferable
    
    def _analyze_career_progression(self, resume: ParsedResume) -> Dict[str, Any]:
        """Analyze candidate's career progression.
        
        Args:
            resume: Resume data
            
        Returns:
            Career progression analysis
        """
        if len(resume.experience) < 2:
            return {'progression_type': 'insufficient_data'}
        
        # Sort experiences by start date (most recent first)
        sorted_exp = sorted(resume.experience, 
                          key=lambda x: self._extract_year_from_date(x.start_date) or 0, 
                          reverse=True)
        
        # Analyze progression patterns
        progression_analysis = {
            'total_positions': len(sorted_exp),
            'average_tenure_months': sum(exp.duration_months for exp in sorted_exp) / len(sorted_exp),
            'progression_type': self._determine_progression_type(sorted_exp),
            'title_progression': self._analyze_title_progression(sorted_exp),
            'company_changes': len(set(exp.company for exp in sorted_exp)),
            'skill_evolution': self._analyze_skill_evolution(sorted_exp)
        }
        
        return progression_analysis
    
    def _determine_progression_type(self, experiences: List[WorkExperience]) -> str:
        """Determine type of career progression.
        
        Args:
            experiences: List of work experiences (sorted by date)
            
        Returns:
            Progression type
        """
        if len(experiences) < 2:
            return 'insufficient_data'
        
        # Analyze title progression
        titles = [exp.job_title.lower() for exp in experiences]
        
        # Check for upward progression keywords
        progression_keywords = ['senior', 'lead', 'principal', 'manager', 'director']
        
        has_progression = False
        for i in range(len(titles) - 1):
            current_title = titles[i]
            previous_title = titles[i + 1]
            
            # Check if current title has more senior keywords
            current_seniority = sum(1 for keyword in progression_keywords if keyword in current_title)
            previous_seniority = sum(1 for keyword in progression_keywords if keyword in previous_title)
            
            if current_seniority > previous_seniority:
                has_progression = True
                break
        
        if has_progression:
            return 'upward_progression'
        
        # Check for lateral moves (same level, different skills/companies)
        unique_companies = set(exp.company for exp in experiences)
        if len(unique_companies) > 1:
            return 'lateral_progression'
        
        return 'stable_progression'
    
    def _analyze_title_progression(self, experiences: List[WorkExperience]) -> List[str]:
        """Analyze progression of job titles.
        
        Args:
            experiences: List of work experiences
            
        Returns:
            List of job titles in chronological order
        """
        return [exp.job_title for exp in reversed(experiences)]
    
    def _analyze_skill_evolution(self, experiences: List[WorkExperience]) -> Dict[str, Any]:
        """Analyze evolution of skills across positions.
        
        Args:
            experiences: List of work experiences
            
        Returns:
            Skill evolution analysis
        """
        all_skills = set()
        skill_timeline = []
        
        for exp in reversed(experiences):  # Chronological order
            exp_skills = set(exp.skills_used)
            all_skills.update(exp_skills)
            skill_timeline.append({
                'position': exp.job_title,
                'skills': list(exp_skills),
                'new_skills': list(exp_skills - set().union(*[s['skills'] for s in skill_timeline]))
            })
        
        return {
            'total_unique_skills': len(all_skills),
            'skill_timeline': skill_timeline,
            'skill_diversity': len(all_skills) / len(experiences) if experiences else 0
        }


class EducationGapAnalyzer:
    """Analyzer for education and certification gaps with alternative paths and ROI analysis.
    
    This class implements comprehensive education analysis including degree requirements,
    certification gaps, alternative qualification paths, and ROI calculations.
    """
    
    def __init__(self):
        """Initialize education gap analyzer."""
        # Degree level hierarchy
        self._degree_levels = {
            'high_school': {'order': 0, 'keywords': ['high school', 'diploma', 'ged']},
            'associate': {'order': 1, 'keywords': ['associate', 'aa', 'as', 'aas']},
            'bachelor': {'order': 2, 'keywords': ['bachelor', 'ba', 'bs', 'bsc', 'beng']},
            'master': {'order': 3, 'keywords': ['master', 'ma', 'ms', 'msc', 'mba', 'meng']},
            'doctoral': {'order': 4, 'keywords': ['phd', 'doctorate', 'doctoral', 'dsc']}
        }
        
        # Field of study mappings
        self._field_mappings = {
            'computer_science': ['computer science', 'cs', 'software engineering', 'information technology'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical'],
            'business': ['business', 'management', 'mba', 'finance', 'economics'],
            'mathematics': ['mathematics', 'math', 'statistics', 'data science'],
            'science': ['physics', 'chemistry', 'biology', 'science'],
            'liberal_arts': ['liberal arts', 'humanities', 'english', 'history', 'psychology']
        }
        
        # Industry-relevant certifications
        self._relevant_certifications = {
            'technology': [
                'AWS Certified Solutions Architect',
                'Google Cloud Professional',
                'Microsoft Azure Fundamentals',
                'Certified Kubernetes Administrator',
                'PMP (Project Management Professional)',
                'Scrum Master Certification',
                'CISSP (Certified Information Systems Security Professional)'
            ],
            'data_science': [
                'Certified Analytics Professional',
                'Google Data Analytics Certificate',
                'Microsoft Certified: Azure Data Scientist',
                'Tableau Desktop Specialist',
                'SAS Certified Specialist'
            ],
            'finance': [
                'CFA (Chartered Financial Analyst)',
                'FRM (Financial Risk Manager)',
                'CPA (Certified Public Accountant)',
                'Series 7 License'
            ],
            'marketing': [
                'Google Ads Certification',
                'HubSpot Content Marketing',
                'Facebook Blueprint Certification',
                'Salesforce Administrator'
            ]
        }
        
        # Alternative education paths
        self._alternative_paths = {
            'bootcamp': {
                'duration_months': 6,
                'cost_range': (5000, 20000),
                'fields': ['programming', 'data science', 'ux design', 'digital marketing']
            },
            'online_courses': {
                'duration_months': 3,
                'cost_range': (100, 2000),
                'fields': ['any']
            },
            'professional_certification': {
                'duration_months': 4,
                'cost_range': (500, 5000),
                'fields': ['cloud computing', 'project management', 'cybersecurity']
            },
            'self_directed_learning': {
                'duration_months': 12,
                'cost_range': (0, 1000),
                'fields': ['programming', 'design', 'marketing']
            }
        }
        
        logger.info("Education gap analyzer initialized")
    
    def analyze_education_gap(self, 
                            resume: ParsedResume, 
                            job: ParsedJobDescription) -> EducationGap:
        """Analyze education and certification gaps.
        
        Args:
            resume: Parsed resume data
            job: Parsed job description data
            
        Returns:
            EducationGap object with detailed analysis
        """
        try:
            logger.debug("Analyzing education gaps")
            
            # Extract education requirements
            degree_required = self._extract_required_degree(job)
            field_required = self._extract_required_field(job)
            
            # Analyze candidate education
            degree_candidate = self._get_highest_degree(resume)
            field_candidate = self._get_primary_field(resume)
            
            # Calculate degree gap
            degree_gap = self._calculate_degree_gap(degree_candidate, degree_required)
            
            # Check field match
            field_match = self._check_field_match(field_candidate, field_required, job)
            
            # Identify certification gaps
            certification_gaps = self._identify_certification_gaps(resume, job)
            
            # Generate alternative paths
            alternative_paths = self._generate_alternative_paths(degree_gap, field_match, job)
            
            # Calculate ROI analysis
            roi_analysis = self._calculate_education_roi(degree_gap, alternative_paths, job)
            
            return EducationGap(
                degree_required=degree_required,
                degree_candidate=degree_candidate,
                degree_gap=degree_gap,
                field_match=field_match,
                certification_gaps=certification_gaps,
                alternative_paths=alternative_paths,
                roi_analysis=roi_analysis
            )
            
        except Exception as e:
            logger.error(f"Education gap analysis failed: {e}")
            return EducationGap(
                degree_required="unknown",
                degree_candidate="unknown",
                degree_gap="unknown",
                field_match=False,
                certification_gaps=[],
                alternative_paths=[],
                roi_analysis={}
            )
    
    def _extract_required_degree(self, job: ParsedJobDescription) -> str:
        """Extract required degree level from job description.
        
        Args:
            job: Job description data
            
        Returns:
            Required degree level
        """
        job_text = str(job.metadata.get('description', '')).lower()
        
        # Look for degree requirements
        for degree, info in self._degree_levels.items():
            if any(keyword in job_text for keyword in info['keywords']):
                return degree
        
        # Check for general education requirements
        if any(phrase in job_text for phrase in ['degree required', 'bachelor required', 'university degree']):
            return 'bachelor'
        
        if any(phrase in job_text for phrase in ['advanced degree', 'graduate degree']):
            return 'master'
        
        return 'bachelor'  # Default assumption for professional roles
    
    def _extract_required_field(self, job: ParsedJobDescription) -> Optional[str]:
        """Extract required field of study from job description.
        
        Args:
            job: Job description data
            
        Returns:
            Required field of study if specified
        """
        job_text = str(job.metadata.get('description', '')).lower()
        
        # Check for field mentions
        for field, keywords in self._field_mappings.items():
            if any(keyword in job_text for keyword in keywords):
                return field
        
        return None
    
    def _get_highest_degree(self, resume: ParsedResume) -> str:
        """Get candidate's highest degree level.
        
        Args:
            resume: Resume data
            
        Returns:
            Highest degree level
        """
        if not resume.education:
            return 'high_school'  # Assume high school if no education listed
        
        highest_order = -1
        highest_degree = 'high_school'
        
        for edu in resume.education:
            degree_text = edu.degree.lower()
            
            for degree, info in self._degree_levels.items():
                if any(keyword in degree_text for keyword in info['keywords']):
                    if info['order'] > highest_order:
                        highest_order = info['order']
                        highest_degree = degree
                    break
        
        return highest_degree
    
    def _get_primary_field(self, resume: ParsedResume) -> Optional[str]:
        """Get candidate's primary field of study.
        
        Args:
            resume: Resume data
            
        Returns:
            Primary field of study
        """
        if not resume.education:
            return None
        
        # Get field from highest degree
        highest_degree_edu = None
        highest_order = -1
        
        for edu in resume.education:
            degree_text = edu.degree.lower()
            
            for degree, info in self._degree_levels.items():
                if any(keyword in degree_text for keyword in info['keywords']):
                    if info['order'] > highest_order:
                        highest_order = info['order']
                        highest_degree_edu = edu
                    break
        
        if highest_degree_edu:
            major_text = highest_degree_edu.major.lower()
            
            for field, keywords in self._field_mappings.items():
                if any(keyword in major_text for keyword in keywords):
                    return field
        
        return None
    
    def _calculate_degree_gap(self, candidate_degree: str, required_degree: str) -> str:
        """Calculate gap between candidate and required degree.
        
        Args:
            candidate_degree: Candidate's degree level
            required_degree: Required degree level
            
        Returns:
            Degree gap description
        """
        candidate_order = self._degree_levels.get(candidate_degree, {}).get('order', 0)
        required_order = self._degree_levels.get(required_degree, {}).get('order', 2)
        
        gap = required_order - candidate_order
        
        if gap > 0:
            return f"needs_{gap}_level_increase"
        elif gap < 0:
            return f"overqualified_by_{abs(gap)}_levels"
        else:
            return "meets_requirement"
    
    def _check_field_match(self, 
                         candidate_field: Optional[str], 
                         required_field: Optional[str], 
                         job: ParsedJobDescription) -> bool:
        """Check if candidate's field matches job requirements.
        
        Args:
            candidate_field: Candidate's field of study
            required_field: Required field of study
            job: Job description data
            
        Returns:
            True if field matches or is acceptable
        """
        if not required_field:
            return True  # No specific field required
        
        if not candidate_field:
            return False  # Field required but candidate has none
        
        if candidate_field == required_field:
            return True  # Exact match
        
        # Check for related fields
        related_fields = {
            'computer_science': ['engineering', 'mathematics'],
            'engineering': ['computer_science', 'mathematics', 'science'],
            'business': ['economics', 'mathematics'],
            'mathematics': ['computer_science', 'engineering', 'science'],
            'science': ['engineering', 'mathematics']
        }
        
        if required_field in related_fields.get(candidate_field, []):
            return True
        
        return False
    
    def _identify_certification_gaps(self, 
                                   resume: ParsedResume, 
                                   job: ParsedJobDescription) -> List[str]:
        """Identify missing certifications relevant to the job.
        
        Args:
            resume: Resume data
            job: Job description data
            
        Returns:
            List of missing relevant certifications
        """
        job_text = str(job.metadata.get('description', '')).lower()
        
        # Get candidate certifications
        candidate_certs = set()
        for cert in resume.certifications:
            candidate_certs.add(cert.name.lower())
        
        # Identify relevant certification categories
        relevant_categories = []
        if any(keyword in job_text for keyword in ['software', 'developer', 'engineer', 'tech']):
            relevant_categories.append('technology')
        
        if any(keyword in job_text for keyword in ['data', 'analytics', 'scientist']):
            relevant_categories.append('data_science')
        
        if any(keyword in job_text for keyword in ['finance', 'financial', 'accounting']):
            relevant_categories.append('finance')
        
        if any(keyword in job_text for keyword in ['marketing', 'digital', 'social media']):
            relevant_categories.append('marketing')
        
        # Find missing certifications
        missing_certs = []
        for category in relevant_categories:
            for cert in self._relevant_certifications.get(category, []):
                if cert.lower() not in candidate_certs:
                    # Check if explicitly mentioned in job
                    if any(part.lower() in job_text for part in cert.split()):
                        missing_certs.append(cert)
        
        return missing_certs[:5]  # Limit to top 5
    
    def _generate_alternative_paths(self, 
                                  degree_gap: str, 
                                  field_match: bool, 
                                  job: ParsedJobDescription) -> List[str]:
        """Generate alternative education/qualification paths.
        
        Args:
            degree_gap: Degree gap analysis
            field_match: Whether field matches
            job: Job description data
            
        Returns:
            List of alternative paths
        """
        alternatives = []
        
        # If degree gap exists
        if 'needs' in degree_gap:
            alternatives.extend([
                "Complete online bachelor's/master's degree program",
                "Pursue professional certifications in relevant field",
                "Attend coding bootcamp or intensive training program",
                "Gain equivalent work experience (2-3 years per degree level)"
            ])
        
        # If field doesn't match
        if not field_match:
            alternatives.extend([
                "Complete specialized courses in required field",
                "Pursue industry-specific certifications",
                "Gain relevant project experience through freelancing",
                "Complete online specialization programs"
            ])
        
        # Job-specific alternatives
        job_text = str(job.metadata.get('description', '')).lower()
        
        if any(keyword in job_text for keyword in ['programming', 'software', 'developer']):
            alternatives.extend([
                "Complete coding bootcamp (3-6 months)",
                "Build portfolio of programming projects",
                "Contribute to open source projects"
            ])
        
        if any(keyword in job_text for keyword in ['data', 'analytics']):
            alternatives.extend([
                "Complete data science bootcamp",
                "Earn Google Data Analytics Certificate",
                "Build data analysis portfolio"
            ])
        
        # Remove duplicates and limit
        return list(dict.fromkeys(alternatives))[:6]
    
    def _calculate_education_roi(self, 
                               degree_gap: str, 
                               alternative_paths: List[str], 
                               job: ParsedJobDescription) -> Dict[str, Any]:
        """Calculate ROI analysis for education investments.
        
        Args:
            degree_gap: Degree gap analysis
            alternative_paths: List of alternative paths
            job: Job description data
            
        Returns:
            ROI analysis dictionary
        """
        roi_analysis = {
            'formal_degree': self._calculate_formal_degree_roi(degree_gap),
            'alternative_paths': self._calculate_alternative_paths_roi(alternative_paths),
            'recommendations': []
        }
        
        # Generate recommendations based on ROI
        formal_roi = roi_analysis['formal_degree']['roi_score']
        best_alternative = max(roi_analysis['alternative_paths'], 
                             key=lambda x: x['roi_score']) if roi_analysis['alternative_paths'] else None
        
        if best_alternative and best_alternative['roi_score'] > formal_roi:
            roi_analysis['recommendations'].append(
                f"Consider {best_alternative['path']} for better ROI"
            )
        else:
            roi_analysis['recommendations'].append(
                "Formal degree may provide best long-term value"
            )
        
        return roi_analysis
    
    def _calculate_formal_degree_roi(self, degree_gap: str) -> Dict[str, Any]:
        """Calculate ROI for formal degree programs.
        
        Args:
            degree_gap: Degree gap analysis
            
        Returns:
            Formal degree ROI analysis
        """
        if 'needs_1' in degree_gap:  # Bachelor's needed
            return {
                'cost_estimate': 40000,  # Average bachelor's cost
                'time_months': 48,
                'salary_increase': 15000,  # Average increase
                'roi_score': 0.6,
                'payback_years': 2.7
            }
        elif 'needs_2' in degree_gap or 'master' in degree_gap:  # Master's needed
            return {
                'cost_estimate': 60000,  # Average master's cost
                'time_months': 24,
                'salary_increase': 20000,  # Average increase
                'roi_score': 0.7,
                'payback_years': 3.0
            }
        else:
            return {
                'cost_estimate': 0,
                'time_months': 0,
                'salary_increase': 0,
                'roi_score': 1.0,
                'payback_years': 0
            }
    
    def _calculate_alternative_paths_roi(self, alternative_paths: List[str]) -> List[Dict[str, Any]]:
        """Calculate ROI for alternative education paths.
        
        Args:
            alternative_paths: List of alternative paths
            
        Returns:
            List of ROI analyses for alternatives
        """
        roi_analyses = []
        
        for path in alternative_paths:
            path_lower = path.lower()
            
            if 'bootcamp' in path_lower:
                roi_analyses.append({
                    'path': path,
                    'cost_estimate': 12000,
                    'time_months': 6,
                    'salary_increase': 12000,
                    'roi_score': 0.8,
                    'payback_years': 1.0
                })
            elif 'certification' in path_lower:
                roi_analyses.append({
                    'path': path,
                    'cost_estimate': 2000,
                    'time_months': 4,
                    'salary_increase': 8000,
                    'roi_score': 0.9,
                    'payback_years': 0.25
                })
            elif 'online' in path_lower:
                roi_analyses.append({
                    'path': path,
                    'cost_estimate': 1000,
                    'time_months': 6,
                    'salary_increase': 5000,
                    'roi_score': 0.85,
                    'payback_years': 0.2
                })
            else:
                roi_analyses.append({
                    'path': path,
                    'cost_estimate': 5000,
                    'time_months': 12,
                    'salary_increase': 7000,
                    'roi_score': 0.7,
                    'payback_years': 0.7
                })
        
        return roi_analyses


class GapAnalysisEngine:
    """Main engine that orchestrates all gap analysis components.
    
    This class integrates skills, experience, and education gap analyzers
    to provide comprehensive gap analysis with improvement recommendations.
    """
    
    def __init__(self, 
                 skills_ontology: Optional[Dict[str, List[str]]] = None,
                 synonym_mapping: Optional[Dict[str, List[str]]] = None):
        """Initialize gap analysis engine.
        
        Args:
            skills_ontology: Skills ontology for categorization
            synonym_mapping: Skill synonym mapping
        """
        self.skills_analyzer = SkillsGapAnalyzer(skills_ontology, synonym_mapping)
        self.experience_analyzer = ExperienceGapAnalyzer()
        self.education_analyzer = EducationGapAnalyzer()
        
        logger.info("Gap analysis engine initialized")
    
    def analyze_gaps(self, 
                    resume: ParsedResume, 
                    job: ParsedJobDescription) -> GapAnalysisResult:
        """Perform comprehensive gap analysis.
        
        Args:
            resume: Parsed resume data
            job: Parsed job description data
            
        Returns:
            Complete gap analysis result
        """
        try:
            logger.info("Performing comprehensive gap analysis")
            
            # Analyze individual gap types
            skills_gaps = self.skills_analyzer.analyze_skills_gaps(resume, job)
            experience_gap = self.experience_analyzer.analyze_experience_gap(resume, job)
            education_gap = self.education_analyzer.analyze_education_gap(resume, job)
            
            # Calculate overall gap score
            overall_gap_score = self._calculate_overall_gap_score(
                skills_gaps, experience_gap, education_gap
            )
            
            # Calculate improvement potential
            improvement_potential = self._calculate_improvement_potential(
                skills_gaps, experience_gap, education_gap
            )
            
            # Generate priority recommendations
            priority_recommendations = self._generate_priority_recommendations(
                skills_gaps, experience_gap, education_gap
            )
            
            # Create metadata
            metadata = {
                'analysis_timestamp': logger.handlers[0].formatter.formatTime(
                    logger.makeRecord('', 0, '', 0, '', (), None)
                ) if logger.handlers else 'unknown',
                'total_skills_gaps': len(skills_gaps),
                'high_priority_skills': len([g for g in skills_gaps if g.priority == 'high']),
                'experience_shortfall_years': experience_gap.shortfall_years,
                'education_gap_severity': education_gap.degree_gap,
                'improvement_areas': len(priority_recommendations)
            }
            
            result = GapAnalysisResult(
                skills_gaps=skills_gaps,
                experience_gap=experience_gap,
                education_gap=education_gap,
                overall_gap_score=overall_gap_score,
                improvement_potential=improvement_potential,
                priority_recommendations=priority_recommendations,
                metadata=metadata
            )
            
            logger.info(f"Gap analysis completed. Overall gap score: {overall_gap_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive gap analysis failed: {e}")
            # Return empty result on failure
            return GapAnalysisResult(
                skills_gaps=[],
                experience_gap=ExperienceGap(0, 0, 0, 0, False, "unknown"),
                education_gap=EducationGap("unknown", "unknown", "unknown", False),
                overall_gap_score=1.0,
                improvement_potential=0.0,
                priority_recommendations=[],
                metadata={'error': str(e)}
            )
    
    def _calculate_overall_gap_score(self, 
                                   skills_gaps: List[SkillGap],
                                   experience_gap: ExperienceGap,
                                   education_gap: EducationGap) -> float:
        """Calculate overall gap score (0-1, lower is better).
        
        Args:
            skills_gaps: List of skill gaps
            experience_gap: Experience gap analysis
            education_gap: Education gap analysis
            
        Returns:
            Overall gap score
        """
        # Skills gap component (0-1)
        if skills_gaps:
            high_priority_count = len([g for g in skills_gaps if g.priority == 'high'])
            skills_score = min(1.0, (high_priority_count * 0.2) + (len(skills_gaps) * 0.05))
        else:
            skills_score = 0.0
        
        # Experience gap component (0-1)
        if experience_gap.shortfall_years > 0:
            exp_score = min(1.0, experience_gap.shortfall_years * 0.2)
        else:
            exp_score = 0.0
        
        # Adjust for relevance
        exp_score *= (1.0 - experience_gap.relevance_score)
        
        # Education gap component (0-1)
        if 'needs' in education_gap.degree_gap:
            edu_score = 0.3
        elif not education_gap.field_match:
            edu_score = 0.2
        else:
            edu_score = 0.0
        
        # Weighted combination
        overall_score = (skills_score * 0.5) + (exp_score * 0.3) + (edu_score * 0.2)
        
        return min(1.0, overall_score)
    
    def _calculate_improvement_potential(self, 
                                       skills_gaps: List[SkillGap],
                                       experience_gap: ExperienceGap,
                                       education_gap: EducationGap) -> float:
        """Calculate potential improvement from addressing gaps.
        
        Args:
            skills_gaps: List of skill gaps
            experience_gap: Experience gap analysis
            education_gap: Education gap analysis
            
        Returns:
            Improvement potential (0-1)
        """
        # Skills improvement potential
        skills_potential = sum(gap.impact_score for gap in skills_gaps[:5])  # Top 5 skills
        skills_potential = min(0.4, skills_potential)  # Cap at 40%
        
        # Experience improvement potential
        if experience_gap.shortfall_years > 0:
            exp_potential = min(0.3, experience_gap.shortfall_years * 0.1)
        else:
            exp_potential = 0.0
        
        # Education improvement potential
        if 'needs' in education_gap.degree_gap:
            edu_potential = 0.2
        elif not education_gap.field_match:
            edu_potential = 0.1
        else:
            edu_potential = 0.0
        
        return skills_potential + exp_potential + edu_potential
    
    def _generate_priority_recommendations(self, 
                                         skills_gaps: List[SkillGap],
                                         experience_gap: ExperienceGap,
                                         education_gap: EducationGap) -> List[str]:
        """Generate priority recommendations for improvement.
        
        Args:
            skills_gaps: List of skill gaps
            experience_gap: Experience gap analysis
            education_gap: Education gap analysis
            
        Returns:
            List of priority recommendations
        """
        recommendations = []
        
        # Top skill recommendations
        high_priority_skills = [g for g in skills_gaps if g.priority == 'high'][:3]
        for skill_gap in high_priority_skills:
            recommendations.append(
                f"Learn {skill_gap.skill} ({skill_gap.category}) - High impact on match score"
            )
        
        # Experience recommendations
        if experience_gap.shortfall_years > 2:
            recommendations.append(
                f"Gain {experience_gap.shortfall_years:.1f} more years of relevant experience"
            )
        elif experience_gap.relevance_score < 0.5:
            recommendations.append(
                "Focus on gaining more relevant industry experience"
            )
        
        # Education recommendations
        if 'needs' in education_gap.degree_gap:
            if education_gap.alternative_paths:
                best_alternative = education_gap.alternative_paths[0]
                recommendations.append(f"Consider {best_alternative} as degree alternative")
            else:
                recommendations.append("Complete required degree program")
        
        # Certification recommendations
        if education_gap.certification_gaps:
            top_cert = education_gap.certification_gaps[0]
            recommendations.append(f"Obtain {top_cert} certification")
        
        return recommendations[:5]  # Limit to top 5