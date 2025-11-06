"""
Sub-scoring engine for detailed resume-job matching analysis.

This module implements detailed sub-scoring for skills, experience, and education
matching between resumes and job descriptions with exact and fuzzy matching,
confidence scoring, and gap analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from .resume_interfaces import ParsedResume, WorkExperience, Education
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SkillMatch:
    """Represents a skill match between resume and job description.
    
    Attributes:
        skill: The skill name
        match_type: Type of match ('exact', 'fuzzy', 'synonym')
        confidence: Confidence score (0-1)
        resume_context: Context where skill appears in resume
        job_context: Context where skill appears in job description
        category: Skill category (technical, soft, tools)
    """
    skill: str
    match_type: str
    confidence: float
    resume_context: str = ""
    job_context: str = ""
    category: str = ""


@dataclass
class SkillsAnalysisResult:
    """Result of skills similarity analysis.
    
    Attributes:
        overall_score: Overall skills similarity score (0-100)
        matched_skills: List of matched skills
        missing_skills: List of skills required but not found in resume
        extra_skills: List of skills in resume but not required
        category_scores: Scores by skill category
        confidence_score: Overall confidence in the analysis
        metadata: Additional analysis metadata
    """
    overall_score: float
    matched_skills: List[SkillMatch]
    missing_skills: List[str]
    extra_skills: List[str]
    category_scores: Dict[str, float]
    confidence_score: float
    metadata: Dict[str, Any]


class SkillsSimilarityScorer:
    """Skills similarity scorer with exact and fuzzy matching capabilities."""
    
    def __init__(self,
                 fuzzy_threshold: float = 0.8,
                 enable_synonym_matching: bool = True,
                 category_weights: Optional[Dict[str, float]] = None):
        """Initialize skills similarity scorer."""
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_synonym_matching = enable_synonym_matching
        
        self.category_weights = category_weights or {
            'technical': 0.4,
            'tools': 0.3,
            'soft': 0.2,
            'domain': 0.1
        }
        
        self.skill_synonyms = {
            'javascript': ['js', 'ecmascript', 'node.js', 'nodejs'],
            'python': ['py'],
            'machine learning': ['ml'],
            'artificial intelligence': ['ai']
        }
        
        self._scoring_stats = {
            'total_comparisons': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'synonym_matches': 0,
            'no_matches': 0
        }
        
        logger.info("Skills similarity scorer initialized")
    
    def calculate_skills_similarity(self,
                                  resume: ParsedResume,
                                  job_description: ParsedJobDescription) -> SkillsAnalysisResult:
        """Calculate comprehensive skills similarity."""
        try:
            logger.debug("Calculating skills similarity")
            
            resume_skills = [skill.lower().strip() for skill in resume.skills if skill]
            job_skills = [skill.lower().strip() for skill in job_description.skills_required if skill]
            
            matched_skills = []
            missing_skills = []
            matched_resume_skills = set()
            
            for job_skill in job_skills:
                best_match = None
                best_score = 0.0
                
                for resume_skill in resume_skills:
                    is_match, match_type, confidence = self._is_skill_match(job_skill, resume_skill)
                    
                    if is_match and confidence > best_score:
                        best_score = confidence
                        best_match = SkillMatch(
                            skill=resume_skill,
                            match_type=match_type,
                            confidence=confidence,
                            category=self._get_skill_category(job_skill, job_description.categories)
                        )
                
                if best_match:
                    matched_skills.append(best_match)
                    matched_resume_skills.add(best_match.skill)
                    self._update_match_stats(best_match.match_type)
                else:
                    missing_skills.append(job_skill)
                    self._scoring_stats['no_matches'] += 1
            
            extra_skills = [skill for skill in resume_skills 
                          if skill not in matched_resume_skills]
            
            category_scores = self._calculate_category_scores(
                matched_skills, missing_skills, job_description.categories
            )
            
            overall_score = self._calculate_overall_skills_score(
                matched_skills, missing_skills, job_skills, category_scores
            )
            
            confidence_score = self._calculate_skills_confidence(
                matched_skills, len(job_skills)
            )
            
            metadata = {
                'total_job_skills': len(job_skills),
                'total_resume_skills': len(resume_skills),
                'matched_count': len(matched_skills),
                'missing_count': len(missing_skills),
                'extra_count': len(extra_skills),
                'match_rate': len(matched_skills) / len(job_skills) if job_skills else 0.0
            }
            
            self._scoring_stats['total_comparisons'] += 1
            
            result = SkillsAnalysisResult(
                overall_score=overall_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                extra_skills=extra_skills,
                category_scores=category_scores,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
            logger.debug(f"Skills similarity calculated: {overall_score:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Skills similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate skills similarity: {e}")
    
    def _is_skill_match(self, job_skill: str, resume_skill: str) -> Tuple[bool, str, float]:
        """Check if two skills match."""
        job_skill_norm = job_skill.lower().strip()
        resume_skill_norm = resume_skill.lower().strip()
        
        # Exact match
        if job_skill_norm == resume_skill_norm:
            return True, 'exact', 1.0
        
        # Substring matches
        if job_skill_norm in resume_skill_norm or resume_skill_norm in job_skill_norm:
            shorter = min(len(job_skill_norm), len(resume_skill_norm))
            longer = max(len(job_skill_norm), len(resume_skill_norm))
            confidence = shorter / longer
            if confidence >= 0.7:
                return True, 'exact', confidence
        
        # Synonym matching
        if self.enable_synonym_matching:
            synonym_confidence = self._check_synonym_match(job_skill_norm, resume_skill_norm)
            if synonym_confidence > 0.8:
                return True, 'synonym', synonym_confidence
        
        # Fuzzy matching
        fuzzy_confidence = SequenceMatcher(None, job_skill_norm, resume_skill_norm).ratio()
        if fuzzy_confidence >= self.fuzzy_threshold:
            return True, 'fuzzy', fuzzy_confidence
        
        return False, 'none', 0.0
    
    def _check_synonym_match(self, skill1: str, skill2: str) -> float:
        """Check if two skills are synonyms."""
        for main_skill, synonyms in self.skill_synonyms.items():
            if skill1 in [main_skill] + synonyms and skill2 in [main_skill] + synonyms:
                return 0.9
        return 0.0
    
    def _get_skill_category(self, skill: str, job_categories: Dict[str, List[str]]) -> str:
        """Get the category of a skill."""
        skill_lower = skill.lower()
        
        for category, skills in job_categories.items():
            if any(skill_lower == s.lower() for s in skills):
                return category
        
        # Default categorization
        if any(tech in skill_lower for tech in ['python', 'java', 'javascript']):
            return 'technical'
        elif any(tool in skill_lower for tool in ['aws', 'docker', 'git']):
            return 'tools'
        elif any(soft in skill_lower for soft in ['leadership', 'communication']):
            return 'soft'
        else:
            return 'domain'
    
    def _calculate_category_scores(self,
                                 matched_skills: List[SkillMatch],
                                 missing_skills: List[str],
                                 job_categories: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate scores for each skill category."""
        category_scores = {}
        
        for category in job_categories.keys():
            category_job_skills = [s.lower() for s in job_categories[category]]
            category_matched = [
                match for match in matched_skills
                if match.category == category
            ]
            
            total_category_skills = len(category_job_skills)
            if total_category_skills == 0:
                category_scores[category] = 100.0
                continue
            
            matched_score = sum(match.confidence for match in category_matched)
            max_possible_score = total_category_skills
            
            category_score = (matched_score / max_possible_score) * 100.0
            category_scores[category] = min(100.0, category_score)
        
        return category_scores
    
    def _calculate_overall_skills_score(self,
                                      matched_skills: List[SkillMatch],
                                      missing_skills: List[str],
                                      job_skills: List[str],
                                      category_scores: Dict[str, float]) -> float:
        """Calculate overall skills similarity score."""
        if not job_skills:
            return 100.0
        
        # Simple match ratio with confidence weighting
        total_confidence = sum(match.confidence for match in matched_skills)
        max_possible_confidence = len(job_skills)
        simple_score = (total_confidence / max_possible_confidence) * 100.0
        
        # Category-weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        else:
            weighted_score = simple_score
        
        # Combine both methods
        final_score = (simple_score * 0.6) + (weighted_score * 0.4)
        
        return min(100.0, max(0.0, final_score))
    
    def _calculate_skills_confidence(self,
                                   matched_skills: List[SkillMatch],
                                   total_job_skills: int) -> float:
        """Calculate confidence score for skills analysis."""
        if total_job_skills == 0:
            return 1.0
        
        match_rate = len(matched_skills) / total_job_skills
        
        if matched_skills:
            avg_match_confidence = np.mean([match.confidence for match in matched_skills])
            quality_factor = avg_match_confidence
        else:
            quality_factor = 0.0
        
        confidence = (match_rate * 0.7) + (quality_factor * 0.3)
        return min(1.0, max(0.0, confidence))
    
    def _update_match_stats(self, match_type: str):
        """Update match statistics."""
        if match_type == 'exact':
            self._scoring_stats['exact_matches'] += 1
        elif match_type == 'fuzzy':
            self._scoring_stats['fuzzy_matches'] += 1
        elif match_type == 'synonym':
            self._scoring_stats['synonym_matches'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for skills scoring."""
        stats = self._scoring_stats.copy()
        
        total_matches = sum([
            stats['exact_matches'],
            stats['fuzzy_matches'],
            stats['synonym_matches']
        ])
        
        if total_matches > 0:
            stats['exact_match_rate'] = stats['exact_matches'] / total_matches
            stats['fuzzy_match_rate'] = stats['fuzzy_matches'] / total_matches
            stats['synonym_match_rate'] = stats['synonym_matches'] / total_matches
        else:
            stats['exact_match_rate'] = 0.0
            stats['fuzzy_match_rate'] = 0.0
            stats['synonym_match_rate'] = 0.0
        
        stats['configuration'] = {
            'fuzzy_threshold': self.fuzzy_threshold,
            'enable_synonym_matching': self.enable_synonym_matching,
            'category_weights': self.category_weights
        }
        
        return stats


@dataclass
class ExperienceAnalysisResult:
    """Result of experience matching analysis."""
    overall_score: float
    years_experience: float
    required_years: float
    seniority_match: float
    industry_relevance: float
    progression_score: float
    confidence_score: float
    metadata: Dict[str, Any]


class ExperienceMatchingScorer:
    """Experience matching scorer for resume-job comparison."""
    
    def __init__(self):
        """Initialize experience matching scorer."""
        self.seniority_thresholds = {
            'entry': (0.0, 2.0),
            'junior': (1.0, 3.0),
            'mid': (2.0, 6.0),
            'senior': (5.0, 10.0),
            'lead': (7.0, 15.0),
            'principal': (10.0, float('inf')),
            'executive': (15.0, float('inf'))
        }
        
        self._experience_stats = {
            'total_analyses': 0,
            'seniority_matches': 0,
            'industry_matches': 0,
            'progression_detected': 0
        }
        
        logger.info("Experience matching scorer initialized")
    
    def calculate_experience_similarity(self,
                                      resume: ParsedResume,
                                      job_description: ParsedJobDescription,
                                      job_text: str = "") -> ExperienceAnalysisResult:
        """Calculate comprehensive experience similarity."""
        try:
            logger.debug("Calculating experience similarity")
            
            # Calculate total years of experience
            total_years = sum(exp.duration_months for exp in resume.experience) / 12.0
            
            # Extract required experience (simplified)
            required_years = 5.0  # Default for senior level
            if 'entry' in job_description.experience_level.lower():
                required_years = 1.0
            elif 'junior' in job_description.experience_level.lower():
                required_years = 2.0
            elif 'senior' in job_description.experience_level.lower():
                required_years = 5.0
            
            # Calculate seniority match (simplified)
            seniority_match = 80.0 if total_years >= required_years else 60.0
            
            # Calculate industry relevance (simplified)
            industry_relevance = 75.0  # Default
            
            # Calculate progression score (simplified)
            progression_score = 70.0 if len(resume.experience) > 1 else 50.0
            
            # Calculate overall score
            overall_score = (
                (total_years / required_years * 40.0 if required_years > 0 else 40.0) +
                (seniority_match * 0.3) +
                (industry_relevance * 0.2) +
                (progression_score * 0.1)
            )
            overall_score = min(100.0, overall_score)
            
            # Calculate confidence
            confidence_score = min(1.0, len(resume.experience) / 3.0)
            
            # Create metadata
            metadata = {
                'total_positions': len(resume.experience),
                'experience_gap': required_years - total_years,
                'avg_position_duration': np.mean([exp.duration_months for exp in resume.experience]) if resume.experience else 0
            }
            
            # Update stats
            self._experience_stats['total_analyses'] += 1
            if seniority_match > 70:
                self._experience_stats['seniority_matches'] += 1
            
            result = ExperienceAnalysisResult(
                overall_score=overall_score,
                years_experience=total_years,
                required_years=required_years,
                seniority_match=seniority_match,
                industry_relevance=industry_relevance,
                progression_score=progression_score,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
            logger.debug(f"Experience similarity calculated: {overall_score:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Experience similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate experience similarity: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for experience scoring."""
        stats = self._experience_stats.copy()
        
        total = stats['total_analyses']
        if total > 0:
            stats['seniority_match_rate'] = stats['seniority_matches'] / total
            stats['industry_match_rate'] = stats['industry_matches'] / total
            stats['progression_detection_rate'] = stats['progression_detected'] / total
        else:
            stats['seniority_match_rate'] = 0.0
            stats['industry_match_rate'] = 0.0
            stats['progression_detection_rate'] = 0.0
        
        stats['configuration'] = {
            'seniority_thresholds': self.seniority_thresholds
        }
        
        return stats