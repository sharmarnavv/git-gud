"""
Comprehensive sub-scoring engine that integrates skills, experience, and education scoring.

This module provides a unified interface for detailed resume-job matching analysis
with comprehensive sub-scores and gap analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .resume_interfaces import ParsedResume
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

# Import the individual scoring components
try:
    from .sub_scoring_engine import SkillsSimilarityScorer, SkillsAnalysisResult
    from .sub_scoring_engine import ExperienceMatchingScorer, ExperienceAnalysisResult
except ImportError:
    # Fallback if sub_scoring_engine has issues
    SkillsSimilarityScorer = None
    ExperienceMatchingScorer = None
    SkillsAnalysisResult = None
    ExperienceAnalysisResult = None

from .education_scoring import EducationRequirementsScorer, EducationAnalysisResult

logger = get_logger(__name__)


@dataclass
class ComprehensiveMatchResult:
    """Comprehensive matching result with all sub-scores.
    
    Attributes:
        overall_score: Overall match score (0-100)
        skills_analysis: Skills matching analysis result
        experience_analysis: Experience matching analysis result
        education_analysis: Education matching analysis result
        weighted_scores: Weighted component scores
        gap_analysis: Comprehensive gap analysis
        improvement_suggestions: Prioritized improvement suggestions
        confidence_score: Overall confidence in the analysis
        metadata: Additional analysis metadata
    """
    overall_score: float
    skills_analysis: Optional[Any]  # SkillsAnalysisResult
    experience_analysis: Optional[Any]  # ExperienceAnalysisResult
    education_analysis: EducationAnalysisResult
    weighted_scores: Dict[str, float]
    gap_analysis: Dict[str, Any]
    improvement_suggestions: List[Dict[str, Any]]
    confidence_score: float
    metadata: Dict[str, Any]


class ComprehensiveSubScoringEngine:
    """Comprehensive sub-scoring engine for resume-job matching.
    
    This class integrates skills, experience, and education scoring to provide
    detailed analysis with gap identification and improvement suggestions.
    """
    
    def __init__(self,
                 component_weights: Optional[Dict[str, float]] = None,
                 skills_config: Optional[Dict[str, Any]] = None,
                 experience_config: Optional[Dict[str, Any]] = None,
                 education_config: Optional[Dict[str, Any]] = None):
        """Initialize comprehensive sub-scoring engine.
        
        Args:
            component_weights: Weights for different components
            skills_config: Configuration for skills scorer
            experience_config: Configuration for experience scorer
            education_config: Configuration for education scorer
        """
        # Component weights (should sum to 1.0)
        self.component_weights = component_weights or {
            'skills': 0.4,
            'experience': 0.35,
            'education': 0.25
        }
        
        # Validate weights
        total_weight = sum(self.component_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Component weights sum to {total_weight}, normalizing to 1.0")
            for key in self.component_weights:
                self.component_weights[key] /= total_weight
        
        # Initialize component scorers
        try:
            if SkillsSimilarityScorer:
                skills_config = skills_config or {}
                self.skills_scorer = SkillsSimilarityScorer(**skills_config)
            else:
                self.skills_scorer = None
                logger.warning("Skills scorer not available")
        except Exception as e:
            logger.error(f"Failed to initialize skills scorer: {e}")
            self.skills_scorer = None
        
        try:
            if ExperienceMatchingScorer:
                experience_config = experience_config or {}
                self.experience_scorer = ExperienceMatchingScorer(**experience_config)
            else:
                self.experience_scorer = None
                logger.warning("Experience scorer not available")
        except Exception as e:
            logger.error(f"Failed to initialize experience scorer: {e}")
            self.experience_scorer = None
        
        try:
            education_config = education_config or {}
            self.education_scorer = EducationRequirementsScorer(**education_config)
        except Exception as e:
            logger.error(f"Failed to initialize education scorer: {e}")
            raise ValueError(f"Education scorer initialization failed: {e}")
        
        # Performance tracking
        self._scoring_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'component_failures': {
                'skills': 0,
                'experience': 0,
                'education': 0
            },
            'average_scores': {
                'overall': 0.0,
                'skills': 0.0,
                'experience': 0.0,
                'education': 0.0
            }
        }
        
        logger.info("Comprehensive sub-scoring engine initialized")
    
    def calculate_comprehensive_match(self,
                                    resume: ParsedResume,
                                    job_description: ParsedJobDescription,
                                    job_text: str = "") -> ComprehensiveMatchResult:
        """Calculate comprehensive match score with detailed sub-analysis.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description data
            job_text: Original job description text
            
        Returns:
            ComprehensiveMatchResult with detailed analysis
        """
        try:
            logger.debug("Starting comprehensive match analysis")
            
            # Initialize results
            skills_analysis = None
            experience_analysis = None
            education_analysis = None
            
            # Calculate skills similarity
            if self.skills_scorer:
                try:
                    skills_analysis = self.skills_scorer.calculate_skills_similarity(
                        resume, job_description
                    )
                    logger.debug(f"Skills analysis completed: {skills_analysis.overall_score:.1f}%")
                except Exception as e:
                    logger.error(f"Skills analysis failed: {e}")
                    self._scoring_stats['component_failures']['skills'] += 1
            
            # Calculate experience similarity
            if self.experience_scorer:
                try:
                    experience_analysis = self.experience_scorer.calculate_experience_similarity(
                        resume, job_description, job_text
                    )
                    logger.debug(f"Experience analysis completed: {experience_analysis.overall_score:.1f}%")
                except Exception as e:
                    logger.error(f"Experience analysis failed: {e}")
                    self._scoring_stats['component_failures']['experience'] += 1
            
            # Calculate education similarity
            try:
                education_analysis = self.education_scorer.calculate_education_similarity(
                    resume, job_description, job_text
                )
                logger.debug(f"Education analysis completed: {education_analysis.overall_score:.1f}%")
            except Exception as e:
                logger.error(f"Education analysis failed: {e}")
                self._scoring_stats['component_failures']['education'] += 1
                # Create default education analysis
                education_analysis = self._create_default_education_analysis()
            
            # Calculate weighted scores
            weighted_scores = self._calculate_weighted_scores(
                skills_analysis, experience_analysis, education_analysis
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(weighted_scores)
            
            # Perform gap analysis
            gap_analysis = self._perform_gap_analysis(
                skills_analysis, experience_analysis, education_analysis
            )
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                gap_analysis, skills_analysis, experience_analysis, education_analysis
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                skills_analysis, experience_analysis, education_analysis
            )
            
            # Create metadata
            metadata = self._create_analysis_metadata(
                resume, job_description, skills_analysis, 
                experience_analysis, education_analysis
            )
            
            # Update performance statistics
            self._update_performance_stats(
                overall_score, skills_analysis, experience_analysis, education_analysis
            )
            
            result = ComprehensiveMatchResult(
                overall_score=overall_score,
                skills_analysis=skills_analysis,
                experience_analysis=experience_analysis,
                education_analysis=education_analysis,
                weighted_scores=weighted_scores,
                gap_analysis=gap_analysis,
                improvement_suggestions=improvement_suggestions,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
            logger.info(f"Comprehensive analysis completed: {overall_score:.1f}% overall match")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive match calculation failed: {e}")
            raise ValueError(f"Failed to calculate comprehensive match: {e}")
    
    def _calculate_weighted_scores(self,
                                 skills_analysis: Optional[Any],
                                 experience_analysis: Optional[Any],
                                 education_analysis: EducationAnalysisResult) -> Dict[str, float]:
        """Calculate weighted component scores.
        
        Args:
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            Dictionary with weighted scores
        """
        weighted_scores = {}
        
        # Skills score
        if skills_analysis:
            skills_score = skills_analysis.overall_score
        else:
            skills_score = 0.0
            logger.warning("Skills analysis not available, using 0.0")
        
        weighted_scores['skills'] = skills_score * self.component_weights['skills']
        
        # Experience score
        if experience_analysis:
            experience_score = experience_analysis.overall_score
        else:
            experience_score = 0.0
            logger.warning("Experience analysis not available, using 0.0")
        
        weighted_scores['experience'] = experience_score * self.component_weights['experience']
        
        # Education score
        education_score = education_analysis.overall_score
        weighted_scores['education'] = education_score * self.component_weights['education']
        
        # Store raw scores for reference
        weighted_scores['raw_skills'] = skills_score
        weighted_scores['raw_experience'] = experience_score
        weighted_scores['raw_education'] = education_score
        
        return weighted_scores
    
    def _calculate_overall_score(self, weighted_scores: Dict[str, float]) -> float:
        """Calculate overall match score from weighted components.
        
        Args:
            weighted_scores: Dictionary with weighted component scores
            
        Returns:
            Overall match score (0-100)
        """
        overall_score = (
            weighted_scores['skills'] +
            weighted_scores['experience'] +
            weighted_scores['education']
        )
        
        return min(100.0, max(0.0, overall_score))
    
    def _perform_gap_analysis(self,
                            skills_analysis: Optional[Any],
                            experience_analysis: Optional[Any],
                            education_analysis: EducationAnalysisResult) -> Dict[str, Any]:
        """Perform comprehensive gap analysis across all components.
        
        Args:
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            Dictionary with gap analysis results
        """
        gap_analysis = {
            'skills_gaps': [],
            'experience_gaps': [],
            'education_gaps': [],
            'priority_gaps': [],
            'impact_analysis': {}
        }
        
        # Skills gaps
        if skills_analysis:
            gap_analysis['skills_gaps'] = skills_analysis.missing_skills
            gap_analysis['impact_analysis']['skills'] = {
                'missing_count': len(skills_analysis.missing_skills),
                'match_rate': len(skills_analysis.matched_skills) / 
                            (len(skills_analysis.matched_skills) + len(skills_analysis.missing_skills))
                            if (skills_analysis.matched_skills or skills_analysis.missing_skills) else 0.0,
                'critical_missing': self._identify_critical_missing_skills(skills_analysis)
            }
        
        # Experience gaps
        if experience_analysis:
            experience_gaps = []
            
            # Years of experience gap
            if experience_analysis.years_experience < experience_analysis.required_years:
                gap_years = experience_analysis.required_years - experience_analysis.years_experience
                experience_gaps.append(f"Need {gap_years:.1f} more years of experience")
            
            # Seniority gap
            if experience_analysis.seniority_match < 70:
                experience_gaps.append("Seniority level mismatch")
            
            # Industry relevance gap
            if experience_analysis.industry_relevance < 70:
                experience_gaps.append("Limited relevant industry experience")
            
            gap_analysis['experience_gaps'] = experience_gaps
            gap_analysis['impact_analysis']['experience'] = {
                'years_gap': max(0, experience_analysis.required_years - experience_analysis.years_experience),
                'seniority_match': experience_analysis.seniority_match,
                'industry_relevance': experience_analysis.industry_relevance
            }
        
        # Education gaps
        gap_analysis['education_gaps'] = education_analysis.missing_requirements
        gap_analysis['impact_analysis']['education'] = {
            'missing_count': len(education_analysis.missing_requirements),
            'degree_match': education_analysis.degree_match_score,
            'field_match': education_analysis.field_match_score,
            'certification_match': education_analysis.certification_score
        }
        
        # Prioritize gaps by impact
        gap_analysis['priority_gaps'] = self._prioritize_gaps(
            gap_analysis, skills_analysis, experience_analysis, education_analysis
        )
        
        return gap_analysis
    
    def _identify_critical_missing_skills(self, skills_analysis: Any) -> List[str]:
        """Identify critical missing skills based on job requirements.
        
        Args:
            skills_analysis: Skills analysis result
            
        Returns:
            List of critical missing skills
        """
        # This would need access to job description categories to determine criticality
        # For now, return first few missing skills as critical
        return skills_analysis.missing_skills[:3] if skills_analysis.missing_skills else []
    
    def _prioritize_gaps(self,
                        gap_analysis: Dict[str, Any],
                        skills_analysis: Optional[Any],
                        experience_analysis: Optional[Any],
                        education_analysis: EducationAnalysisResult) -> List[Dict[str, Any]]:
        """Prioritize gaps by impact and feasibility.
        
        Args:
            gap_analysis: Gap analysis results
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            List of prioritized gaps with impact scores
        """
        prioritized_gaps = []
        
        # Skills gaps (high impact, medium feasibility)
        for skill in gap_analysis['skills_gaps'][:5]:  # Top 5 missing skills
            prioritized_gaps.append({
                'type': 'skill',
                'description': f"Missing skill: {skill}",
                'impact': 'high',
                'feasibility': 'medium',
                'priority_score': 8
            })
        
        # Experience gaps (medium impact, low feasibility)
        for exp_gap in gap_analysis['experience_gaps']:
            if 'years' in exp_gap.lower():
                feasibility = 'low'
                impact = 'high'
                priority_score = 6
            else:
                feasibility = 'medium'
                impact = 'medium'
                priority_score = 5
            
            prioritized_gaps.append({
                'type': 'experience',
                'description': exp_gap,
                'impact': impact,
                'feasibility': feasibility,
                'priority_score': priority_score
            })
        
        # Education gaps (low-medium impact, medium feasibility)
        for edu_gap in gap_analysis['education_gaps']:
            if 'certification' in edu_gap.lower():
                feasibility = 'high'
                impact = 'medium'
                priority_score = 7
            elif 'degree' in edu_gap.lower():
                feasibility = 'low'
                impact = 'medium'
                priority_score = 4
            else:
                feasibility = 'medium'
                impact = 'low'
                priority_score = 3
            
            prioritized_gaps.append({
                'type': 'education',
                'description': edu_gap,
                'impact': impact,
                'feasibility': feasibility,
                'priority_score': priority_score
            })
        
        # Sort by priority score (descending)
        prioritized_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized_gaps[:10]  # Return top 10 priority gaps
    
    def _generate_improvement_suggestions(self,
                                        gap_analysis: Dict[str, Any],
                                        skills_analysis: Optional[Any],
                                        experience_analysis: Optional[Any],
                                        education_analysis: EducationAnalysisResult) -> List[Dict[str, Any]]:
        """Generate prioritized improvement suggestions.
        
        Args:
            gap_analysis: Gap analysis results
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            List of improvement suggestions with priorities
        """
        suggestions = []
        
        # Skills improvement suggestions
        if skills_analysis and skills_analysis.missing_skills:
            for skill in skills_analysis.missing_skills[:3]:  # Top 3 missing skills
                suggestions.append({
                    'category': 'skills',
                    'type': 'skill_acquisition',
                    'title': f"Learn {skill}",
                    'description': f"Acquire {skill} through online courses, tutorials, or hands-on projects",
                    'priority': 'high',
                    'effort': 'medium',
                    'timeframe': '1-3 months',
                    'impact_score': 8
                })
        
        # Experience improvement suggestions
        if experience_analysis:
            if experience_analysis.years_experience < experience_analysis.required_years:
                suggestions.append({
                    'category': 'experience',
                    'type': 'experience_highlighting',
                    'title': "Highlight relevant experience",
                    'description': "Emphasize transferable skills and relevant projects to demonstrate equivalent experience",
                    'priority': 'high',
                    'effort': 'low',
                    'timeframe': 'immediate',
                    'impact_score': 7
                })
            
            if experience_analysis.industry_relevance < 70:
                suggestions.append({
                    'category': 'experience',
                    'type': 'industry_transition',
                    'title': "Demonstrate industry knowledge",
                    'description': "Highlight transferable skills and show understanding of industry-specific challenges",
                    'priority': 'medium',
                    'effort': 'medium',
                    'timeframe': '1-2 months',
                    'impact_score': 6
                })
        
        # Education improvement suggestions
        if education_analysis.missing_requirements:
            for req in education_analysis.missing_requirements[:2]:  # Top 2 missing requirements
                if 'certification' in req.lower():
                    suggestions.append({
                        'category': 'education',
                        'type': 'certification',
                        'title': f"Obtain {req.split(':', 1)[1].strip()} certification",
                        'description': f"Pursue {req.split(':', 1)[1].strip()} to meet job requirements",
                        'priority': 'medium',
                        'effort': 'medium',
                        'timeframe': '2-6 months',
                        'impact_score': 7
                    })
                elif 'degree' in req.lower():
                    suggestions.append({
                        'category': 'education',
                        'type': 'education_alternative',
                        'title': "Emphasize equivalent qualifications",
                        'description': "Highlight professional experience and certifications as degree alternatives",
                        'priority': 'medium',
                        'effort': 'low',
                        'timeframe': 'immediate',
                        'impact_score': 5
                    })
        
        # Add alternative path suggestions from education analysis
        for alt_path in education_analysis.alternative_paths[:2]:
            suggestions.append({
                'category': 'education',
                'type': 'alternative_path',
                'title': "Alternative qualification path",
                'description': alt_path,
                'priority': 'low',
                'effort': 'medium',
                'timeframe': '3-12 months',
                'impact_score': 4
            })
        
        # Sort suggestions by impact score (descending)
        suggestions.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return suggestions[:8]  # Return top 8 suggestions
    
    def _calculate_overall_confidence(self,
                                    skills_analysis: Optional[Any],
                                    experience_analysis: Optional[Any],
                                    education_analysis: EducationAnalysisResult) -> float:
        """Calculate overall confidence score for the analysis.
        
        Args:
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            Overall confidence score (0-1)
        """
        confidence_scores = []
        
        # Skills confidence
        if skills_analysis:
            confidence_scores.append(skills_analysis.confidence_score)
        
        # Experience confidence
        if experience_analysis:
            confidence_scores.append(experience_analysis.confidence_score)
        
        # Education confidence
        confidence_scores.append(education_analysis.confidence_score)
        
        # Calculate weighted average
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Default confidence
    
    def _create_analysis_metadata(self,
                                resume: ParsedResume,
                                job_description: ParsedJobDescription,
                                skills_analysis: Optional[Any],
                                experience_analysis: Optional[Any],
                                education_analysis: EducationAnalysisResult) -> Dict[str, Any]:
        """Create comprehensive analysis metadata.
        
        Args:
            resume: Parsed resume data
            job_description: Parsed job description data
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
            
        Returns:
            Dictionary with analysis metadata
        """
        metadata = {
            'analysis_components': {
                'skills_available': skills_analysis is not None,
                'experience_available': experience_analysis is not None,
                'education_available': True
            },
            'resume_summary': {
                'total_skills': len(resume.skills),
                'total_experience_entries': len(resume.experience),
                'total_education_entries': len(resume.education),
                'total_certifications': len(resume.certifications)
            },
            'job_summary': {
                'required_skills': len(job_description.skills_required),
                'experience_level': job_description.experience_level,
                'tools_mentioned': len(job_description.tools_mentioned)
            },
            'component_weights': self.component_weights.copy(),
            'analysis_timestamp': self._get_timestamp(),
            'scoring_engine_version': '1.0.0'
        }
        
        # Add component-specific metadata
        if skills_analysis:
            metadata['skills_metadata'] = skills_analysis.metadata
        
        if experience_analysis:
            metadata['experience_metadata'] = experience_analysis.metadata
        
        metadata['education_metadata'] = education_analysis.metadata
        
        return metadata
    
    def _create_default_education_analysis(self) -> EducationAnalysisResult:
        """Create default education analysis when scoring fails.
        
        Returns:
            Default EducationAnalysisResult
        """
        from .education_scoring import EducationAnalysisResult
        
        return EducationAnalysisResult(
            overall_score=50.0,
            degree_match_score=50.0,
            field_match_score=50.0,
            institution_score=50.0,
            certification_score=50.0,
            matched_requirements=[],
            missing_requirements=[],
            alternative_paths=[],
            confidence_score=0.0,
            metadata={'error': 'Education analysis failed, using default values'}
        )
    
    def _update_performance_stats(self,
                                overall_score: float,
                                skills_analysis: Optional[Any],
                                experience_analysis: Optional[Any],
                                education_analysis: EducationAnalysisResult):
        """Update performance statistics.
        
        Args:
            overall_score: Overall match score
            skills_analysis: Skills analysis result
            experience_analysis: Experience analysis result
            education_analysis: Education analysis result
        """
        self._scoring_stats['total_analyses'] += 1
        
        # Track successful analysis
        if skills_analysis or experience_analysis or education_analysis:
            self._scoring_stats['successful_analyses'] += 1
        
        # Update running averages
        n = self._scoring_stats['total_analyses']
        
        # Overall score average
        prev_avg = self._scoring_stats['average_scores']['overall']
        self._scoring_stats['average_scores']['overall'] = (
            (prev_avg * (n - 1) + overall_score) / n
        )
        
        # Component score averages
        if skills_analysis:
            prev_avg = self._scoring_stats['average_scores']['skills']
            self._scoring_stats['average_scores']['skills'] = (
                (prev_avg * (n - 1) + skills_analysis.overall_score) / n
            )
        
        if experience_analysis:
            prev_avg = self._scoring_stats['average_scores']['experience']
            self._scoring_stats['average_scores']['experience'] = (
                (prev_avg * (n - 1) + experience_analysis.overall_score) / n
            )
        
        prev_avg = self._scoring_stats['average_scores']['education']
        self._scoring_stats['average_scores']['education'] = (
            (prev_avg * (n - 1) + education_analysis.overall_score) / n
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata.
        
        Returns:
            ISO format timestamp string
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._scoring_stats.copy()
        
        # Calculate success rates
        total = stats['total_analyses']
        if total > 0:
            stats['success_rate'] = stats['successful_analyses'] / total
            
            # Component failure rates
            component_failures = stats['component_failures'].copy()
            for component in component_failures:
                stats['component_failures'][f'{component}_failure_rate'] = (
                    component_failures[component] / total
                )
        else:
            stats['success_rate'] = 0.0
        
        # Add component scorer stats if available
        if self.skills_scorer and hasattr(self.skills_scorer, 'get_performance_stats'):
            stats['skills_scorer_stats'] = self.skills_scorer.get_performance_stats()
        
        if self.experience_scorer and hasattr(self.experience_scorer, 'get_performance_stats'):
            stats['experience_scorer_stats'] = self.experience_scorer.get_performance_stats()
        
        if hasattr(self.education_scorer, 'get_performance_stats'):
            stats['education_scorer_stats'] = self.education_scorer.get_performance_stats()
        
        # Add configuration info
        stats['configuration'] = {
            'component_weights': self.component_weights,
            'components_available': {
                'skills': self.skills_scorer is not None,
                'experience': self.experience_scorer is not None,
                'education': True
            }
        }
        
        return stats
    
    def update_component_weights(self, new_weights: Dict[str, float]):
        """Update component weights for scoring.
        
        Args:
            new_weights: New weights for components
        """
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"New weights sum to {total_weight}, normalizing to 1.0")
            for key in new_weights:
                new_weights[key] /= total_weight
        
        self.component_weights.update(new_weights)
        logger.info(f"Updated component weights: {self.component_weights}")
    
    def get_component_availability(self) -> Dict[str, bool]:
        """Get availability status of scoring components.
        
        Returns:
            Dictionary indicating which components are available
        """
        return {
            'skills': self.skills_scorer is not None,
            'experience': self.experience_scorer is not None,
            'education': True
        }