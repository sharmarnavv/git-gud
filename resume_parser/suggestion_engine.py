"""
Comprehensive suggestion engine for resume improvement.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from resume_parser.resume_interfaces import ParsedResume
from resume_parser.ats_optimization_system import ATSOptimizationSystem, ATSSuggestion
from resume_parser.gap_analysis import GapAnalysisResult, SkillGap, ExperienceGap, EducationGap
from job_parser.interfaces import ParsedJobDescription
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


class SuggestionCategory(Enum):
    """Categories of suggestions."""
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    FORMATTING = "formatting"
    ATS_OPTIMIZATION = "ats_optimization"
    KEYWORDS = "keywords"


class SuggestionPriority(Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Suggestion:
    """Unified suggestion data structure."""
    id: str
    category: SuggestionCategory
    priority: SuggestionPriority
    title: str
    description: str
    impact_score: float
    feasibility_score: float
    implementation_effort: str
    timeframe: str
    specific_actions: List[str] = field(default_factory=list)
    rationale: str = ""
    examples: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestionEngineResult:
    """Result from suggestion engine with all suggestions and analysis."""
    suggestions: List[Suggestion]
    prioritized_suggestions: List[Suggestion]
    suggestions_by_category: Dict[str, List[Suggestion]]
    quick_wins: List[Suggestion]
    long_term_improvements: List[Suggestion]
    overall_improvement_potential: float
    personalization_applied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class SuggestionEngine:
    """Main orchestrator for comprehensive suggestion generation."""
    
    def __init__(self, ats_system: Optional[ATSOptimizationSystem] = None,
                 user_preferences: Optional[Dict[str, Any]] = None):
        """Initialize the suggestion engine."""
        self.ats_system = ats_system or ATSOptimizationSystem()
        self.user_preferences = user_preferences or {}
        self.ranking_weights = {
            'impact_score': 0.4, 'feasibility_score': 0.3,
            'priority': 0.2, 'user_preference': 0.1
        }
        self._suggestion_counter = 0
        logger.info("Suggestion engine initialized")
    
    def generate_suggestions(self, resume: ParsedResume, job_description: ParsedJobDescription,
                           gap_analysis: Optional[GapAnalysisResult] = None,
                           similarity_score: Optional[float] = None) -> SuggestionEngineResult:
        """Generate comprehensive suggestions for resume improvement."""
        try:
            logger.info("Generating comprehensive suggestions")
            all_suggestions = []
            
            if gap_analysis and gap_analysis.skills_gaps:
                all_suggestions.extend(self._generate_skills_suggestions(
                    gap_analysis.skills_gaps, resume, job_description))
            
            if gap_analysis and gap_analysis.experience_gap:
                all_suggestions.extend(self._generate_experience_suggestions(
                    gap_analysis.experience_gap, resume, job_description))
            
            if gap_analysis and gap_analysis.education_gap:
                all_suggestions.extend(self._generate_education_suggestions(
                    gap_analysis.education_gap, resume, job_description))
            
            all_suggestions.extend(self._generate_ats_suggestions(resume, job_description))
            all_suggestions.extend(self._generate_formatting_suggestions(resume))
            
            ranked = self._rank_suggestions(all_suggestions)
            personalized = self._apply_personalization(ranked)
            by_category = self._categorize_suggestions(personalized)
            quick_wins = self._identify_quick_wins(personalized)
            long_term = self._identify_long_term_improvements(personalized)
            improvement = self._calculate_improvement_potential(personalized, similarity_score)
            
            result = SuggestionEngineResult(
                suggestions=personalized, prioritized_suggestions=personalized[:10],
                suggestions_by_category=by_category, quick_wins=quick_wins,
                long_term_improvements=long_term, overall_improvement_potential=improvement,
                personalization_applied=bool(self.user_preferences),
                metadata={'total_suggestions': len(personalized),
                         'current_similarity_score': similarity_score,
                         'categories_covered': list(by_category.keys())}
            )
            
            logger.info(f"Generated {len(personalized)} suggestions")
            return result
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return SuggestionEngineResult(
                suggestions=[], prioritized_suggestions=[], suggestions_by_category={},
                quick_wins=[], long_term_improvements=[], overall_improvement_potential=0.0,
                personalization_applied=False, metadata={'error': str(e)})
    
    def _generate_skills_suggestions(self, skills_gaps: List[SkillGap], resume: ParsedResume,
                                    job: ParsedJobDescription) -> List[Suggestion]:
        """Generate suggestions for skill gaps."""
        suggestions = []
        try:
            high_priority = [g for g in skills_gaps if g.priority == 'high']
            for gap in high_priority[:5]:
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.SKILLS,
                    priority=self._map_priority(gap.priority),
                    title=f"Add {gap.skill} to your skillset",
                    description=f"The job requires {gap.skill}, missing from your resume.",
                    impact_score=gap.impact_score,
                    feasibility_score=self._calculate_skill_feasibility(gap),
                    implementation_effort=self._determine_skill_effort(gap),
                    timeframe=self._estimate_skill_timeframe(gap),
                    specific_actions=[f"Learn {gap.skill} through online courses",
                                    f"Practice {gap.skill} with projects",
                                    f"Add {gap.skill} to resume skills section"],
                    rationale=f"Skill in {gap.category} category with high importance.",
                    examples=self._generate_skill_examples(gap, resume),
                    resources=gap.learning_resources, metadata={'gap': gap}
                ))
            
            if skills_gaps:
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.SKILLS,
                    priority=SuggestionPriority.MEDIUM,
                    title="Better highlight your existing skills",
                    description="Ensure relevant skills are prominently displayed.",
                    impact_score=0.6, feasibility_score=0.9,
                    implementation_effort="easy", timeframe="immediate",
                    specific_actions=["Move skills section near top",
                                    "Use specific skill names matching job",
                                    "Mention skills in experience descriptions"],
                    rationale="Proper presentation helps ATS and recruiters."
                ))
            return suggestions
        except Exception as e:
            logger.error(f"Skills suggestion generation failed: {e}")
            return []
    
    def _generate_experience_suggestions(self, experience_gap: ExperienceGap, resume: ParsedResume,
                                       job: ParsedJobDescription) -> List[Suggestion]:
        """Generate suggestions for experience gaps."""
        suggestions = []
        try:
            if experience_gap.shortfall_years > 0:
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.EXPERIENCE,
                    priority=SuggestionPriority.HIGH,
                    title="Emphasize relevant experience and transferable skills",
                    description=f"You have {experience_gap.years_candidate:.1f} years but job requires {experience_gap.years_required:.1f}.",
                    impact_score=0.7, feasibility_score=0.8,
                    implementation_effort="medium", timeframe="immediate",
                    specific_actions=["Highlight all relevant projects including side projects",
                                    "Emphasize transferable skills from other roles",
                                    "Quantify achievements to demonstrate impact"],
                    rationale="Diverse experience sources can compensate for years shortfall.",
                    resources=experience_gap.transferable_skills
                ))
            
            if not experience_gap.industry_match:
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.EXPERIENCE,
                    priority=SuggestionPriority.HIGH,
                    title="Bridge industry transition with transferable skills",
                    description="Your experience is in a different industry.",
                    impact_score=0.8, feasibility_score=0.7,
                    implementation_effort="medium", timeframe="immediate",
                    specific_actions=["Identify and emphasize transferable skills",
                                    "Research target industry and use relevant terminology",
                                    "Highlight any related projects or experience"],
                    rationale="Transferable skills help overcome industry transition concerns.",
                    resources=experience_gap.transferable_skills
                ))
            return suggestions
        except Exception as e:
            logger.error(f"Experience suggestion generation failed: {e}")
            return []
    
    def _generate_education_suggestions(self, education_gap: EducationGap, resume: ParsedResume,
                                       job: ParsedJobDescription) -> List[Suggestion]:
        """Generate suggestions for education gaps."""
        suggestions = []
        try:
            if education_gap.degree_gap and education_gap.degree_gap != "none":
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.EDUCATION,
                    priority=SuggestionPriority.MEDIUM,
                    title="Address education requirements",
                    description=f"Job requires {education_gap.degree_required} but you have {education_gap.degree_candidate}.",
                    impact_score=0.6, feasibility_score=0.5,
                    implementation_effort="hard", timeframe="6-24 months",
                    specific_actions=["Consider pursuing relevant degree or certification",
                                    "Highlight equivalent professional experience",
                                    "Emphasize relevant coursework and training"],
                    rationale="Education requirements can sometimes be offset by experience.",
                    resources=education_gap.alternative_paths
                ))
            
            if education_gap.certification_gaps:
                for cert in education_gap.certification_gaps[:3]:
                    suggestions.append(Suggestion(
                        id=self._generate_suggestion_id(), category=SuggestionCategory.EDUCATION,
                        priority=SuggestionPriority.MEDIUM,
                        title=f"Obtain {cert} certification",
                        description=f"This certification is mentioned in job requirements.",
                        impact_score=0.7, feasibility_score=0.7,
                        implementation_effort="medium", timeframe="2-6 months",
                        specific_actions=[f"Research {cert} certification requirements",
                                        f"Enroll in {cert} preparation course",
                                        f"Schedule and complete {cert} exam"],
                        rationale="Certifications demonstrate commitment and expertise."
                    ))
            return suggestions
        except Exception as e:
            logger.error(f"Education suggestion generation failed: {e}")
            return []
    
    def _generate_ats_suggestions(self, resume: ParsedResume,
                                 job: ParsedJobDescription) -> List[Suggestion]:
        """Generate ATS optimization suggestions."""
        suggestions = []
        try:
            ats_suggestions = self.ats_system.generate_ats_suggestions(resume, job)
            for ats_sugg in ats_suggestions[:5]:
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(), category=SuggestionCategory.ATS_OPTIMIZATION,
                    priority=self._map_priority(ats_sugg.priority), title=ats_sugg.title,
                    description=ats_sugg.description, impact_score=ats_sugg.impact_score,
                    feasibility_score=self._map_effort_to_feasibility(ats_sugg.implementation_effort),
                    implementation_effort=ats_sugg.implementation_effort, timeframe="immediate",
                    specific_actions=ats_sugg.specific_actions,
                    rationale="ATS optimization improves resume parsing and visibility.",
                    examples=ats_sugg.integration_examples, metadata={'ats_suggestion': ats_sugg}
                ))
            return suggestions
        except Exception as e:
            logger.error(f"ATS suggestion generation failed: {e}")
            return []
    
    def _generate_formatting_suggestions(self, resume: ParsedResume) -> List[Suggestion]:
        """Generate formatting suggestions."""
        suggestions = []
        try:
            suggestions.append(Suggestion(
                id=self._generate_suggestion_id(), category=SuggestionCategory.FORMATTING,
                priority=SuggestionPriority.MEDIUM,
                title="Optimize resume structure and formatting",
                description="Ensure resume has clear structure and professional formatting.",
                impact_score=0.5, feasibility_score=0.9,
                implementation_effort="easy", timeframe="immediate",
                specific_actions=["Use consistent formatting throughout",
                                "Ensure clear section headers",
                                "Use bullet points for readability",
                                "Keep resume to 1-2 pages"],
                rationale="Professional formatting improves readability and first impression."
            ))
            return suggestions
        except Exception as e:
            logger.error(f"Formatting suggestion generation failed: {e}")
            return []
    
    def _rank_suggestions(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Rank suggestions by impact and feasibility."""
        try:
            priority_scores = {SuggestionPriority.CRITICAL: 1.0, SuggestionPriority.HIGH: 0.8,
                             SuggestionPriority.MEDIUM: 0.5, SuggestionPriority.LOW: 0.2}
            
            def ranking_score(sugg: Suggestion) -> float:
                return (sugg.impact_score * self.ranking_weights['impact_score'] +
                       sugg.feasibility_score * self.ranking_weights['feasibility_score'] +
                       priority_scores.get(sugg.priority, 0.5) * self.ranking_weights['priority'])
            
            return sorted(suggestions, key=ranking_score, reverse=True)
        except Exception as e:
            logger.error(f"Suggestion ranking failed: {e}")
            return suggestions
    
    def _apply_personalization(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Apply user preferences to filter and adjust suggestions."""
        if not self.user_preferences:
            return suggestions
        
        try:
            personalized = []
            focus_areas = self.user_preferences.get('focus_areas', [])
            exclude_categories = self.user_preferences.get('exclude_categories', [])
            max_timeframe = self.user_preferences.get('max_timeframe', None)
            
            for sugg in suggestions:
                if sugg.category.value in exclude_categories:
                    continue
                if max_timeframe and self._timeframe_exceeds(sugg.timeframe, max_timeframe):
                    continue
                if focus_areas and sugg.category.value in focus_areas:
                    sugg.impact_score = min(1.0, sugg.impact_score * 1.2)
                personalized.append(sugg)
            
            return personalized
        except Exception as e:
            logger.error(f"Personalization failed: {e}")
            return suggestions
    
    def _categorize_suggestions(self, suggestions: List[Suggestion]) -> Dict[str, List[Suggestion]]:
        """Group suggestions by category."""
        categorized = {}
        for sugg in suggestions:
            category = sugg.category.value
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(sugg)
        return categorized
    
    def _identify_quick_wins(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Identify easy-to-implement high-impact suggestions."""
        return [s for s in suggestions
                if s.implementation_effort == "easy" and s.impact_score >= 0.5][:5]
    
    def _identify_long_term_improvements(self, suggestions: List[Suggestion]) -> List[Suggestion]:
        """Identify longer-term improvement suggestions."""
        return [s for s in suggestions
                if s.implementation_effort in ["medium", "hard"] and s.impact_score >= 0.6][:5]
    
    def _calculate_improvement_potential(self, suggestions: List[Suggestion],
                                       current_score: Optional[float]) -> float:
        """Calculate potential improvement from implementing suggestions."""
        try:
            if not suggestions:
                return 0.0
            total_potential = sum(s.impact_score * s.feasibility_score for s in suggestions[:10])
            normalized_potential = min(0.4, total_potential * 0.05)
            return round(normalized_potential, 2)
        except Exception as e:
            logger.error(f"Improvement potential calculation failed: {e}")
            return 0.0
    
    def _generate_suggestion_id(self) -> str:
        """Generate unique suggestion ID."""
        self._suggestion_counter += 1
        return f"sugg_{self._suggestion_counter:04d}"
    
    def _map_priority(self, priority_str: str) -> SuggestionPriority:
        """Map string priority to SuggestionPriority enum."""
        mapping = {'critical': SuggestionPriority.CRITICAL, 'high': SuggestionPriority.HIGH,
                  'medium': SuggestionPriority.MEDIUM, 'low': SuggestionPriority.LOW}
        return mapping.get(priority_str.lower(), SuggestionPriority.MEDIUM)
    
    def _calculate_skill_feasibility(self, gap: SkillGap) -> float:
        """Calculate feasibility of acquiring a skill."""
        if gap.alternatives:
            return 0.8
        if gap.category in ['soft_skills', 'tools']:
            return 0.9
        return 0.6
    
    def _determine_skill_effort(self, gap: SkillGap) -> str:
        """Determine effort required to acquire skill."""
        if gap.category in ['soft_skills', 'tools']:
            return "easy"
        if gap.category in ['programming_languages', 'frameworks']:
            return "medium"
        return "hard"
    
    def _estimate_skill_timeframe(self, gap: SkillGap) -> str:
        """Estimate timeframe to acquire skill."""
        effort = self._determine_skill_effort(gap)
        if effort == "easy":
            return "1-2 weeks"
        elif effort == "medium":
            return "1-3 months"
        return "3-6 months"
    
    def _generate_skill_examples(self, gap: SkillGap, resume: ParsedResume) -> List[str]:
        """Generate examples of how to integrate skill."""
        examples = []
        if resume.experience:
            exp = resume.experience[0]
            examples.append(f"In {exp.job_title} role: 'Utilized {gap.skill} to improve system performance'")
        examples.append(f"Skills section: Add '{gap.skill}' with proficiency level")
        return examples
    
    def _map_effort_to_feasibility(self, effort: str) -> float:
        """Map implementation effort to feasibility score."""
        mapping = {'easy': 0.9, 'medium': 0.7, 'hard': 0.4}
        return mapping.get(effort.lower(), 0.7)
    
    def _timeframe_exceeds(self, timeframe: str, max_timeframe: str) -> bool:
        """Check if timeframe exceeds maximum."""
        timeframe_order = ['immediate', '1-2 weeks', '1-3 months', '2-6 months',
                          '3-6 months', '6-12 months', '6-24 months', '12+ months']
        try:
            tf_idx = next((i for i, tf in enumerate(timeframe_order) if tf in timeframe.lower()), 0)
            max_idx = next((i for i, tf in enumerate(timeframe_order) if tf in max_timeframe.lower()), len(timeframe_order))
            return tf_idx > max_idx
        except:
            return False
