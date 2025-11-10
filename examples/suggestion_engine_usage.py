"""
Example usage of the SuggestionEngine for generating resume improvement suggestions.

This example demonstrates how to use the SuggestionEngine to generate comprehensive,
prioritized suggestions for improving a resume based on job requirements.
"""

from resume_parser import SuggestionEngine, ParsedResume, ContactInfo, WorkExperience, Education
from resume_parser.gap_analysis import GapAnalysisResult, SkillGap, ExperienceGap, EducationGap
from job_parser.interfaces import ParsedJobDescription


def main():
    """Demonstrate SuggestionEngine usage."""
    
    # Create sample resume data
    resume = ParsedResume(
        contact_info=ContactInfo(
            name="Jane Smith",
            email="jane.smith@email.com",
            phone="555-0123",
            address="San Francisco, CA",
            linkedin="linkedin.com/in/janesmith",
            github="github.com/janesmith"
        ),
        skills=["Python", "JavaScript", "SQL", "Git"],
        experience=[
            WorkExperience(
                job_title="Software Developer",
                company="Tech Startup Inc",
                start_date="2021-06",
                end_date="2024-01",
                description="Developed web applications using Python and JavaScript",
                duration_months=31
            )
        ],
        education=[
            Education(
                degree="Bachelor of Science",
                major="Computer Science",
                institution="State University",
                graduation_date="2021-05"
            )
        ],
        certifications=[],
        metadata={}
    )
    
    # Create sample job description
    job = ParsedJobDescription(
        skills_required=["Python", "JavaScript", "React", "AWS", "Docker", "Kubernetes"],
        experience_level="senior",
        tools_mentioned=["Git", "Jenkins", "Terraform"],
        confidence_scores={
            "Python": 0.95, "JavaScript": 0.90, "React": 0.85,
            "AWS": 0.90, "Docker": 0.80, "Kubernetes": 0.75
        },
        categories={
            "programming": ["Python", "JavaScript"],
            "frameworks": ["React"],
            "cloud": ["AWS"],
            "devops": ["Docker", "Kubernetes"]
        },
        metadata={
            "job_title": "Senior Software Engineer",
            "description": "Looking for a Senior Software Engineer with 5+ years experience",
            "years_required": 5
        }
    )
    
    # Create gap analysis (normally this would come from GapAnalyzer)
    gap_analysis = GapAnalysisResult(
        skills_gaps=[
            SkillGap(
                skill="React",
                category="frameworks",
                priority="high",
                confidence=0.9,
                alternatives=["Vue.js"],
                learning_resources=["React Official Docs", "React Course on Udemy"],
                impact_score=0.85
            ),
            SkillGap(
                skill="AWS",
                category="cloud_platforms",
                priority="high",
                confidence=0.9,
                alternatives=[],
                learning_resources=["AWS Certified Developer Course"],
                impact_score=0.80
            ),
            SkillGap(
                skill="Docker",
                category="devops",
                priority="medium",
                confidence=0.85,
                alternatives=[],
                learning_resources=["Docker Documentation"],
                impact_score=0.70
            )
        ],
        experience_gap=ExperienceGap(
            years_required=5.0,
            years_candidate=2.6,
            shortfall_years=2.4,
            relevance_score=0.75,
            industry_match=True,
            seniority_gap="mid_to_senior",
            transferable_skills=["Python", "JavaScript", "Problem Solving"],
            progression_analysis={"growth_rate": "steady"}
        ),
        education_gap=EducationGap(
            degree_required="Bachelor's",
            degree_candidate="Bachelor's",
            degree_gap="none",
            field_match=True,
            certification_gaps=["AWS Certified Developer"],
            alternative_paths=[],
            roi_analysis={}
        ),
        overall_gap_score=0.35,
        improvement_potential=0.30,
        priority_recommendations=[],
        metadata={}
    )
    
    # Initialize SuggestionEngine with optional user preferences
    user_preferences = {
        'focus_areas': ['skills', 'experience'],  # Focus on these categories
        'max_timeframe': '6 months',  # Only show suggestions achievable in 6 months
        'exclude_categories': []  # Don't exclude any categories
    }
    
    engine = SuggestionEngine(user_preferences=user_preferences)
    
    # Generate suggestions
    print("Generating comprehensive resume improvement suggestions...\n")
    result = engine.generate_suggestions(
        resume=resume,
        job_description=job,
        gap_analysis=gap_analysis,
        similarity_score=0.62  # Current match score
    )
    
    # Display results
    print(f"{'='*80}")
    print(f"SUGGESTION ENGINE RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total Suggestions Generated: {len(result.suggestions)}")
    print(f"Overall Improvement Potential: +{result.overall_improvement_potential:.0%}")
    print(f"Current Match Score: 62%")
    print(f"Projected Match Score: {62 + (result.overall_improvement_potential * 100):.0f}%")
    print(f"Personalization Applied: {result.personalization_applied}\n")
    
    # Show suggestions by category
    print(f"{'='*80}")
    print(f"SUGGESTIONS BY CATEGORY")
    print(f"{'='*80}\n")
    
    for category, suggestions in result.suggestions_by_category.items():
        print(f"{category.upper()}: {len(suggestions)} suggestions")
    
    # Show quick wins
    print(f"\n{'='*80}")
    print(f"QUICK WINS (Easy & High Impact)")
    print(f"{'='*80}\n")
    
    for i, sugg in enumerate(result.quick_wins, 1):
        print(f"{i}. {sugg.title}")
        print(f"   Impact: {sugg.impact_score:.0%} | Effort: {sugg.implementation_effort} | "
              f"Timeframe: {sugg.timeframe}")
        print(f"   {sugg.description}\n")
    
    # Show top prioritized suggestions
    print(f"{'='*80}")
    print(f"TOP PRIORITY SUGGESTIONS")
    print(f"{'='*80}\n")
    
    for i, sugg in enumerate(result.prioritized_suggestions[:5], 1):
        print(f"{i}. [{sugg.priority.value.upper()}] {sugg.title}")
        print(f"   Category: {sugg.category.value}")
        print(f"   Impact: {sugg.impact_score:.0%} | Feasibility: {sugg.feasibility_score:.0%}")
        print(f"   Effort: {sugg.implementation_effort} | Timeframe: {sugg.timeframe}")
        print(f"   \n   Description: {sugg.description}")
        
        if sugg.specific_actions:
            print(f"   \n   Specific Actions:")
            for action in sugg.specific_actions[:3]:
                print(f"   • {action}")
        
        if sugg.rationale:
            print(f"   \n   Why this matters: {sugg.rationale}")
        
        print()
    
    # Show long-term improvements
    if result.long_term_improvements:
        print(f"{'='*80}")
        print(f"LONG-TERM IMPROVEMENTS")
        print(f"{'='*80}\n")
        
        for i, sugg in enumerate(result.long_term_improvements, 1):
            print(f"{i}. {sugg.title}")
            print(f"   Impact: {sugg.impact_score:.0%} | Timeframe: {sugg.timeframe}")
            print(f"   {sugg.description}\n")
    
    print(f"{'='*80}")
    print("Suggestion generation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
