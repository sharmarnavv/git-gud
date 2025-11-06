"""
Simple test for the sub-scoring system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription
from resume_parser.education_scoring import EducationRequirementsScorer
from resume_parser.comprehensive_sub_scoring import ComprehensiveSubScoringEngine

def create_test_data():
    """Create simple test data."""
    # Simple resume
    resume = ParsedResume(
        contact_info=ContactInfo(name="Test User"),
        skills=["Python", "JavaScript", "React", "Machine Learning"],
        experience=[
            WorkExperience(
                job_title="Software Engineer",
                company="Tech Corp",
                start_date="2020",
                end_date="2023",
                description="Developed web applications",
                duration_months=36
            )
        ],
        education=[
            Education(
                degree="Bachelor of Science",
                major="Computer Science",
                institution="State University",
                graduation_date="2020"
            )
        ],
        certifications=[
            Certification(
                name="AWS Certified Developer",
                issuer="Amazon",
                issue_date="2022"
            )
        ]
    )
    
    # Simple job description
    job_description = ParsedJobDescription(
        skills_required=["Python", "JavaScript", "AWS", "Docker"],
        experience_level="mid-level",
        tools_mentioned=["Git", "Jenkins"],
        confidence_scores={"Python": 0.9, "JavaScript": 0.8},
        categories={
            "technical": ["Python", "JavaScript"],
            "tools": ["AWS", "Docker", "Git", "Jenkins"]
        },
        metadata={}
    )
    
    job_text = """
    Software Engineer Position
    
    Requirements:
    - Bachelor's degree in Computer Science
    - 3+ years of experience
    - Python and JavaScript skills
    - AWS experience preferred
    """
    
    return resume, job_description, job_text

def test_education_scoring():
    """Test education scoring."""
    print("Testing Education Scoring...")
    
    try:
        scorer = EducationRequirementsScorer()
        resume, job_description, job_text = create_test_data()
        
        result = scorer.calculate_education_similarity(resume, job_description, job_text)
        
        print(f"✅ Education Score: {result.overall_score:.1f}%")
        print(f"   Degree Match: {result.degree_match_score:.1f}%")
        print(f"   Field Match: {result.field_match_score:.1f}%")
        print(f"   Certification Score: {result.certification_score:.1f}%")
        print(f"   Confidence: {result.confidence_score:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Education scoring failed: {e}")
        return False

def test_comprehensive_scoring():
    """Test comprehensive scoring."""
    print("\nTesting Comprehensive Scoring...")
    
    try:
        scorer = ComprehensiveSubScoringEngine()
        resume, job_description, job_text = create_test_data()
        
        result = scorer.calculate_comprehensive_match(resume, job_description, job_text)
        
        print(f"✅ Overall Score: {result.overall_score:.1f}%")
        print(f"   Confidence: {result.confidence_score:.2f}")
        
        print(f"   Component Scores:")
        for component, score in result.weighted_scores.items():
            if not component.startswith('raw_'):
                raw_score = result.weighted_scores.get(f'raw_{component}', 0)
                print(f"     {component}: {score:.1f}% (raw: {raw_score:.1f}%)")
        
        print(f"   Gap Analysis:")
        if result.gap_analysis['skills_gaps']:
            print(f"     Missing Skills: {result.gap_analysis['skills_gaps'][:3]}")
        
        print(f"   Top Suggestions:")
        for i, suggestion in enumerate(result.improvement_suggestions[:2], 1):
            print(f"     {i}. {suggestion['title']}")
        
        return True
    except Exception as e:
        print(f"❌ Comprehensive scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests."""
    print("🚀 Running Simple Sub-Scoring Tests")
    print("=" * 50)
    
    results = []
    results.append(test_education_scoring())
    results.append(test_comprehensive_scoring())
    
    print("\n" + "=" * 50)
    print("Test Results:")
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)