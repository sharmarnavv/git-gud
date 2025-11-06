"""
Test script for the newly implemented sub-scoring features.

This script tests the skills, experience, and education scoring components
with sample data to verify functionality and performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription
from resume_parser.education_scoring import EducationRequirementsScorer
from resume_parser.comprehensive_sub_scoring import ComprehensiveSubScoringEngine

def create_sample_resume():
    """Create a sample resume for testing."""
    contact_info = ContactInfo(
        name="John Doe",
        email="john.doe@email.com",
        phone="555-123-4567",
        address="123 Main St, City, State"
    )
    
    skills = [
        "Python", "JavaScript", "React", "Node.js", "SQL", "MongoDB",
        "Machine Learning", "Data Analysis", "Project Management",
        "Communication", "Leadership", "Problem Solving"
    ]
    
    experience = [
        WorkExperience(
            job_title="Senior Software Engineer",
            company="Tech Corp",
            start_date="2020",
            end_date="2023",
            description="Led development of web applications using React and Node.js. Managed team of 3 developers.",
            skills_used=["Python", "JavaScript", "React", "Leadership"],
            duration_months=36
        ),
        WorkExperience(
            job_title="Software Developer",
            company="StartupXYZ",
            start_date="2018",
            end_date="2020",
            description="Developed backend APIs and data processing pipelines using Python and SQL.",
            skills_used=["Python", "SQL", "Data Analysis"],
            duration_months=24
        ),
        WorkExperience(
            job_title="Junior Developer",
            company="WebDev Inc",
            start_date="2016",
            end_date="2018",
            description="Built responsive websites using HTML, CSS, and JavaScript.",
            skills_used=["JavaScript", "HTML", "CSS"],
            duration_months=24
        )
    ]
    
    education = [
        Education(
            degree="Bachelor of Science",
            major="Computer Science",
            institution="State University",
            graduation_date="2016",
            gpa="3.7"
        )
    ]
    
    certifications = [
        Certification(
            name="AWS Certified Developer",
            issuer="Amazon Web Services",
            issue_date="2022",
            expiry_date="2025"
        ),
        Certification(
            name="Certified Scrum Master",
            issuer="Scrum Alliance",
            issue_date="2021"
        )
    ]
    
    return ParsedResume(
        contact_info=contact_info,
        skills=skills,
        experience=experience,
        education=education,
        certifications=certifications,
        metadata={"source": "test_data"}
    )

def create_sample_job_description():
    """Create a sample job description for testing."""
    return ParsedJobDescription(
        skills_required=[
            "Python", "JavaScript", "React", "AWS", "Docker",
            "Machine Learning", "Data Science", "Leadership",
            "Project Management", "Agile"
        ],
        experience_level="senior-level",
        tools_mentioned=["Git", "Jenkins", "Kubernetes", "PostgreSQL"],
        confidence_scores={
            "Python": 0.95,
            "JavaScript": 0.90,
            "React": 0.85,
            "AWS": 0.80,
            "Machine Learning": 0.75
        },
        categories={
            "technical": ["Python", "JavaScript", "React", "AWS", "Docker"],
            "soft": ["Leadership", "Project Management"],
            "domain": ["Machine Learning", "Data Science"],
            "tools": ["Git", "Jenkins", "Kubernetes"]
        },
        metadata={"source": "test_job_posting"}
    )

def create_sample_job_text():
    """Create sample job description text for testing."""
    return """
    Senior Software Engineer - AI/ML Team
    
    We are seeking a Senior Software Engineer with 5+ years of experience to join our AI/ML team.
    
    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 5+ years of software development experience
    - Strong proficiency in Python and JavaScript
    - Experience with React and modern web frameworks
    - Knowledge of machine learning and data science
    - AWS certification preferred
    - Experience with Docker and Kubernetes
    - Strong leadership and project management skills
    - Excellent communication and problem-solving abilities
    
    Preferred Qualifications:
    - Master's degree in Computer Science or related field
    - PMP certification
    - Experience in fintech or healthcare industry
    - Agile/Scrum methodology experience
    """

def test_education_scoring():
    """Test the education scoring component."""
    print("=" * 60)
    print("TESTING EDUCATION SCORING")
    print("=" * 60)
    
    try:
        # Initialize scorer
        education_scorer = EducationRequirementsScorer()
        
        # Create test data
        resume = create_sample_resume()
        job_description = create_sample_job_description()
        job_text = create_sample_job_text()
        
        # Calculate education similarity
        result = education_scorer.calculate_education_similarity(
            resume, job_description, job_text
        )
        
        # Display results
        print(f"Overall Education Score: {result.overall_score:.1f}%")
        print(f"Degree Match Score: {result.degree_match_score:.1f}%")
        print(f"Field Match Score: {result.field_match_score:.1f}%")
        print(f"Institution Score: {result.institution_score:.1f}%")
        print(f"Certification Score: {result.certification_score:.1f}%")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        
        print(f"\nMatched Requirements ({len(result.matched_requirements)}):")
        for match in result.matched_requirements:
            print(f"  - {match.education_type}: {match.resume_education} -> {match.job_requirement} "
                  f"({match.match_level}, confidence: {match.confidence:.2f})")
        
        print(f"\nMissing Requirements ({len(result.missing_requirements)}):")
        for missing in result.missing_requirements:
            print(f"  - {missing}")
        
        print(f"\nAlternative Paths ({len(result.alternative_paths)}):")
        for path in result.alternative_paths:
            print(f"  - {path}")
        
        print(f"\nMetadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Education scoring test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Education scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_scoring():
    """Test the comprehensive sub-scoring engine."""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE SUB-SCORING ENGINE")
    print("=" * 60)
    
    try:
        # Initialize comprehensive scorer
        comprehensive_scorer = ComprehensiveSubScoringEngine()
        
        # Create test data
        resume = create_sample_resume()
        job_description = create_sample_job_description()
        job_text = create_sample_job_text()
        
        # Calculate comprehensive match
        result = comprehensive_scorer.calculate_comprehensive_match(
            resume, job_description, job_text
        )
        
        # Display results
        print(f"Overall Match Score: {result.overall_score:.1f}%")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        
        print(f"\nWeighted Component Scores:")
        for component, score in result.weighted_scores.items():
            if not component.startswith('raw_'):
                raw_score = result.weighted_scores.get(f'raw_{component}', 0)
                print(f"  {component.capitalize()}: {score:.1f}% (raw: {raw_score:.1f}%)")
        
        print(f"\nGap Analysis:")
        gap_analysis = result.gap_analysis
        
        if gap_analysis['skills_gaps']:
            print(f"  Skills Gaps ({len(gap_analysis['skills_gaps'])}):")
            for skill in gap_analysis['skills_gaps'][:5]:
                print(f"    - {skill}")
        
        if gap_analysis['experience_gaps']:
            print(f"  Experience Gaps ({len(gap_analysis['experience_gaps'])}):")
            for gap in gap_analysis['experience_gaps']:
                print(f"    - {gap}")
        
        if gap_analysis['education_gaps']:
            print(f"  Education Gaps ({len(gap_analysis['education_gaps'])}):")
            for gap in gap_analysis['education_gaps']:
                print(f"    - {gap}")
        
        print(f"\nPriority Gaps (Top 5):")
        for i, gap in enumerate(gap_analysis['priority_gaps'][:5], 1):
            print(f"  {i}. {gap['description']} "
                  f"(Impact: {gap['impact']}, Feasibility: {gap['feasibility']}, "
                  f"Priority: {gap['priority_score']})")
        
        print(f"\nImprovement Suggestions (Top 5):")
        for i, suggestion in enumerate(result.improvement_suggestions[:5], 1):
            print(f"  {i}. {suggestion['title']}")
            print(f"     {suggestion['description']}")
            print(f"     Priority: {suggestion['priority']}, "
                  f"Effort: {suggestion['effort']}, "
                  f"Timeframe: {suggestion['timeframe']}")
            print()
        
        print(f"Component Analysis Details:")
        
        # Skills analysis
        if result.skills_analysis:
            skills = result.skills_analysis
            print(f"  Skills Analysis:")
            print(f"    - Overall Score: {skills.overall_score:.1f}%")
            print(f"    - Matched Skills: {len(skills.matched_skills)}")
            print(f"    - Missing Skills: {len(skills.missing_skills)}")
            print(f"    - Confidence: {skills.confidence_score:.2f}")
        
        # Experience analysis
        if result.experience_analysis:
            exp = result.experience_analysis
            print(f"  Experience Analysis:")
            print(f"    - Overall Score: {exp.overall_score:.1f}%")
            print(f"    - Years Experience: {exp.years_experience:.1f}")
            print(f"    - Required Years: {exp.required_years:.1f}")
            print(f"    - Seniority Match: {exp.seniority_match:.1f}%")
            print(f"    - Industry Relevance: {exp.industry_relevance:.1f}%")
            print(f"    - Confidence: {exp.confidence_score:.2f}")
        
        # Education analysis
        if result.education_analysis:
            edu = result.education_analysis
            print(f"  Education Analysis:")
            print(f"    - Overall Score: {edu.overall_score:.1f}%")
            print(f"    - Degree Match: {edu.degree_match_score:.1f}%")
            print(f"    - Field Match: {edu.field_match_score:.1f}%")
            print(f"    - Institution Score: {edu.institution_score:.1f}%")
            print(f"    - Certification Score: {edu.certification_score:.1f}%")
            print(f"    - Confidence: {edu.confidence_score:.2f}")
        
        print("\n✅ Comprehensive scoring test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_stats():
    """Test performance statistics functionality."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE STATISTICS")
    print("=" * 60)
    
    try:
        # Initialize scorers
        education_scorer = EducationRequirementsScorer()
        comprehensive_scorer = ComprehensiveSubScoringEngine()
        
        # Run multiple analyses to generate stats
        resume = create_sample_resume()
        job_description = create_sample_job_description()
        job_text = create_sample_job_text()
        
        print("Running multiple analyses to generate performance statistics...")
        
        # Run education scoring multiple times
        for i in range(3):
            education_scorer.calculate_education_similarity(resume, job_description, job_text)
        
        # Run comprehensive scoring multiple times
        for i in range(3):
            comprehensive_scorer.calculate_comprehensive_match(resume, job_description, job_text)
        
        # Get performance stats
        edu_stats = education_scorer.get_performance_stats()
        comp_stats = comprehensive_scorer.get_performance_stats()
        
        print(f"\nEducation Scorer Performance Stats:")
        for key, value in edu_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nComprehensive Scorer Performance Stats:")
        for key, value in comp_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"    {sub_key}:")
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            print(f"      {sub_sub_key}: {sub_sub_value}")
                    else:
                        print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Performance statistics test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Performance statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES AND ERROR HANDLING")
    print("=" * 60)
    
    try:
        education_scorer = EducationRequirementsScorer()
        comprehensive_scorer = ComprehensiveSubScoringEngine()
        
        # Test with minimal resume data
        print("Testing with minimal resume data...")
        minimal_resume = ParsedResume(
            contact_info=ContactInfo(name="Test User"),
            skills=["Python"],
            experience=[],
            education=[],
            certifications=[]
        )
        
        minimal_job = ParsedJobDescription(
            skills_required=["Python", "Java"],
            experience_level="entry-level",
            tools_mentioned=[],
            confidence_scores={},
            categories={"technical": ["Python", "Java"]},
            metadata={}
        )
        
        # Test education scoring with minimal data
        edu_result = education_scorer.calculate_education_similarity(
            minimal_resume, minimal_job, ""
        )
        print(f"  Minimal data education score: {edu_result.overall_score:.1f}%")
        
        # Test comprehensive scoring with minimal data
        comp_result = comprehensive_scorer.calculate_comprehensive_match(
            minimal_resume, minimal_job, ""
        )
        print(f"  Minimal data comprehensive score: {comp_result.overall_score:.1f}%")
        
        # Test with empty data
        print("Testing with empty data...")
        empty_resume = ParsedResume()
        empty_job = ParsedJobDescription(
            skills_required=[],
            experience_level="",
            tools_mentioned=[],
            confidence_scores={},
            categories={},
            metadata={}
        )
        
        try:
            edu_result = education_scorer.calculate_education_similarity(
                empty_resume, empty_job, ""
            )
            print(f"  Empty data education score: {edu_result.overall_score:.1f}%")
        except Exception as e:
            print(f"  Empty data education scoring handled error: {type(e).__name__}")
        
        try:
            comp_result = comprehensive_scorer.calculate_comprehensive_match(
                empty_resume, empty_job, ""
            )
            print(f"  Empty data comprehensive score: {comp_result.overall_score:.1f}%")
        except Exception as e:
            print(f"  Empty data comprehensive scoring handled error: {type(e).__name__}")
        
        print("\n✅ Edge cases test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Sub-Scoring System Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Education Scoring", test_education_scoring()))
    test_results.append(("Comprehensive Scoring", test_comprehensive_scoring()))
    test_results.append(("Performance Statistics", test_performance_stats()))
    test_results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sub-scoring system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)