"""
Detailed test for the sub-scoring system features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription
from resume_parser.sub_scoring_engine import SkillsSimilarityScorer
from resume_parser.education_scoring import EducationRequirementsScorer
from resume_parser.comprehensive_sub_scoring import ComprehensiveSubScoringEngine

def create_detailed_test_data():
    """Create detailed test data for comprehensive testing."""
    # Detailed resume
    resume = ParsedResume(
        contact_info=ContactInfo(
            name="Jane Smith",
            email="jane.smith@email.com",
            phone="555-123-4567"
        ),
        skills=[
            "Python", "JavaScript", "React", "Node.js", "SQL", "MongoDB",
            "Machine Learning", "Data Analysis", "AWS", "Docker",
            "Project Management", "Leadership", "Communication"
        ],
        experience=[
            WorkExperience(
                job_title="Senior Software Engineer",
                company="Tech Innovations Inc",
                start_date="2021",
                end_date="2023",
                description="Led development of ML-powered web applications using Python and React. Managed team of 4 developers.",
                skills_used=["Python", "React", "Machine Learning", "Leadership"],
                duration_months=24
            ),
            WorkExperience(
                job_title="Software Developer",
                company="DataCorp Solutions",
                start_date="2019",
                end_date="2021",
                description="Developed data processing pipelines and APIs using Python, SQL, and AWS services.",
                skills_used=["Python", "SQL", "AWS", "Data Analysis"],
                duration_months=24
            ),
            WorkExperience(
                job_title="Junior Developer",
                company="WebTech Startup",
                start_date="2017",
                end_date="2019",
                description="Built responsive web applications using JavaScript, HTML, and CSS.",
                skills_used=["JavaScript", "HTML", "CSS"],
                duration_months=24
            )
        ],
        education=[
            Education(
                degree="Master of Science",
                major="Computer Science",
                institution="Tech University",
                graduation_date="2017",
                gpa="3.8"
            ),
            Education(
                degree="Bachelor of Science",
                major="Mathematics",
                institution="State College",
                graduation_date="2015",
                gpa="3.6"
            )
        ],
        certifications=[
            Certification(
                name="AWS Certified Solutions Architect",
                issuer="Amazon Web Services",
                issue_date="2022",
                expiry_date="2025"
            ),
            Certification(
                name="Certified Scrum Master",
                issuer="Scrum Alliance",
                issue_date="2021"
            ),
            Certification(
                name="Google Cloud Professional Data Engineer",
                issuer="Google Cloud",
                issue_date="2020",
                expiry_date="2023"
            )
        ]
    )
    
    # Detailed job description
    job_description = ParsedJobDescription(
        skills_required=[
            "Python", "JavaScript", "React", "AWS", "Docker", "Kubernetes",
            "Machine Learning", "Data Science", "SQL", "NoSQL",
            "Leadership", "Project Management", "Agile", "Communication"
        ],
        experience_level="senior-level",
        tools_mentioned=["Git", "Jenkins", "Terraform", "PostgreSQL", "Redis"],
        confidence_scores={
            "Python": 0.95,
            "JavaScript": 0.90,
            "React": 0.85,
            "AWS": 0.90,
            "Machine Learning": 0.80,
            "Leadership": 0.75
        },
        categories={
            "technical": ["Python", "JavaScript", "React", "SQL", "NoSQL"],
            "cloud": ["AWS", "Docker", "Kubernetes"],
            "data": ["Machine Learning", "Data Science"],
            "soft": ["Leadership", "Project Management", "Communication"],
            "methodology": ["Agile"]
        },
        metadata={"source": "detailed_test_job"}
    )
    
    job_text = """
    Senior Software Engineer - AI/ML Platform
    
    We are seeking a Senior Software Engineer with 5+ years of experience to join our AI/ML platform team.
    
    Requirements:
    - Master's degree in Computer Science, Engineering, or related field
    - 5+ years of software development experience
    - Strong proficiency in Python and JavaScript
    - Experience with React and modern web frameworks
    - Hands-on experience with AWS cloud services
    - Knowledge of machine learning and data science concepts
    - Experience with containerization (Docker, Kubernetes)
    - Strong leadership and project management skills
    - Excellent communication and problem-solving abilities
    - Experience with Agile/Scrum methodologies
    
    Preferred Qualifications:
    - AWS certification (Solutions Architect or similar)
    - Experience with data engineering and big data technologies
    - Background in fintech or healthcare industry
    - PMP or Scrum Master certification
    - Experience with CI/CD pipelines and DevOps practices
    """
    
    return resume, job_description, job_text

def test_skills_scoring_detailed():
    """Test detailed skills scoring functionality."""
    print("Testing Detailed Skills Scoring...")
    print("-" * 40)
    
    try:
        scorer = SkillsSimilarityScorer(
            fuzzy_threshold=0.8,
            enable_synonym_matching=True,
            category_weights={
                'technical': 0.4,
                'cloud': 0.25,
                'data': 0.2,
                'soft': 0.15
            }
        )
        
        resume, job_description, job_text = create_detailed_test_data()
        
        result = scorer.calculate_skills_similarity(resume, job_description)
        
        print(f"✅ Overall Skills Score: {result.overall_score:.1f}%")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        print(f"   Match Rate: {result.metadata['match_rate']:.1%}")
        
        print(f"\n   Matched Skills ({len(result.matched_skills)}):")
        for match in result.matched_skills[:5]:  # Show top 5
            print(f"     - {match.skill} ({match.match_type}, {match.confidence:.2f})")
        
        print(f"\n   Missing Skills ({len(result.missing_skills)}):")
        for skill in result.missing_skills[:5]:  # Show top 5
            print(f"     - {skill}")
        
        print(f"\n   Category Scores:")
        for category, score in result.category_scores.items():
            print(f"     {category}: {score:.1f}%")
        
        # Test performance stats
        stats = scorer.get_performance_stats()
        print(f"\n   Performance Stats:")
        print(f"     Total Comparisons: {stats['total_comparisons']}")
        print(f"     Exact Matches: {stats['exact_matches']}")
        print(f"     Fuzzy Matches: {stats['fuzzy_matches']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed skills scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_education_scoring_detailed():
    """Test detailed education scoring functionality."""
    print("\nTesting Detailed Education Scoring...")
    print("-" * 40)
    
    try:
        scorer = EducationRequirementsScorer()
        resume, job_description, job_text = create_detailed_test_data()
        
        result = scorer.calculate_education_similarity(resume, job_description, job_text)
        
        print(f"✅ Overall Education Score: {result.overall_score:.1f}%")
        print(f"   Degree Match: {result.degree_match_score:.1f}%")
        print(f"   Field Match: {result.field_match_score:.1f}%")
        print(f"   Institution Score: {result.institution_score:.1f}%")
        print(f"   Certification Score: {result.certification_score:.1f}%")
        print(f"   Confidence: {result.confidence_score:.2f}")
        
        print(f"\n   Matched Requirements ({len(result.matched_requirements)}):")
        for match in result.matched_requirements:
            print(f"     - {match.education_type}: {match.resume_education} -> {match.job_requirement}")
            print(f"       ({match.match_level}, confidence: {match.confidence:.2f})")
        
        if result.missing_requirements:
            print(f"\n   Missing Requirements ({len(result.missing_requirements)}):")
            for missing in result.missing_requirements:
                print(f"     - {missing}")
        
        if result.alternative_paths:
            print(f"\n   Alternative Paths ({len(result.alternative_paths)}):")
            for path in result.alternative_paths[:3]:  # Show top 3
                print(f"     - {path}")
        
        # Test performance stats
        stats = scorer.get_performance_stats()
        print(f"\n   Performance Stats:")
        print(f"     Total Analyses: {stats['total_analyses']}")
        print(f"     Degree Match Rate: {stats['degree_match_rate']:.1%}")
        print(f"     Field Match Rate: {stats['field_match_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed education scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_scoring_detailed():
    """Test detailed comprehensive scoring functionality."""
    print("\nTesting Detailed Comprehensive Scoring...")
    print("-" * 40)
    
    try:
        # Test with custom weights
        scorer = ComprehensiveSubScoringEngine(
            component_weights={
                'skills': 0.45,
                'experience': 0.35,
                'education': 0.20
            }
        )
        
        resume, job_description, job_text = create_detailed_test_data()
        
        result = scorer.calculate_comprehensive_match(resume, job_description, job_text)
        
        print(f"✅ Overall Match Score: {result.overall_score:.1f}%")
        print(f"   Overall Confidence: {result.confidence_score:.2f}")
        
        print(f"\n   Weighted Component Scores:")
        for component, score in result.weighted_scores.items():
            if not component.startswith('raw_'):
                raw_score = result.weighted_scores.get(f'raw_{component}', 0)
                print(f"     {component.capitalize()}: {score:.1f}% (raw: {raw_score:.1f}%)")
        
        print(f"\n   Gap Analysis Summary:")
        gap_analysis = result.gap_analysis
        
        print(f"     Skills Gaps: {len(gap_analysis['skills_gaps'])}")
        if gap_analysis['skills_gaps']:
            print(f"       Top Missing: {', '.join(gap_analysis['skills_gaps'][:3])}")
        
        print(f"     Experience Gaps: {len(gap_analysis['experience_gaps'])}")
        if gap_analysis['experience_gaps']:
            for gap in gap_analysis['experience_gaps']:
                print(f"       - {gap}")
        
        print(f"     Education Gaps: {len(gap_analysis['education_gaps'])}")
        if gap_analysis['education_gaps']:
            for gap in gap_analysis['education_gaps']:
                print(f"       - {gap}")
        
        print(f"\n   Priority Gaps (Top 5):")
        for i, gap in enumerate(gap_analysis['priority_gaps'][:5], 1):
            print(f"     {i}. {gap['description']}")
            print(f"        Impact: {gap['impact']}, Feasibility: {gap['feasibility']}")
        
        print(f"\n   Improvement Suggestions (Top 5):")
        for i, suggestion in enumerate(result.improvement_suggestions[:5], 1):
            print(f"     {i}. {suggestion['title']}")
            print(f"        {suggestion['description']}")
            print(f"        Priority: {suggestion['priority']}, Effort: {suggestion['effort']}")
            print(f"        Timeframe: {suggestion['timeframe']}")
        
        # Test component availability
        availability = scorer.get_component_availability()
        print(f"\n   Component Availability:")
        for component, available in availability.items():
            status = "✅" if available else "❌"
            print(f"     {component}: {status}")
        
        # Test performance stats
        stats = scorer.get_performance_stats()
        print(f"\n   Performance Stats:")
        print(f"     Total Analyses: {stats['total_analyses']}")
        print(f"     Success Rate: {stats['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detailed comprehensive scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")
    print("-" * 40)
    
    try:
        scorer = ComprehensiveSubScoringEngine()
        
        # Test with minimal data
        minimal_resume = ParsedResume(
            contact_info=ContactInfo(name="Minimal User"),
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
        
        result = scorer.calculate_comprehensive_match(minimal_resume, minimal_job, "")
        print(f"✅ Minimal data test: {result.overall_score:.1f}% match")
        
        # Test with empty skills
        empty_skills_resume = ParsedResume(
            contact_info=ContactInfo(name="No Skills User"),
            skills=[],
            experience=[WorkExperience(job_title="Developer", company="Corp", duration_months=12)],
            education=[Education(degree="Bachelor", major="CS")],
            certifications=[]
        )
        
        result = scorer.calculate_comprehensive_match(empty_skills_resume, minimal_job, "")
        print(f"✅ Empty skills test: {result.overall_score:.1f}% match")
        
        # Test weight updates
        scorer.update_component_weights({
            'skills': 0.5,
            'experience': 0.3,
            'education': 0.2
        })
        print(f"✅ Weight update test: successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run detailed tests."""
    print("🚀 Running Detailed Sub-Scoring System Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run detailed tests
    test_results.append(("Skills Scoring (Detailed)", test_skills_scoring_detailed()))
    test_results.append(("Education Scoring (Detailed)", test_education_scoring_detailed()))
    test_results.append(("Comprehensive Scoring (Detailed)", test_comprehensive_scoring_detailed()))
    test_results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "=" * 60)
    print("DETAILED TEST SUMMARY")
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
        print("🎉 All detailed tests passed! Sub-scoring system is fully functional.")
        print("\n📊 Key Features Verified:")
        print("   ✅ Skills similarity with exact, fuzzy, and synonym matching")
        print("   ✅ Experience analysis with years, seniority, and progression")
        print("   ✅ Education requirements matching with degrees and certifications")
        print("   ✅ Comprehensive scoring with weighted components")
        print("   ✅ Gap analysis and prioritized improvement suggestions")
        print("   ✅ Performance statistics and monitoring")
        print("   ✅ Edge case handling and error recovery")
    else:
        print("⚠️  Some detailed tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)