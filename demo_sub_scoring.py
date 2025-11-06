"""
Demonstration of the Sub-Scoring System Features

This script showcases the key capabilities of the newly implemented
sub-scoring system for resume-job matching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription
from resume_parser.comprehensive_sub_scoring import ComprehensiveSubScoringEngine

def create_demo_data():
    """Create realistic demo data."""
    # Candidate Profile: Mid-level Software Engineer
    resume = ParsedResume(
        contact_info=ContactInfo(
            name="Alex Johnson",
            email="alex.johnson@email.com",
            phone="555-987-6543"
        ),
        skills=[
            # Strong in some areas
            "Python", "JavaScript", "React", "SQL", "Git",
            # Some experience
            "Node.js", "MongoDB", "HTML", "CSS",
            # Soft skills
            "Problem Solving", "Team Collaboration", "Communication"
        ],
        experience=[
            WorkExperience(
                job_title="Software Developer",
                company="TechStart Inc",
                start_date="2021",
                end_date="2023",
                description="Developed web applications using React and Node.js. Worked with SQL databases and REST APIs.",
                skills_used=["Python", "JavaScript", "React", "SQL"],
                duration_months=24
            ),
            WorkExperience(
                job_title="Junior Developer",
                company="WebSolutions LLC",
                start_date="2019",
                end_date="2021",
                description="Built responsive websites and learned full-stack development.",
                skills_used=["JavaScript", "HTML", "CSS", "MongoDB"],
                duration_months=24
            )
        ],
        education=[
            Education(
                degree="Bachelor of Science",
                major="Computer Science",
                institution="State University",
                graduation_date="2019",
                gpa="3.5"
            )
        ],
        certifications=[
            Certification(
                name="JavaScript Developer Certification",
                issuer="FreeCodeCamp",
                issue_date="2020"
            )
        ]
    )
    
    # Job Posting: Senior Full-Stack Developer
    job_description = ParsedJobDescription(
        skills_required=[
            # Required technical skills
            "Python", "JavaScript", "React", "Node.js", "SQL",
            # Preferred skills (candidate is missing some)
            "AWS", "Docker", "Kubernetes", "TypeScript", "GraphQL",
            # Soft skills
            "Leadership", "Project Management", "Communication"
        ],
        experience_level="senior-level",
        tools_mentioned=["Git", "Jenkins", "PostgreSQL", "Redis"],
        confidence_scores={
            "Python": 0.95,
            "JavaScript": 0.90,
            "React": 0.85,
            "AWS": 0.80,
            "Leadership": 0.75
        },
        categories={
            "backend": ["Python", "Node.js", "SQL", "PostgreSQL"],
            "frontend": ["JavaScript", "React", "TypeScript"],
            "devops": ["AWS", "Docker", "Kubernetes", "Jenkins"],
            "soft": ["Leadership", "Project Management", "Communication"]
        },
        metadata={"source": "demo_job_posting"}
    )
    
    job_text = """
    Senior Full-Stack Developer - Growing Tech Company
    
    We're looking for a Senior Full-Stack Developer to join our dynamic team!
    
    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 5+ years of software development experience
    - Strong proficiency in Python and JavaScript
    - Experience with React and Node.js
    - Solid understanding of SQL databases
    - AWS cloud experience preferred
    - Knowledge of containerization (Docker, Kubernetes)
    - Leadership experience and project management skills
    
    Nice to Have:
    - TypeScript and GraphQL experience
    - DevOps experience with CI/CD pipelines
    - Previous startup experience
    """
    
    return resume, job_description, job_text

def run_demo():
    """Run the comprehensive demo."""
    print("🚀 Sub-Scoring System Demonstration")
    print("=" * 60)
    print("Candidate: Alex Johnson (Mid-level Software Engineer)")
    print("Position: Senior Full-Stack Developer")
    print("=" * 60)
    
    # Create test data
    resume, job_description, job_text = create_demo_data()
    
    # Initialize comprehensive scorer
    scorer = ComprehensiveSubScoringEngine(
        component_weights={
            'skills': 0.4,      # 40% weight on skills
            'experience': 0.35, # 35% weight on experience  
            'education': 0.25   # 25% weight on education
        }
    )
    
    # Calculate comprehensive match
    print("🔍 Analyzing candidate fit...")
    result = scorer.calculate_comprehensive_match(resume, job_description, job_text)
    
    # Display overall results
    print(f"\n📊 OVERALL MATCH ANALYSIS")
    print("-" * 40)
    print(f"Overall Match Score: {result.overall_score:.1f}%")
    print(f"Confidence Level: {result.confidence_score:.1%}")
    
    # Component breakdown
    print(f"\n🔧 COMPONENT BREAKDOWN")
    print("-" * 40)
    for component, score in result.weighted_scores.items():
        if not component.startswith('raw_'):
            raw_score = result.weighted_scores.get(f'raw_{component}', 0)
            weight = scorer.component_weights.get(component, 0)
            print(f"{component.capitalize():12}: {score:5.1f}% (raw: {raw_score:5.1f}%, weight: {weight:.0%})")
    
    # Skills analysis
    if result.skills_analysis:
        skills = result.skills_analysis
        print(f"\n🎯 SKILLS ANALYSIS")
        print("-" * 40)
        print(f"Skills Match Rate: {skills.metadata['match_rate']:.1%}")
        print(f"Matched Skills ({len(skills.matched_skills)}):")
        for match in skills.matched_skills[:5]:
            print(f"  ✅ {match.skill} ({match.match_type}, {match.confidence:.0%})")
        
        if skills.missing_skills:
            print(f"\nMissing Skills ({len(skills.missing_skills)}):")
            for skill in skills.missing_skills[:5]:
                print(f"  ❌ {skill}")
    
    # Experience analysis
    if result.experience_analysis:
        exp = result.experience_analysis
        print(f"\n💼 EXPERIENCE ANALYSIS")
        print("-" * 40)
        print(f"Years of Experience: {exp.years_experience:.1f} years")
        print(f"Required Experience: {exp.required_years:.1f} years")
        print(f"Experience Gap: {exp.required_years - exp.years_experience:+.1f} years")
        print(f"Seniority Match: {exp.seniority_match:.1f}%")
        print(f"Industry Relevance: {exp.industry_relevance:.1f}%")
    
    # Education analysis
    if result.education_analysis:
        edu = result.education_analysis
        print(f"\n🎓 EDUCATION ANALYSIS")
        print("-" * 40)
        print(f"Degree Match: {edu.degree_match_score:.1f}%")
        print(f"Field Match: {edu.field_match_score:.1f}%")
        print(f"Certification Score: {edu.certification_score:.1f}%")
        
        if edu.matched_requirements:
            print(f"\nMatched Requirements:")
            for match in edu.matched_requirements[:3]:
                print(f"  ✅ {match.education_type}: {match.resume_education}")
    
    # Gap analysis
    print(f"\n🔍 GAP ANALYSIS")
    print("-" * 40)
    gap_analysis = result.gap_analysis
    
    print(f"Priority Gaps:")
    for i, gap in enumerate(gap_analysis['priority_gaps'][:5], 1):
        impact_emoji = "🔴" if gap['impact'] == 'high' else "🟡" if gap['impact'] == 'medium' else "🟢"
        print(f"  {i}. {impact_emoji} {gap['description']}")
        print(f"     Impact: {gap['impact']}, Feasibility: {gap['feasibility']}")
    
    # Improvement suggestions
    print(f"\n💡 IMPROVEMENT SUGGESTIONS")
    print("-" * 40)
    for i, suggestion in enumerate(result.improvement_suggestions[:5], 1):
        priority_emoji = "🔥" if suggestion['priority'] == 'high' else "⚡" if suggestion['priority'] == 'medium' else "💭"
        print(f"{i}. {priority_emoji} {suggestion['title']}")
        print(f"   {suggestion['description']}")
        print(f"   ⏱️  {suggestion['timeframe']} | 💪 {suggestion['effort']} effort")
        print()
    
    # Recommendation
    print(f"🎯 HIRING RECOMMENDATION")
    print("-" * 40)
    
    if result.overall_score >= 80:
        recommendation = "🟢 STRONG MATCH - Highly recommended for interview"
    elif result.overall_score >= 65:
        recommendation = "🟡 GOOD MATCH - Recommended with some skill development"
    elif result.overall_score >= 50:
        recommendation = "🟠 MODERATE MATCH - Consider for junior/mid-level roles"
    else:
        recommendation = "🔴 WEAK MATCH - Not recommended for this position"
    
    print(recommendation)
    
    # Key strengths and concerns
    print(f"\nKey Strengths:")
    strengths = []
    if result.skills_analysis and result.skills_analysis.metadata['match_rate'] > 0.6:
        strengths.append("Strong technical skill alignment")
    if result.experience_analysis and result.experience_analysis.years_experience >= 3:
        strengths.append("Solid professional experience")
    if result.education_analysis and result.education_analysis.degree_match_score > 80:
        strengths.append("Educational requirements met")
    
    for strength in strengths:
        print(f"  ✅ {strength}")
    
    print(f"\nKey Concerns:")
    concerns = []
    if result.skills_analysis and len(result.skills_analysis.missing_skills) > 3:
        concerns.append(f"Missing {len(result.skills_analysis.missing_skills)} required skills")
    if result.experience_analysis and result.experience_analysis.years_experience < result.experience_analysis.required_years:
        gap = result.experience_analysis.required_years - result.experience_analysis.years_experience
        concerns.append(f"Experience gap of {gap:.1f} years")
    
    for concern in concerns:
        print(f"  ⚠️  {concern}")
    
    print(f"\n" + "=" * 60)
    print("✨ Analysis Complete! The sub-scoring system provides detailed")
    print("   insights for data-driven hiring decisions.")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()