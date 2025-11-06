#!/usr/bin/env python3
"""
Gap Analysis System Demo

This demo showcases the comprehensive gap analysis system for resume-job matching,
including skills gaps, experience analysis, and education requirements.
"""

import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_parser.gap_analysis import GapAnalysisEngine
from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 40)


def create_candidate_resume():
    """Create a realistic candidate resume for demo."""
    contact = ContactInfo(
        name="Sarah Johnson",
        email="sarah.johnson@email.com",
        phone="(555) 123-4567",
        address="San Francisco, CA",
        linkedin="linkedin.com/in/sarahjohnson",
        github="github.com/sarahjohnson"
    )
    
    experience = [
        WorkExperience(
            job_title="Software Developer",
            company="TechStart Inc",
            start_date="2021-03",
            end_date="2024-01",
            description="Developed web applications using Python Flask and React. Built REST APIs and worked with MySQL databases. Collaborated with cross-functional teams using Agile methodologies. Implemented automated testing and CI/CD pipelines.",
            skills_used=["python", "flask", "react", "javascript", "mysql", "git", "agile", "rest api"],
            duration_months=34
        ),
        WorkExperience(
            job_title="Junior Web Developer",
            company="Digital Solutions LLC",
            start_date="2019-06",
            end_date="2021-03",
            description="Created responsive websites using HTML, CSS, and JavaScript. Worked with WordPress and basic PHP. Learned version control with Git and collaborated with design teams.",
            skills_used=["html", "css", "javascript", "php", "wordpress", "git"],
            duration_months=21
        ),
        WorkExperience(
            job_title="IT Support Intern",
            company="Local Business Corp",
            start_date="2018-09",
            end_date="2019-05",
            description="Provided technical support to employees. Troubleshot hardware and software issues. Maintained computer systems and networks.",
            skills_used=["troubleshooting", "windows", "networking"],
            duration_months=8
        )
    ]
    
    education = [
        Education(
            degree="Bachelor of Science",
            major="Information Technology",
            institution="State University",
            graduation_date="2019-05",
            gpa="3.4"
        )
    ]
    
    certifications = [
        Certification(
            name="AWS Cloud Practitioner",
            issuer="Amazon Web Services",
            issue_date="2022-08",
            expiry_date="2025-08"
        )
    ]
    
    return ParsedResume(
        contact_info=contact,
        skills=["python", "flask", "react", "javascript", "html", "css", "mysql", "git", "agile", "rest api", "php", "wordpress"],
        experience=experience,
        education=education,
        certifications=certifications,
        metadata={"parsing_source": "demo", "total_experience_years": 5.25}
    )


def create_senior_job_posting():
    """Create a senior-level job posting for comparison."""
    return ParsedJobDescription(
        skills_required=[
            "python", "django", "postgresql", "docker", "kubernetes", 
            "aws", "redis", "elasticsearch", "microservices", "graphql",
            "terraform", "jenkins", "monitoring", "leadership"
        ],
        experience_level="senior-level",
        tools_mentioned=["git", "jira", "confluence", "datadog", "prometheus"],
        confidence_scores={
            "python": 0.95,
            "django": 0.90,
            "postgresql": 0.85,
            "docker": 0.88,
            "kubernetes": 0.82,
            "aws": 0.80,
            "microservices": 0.75,
            "leadership": 0.70
        },
        categories={
            "programming_languages": ["python"],
            "frameworks": ["django"],
            "databases": ["postgresql", "redis", "elasticsearch"],
            "cloud_platforms": ["aws"],
            "tools": ["docker", "kubernetes", "terraform", "jenkins", "git", "jira"],
            "soft_skills": ["leadership"],
            "methodologies": ["microservices"]
        },
        metadata={
            "description": """
            We are seeking a Senior Python Developer with 5-7 years of experience to join our growing engineering team. 
            
            Requirements:
            - Bachelor's degree in Computer Science or related field
            - 5+ years of professional Python development experience
            - Strong experience with Django framework and RESTful API development
            - Proficiency with PostgreSQL and database optimization
            - Experience with containerization (Docker) and orchestration (Kubernetes)
            - AWS cloud platform experience (EC2, S3, RDS, Lambda)
            - Knowledge of microservices architecture and distributed systems
            - Experience with CI/CD pipelines and DevOps practices
            - Leadership experience mentoring junior developers
            - Strong problem-solving and communication skills
            
            Preferred:
            - Master's degree in Computer Science
            - Experience with GraphQL and modern API design
            - Knowledge of monitoring tools (DataDog, Prometheus)
            - Terraform for infrastructure as code
            - Redis and Elasticsearch experience
            
            This is a senior-level position requiring technical leadership and architectural decision-making.
            """
        }
    )


def display_candidate_profile(resume):
    """Display candidate profile summary."""
    print_section("👤 Candidate Profile")
    
    print(f"Name: {resume.contact_info.name}")
    print(f"Email: {resume.contact_info.email}")
    print(f"Location: {resume.contact_info.address}")
    
    total_exp = sum(exp.duration_months for exp in resume.experience) / 12
    print(f"Total Experience: {total_exp:.1f} years")
    
    print(f"\n🎓 Education:")
    for edu in resume.education:
        print(f"  • {edu.degree} in {edu.major} - {edu.institution} ({edu.graduation_date})")
    
    print(f"\n📜 Certifications:")
    for cert in resume.certifications:
        print(f"  • {cert.name} - {cert.issuer}")
    
    print(f"\n💼 Recent Experience:")
    for exp in resume.experience[:2]:  # Show top 2 positions
        print(f"  • {exp.job_title} at {exp.company} ({exp.duration_months} months)")
    
    print(f"\n🛠️ Key Skills:")
    skills_display = ", ".join(resume.skills[:10])  # Show first 10 skills
    if len(resume.skills) > 10:
        skills_display += f" ... and {len(resume.skills) - 10} more"
    print(f"  {skills_display}")


def display_job_requirements(job):
    """Display job requirements summary."""
    print_section("💼 Job Requirements")
    
    print(f"Position Level: {job.experience_level.title()}")
    print(f"Required Skills: {len(job.skills_required)} skills")
    
    print(f"\n🔧 Technical Requirements:")
    for category, skills in job.categories.items():
        if skills and category != 'soft_skills':
            category_name = category.replace('_', ' ').title()
            print(f"  • {category_name}: {', '.join(skills[:5])}")
            if len(skills) > 5:
                print(f"    ... and {len(skills) - 5} more")
    
    print(f"\n🤝 Soft Skills:")
    soft_skills = job.categories.get('soft_skills', [])
    if soft_skills:
        print(f"  • {', '.join(soft_skills)}")
    else:
        print("  • Leadership, Communication, Problem Solving (inferred)")


def display_skills_gap_analysis(skills_gaps):
    """Display detailed skills gap analysis."""
    print_section("🎯 Skills Gap Analysis")
    
    if not skills_gaps:
        print("✅ No significant skill gaps identified!")
        return
    
    # Group by priority
    high_priority = [g for g in skills_gaps if g.priority == 'high']
    medium_priority = [g for g in skills_gaps if g.priority == 'medium']
    low_priority = [g for g in skills_gaps if g.priority == 'low']
    
    print(f"Total Missing Skills: {len(skills_gaps)}")
    print(f"  🔴 High Priority: {len(high_priority)}")
    print(f"  🟡 Medium Priority: {len(medium_priority)}")
    print(f"  🟢 Low Priority: {len(low_priority)}")
    
    # Show high priority gaps in detail
    if high_priority:
        print(f"\n🔴 HIGH PRIORITY GAPS:")
        for gap in high_priority[:5]:  # Show top 5
            print(f"\n  📌 {gap.skill.upper()}")
            print(f"     Category: {gap.category.replace('_', ' ').title()}")
            print(f"     Impact Score: {gap.impact_score:.2f}/1.0")
            print(f"     Confidence: {gap.confidence:.2f}/1.0")
            
            if gap.alternatives:
                print(f"     Similar Skills You Have: {', '.join(gap.alternatives)}")
            
            if gap.learning_resources:
                print(f"     Learning Path:")
                for resource in gap.learning_resources[:2]:
                    print(f"       • {resource}")
    
    # Show medium priority summary
    if medium_priority:
        print(f"\n🟡 MEDIUM PRIORITY GAPS:")
        medium_skills = [g.skill for g in medium_priority[:8]]
        print(f"  {', '.join(medium_skills)}")
        if len(medium_priority) > 8:
            print(f"  ... and {len(medium_priority) - 8} more")


def display_experience_analysis(exp_gap):
    """Display experience gap analysis."""
    print_section("📈 Experience Analysis")
    
    print(f"Required Experience: {exp_gap.years_required} years")
    print(f"Your Experience: {exp_gap.years_candidate:.1f} years")
    
    if exp_gap.shortfall_years > 0:
        print(f"⚠️  Experience Gap: {exp_gap.shortfall_years:.1f} years short")
    elif exp_gap.shortfall_years < 0:
        print(f"✅ Experience Surplus: {abs(exp_gap.shortfall_years):.1f} years over requirement")
    else:
        print(f"✅ Experience Level: Meets requirement")
    
    print(f"\n📊 Experience Quality:")
    print(f"  Relevance Score: {exp_gap.relevance_score:.2f}/1.0")
    print(f"  Industry Match: {'✅ Yes' if exp_gap.industry_match else '❌ No'}")
    print(f"  Seniority Level: {exp_gap.seniority_gap.replace('_', ' ').title()}")
    
    if exp_gap.transferable_skills:
        print(f"\n🔄 Transferable Skills:")
        for skill in exp_gap.transferable_skills:
            print(f"  • {skill.replace('_', ' ').title()}")
    
    if exp_gap.progression_analysis:
        prog = exp_gap.progression_analysis
        print(f"\n📈 Career Progression:")
        print(f"  Pattern: {prog.get('progression_type', 'unknown').replace('_', ' ').title()}")
        print(f"  Total Positions: {prog.get('total_positions', 0)}")
        print(f"  Average Tenure: {prog.get('average_tenure_months', 0):.1f} months")
        print(f"  Company Changes: {prog.get('company_changes', 0)}")


def display_education_analysis(edu_gap):
    """Display education gap analysis."""
    print_section("🎓 Education & Certification Analysis")
    
    print(f"Required Degree: {edu_gap.degree_required.replace('_', ' ').title()}")
    print(f"Your Degree: {edu_gap.degree_candidate.replace('_', ' ').title()}")
    
    if edu_gap.degree_gap == "meets_requirement":
        print(f"✅ Degree Requirement: Met")
    elif "needs" in edu_gap.degree_gap:
        print(f"⚠️  Degree Gap: {edu_gap.degree_gap.replace('_', ' ')}")
    elif "overqualified" in edu_gap.degree_gap:
        print(f"✅ Degree Level: {edu_gap.degree_gap.replace('_', ' ')}")
    
    print(f"Field of Study Match: {'✅ Yes' if edu_gap.field_match else '❌ No'}")
    
    if edu_gap.certification_gaps:
        print(f"\n📜 Missing Certifications:")
        for cert in edu_gap.certification_gaps[:3]:
            print(f"  • {cert}")
        if len(edu_gap.certification_gaps) > 3:
            print(f"  ... and {len(edu_gap.certification_gaps) - 3} more")
    
    if edu_gap.alternative_paths:
        print(f"\n🛤️  Alternative Learning Paths:")
        for path in edu_gap.alternative_paths[:4]:
            print(f"  • {path}")
    
    if edu_gap.roi_analysis:
        roi = edu_gap.roi_analysis
        formal_roi = roi.get('formal_degree', {})
        
        print(f"\n💰 Education ROI Analysis:")
        if formal_roi:
            print(f"  Formal Degree:")
            print(f"    Cost: ${formal_roi.get('cost_estimate', 0):,}")
            print(f"    Time: {formal_roi.get('time_months', 0)} months")
            print(f"    ROI Score: {formal_roi.get('roi_score', 0):.2f}/1.0")
        
        alternatives = roi.get('alternative_paths', [])
        if alternatives:
            best_alt = max(alternatives, key=lambda x: x.get('roi_score', 0))
            print(f"  Best Alternative: {best_alt.get('path', 'Unknown')}")
            print(f"    ROI Score: {best_alt.get('roi_score', 0):.2f}/1.0")


def display_overall_analysis(result):
    """Display overall gap analysis results."""
    print_section("📊 Overall Analysis & Recommendations")
    
    # Overall scores
    gap_score_percent = result.overall_gap_score * 100
    improvement_percent = result.improvement_potential * 100
    match_score_percent = (1 - result.overall_gap_score) * 100
    
    print(f"📈 MATCH ANALYSIS:")
    print(f"  Current Match Score: {match_score_percent:.1f}%")
    print(f"  Gap Score: {gap_score_percent:.1f}%")
    print(f"  Improvement Potential: +{improvement_percent:.1f}%")
    
    # Match level assessment
    if match_score_percent >= 80:
        match_level = "🟢 EXCELLENT MATCH"
    elif match_score_percent >= 65:
        match_level = "🟡 GOOD MATCH"
    elif match_score_percent >= 50:
        match_level = "🟠 MODERATE MATCH"
    else:
        match_level = "🔴 POOR MATCH"
    
    print(f"  Assessment: {match_level}")
    
    # Priority recommendations
    if result.priority_recommendations:
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(result.priority_recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Metadata insights
    metadata = result.metadata
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"  Total Skill Gaps: {metadata.get('total_skills_gaps', 0)}")
    print(f"  High Priority Skills: {metadata.get('high_priority_skills', 0)}")
    print(f"  Experience Gap: {metadata.get('experience_shortfall_years', 0):.1f} years")
    print(f"  Education Status: {metadata.get('education_gap_severity', 'unknown')}")


def run_demo():
    """Run the complete gap analysis demo."""
    print_header("🚀 RESUME-JOB GAP ANALYSIS DEMO")
    print("This demo showcases comprehensive gap analysis between a candidate and job posting.")
    
    # Create demo data
    print("\n🔄 Initializing demo data...")
    candidate = create_candidate_resume()
    job_posting = create_senior_job_posting()
    
    # Display profiles
    display_candidate_profile(candidate)
    display_job_requirements(job_posting)
    
    # Run gap analysis
    print("\n🔄 Performing comprehensive gap analysis...")
    engine = GapAnalysisEngine()
    result = engine.analyze_gaps(candidate, job_posting)
    
    # Display results
    display_skills_gap_analysis(result.skills_gaps)
    display_experience_analysis(result.experience_gap)
    display_education_analysis(result.education_gap)
    display_overall_analysis(result)
    
    # Final summary
    print_header("✨ DEMO COMPLETED")
    print("The gap analysis system has successfully identified:")
    print("• Missing technical skills with learning recommendations")
    print("• Experience gaps and career progression analysis")
    print("• Education requirements and alternative paths")
    print("• Overall match assessment with improvement potential")
    print("\nThis comprehensive analysis helps both candidates and recruiters")
    print("understand fit and identify specific areas for improvement.")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)