#!/usr/bin/env python3
"""
Basic Usage Example - Resume-Job Matcher System

This example demonstrates the fundamental usage of the Resume-Job Matcher System,
showing how to parse resumes, analyze job descriptions, and calculate similarity scores.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from resume_parser import ResumeParser
from job_parser import JobDescriptionParser
from resume_parser.similarity_engine import SimilarityEngine


def main():
    """Demonstrate basic usage of the Resume-Job Matcher System."""
    
    print("🚀 Resume-Job Matcher System - Basic Usage Example")
    print("=" * 60)
    
    # Sample resume text
    resume_text = """
    John Smith
    Software Engineer
    Email: john.smith@email.com
    Phone: (555) 123-4567
    
    SKILLS
    • Programming: Python, JavaScript, Java, SQL
    • Frameworks: Django, React, Node.js, Flask
    • Databases: PostgreSQL, MongoDB, Redis
    • Cloud: AWS, Docker, Kubernetes
    • Tools: Git, Jenkins, JIRA
    
    EXPERIENCE
    Senior Software Engineer | TechCorp Inc | 2021-2024
    • Led development of microservices architecture
    • Implemented CI/CD pipelines reducing deployment time by 50%
    • Mentored team of 4 junior developers
    • Built scalable APIs serving 1M+ requests daily
    
    Software Developer | StartupXYZ | 2019-2021
    • Developed full-stack web applications using React and Python
    • Designed and optimized database schemas
    • Collaborated with cross-functional teams in Agile environment
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2019
    """
    
    # Sample job description
    job_text = """
    Senior Python Developer - Remote
    
    We are seeking an experienced Senior Python Developer to join our growing team.
    
    REQUIREMENTS:
    • 5+ years of Python development experience
    • Strong experience with Django or Flask frameworks
    • Knowledge of cloud platforms (AWS preferred)
    • Experience with containerization (Docker, Kubernetes)
    • Database experience (PostgreSQL, MongoDB)
    • Understanding of microservices architecture
    • Experience with CI/CD pipelines
    • Strong communication and leadership skills
    
    PREFERRED:
    • React or other frontend framework experience
    • DevOps experience
    • Agile/Scrum methodology experience
    • Previous startup experience
    
    RESPONSIBILITIES:
    • Design and implement scalable Python applications
    • Lead technical discussions and architectural decisions
    • Mentor junior developers
    • Collaborate with product and design teams
    • Ensure code quality and best practices
    """
    
    try:
        # Step 1: Initialize components
        print("\n🔧 Initializing system components...")
        resume_parser = ResumeParser(enable_semantic_matching=True)
        job_parser = JobDescriptionParser()
        similarity_engine = SimilarityEngine(enable_caching=True)
        print("✅ Components initialized successfully")
        
        # Step 2: Parse resume
        print("\n📄 Parsing resume...")
        # Note: In real usage, you'd parse from file: resume_parser.parse_resume("resume.pdf")
        # For this example, we'll create a mock parsed resume
        from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education
        
        resume = ParsedResume(
            contact_info=ContactInfo(
                name="John Smith",
                email="john.smith@email.com",
                phone="(555) 123-4567"
            ),
            skills=[
                "Python", "JavaScript", "Java", "SQL", "Django", "React", 
                "Node.js", "Flask", "PostgreSQL", "MongoDB", "Redis", 
                "AWS", "Docker", "Kubernetes", "Git", "Jenkins", "JIRA"
            ],
            experience=[
                WorkExperience(
                    job_title="Senior Software Engineer",
                    company="TechCorp Inc",
                    start_date="2021",
                    end_date="2024",
                    description="Led development of microservices architecture",
                    duration_months=36
                ),
                WorkExperience(
                    job_title="Software Developer",
                    company="StartupXYZ", 
                    start_date="2019",
                    end_date="2021",
                    description="Developed full-stack web applications",
                    duration_months=24
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Science",
                    major="Computer Science",
                    institution="University of Technology",
                    graduation_date="2019"
                )
            ]
        )
        print("✅ Resume parsed successfully")
        print(f"   📊 Extracted: {len(resume.skills)} skills, {len(resume.experience)} positions")
        
        # Step 3: Parse job description
        print("\n📋 Parsing job description...")
        job = job_parser.parse_job_description(job_text)
        print("✅ Job description parsed successfully")
        print(f"   📊 Found: {len(job.skills_required)} required skills")
        print(f"   🎯 Experience level: {job.experience_level}")
        
        # Step 4: Calculate similarity
        print("\n🧮 Calculating similarity...")
        result = similarity_engine.calculate_comprehensive_similarity(
            resume=resume,
            job_description=job,
            resume_text=resume_text,
            job_text=job_text,
            include_sub_scores=True
        )
        print("✅ Similarity calculation completed")
        
        # Step 5: Display results
        print("\n" + "=" * 60)
        print("📊 SIMILARITY ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"\n🎯 Overall Match Score: {result.overall_score:.1f}%")
        
        print(f"\n📈 Component Breakdown:")
        for component, score in result.component_scores.items():
            emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            print(f"   {emoji} {component.title()}: {score:.1f}%")
        
        # Display sub-scores if available
        if hasattr(result, 'sub_scores') and result.sub_scores:
            print(f"\n🔍 Detailed Analysis:")
            
            # Skills analysis
            if 'skills' in result.sub_scores:
                skills_data = result.sub_scores['skills']
                if isinstance(skills_data, dict):
                    matched = skills_data.get('matched_skills', 0)
                    missing = skills_data.get('missing_skills', 0)
                    print(f"   Skills: {matched} matched, {missing} missing")
            
            # Experience analysis
            if 'experience' in result.sub_scores:
                exp_data = result.sub_scores['experience']
                if isinstance(exp_data, dict):
                    years = exp_data.get('years_experience', 0)
                    required = exp_data.get('required_years', 0)
                    print(f"   Experience: {years:.1f} years (required: {required:.1f})")
        
        # Display recommendations
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(result.recommendations[:5], 1):
            print(f"   {i}. {rec}")
        
        # Match assessment
        print(f"\n🎯 Assessment:")
        if result.overall_score >= 80:
            assessment = "🟢 EXCELLENT MATCH - Highly recommended for interview"
        elif result.overall_score >= 65:
            assessment = "🟡 GOOD MATCH - Strong candidate with minor gaps"
        elif result.overall_score >= 50:
            assessment = "🟠 MODERATE MATCH - Potential with development"
        else:
            assessment = "🔴 WEAK MATCH - Significant gaps identified"
        
        print(f"   {assessment}")
        
        print(f"\n" + "=" * 60)
        print("✨ Analysis completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Basic usage example completed successfully!")
        print("\n📚 Next steps:")
        print("   • Try with your own resume and job description")
        print("   • Explore batch processing: examples/batch_processing.py")
        print("   • Check advanced features: examples/custom_configuration.py")
    else:
        print("\n💥 Example failed. Please check the error messages above.")
        sys.exit(1)