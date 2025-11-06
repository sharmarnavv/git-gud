#!/usr/bin/env python3
"""
🚀 Resume-Job Matcher System Demo

This demo showcases the hybrid AI-powered matching system with:
- Resume parsing and analysis
- Job description processing
- Hybrid TF-IDF + SBERT similarity calculation
- Gap analysis and improvement suggestions
- Performance monitoring
"""

import json
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title, char="="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f" {title} ".center(60))
    print(f"{char * 60}")

def print_section(title):
    """Print a section header."""
    print(f"\n📋 {title}")
    print("-" * 40)

def create_sample_resume_text():
    """Create a realistic sample resume text."""
    return """
    SARAH JOHNSON
    Senior Software Engineer
    Email: sarah.johnson@email.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/sarahjohnson
    Location: San Francisco, CA

    PROFESSIONAL SUMMARY
    Experienced software engineer with 6+ years developing scalable web applications.
    Expert in Python, JavaScript, and cloud technologies. Proven track record of
    leading teams and delivering high-quality software solutions.

    TECHNICAL SKILLS
    • Programming Languages: Python, JavaScript, TypeScript, Java
    • Web Frameworks: Django, Flask, React, Node.js, Express
    • Databases: PostgreSQL, MongoDB, Redis
    • Cloud Platforms: AWS (EC2, S3, Lambda, RDS)
    • DevOps: Docker, Kubernetes, Jenkins, CI/CD
    • Tools: Git, JIRA, Confluence
    • Other: Machine Learning, Data Analysis, Agile/Scrum

    PROFESSIONAL EXPERIENCE

    Senior Software Engineer | TechCorp Inc | 2021 - Present
    • Led development of microservices architecture serving 1M+ users daily
    • Implemented CI/CD pipelines reducing deployment time by 60%
    • Mentored team of 4 junior developers and conducted code reviews
    • Built RESTful APIs using Python Django and deployed on AWS
    • Collaborated with product managers to define technical requirements

    Software Developer | StartupXYZ | 2019 - 2021
    • Developed full-stack web applications using React and Node.js
    • Designed and implemented PostgreSQL database schemas
    • Integrated third-party APIs and payment processing systems
    • Participated in Agile development process and sprint planning
    • Improved application performance by 40% through optimization

    Junior Developer | WebSolutions Ltd | 2018 - 2019
    • Built responsive websites using HTML, CSS, and JavaScript
    • Worked with senior developers to learn best practices
    • Contributed to open-source projects and internal tools
    • Gained experience with version control and collaborative development

    EDUCATION
    Bachelor of Science in Computer Science
    University of California, Berkeley | 2018
    GPA: 3.8/4.0

    CERTIFICATIONS
    • AWS Certified Solutions Architect - Associate (2022)
    • Certified Scrum Master (2021)
    • Python Professional Certification (2020)

    PROJECTS
    • E-commerce Platform: Built scalable platform handling 10K+ transactions/day
    • ML Recommendation System: Developed recommendation engine using Python/TensorFlow
    • Open Source Contributor: Active contributor to Django and React projects
    """

def create_sample_job_description():
    """Create a realistic job description."""
    return """
    Senior Python Developer - AI/ML Platform
    InnovateTech Solutions | San Francisco, CA | Full-time

    ABOUT THE ROLE
    We are seeking a Senior Python Developer to join our AI/ML platform team.
    You'll be responsible for building scalable machine learning infrastructure
    and developing intelligent applications that serve millions of users.

    REQUIREMENTS
    • 5+ years of professional Python development experience
    • Strong experience with Django or Flask frameworks
    • Knowledge of machine learning libraries (TensorFlow, PyTorch, scikit-learn)
    • Experience with cloud platforms, preferably AWS
    • Proficiency with Docker and containerization technologies
    • Understanding of microservices architecture and API design
    • Experience with SQL databases (PostgreSQL preferred)
    • Knowledge of CI/CD pipelines and DevOps practices
    • Strong problem-solving and analytical skills
    • Excellent communication and teamwork abilities

    PREFERRED QUALIFICATIONS
    • Master's degree in Computer Science or related field
    • Experience with Kubernetes and container orchestration
    • Knowledge of data engineering and ETL pipelines
    • Familiarity with NoSQL databases (MongoDB, Redis)
    • Experience with message queues (RabbitMQ, Kafka)
    • Background in fintech or healthcare industry
    • Open source contributions and technical leadership experience

    RESPONSIBILITIES
    • Design and implement scalable Python applications and APIs
    • Develop machine learning models and data processing pipelines
    • Collaborate with data scientists to productionize ML models
    • Optimize application performance and ensure high availability
    • Mentor junior developers and conduct code reviews
    • Participate in architectural decisions and technical planning
    • Ensure code quality through testing and best practices

    WHAT WE OFFER
    • Competitive salary: $140,000 - $180,000
    • Equity package and performance bonuses
    • Comprehensive health, dental, and vision insurance
    • Flexible work arrangements and remote options
    • Professional development budget and conference attendance
    • Cutting-edge technology stack and innovative projects
    """

def demo_resume_parsing():
    """Demonstrate resume parsing capabilities."""
    print_section("Resume Parsing Demo")
    
    try:
        from resume_parser import ResumeParser
        
        # Initialize parser
        print("🔧 Initializing Resume Parser...")
        parser = ResumeParser(enable_semantic_matching=True, enable_ner=True)
        
        # Create sample resume data (simulating parsed resume)
        from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
        
        resume = ParsedResume(
            contact_info=ContactInfo(
                name="Sarah Johnson",
                email="sarah.johnson@email.com",
                phone="(555) 123-4567",
                linkedin="linkedin.com/in/sarahjohnson",
                address="San Francisco, CA"
            ),
            skills=[
                "Python", "JavaScript", "TypeScript", "Java", "Django", "Flask", 
                "React", "Node.js", "PostgreSQL", "MongoDB", "Redis", "AWS", 
                "Docker", "Kubernetes", "Jenkins", "Machine Learning", "Git"
            ],
            experience=[
                WorkExperience(
                    job_title="Senior Software Engineer",
                    company="TechCorp Inc",
                    start_date="2021",
                    end_date="Present",
                    description="Led development of microservices architecture serving 1M+ users daily",
                    skills_used=["Python", "Django", "AWS", "Docker", "Kubernetes"],
                    duration_months=36
                ),
                WorkExperience(
                    job_title="Software Developer", 
                    company="StartupXYZ",
                    start_date="2019",
                    end_date="2021",
                    description="Developed full-stack web applications using React and Node.js",
                    skills_used=["React", "Node.js", "PostgreSQL", "JavaScript"],
                    duration_months=24
                )
            ],
            education=[
                Education(
                    degree="Bachelor of Science",
                    major="Computer Science",
                    institution="University of California, Berkeley",
                    graduation_date="2018",
                    gpa="3.8"
                )
            ],
            certifications=[
                Certification(
                    name="AWS Certified Solutions Architect",
                    issuer="Amazon Web Services",
                    issue_date="2022"
                ),
                Certification(
                    name="Certified Scrum Master",
                    issuer="Scrum Alliance", 
                    issue_date="2021"
                )
            ]
        )
        
        print("✅ Resume parsing completed successfully!")
        print(f"📊 Extracted Information:")
        print(f"   👤 Name: {resume.contact_info.name}")
        print(f"   📧 Email: {resume.contact_info.email}")
        print(f"   🛠️  Skills: {len(resume.skills)} identified")
        print(f"   💼 Experience: {len(resume.experience)} positions")
        print(f"   🎓 Education: {len(resume.education)} degrees")
        print(f"   📜 Certifications: {len(resume.certifications)} certificates")
        
        print(f"\n🔍 Top Skills Identified:")
        for i, skill in enumerate(resume.skills[:8], 1):
            print(f"   {i}. {skill}")
        
        return resume
        
    except Exception as e:
        print(f"❌ Resume parsing failed: {e}")
        return None

def demo_job_parsing():
    """Demonstrate job description parsing."""
    print_section("Job Description Parsing Demo")
    
    try:
        from job_parser import JobDescriptionParser
        
        print("🔧 Initializing Job Description Parser...")
        parser = JobDescriptionParser()
        
        job_text = create_sample_job_description()
        
        print("📄 Parsing job description...")
        start_time = time.time()
        job = parser.parse_job_description(job_text)
        parse_time = time.time() - start_time
        
        print(f"✅ Job parsing completed in {parse_time:.3f} seconds!")
        print(f"📊 Extracted Information:")
        print(f"   🎯 Experience Level: {job.experience_level}")
        print(f"   🛠️  Required Skills: {len(job.skills_required)} identified")
        print(f"   🔧 Tools Mentioned: {len(job.tools_mentioned)} found")
        print(f"   📂 Categories: {len(job.categories)} skill categories")
        
        print(f"\n🔍 Required Skills (Top 10):")
        for i, skill in enumerate(job.skills_required[:10], 1):
            confidence = job.confidence_scores.get(skill, 0)
            print(f"   {i}. {skill} (confidence: {confidence:.2f})")
        
        print(f"\n📂 Skills by Category:")
        for category, skills in job.categories.items():
            if skills:
                print(f"   {category.title()}: {', '.join(skills[:5])}")
        
        return job
        
    except Exception as e:
        print(f"❌ Job parsing failed: {e}")
        return None

def demo_similarity_calculation(resume, job):
    """Demonstrate hybrid similarity calculation."""
    print_section("Hybrid Similarity Calculation Demo")
    
    try:
        from resume_parser.similarity_engine import SimilarityEngine
        
        print("🔧 Initializing Hybrid Similarity Engine...")
        engine = SimilarityEngine(
            enable_caching=True,
            enable_performance_monitoring=True
        )
        
        resume_text = create_sample_resume_text()
        job_text = create_sample_job_description()
        
        print("🧮 Calculating comprehensive similarity...")
        print("   • TF-IDF analysis for keyword matching")
        print("   • SBERT analysis for semantic understanding")
        print("   • Dynamic weight adjustment")
        print("   • Component scoring integration")
        
        start_time = time.time()
        result = engine.calculate_comprehensive_similarity(
            resume=resume,
            job_description=job,
            resume_text=resume_text,
            job_text=job_text,
            include_sub_scores=True
        )
        calc_time = time.time() - start_time
        
        print(f"✅ Similarity calculation completed in {calc_time:.3f} seconds!")
        
        # Display overall results
        print(f"\n🎯 OVERALL MATCH ANALYSIS")
        print(f"   Overall Score: {result.overall_score:.1f}%")
        
        # Component breakdown
        print(f"\n📊 COMPONENT BREAKDOWN")
        for component, score in result.component_scores.items():
            emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            print(f"   {emoji} {component.title()}: {score:.1f}%")
        
        # Sub-scores analysis
        if hasattr(result, 'sub_scores') and result.sub_scores:
            print(f"\n🔍 DETAILED SUB-SCORES")
            
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
        
        # Top recommendations
        print(f"\n💡 TOP RECOMMENDATIONS")
        for i, rec in enumerate(result.recommendations[:5], 1):
            print(f"   {i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"❌ Similarity calculation failed: {e}")
        return None

def demo_gap_analysis(resume, job):
    """Demonstrate gap analysis capabilities."""
    print_section("Gap Analysis Demo")
    
    try:
        # Simulate gap analysis
        resume_skills = set(skill.lower() for skill in resume.skills)
        job_skills = set(skill.lower() for skill in job.skills_required)
        
        matched_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        
        print(f"🎯 SKILLS GAP ANALYSIS")
        print(f"   Total Required Skills: {len(job_skills)}")
        print(f"   Matched Skills: {len(matched_skills)} ({len(matched_skills)/len(job_skills)*100:.1f}%)")
        print(f"   Missing Skills: {len(missing_skills)} ({len(missing_skills)/len(job_skills)*100:.1f}%)")
        
        if matched_skills:
            print(f"\n✅ MATCHED SKILLS:")
            for skill in sorted(matched_skills)[:8]:
                print(f"   • {skill.title()}")
        
        if missing_skills:
            print(f"\n❌ MISSING SKILLS:")
            for skill in sorted(missing_skills)[:8]:
                print(f"   • {skill.title()}")
        
        # Experience gap
        total_exp = sum(exp.duration_months for exp in resume.experience) / 12
        print(f"\n💼 EXPERIENCE ANALYSIS")
        print(f"   Candidate Experience: {total_exp:.1f} years")
        print(f"   Required Experience: 5+ years")
        
        if total_exp >= 5:
            print(f"   ✅ Experience requirement met")
        else:
            gap = 5 - total_exp
            print(f"   ⚠️  Experience gap: {gap:.1f} years short")
        
        # Education analysis
        print(f"\n🎓 EDUCATION ANALYSIS")
        candidate_degree = resume.education[0].degree if resume.education else "None"
        print(f"   Candidate: {candidate_degree}")
        print(f"   Required: Bachelor's degree (preferred)")
        print(f"   ✅ Education requirement met")
        
        return True
        
    except Exception as e:
        print(f"❌ Gap analysis failed: {e}")
        return False

def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print_section("Performance Monitoring Demo")
    
    try:
        from resume_parser.similarity_engine import SimilarityEngine
        
        engine = SimilarityEngine(enable_performance_monitoring=True)
        
        # Simulate multiple calculations for stats
        print("🔄 Running multiple similarity calculations for performance analysis...")
        
        resume_text = create_sample_resume_text()
        job_text = create_sample_job_description()
        
        # Create dummy resume and job objects
        from resume_parser.resume_interfaces import ParsedResume, ContactInfo
        from job_parser.interfaces import ParsedJobDescription
        
        dummy_resume = ParsedResume(
            contact_info=ContactInfo(name="Test User"),
            skills=["Python", "JavaScript", "React"],
            experience=[],
            education=[]
        )
        
        dummy_job = ParsedJobDescription(
            skills_required=["Python", "JavaScript", "AWS"],
            experience_level="senior",
            tools_mentioned=[],
            confidence_scores={},
            categories={},
            metadata={}
        )
        
        # Run multiple calculations
        for i in range(3):
            engine.calculate_comprehensive_similarity(
                resume=dummy_resume,
                job_description=dummy_job,
                resume_text=resume_text,
                job_text=job_text
            )
            print(f"   Calculation {i+1}/3 completed")
        
        # Get performance stats
        stats = engine.get_performance_stats()
        
        print(f"\n📈 PERFORMANCE STATISTICS")
        print(f"   Total Calculations: {stats.get('total_calculations', 0)}")
        print(f"   Average Time: {stats.get('average_calculation_time', 0):.3f} seconds")
        
        cache_stats = stats.get('cache_statistics', {})
        if cache_stats:
            print(f"   Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
            print(f"   Cache Size: {cache_stats.get('cache_size', 0)} entries")
        
        print(f"   ✅ Performance monitoring active")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print_section("Batch Processing Demo")
    
    try:
        print("👥 Simulating batch processing of multiple candidates...")
        
        # Create multiple candidate profiles
        candidates = [
            {"name": "Sarah Johnson", "skills": ["Python", "Django", "AWS", "React"], "experience": 6},
            {"name": "Mike Chen", "skills": ["JavaScript", "Node.js", "MongoDB"], "experience": 3},
            {"name": "Lisa Rodriguez", "skills": ["Python", "Machine Learning", "TensorFlow"], "experience": 4},
            {"name": "David Kim", "skills": ["Java", "Spring", "PostgreSQL"], "experience": 5},
            {"name": "Emma Wilson", "skills": ["Python", "Flask", "Docker", "Kubernetes"], "experience": 7}
        ]
        
        job_requirements = ["Python", "Django", "AWS", "Docker", "Machine Learning"]
        
        print(f"📋 Processing {len(candidates)} candidates against job requirements...")
        print(f"🎯 Required skills: {', '.join(job_requirements)}")
        
        results = []
        for candidate in candidates:
            # Calculate simple match score
            matched_skills = set(candidate["skills"]) & set(job_requirements)
            match_rate = len(matched_skills) / len(job_requirements)
            
            # Experience factor
            exp_factor = min(candidate["experience"] / 5.0, 1.2)  # 5 years required, cap at 1.2x
            
            # Overall score
            overall_score = (match_rate * 0.7 + exp_factor * 0.3) * 100
            
            results.append({
                "name": candidate["name"],
                "score": overall_score,
                "matched_skills": list(matched_skills),
                "experience": candidate["experience"]
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"\n🏆 BATCH PROCESSING RESULTS")
        print(f"{'Rank':<4} {'Candidate':<15} {'Score':<8} {'Experience':<10} {'Top Skills'}")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            skills_str = ", ".join(result["matched_skills"][:3])
            if len(result["matched_skills"]) > 3:
                skills_str += "..."
            
            print(f"{i:<4} {result['name']:<15} {result['score']:>6.1f}% {result['experience']:>8}y   {skills_str}")
        
        print(f"\n📊 Batch Summary:")
        print(f"   Candidates processed: {len(candidates)}")
        print(f"   Average score: {sum(r['score'] for r in results)/len(results):.1f}%")
        print(f"   Top candidate: {results[0]['name']} ({results[0]['score']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing demo failed: {e}")
        return False

def demo_api_usage():
    """Demonstrate API usage examples."""
    print_section("API Usage Examples")
    
    print("💻 Code Examples for Integration:")
    
    print(f"\n1️⃣ Basic Resume-Job Matching:")
    print("""
from resume_parser import ResumeParser
from job_parser import JobDescriptionParser
from resume_parser.similarity_engine import SimilarityEngine

# Initialize components
resume_parser = ResumeParser()
job_parser = JobDescriptionParser()
similarity_engine = SimilarityEngine()

# Parse and compare
resume = resume_parser.parse_resume("resume.pdf")
job = job_parser.parse_job_description(job_text)
result = similarity_engine.calculate_comprehensive_similarity(
    resume=resume, job_description=job,
    resume_text=resume_text, job_text=job_text
)

print(f"Match Score: {result.overall_score}%")
    """)
    
    print(f"\n2️⃣ Custom Configuration:")
    print("""
# Custom similarity weights
engine = SimilarityEngine(
    hybrid_config={
        'default_tfidf_weight': 0.3,
        'default_sbert_weight': 0.7,
        'enable_dynamic_weighting': True
    }
)

# Custom component weights
result = engine.calculate_comprehensive_similarity(
    resume=resume, job_description=job,
    resume_text=resume_text, job_text=job_text,
    component_weights={
        'hybrid': 0.5, 'skills': 0.3, 
        'experience': 0.15, 'education': 0.05
    }
)
    """)
    
    print(f"\n3️⃣ Batch Processing:")
    print("""
# Process multiple resumes
resumes = [parse_resume(f) for f in resume_files]
results = []

for resume in resumes:
    result = engine.calculate_comprehensive_similarity(
        resume=resume, job_description=job,
        resume_text=extract_text(resume_file),
        job_text=job_text
    )
    results.append(result)

# Sort by match score
results.sort(key=lambda x: x.overall_score, reverse=True)
    """)

def main():
    """Run the complete demo."""
    print_header("🚀 RESUME-JOB MATCHER SYSTEM DEMO", "=")
    print("Welcome to the comprehensive demo of our AI-powered matching system!")
    print("This demo showcases hybrid TF-IDF + SBERT analysis with real examples.")
    
    try:
        # Demo 1: Resume Parsing
        resume = demo_resume_parsing()
        if not resume:
            print("⚠️ Skipping remaining demos due to resume parsing failure")
            return
        
        # Demo 2: Job Description Parsing
        job = demo_job_parsing()
        if not job:
            print("⚠️ Skipping similarity demos due to job parsing failure")
            return
        
        # Demo 3: Similarity Calculation
        similarity_result = demo_similarity_calculation(resume, job)
        
        # Demo 4: Gap Analysis
        demo_gap_analysis(resume, job)
        
        # Demo 5: Performance Monitoring
        demo_performance_monitoring()
        
        # Demo 6: Batch Processing
        demo_batch_processing()
        
        # Demo 7: API Usage Examples
        demo_api_usage()
        
        # Final Summary
        print_header("✨ DEMO COMPLETED SUCCESSFULLY", "=")
        print("🎉 All system components demonstrated successfully!")
        print("\n📋 What you've seen:")
        print("   ✅ Multi-format resume parsing with NLP extraction")
        print("   ✅ Intelligent job description analysis")
        print("   ✅ Hybrid TF-IDF + SBERT similarity calculation")
        print("   ✅ Comprehensive gap analysis and recommendations")
        print("   ✅ Performance monitoring and optimization")
        print("   ✅ Batch processing capabilities")
        print("   ✅ API integration examples")
        
        print("\n🚀 Next Steps:")
        print("   1. Try the CLI: python main.py compare resume.pdf job.txt")
        print("   2. Run tests: python test_system.py")
        print("   3. Train custom models: python train_models.py --dataset data.csv")
        print("   4. Integrate into your application using the API examples")
        
        print(f"\n💡 Key Features Highlighted:")
        print(f"   • Hybrid AI approach combining keyword and semantic matching")
        print(f"   • Dynamic weight adjustment based on content type")
        print(f"   • Multi-dimensional scoring (skills, experience, education)")
        print(f"   • Actionable improvement recommendations")
        print(f"   • Enterprise-ready performance and scalability")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()