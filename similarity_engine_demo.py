#!/usr/bin/env python3
"""
Comprehensive Demo of the SimilarityEngine

This demo showcases the main features of the SimilarityEngine including:
- Comprehensive similarity calculation with detailed reporting
- Component score breakdown (TF-IDF, SBERT, Skills, Experience)
- Performance monitoring and caching
- Batch processing capabilities
- Detailed recommendations and analysis
"""

import json
import time
from resume_parser.similarity_engine import SimilarityEngine
from resume_parser.resume_interfaces import ParsedResume, ContactInfo, WorkExperience, Education, Certification
from job_parser.interfaces import ParsedJobDescription


def create_sample_resume_1():
    """Create a sample resume for a Senior Python Developer."""
    contact = ContactInfo(
        name="Sarah Johnson",
        email="sarah.johnson@email.com",
        phone="555-123-4567",
        linkedin="linkedin.com/in/sarahjohnson",
        github="github.com/sarahjohnson"
    )
    
    experience = [
        WorkExperience(
            job_title="Senior Software Engineer",
            company="TechCorp Inc",
            start_date="2021-03-01",
            end_date="2024-01-01",
            description="Led development of microservices architecture using Python, Django, and AWS. Managed team of 4 developers and implemented CI/CD pipelines. Built RESTful APIs serving 1M+ requests daily.",
            skills_used=["Python", "Django", "AWS", "Docker", "PostgreSQL", "Redis", "Git"],
            duration_months=34
        ),
        WorkExperience(
            job_title="Software Developer",
            company="StartupXYZ",
            start_date="2019-06-01",
            end_date="2021-02-28",
            description="Developed web applications using Python Flask and React. Implemented machine learning models for recommendation system. Worked with PostgreSQL and MongoDB databases.",
            skills_used=["Python", "Flask", "React", "JavaScript", "PostgreSQL", "MongoDB", "Machine Learning"],
            duration_months=21
        ),
        WorkExperience(
            job_title="Junior Developer",
            company="WebSolutions Ltd",
            start_date="2018-01-01",
            end_date="2019-05-31",
            description="Built responsive web applications using HTML, CSS, JavaScript, and Python. Collaborated with design team to implement user interfaces.",
            skills_used=["Python", "JavaScript", "HTML", "CSS", "MySQL"],
            duration_months=17
        )
    ]
    
    education = [
        Education(
            degree="Bachelor of Science in Computer Science",
            major="Computer Science",
            institution="University of Technology",
            graduation_date="2017-12-15",
            gpa="3.8"
        )
    ]
    
    certifications = [
        Certification(
            name="AWS Certified Solutions Architect",
            issuer="Amazon Web Services",
            issue_date="2022-06-15",
            expiry_date="2025-06-15"
        ),
        Certification(
            name="Python Professional Certification",
            issuer="Python Institute",
            issue_date="2021-03-10"
        )
    ]
    
    return ParsedResume(
        contact_info=contact,
        skills=[
            "Python", "Django", "Flask", "FastAPI", "JavaScript", "React", "Vue.js",
            "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB", "Redis",
            "Git", "CI/CD", "Machine Learning", "REST APIs", "Microservices",
            "Agile", "Scrum", "Leadership", "Team Management"
        ],
        experience=experience,
        education=education,
        certifications=certifications,
        metadata={"years_experience": 6, "source": "demo"}
    )


def create_sample_resume_2():
    """Create a sample resume for a Junior Frontend Developer."""
    contact = ContactInfo(
        name="Alex Chen",
        email="alex.chen@email.com",
        phone="555-987-6543",
        github="github.com/alexchen"
    )
    
    experience = [
        WorkExperience(
            job_title="Frontend Developer",
            company="Digital Agency",
            start_date="2022-09-01",
            end_date="2024-01-01",
            description="Developed responsive web applications using React and TypeScript. Collaborated with UX designers to implement pixel-perfect designs. Optimized application performance and accessibility.",
            skills_used=["React", "TypeScript", "JavaScript", "HTML", "CSS", "Sass"],
            duration_months=16
        ),
        WorkExperience(
            job_title="Web Developer Intern",
            company="Creative Studio",
            start_date="2022-01-01",
            end_date="2022-08-31",
            description="Built landing pages and marketing websites using HTML, CSS, and JavaScript. Learned React fundamentals and contributed to team projects.",
            skills_used=["HTML", "CSS", "JavaScript", "React", "Git"],
            duration_months=8
        )
    ]
    
    education = [
        Education(
            degree="Bachelor of Arts in Web Design",
            major="Web Design and Development",
            institution="Design College",
            graduation_date="2021-12-15",
            gpa="3.6"
        )
    ]
    
    return ParsedResume(
        contact_info=contact,
        skills=[
            "React", "JavaScript", "TypeScript", "HTML", "CSS", "Sass", "LESS",
            "Node.js", "Express", "Git", "Webpack", "Responsive Design",
            "UI/UX", "Figma", "Adobe Creative Suite", "Agile"
        ],
        experience=experience,
        education=education,
        metadata={"years_experience": 2, "source": "demo"}
    )


def create_sample_job_description():
    """Create a sample job description for a Senior Python Developer position."""
    return ParsedJobDescription(
        skills_required=[
            "Python", "Django", "FastAPI", "AWS", "Docker", "PostgreSQL",
            "Redis", "Git", "CI/CD", "REST APIs", "Microservices",
            "Leadership", "Team Management", "Agile", "Scrum"
        ],
        experience_level="Senior (5+ years)",
        tools_mentioned=[
            "Python", "Django", "FastAPI", "AWS", "Docker", "Kubernetes",
            "PostgreSQL", "Redis", "Git", "Jenkins", "Terraform"
        ],
        confidence_scores={
            "Python": 0.95,
            "Django": 0.90,
            "AWS": 0.85,
            "Docker": 0.80,
            "PostgreSQL": 0.75,
            "Leadership": 0.70
        },
        categories={
            "technical": ["Python", "Django", "FastAPI", "PostgreSQL", "Redis"],
            "tools": ["AWS", "Docker", "Kubernetes", "Git", "Jenkins"],
            "soft": ["Leadership", "Team Management", "Communication"],
            "methodologies": ["Agile", "Scrum", "CI/CD", "Microservices"]
        },
        metadata={
            "company": "InnovateTech Solutions",
            "location": "San Francisco, CA",
            "salary_range": "$120,000 - $160,000",
            "remote_friendly": True,
            "source": "demo"
        }
    )


def print_separator(title):
    """Print a formatted separator with title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80))
    print("="*80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def demo_single_similarity_calculation():
    """Demonstrate single similarity calculation with detailed analysis."""
    print_separator("SINGLE SIMILARITY CALCULATION DEMO")
    
    # Initialize the SimilarityEngine
    print("Initializing SimilarityEngine...")
    engine = SimilarityEngine(
        enable_caching=True,
        cache_size_limit=100,
        enable_performance_monitoring=True
    )
    print("✓ SimilarityEngine initialized successfully")
    
    # Create test data
    resume = create_sample_resume_1()
    job = create_sample_job_description()
    
    # Create realistic text content
    resume_text = """
    Sarah Johnson - Senior Software Engineer
    
    Experienced software engineer with 6+ years developing scalable web applications.
    Expert in Python, Django, and AWS cloud services. Led teams and implemented
    microservices architecture serving millions of users daily.
    
    Technical Skills: Python, Django, Flask, FastAPI, JavaScript, React, AWS, Docker,
    Kubernetes, PostgreSQL, MongoDB, Redis, Git, CI/CD, Machine Learning
    
    Experience:
    - Senior Software Engineer at TechCorp Inc (2021-2024)
    - Software Developer at StartupXYZ (2019-2021)  
    - Junior Developer at WebSolutions Ltd (2018-2019)
    
    Education: BS Computer Science, University of Technology
    Certifications: AWS Solutions Architect, Python Professional
    """
    
    job_text = """
    Senior Python Developer - InnovateTech Solutions
    
    We are seeking a Senior Python Developer to join our growing engineering team.
    The ideal candidate will have 5+ years of experience building scalable web applications
    using Python, Django, and cloud technologies.
    
    Required Skills:
    - 5+ years Python development experience
    - Strong experience with Django or FastAPI
    - AWS cloud services (EC2, S3, RDS, Lambda)
    - Docker and containerization
    - PostgreSQL and Redis
    - Git version control and CI/CD pipelines
    - REST API design and microservices architecture
    - Leadership and team management experience
    - Agile/Scrum methodologies
    
    Preferred Skills:
    - Kubernetes orchestration
    - Machine learning experience
    - Frontend technologies (React, JavaScript)
    
    This is a senior-level position offering competitive salary, equity, and remote work options.
    """
    
    print(f"\nAnalyzing resume for: {resume.contact_info.name}")
    print(f"Job position: Senior Python Developer at InnovateTech Solutions")
    
    # Calculate comprehensive similarity
    start_time = time.time()
    report = engine.calculate_comprehensive_similarity(
        resume=resume,
        job_description=job,
        resume_text=resume_text,
        job_text=job_text,
        include_sub_scores=True
    )
    calculation_time = time.time() - start_time
    
    # Display results
    print_subsection("OVERALL SIMILARITY SCORE")
    print(f"🎯 Overall Match Score: {report.overall_score:.1f}%")
    print(f"⏱️  Calculation Time: {calculation_time:.3f} seconds")
    
    print_subsection("COMPONENT SCORES BREAKDOWN")
    for component, score in report.component_scores.items():
        emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        print(f"{emoji} {component.title()}: {score:.1f}%")
    
    print_subsection("DETAILED SUB-SCORES")
    for category, data in report.sub_scores.items():
        if isinstance(data, dict) and 'overall_score' in data:
            print(f"\n{category.title()} Analysis:")
            print(f"  Score: {data['overall_score']:.1f}%")
            if 'confidence' in data:
                print(f"  Confidence: {data['confidence']:.2f}")
            
            # Show specific details for skills
            if category == 'skills' and 'matched_skills' in data:
                print(f"  Matched Skills: {data['matched_skills']}")
                print(f"  Missing Skills: {data['missing_skills']}")
            
            # Show specific details for experience
            elif category == 'experience':
                if 'years_experience' in data and 'required_years' in data:
                    print(f"  Years Experience: {data['years_experience']:.1f}")
                    print(f"  Required Years: {data['required_years']:.1f}")
                    gap = data['years_experience'] - data['required_years']
                    if gap >= 0:
                        print(f"  Experience Gap: +{gap:.1f} years (exceeds requirement)")
                    else:
                        print(f"  Experience Gap: {gap:.1f} years (below requirement)")
    
    print_subsection("TOP RECOMMENDATIONS")
    for i, recommendation in enumerate(report.recommendations[:5], 1):
        print(f"{i}. {recommendation}")
    
    print_subsection("ANALYSIS METADATA")
    metadata = report.metadata
    print(f"Components Calculated: {', '.join(metadata.get('components_calculated', []))}")
    print(f"Sub-scores Included: {metadata.get('sub_scores_included', False)}")
    print(f"Cache Used: {metadata.get('cache_used', False)}")
    
    return engine, report


def demo_batch_processing(engine):
    """Demonstrate batch processing capabilities."""
    print_separator("BATCH PROCESSING DEMO")
    
    # Create multiple resumes
    resumes = [
        create_sample_resume_1(),  # Senior developer - good match
        create_sample_resume_2(),  # Junior frontend - partial match
    ]
    
    # Add a third resume programmatically
    resume_3 = ParsedResume(
        contact_info=ContactInfo(name="Mike Wilson", email="mike@email.com"),
        skills=["Java", "Spring", "MySQL", "Maven"],
        experience=[
            WorkExperience(
                job_title="Java Developer",
                company="Enterprise Corp",
                duration_months=36,
                description="Developed enterprise applications using Java and Spring framework"
            )
        ],
        education=[
            Education(degree="BS Computer Engineering", major="Computer Engineering")
        ],
        metadata={"years_experience": 3}
    )
    resumes.append(resume_3)
    
    job = create_sample_job_description()
    
    # Create corresponding resume texts
    resume_texts = [
        "Senior Python developer with Django and AWS experience, 6+ years",
        "Frontend developer with React and JavaScript skills, 2 years experience",
        "Java developer with Spring framework experience, 3 years in enterprise applications"
    ]
    
    job_text = "Senior Python Developer position requiring Django, AWS, and leadership skills"
    
    print(f"Processing {len(resumes)} resumes against the job description...")
    
    # Track progress
    def progress_callback(current, total, score):
        print(f"  Processed {current}/{total} - Latest score: {score:.1f}%")
    
    # Calculate batch similarities
    start_time = time.time()
    reports = engine.calculate_batch_similarity(
        resumes=resumes,
        job_description=job,
        resume_texts=resume_texts,
        job_text=job_text,
        include_sub_scores=True,
        progress_callback=progress_callback
    )
    batch_time = time.time() - start_time
    
    print(f"\n✓ Batch processing completed in {batch_time:.3f} seconds")
    
    print_subsection("BATCH RESULTS SUMMARY")
    
    # Sort by similarity score
    sorted_results = sorted(
        [(i, report) for i, report in enumerate(reports)],
        key=lambda x: x[1].overall_score,
        reverse=True
    )
    
    print("Ranking | Candidate | Overall Score | Top Component Scores")
    print("-" * 70)
    
    for rank, (idx, report) in enumerate(sorted_results, 1):
        candidate_name = resumes[idx].contact_info.name
        overall_score = report.overall_score
        
        # Get top 2 component scores
        top_components = sorted(
            report.component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        component_str = ", ".join([f"{comp}: {score:.0f}%" for comp, score in top_components])
        
        print(f"   {rank}    | {candidate_name:<12} | {overall_score:>6.1f}%     | {component_str}")
    
    print_subsection("DETAILED ANALYSIS FOR TOP CANDIDATE")
    
    best_idx, best_report = sorted_results[0]
    best_candidate = resumes[best_idx]
    
    print(f"🏆 Top Candidate: {best_candidate.contact_info.name}")
    print(f"📧 Email: {best_candidate.contact_info.email}")
    print(f"🎯 Match Score: {best_report.overall_score:.1f}%")
    
    print("\nStrengths:")
    # Get skills analysis
    skills_data = best_report.sub_scores.get('skills', {})
    if isinstance(skills_data, dict):
        matched_count = skills_data.get('matched_skills', 0)
        missing_count = skills_data.get('missing_skills', 0)
        print(f"  • Matched {matched_count} required skills")
        if missing_count > 0:
            print(f"  • Missing {missing_count} required skills")
    
    # Get experience analysis
    exp_data = best_report.sub_scores.get('experience', {})
    if isinstance(exp_data, dict):
        years_exp = exp_data.get('years_experience', 0)
        required_years = exp_data.get('required_years', 0)
        if years_exp >= required_years:
            print(f"  • {years_exp:.1f} years experience (exceeds {required_years:.1f} year requirement)")
        else:
            print(f"  • {years_exp:.1f} years experience ({required_years - years_exp:.1f} years below requirement)")
    
    print(f"\nTop Recommendations for {best_candidate.contact_info.name}:")
    for i, rec in enumerate(best_report.recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    return reports


def demo_performance_monitoring(engine):
    """Demonstrate performance monitoring capabilities."""
    print_separator("PERFORMANCE MONITORING DEMO")
    
    # Get comprehensive performance stats
    stats = engine.get_performance_stats()
    
    print_subsection("ENGINE PERFORMANCE STATISTICS")
    
    print(f"Total Calculations: {stats['total_calculations']}")
    print(f"Single Calculations: {stats['single_calculations']}")
    print(f"Batch Calculations: {stats['batch_calculations']}")
    print(f"Average Calculation Time: {stats['average_calculation_time']:.3f} seconds")
    print(f"Total Calculation Time: {stats['total_calculation_time']:.3f} seconds")
    print(f"Error Count: {stats['error_count']}")
    
    print_subsection("CACHING PERFORMANCE")
    
    cache_stats = stats.get('cache_statistics', {})
    print(f"Cache Enabled: {cache_stats.get('cache_enabled', False)}")
    print(f"Cache Size: {cache_stats.get('cache_size', 0)} entries")
    print(f"Cache Size Limit: {cache_stats.get('cache_size_limit', 0)}")
    print(f"Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
    
    print_subsection("COMPONENT USAGE")
    
    component_usage = stats.get('component_usage', {})
    for component, count in component_usage.items():
        print(f"{component.title()}: {count} calculations")
    
    print_subsection("COMPONENT PERFORMANCE")
    
    component_stats = stats.get('component_statistics', {})
    
    # Hybrid calculator stats
    if 'hybrid_calculator' in component_stats:
        hybrid_stats = component_stats['hybrid_calculator']
        print(f"\nHybrid Calculator:")
        print(f"  Total Calculations: {hybrid_stats.get('total_calculations', 0)}")
        print(f"  Dynamic Weight Adjustments: {hybrid_stats.get('dynamic_weight_adjustments', 0)}")
        print(f"  Average TF-IDF Weight: {hybrid_stats.get('average_tfidf_weight', 0):.2f}")
        print(f"  Average SBERT Weight: {hybrid_stats.get('average_sbert_weight', 0):.2f}")
    
    # Skills scorer stats
    if 'skills_scorer' in component_stats:
        skills_stats = component_stats['skills_scorer']
        print(f"\nSkills Scorer:")
        print(f"  Total Comparisons: {skills_stats.get('total_comparisons', 0)}")
        print(f"  Exact Matches: {skills_stats.get('exact_matches', 0)}")
        print(f"  Fuzzy Matches: {skills_stats.get('fuzzy_matches', 0)}")
        print(f"  Exact Match Rate: {skills_stats.get('exact_match_rate', 0):.1%}")


def demo_similarity_breakdown(engine, report):
    """Demonstrate detailed similarity breakdown analysis."""
    print_separator("SIMILARITY BREAKDOWN ANALYSIS")
    
    breakdown = engine.get_similarity_breakdown(report)
    
    print_subsection("SCORE COMPOSITION")
    
    print(f"Overall Score: {breakdown['overall_score']:.1f}%")
    
    print("\nComponent Contributions:")
    weighted_contributions = breakdown['score_components'].get('weighted_contributions', {})
    component_weights = breakdown['score_components'].get('component_weights', {})
    
    for component, contribution in weighted_contributions.items():
        weight = component_weights.get(component, 0)
        raw_score = breakdown['score_components']['component_scores'].get(component, 0)
        print(f"  {component.title()}: {raw_score:.1f}% × {weight:.1f} weight = {contribution:.1f} points")
    
    print_subsection("STRENGTHS & WEAKNESSES")
    
    strengths = breakdown.get('key_strengths', [])
    weaknesses = breakdown.get('key_weaknesses', [])
    
    if strengths:
        print("🟢 Key Strengths:")
        for strength in strengths:
            print(f"  • {strength}")
    
    if weaknesses:
        print("\n🔴 Areas for Improvement:")
        for weakness in weaknesses:
            print(f"  • {weakness}")
    
    print_subsection("IMPROVEMENT RECOMMENDATIONS")
    
    improvements = breakdown.get('improvement_areas', [])
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")
    
    print_subsection("CONFIDENCE INDICATORS")
    
    confidence = breakdown.get('confidence_indicators', {})
    print(f"Calculation Time: {confidence.get('calculation_time', 0):.3f} seconds")
    print(f"Components Calculated: {confidence.get('components_calculated', 0)}")
    print(f"Sub-scores Available: {confidence.get('sub_scores_available', 0)}")
    print(f"Cache Used: {confidence.get('cache_used', False)}")


def demo_caching_benefits(engine):
    """Demonstrate caching performance benefits."""
    print_separator("CACHING PERFORMANCE DEMO")
    
    resume = create_sample_resume_1()
    job = create_sample_job_description()
    resume_text = "Python developer with Django experience"
    job_text = "Senior Python developer position"
    
    print("Testing caching performance benefits...")
    
    # First calculation (cache miss)
    print("\n1. First calculation (cache miss):")
    start_time = time.time()
    report1 = engine.calculate_comprehensive_similarity(
        resume=resume,
        job_description=job,
        resume_text=resume_text,
        job_text=job_text,
        include_sub_scores=True
    )
    time1 = time.time() - start_time
    print(f"   Time: {time1:.3f} seconds")
    print(f"   Score: {report1.overall_score:.1f}%")
    
    # Second calculation (cache hit)
    print("\n2. Second calculation (cache hit):")
    start_time = time.time()
    report2 = engine.calculate_comprehensive_similarity(
        resume=resume,
        job_description=job,
        resume_text=resume_text,
        job_text=job_text,
        include_sub_scores=True
    )
    time2 = time.time() - start_time
    print(f"   Time: {time2:.3f} seconds")
    print(f"   Score: {report2.overall_score:.1f}%")
    
    # Performance improvement
    if time1 > 0:
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n📈 Performance Improvement:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time Saved: {(time1 - time2)*1000:.1f} milliseconds")
    
    # Cache statistics
    stats = engine.get_performance_stats()
    cache_stats = stats.get('cache_statistics', {})
    print(f"\n📊 Cache Statistics:")
    print(f"   Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
    print(f"   Cache Size: {cache_stats.get('cache_size', 0)} entries")


def main():
    """Run the comprehensive SimilarityEngine demo."""
    print("🚀 SIMILARITY ENGINE COMPREHENSIVE DEMO")
    print("This demo showcases all major features of the SimilarityEngine")
    
    try:
        # Demo 1: Single similarity calculation
        engine, sample_report = demo_single_similarity_calculation()
        
        # Demo 2: Batch processing
        batch_reports = demo_batch_processing(engine)
        
        # Demo 3: Performance monitoring
        demo_performance_monitoring(engine)
        
        # Demo 4: Similarity breakdown
        demo_similarity_breakdown(engine, sample_report)
        
        # Demo 5: Caching benefits
        demo_caching_benefits(engine)
        
        print_separator("DEMO COMPLETED SUCCESSFULLY")
        print("✅ All SimilarityEngine features demonstrated successfully!")
        print("\nKey Features Showcased:")
        print("  • Comprehensive similarity calculation with detailed reporting")
        print("  • Multi-component scoring (TF-IDF, SBERT, Skills, Experience)")
        print("  • Batch processing with progress tracking")
        print("  • Performance monitoring and statistics")
        print("  • Intelligent caching for improved performance")
        print("  • Detailed analysis breakdown and recommendations")
        print("  • Component integration and orchestration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)