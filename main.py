#!/usr/bin/env python3
"""
Resume-Job Matcher - Comprehensive AI-Powered Resume Analysis

Usage: python main.py <resume_file.pdf>

The job description is read from 'job_description.txt' in the current directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from resume_parser import ResumeParser, SuggestionEngine
from resume_parser.similarity_engine import SimilarityEngine
from resume_parser.gap_analysis import SkillsGapAnalyzer, ExperienceGapAnalyzer, EducationGapAnalyzer, GapAnalysisResult
from job_parser import JobDescriptionParser


JOB_DESCRIPTION_FILE = "job_description.txt"


def print_header(title: str, char: str = "=") -> None:
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f" {title} ".center(80))
    print(f"{char * 80}\n")


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{title}")
    print("-" * 80)


def print_progress(message: str) -> None:
    """Print progress message."""
    print(f"🔄 {message}")


def main():
    """Main entry point for resume-job matching."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Resume-Job Matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py resume.pdf
  python main.py my_resume.pdf --output results.json
  python main.py resume.pdf --job custom_job.txt

The default job description is read from 'job_description.txt'
        """
    )
    
    parser.add_argument(
        'resume',
        help='Path to resume file (PDF, DOCX, or TXT)'
    )
    
    parser.add_argument(
        '--job',
        default=JOB_DESCRIPTION_FILE,
        help=f'Path to job description file (default: {JOB_DESCRIPTION_FILE})'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Save detailed results to JSON file'
    )
    
    parser.add_argument(
        '--no-suggestions',
        action='store_true',
        help='Skip generating improvement suggestions (faster)'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    resume_path = Path(args.resume)
    job_path = Path(args.job)
    
    if not resume_path.exists():
        print(f"❌ Error: Resume file '{args.resume}' not found!", file=sys.stderr)
        sys.exit(1)
    
    if not job_path.exists():
        print(f"❌ Error: Job description file '{args.job}' not found!", file=sys.stderr)
        print(f"💡 Tip: Create a '{JOB_DESCRIPTION_FILE}' file with the job posting", file=sys.stderr)
        sys.exit(1)
    
    try:
        print_header("🚀 RESUME-JOB MATCHER - AI-POWERED ANALYSIS")
        print(f"📄 Resume: {args.resume}")
        print(f"💼 Job Description: {args.job}")
        
        # Initialize components
        print_progress("Loading AI models and initializing parsers...")
        resume_parser = ResumeParser()
        job_parser = JobDescriptionParser()
        similarity_engine = SimilarityEngine()
        
        # Parse resume
        print_progress("Parsing resume...")
        resume = resume_parser.parse_resume(str(resume_path))
        
        # Read resume text
        try:
            with open(resume_path, 'r', encoding='utf-8', errors='ignore') as f:
                resume_text = f.read()
        except:
            resume_text = ""
        
        # Parse job description
        print_progress("Parsing job description...")
        with open(job_path, 'r', encoding='utf-8') as f:
            job_text = f.read()
        
        job = job_parser.parse_job_description(job_text)
        
        # Calculate similarity
        print_progress("Calculating comprehensive similarity score...")
        similarity_result = similarity_engine.calculate_comprehensive_similarity(
            resume=resume,
            job_description=job,
            resume_text=resume_text,
            job_text=job_text,
            include_sub_scores=True
        )
        
        # Display results
        print_header("📊 MATCH ANALYSIS RESULTS")
        
        # Overall score with visual indicator
        score = similarity_result.overall_score
        if score >= 80:
            emoji, rating, color = "🟢", "EXCELLENT MATCH", "green"
        elif score >= 60:
            emoji, rating, color = "🟡", "GOOD MATCH", "yellow"
        elif score >= 40:
            emoji, rating, color = "🟠", "MODERATE MATCH", "orange"
        else:
            emoji, rating, color = "🔴", "WEAK MATCH", "red"
        
        print(f"{emoji} Overall Match Score: {score:.1f}% - {rating}\n")
        
        # Component breakdown with progress bars
        print_section("📈 COMPONENT BREAKDOWN")
        for component, comp_score in similarity_result.component_scores.items():
            bar_length = int(comp_score / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{component.title():<20} {bar} {comp_score:>6.1f}%")
        
        # Skills analysis
        print_section("🛠️  SKILLS ANALYSIS")
        resume_skills = set(s.lower() for s in resume.skills)
        job_skills = set(s.lower() for s in job.skills_required)
        matched_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills
        
        match_rate = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0
        
        print(f"Skills Match Rate: {match_rate:.1f}% ({len(matched_skills)}/{len(job_skills)})")
        
        print(f"\n✅ Matched Skills ({len(matched_skills)}):")
        for i, skill in enumerate(sorted(matched_skills)[:15], 1):
            print(f"   {i:2d}. {skill.title()}")
        if len(matched_skills) > 15:
            print(f"   ... and {len(matched_skills) - 15} more")
        
        if missing_skills:
            print(f"\n❌ Missing Skills ({len(missing_skills)}):")
            for i, skill in enumerate(sorted(missing_skills)[:15], 1):
                print(f"   {i:2d}. {skill.title()}")
            if len(missing_skills) > 15:
                print(f"   ... and {len(missing_skills) - 15} more")
        
        # Experience analysis
        print_section("💼 EXPERIENCE ANALYSIS")
        total_exp_months = sum(exp.duration_months for exp in resume.experience)
        total_exp_years = total_exp_months / 12
        
        print(f"Total Experience: {total_exp_years:.1f} years")
        print(f"Required Level: {job.experience_level.title()}")
        print(f"\nPositions Held:")
        for i, exp in enumerate(resume.experience[:5], 1):
            duration = exp.duration_months / 12
            print(f"   {i}. {exp.job_title} at {exp.company} ({duration:.1f} years)")
        
        # Education analysis
        print_section("🎓 EDUCATION")
        if resume.education:
            for edu in resume.education:
                print(f"• {edu.degree} in {edu.major}")
                print(f"  {edu.institution}")
                if edu.graduation_date:
                    print(f"  Graduated: {edu.graduation_date}")
                if edu.gpa:
                    print(f"  GPA: {edu.gpa}")
        else:
            print("No education information found in resume")
        
        # Certifications
        if resume.certifications:
            print_section("📜 CERTIFICATIONS")
            for cert in resume.certifications:
                print(f"• {cert.name}")
                if cert.issuer:
                    print(f"  Issued by: {cert.issuer} ({cert.issue_date})")
        
        # Top recommendations
        print_section("💡 INITIAL RECOMMENDATIONS")
        for i, rec in enumerate(similarity_result.recommendations[:10], 1):
            print(f"{i:2d}. {rec}")
        
        # Generate comprehensive suggestions if not disabled
        if not args.no_suggestions:
            print_progress("Generating comprehensive improvement suggestions...")
            
            # Perform gap analysis
            skills_analyzer = SkillsGapAnalyzer()
            skills_gaps = skills_analyzer.analyze_skills_gaps(resume, job)
            
            experience_analyzer = ExperienceGapAnalyzer()
            experience_gap = experience_analyzer.analyze_experience_gap(resume, job)
            
            education_analyzer = EducationGapAnalyzer()
            education_gap = education_analyzer.analyze_education_gap(resume, job)
            
            gap_analysis = GapAnalysisResult(
                skills_gaps=skills_gaps,
                experience_gap=experience_gap,
                education_gap=education_gap,
                overall_gap_score=1.0 - (score / 100),
                improvement_potential=0.3,
                priority_recommendations=[],
                metadata={}
            )
            
            # Generate suggestions
            suggestion_engine = SuggestionEngine()
            suggestions_result = suggestion_engine.generate_suggestions(
                resume=resume,
                job_description=job,
                gap_analysis=gap_analysis,
                similarity_score=score / 100
            )
            
            # Display quick wins
            if suggestions_result.quick_wins:
                print_section("🚀 QUICK WINS (Easy & High Impact)")
                for i, sugg in enumerate(suggestions_result.quick_wins[:5], 1):
                    print(f"\n{i}. {sugg.title}")
                    print(f"   Impact: {sugg.impact_score:.0%} | Effort: {sugg.implementation_effort} | Time: {sugg.timeframe}")
                    print(f"   {sugg.description}")
                    if sugg.specific_actions:
                        print(f"   Actions:")
                        for action in sugg.specific_actions[:3]:
                            print(f"   • {action}")
            
            # Display top priority suggestions
            print_section("⭐ TOP PRIORITY IMPROVEMENTS")
            for i, sugg in enumerate(suggestions_result.prioritized_suggestions[:8], 1):
                print(f"\n{i}. [{sugg.priority.value.upper()}] {sugg.title}")
                print(f"   Category: {sugg.category.value.title()} | Impact: {sugg.impact_score:.0%} | Feasibility: {sugg.feasibility_score:.0%}")
                print(f"   {sugg.description}")
                if sugg.specific_actions:
                    print(f"   Key Actions:")
                    for action in sugg.specific_actions[:2]:
                        print(f"   • {action}")
            
            # Summary with improvement potential
            print_header("📋 IMPROVEMENT SUMMARY")
            print(f"Current Match Score: {score:.1f}%")
            print(f"Improvement Potential: +{suggestions_result.overall_improvement_potential:.0%}")
            projected_score = min(100, score + (suggestions_result.overall_improvement_potential * 100))
            print(f"Projected Score: {projected_score:.1f}%")
            print(f"\nTotal Suggestions Generated: {len(suggestions_result.suggestions)}")
            print(f"Quick Wins Identified: {len(suggestions_result.quick_wins)}")
            print(f"Long-term Improvements: {len(suggestions_result.long_term_improvements)}")
            
            # Save detailed results if requested
            if args.output:
                output_data = {
                    'overall_score': score,
                    'rating': rating,
                    'component_scores': similarity_result.component_scores,
                    'skills_analysis': {
                        'matched': list(matched_skills),
                        'missing': list(missing_skills),
                        'match_rate': match_rate
                    },
                    'experience': {
                        'years': total_exp_years,
                        'required_level': job.experience_level,
                        'positions': [
                            {
                                'title': exp.job_title,
                                'company': exp.company,
                                'duration_years': exp.duration_months / 12
                            }
                            for exp in resume.experience
                        ]
                    },
                    'education': [
                        {
                            'degree': edu.degree,
                            'major': edu.major,
                            'institution': edu.institution,
                            'graduation_date': edu.graduation_date
                        }
                        for edu in resume.education
                    ],
                    'recommendations': similarity_result.recommendations,
                    'improvement_potential': suggestions_result.overall_improvement_potential,
                    'projected_score': projected_score,
                    'suggestions': {
                        'quick_wins': [
                            {
                                'title': s.title,
                                'category': s.category.value,
                                'impact': s.impact_score,
                                'effort': s.implementation_effort,
                                'timeframe': s.timeframe,
                                'description': s.description,
                                'actions': s.specific_actions
                            }
                            for s in suggestions_result.quick_wins
                        ],
                        'top_priorities': [
                            {
                                'title': s.title,
                                'category': s.category.value,
                                'priority': s.priority.value,
                                'impact': s.impact_score,
                                'feasibility': s.feasibility_score,
                                'description': s.description,
                                'actions': s.specific_actions,
                                'rationale': s.rationale
                            }
                            for s in suggestions_result.prioritized_suggestions[:10]
                        ]
                    }
                }
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Detailed results saved to: {args.output}")
        
        print_header("✨ ANALYSIS COMPLETE")
        print("Thank you for using Resume-Job Matcher!")
        print("Good luck with your application! 🍀\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()