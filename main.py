#!/usr/bin/env python3
"""Resume-Job Matcher CLI"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from resume_parser import ResumeParser
from job_parser import JobDescriptionParser
from resume_parser.similarity_engine import SimilarityEngine


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_resume(file_path: str) -> Dict[str, Any]:
    """Parse resume file."""
    parser = ResumeParser()
    result = parser.parse_resume(file_path)
    return result.to_json()


def parse_job(text: str) -> Dict[str, Any]:
    """Parse job description text."""
    parser = JobDescriptionParser()
    result = parser.parse_job_description(text)
    return result.to_json()


def calculate_similarity(resume_file: str, job_text: str) -> Dict[str, Any]:
    """Calculate similarity between resume and job."""
    resume_parser = ResumeParser()
    job_parser = JobDescriptionParser()
    similarity_engine = SimilarityEngine()
    
    resume = resume_parser.parse_resume(resume_file)
    job = job_parser.parse_job_description(job_text)
    
    with open(resume_file, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    result = similarity_engine.calculate_comprehensive_similarity(
        resume=resume,
        job_description=job,
        resume_text=resume_text,
        job_text=job_text,
        include_sub_scores=True
    )
    
    return {
        'overall_score': result.overall_score,
        'component_scores': result.component_scores,
        'sub_scores': result.sub_scores,
        'recommendations': result.recommendations
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Resume-Job Matcher")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Parse resume command
    resume_parser = subparsers.add_parser('parse-resume', help='Parse resume file')
    resume_parser.add_argument('file', help='Resume file path')
    resume_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Parse job command
    job_parser = subparsers.add_parser('parse-job', help='Parse job description')
    job_parser.add_argument('file', help='Job description text file')
    job_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare resume to job')
    compare_parser.add_argument('resume', help='Resume file path')
    compare_parser.add_argument('job', help='Job description text file')
    compare_parser.add_argument('-o', '--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'parse-resume':
            result = parse_resume(args.file)
            
        elif args.command == 'parse-job':
            with open(args.file, 'r', encoding='utf-8') as f:
                job_text = f.read()
            result = parse_job(job_text)
            
        elif args.command == 'compare':
            with open(args.job, 'r', encoding='utf-8') as f:
                job_text = f.read()
            result = calculate_similarity(args.resume, job_text)
        
        if args.output:
            save_json(result, Path(args.output))
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()