#!/usr/bin/env python3
"""
Test script to validate job description parser against test dataset
"""

import json
from job_parser.parser import JobDescriptionParser
from job_parser.interfaces import JobDescription

def load_test_data(filename):
    """Load test job descriptions from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_parser_output(test_case, parsed_result):
    """Compare parser output with expected results"""
    expected = test_case['expected_output']
    
    # Extract skills from parsed result
    technical_skills = [skill.name.lower() for skill in parsed_result.technical_skills]
    soft_skills = [skill.name.lower() for skill in parsed_result.soft_skills]
    
    # Compare technical skills
    expected_technical = [skill.lower() for skill in expected['technical_skills']]
    technical_matches = sum(1 for skill in expected_technical if any(skill in ts for ts in technical_skills))
    technical_precision = technical_matches / len(expected_technical) if expected_technical else 0
    
    # Compare soft skills
    expected_soft = [skill.lower() for skill in expected['soft_skills']]
    soft_matches = sum(1 for skill in expected_soft if any(skill in ss for ss in soft_skills))
    soft_precision = soft_matches / len(expected_soft) if expected_soft else 0
    
    return {
        'technical_precision': technical_precision,
        'soft_precision': soft_precision,
        'technical_matches': technical_matches,
        'soft_matches': soft_matches,
        'expected_technical_count': len(expected_technical),
        'expected_soft_count': len(expected_soft),
        'found_technical_count': len(technical_skills),
        'found_soft_count': len(soft_skills)
    }

def main():
    """Run validation tests"""
    parser = JobDescriptionParser()
    
    # Load test datasets
    test_files = ['test_job_descriptions.json', 'test_job_descriptions_extended.json']
    
    all_results = []
    
    for test_file in test_files:
        try:
            test_data = load_test_data(test_file)
            print(f"\n=== Testing with {test_file} ===")
            
            for test_case in test_data:
                print(f"\nTesting: {test_case['title']} ({test_case['experience_level']})")
                
                # Create JobDescription object
                job_desc = JobDescription(
                    title=test_case['title'],
                    company=test_case['company'],
                    description=test_case['description']
                )
                
                # Parse the job description
                try:
                    result = parser.parse(job_desc)
                    validation = validate_parser_output(test_case, result)
                    
                    print(f"  Technical Skills: {validation['technical_matches']}/{validation['expected_technical_count']} "
                          f"({validation['technical_precision']:.2%})")
                    print(f"  Soft Skills: {validation['soft_matches']}/{validation['expected_soft_count']} "
                          f"({validation['soft_precision']:.2%})")
                    
                    all_results.append({
                        'job_id': test_case['id'],
                        'title': test_case['title'],
                        'validation': validation
                    })
                    
                except Exception as e:
                    print(f"  Error parsing job description: {e}")
                    
        except FileNotFoundError:
            print(f"Test file {test_file} not found, skipping...")
    
    # Summary statistics
    if all_results:
        avg_technical = sum(r['validation']['technical_precision'] for r in all_results) / len(all_results)
        avg_soft = sum(r['validation']['soft_precision'] for r in all_results) / len(all_results)
        
        print(f"\n=== SUMMARY ===")
        print(f"Total test cases: {len(all_results)}")
        print(f"Average technical skills precision: {avg_technical:.2%}")
        print(f"Average soft skills precision: {avg_soft:.2%}")
        print(f"Overall average precision: {(avg_technical + avg_soft) / 2:.2%}")

if __name__ == "__main__":
    main()