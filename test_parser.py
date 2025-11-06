#!/usr/bin/env python3
"""
Simple test script for the Job Description Parser.

This script demonstrates different ways to test the parser with various configurations.
"""

import json
import time
from job_parser import JobDescriptionParser
from job_parser.config import ParserConfig


def test_single_job():
    """Test parsing a single job description."""
    print("=" * 60)
    print("TEST 1: Single Job Description Parsing")
    print("=" * 60)
    
    # Sample job description
    job_text = """
    Senior Python Developer - Remote
    
    We are looking for an experienced Python developer to join our team.
    
    Requirements:
    - 5+ years of Python development experience
    - Experience with Django and Flask frameworks
    - Knowledge of PostgreSQL and Redis
    - Familiarity with AWS cloud services
    - Experience with Docker and Kubernetes
    - Strong problem-solving skills
    - Excellent communication abilities
    
    Nice to have:
    - Machine learning experience with scikit-learn
    - React.js frontend development
    - CI/CD pipeline experience with Jenkins
    """
    
    # Initialize parser
    parser = JobDescriptionParser()
    
    # Parse the job
    start_time = time.time()
    result = parser.parse_job_description(job_text)
    processing_time = time.time() - start_time
    
    # Display results
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Skills found: {len(result.skills_required)}")
    print(f"Experience level: {result.experience_level}")
    print(f"Tools mentioned: {len(result.tools_mentioned)}")
    
    print("\nTop skills by confidence:")
    for skill in result.skills_required[:10]:
        confidence = result.confidence_scores.get(skill, 0)
        print(f"  - {skill}: {confidence:.3f}")
    
    print("\nSkills by category:")
    for category, skills in result.categories.items():
        if skills:
            print(f"  {category.title()}: {', '.join(skills[:5])}")
    
    return result


def test_batch_processing():
    """Test batch processing with multiple job descriptions."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Processing")
    print("=" * 60)
    
    # Load test data
    try:
        with open('test_job_descriptions.json', 'r') as f:
            test_jobs = json.load(f)
    except FileNotFoundError:
        print("test_job_descriptions.json not found. Skipping batch test.")
        return
    
    # Extract job texts
    job_texts = [job['description'] for job in test_jobs]
    
    # Initialize parser
    parser = JobDescriptionParser()
    
    # Test batch processing
    start_time = time.time()
    results = parser.parse_job_descriptions_batch(job_texts)
    processing_time = time.time() - start_time
    
    # Display results
    print(f"Processed {len(results)} jobs in {processing_time:.3f} seconds")
    print(f"Throughput: {len(results)/processing_time:.1f} jobs/second")
    
    # Summary statistics
    total_skills = sum(len(r.skills_required) for r in results)
    experience_levels = [r.experience_level for r in results]
    
    print(f"Total skills extracted: {total_skills}")
    print(f"Average skills per job: {total_skills/len(results):.1f}")
    
    # Experience level distribution
    from collections import Counter
    exp_counts = Counter(experience_levels)
    print("\nExperience level distribution:")
    for level, count in exp_counts.items():
        print(f"  {level}: {count}")
    
    return results


def test_performance_optimization():
    """Test performance optimization features."""
    print("\n" + "=" * 60)
    print("TEST 3: Performance Optimization")
    print("=" * 60)
    
    # Initialize parser
    parser = JobDescriptionParser()
    
    # Test optimization for batch processing
    print("Optimizing for batch processing...")
    parser.optimize_for_batch_processing(expected_batch_size=10, max_memory_mb=256.0)
    
    # Get performance statistics
    stats = parser.get_performance_stats()
    
    print("Performance Statistics:")
    print(f"  Batch optimization: {stats['optimization']['is_optimized_for_batch']}")
    print(f"  Precomputed embeddings: {stats['optimization']['ontology_embeddings_precomputed']}")
    
    if 'semantic_matcher' in stats['components']:
        semantic_stats = stats['components']['semantic_matcher']
        print(f"  Model loaded: {semantic_stats.get('model_loaded', False)}")
        print(f"  Model warmed up: {semantic_stats.get('model_warmed_up', False)}")
        print(f"  Cache hit rate: {semantic_stats.get('cache_hit_rate', 0):.3f}")
    
    return stats


def test_custom_configuration():
    """Test parser with custom configuration."""
    print("\n" + "=" * 60)
    print("TEST 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = ParserConfig()
    config.similarity_threshold = 0.6  # Lower threshold for more matches
    config.max_text_length = 1000
    
    print(f"Custom config - Threshold: {config.similarity_threshold}")
    print(f"Custom config - Max length: {config.max_text_length}")
    
    # Initialize parser with custom config
    parser = JobDescriptionParser(config)
    
    # Test with a simple job description
    job_text = "Python developer with machine learning experience using TensorFlow and scikit-learn."
    
    result = parser.parse_job_description(job_text)
    
    print(f"Skills found with lower threshold: {len(result.skills_required)}")
    print(f"Skills: {', '.join(result.skills_required)}")
    
    return result


def test_json_output():
    """Test JSON output formatting."""
    print("\n" + "=" * 60)
    print("TEST 5: JSON Output")
    print("=" * 60)
    
    parser = JobDescriptionParser()
    
    job_text = "Senior React developer with TypeScript and Node.js experience."
    
    # Get JSON output
    json_output = parser.parse_job_description_to_json(job_text)
    
    print("JSON Output (first 500 characters):")
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
    
    # Parse and validate JSON
    try:
        parsed_json = json.loads(json_output)
        print(f"\nJSON validation: ✓ Valid")
        print(f"Skills in JSON: {len(parsed_json.get('skills_required', []))}")
    except json.JSONDecodeError as e:
        print(f"\nJSON validation: ✗ Invalid - {e}")
    
    return json_output


def main():
    """Run all tests."""
    print("Job Description Parser - Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_single_job()
        test_batch_processing()
        test_performance_optimization()
        test_custom_configuration()
        test_json_output()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()