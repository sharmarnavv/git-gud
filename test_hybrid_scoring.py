#!/usr/bin/env python3
"""
Test script to verify the hybrid scoring algorithm implementation.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_parser.similarity_engine import HybridSimilarityCalculator

def test_hybrid_scoring():
    """Test the hybrid scoring algorithm implementation."""
    
    # Sample resume text
    resume_text = """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Machine learning and data science projects
    - AWS cloud infrastructure management
    - Team leadership and project management
    
    Skills:
    - Python, JavaScript, SQL
    - TensorFlow, PyTorch
    - Docker, Kubernetes
    - Communication, problem-solving
    
    Education:
    - BS Computer Science, MIT
    """
    
    # Sample job description
    job_text = """
    Senior Software Engineer Position
    
    We are looking for an experienced software engineer with:
    - 3+ years Python development experience
    - Machine learning and AI experience
    - Cloud platforms (AWS preferred)
    - Strong communication skills
    - Leadership experience
    
    Required Skills:
    - Python, SQL, JavaScript
    - Machine learning frameworks (TensorFlow/PyTorch)
    - Docker containerization
    - Team collaboration
    """
    
    try:
        print("Initializing Hybrid Similarity Calculator...")
        calculator = HybridSimilarityCalculator(
            default_tfidf_weight=0.4,
            default_sbert_weight=0.6,
            enable_dynamic_weighting=True,
            score_calibration=True
        )
        
        print("Calculating hybrid similarity...")
        result = calculator.calculate_similarity(resume_text, job_text)
        
        print(f"\n=== Hybrid Similarity Results ===")
        print(f"Overall Similarity Score: {result['similarity_score']:.1f}%")
        
        component_scores = result['component_scores']
        print(f"\nComponent Scores:")
        print(f"  TF-IDF Score: {component_scores['tfidf_score']:.1f}% (weight: {component_scores['tfidf_weight']:.3f})")
        print(f"  SBERT Score: {component_scores['sbert_score']:.1f}% (weight: {component_scores['sbert_weight']:.3f})")
        
        metadata = result['metadata']
        print(f"\nMetadata:")
        print(f"  Dynamic weighting used: {metadata['dynamic_weighting_used']}")
        print(f"  Score calibration applied: {metadata['score_calibration_applied']}")
        print(f"  Resume length: {metadata['resume_length']} words")
        print(f"  Job length: {metadata['job_length']} words")
        
        content_analysis = metadata['content_analysis']
        print(f"\nContent Analysis:")
        print(f"  Technical keyword ratio: {content_analysis['technical_keyword_ratio']:.3f}")
        print(f"  Soft skill keyword ratio: {content_analysis['soft_skill_keyword_ratio']:.3f}")
        
        # Test with custom weights
        print(f"\n=== Testing Custom Weights ===")
        custom_result = calculator.calculate_similarity(
            resume_text, job_text, 
            custom_weights={'tfidf': 0.7, 'sbert': 0.3}
        )
        print(f"Custom weights similarity: {custom_result['similarity_score']:.1f}%")
        
        # Get performance stats
        stats = calculator.get_performance_stats()
        print(f"\n=== Performance Stats ===")
        print(f"Total calculations: {stats['total_calculations']}")
        print(f"Dynamic weight adjustments: {stats['dynamic_weight_adjustments']}")
        print(f"Calibration applications: {stats['calibration_applications']}")
        
        print("\n✅ Hybrid scoring algorithm test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_scoring()
    sys.exit(0 if success else 1)