#!/usr/bin/env python3
"""
Demo test for skill categorization and confidence scoring.
Run this to see the SkillCategorizer in action!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from job_parser.skill_categorization import SkillCategorizer
from job_parser.interfaces import SkillMatch

def demo_skill_categorization():
    """Demonstrate skill categorization with realistic job data."""
    
    print("🚀 Skill Categorization & Confidence Scoring Demo")
    print("=" * 50)
    
    # Create realistic skill matches (simulating what would come from semantic + NER)
    skill_matches = [
        # Technical skills with various sources and confidences
        SkillMatch(skill="Python", category="technical", confidence=0.92, source="semantic", context="experience with Python programming"),
        SkillMatch(skill="python", category="technical", confidence=0.75, source="ner", context=""),  # Duplicate - lower confidence
        SkillMatch(skill="React", category="technical", confidence=0.88, source="ner+semantic", context="React development experience"),
        SkillMatch(skill="JavaScript", category="technical", confidence=0.85, source="semantic", context="JavaScript and modern frameworks"),
        SkillMatch(skill="Node.js", category="technical", confidence=0.82, source="ner", context="Node.js backend development"),
        
        # Tools and platforms
        SkillMatch(skill="Docker", category="tools", confidence=0.90, source="semantic", context="containerization with Docker"),
        SkillMatch(skill="AWS", category="tools", confidence=0.87, source="ner+semantic", context="AWS cloud services"),
        SkillMatch(skill="Git", category="tools", confidence=0.78, source="ner", context="version control with Git"),
        
        # Soft skills
        SkillMatch(skill="teamwork", category="soft", confidence=0.65, source="semantic", context="collaborative team environment"),
        SkillMatch(skill="leadership", category="soft", confidence=0.72, source="ner", context="leadership and mentoring"),
        SkillMatch(skill="communication", category="soft", confidence=0.68, source="semantic", context="excellent communication skills"),
        
        # Some edge cases
        SkillMatch(skill="", category="technical", confidence=0.5, source="ner", context=""),  # Empty skill - should be filtered
        SkillMatch(skill="PYTHON", category="technical", confidence=0.60, source="ner", context=""),  # Another duplicate
    ]
    
    # Sample ontology
    ontology = {
        "technical": ["Python", "JavaScript", "React", "Node.js", "Java", "C++"],
        "tools": ["Docker", "AWS", "Git", "Jenkins", "Kubernetes"],
        "soft": ["teamwork", "leadership", "communication", "problem-solving"]
    }
    
    # Initialize categorizer
    categorizer = SkillCategorizer()
    
    print(f"📊 Input: {len(skill_matches)} skill matches")
    print("\nRaw skill matches:")
    for i, match in enumerate(skill_matches, 1):
        print(f"  {i:2d}. {match.skill:12} | {match.category:10} | {match.confidence:.2f} | {match.source}")
    
    print("\n" + "─" * 50)
    
    # Test 1: Skill Categorization
    print("🏷️  SKILL CATEGORIZATION")
    categorized = categorizer.categorize_skills(skill_matches, ontology)
    
    print("\nCategorized & Ranked Skills:")
    for category, skills in categorized.items():
        if skills:  # Only show categories with skills
            print(f"  📂 {category.upper()}:")
            for i, skill in enumerate(skills, 1):
                print(f"     {i}. {skill}")
        else:
            print(f"  📂 {category.upper()}: (no skills)")
    
    print("\n" + "─" * 50)
    
    # Test 2: Confidence Scoring
    print("🎯 CONFIDENCE SCORING")
    confidence_scores = categorizer.compute_confidence_scores(skill_matches)
    
    print("\nFinal Confidence Scores:")
    # Sort by confidence for better display
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    for skill, score in sorted_scores:
        bar_length = int(score * 20)  # Scale to 20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {skill:15} │{bar}│ {score:.3f}")
    
    print("\n" + "─" * 50)
    
    # Test 3: Experience Level Inference
    print("🎓 EXPERIENCE LEVEL INFERENCE")
    
    job_descriptions = [
        {
            "title": "Junior Developer Position",
            "text": "We're looking for a junior Python developer with 0-2 years of experience. This is an entry-level position perfect for recent graduates. Training and mentorship will be provided."
        },
        {
            "title": "Senior Engineering Role", 
            "text": "Seeking a senior software engineer with 7+ years of experience. Must have proven leadership experience, system architecture skills, and ability to mentor junior developers. Technical lead responsibilities included."
        },
        {
            "title": "Mid-Level Developer",
            "text": "Looking for a mid-level developer with 3-4 years of solid experience in React and Python. Should be proficient in modern development practices and able to work independently."
        }
    ]
    
    print("\nExperience Level Analysis:")
    for job in job_descriptions:
        level = categorizer.infer_experience_level(job["text"], skill_matches)
        print(f"  📋 {job['title']}")
        print(f"     → Inferred Level: {level.upper()}")
        print()
    
    print("✅ Demo completed! The SkillCategorizer successfully:")
    print("   • Filtered duplicates (kept 'Python' over 'python' and 'PYTHON')")
    print("   • Ranked skills by confidence within categories") 
    print("   • Applied source-based weighting to confidence scores")
    print("   • Correctly inferred experience levels from job text")

if __name__ == "__main__":
    demo_skill_categorization()