"""
Skill categorization and confidence scoring system.

This module handles the final processing of extracted skills including
categorization, duplicate filtering, and confidence score computation.
"""

from typing import Dict, List

from .interfaces import SkillCategorizerInterface, SkillMatch
from .logging_config import get_logger


class SkillCategorizer(SkillCategorizerInterface):
    """Handles skill categorization and confidence scoring.
    
    This class implements the final processing stage that categorizes
    extracted skills and computes final confidence scores.
    """
    
    def __init__(self):
        """Initialize the skill categorizer."""
        self.logger = get_logger(__name__)
    
    def categorize_skills(self, 
                         skill_matches: List[SkillMatch], 
                         ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Categorize and rank extracted skills.
        
        This method organizes extracted skills by their categories, filters out
        duplicates while preserving the highest confidence scores, and ranks
        skills within each category by descending confidence levels.
        
        Args:
            skill_matches: List of matched skills with confidence scores
            ontology: Skills ontology for categorization validation
            
        Returns:
            Dictionary mapping categories to ranked skill lists
        """
        self.logger.info(f"Categorizing {len(skill_matches)} extracted skills")
        
        if not skill_matches:
            self.logger.warning("No skill matches provided for categorization")
            return {category: [] for category in ontology.keys()}
        
        # Initialize categories from ontology
        categorized_skills = {category: {} for category in ontology.keys()}
        
        # Add any categories found in skill matches that aren't in ontology
        for skill_match in skill_matches:
            if skill_match.category not in categorized_skills:
                categorized_skills[skill_match.category] = {}
                self.logger.debug(f"Added new category from skill matches: {skill_match.category}")
        
        # Process each skill match
        for skill_match in skill_matches:
            skill_name = skill_match.skill.strip()
            category = skill_match.category
            confidence = skill_match.confidence
            
            # Skip empty or invalid skills
            if not skill_name or len(skill_name.strip()) < 2:
                continue
            
            # Normalize skill name for duplicate detection (case-insensitive)
            normalized_skill = skill_name.lower().strip()
            
            # Check if this skill already exists in the category
            existing_skill = None
            existing_confidence = 0.0
            
            # Find existing skill with same normalized name
            for existing_name, existing_conf in categorized_skills[category].items():
                if existing_name.lower().strip() == normalized_skill:
                    existing_skill = existing_name
                    existing_confidence = existing_conf
                    break
            
            # Add or update skill based on confidence
            if existing_skill is None:
                # New skill - add it
                categorized_skills[category][skill_name] = confidence
                self.logger.debug(f"Added new skill: {skill_name} ({category}) - {confidence:.3f}")
            elif confidence > existing_confidence:
                # Higher confidence - replace existing
                del categorized_skills[category][existing_skill]
                categorized_skills[category][skill_name] = confidence
                self.logger.debug(f"Updated skill: {skill_name} ({category}) - {confidence:.3f} (was {existing_confidence:.3f})")
            else:
                # Lower confidence - keep existing
                self.logger.debug(f"Kept existing skill: {existing_skill} ({category}) - {existing_confidence:.3f} (vs {confidence:.3f})")
        
        # Convert to ranked lists (sorted by confidence descending)
        ranked_categories = {}
        total_skills = 0
        
        for category, skills_dict in categorized_skills.items():
            # Sort skills by confidence (descending) and extract skill names
            sorted_skills = sorted(skills_dict.items(), key=lambda x: x[1], reverse=True)
            ranked_categories[category] = [skill_name for skill_name, _ in sorted_skills]
            
            skill_count = len(ranked_categories[category])
            total_skills += skill_count
            
            if skill_count > 0:
                self.logger.info(f"Category '{category}': {skill_count} skills (top: {ranked_categories[category][0]})")
        
        self.logger.info(f"Categorization complete: {total_skills} unique skills across {len(ranked_categories)} categories")
        
        return ranked_categories
    
    def compute_confidence_scores(self, skill_matches: List[SkillMatch]) -> Dict[str, float]:
        """Compute final confidence scores for skills.
        
        This method combines similarity scores and NER probabilities using a weighted
        confidence calculation system. It also considers the source of the match
        and applies appropriate weighting factors.
        
        Args:
            skill_matches: List of matched skills with individual confidences
            
        Returns:
            Dictionary mapping skills to final confidence scores (0.0-1.0)
        """
        self.logger.info(f"Computing confidence scores for {len(skill_matches)} skill matches")
        
        if not skill_matches:
            self.logger.warning("No skill matches provided for confidence scoring")
            return {}
        
        # Confidence weighting factors based on source
        source_weights = {
            'semantic': 0.6,      # Semantic matching is reliable but can be broad
            'ner': 0.4,           # NER is precise but may miss context
            'ner+semantic': 0.8,  # Combined sources get higher weight
            'semantic+ner': 0.8   # Alternative naming for combined
        }
        
        # Base confidence adjustments based on match characteristics
        confidence_adjustments = {
            'exact_match': 0.1,     # Boost for exact ontology matches
            'partial_match': 0.0,   # No adjustment for partial matches
            'context_boost': 0.05,  # Small boost if context is available
            'high_similarity': 0.1, # Boost for very high similarity (>0.9)
            'multiple_sources': 0.15 # Significant boost for multiple source confirmation
        }
        
        final_scores = {}
        processed_skills = {}  # Track skills to handle duplicates
        
        for skill_match in skill_matches:
            skill_name = skill_match.skill.strip()
            base_confidence = skill_match.confidence
            source = skill_match.source
            context = skill_match.context
            
            # Skip invalid skills
            if not skill_name or base_confidence < 0:
                continue
            
            # Normalize skill name for duplicate handling
            normalized_skill = skill_name.lower().strip()
            
            # Apply source weighting
            source_weight = source_weights.get(source, 0.5)  # Default weight for unknown sources
            weighted_confidence = base_confidence * source_weight
            
            # Apply confidence adjustments
            adjustments = 0.0
            
            # High similarity boost
            if base_confidence > 0.9:
                adjustments += confidence_adjustments['high_similarity']
                self.logger.debug(f"High similarity boost for {skill_name}: +{confidence_adjustments['high_similarity']}")
            
            # Context availability boost
            if context and len(context.strip()) > 0:
                adjustments += confidence_adjustments['context_boost']
                self.logger.debug(f"Context boost for {skill_name}: +{confidence_adjustments['context_boost']}")
            
            # Multiple sources boost
            if '+' in source or 'semantic' in source and 'ner' in source:
                adjustments += confidence_adjustments['multiple_sources']
                self.logger.debug(f"Multiple sources boost for {skill_name}: +{confidence_adjustments['multiple_sources']}")
            
            # Calculate final confidence
            final_confidence = min(1.0, weighted_confidence + adjustments)
            
            # Handle duplicates - keep highest confidence
            if normalized_skill in processed_skills:
                existing_skill_name, existing_confidence = processed_skills[normalized_skill]
                if final_confidence > existing_confidence:
                    # Replace with higher confidence version
                    del final_scores[existing_skill_name]
                    final_scores[skill_name] = final_confidence
                    processed_skills[normalized_skill] = (skill_name, final_confidence)
                    self.logger.debug(f"Updated {skill_name}: {final_confidence:.3f} (was {existing_confidence:.3f})")
                else:
                    # Keep existing higher confidence version
                    self.logger.debug(f"Kept existing {existing_skill_name}: {existing_confidence:.3f} (vs {final_confidence:.3f})")
            else:
                # New skill
                final_scores[skill_name] = final_confidence
                processed_skills[normalized_skill] = (skill_name, final_confidence)
                self.logger.debug(f"Added {skill_name}: {final_confidence:.3f} (base: {base_confidence:.3f}, source: {source})")
        
        # Log confidence distribution
        if final_scores:
            confidences = list(final_scores.values())
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            self.logger.info(f"Confidence scoring complete: {len(final_scores)} skills")
            self.logger.info(f"Confidence stats - Avg: {avg_confidence:.3f}, Max: {max_confidence:.3f}, Min: {min_confidence:.3f}")
        
        return final_scores
    
    def infer_experience_level(self, text: str, skill_matches: List[SkillMatch]) -> str:
        """Infer experience level from job description text and extracted skills.
        
        This method analyzes keywords and patterns in the job description text
        to determine the likely experience level (entry, mid, senior) required
        for the position.
        
        Args:
            text: Original job description text
            skill_matches: List of extracted skill matches
            
        Returns:
            Experience level string: 'entry-level', 'mid-level', or 'senior-level'
        """
        self.logger.info("Inferring experience level from job description")
        
        if not text:
            self.logger.warning("No text provided for experience level inference")
            return "mid-level"  # Default fallback
        
        text_lower = text.lower()
        
        # Experience level indicators
        entry_indicators = [
            'entry level', 'entry-level', 'junior', 'graduate', 'new grad',
            'recent graduate', 'internship', 'trainee', '0-2 years',
            'no experience required', 'fresh', 'beginner', 'starting',
            'learn on the job', 'training provided', '0+ years'
        ]
        
        senior_indicators = [
            'senior', 'lead', 'principal', 'architect', 'manager',
            'director', 'head of', 'chief', 'expert', 'specialist',
            '5+ years', '7+ years', '10+ years', 'extensive experience',
            'proven track record', 'leadership', 'mentoring', 'team lead',
            'technical lead', 'staff engineer', 'distinguished'
        ]
        
        mid_indicators = [
            'mid level', 'mid-level', 'intermediate', '2-5 years',
            '3+ years', '4+ years', 'some experience', 'solid experience',
            'proficient', 'competent', 'experienced'
        ]
        
        # Count indicators
        entry_count = sum(1 for indicator in entry_indicators if indicator in text_lower)
        senior_count = sum(1 for indicator in senior_indicators if indicator in text_lower)
        mid_count = sum(1 for indicator in mid_indicators if indicator in text_lower)
        
        # Additional scoring based on skill complexity and quantity
        skill_complexity_score = 0
        if skill_matches:
            # More skills might indicate higher level position
            skill_count = len(skill_matches)
            if skill_count > 15:
                skill_complexity_score += 2  # Many skills = senior
            elif skill_count > 8:
                skill_complexity_score += 1  # Moderate skills = mid
            
            # Check for leadership/architecture skills
            leadership_skills = [
                'leadership', 'management', 'architecture', 'design patterns',
                'system design', 'mentoring', 'code review', 'technical strategy'
            ]
            
            for skill_match in skill_matches:
                skill_lower = skill_match.skill.lower()
                if any(leadership in skill_lower for leadership in leadership_skills):
                    skill_complexity_score += 1
        
        # Combine scores
        total_entry = entry_count
        total_mid = mid_count + (1 if skill_complexity_score == 1 else 0)
        total_senior = senior_count + skill_complexity_score
        
        # Determine experience level
        if total_senior > total_entry and total_senior > total_mid:
            experience_level = "senior-level"
        elif total_entry > total_mid and total_entry > total_senior:
            experience_level = "entry-level"
        else:
            experience_level = "mid-level"  # Default to mid-level
        
        self.logger.info(f"Experience level inference: {experience_level}")
        self.logger.debug(f"Scores - Entry: {total_entry}, Mid: {total_mid}, Senior: {total_senior}")
        
        return experience_level