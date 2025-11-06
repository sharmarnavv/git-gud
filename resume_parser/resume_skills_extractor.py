"""
Resume skills extraction module.

This module provides functionality to extract skills from resume text,
extending the existing skills ontology for resume-specific contexts.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Set
import logging

from .resume_interfaces import ResumeSkillsExtractorInterface
from .resume_exceptions import SkillsExtractionError
from job_parser.logging_config import get_logger
from job_parser.semantic_matching import SemanticMatcher
from job_parser.interfaces import SkillMatch
from job_parser.ner_extraction import NERExtractor
from job_parser.skill_categorization import SkillCategorizer
from .resume_ontology_enhancer import ResumeOntologyEnhancer


class ResumeSkillsExtractor(ResumeSkillsExtractorInterface):
    """Skills extractor specialized for resume parsing."""
    
    def __init__(self, semantic_matcher: Optional[SemanticMatcher] = None, 
                 ner_extractor: Optional[NERExtractor] = None,
                 skill_categorizer: Optional[SkillCategorizer] = None):
        """Initialize the resume skills extractor.
        
        Args:
            semantic_matcher: Optional semantic matcher instance
            ner_extractor: Optional NER extractor instance
            skill_categorizer: Optional skill categorizer instance
        """
        self.logger = get_logger(__name__)
        
        # Initialize semantic matcher
        self.semantic_matcher = semantic_matcher
        if self.semantic_matcher is None:
            try:
                self.semantic_matcher = SemanticMatcher(
                    model_name='all-MiniLM-L6-v2',
                    threshold=0.5,  # Lower threshold for resume context
                    enable_caching=True
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize semantic matcher: {e}")
                self.semantic_matcher = None
        
        # Initialize NER extractor
        self.ner_extractor = ner_extractor
        if self.ner_extractor is None:
            try:
                self.ner_extractor = NERExtractor()
            except Exception as e:
                self.logger.warning(f"Failed to initialize NER extractor: {e}")
                self.ner_extractor = None
        
        # Initialize skill categorizer
        self.skill_categorizer = skill_categorizer
        if self.skill_categorizer is None:
            self.skill_categorizer = SkillCategorizer()
        
        # Initialize ontology enhancer
        self.ontology_enhancer = ResumeOntologyEnhancer()
        
        # Resume-specific skill patterns
        self._compile_skill_patterns()
        
        # Extended context keywords for resume analysis
        self.skill_context_keywords = {
            'technical': ['technologies', 'tools', 'programming', 'languages', 'frameworks', 'libraries', 
                         'technical skills', 'core competencies', 'expertise'],
            'experience': ['experience', 'worked', 'used', 'implemented', 'developed', 'built', 
                          'utilized', 'applied', 'leveraged', 'employed'],
            'proficient': ['proficient', 'skilled', 'expert', 'advanced', 'intermediate', 'beginner',
                          'familiar', 'knowledgeable', 'competent', 'experienced'],
            'projects': ['projects', 'project', 'built', 'created', 'developed', 'designed',
                        'architected', 'delivered', 'launched', 'deployed'],
            'achievements': ['achieved', 'accomplished', 'delivered', 'improved', 'optimized',
                           'increased', 'reduced', 'enhanced', 'streamlined']
        }
        
        # Resume-specific confidence weights
        self.confidence_weights = {
            'skills_section': 0.9,      # High confidence for dedicated skills sections
            'experience_context': 0.7,   # Medium-high for experience descriptions
            'project_context': 0.8,      # High for project descriptions
            'achievement_context': 0.6,  # Medium for achievement mentions
            'semantic_match': 0.6,       # Medium for semantic matches
            'ner_match': 0.7,           # Medium-high for NER matches
            'exact_match': 0.95,        # Very high for exact ontology matches
            'fuzzy_match': 0.5          # Lower for fuzzy matches
        }
    
    def _compile_skill_patterns(self):
        """Compile regex patterns for skill extraction."""
        
        # Common skill section headers
        self.skill_section_patterns = [
            re.compile(r'(?:technical\s+)?skills?:?', re.IGNORECASE),
            re.compile(r'technologies?:?', re.IGNORECASE),
            re.compile(r'programming\s+languages?:?', re.IGNORECASE),
            re.compile(r'tools?\s+(?:and\s+)?technologies?:?', re.IGNORECASE),
            re.compile(r'core\s+competencies:?', re.IGNORECASE),
            re.compile(r'technical\s+expertise:?', re.IGNORECASE)
        ]
        
        # Skill list patterns (comma-separated, bullet points, etc.)
        self.skill_list_patterns = [
            re.compile(r'([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})(?:,|;|\||$)', re.MULTILINE),
            re.compile(r'•\s*([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})', re.MULTILINE),
            re.compile(r'-\s*([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})', re.MULTILINE),
            re.compile(r'\*\s*([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})', re.MULTILINE)
        ]
        
        # Experience context patterns
        self.experience_patterns = [
            re.compile(r'(?:experience\s+(?:with|in)|worked\s+with|used|utilizing|implementing)\s+([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})', re.IGNORECASE),
            re.compile(r'(?:proficient\s+(?:in|with)|skilled\s+(?:in|with)|expert\s+(?:in|with))\s+([A-Za-z][A-Za-z0-9\+\#\.\-\s]{1,30})', re.IGNORECASE)
        ]
    
    def extract_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[str]:
        """Extract skills from resume text using multiple extraction methods.
        
        Args:
            text: Resume text content
            ontology: Skills ontology for matching
            
        Returns:
            List of extracted skills with confidence-based ranking
        """
        try:
            self.logger.info("Starting comprehensive resume skills extraction")
            
            # Enhance ontology for resume context
            enhanced_ontology = self.ontology_enhancer.enhance_ontology(ontology)
            
            # Collect all skill matches with confidence scores
            all_skill_matches = []
            
            # 1. Extract from dedicated skills sections (highest confidence)
            section_matches = self._extract_from_skill_sections_with_confidence(text, enhanced_ontology)
            all_skill_matches.extend(section_matches)
            
            # 2. Extract from experience descriptions
            experience_matches = self._extract_from_experience_context_with_confidence(text, enhanced_ontology)
            all_skill_matches.extend(experience_matches)
            
            # 3. Extract from project descriptions
            project_matches = self._extract_from_project_context(text, enhanced_ontology)
            all_skill_matches.extend(project_matches)
            
            # 4. Use semantic matching if available
            if self.semantic_matcher:
                semantic_matches = self._extract_with_semantic_matching_enhanced(text, enhanced_ontology)
                all_skill_matches.extend(semantic_matches)
            
            # 5. Use NER extraction if available
            if self.ner_extractor:
                ner_matches = self._extract_with_ner_enhanced(text, enhanced_ontology)
                all_skill_matches.extend(ner_matches)
            
            # 6. Merge and deduplicate matches
            merged_matches = self._merge_and_deduplicate_matches(all_skill_matches)
            
            # 7. Compute final confidence scores
            final_scores = self.skill_categorizer.compute_confidence_scores(merged_matches)
            
            # 8. Filter by minimum confidence and return sorted list
            min_confidence = 0.3  # Resume-specific minimum confidence
            validated_skills = [
                skill for skill, confidence in final_scores.items() 
                if confidence >= min_confidence
            ]
            
            # Sort by confidence (highest first)
            validated_skills.sort(key=lambda skill: final_scores[skill], reverse=True)
            
            self.logger.info(f"Extracted {len(validated_skills)} skills from resume "
                           f"(from {len(all_skill_matches)} total matches)")
            
            return validated_skills
            
        except Exception as e:
            self.logger.error(f"Skills extraction failed: {e}")
            raise SkillsExtractionError(f"Failed to extract skills: {e}", cause=e)
    
    def _extract_from_skill_sections_with_confidence(self, text: str, ontology: Dict[str, List[str]]) -> List[SkillMatch]:
        """Extract skills from dedicated skill sections with confidence scoring."""
        skill_matches = []
        lines = text.split('\n')
        
        in_skill_section = False
        skill_section_lines = []
        section_start_line = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line starts a skill section
            if any(pattern.search(line) for pattern in self.skill_section_patterns):
                in_skill_section = True
                section_start_line = line_num
                # Include the current line if it has content after the header
                header_content = self._extract_content_after_header(line)
                if header_content:
                    skill_section_lines.append(header_content)
                continue
            
            # If we're in a skill section, collect lines until we hit another section
            if in_skill_section:
                if self._is_new_section_header(line):
                    in_skill_section = False
                    # Process collected section
                    if skill_section_lines:
                        section_matches = self._parse_skill_section_with_confidence(
                            skill_section_lines, ontology, section_start_line
                        )
                        skill_matches.extend(section_matches)
                        skill_section_lines = []
                else:
                    skill_section_lines.append(line)
        
        # Process any remaining skill section content
        if skill_section_lines:
            section_matches = self._parse_skill_section_with_confidence(
                skill_section_lines, ontology, section_start_line
            )
            skill_matches.extend(section_matches)
        
        return skill_matches
    
    def _parse_skill_section_with_confidence(self, section_lines: List[str], 
                                           ontology: Dict[str, List[str]], 
                                           start_line: int) -> List[SkillMatch]:
        """Parse skill section content and create SkillMatch objects."""
        skill_matches = []
        section_text = '\n'.join(section_lines)
        
        # Extract skills using various patterns
        extracted_skills = self._parse_skill_lists_enhanced(section_text, ontology)
        
        for skill, category in extracted_skills:
            # High confidence for skills found in dedicated sections
            confidence = self.confidence_weights['skills_section']
            
            # Create context from surrounding text
            context = self._create_section_context(section_lines, skill)
            
            skill_match = SkillMatch(
                skill=skill,
                category=category,
                confidence=confidence,
                source="skills_section",
                context=context
            )
            skill_matches.append(skill_match)
        
        return skill_matches
    
    def _extract_from_experience_context_with_confidence(self, text: str, 
                                                       ontology: Dict[str, List[str]]) -> List[SkillMatch]:
        """Extract skills mentioned in experience context with confidence scoring."""
        skill_matches = []
        
        # Split text into sentences for better context analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            # Check if sentence contains experience-related keywords
            if self._contains_experience_keywords(sentence):
                # Extract skills from this sentence
                sentence_skills = self._extract_skills_from_sentence(sentence, ontology)
                
                for skill, category in sentence_skills:
                    # Calculate confidence based on context
                    confidence = self._calculate_experience_confidence(sentence, skill)
                    
                    skill_match = SkillMatch(
                        skill=skill,
                        category=category,
                        confidence=confidence,
                        source="experience_context",
                        context=sentence
                    )
                    skill_matches.append(skill_match)
        
        return skill_matches
    
    def _extract_from_project_context(self, text: str, ontology: Dict[str, List[str]]) -> List[SkillMatch]:
        """Extract skills from project descriptions."""
        skill_matches = []
        
        # Find project sections
        project_sections = self._identify_project_sections(text)
        
        for project_text in project_sections:
            # Extract skills from project description
            project_skills = self._extract_skills_from_sentence(project_text, ontology)
            
            for skill, category in project_skills:
                # High confidence for project context
                confidence = self.confidence_weights['project_context']
                
                skill_match = SkillMatch(
                    skill=skill,
                    category=category,
                    confidence=confidence,
                    source="project_context",
                    context=project_text[:200] + "..." if len(project_text) > 200 else project_text
                )
                skill_matches.append(skill_match)
        
        return skill_matches
    
    def _extract_with_semantic_matching_enhanced(self, text: str, 
                                               ontology: Dict[str, List[str]]) -> List[SkillMatch]:
        """Extract skills using enhanced semantic matching."""
        skill_matches = []
        
        try:
            # Split text into meaningful chunks (sentences and bullet points)
            text_chunks = self._split_text_into_chunks(text)
            
            # Use semantic matcher with resume-specific threshold
            semantic_matches = self.semantic_matcher.find_skill_matches(
                text_chunks, ontology, threshold=0.5  # Lower threshold for resumes
            )
            
            # Convert to SkillMatch objects
            for category, matches in semantic_matches.items():
                for skill, confidence in matches:
                    # Adjust confidence for resume context
                    adjusted_confidence = confidence * self.confidence_weights['semantic_match']
                    
                    # Find best matching context
                    context = self._find_best_context_for_skill(text_chunks, skill)
                    
                    skill_match = SkillMatch(
                        skill=skill,
                        category=category,
                        confidence=adjusted_confidence,
                        source="semantic_matching",
                        context=context
                    )
                    skill_matches.append(skill_match)
            
            return skill_matches
            
        except Exception as e:
            self.logger.warning(f"Enhanced semantic skill extraction failed: {e}")
            return []
    
    def _extract_with_ner_enhanced(self, text: str, ontology: Dict[str, List[str]]) -> List[SkillMatch]:
        """Extract skills using enhanced NER extraction."""
        skill_matches = []
        
        try:
            # Use NER extractor to find entities
            ner_matches = self.ner_extractor.extract_entities(text)
            
            # Filter and enhance NER matches for resume context
            for ner_match in ner_matches:
                # Validate against ontology
                if self._validate_skill_against_ontology(ner_match.skill, ontology):
                    # Adjust confidence for resume context
                    adjusted_confidence = ner_match.confidence * self.confidence_weights['ner_match']
                    
                    # Update category if needed based on ontology
                    category = self._determine_skill_category(ner_match.skill, ontology)
                    
                    enhanced_match = SkillMatch(
                        skill=ner_match.skill,
                        category=category,
                        confidence=adjusted_confidence,
                        source="ner_extraction",
                        context=ner_match.context
                    )
                    skill_matches.append(enhanced_match)
            
            return skill_matches
            
        except Exception as e:
            self.logger.warning(f"Enhanced NER skill extraction failed: {e}")
            return []
    
    def _parse_skill_lists(self, text: str, ontology: Dict[str, List[str]]) -> List[str]:
        """Parse skill lists from text using various patterns."""
        skills = []
        
        for pattern in self.skill_list_patterns:
            matches = pattern.findall(text)
            for match in matches:
                skill_candidate = match.strip()
                if self._is_valid_skill_candidate(skill_candidate, ontology):
                    skills.append(skill_candidate)
        
        return skills
    
    def _is_valid_skill_candidate(self, candidate: str, ontology: Dict[str, List[str]]) -> bool:
        """Check if a candidate string is likely a valid skill."""
        candidate = candidate.strip()
        
        # Basic validation
        if len(candidate) < 2 or len(candidate) > 50:
            return False
        
        # Check against ontology (fuzzy matching)
        candidate_lower = candidate.lower()
        for category_skills in ontology.values():
            for skill in category_skills:
                if candidate_lower == skill.lower():
                    return True
                # Check for partial matches
                if candidate_lower in skill.lower() or skill.lower() in candidate_lower:
                    if len(candidate) >= len(skill) * 0.7:  # At least 70% match
                        return True
        
        # Check for common skill patterns
        skill_indicators = ['++', '#', '.js', '.py', 'sql', 'api', 'framework', 'library']
        if any(indicator in candidate_lower for indicator in skill_indicators):
            return True
        
        return False
    
    def _extract_content_after_header(self, line: str) -> str:
        """Extract content after skill section header."""
        for pattern in self.skill_section_patterns:
            match = pattern.search(line)
            if match:
                # Return content after the matched header
                return line[match.end():].strip()
        return ""
    
    def _is_new_section_header(self, line: str) -> bool:
        """Check if line is a new section header."""
        common_headers = [
            'experience', 'education', 'projects', 'certifications',
            'achievements', 'awards', 'publications', 'references'
        ]
        
        line_lower = line.lower().strip()
        
        # Check for common section headers
        for header in common_headers:
            if line_lower.startswith(header) and ':' in line:
                return True
        
        # Check for all caps headers
        if line.isupper() and len(line.split()) <= 3:
            return True
        
        return False
    
    def analyze_skill_context(self, text: str, skill: str) -> Dict[str, Any]:
        """Analyze context where skill is mentioned with enhanced analysis.
        
        Args:
            text: Resume text content
            skill: Skill to analyze
            
        Returns:
            Dictionary with comprehensive context analysis
        """
        try:
            # Normalize skill name for better matching
            normalized_skill = self.ontology_enhancer.normalize_skill_name(skill)
            skill_variations = self.ontology_enhancer.get_skill_variations(normalized_skill)
            
            context_analysis = {
                'skill': skill,
                'normalized_skill': normalized_skill,
                'skill_variations': skill_variations,
                'mentions': [],
                'contexts': [],
                'confidence_factors': [],
                'overall_confidence': 0.0,
                'section_analysis': {},
                'experience_years': None,
                'proficiency_level': None
            }
            
            # Find all mentions of the skill and its variations
            all_patterns = [re.compile(re.escape(var), re.IGNORECASE) for var in skill_variations]
            
            # Analyze by sections
            sections = self._split_resume_into_sections(text)
            
            for section_name, section_text in sections.items():
                section_mentions = []
                
                sentences = [s.strip() for s in section_text.split('.') if s.strip()]
                
                for i, sentence in enumerate(sentences):
                    for pattern, variation in zip(all_patterns, skill_variations):
                        if pattern.search(sentence):
                            context_info = {
                                'sentence': sentence,
                                'sentence_index': i,
                                'section': section_name,
                                'skill_variation': variation,
                                'context_type': self._classify_context_enhanced(sentence, section_name),
                                'confidence_boost': self._calculate_context_confidence_enhanced(sentence, section_name),
                                'proficiency_indicators': self._extract_proficiency_indicators(sentence),
                                'experience_indicators': self._extract_experience_indicators(sentence)
                            }
                            section_mentions.append(context_info)
                            context_analysis['mentions'].append(context_info)
                
                if section_mentions:
                    context_analysis['section_analysis'][section_name] = {
                        'mention_count': len(section_mentions),
                        'avg_confidence': sum(m['confidence_boost'] for m in section_mentions) / len(section_mentions),
                        'context_types': list(set(m['context_type'] for m in section_mentions))
                    }
            
            # Analyze overall context
            if context_analysis['mentions']:
                context_types = [mention['context_type'] for mention in context_analysis['mentions']]
                confidence_boosts = [mention['confidence_boost'] for mention in context_analysis['mentions']]
                
                context_analysis['contexts'] = list(set(context_types))
                context_analysis['confidence_factors'] = confidence_boosts
                context_analysis['overall_confidence'] = min(1.0, sum(confidence_boosts) / len(confidence_boosts))
                
                # Extract experience years if mentioned
                context_analysis['experience_years'] = self._extract_experience_years(text, skill_variations)
                
                # Determine proficiency level
                context_analysis['proficiency_level'] = self._determine_proficiency_level(context_analysis['mentions'])
            
            return context_analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced context analysis failed for skill '{skill}': {e}")
            return {
                'skill': skill,
                'mentions': [],
                'contexts': [],
                'confidence_factors': [],
                'overall_confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_context(self, sentence: str) -> str:
        """Classify the context type of a sentence."""
        sentence_lower = sentence.lower()
        
        # Check for different context types
        if any(keyword in sentence_lower for keyword in self.skill_context_keywords['technical']):
            return 'technical_section'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['experience']):
            return 'experience_description'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['proficient']):
            return 'proficiency_statement'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['projects']):
            return 'project_description'
        else:
            return 'general_mention'
    
    def _calculate_context_confidence(self, sentence: str) -> float:
        """Calculate confidence boost based on context."""
        sentence_lower = sentence.lower()
        confidence = 0.5  # Base confidence
        
        # Boost for specific context indicators
        if any(word in sentence_lower for word in ['expert', 'advanced', 'proficient']):
            confidence += 0.3
        elif any(word in sentence_lower for word in ['experienced', 'skilled', 'familiar']):
            confidence += 0.2
        elif any(word in sentence_lower for word in ['used', 'worked', 'implemented']):
            confidence += 0.1
        
        # Boost for quantified experience
        if re.search(r'\d+\s*(?:years?|months?)', sentence_lower):
            confidence += 0.2
        
        # Boost for project context
        if any(word in sentence_lower for word in ['project', 'built', 'developed', 'created']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _validate_and_filter_skills(self, skills: List[str], ontology: Dict[str, List[str]]) -> List[str]:
        """Validate and filter extracted skills."""
        validated_skills = []
        seen_skills = set()
        
        for skill in skills:
            skill = skill.strip()
            skill_lower = skill.lower()
            
            # Skip duplicates (case-insensitive)
            if skill_lower in seen_skills:
                continue
            
            # Skip very short or very long skills
            if len(skill) < 2 or len(skill) > 50:
                continue
            
            # Skip common non-skills
            non_skills = {'and', 'or', 'with', 'using', 'including', 'such', 'as', 'etc'}
            if skill_lower in non_skills:
                continue
            
            # Add valid skill
            validated_skills.append(skill)
            seen_skills.add(skill_lower)
        
        return validated_skills
    
    def _merge_and_deduplicate_matches(self, skill_matches: List[SkillMatch]) -> List[SkillMatch]:
        """Merge and deduplicate skill matches, keeping highest confidence."""
        merged_matches = {}
        
        for match in skill_matches:
            skill_key = match.skill.lower().strip()
            
            if skill_key in merged_matches:
                # Keep match with higher confidence
                existing_match = merged_matches[skill_key]
                if match.confidence > existing_match.confidence:
                    # Combine sources if different
                    if existing_match.source != match.source:
                        match.source = f"{existing_match.source}+{match.source}"
                    merged_matches[skill_key] = match
                else:
                    # Update source to show multiple detections
                    if existing_match.source != match.source:
                        existing_match.source = f"{existing_match.source}+{match.source}"
            else:
                merged_matches[skill_key] = match
        
        return list(merged_matches.values())
    
    def _parse_skill_lists_enhanced(self, text: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Enhanced skill list parsing with better pattern recognition."""
        skills_with_categories = []
        
        # Try different parsing strategies
        strategies = [
            self._parse_comma_separated_skills,
            self._parse_bullet_point_skills,
            self._parse_line_separated_skills,
            self._parse_categorized_skills
        ]
        
        for strategy in strategies:
            strategy_results = strategy(text, ontology)
            skills_with_categories.extend(strategy_results)
        
        return skills_with_categories
    
    def _parse_comma_separated_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Parse comma-separated skill lists."""
        skills = []
        
        # Split by commas and clean up
        candidates = [s.strip() for s in text.split(',') if s.strip()]
        
        for candidate in candidates:
            # Remove common prefixes/suffixes
            cleaned = re.sub(r'^[-•*\s]+|[,;.\s]+$', '', candidate)
            
            if self._is_valid_skill_candidate(cleaned, ontology):
                category = self._determine_skill_category(cleaned, ontology)
                skills.append((cleaned, category))
        
        return skills
    
    def _parse_bullet_point_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Parse bullet point skill lists."""
        skills = []
        
        # Match various bullet point patterns
        bullet_patterns = [
            r'•\s*([^\n•]+)',
            r'-\s*([^\n-]+)',
            r'\*\s*([^\n*]+)',
            r'○\s*([^\n○]+)'
        ]
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                cleaned = match.strip()
                if self._is_valid_skill_candidate(cleaned, ontology):
                    category = self._determine_skill_category(cleaned, ontology)
                    skills.append((cleaned, category))
        
        return skills
    
    def _parse_line_separated_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Parse line-separated skill lists."""
        skills = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and self._is_valid_skill_candidate(line, ontology):
                category = self._determine_skill_category(line, ontology)
                skills.append((line, category))
        
        return skills
    
    def _parse_categorized_skills(self, text: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Parse skills that are already categorized in the text."""
        skills = []
        
        # Look for category headers followed by skills
        category_patterns = {
            'technical': r'(?:programming|technical|languages?|frameworks?|technologies?):\s*([^\n]+)',
            'tools': r'(?:tools?|platforms?|software):\s*([^\n]+)',
            'soft': r'(?:soft|interpersonal|communication):\s*([^\n]+)'
        }
        
        for category, pattern in category_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Parse the skills list after the category header
                category_skills = self._parse_comma_separated_skills(match, ontology)
                # Override category with detected one
                skills.extend([(skill, category) for skill, _ in category_skills])
        
        return skills
    
    def _contains_experience_keywords(self, sentence: str) -> bool:
        """Check if sentence contains experience-related keywords."""
        sentence_lower = sentence.lower()
        
        experience_keywords = [
            'experience', 'worked', 'used', 'implemented', 'developed', 'built',
            'utilized', 'applied', 'leveraged', 'employed', 'responsible',
            'managed', 'led', 'created', 'designed', 'architected'
        ]
        
        return any(keyword in sentence_lower for keyword in experience_keywords)
    
    def _extract_skills_from_sentence(self, sentence: str, ontology: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Extract skills from a single sentence."""
        skills = []
        
        # Check for exact matches first
        for category, category_skills in ontology.items():
            for skill in category_skills:
                if skill.lower() in sentence.lower():
                    skills.append((skill, category))
        
        # Check for partial matches and variations
        words = sentence.split()
        for word_combo in self._generate_word_combinations(words):
            combo_text = ' '.join(word_combo)
            if self._is_valid_skill_candidate(combo_text, ontology):
                category = self._determine_skill_category(combo_text, ontology)
                skills.append((combo_text, category))
        
        return skills
    
    def _generate_word_combinations(self, words: List[str], max_length: int = 3) -> List[List[str]]:
        """Generate word combinations up to max_length."""
        combinations = []
        
        for i in range(len(words)):
            for j in range(i + 1, min(i + max_length + 1, len(words) + 1)):
                combo = words[i:j]
                if len(combo) <= max_length:
                    combinations.append(combo)
        
        return combinations
    
    def _calculate_experience_confidence(self, sentence: str, skill: str) -> float:
        """Calculate confidence for skills found in experience context."""
        base_confidence = self.confidence_weights['experience_context']
        
        sentence_lower = sentence.lower()
        
        # Boost for specific experience indicators
        if any(indicator in sentence_lower for indicator in ['expert', 'advanced', 'proficient']):
            base_confidence += 0.2
        elif any(indicator in sentence_lower for indicator in ['experienced', 'skilled']):
            base_confidence += 0.1
        
        # Boost for quantified experience
        if re.search(r'\d+\s*(?:years?|months?)', sentence_lower):
            base_confidence += 0.15
        
        # Boost for achievement context
        if any(indicator in sentence_lower for indicator in self.skill_context_keywords['achievements']):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _identify_project_sections(self, text: str) -> List[str]:
        """Identify and extract project sections from resume text."""
        project_sections = []
        
        # Look for project headers
        project_patterns = [
            r'(?:projects?|portfolio):\s*([^:]+?)(?=\n[A-Z]|\n\n|\Z)',
            r'(?:key\s+projects?|notable\s+projects?):\s*([^:]+?)(?=\n[A-Z]|\n\n|\Z)'
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            project_sections.extend(matches)
        
        # Also look for bullet points that might be projects
        lines = text.split('\n')
        in_project_section = False
        current_project = []
        
        for line in lines:
            line = line.strip()
            
            if re.match(r'(?:projects?|portfolio)', line, re.IGNORECASE):
                in_project_section = True
                continue
            
            if in_project_section:
                if self._is_new_section_header(line):
                    in_project_section = False
                    if current_project:
                        project_sections.append('\n'.join(current_project))
                        current_project = []
                else:
                    current_project.append(line)
        
        # Add any remaining project content
        if current_project:
            project_sections.append('\n'.join(current_project))
        
        return project_sections
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into meaningful chunks for semantic analysis."""
        chunks = []
        
        # Split by sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks.extend(sentences)
        
        # Split by bullet points
        bullet_points = re.findall(r'[•\-*]\s*([^\n•\-*]+)', text)
        chunks.extend([bp.strip() for bp in bullet_points if bp.strip()])
        
        # Split by lines (for structured content)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        chunks.extend(lines)
        
        # Remove duplicates and very short chunks
        unique_chunks = list(set(chunk for chunk in chunks if len(chunk) > 10))
        
        return unique_chunks
    
    def _find_best_context_for_skill(self, text_chunks: List[str], skill: str) -> str:
        """Find the best context chunk that mentions the skill."""
        skill_lower = skill.lower()
        
        best_context = ""
        best_score = 0
        
        for chunk in text_chunks:
            if skill_lower in chunk.lower():
                # Score based on context quality
                score = len(chunk)  # Longer contexts are generally better
                
                # Boost for experience/project keywords
                if any(keyword in chunk.lower() for keyword in self.skill_context_keywords['experience']):
                    score += 50
                if any(keyword in chunk.lower() for keyword in self.skill_context_keywords['projects']):
                    score += 40
                
                if score > best_score:
                    best_score = score
                    best_context = chunk
        
        return best_context[:200] + "..." if len(best_context) > 200 else best_context
    
    def _validate_skill_against_ontology(self, skill: str, ontology: Dict[str, List[str]]) -> bool:
        """Validate if a skill exists in the ontology or is similar enough."""
        skill_lower = skill.lower()
        
        # Check exact matches
        for category_skills in ontology.values():
            for ontology_skill in category_skills:
                if skill_lower == ontology_skill.lower():
                    return True
                
                # Check partial matches (fuzzy matching)
                if (skill_lower in ontology_skill.lower() or 
                    ontology_skill.lower() in skill_lower):
                    # Require at least 70% similarity
                    if len(skill) >= len(ontology_skill) * 0.7:
                        return True
        
        return False
    
    def _determine_skill_category(self, skill: str, ontology: Dict[str, List[str]]) -> str:
        """Determine the category of a skill based on the ontology."""
        skill_lower = skill.lower()
        
        # Check exact matches first
        for category, category_skills in ontology.items():
            for ontology_skill in category_skills:
                if skill_lower == ontology_skill.lower():
                    return category
        
        # Check partial matches
        for category, category_skills in ontology.items():
            for ontology_skill in category_skills:
                if (skill_lower in ontology_skill.lower() or 
                    ontology_skill.lower() in skill_lower):
                    return category
        
        # Default categorization based on common patterns
        if any(indicator in skill_lower for indicator in 
               ['python', 'java', 'javascript', 'programming', 'framework', 'library']):
            return 'technical'
        elif any(indicator in skill_lower for indicator in 
                ['git', 'docker', 'aws', 'azure', 'tool', 'platform']):
            return 'tools'
        else:
            return 'soft'  # Default fallback
    
    def _create_section_context(self, section_lines: List[str], skill: str) -> str:
        """Create context from skill section lines."""
        # Find the line containing the skill
        skill_lower = skill.lower()
        
        for i, line in enumerate(section_lines):
            if skill_lower in line.lower():
                # Include surrounding lines for context
                start = max(0, i - 1)
                end = min(len(section_lines), i + 2)
                context_lines = section_lines[start:end]
                context = ' '.join(context_lines)
                return context[:200] + "..." if len(context) > 200 else context
        
        # Fallback to first few lines
        context = ' '.join(section_lines[:3])
        return context[:200] + "..." if len(context) > 200 else context
    
    def _split_resume_into_sections(self, text: str) -> Dict[str, str]:
        """Split resume text into logical sections."""
        sections = {
            'skills': '',
            'experience': '',
            'projects': '',
            'education': '',
            'summary': '',
            'other': ''
        }
        
        lines = text.split('\n')
        current_section = 'other'
        section_content = []
        
        section_headers = {
            'skills': ['skills', 'technical skills', 'core competencies', 'technologies'],
            'experience': ['experience', 'work experience', 'employment', 'professional experience'],
            'projects': ['projects', 'key projects', 'notable projects', 'portfolio'],
            'education': ['education', 'academic background', 'qualifications'],
            'summary': ['summary', 'profile', 'objective', 'about']
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            new_section = None
            for section, headers in section_headers.items():
                if any(header in line_lower for header in headers):
                    new_section = section
                    break
            
            if new_section:
                # Save previous section content
                if section_content:
                    sections[current_section] += '\n'.join(section_content)
                
                # Start new section
                current_section = new_section
                section_content = []
            else:
                section_content.append(line)
        
        # Save final section content
        if section_content:
            sections[current_section] += '\n'.join(section_content)
        
        return sections
    
    def _classify_context_enhanced(self, sentence: str, section: str) -> str:
        """Enhanced context classification considering section information."""
        sentence_lower = sentence.lower()
        
        # Section-based classification
        if section == 'skills':
            return 'skills_section'
        elif section == 'experience':
            return 'work_experience'
        elif section == 'projects':
            return 'project_description'
        elif section == 'education':
            return 'educational_context'
        
        # Content-based classification
        if any(keyword in sentence_lower for keyword in self.skill_context_keywords['technical']):
            return 'technical_context'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['experience']):
            return 'experience_description'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['proficient']):
            return 'proficiency_statement'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['projects']):
            return 'project_context'
        elif any(keyword in sentence_lower for keyword in self.skill_context_keywords['achievements']):
            return 'achievement_context'
        else:
            return 'general_mention'
    
    def _calculate_context_confidence_enhanced(self, sentence: str, section: str) -> float:
        """Enhanced confidence calculation considering section and content."""
        base_confidence = 0.5
        sentence_lower = sentence.lower()
        
        # Section-based confidence
        section_weights = {
            'skills': 0.9,
            'experience': 0.7,
            'projects': 0.8,
            'education': 0.6,
            'summary': 0.5
        }
        
        base_confidence = section_weights.get(section, 0.5)
        
        # Content-based boosts
        if any(word in sentence_lower for word in ['expert', 'advanced', 'proficient']):
            base_confidence += 0.3
        elif any(word in sentence_lower for word in ['experienced', 'skilled', 'familiar']):
            base_confidence += 0.2
        elif any(word in sentence_lower for word in ['used', 'worked', 'implemented']):
            base_confidence += 0.1
        
        # Quantified experience boost
        if re.search(r'\d+\s*(?:years?|months?)', sentence_lower):
            base_confidence += 0.2
        
        # Achievement context boost
        if any(word in sentence_lower for word in ['achieved', 'delivered', 'improved', 'optimized']):
            base_confidence += 0.15
        
        # Project context boost
        if any(word in sentence_lower for word in ['built', 'developed', 'created', 'designed']):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _extract_proficiency_indicators(self, sentence: str) -> List[str]:
        """Extract proficiency level indicators from sentence."""
        sentence_lower = sentence.lower()
        indicators = []
        
        proficiency_levels = {
            'expert': ['expert', 'expertise', 'mastery', 'master'],
            'advanced': ['advanced', 'senior', 'lead', 'principal'],
            'intermediate': ['intermediate', 'proficient', 'competent', 'solid'],
            'beginner': ['beginner', 'basic', 'fundamental', 'learning', 'familiar']
        }
        
        for level, keywords in proficiency_levels.items():
            if any(keyword in sentence_lower for keyword in keywords):
                indicators.append(level)
        
        return indicators
    
    def _extract_experience_indicators(self, sentence: str) -> Dict[str, Any]:
        """Extract experience-related indicators from sentence."""
        indicators = {
            'years': None,
            'months': None,
            'projects_count': None,
            'team_size': None
        }
        
        # Extract years of experience
        years_match = re.search(r'(\d+)\s*(?:years?|yrs?)', sentence.lower())
        if years_match:
            indicators['years'] = int(years_match.group(1))
        
        # Extract months of experience
        months_match = re.search(r'(\d+)\s*months?', sentence.lower())
        if months_match:
            indicators['months'] = int(months_match.group(1))
        
        # Extract project count
        projects_match = re.search(r'(\d+)\s*projects?', sentence.lower())
        if projects_match:
            indicators['projects_count'] = int(projects_match.group(1))
        
        # Extract team size
        team_match = re.search(r'team\s*of\s*(\d+)', sentence.lower())
        if team_match:
            indicators['team_size'] = int(team_match.group(1))
        
        return indicators
    
    def _extract_experience_years(self, text: str, skill_variations: List[str]) -> Optional[int]:
        """Extract years of experience for a specific skill."""
        text_lower = text.lower()
        
        # Look for patterns like "5 years of Python experience"
        for skill in skill_variations:
            skill_lower = skill.lower()
            
            # Pattern: X years of [skill] experience
            pattern1 = rf'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience\s*)?(?:with\s*|in\s*|using\s*)?{re.escape(skill_lower)}'
            match1 = re.search(pattern1, text_lower)
            if match1:
                return int(match1.group(1))
            
            # Pattern: [skill] - X years
            pattern2 = rf'{re.escape(skill_lower)}\s*[-:]\s*(\d+)\s*(?:years?|yrs?)'
            match2 = re.search(pattern2, text_lower)
            if match2:
                return int(match2.group(1))
            
            # Pattern: X+ years [skill]
            pattern3 = rf'(\d+)\+?\s*(?:years?|yrs?)\s*{re.escape(skill_lower)}'
            match3 = re.search(pattern3, text_lower)
            if match3:
                return int(match3.group(1))
        
        return None
    
    def _determine_proficiency_level(self, mentions: List[Dict[str, Any]]) -> Optional[str]:
        """Determine overall proficiency level from all mentions."""
        if not mentions:
            return None
        
        # Collect all proficiency indicators
        all_indicators = []
        for mention in mentions:
            all_indicators.extend(mention.get('proficiency_indicators', []))
        
        if not all_indicators:
            return None
        
        # Determine highest proficiency level mentioned
        proficiency_hierarchy = ['expert', 'advanced', 'intermediate', 'beginner']
        
        for level in proficiency_hierarchy:
            if level in all_indicators:
                return level
        
        return None
    
    def get_skill_confidence_score(self, skill: str, text: str, ontology: Dict[str, List[str]]) -> float:
        """Get confidence score for a specific skill in resume context.
        
        Args:
            skill: Skill to analyze
            text: Resume text
            ontology: Skills ontology
            
        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            # Enhance ontology
            enhanced_ontology = self.ontology_enhancer.enhance_ontology(ontology)
            
            # Analyze context
            context_analysis = self.analyze_skill_context(text, skill)
            
            # Base confidence from context analysis
            base_confidence = context_analysis.get('overall_confidence', 0.0)
            
            # Adjust based on skill validation
            if self._validate_skill_against_ontology(skill, enhanced_ontology):
                base_confidence += 0.2  # Boost for ontology validation
            
            # Adjust based on proficiency level
            proficiency_level = context_analysis.get('proficiency_level')
            if proficiency_level:
                proficiency_boosts = {
                    'expert': 0.3,
                    'advanced': 0.2,
                    'intermediate': 0.1,
                    'beginner': 0.05
                }
                base_confidence += proficiency_boosts.get(proficiency_level, 0)
            
            # Adjust based on experience years
            experience_years = context_analysis.get('experience_years')
            if experience_years:
                if experience_years >= 5:
                    base_confidence += 0.2
                elif experience_years >= 2:
                    base_confidence += 0.1
                elif experience_years >= 1:
                    base_confidence += 0.05
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence score for skill '{skill}': {e}")
            return 0.0