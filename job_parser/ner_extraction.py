"""
Named Entity Recognition system for skill extraction.

This module implements NER-based skill extraction using spaCy models
with custom patterns for technical skills and tools.
"""

import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Any, Set
import re

from .interfaces import NERExtractorInterface, SkillMatch
from .exceptions import ModelLoadError
from .logging_config import get_logger


class NERExtractor(NERExtractorInterface):
    """Implements Named Entity Recognition for skill extraction.
    
    This class uses spaCy NER models with custom patterns to identify
    technical skills, tools, and other relevant entities in job descriptions.
    """
    
    def __init__(self):
        """Initialize the NER extractor with spaCy model and custom patterns.
        
        Raises:
            ModelLoadError: If spaCy model loading fails
        """
        self.logger = get_logger(__name__)
        self.nlp = None
        self.matcher = None
        
        # Load spaCy model with error handling
        self._load_spacy_model()
        
        # Add custom patterns for programming languages and tools
        self._add_custom_patterns()
        
        self.logger.info("NERExtractor initialized successfully")
    
    def _load_spacy_model(self) -> None:
        """Load the spaCy en_core_web_sm model with error handling.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            self.logger.info("Loading spaCy en_core_web_sm model")
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = Matcher(self.nlp.vocab)
            self.logger.info("spaCy model loaded successfully")
            
        except OSError as e:
            error_msg = (
                "Failed to load spaCy model 'en_core_web_sm'. "
                "Please install it using: python -m spacy download en_core_web_sm"
            )
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
        except Exception as e:
            error_msg = f"Unexpected error loading spaCy model: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def _add_custom_patterns(self) -> None:
        """Add custom patterns for programming languages and tools.
        
        This method defines spaCy Matcher patterns for technical skills
        that might not be recognized by standard NER.
        """
        try:
            # Programming languages patterns
            programming_languages = [
                [{"LOWER": "python"}],
                [{"LOWER": "javascript"}], [{"LOWER": "js"}],
                [{"LOWER": "typescript"}], [{"LOWER": "ts"}],
                [{"LOWER": "java"}],
                [{"LOWER": "c++"}], [{"LOWER": "cpp"}],
                [{"LOWER": "c#"}], [{"LOWER": "csharp"}],
                [{"LOWER": "go"}], [{"LOWER": "golang"}],
                [{"LOWER": "rust"}],
                [{"LOWER": "php"}],
                [{"LOWER": "ruby"}],
                [{"LOWER": "swift"}],
                [{"LOWER": "kotlin"}],
                [{"LOWER": "scala"}],
                [{"LOWER": "r"}],
                [{"LOWER": "matlab"}],
                [{"LOWER": "sql"}],
                [{"LOWER": "html"}],
                [{"LOWER": "css"}],
                [{"LOWER": "bash"}], [{"LOWER": "shell"}]
            ]
            
            # Frameworks and libraries patterns
            frameworks_libraries = [
                [{"LOWER": "react"}], [{"LOWER": "reactjs"}],
                [{"LOWER": "angular"}], [{"LOWER": "angularjs"}],
                [{"LOWER": "vue"}, {"LOWER": "js"}], [{"LOWER": "vuejs"}],
                [{"LOWER": "node"}, {"LOWER": "js"}], [{"LOWER": "nodejs"}],
                [{"LOWER": "express"}], [{"LOWER": "expressjs"}],
                [{"LOWER": "django"}],
                [{"LOWER": "flask"}],
                [{"LOWER": "spring"}], [{"LOWER": "springboot"}],
                [{"LOWER": "tensorflow"}],
                [{"LOWER": "pytorch"}],
                [{"LOWER": "scikit-learn"}], [{"LOWER": "sklearn"}],
                [{"LOWER": "pandas"}],
                [{"LOWER": "numpy"}],
                [{"LOWER": "matplotlib"}],
                [{"LOWER": "seaborn"}],
                [{"LOWER": "opencv"}],
                [{"LOWER": "keras"}],
                [{"LOWER": "fastapi"}],
                [{"LOWER": "laravel"}],
                [{"LOWER": "rails"}], [{"LOWER": "ruby"}, {"LOWER": "on"}, {"LOWER": "rails"}]
            ]
            
            # Tools and platforms patterns
            tools_platforms = [
                [{"LOWER": "git"}],
                [{"LOWER": "github"}],
                [{"LOWER": "gitlab"}],
                [{"LOWER": "docker"}],
                [{"LOWER": "kubernetes"}], [{"LOWER": "k8s"}],
                [{"LOWER": "aws"}], [{"LOWER": "amazon"}, {"LOWER": "web"}, {"LOWER": "services"}],
                [{"LOWER": "azure"}],
                [{"LOWER": "gcp"}], [{"LOWER": "google"}, {"LOWER": "cloud"}],
                [{"LOWER": "jenkins"}],
                [{"LOWER": "terraform"}],
                [{"LOWER": "ansible"}],
                [{"LOWER": "redis"}],
                [{"LOWER": "mongodb"}], [{"LOWER": "mongo"}],
                [{"LOWER": "postgresql"}], [{"LOWER": "postgres"}],
                [{"LOWER": "mysql"}],
                [{"LOWER": "elasticsearch"}],
                [{"LOWER": "kafka"}],
                [{"LOWER": "rabbitmq"}],
                [{"LOWER": "nginx"}],
                [{"LOWER": "apache"}],
                [{"LOWER": "linux"}],
                [{"LOWER": "ubuntu"}],
                [{"LOWER": "centos"}],
                [{"LOWER": "windows"}],
                [{"LOWER": "macos"}]
            ]
            
            # Database patterns
            databases = [
                [{"LOWER": "sql"}, {"LOWER": "server"}],
                [{"LOWER": "oracle"}],
                [{"LOWER": "sqlite"}],
                [{"LOWER": "cassandra"}],
                [{"LOWER": "dynamodb"}],
                [{"LOWER": "firebase"}]
            ]
            
            # Add all patterns to matcher
            for pattern in programming_languages:
                self.matcher.add("PROGRAMMING_LANGUAGE", [pattern])
            
            for pattern in frameworks_libraries:
                self.matcher.add("FRAMEWORK_LIBRARY", [pattern])
            
            for pattern in tools_platforms:
                self.matcher.add("TOOL_PLATFORM", [pattern])
            
            for pattern in databases:
                self.matcher.add("DATABASE", [pattern])
            
            self.logger.info(f"Added {len(programming_languages + frameworks_libraries + tools_platforms + databases)} custom patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to add custom patterns: {e}")
            # Don't raise exception here as this is not critical for basic functionality
    
    def extract_entities(self, text: str) -> List[SkillMatch]:
        """Extract named entities with confidence scores.
        
        Args:
            text: Job description text
            
        Returns:
            List of SkillMatch objects for extracted entities
            
        Raises:
            ModelLoadError: If NER model loading fails
        """
        if not self.nlp or not self.matcher:
            raise ModelLoadError("NER model not properly initialized")
        
        try:
            self.logger.info("Extracting named entities from text")
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            skill_matches = []
            processed_skills = set()  # To avoid duplicates
            
            # Extract standard NER entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE", "PERSON"]:
                    skill_text = ent.text.strip().lower()
                    
                    # Skip if already processed or too short/generic
                    if (skill_text in processed_skills or 
                        len(skill_text) < 2 or 
                        skill_text in {"the", "and", "or", "of", "in", "to", "for", "with"}):
                        continue
                    
                    # Determine category based on entity label
                    category = self._determine_category_from_ner(ent.label_, skill_text)
                    
                    # Basic confidence based on entity confidence (if available)
                    confidence = getattr(ent, 'confidence', 0.6)  # Default confidence for NER
                    
                    skill_match = SkillMatch(
                        skill=ent.text.strip(),
                        category=category,
                        confidence=confidence,
                        source="ner",
                        context=self._extract_context(doc, ent.start, ent.end)
                    )
                    
                    skill_matches.append(skill_match)
                    processed_skills.add(skill_text)
            
            # Extract custom pattern matches
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                skill_text = span.text.strip().lower()
                
                # Skip if already processed
                if skill_text in processed_skills:
                    continue
                
                # Get pattern label
                pattern_label = self.nlp.vocab.strings[match_id]
                category = self._determine_category_from_pattern(pattern_label)
                
                # Higher confidence for custom patterns as they are more specific
                confidence = 0.8
                
                skill_match = SkillMatch(
                    skill=span.text.strip(),
                    category=category,
                    confidence=confidence,
                    source="ner",
                    context=self._extract_context(doc, start, end)
                )
                
                skill_matches.append(skill_match)
                processed_skills.add(skill_text)
            
            self.logger.info(f"Extracted {len(skill_matches)} entities using NER")
            return skill_matches
            
        except Exception as e:
            error_msg = f"Failed to extract entities: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def _determine_category_from_ner(self, ner_label: str, skill_text: str) -> str:
        """Determine skill category from NER label and text content.
        
        Args:
            ner_label: spaCy NER label
            skill_text: The skill text (lowercase)
            
        Returns:
            Category string (technical, soft, tools)
        """
        # Check if it's a known technical term
        technical_indicators = {
            "python", "java", "javascript", "react", "angular", "sql", 
            "machine learning", "ai", "data science", "backend", "frontend"
        }
        
        if any(indicator in skill_text for indicator in technical_indicators):
            return "technical"
        
        # Organizations might be tools/platforms
        if ner_label == "ORG":
            return "tools"
        
        # Products are usually tools
        if ner_label == "PRODUCT":
            return "tools"
        
        # Default to technical for most NER entities
        return "technical"
    
    def _determine_category_from_pattern(self, pattern_label: str) -> str:
        """Determine skill category from custom pattern label.
        
        Args:
            pattern_label: Custom pattern label
            
        Returns:
            Category string (technical, soft, tools)
        """
        category_mapping = {
            "PROGRAMMING_LANGUAGE": "technical",
            "FRAMEWORK_LIBRARY": "technical", 
            "TOOL_PLATFORM": "tools",
            "DATABASE": "tools"
        }
        
        return category_mapping.get(pattern_label, "technical")
    
    def _extract_context(self, doc, start: int, end: int, window: int = 5) -> str:
        """Extract context around a matched entity.
        
        Args:
            doc: spaCy Doc object
            start: Start token index
            end: End token index
            window: Number of tokens to include on each side
            
        Returns:
            Context string around the entity
        """
        context_start = max(0, start - window)
        context_end = min(len(doc), end + window)
        
        context_tokens = []
        for i in range(context_start, context_end):
            if i == start:
                context_tokens.append(f"**{doc[start:end].text}**")
                # Skip to end of entity
                continue
            elif start <= i < end:
                # Skip tokens within the entity (already added above)
                continue
            else:
                context_tokens.append(doc[i].text)
        
        return " ".join(context_tokens).strip()
    
    def merge_with_semantic_results(self, 
                                  ner_matches: List[SkillMatch], 
                                  semantic_matches: Dict[str, List[tuple]]) -> List[SkillMatch]:
        """Merge NER results with semantic matching results.
        
        This method combines NER-extracted skills with semantically matched skills,
        handling duplicates by preserving the highest confidence score and 
        combining information from both sources.
        
        Args:
            ner_matches: List of SkillMatch objects from NER extraction
            semantic_matches: Dictionary mapping categories to (skill, confidence) tuples
            
        Returns:
            List of merged SkillMatch objects with combined results
        """
        try:
            self.logger.info("Merging NER results with semantic matching results")
            
            # Create a dictionary to track skills by normalized name
            merged_skills = {}
            
            # Add NER matches first
            for ner_match in ner_matches:
                skill_key = ner_match.skill.lower().strip()
                merged_skills[skill_key] = ner_match
            
            # Add semantic matches, handling duplicates
            for category, skill_tuples in semantic_matches.items():
                for skill, confidence in skill_tuples:
                    skill_key = skill.lower().strip()
                    
                    if skill_key in merged_skills:
                        # Skill already exists from NER, update if semantic has higher confidence
                        existing_match = merged_skills[skill_key]
                        if confidence > existing_match.confidence:
                            # Update with semantic match info but preserve NER context if available
                            context = existing_match.context if existing_match.context else ""
                            merged_skills[skill_key] = SkillMatch(
                                skill=skill,  # Use semantic match formatting
                                category=category,
                                confidence=confidence,
                                source="ner+semantic",  # Indicate combined source
                                context=context
                            )
                        else:
                            # Keep NER match but update source to indicate both were found
                            existing_match.source = "ner+semantic"
                    else:
                        # New skill from semantic matching
                        merged_skills[skill_key] = SkillMatch(
                            skill=skill,
                            category=category,
                            confidence=confidence,
                            source="semantic",
                            context=""
                        )
            
            # Convert back to list and sort by confidence
            merged_list = list(merged_skills.values())
            merged_list.sort(key=lambda x: x.confidence, reverse=True)
            
            self.logger.info(f"Merged results: {len(ner_matches)} NER + semantic matches = {len(merged_list)} total")
            
            return merged_list
            
        except Exception as e:
            self.logger.error(f"Failed to merge NER and semantic results: {e}")
            # Return NER matches as fallback
            return ner_matches
    
    def extract_organizations_and_tools(self, text: str) -> List[SkillMatch]:
        """Extract organization and tool entities specifically.
        
        This method focuses on extracting organizations and tools that might
        be relevant as technologies or platforms in job descriptions.
        
        Args:
            text: Job description text
            
        Returns:
            List of SkillMatch objects for organizations and tools
        """
        if not self.nlp:
            raise ModelLoadError("NER model not properly initialized")
        
        try:
            self.logger.info("Extracting organizations and tools")
            
            doc = self.nlp(text)
            org_tool_matches = []
            processed_entities = set()
            
            # Extract organizations that might be tech companies or platforms
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"]:
                    entity_text = ent.text.strip()
                    entity_lower = entity_text.lower()
                    
                    # Skip if already processed or too generic
                    if (entity_lower in processed_entities or 
                        len(entity_lower) < 2 or
                        entity_lower in {"company", "team", "department", "group"}):
                        continue
                    
                    # Check if it's likely a tech organization/tool
                    if self._is_tech_organization(entity_lower):
                        skill_match = SkillMatch(
                            skill=entity_text,
                            category="tools",
                            confidence=0.7,  # Medium confidence for org/tool extraction
                            source="ner",
                            context=self._extract_context(doc, ent.start, ent.end)
                        )
                        
                        org_tool_matches.append(skill_match)
                        processed_entities.add(entity_lower)
            
            self.logger.info(f"Extracted {len(org_tool_matches)} organization/tool entities")
            return org_tool_matches
            
        except Exception as e:
            self.logger.error(f"Failed to extract organizations and tools: {e}")
            return []
    
    def _is_tech_organization(self, org_name: str) -> bool:
        """Check if an organization name is likely tech-related.
        
        Args:
            org_name: Organization name (lowercase)
            
        Returns:
            True if likely tech-related, False otherwise
        """
        # Known tech companies and platforms
        tech_orgs = {
            "google", "microsoft", "amazon", "apple", "facebook", "meta",
            "netflix", "uber", "airbnb", "spotify", "slack", "zoom",
            "salesforce", "oracle", "ibm", "intel", "nvidia", "amd",
            "github", "gitlab", "atlassian", "jira", "confluence",
            "docker", "kubernetes", "jenkins", "terraform", "ansible"
        }
        
        # Check direct matches
        if org_name in tech_orgs:
            return True
        
        # Check for tech-related keywords
        tech_keywords = {
            "tech", "software", "cloud", "data", "ai", "ml", "analytics",
            "platform", "api", "web", "mobile", "app", "system", "digital"
        }
        
        return any(keyword in org_name for keyword in tech_keywords)