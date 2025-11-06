"""
Main JobDescriptionParser orchestrator class.

This module contains the primary parser class that coordinates all components
to transform job descriptions into structured data.
"""

import gc
from typing import Dict, Any, Optional, List

from .interfaces import (
    JobDescriptionParserInterface, 
    ParsedJobDescription, 
    JobDescriptionInput,
    SkillMatch
)
from .config import ParserConfig, CONFIG
from .exceptions import JobParserError, InputValidationError, OntologyLoadError, ModelLoadError
from .logging_config import get_logger
from .ontology import OntologyLoader
from .preprocessing import TextPreprocessor
from .semantic_matching import SemanticMatcher
from .ner_extraction import NERExtractor
from .skill_categorization import SkillCategorizer


class JobDescriptionParser(JobDescriptionParserInterface):
    """Main orchestrator class for job description parsing.
    
    This class coordinates all parser components to process job descriptions
    through the complete pipeline from text input to structured JSON output.
    
    Attributes:
        config: Parser configuration settings
        logger: Logger instance for this parser
    """
    
    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize the job description parser.
        
        Creates constructor with model and ontology initialization as per requirements.
        Loads all necessary components and validates configuration.
        
        Args:
            config: Optional parser configuration. Uses global CONFIG if None.
            
        Raises:
            JobParserError: If initialization fails
        """
        self.config = config or CONFIG
        self.logger = get_logger(__name__)
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            raise JobParserError(f"Invalid configuration: {e}", cause=e)
        
        # Initialize component instances
        self._ontology_loader = None
        self._text_preprocessor = None
        self._semantic_matcher = None
        self._ner_extractor = None
        self._skill_categorizer = None
        self._ontology = None
        
        # Performance optimization attributes
        self._ontology_embeddings = None
        self._is_optimized_for_batch = False
        self._processing_stats = {
            'total_jobs_processed': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Initialize models and ontology with comprehensive error handling
        try:
            self._initialize_components()
        except Exception as e:
            self.logger.error(f"Failed to initialize parser components: {e}")
            raise JobParserError(f"Parser initialization failed: {e}", cause=e)
        
        self.logger.info("JobDescriptionParser initialized successfully")
    
    def parse_job_description(self, job_desc: str) -> ParsedJobDescription:
        """Parse job description into structured data.
        
        This is the main entry point that orchestrates the entire parsing pipeline.
        Implements comprehensive error handling with try-catch blocks as required.
        
        Args:
            job_desc: Raw job description text
            
        Returns:
            ParsedJobDescription object with extracted information
            
        Raises:
            JobParserError: If parsing fails at any stage
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Starting job description parsing")
            
            # Step 1: Validate input with comprehensive error handling
            try:
                job_input = JobDescriptionInput(
                    text=job_desc, 
                    max_length=self.config.max_text_length
                )
                job_input.validate()
            except (TypeError, InputValidationError) as e:
                self.logger.error(f"Input validation failed: {e}")
                raise InputValidationError(f"Invalid input: {e}", cause=e)
            
            # Step 2: Preprocess text with error handling
            try:
                preprocessed_sentences = self._text_preprocessor.preprocess(job_desc)
                self.logger.debug(f"Preprocessed into {len(preprocessed_sentences)} sentences")
            except Exception as e:
                self.logger.error(f"Text preprocessing failed: {e}")
                raise JobParserError(f"Failed to preprocess text: {e}", cause=e)
            
            # Step 3: Extract skills using semantic matching (if enabled)
            semantic_matches = {}
            if self.config.enable_semantic:
                try:
                    # Use optimized matching if ontology embeddings are precomputed
                    if self._ontology_embeddings and self._is_optimized_for_batch:
                        semantic_matches = self._semantic_matcher.find_skill_matches_optimized(
                            preprocessed_sentences,
                            self._ontology_embeddings,
                            self._ontology,
                            self.config.similarity_threshold
                        )
                        self.logger.debug(f"Optimized semantic matching found matches in {len(semantic_matches)} categories")
                    else:
                        semantic_matches = self._semantic_matcher.find_skill_matches(
                            preprocessed_sentences, 
                            self._ontology, 
                            self.config.similarity_threshold
                        )
                        self.logger.debug(f"Semantic matching found matches in {len(semantic_matches)} categories")
                except Exception as e:
                    self.logger.warning(f"Semantic matching failed, continuing without: {e}")
                    semantic_matches = {}
            
            # Step 4: Extract entities using NER (if enabled)
            ner_matches = []
            if self.config.enable_ner:
                try:
                    ner_matches = self._ner_extractor.extract_entities(job_desc)
                    self.logger.debug(f"NER extraction found {len(ner_matches)} entities")
                except Exception as e:
                    self.logger.warning(f"NER extraction failed, continuing without: {e}")
                    ner_matches = []
            
            # Step 5: Merge and categorize skills with error handling
            try:
                # Convert semantic matches to SkillMatch objects
                all_skill_matches = self._convert_semantic_to_skill_matches(semantic_matches)
                
                # Merge with NER results
                if ner_matches:
                    merged_matches = self._ner_extractor.merge_with_semantic_results(
                        ner_matches, semantic_matches
                    )
                    all_skill_matches = merged_matches
                
                # Categorize and compute final scores
                categorized_skills = self._skill_categorizer.categorize_skills(
                    all_skill_matches, self._ontology
                )
                confidence_scores = self._skill_categorizer.compute_confidence_scores(all_skill_matches)
                experience_level = self._skill_categorizer.infer_experience_level(job_desc, all_skill_matches)
                
                self.logger.debug(f"Categorized {sum(len(skills) for skills in categorized_skills.values())} skills")
                
            except Exception as e:
                self.logger.error(f"Skill categorization failed: {e}")
                raise JobParserError(f"Failed to categorize skills: {e}", cause=e)
            
            # Step 6: Extract tools and create final output
            try:
                # Extract all skills for skills_required
                all_skills = []
                tools_mentioned = []
                
                for category, skills in categorized_skills.items():
                    all_skills.extend(skills)
                    if category == "tools":
                        tools_mentioned.extend(skills)
                
                # Create metadata
                metadata = {
                    "parser_version": "1.0.0",
                    "processing_stats": {
                        "input_length_words": len(job_desc.split()),
                        "sentences_processed": len(preprocessed_sentences),
                        "semantic_matches": sum(len(matches) for matches in semantic_matches.values()),
                        "ner_matches": len(ner_matches),
                        "total_skills_found": len(all_skills)
                    },
                    "config": {
                        "similarity_threshold": self.config.similarity_threshold,
                        "max_text_length": self.config.max_text_length,
                        "enable_ner": self.config.enable_ner,
                        "enable_semantic": self.config.enable_semantic
                    }
                }
                
                # Create final result
                result = ParsedJobDescription(
                    skills_required=all_skills,
                    experience_level=experience_level,
                    tools_mentioned=tools_mentioned,
                    confidence_scores=confidence_scores,
                    categories=categorized_skills,
                    metadata=metadata
                )
                
                # Update performance statistics
                processing_time = time.time() - start_time
                self._processing_stats['total_jobs_processed'] += 1
                
                # Update average processing time
                total_jobs = self._processing_stats['total_jobs_processed']
                current_avg = self._processing_stats['average_processing_time']
                self._processing_stats['average_processing_time'] = (
                    (current_avg * (total_jobs - 1) + processing_time) / total_jobs
                )
                
                # Update cache hit rate if semantic matching is enabled
                if self.config.enable_semantic and self._semantic_matcher:
                    semantic_stats = self._semantic_matcher.get_performance_stats()
                    self._processing_stats['cache_hit_rate'] = semantic_stats.get('cache_hit_rate', 0.0)
                
                self.logger.info(f"Parsing completed successfully: {len(all_skills)} skills, {experience_level} level "
                               f"(processed in {processing_time:.3f}s)")
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to create final output: {e}")
                raise JobParserError(f"Failed to create parsing result: {e}", cause=e)
            
        except (InputValidationError, JobParserError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            self.logger.error(f"Unexpected parsing error: {e}")
            raise JobParserError(f"Unexpected error during parsing: {e}", cause=e)
    
    def _initialize_components(self) -> None:
        """Initialize all parser components.
        
        Loads ontology and initializes all processing components with comprehensive
        error handling as required.
        
        Raises:
            JobParserError: If component initialization fails
        """
        try:
            self.logger.info("Initializing parser components")
            
            # Initialize ontology loader and load skills ontology
            try:
                self._ontology_loader = OntologyLoader()
                self._ontology = self._ontology_loader.load_ontology_with_fallback(
                    self.config.ontology_path
                )
                self.logger.info(f"Loaded ontology with {sum(len(skills) for skills in self._ontology.values())} skills")
            except Exception as e:
                self.logger.error(f"Failed to load ontology: {e}")
                raise OntologyLoadError(f"Ontology initialization failed: {e}", cause=e)
            
            # Initialize text preprocessor
            try:
                self._text_preprocessor = TextPreprocessor()
                self.logger.info("Text preprocessor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize text preprocessor: {e}")
                raise ModelLoadError(f"Text preprocessor initialization failed: {e}", cause=e)
            
            # Initialize semantic matcher (if enabled)
            if self.config.enable_semantic:
                try:
                    self._semantic_matcher = SemanticMatcher(
                        model_name=self.config.model_name,
                        threshold=self.config.similarity_threshold,
                        enable_caching=True,  # Enable caching for performance
                        batch_size=32  # Optimized batch size
                    )
                    self.logger.info(f"Semantic matcher initialized with model: {self.config.model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize semantic matcher: {e}")
                    raise ModelLoadError(f"Semantic matcher initialization failed: {e}", cause=e)
            else:
                self.logger.info("Semantic matching disabled in configuration")
            
            # Initialize NER extractor (if enabled)
            if self.config.enable_ner:
                try:
                    self._ner_extractor = NERExtractor()
                    self.logger.info("NER extractor initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize NER extractor: {e}")
                    raise ModelLoadError(f"NER extractor initialization failed: {e}", cause=e)
            else:
                self.logger.info("NER extraction disabled in configuration")
            
            # Initialize skill categorizer
            try:
                self._skill_categorizer = SkillCategorizer()
                self.logger.info("Skill categorizer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize skill categorizer: {e}")
                raise JobParserError(f"Skill categorizer initialization failed: {e}", cause=e)
            
            self.logger.info("All components initialized successfully")
            
        except (OntologyLoadError, ModelLoadError, JobParserError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during component initialization: {e}")
            raise JobParserError(f"Failed to initialize parser components: {e}", cause=e)
    
    def get_config(self) -> ParserConfig:
        """Get current parser configuration.
        
        Returns:
            Current parser configuration object
        """
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update parser configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            JobParserError: If configuration update fails
        """
        try:
            # Update configuration attributes
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    raise ValueError(f"Unknown configuration parameter: {key}")
            
            # Validate updated configuration
            self.config.validate()
            
            self.logger.info(f"Configuration updated: {kwargs}")
            
        except Exception as e:
            raise JobParserError(f"Failed to update configuration: {e}", cause=e)
    
    def _convert_semantic_to_skill_matches(self, semantic_matches: Dict[str, List[tuple]]) -> List[SkillMatch]:
        """Convert semantic matching results to SkillMatch objects.
        
        Args:
            semantic_matches: Dictionary mapping categories to (skill, confidence) tuples
            
        Returns:
            List of SkillMatch objects from semantic matching
        """
        skill_matches = []
        
        for category, skill_tuples in semantic_matches.items():
            for skill, confidence in skill_tuples:
                skill_match = SkillMatch(
                    skill=skill,
                    category=category,
                    confidence=confidence,
                    source="semantic",
                    context=""
                )
                skill_matches.append(skill_match)
        
        return skill_matches
    
    def parse_job_description_to_json(self, job_desc: str) -> str:
        """Parse job description and return JSON string output.
        
        Implements JSON serialization with proper formatting as required.
        Handles no matches scenario with explanatory notes.
        
        Args:
            job_desc: Raw job description text
            
        Returns:
            JSON string with structured output and all required fields
            
        Raises:
            JobParserError: If parsing or JSON serialization fails
        """
        try:
            # Parse the job description
            parsed_result = self.parse_job_description(job_desc)
            
            # Convert to JSON-serializable format
            json_output = self._format_json_output(parsed_result)
            
            # Serialize to JSON string with proper formatting
            import json
            return json.dumps(json_output, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON output: {e}")
            raise JobParserError(f"JSON output generation failed: {e}", cause=e)
    
    def _format_json_output(self, parsed_result: ParsedJobDescription) -> Dict[str, Any]:
        """Format parsed result into structured JSON output.
        
        Creates structured output with all required fields and handles
        no matches scenario with explanatory notes as required.
        
        Args:
            parsed_result: ParsedJobDescription object to format
            
        Returns:
            Dictionary with structured output ready for JSON serialization
        """
        # Check if we have any matches
        has_matches = (
            len(parsed_result.skills_required) > 0 or 
            len(parsed_result.tools_mentioned) > 0 or
            any(len(skills) > 0 for skills in parsed_result.categories.values())
        )
        
        # Base output structure
        json_output = {
            "skills_required": parsed_result.skills_required,
            "experience_level": parsed_result.experience_level,
            "tools_mentioned": parsed_result.tools_mentioned,
            "confidence_scores": parsed_result.confidence_scores,
            "categories": parsed_result.categories,
            "metadata": parsed_result.metadata
        }
        
        # Handle no matches scenario with explanatory notes
        if not has_matches:
            json_output["notes"] = {
                "no_matches_found": True,
                "explanation": "No skills were extracted from the job description. This could be due to:",
                "possible_reasons": [
                    "The job description may not contain recognizable technical skills",
                    "The text might be too short or lack specific skill requirements",
                    "Skills mentioned may not be in the current ontology",
                    "The similarity threshold may be too high for this content"
                ],
                "suggestions": [
                    "Try lowering the similarity threshold",
                    "Check if the job description contains clear skill requirements",
                    "Verify the skills ontology contains relevant skills for this domain",
                    "Ensure the input text is a complete job description"
                ]
            }
            
            # Add empty structure for consistency
            if not json_output["skills_required"]:
                json_output["skills_required"] = []
            if not json_output["tools_mentioned"]:
                json_output["tools_mentioned"] = []
            if not json_output["confidence_scores"]:
                json_output["confidence_scores"] = {}
            if not any(json_output["categories"].values()):
                json_output["categories"] = {category: [] for category in json_output["categories"]}
        else:
            # Add success metadata for matches found
            json_output["notes"] = {
                "matches_found": True,
                "total_skills": len(parsed_result.skills_required),
                "total_categories": len([cat for cat, skills in parsed_result.categories.items() if skills]),
                "processing_summary": f"Successfully extracted {len(parsed_result.skills_required)} skills across {len([cat for cat, skills in parsed_result.categories.items() if skills])} categories"
            }
        
        # Add parsing timestamp
        from datetime import datetime
        json_output["metadata"]["parsed_at"] = datetime.utcnow().isoformat() + "Z"
        
        return json_output
    
    def get_parsing_summary(self, parsed_result: ParsedJobDescription) -> Dict[str, Any]:
        """Get a summary of parsing results for debugging and monitoring.
        
        Args:
            parsed_result: ParsedJobDescription object to summarize
            
        Returns:
            Dictionary with parsing summary statistics
        """
        return {
            "total_skills_found": len(parsed_result.skills_required),
            "skills_by_category": {
                category: len(skills) 
                for category, skills in parsed_result.categories.items()
            },
            "average_confidence": (
                sum(parsed_result.confidence_scores.values()) / len(parsed_result.confidence_scores)
                if parsed_result.confidence_scores else 0.0
            ),
            "experience_level": parsed_result.experience_level,
            "tools_count": len(parsed_result.tools_mentioned),
            "processing_stats": parsed_result.metadata.get("processing_stats", {})
        }
    
    def optimize_for_batch_processing(self, expected_batch_size: int, max_memory_mb: float = 512.0) -> None:
        """Optimize parser for batch processing of multiple job descriptions.
        
        This method configures the parser components for efficient batch processing
        by enabling caching, adjusting batch sizes, pre-warming models, and 
        precomputing ontology embeddings for maximum performance.
        
        Args:
            expected_batch_size: Expected number of job descriptions to process
            max_memory_mb: Maximum memory usage limit in MB
        """
        self.logger.info(f"Optimizing parser for batch processing: "
                        f"batch_size={expected_batch_size}, max_memory_mb={max_memory_mb}")
        
        try:
            # Optimize semantic matcher if enabled
            if self.config.enable_semantic and self._semantic_matcher:
                self._semantic_matcher.optimize_for_batch_processing(
                    expected_batch_size, max_memory_mb
                )
                
                # Precompute ontology embeddings for maximum performance
                if self._ontology and not self._ontology_embeddings:
                    self.logger.info("Precomputing ontology embeddings for batch optimization")
                    self._ontology_embeddings = self._semantic_matcher.precompute_ontology_embeddings(
                        self._ontology
                    )
                    self.logger.info("Ontology embeddings precomputed successfully")
            
            # Pre-warm components by processing a small test input
            test_input = "Software engineer with Python experience"
            try:
                import time
                start_time = time.time()
                _ = self.parse_job_description(test_input)
                warmup_time = time.time() - start_time
                self.logger.debug(f"Parser pre-warming completed successfully in {warmup_time:.3f}s")
            except Exception as e:
                self.logger.warning(f"Parser pre-warming failed: {e}")
            
            # Mark as optimized for batch processing
            self._is_optimized_for_batch = True
            
            self.logger.info("Parser optimization for batch processing completed")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize parser for batch processing: {e}")
            raise JobParserError(f"Batch optimization failed: {e}", cause=e)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for all parser components.
        
        Returns:
            Dictionary with detailed performance metrics
        """
        stats = {
            "parser_version": "1.0.0",
            "config": {
                "ontology_path": self.config.ontology_path,
                "similarity_threshold": self.config.similarity_threshold,
                "max_text_length": self.config.max_text_length,
                "enable_ner": self.config.enable_ner,
                "enable_semantic": self.config.enable_semantic,
                "model_name": self.config.model_name
            },
            "optimization": {
                "is_optimized_for_batch": self._is_optimized_for_batch,
                "ontology_embeddings_precomputed": self._ontology_embeddings is not None,
                "processing_stats": self._processing_stats.copy()
            },
            "components": {}
        }
        
        # Get semantic matcher stats
        if self.config.enable_semantic and self._semantic_matcher:
            try:
                stats["components"]["semantic_matcher"] = self._semantic_matcher.get_performance_stats()
            except Exception as e:
                self.logger.warning(f"Failed to get semantic matcher stats: {e}")
                stats["components"]["semantic_matcher"] = {"error": str(e)}
        
        # Get ontology stats
        if self._ontology:
            ontology_stats = {
                "total_skills": sum(len(skills) for skills in self._ontology.values()),
                "categories": list(self._ontology.keys()),
                "skills_per_category": {
                    category: len(skills) for category, skills in self._ontology.items()
                }
            }
            
            # Add embedding stats if available
            if self._ontology_embeddings:
                ontology_stats["embeddings_precomputed"] = True
                ontology_stats["embedding_shapes"] = {
                    category: embeddings.shape for category, embeddings in self._ontology_embeddings.items()
                }
            else:
                ontology_stats["embeddings_precomputed"] = False
            
            stats["components"]["ontology"] = ontology_stats
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all performance caches to free memory.
        
        This method clears model caches, embedding caches, precomputed embeddings,
        and forces garbage collection to free up memory resources.
        """
        self.logger.info("Clearing parser caches")
        
        try:
            # Clear semantic matcher caches
            if self.config.enable_semantic and self._semantic_matcher:
                self._semantic_matcher.clear_caches()
            
            # Clear precomputed ontology embeddings
            if self._ontology_embeddings:
                self.logger.info("Clearing precomputed ontology embeddings")
                self._ontology_embeddings = None
                self._is_optimized_for_batch = False
            
            # Reset processing statistics
            self._processing_stats = {
                'total_jobs_processed': 0,
                'average_processing_time': 0.0,
                'cache_hit_rate': 0.0
            }
            
            # Import and clear global caches
            from .performance import clear_all_caches
            clear_all_caches()
            
            self.logger.info("Successfully cleared all parser caches")
            
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
            raise JobParserError(f"Cache clearing failed: {e}", cause=e)
    
    def parse_job_descriptions_batch(self, job_descriptions: List[str]) -> List[ParsedJobDescription]:
        """Parse multiple job descriptions efficiently in batch.
        
        This method provides optimized batch processing for multiple job descriptions
        with automatic performance optimization and memory management.
        
        Args:
            job_descriptions: List of job description texts to parse
            
        Returns:
            List of ParsedJobDescription objects
            
        Raises:
            JobParserError: If batch parsing fails
        """
        if not job_descriptions:
            return []
        
        import time
        from .performance import MemoryManager
        
        batch_start_time = time.time()
        self.logger.info(f"Starting optimized batch parsing of {len(job_descriptions)} job descriptions")
        
        try:
            # Optimize for batch processing with memory estimation
            memory_estimate = MemoryManager.estimate_memory_usage(job_descriptions)
            self.logger.info(f"Estimated memory usage: {memory_estimate['total_estimated_mb']:.1f} MB")
            
            # Adjust batch processing based on memory constraints
            max_memory_mb = min(1024.0, memory_estimate['total_estimated_mb'] * 1.5)  # 50% buffer
            self.optimize_for_batch_processing(len(job_descriptions), max_memory_mb)
            
            results = []
            chunk_size = 10  # Process in smaller chunks to manage memory
            
            # Process in chunks for better memory management
            for chunk_start in range(0, len(job_descriptions), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(job_descriptions))
                chunk = job_descriptions[chunk_start:chunk_end]
                
                self.logger.debug(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(job_descriptions) + chunk_size - 1)//chunk_size}")
                
                # Process chunk
                for i, job_desc in enumerate(chunk):
                    global_index = chunk_start + i
                    try:
                        self.logger.debug(f"Processing job description {global_index+1}/{len(job_descriptions)}")
                        result = self.parse_job_description(job_desc)
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to parse job description {global_index+1}: {e}")
                        # Create error result to maintain batch consistency
                        error_result = ParsedJobDescription(
                            skills_required=[],
                            experience_level="unknown",
                            tools_mentioned=[],
                            confidence_scores={},
                            categories={},
                            metadata={
                                "parsing_failed": True,
                                "error_message": str(e),
                                "job_index": global_index
                            }
                        )
                        results.append(error_result)
                
                # Force garbage collection between chunks to manage memory
                if chunk_end < len(job_descriptions):
                    gc_stats = MemoryManager.force_garbage_collection()
                    self.logger.debug(f"Garbage collection: {gc_stats['objects_collected']} objects collected")
            
            # Log batch summary with performance metrics
            batch_time = time.time() - batch_start_time
            successful_parses = sum(1 for r in results if not r.metadata.get("parsing_failed", False))
            
            self.logger.info(f"Optimized batch parsing completed in {batch_time:.2f}s: "
                           f"{successful_parses}/{len(job_descriptions)} successful "
                           f"({successful_parses/batch_time:.1f} jobs/sec)")
            
            # Log performance statistics
            perf_stats = self.get_performance_stats()
            optimization_stats = perf_stats.get('optimization', {})
            self.logger.info(f"Batch optimization stats: "
                           f"precomputed_embeddings={optimization_stats.get('ontology_embeddings_precomputed', False)}, "
                           f"avg_processing_time={optimization_stats.get('processing_stats', {}).get('average_processing_time', 0):.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch parsing failed: {e}")
            raise JobParserError(f"Failed to parse job descriptions batch: {e}", cause=e)