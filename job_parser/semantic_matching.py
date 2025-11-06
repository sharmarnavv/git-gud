"""
Semantic matching engine using Sentence-BERT embeddings.

This module implements semantic similarity matching between job description text
and skills ontology using pre-trained Sentence-BERT models for high-quality
embeddings and cosine similarity computation with performance optimizations.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .interfaces import SemanticMatcherInterface
from .exceptions import ModelLoadError
from .performance import get_cached_model, BatchProcessor, MemoryManager

logger = logging.getLogger(__name__)


class SemanticMatcher(SemanticMatcherInterface):
    """Semantic matching engine using Sentence-BERT embeddings with performance optimizations.
    
    This class implements semantic similarity matching between job description
    sentences and skills from the ontology using pre-trained Sentence-BERT
    models with caching, batch processing, and memory management optimizations.
    
    Attributes:
        model: Loaded Sentence-BERT model for embedding generation
        model_name: Name of the Sentence-BERT model to use
        threshold: Minimum similarity threshold for skill matches
        batch_processor: Optimized batch processor for large-scale operations
    """
    
    def __init__(self, 
                 model_name: str = './trained_model/trained_model', 
                 threshold: float = 0.7,
                 enable_caching: bool = True,
                 batch_size: int = 32):
        """Initialize SemanticMatcher with optimized Sentence-BERT model.
        
        Args:
            model_name: Name of the Sentence-BERT model to load
            threshold: Default similarity threshold for matches
            enable_caching: Whether to enable model and embedding caching
            batch_size: Batch size for processing large text collections
            
        Raises:
            ModelLoadError: If model loading fails
        """
        self.model_name = model_name
        self.threshold = threshold
        self.enable_caching = enable_caching
        self.model: Optional[SentenceTransformer] = None
        
        # Performance optimization attributes
        self._model_warmed_up = False
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0
        self._batch_processing_stats = {
            'total_batches': 0,
            'total_texts_processed': 0,
            'average_batch_time': 0.0
        }
        
        # Initialize batch processor for performance optimization
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            enable_caching=enable_caching
        )
        
        try:
            logger.info(f"Loading Sentence-BERT model: {model_name}")
            
            # Check if it's a local path (fine-tuned model) or HuggingFace model name
            import os
            is_local_model = os.path.exists(model_name) and os.path.isdir(model_name)
            
            if is_local_model:
                logger.info(f"Loading fine-tuned model from local path: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded fine-tuned model: {model_name}")
            elif enable_caching:
                # Use cached model for better performance (for HuggingFace models)
                self.model = get_cached_model(model_name)
                logger.info(f"Successfully loaded cached model: {model_name}")
            else:
                # Load model directly
                self.model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
                
        except Exception as e:
            error_msg = f"Failed to load Sentence-BERT model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings for input texts with optimization.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            ModelLoadError: If model is not loaded or embedding generation fails
        """
        if self.model is None:
            raise ModelLoadError("Sentence-BERT model not loaded")
        
        if not texts:
            return np.array([])
        
        try:
            import time
            start_time = time.time()
            
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            # Warm up model if not already done
            if not self._model_warmed_up:
                self._warm_up_model()
            
            # Use batch processor for optimized embedding generation
            if len(texts) > 8:  # Use batch processing for larger collections
                embeddings = self.batch_processor.process_texts_batch(texts, self.model)
                self._batch_processing_stats['total_batches'] += 1
                self._batch_processing_stats['total_texts_processed'] += len(texts)
            else:
                # For small collections, use direct encoding with caching
                if self.enable_caching and hasattr(self.batch_processor, 'embedding_cache'):
                    embeddings = self.batch_processor.embedding_cache.get_embeddings_batch(texts, self.model)
                    # Track cache performance
                    cache_info = self.batch_processor.embedding_cache.get_cache_info()
                    self._embedding_cache_hits += cache_info.get('cache_size', 0)
                else:
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    self._embedding_cache_misses += len(texts)
            
            # Update performance statistics
            processing_time = time.time() - start_time
            if self._batch_processing_stats['total_batches'] > 0:
                self._batch_processing_stats['average_batch_time'] = (
                    (self._batch_processing_stats['average_batch_time'] * (self._batch_processing_stats['total_batches'] - 1) + processing_time) /
                    self._batch_processing_stats['total_batches']
                )
            
            logger.debug(f"Generated embeddings with shape: {embeddings.shape} in {processing_time:.3f}s")
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _compute_similarity_matrix(self, 
                                 sentence_embeddings: np.ndarray, 
                                 skill_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between sentences and skills with optimization.
        
        Args:
            sentence_embeddings: Embeddings for job description sentences
            skill_embeddings: Embeddings for ontology skills
            
        Returns:
            Similarity matrix with shape (n_sentences, n_skills)
        """
        if sentence_embeddings.size == 0 or skill_embeddings.size == 0:
            return np.array([])
        
        try:
            # Estimate memory usage for similarity computation
            memory_estimate = MemoryManager.estimate_memory_usage(
                ['dummy'] * (sentence_embeddings.shape[0] + skill_embeddings.shape[0])
            )
            
            # Use chunked computation for large matrices to manage memory
            if memory_estimate['similarity_matrix_mb'] > 100:  # 100MB threshold
                logger.info(f"Using chunked similarity computation for large matrix "
                           f"({sentence_embeddings.shape[0]}x{skill_embeddings.shape[0]})")
                
                # Compute in chunks to manage memory
                chunk_size = max(1, int(100 * sentence_embeddings.shape[0] / memory_estimate['similarity_matrix_mb']))
                similarity_chunks = []
                
                for i in range(0, sentence_embeddings.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, sentence_embeddings.shape[0])
                    chunk_similarities = cosine_similarity(
                        sentence_embeddings[i:end_idx], 
                        skill_embeddings
                    )
                    similarity_chunks.append(chunk_similarities)
                    
                    # Force garbage collection between chunks
                    if len(similarity_chunks) % 5 == 0:
                        MemoryManager.force_garbage_collection()
                
                similarity_matrix = np.vstack(similarity_chunks)
            else:
                # Compute similarity matrix directly for smaller matrices
                similarity_matrix = cosine_similarity(sentence_embeddings, skill_embeddings)
            
            logger.debug(f"Computed similarity matrix with shape: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            error_msg = f"Failed to compute similarity matrix: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def find_skill_matches(self, 
                          sentences: List[str], 
                          ontology: Dict[str, List[str]], 
                          threshold: float = None) -> Dict[str, List[Tuple[str, float]]]:
        """Find semantic matches between job text and ontology skills.
        
        This method computes semantic similarity between job description sentences
        and skills from the ontology, returning matches above the specified threshold
        organized by category.
        
        Args:
            sentences: Preprocessed job description sentences
            ontology: Skills ontology dictionary with categories as keys
            threshold: Minimum similarity threshold (uses instance default if None)
            
        Returns:
            Dictionary mapping categories to lists of (skill, confidence) tuples
            sorted by confidence in descending order
            
        Raises:
            ModelLoadError: If semantic model operations fail
        """
        if threshold is None:
            threshold = self.threshold
        
        logger.info(f"Finding skill matches with threshold: {threshold}")
        
        if not sentences:
            logger.warning("No sentences provided for matching")
            return {category: [] for category in ontology.keys()}
        
        if not ontology:
            logger.warning("Empty ontology provided")
            return {}
        
        # Flatten all skills from ontology for batch embedding
        all_skills = []
        skill_to_category = {}
        
        for category, skills in ontology.items():
            for skill in skills:
                all_skills.append(skill)
                skill_to_category[skill] = category
        
        if not all_skills:
            logger.warning("No skills found in ontology")
            return {category: [] for category in ontology.keys()}
        
        try:
            # Generate embeddings for sentences and skills
            logger.debug(f"Processing {len(sentences)} sentences and {len(all_skills)} skills")
            sentence_embeddings = self._generate_embeddings(sentences)
            skill_embeddings = self._generate_embeddings(all_skills)
            
            # Compute similarity matrix
            similarity_matrix = self._compute_similarity_matrix(sentence_embeddings, skill_embeddings)
            
            # Find matches above threshold
            matches_by_category = {category: [] for category in ontology.keys()}
            
            # Get maximum similarity for each skill across all sentences
            if similarity_matrix.size > 0:
                max_similarities = np.max(similarity_matrix, axis=0)
                
                for skill_idx, skill in enumerate(all_skills):
                    max_similarity = max_similarities[skill_idx]
                    
                    if max_similarity >= threshold:
                        category = skill_to_category[skill]
                        matches_by_category[category].append((skill, float(max_similarity)))
                        logger.debug(f"Found match: {skill} ({category}) - {max_similarity:.3f}")
            
            # Sort matches by confidence (descending)
            for category in matches_by_category:
                matches_by_category[category].sort(key=lambda x: x[1], reverse=True)
            
            total_matches = sum(len(matches) for matches in matches_by_category.values())
            logger.info(f"Found {total_matches} skill matches above threshold {threshold}")
            
            return matches_by_category
            
        except Exception as e:
            error_msg = f"Failed to find skill matches: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def get_top_matches_per_category(self, 
                                   matches_by_category: Dict[str, List[Tuple[str, float]]], 
                                   top_n: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Get top N matches per category from skill matches.
        
        Args:
            matches_by_category: Dictionary of category to skill matches
            top_n: Maximum number of top matches to return per category
            
        Returns:
            Dictionary with top N matches per category
        """
        top_matches = {}
        for category, matches in matches_by_category.items():
            top_matches[category] = matches[:top_n]
        
        return top_matches
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "not_loaded", "model_name": self.model_name}
        
        info = {
            "status": "loaded",
            "model_name": self.model_name,
            "max_seq_length": str(self.model.max_seq_length),
            "embedding_dimension": str(self.model.get_sentence_embedding_dimension()),
            "caching_enabled": str(self.enable_caching)
        }
        
        # Add performance statistics if available
        if hasattr(self, 'batch_processor'):
            perf_stats = self.batch_processor.get_performance_stats()
            info.update({
                "batch_size": str(perf_stats.get('batch_size', 'unknown')),
                "max_memory_mb": str(perf_stats.get('max_memory_mb', 'unknown'))
            })
        
        return info
    
    def optimize_for_batch_processing(self, expected_batch_size: int, max_memory_mb: float = 512.0) -> None:
        """Optimize matcher configuration for batch processing.
        
        Args:
            expected_batch_size: Expected number of texts to process in batches
            max_memory_mb: Maximum memory usage limit in MB
        """
        logger.info(f"Optimizing for batch processing: batch_size={expected_batch_size}, "
                   f"max_memory_mb={max_memory_mb}")
        
        # Update batch processor configuration
        self.batch_processor = BatchProcessor(
            batch_size=min(expected_batch_size, 64),  # Cap at 64 for memory safety
            max_memory_mb=max_memory_mb,
            enable_caching=self.enable_caching
        )
        
        # Pre-warm the model cache if caching is enabled
        if self.enable_caching and self.model:
            logger.debug("Pre-warming model cache")
            # Generate a small embedding to ensure model is cached
            _ = self.model.encode(["test"], convert_to_numpy=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics.
        
        Returns:
            Dictionary with performance metrics and cache statistics
        """
        stats = {
            'model_name': self.model_name,
            'threshold': self.threshold,
            'caching_enabled': self.enable_caching,
            'model_loaded': self.model is not None,
            'model_warmed_up': self._model_warmed_up,
            'embedding_cache_hits': self._embedding_cache_hits,
            'embedding_cache_misses': self._embedding_cache_misses,
            'cache_hit_rate': (
                self._embedding_cache_hits / max(1, self._embedding_cache_hits + self._embedding_cache_misses)
            ),
            'batch_processing_stats': self._batch_processing_stats.copy()
        }
        
        if hasattr(self, 'batch_processor'):
            batch_stats = self.batch_processor.get_performance_stats()
            stats['batch_processor'] = batch_stats
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all performance caches to free memory."""
        if self.enable_caching and hasattr(self.batch_processor, 'embedding_cache'):
            self.batch_processor.embedding_cache.clear_cache()
            logger.info("Cleared embedding caches")
        
        # Reset performance statistics
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0
        self._batch_processing_stats = {
            'total_batches': 0,
            'total_texts_processed': 0,
            'average_batch_time': 0.0
        }
        
        # Force garbage collection
        MemoryManager.force_garbage_collection()
    
    def _warm_up_model(self) -> None:
        """Warm up the model by processing a small test input.
        
        This helps ensure the model is fully loaded and ready for processing,
        reducing latency for the first real request.
        """
        if self.model and not self._model_warmed_up:
            try:
                logger.debug("Warming up Sentence-BERT model")
                # Process a small test sentence to warm up the model
                test_texts = ["Software engineer with Python experience", "Data scientist with machine learning skills"]
                _ = self.model.encode(test_texts, convert_to_numpy=True)
                self._model_warmed_up = True
                logger.debug("Model warm-up completed successfully")
            except Exception as e:
                logger.warning(f"Model warm-up failed: {e}")
    
    def precompute_ontology_embeddings(self, ontology: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Precompute and cache embeddings for all ontology skills.
        
        This method precomputes embeddings for all skills in the ontology
        to improve performance during skill matching operations.
        
        Args:
            ontology: Skills ontology dictionary
            
        Returns:
            Dictionary mapping categories to skill embedding arrays
        """
        if not self.model:
            raise ModelLoadError("Model not loaded")
        
        logger.info("Precomputing ontology embeddings for performance optimization")
        
        ontology_embeddings = {}
        total_skills = sum(len(skills) for skills in ontology.values())
        processed_skills = 0
        
        for category, skills in ontology.items():
            if skills:
                logger.debug(f"Computing embeddings for {len(skills)} skills in category: {category}")
                
                # Use optimized embedding generation
                skill_embeddings = self._generate_embeddings(skills)
                ontology_embeddings[category] = skill_embeddings
                
                processed_skills += len(skills)
                logger.debug(f"Progress: {processed_skills}/{total_skills} skills processed")
        
        logger.info(f"Precomputed embeddings for {processed_skills} skills across {len(ontology_embeddings)} categories")
        return ontology_embeddings
    
    def find_skill_matches_optimized(self, 
                                   sentences: List[str], 
                                   ontology_embeddings: Dict[str, np.ndarray],
                                   ontology_skills: Dict[str, List[str]],
                                   threshold: float = None) -> Dict[str, List[Tuple[str, float]]]:
        """Optimized skill matching using precomputed ontology embeddings.
        
        This method uses precomputed ontology embeddings to significantly
        speed up skill matching operations, especially for repeated processing.
        
        Args:
            sentences: Preprocessed job description sentences
            ontology_embeddings: Precomputed embeddings for ontology skills
            ontology_skills: Original ontology skills for result mapping
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary mapping categories to lists of (skill, confidence) tuples
        """
        if threshold is None:
            threshold = self.threshold
        
        logger.info(f"Finding skill matches using precomputed embeddings with threshold: {threshold}")
        
        if not sentences:
            logger.warning("No sentences provided for matching")
            return {category: [] for category in ontology_skills.keys()}
        
        try:
            # Generate embeddings for sentences only
            sentence_embeddings = self._generate_embeddings(sentences)
            
            matches_by_category = {}
            
            for category, skill_embeddings in ontology_embeddings.items():
                if category not in ontology_skills or not ontology_skills[category]:
                    matches_by_category[category] = []
                    continue
                
                # Compute similarity matrix for this category
                similarity_matrix = self._compute_similarity_matrix(sentence_embeddings, skill_embeddings)
                
                # Find matches above threshold
                category_matches = []
                if similarity_matrix.size > 0:
                    max_similarities = np.max(similarity_matrix, axis=0)
                    
                    for skill_idx, skill in enumerate(ontology_skills[category]):
                        max_similarity = max_similarities[skill_idx]
                        
                        if max_similarity >= threshold:
                            category_matches.append((skill, float(max_similarity)))
                
                # Sort by confidence
                category_matches.sort(key=lambda x: x[1], reverse=True)
                matches_by_category[category] = category_matches
            
            total_matches = sum(len(matches) for matches in matches_by_category.values())
            logger.info(f"Found {total_matches} skill matches using optimized matching")
            
            return matches_by_category
            
        except Exception as e:
            logger.error(f"Optimized skill matching failed: {e}")
            # Fallback to regular matching
            logger.info("Falling back to regular skill matching")
            return self.find_skill_matches(sentences, ontology_skills, threshold)