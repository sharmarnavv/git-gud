"""
Performance optimization utilities for the Job Description Parser.

This module provides caching, memory management, and batch processing
optimizations to improve parser performance and scalability.
"""

import gc
import logging
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe cache for ML models to enable reuse across requests.
    
    This class implements a singleton pattern to ensure models are loaded
    once and reused across multiple parser instances, reducing memory usage
    and initialization time.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model cache."""
        if not self._initialized:
            self._models: Dict[str, SentenceTransformer] = {}
            self._model_locks: Dict[str, threading.Lock] = {}
            self._access_times: Dict[str, float] = {}
            self._max_models = 3  # Maximum number of cached models
            self._initialized = True
            logger.info("ModelCache initialized")
    
    def get_model(self, model_name: str) -> SentenceTransformer:
        """Get or load a Sentence-BERT model with caching.
        
        Args:
            model_name: Name of the Sentence-BERT model to load
            
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        # Create lock for this model if it doesn't exist
        if model_name not in self._model_locks:
            with self._lock:
                if model_name not in self._model_locks:
                    self._model_locks[model_name] = threading.Lock()
        
        # Check if model is already cached
        if model_name in self._models:
            self._access_times[model_name] = time.time()
            logger.debug(f"Retrieved cached model: {model_name}")
            return self._models[model_name]
        
        # Load model with thread safety
        with self._model_locks[model_name]:
            # Double-check pattern
            if model_name in self._models:
                self._access_times[model_name] = time.time()
                return self._models[model_name]
            
            try:
                logger.info(f"Loading new model into cache: {model_name}")
                
                # Check if we need to evict old models
                self._evict_if_needed()
                
                # Load the model
                model = SentenceTransformer(model_name)
                
                # Cache the model
                self._models[model_name] = model
                self._access_times[model_name] = time.time()
                
                logger.info(f"Successfully cached model: {model_name}")
                return model
                
            except Exception as e:
                error_msg = f"Failed to load model '{model_name}': {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used models if cache is full."""
        if len(self._models) >= self._max_models:
            # Find least recently used model
            lru_model = min(self._access_times.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Evicting LRU model from cache: {lru_model}")
            
            # Remove from cache
            del self._models[lru_model]
            del self._access_times[lru_model]
            del self._model_locks[lru_model]
            
            # Force garbage collection
            gc.collect()
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        with self._lock:
            logger.info("Clearing model cache")
            self._models.clear()
            self._access_times.clear()
            self._model_locks.clear()
            gc.collect()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_models': list(self._models.keys()),
            'cache_size': len(self._models),
            'max_cache_size': self._max_models,
            'access_times': dict(self._access_times)
        }


class EmbeddingCache:
    """LRU cache for sentence embeddings to avoid recomputation.
    
    This class caches embeddings for frequently used text to improve
    performance when processing similar job descriptions or skills.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        logger.info(f"EmbeddingCache initialized with max_size: {max_size}")
    
    def get_embedding(self, text: str, model: SentenceTransformer) -> np.ndarray:
        """Get or compute embedding for text.
        
        Args:
            text: Text to embed
            model: SentenceTransformer model to use
            
        Returns:
            Embedding vector as numpy array
        """
        # Create cache key (include model name for uniqueness)
        cache_key = f"{model._modules['0'].auto_model.name_or_path}:{hash(text)}"
        
        with self._lock:
            # Check if embedding is cached
            if cache_key in self._cache:
                self._access_times[cache_key] = time.time()
                logger.debug(f"Retrieved cached embedding for text hash: {hash(text)}")
                return self._cache[cache_key].copy()
            
            # Evict if cache is full
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Compute embedding
            embedding = model.encode([text], convert_to_numpy=True)[0]
            
            # Cache the embedding
            self._cache[cache_key] = embedding.copy()
            self._access_times[cache_key] = time.time()
            
            logger.debug(f"Cached new embedding for text hash: {hash(text)}")
            return embedding
    
    def get_embeddings_batch(self, texts: List[str], model: SentenceTransformer) -> np.ndarray:
        """Get or compute embeddings for multiple texts with partial caching.
        
        Args:
            texts: List of texts to embed
            model: SentenceTransformer model to use
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        model_name = model._modules['0'].auto_model.name_or_path
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        with self._lock:
            # Check which embeddings are cached
            for i, text in enumerate(texts):
                cache_key = f"{model_name}:{hash(text)}"
                
                if cache_key in self._cache:
                    self._access_times[cache_key] = time.time()
                    embeddings.append(self._cache[cache_key].copy())
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_compute.append(text)
                    indices_to_compute.append(i)
        
        # Compute missing embeddings in batch
        if texts_to_compute:
            logger.debug(f"Computing {len(texts_to_compute)} new embeddings out of {len(texts)} total")
            new_embeddings = model.encode(texts_to_compute, convert_to_numpy=True)
            
            with self._lock:
                # Cache new embeddings and update results
                for j, (text, embedding) in enumerate(zip(texts_to_compute, new_embeddings)):
                    cache_key = f"{model_name}:{hash(text)}"
                    
                    # Evict if needed
                    if len(self._cache) >= self.max_size:
                        self._evict_lru()
                    
                    # Cache the embedding
                    self._cache[cache_key] = embedding.copy()
                    self._access_times[cache_key] = time.time()
                    
                    # Update results
                    original_index = indices_to_compute[j]
                    embeddings[original_index] = embedding
        
        return np.array(embeddings)
    
    def _evict_lru(self) -> None:
        """Evict least recently used embedding."""
        if self._access_times:
            lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            del self._cache[lru_key]
            del self._access_times[lru_key]
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            gc.collect()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache information
        """
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.max_size,
                'hit_rate': len(self._cache) / max(1, len(self._access_times))
            }


class MemoryManager:
    """Memory management utilities for large text processing.
    
    This class provides utilities for managing memory usage during
    batch processing and large text operations.
    """
    
    @staticmethod
    def chunk_texts(texts: List[str], chunk_size: int = 32) -> List[List[str]]:
        """Split texts into smaller chunks for batch processing.
        
        Args:
            texts: List of texts to chunk
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunks.append(chunk)
        
        logger.debug(f"Split {len(texts)} texts into {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    @staticmethod
    def estimate_memory_usage(texts: List[str], embedding_dim: int = 384) -> Dict[str, float]:
        """Estimate memory usage for text processing.
        
        Args:
            texts: List of texts to process
            embedding_dim: Dimension of embeddings (default for all-MiniLM-L6-v2)
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Estimate text size (assuming UTF-8 encoding)
        text_size_bytes = sum(len(text.encode('utf-8')) for text in texts)
        
        # Estimate embedding size (float32 = 4 bytes per element)
        embedding_size_bytes = len(texts) * embedding_dim * 4
        
        # Estimate similarity matrix size (if computing pairwise similarities)
        similarity_matrix_bytes = len(texts) * len(texts) * 4
        
        return {
            'text_size_mb': text_size_bytes / (1024 * 1024),
            'embeddings_mb': embedding_size_bytes / (1024 * 1024),
            'similarity_matrix_mb': similarity_matrix_bytes / (1024 * 1024),
            'total_estimated_mb': (text_size_bytes + embedding_size_bytes + similarity_matrix_bytes) / (1024 * 1024)
        }
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return statistics.
        
        Returns:
            Dictionary with garbage collection statistics
        """
        logger.debug("Forcing garbage collection")
        
        # Get initial counts
        initial_counts = [len(gc.get_objects())]
        
        # Force collection for all generations
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        # Get final counts
        final_counts = [len(gc.get_objects())]
        
        stats = {
            'objects_before': initial_counts[0],
            'objects_after': final_counts[0],
            'objects_collected': initial_counts[0] - final_counts[0],
            'collections_by_generation': collected
        }
        
        logger.debug(f"GC stats: {stats}")
        return stats
    
    @staticmethod
    def optimize_memory_for_large_texts(texts: List[str], max_memory_mb: float = 512.0) -> Dict[str, Any]:
        """Optimize memory usage for processing large text collections.
        
        Args:
            texts: List of texts to analyze
            max_memory_mb: Maximum memory threshold in MB
            
        Returns:
            Dictionary with optimization recommendations
        """
        memory_estimate = MemoryManager.estimate_memory_usage(texts)
        
        recommendations = {
            'estimated_memory_mb': memory_estimate['total_estimated_mb'],
            'within_limits': memory_estimate['total_estimated_mb'] <= max_memory_mb,
            'recommended_chunk_size': len(texts),
            'should_use_chunking': False,
            'memory_optimization_needed': False
        }
        
        if memory_estimate['total_estimated_mb'] > max_memory_mb:
            # Calculate optimal chunk size
            memory_ratio = max_memory_mb / memory_estimate['total_estimated_mb']
            recommended_chunk_size = max(1, int(len(texts) * memory_ratio * 0.8))  # 20% safety margin
            
            recommendations.update({
                'recommended_chunk_size': recommended_chunk_size,
                'should_use_chunking': True,
                'memory_optimization_needed': True,
                'estimated_chunks': (len(texts) + recommended_chunk_size - 1) // recommended_chunk_size
            })
            
            logger.info(f"Memory optimization recommended: "
                       f"chunk size {recommended_chunk_size}, "
                       f"estimated {recommendations['estimated_chunks']} chunks")
        
        return recommendations
    
    @staticmethod
    def get_memory_usage_stats() -> Dict[str, Any]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            stats = {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'gc_objects': len(gc.get_objects())
            }
            
            return stats
            
        except ImportError:
            # Fallback if psutil is not available
            return {
                'gc_objects': len(gc.get_objects()),
                'note': 'Install psutil for detailed memory statistics'
            }


class BatchProcessor:
    """Optimized batch processing for multiple job descriptions.
    
    This class provides efficient batch processing capabilities with
    memory management and progress tracking.
    """
    
    def __init__(self, 
                 batch_size: int = 16,
                 max_memory_mb: float = 512.0,
                 enable_caching: bool = True):
        """Initialize batch processor.
        
        Args:
            batch_size: Default batch size for processing
            max_memory_mb: Maximum memory usage threshold in MB
            enable_caching: Whether to enable embedding caching
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.enable_caching = enable_caching
        
        # Initialize caches if enabled
        if enable_caching:
            self.model_cache = ModelCache()
            self.embedding_cache = EmbeddingCache()
        else:
            self.model_cache = None
            self.embedding_cache = None
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, "
                   f"max_memory_mb={max_memory_mb}, caching={enable_caching}")
    
    def process_texts_batch(self, 
                           texts: List[str],
                           model: SentenceTransformer,
                           progress_callback: Optional[callable] = None) -> np.ndarray:
        """Process texts in optimized batches with adaptive memory management.
        
        Args:
            texts: List of texts to process
            model: SentenceTransformer model to use
            progress_callback: Optional callback for progress updates
            
        Returns:
            Array of embeddings for all texts
        """
        if not texts:
            return np.array([])
        
        import time
        start_time = time.time()
        
        # Estimate memory usage and adjust batch size if needed
        memory_estimate = MemoryManager.estimate_memory_usage(texts)
        
        if memory_estimate['total_estimated_mb'] > self.max_memory_mb:
            # Reduce batch size to fit memory constraints
            adjusted_batch_size = max(1, int(self.batch_size * self.max_memory_mb / memory_estimate['total_estimated_mb']))
            logger.warning(f"Reducing batch size from {self.batch_size} to {adjusted_batch_size} "
                          f"due to memory constraints ({memory_estimate['total_estimated_mb']:.1f} MB estimated)")
            batch_size = adjusted_batch_size
        else:
            batch_size = self.batch_size
        
        # Process in chunks with adaptive sizing
        chunks = MemoryManager.chunk_texts(texts, batch_size)
        all_embeddings = []
        processing_times = []
        
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            
            if progress_callback:
                progress = (i + 1) / len(chunks)
                progress_callback(progress, f"Processing batch {i+1}/{len(chunks)}")
            
            # Use cached embeddings if available
            if self.enable_caching and self.embedding_cache:
                chunk_embeddings = self.embedding_cache.get_embeddings_batch(chunk, model)
            else:
                chunk_embeddings = model.encode(chunk, convert_to_numpy=True)
            
            all_embeddings.append(chunk_embeddings)
            
            # Track processing time for adaptive optimization
            chunk_time = time.time() - chunk_start_time
            processing_times.append(chunk_time)
            
            # Adaptive batch size adjustment based on processing time
            if len(processing_times) > 2:
                avg_time = sum(processing_times[-3:]) / 3
                if avg_time > 2.0 and batch_size > 4:  # If taking too long, reduce batch size
                    batch_size = max(4, int(batch_size * 0.8))
                    logger.debug(f"Reducing batch size to {batch_size} due to slow processing")
                elif avg_time < 0.5 and batch_size < self.batch_size:  # If too fast, increase batch size
                    batch_size = min(self.batch_size, int(batch_size * 1.2))
                    logger.debug(f"Increasing batch size to {batch_size} for better throughput")
            
            # Force garbage collection between large chunks
            if len(chunk) > 8 or (i + 1) % 5 == 0:  # Every 5 chunks or large chunks
                MemoryManager.force_garbage_collection()
        
        # Combine all embeddings
        if all_embeddings:
            result = np.vstack(all_embeddings)
            total_time = time.time() - start_time
            throughput = len(texts) / total_time
            
            logger.info(f"Processed {len(texts)} texts in {len(chunks)} batches, "
                       f"final shape: {result.shape}, "
                       f"time: {total_time:.2f}s, throughput: {throughput:.1f} texts/sec")
            return result
        else:
            return np.array([])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the batch processor.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'batch_size': self.batch_size,
            'max_memory_mb': self.max_memory_mb,
            'caching_enabled': self.enable_caching
        }
        
        if self.enable_caching:
            if self.model_cache:
                stats['model_cache'] = self.model_cache.get_cache_info()
            if self.embedding_cache:
                stats['embedding_cache'] = self.embedding_cache.get_cache_info()
        
        return stats


# Global instances for reuse
_model_cache = ModelCache()
_embedding_cache = EmbeddingCache()


def get_cached_model(model_name: str) -> SentenceTransformer:
    """Get a cached Sentence-BERT model.
    
    Args:
        model_name: Name of the model to retrieve
        
    Returns:
        Cached SentenceTransformer model
    """
    return _model_cache.get_model(model_name)


def clear_all_caches() -> None:
    """Clear all performance caches."""
    _model_cache.clear_cache()
    _embedding_cache.clear_cache()
    logger.info("Cleared all performance caches")


@lru_cache(maxsize=128)
def compute_similarity_threshold_stats(similarities: tuple, threshold: float) -> Dict[str, float]:
    """Compute statistics for similarity scores above threshold (cached).
    
    Args:
        similarities: Tuple of similarity scores (for hashability)
        threshold: Similarity threshold
        
    Returns:
        Dictionary with similarity statistics
    """
    similarities_array = np.array(similarities)
    above_threshold = similarities_array[similarities_array >= threshold]
    
    if len(above_threshold) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'count': len(above_threshold),
        'mean': float(np.mean(above_threshold)),
        'std': float(np.std(above_threshold)),
        'min': float(np.min(above_threshold)),
        'max': float(np.max(above_threshold))
    }