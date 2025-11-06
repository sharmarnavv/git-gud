"""
Similarity calculation engine for resume-job matching.

This module implements TF-IDF and SBERT-based similarity calculations
for comparing resumes against job descriptions with hybrid scoring.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from .resume_interfaces import ParsedResume
from job_parser.interfaces import ParsedJobDescription
from job_parser.semantic_matching import SemanticMatcher
from job_parser.logging_config import get_logger

logger = get_logger(__name__)


class TFIDFSimilarityCalculator:
    """TF-IDF based similarity calculator for resume-job matching.
    
    This class implements TF-IDF vectorization and cosine similarity calculation
    for fast keyword-based matching between resumes and job descriptions.
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 min_df: int = 1,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2),
                 stop_words: str = 'english'):
        """Initialize TF-IDF similarity calculator.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to extract
            stop_words: Stop words to remove
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9+#\.]*\b'  # Include technical terms like C++, C#
        )
        
        # Cache for fitted vectorizer and job description vectors
        self._fitted_vectorizer = None
        self._job_vector_cache = {}
        
        # Performance tracking
        self._vectorization_stats = {
            'total_vectorizations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("TF-IDF similarity calculator initialized")
    
    def calculate_similarity(self, 
                           resume_text: str, 
                           job_text: str,
                           keyword_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate TF-IDF similarity between resume and job description.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            keyword_weights: Optional weights for important keywords
            
        Returns:
            Dictionary with similarity score and detailed analysis
        """
        try:
            logger.debug("Calculating TF-IDF similarity")
            
            # Preprocess texts
            resume_processed = self._preprocess_text(resume_text)
            job_processed = self._preprocess_text(job_text)
            
            # Create document corpus
            documents = [resume_processed, job_processed]
            
            # Fit and transform documents
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            base_similarity = float(similarity_matrix[0, 1])  # Resume vs Job similarity
            
            # Apply keyword importance weighting if provided
            weighted_similarity = base_similarity
            if keyword_weights:
                weighted_similarity = self._apply_keyword_weighting(
                    resume_processed, job_processed, base_similarity, keyword_weights
                )
            
            # Get feature analysis
            feature_analysis = self._analyze_features(
                tfidf_matrix, resume_processed, job_processed
            )
            
            # Update performance stats
            self._vectorization_stats['total_vectorizations'] += 1
            
            result = {
                'similarity_score': weighted_similarity,
                'base_similarity': base_similarity,
                'feature_analysis': feature_analysis,
                'metadata': {
                    'resume_length': len(resume_text.split()),
                    'job_length': len(job_text.split()),
                    'vocabulary_size': len(self.vectorizer.vocabulary_),
                    'keyword_weighting_applied': keyword_weights is not None
                }
            }
            
            logger.debug(f"TF-IDF similarity calculated: {weighted_similarity:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"TF-IDF similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate TF-IDF similarity: {e}")
    
    def calculate_batch_similarity(self, 
                                 resume_texts: List[str], 
                                 job_text: str,
                                 keyword_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Calculate TF-IDF similarity for multiple resumes against one job.
        
        Args:
            resume_texts: List of resume text contents
            job_text: Job description text content
            keyword_weights: Optional weights for important keywords
            
        Returns:
            List of similarity results for each resume
        """
        try:
            logger.info(f"Calculating batch TF-IDF similarity for {len(resume_texts)} resumes")
            
            # Preprocess all texts
            job_processed = self._preprocess_text(job_text)
            resume_processed = [self._preprocess_text(text) for text in resume_texts]
            
            # Create document corpus (job + all resumes)
            documents = [job_processed] + resume_processed
            
            # Fit and transform documents
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Calculate similarities (job is at index 0, resumes start at index 1)
            job_vector = tfidf_matrix[0:1]  # Keep as matrix for broadcasting
            resume_vectors = tfidf_matrix[1:]
            
            # Calculate cosine similarities
            similarity_matrix = cosine_similarity(resume_vectors, job_vector)
            base_similarities = similarity_matrix.flatten()
            
            # Process results for each resume
            results = []
            for i, (resume_text, base_similarity) in enumerate(zip(resume_texts, base_similarities)):
                # Apply keyword weighting if provided
                weighted_similarity = base_similarity
                if keyword_weights:
                    weighted_similarity = self._apply_keyword_weighting(
                        resume_processed[i], job_processed, base_similarity, keyword_weights
                    )
                
                # Get feature analysis for this resume
                resume_vector = tfidf_matrix[i + 1:i + 2]  # +1 because job is at index 0
                feature_analysis = self._analyze_features_single(
                    resume_vector, job_vector, resume_processed[i], job_processed
                )
                
                result = {
                    'similarity_score': float(weighted_similarity),
                    'base_similarity': float(base_similarity),
                    'feature_analysis': feature_analysis,
                    'metadata': {
                        'resume_index': i,
                        'resume_length': len(resume_text.split()),
                        'job_length': len(job_text.split()),
                        'vocabulary_size': len(self.vectorizer.vocabulary_),
                        'keyword_weighting_applied': keyword_weights is not None
                    }
                }
                results.append(result)
            
            # Update performance stats
            self._vectorization_stats['total_vectorizations'] += len(resume_texts)
            
            logger.info(f"Batch TF-IDF similarity calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"Batch TF-IDF similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate batch TF-IDF similarity: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF vectorization.
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        processed = text.lower()
        
        # Remove extra whitespace and normalize
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove special characters but keep technical terms
        processed = re.sub(r'[^\w\s+#\.\-]', ' ', processed)
        
        # Normalize common technical terms
        tech_normalizations = {
            r'\bc\+\+\b': 'cplusplus',
            r'\bc#\b': 'csharp',
            r'\bf#\b': 'fsharp',
            r'\bnode\.js\b': 'nodejs',
            r'\breact\.js\b': 'reactjs',
            r'\bvue\.js\b': 'vuejs',
            r'\bangular\.js\b': 'angularjs'
        }
        
        for pattern, replacement in tech_normalizations.items():
            processed = re.sub(pattern, replacement, processed)
        
        return processed.strip()
    
    def _apply_keyword_weighting(self, 
                               resume_text: str, 
                               job_text: str, 
                               base_similarity: float,
                               keyword_weights: Dict[str, float]) -> float:
        """Apply keyword importance weighting to similarity score.
        
        Args:
            resume_text: Preprocessed resume text
            job_text: Preprocessed job text
            base_similarity: Base TF-IDF similarity score
            keyword_weights: Dictionary of keyword weights
            
        Returns:
            Weighted similarity score
        """
        try:
            # Extract keywords from job description
            job_keywords = set(job_text.lower().split())
            resume_keywords = set(resume_text.lower().split())
            
            # Calculate keyword match bonus
            total_weight = 0.0
            matched_weight = 0.0
            
            for keyword, weight in keyword_weights.items():
                keyword_lower = keyword.lower()
                total_weight += weight
                
                # Check if keyword appears in both texts
                if keyword_lower in job_keywords and keyword_lower in resume_keywords:
                    matched_weight += weight
            
            # Calculate keyword match ratio
            keyword_match_ratio = matched_weight / total_weight if total_weight > 0 else 0.0
            
            # Apply weighting (boost similarity based on important keyword matches)
            weight_boost = keyword_match_ratio * 0.2  # Max 20% boost
            weighted_similarity = min(1.0, base_similarity + weight_boost)
            
            logger.debug(f"Keyword weighting applied: {base_similarity:.3f} -> {weighted_similarity:.3f}")
            return weighted_similarity
            
        except Exception as e:
            logger.warning(f"Keyword weighting failed: {e}")
            return base_similarity
    
    def _analyze_features(self, 
                         tfidf_matrix: np.ndarray, 
                         resume_text: str, 
                         job_text: str) -> Dict[str, Any]:
        """Analyze TF-IDF features for detailed similarity breakdown.
        
        Args:
            tfidf_matrix: TF-IDF matrix for documents
            resume_text: Preprocessed resume text
            job_text: Preprocessed job text
            
        Returns:
            Dictionary with feature analysis
        """
        try:
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get vectors for resume and job
            resume_vector = tfidf_matrix[0].toarray().flatten()
            job_vector = tfidf_matrix[1].toarray().flatten()
            
            # Find top features for each document
            resume_top_indices = np.argsort(resume_vector)[-10:][::-1]
            job_top_indices = np.argsort(job_vector)[-10:][::-1]
            
            resume_top_features = [
                (feature_names[i], float(resume_vector[i])) 
                for i in resume_top_indices if resume_vector[i] > 0
            ]
            
            job_top_features = [
                (feature_names[i], float(job_vector[i])) 
                for i in job_top_indices if job_vector[i] > 0
            ]
            
            # Find common features
            common_features = []
            for i, feature in enumerate(feature_names):
                if resume_vector[i] > 0 and job_vector[i] > 0:
                    common_score = min(resume_vector[i], job_vector[i])
                    common_features.append((feature, float(common_score)))
            
            # Sort common features by score
            common_features.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'resume_top_features': resume_top_features[:5],
                'job_top_features': job_top_features[:5],
                'common_features': common_features[:10],
                'total_features': len(feature_names),
                'resume_feature_count': int(np.sum(resume_vector > 0)),
                'job_feature_count': int(np.sum(job_vector > 0)),
                'common_feature_count': len(common_features)
            }
            
        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            return {
                'resume_top_features': [],
                'job_top_features': [],
                'common_features': [],
                'total_features': 0,
                'resume_feature_count': 0,
                'job_feature_count': 0,
                'common_feature_count': 0
            }
    
    def _analyze_features_single(self, 
                               resume_vector: np.ndarray, 
                               job_vector: np.ndarray,
                               resume_text: str, 
                               job_text: str) -> Dict[str, Any]:
        """Analyze features for a single resume-job pair in batch processing.
        
        Args:
            resume_vector: TF-IDF vector for resume
            job_vector: TF-IDF vector for job
            resume_text: Preprocessed resume text
            job_text: Preprocessed job text
            
        Returns:
            Dictionary with feature analysis
        """
        try:
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Flatten vectors
            resume_vec = resume_vector.toarray().flatten()
            job_vec = job_vector.toarray().flatten()
            
            # Find common features
            common_features = []
            for i, feature in enumerate(feature_names):
                if resume_vec[i] > 0 and job_vec[i] > 0:
                    common_score = min(resume_vec[i], job_vec[i])
                    common_features.append((feature, float(common_score)))
            
            # Sort by score
            common_features.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'common_features': common_features[:5],  # Top 5 for batch processing
                'common_feature_count': len(common_features),
                'resume_feature_count': int(np.sum(resume_vec > 0)),
                'job_feature_count': int(np.sum(job_vec > 0))
            }
            
        except Exception as e:
            logger.warning(f"Single feature analysis failed: {e}")
            return {
                'common_features': [],
                'common_feature_count': 0,
                'resume_feature_count': 0,
                'job_feature_count': 0
            }
    
    def extract_job_keywords(self, job_text: str, top_n: int = 20) -> Dict[str, float]:
        """Extract important keywords from job description for weighting.
        
        Args:
            job_text: Job description text
            top_n: Number of top keywords to extract
            
        Returns:
            Dictionary mapping keywords to importance weights
        """
        try:
            # Preprocess job text
            processed_text = self._preprocess_text(job_text)
            
            # Fit vectorizer on job text
            tfidf_matrix = self.vectorizer.fit_transform([processed_text])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray().flatten()
            
            # Get top features
            top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
            
            # Create keyword weights (normalize to 0-1 range)
            max_score = np.max(tfidf_scores) if len(tfidf_scores) > 0 else 1.0
            keyword_weights = {}
            
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    keyword = feature_names[idx]
                    weight = float(tfidf_scores[idx] / max_score)
                    keyword_weights[keyword] = weight
            
            logger.debug(f"Extracted {len(keyword_weights)} job keywords")
            return keyword_weights
            
        except Exception as e:
            logger.error(f"Job keyword extraction failed: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the TF-IDF calculator.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._vectorization_stats.copy()
        
        # Calculate cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        # Add configuration info
        stats['configuration'] = {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'stop_words': self.stop_words
        }
        
        return stats
    
    def clear_cache(self):
        """Clear vectorizer cache to free memory."""
        self._fitted_vectorizer = None
        self._job_vector_cache.clear()
        
        # Reset performance stats
        self._vectorization_stats = {
            'total_vectorizations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("TF-IDF calculator cache cleared")


class SBERTSimilarityCalculator:
    """SBERT-based semantic similarity calculator for resume-job matching.
    
    This class extends the existing semantic matching capabilities to handle
    resume-job comparisons with sentence-level similarity and caching.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 enable_caching: bool = True,
                 batch_size: int = 32):
        """Initialize SBERT similarity calculator.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
            enable_caching: Whether to enable embedding caching
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.batch_size = batch_size
        
        # Initialize semantic matcher from existing job parser
        try:
            self.semantic_matcher = SemanticMatcher(
                model_name=model_name,
                threshold=0.0,  # We'll handle thresholding separately
                enable_caching=enable_caching,
                batch_size=batch_size
            )
            logger.info(f"SBERT similarity calculator initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SBERT calculator: {e}")
            raise ValueError(f"SBERT initialization failed: {e}")
        
        # Embedding cache for resume-job pairs
        self._embedding_cache = {}
        self._similarity_cache = {}
        
        # Performance tracking
        self._calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'sentence_level_calculations': 0,
            'document_level_calculations': 0
        }
    
    def calculate_similarity(self, 
                           resume_text: str, 
                           job_text: str,
                           use_sentence_level: bool = True) -> Dict[str, Any]:
        """Calculate SBERT semantic similarity between resume and job description.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            use_sentence_level: Whether to use sentence-level similarity
            
        Returns:
            Dictionary with similarity score and detailed analysis
        """
        try:
            logger.debug("Calculating SBERT semantic similarity")
            
            # Check cache first
            cache_key = self._generate_cache_key(resume_text, job_text, use_sentence_level)
            if self.enable_caching and cache_key in self._similarity_cache:
                self._calculation_stats['cache_hits'] += 1
                logger.debug("SBERT similarity retrieved from cache")
                return self._similarity_cache[cache_key]
            
            self._calculation_stats['cache_misses'] += 1
            
            if use_sentence_level:
                result = self._calculate_sentence_level_similarity(resume_text, job_text)
                self._calculation_stats['sentence_level_calculations'] += 1
            else:
                result = self._calculate_document_level_similarity(resume_text, job_text)
                self._calculation_stats['document_level_calculations'] += 1
            
            # Cache result
            if self.enable_caching:
                self._similarity_cache[cache_key] = result
            
            # Update performance stats
            self._calculation_stats['total_calculations'] += 1
            
            logger.debug(f"SBERT similarity calculated: {result['similarity_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"SBERT similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate SBERT similarity: {e}")
    
    def calculate_batch_similarity(self, 
                                 resume_texts: List[str], 
                                 job_text: str,
                                 use_sentence_level: bool = True) -> List[Dict[str, Any]]:
        """Calculate SBERT similarity for multiple resumes against one job.
        
        Args:
            resume_texts: List of resume text contents
            job_text: Job description text content
            use_sentence_level: Whether to use sentence-level similarity
            
        Returns:
            List of similarity results for each resume
        """
        try:
            logger.info(f"Calculating batch SBERT similarity for {len(resume_texts)} resumes")
            
            results = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(resume_texts), self.batch_size):
                batch_end = min(i + self.batch_size, len(resume_texts))
                batch_resumes = resume_texts[i:batch_end]
                
                logger.debug(f"Processing SBERT batch {i//self.batch_size + 1}")
                
                # Calculate similarity for each resume in batch
                batch_results = []
                for j, resume_text in enumerate(batch_resumes):
                    try:
                        result = self.calculate_similarity(resume_text, job_text, use_sentence_level)
                        result['metadata']['resume_index'] = i + j
                        batch_results.append(result)
                    except Exception as e:
                        logger.warning(f"SBERT calculation failed for resume {i+j}: {e}")
                        # Create error result
                        error_result = {
                            'similarity_score': 0.0,
                            'confidence_score': 0.0,
                            'sentence_similarities': [],
                            'metadata': {
                                'resume_index': i + j,
                                'error': str(e),
                                'calculation_failed': True
                            }
                        }
                        batch_results.append(error_result)
                
                results.extend(batch_results)
            
            logger.info(f"Batch SBERT similarity calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"Batch SBERT similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate batch SBERT similarity: {e}")
    
    def _calculate_sentence_level_similarity(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Calculate sentence-level semantic similarity.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            
        Returns:
            Dictionary with detailed sentence-level similarity analysis
        """
        try:
            # Split texts into sentences
            resume_sentences = self._split_into_sentences(resume_text)
            job_sentences = self._split_into_sentences(job_text)
            
            if not resume_sentences or not job_sentences:
                return self._create_empty_result("Empty sentences")
            
            # Generate embeddings for all sentences
            all_sentences = resume_sentences + job_sentences
            embeddings = self.semantic_matcher._generate_embeddings(all_sentences)
            
            # Split embeddings
            resume_embeddings = embeddings[:len(resume_sentences)]
            job_embeddings = embeddings[len(resume_sentences):]
            
            # Calculate similarity matrix between all resume and job sentences
            similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
            
            # Analyze sentence-level similarities
            sentence_similarities = []
            max_similarities = []
            
            for i, resume_sentence in enumerate(resume_sentences):
                sentence_sims = similarity_matrix[i]
                max_sim_idx = np.argmax(sentence_sims)
                max_similarity = float(sentence_sims[max_sim_idx])
                
                sentence_similarities.append({
                    'resume_sentence': resume_sentence[:100] + "..." if len(resume_sentence) > 100 else resume_sentence,
                    'best_match_job_sentence': job_sentences[max_sim_idx][:100] + "..." if len(job_sentences[max_sim_idx]) > 100 else job_sentences[max_sim_idx],
                    'similarity': max_similarity
                })
                
                max_similarities.append(max_similarity)
            
            # Calculate overall similarity scores
            overall_similarity = float(np.mean(max_similarities))
            confidence_score = self._calculate_confidence_score(similarity_matrix, max_similarities)
            
            # Find top sentence matches
            top_matches = sorted(sentence_similarities, key=lambda x: x['similarity'], reverse=True)[:5]
            
            return {
                'similarity_score': overall_similarity,
                'confidence_score': confidence_score,
                'sentence_similarities': top_matches,
                'metadata': {
                    'calculation_type': 'sentence_level',
                    'resume_sentences': len(resume_sentences),
                    'job_sentences': len(job_sentences),
                    'max_sentence_similarity': float(np.max(max_similarities)),
                    'min_sentence_similarity': float(np.min(max_similarities)),
                    'std_sentence_similarity': float(np.std(max_similarities))
                }
            }
            
        except Exception as e:
            logger.error(f"Sentence-level similarity calculation failed: {e}")
            return self._create_empty_result(f"Calculation error: {e}")
    
    def _calculate_document_level_similarity(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Calculate document-level semantic similarity.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            
        Returns:
            Dictionary with document-level similarity analysis
        """
        try:
            # Generate embeddings for entire documents
            documents = [resume_text, job_text]
            embeddings = self.semantic_matcher._generate_embeddings(documents)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            similarity_score = float(similarity_matrix[0, 1])
            
            # Calculate confidence based on embedding quality
            resume_embedding = embeddings[0]
            job_embedding = embeddings[1]
            
            # Simple confidence metric based on embedding norms
            resume_norm = float(np.linalg.norm(resume_embedding))
            job_norm = float(np.linalg.norm(job_embedding))
            confidence_score = min(resume_norm, job_norm) / max(resume_norm, job_norm)
            
            return {
                'similarity_score': similarity_score,
                'confidence_score': confidence_score,
                'sentence_similarities': [],  # Empty for document-level
                'metadata': {
                    'calculation_type': 'document_level',
                    'resume_length': len(resume_text.split()),
                    'job_length': len(job_text.split()),
                    'resume_embedding_norm': resume_norm,
                    'job_embedding_norm': job_norm
                }
            }
            
        except Exception as e:
            logger.error(f"Document-level similarity calculation failed: {e}")
            return self._create_empty_result(f"Calculation error: {e}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for sentence-level analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_confidence_score(self, similarity_matrix: np.ndarray, max_similarities: List[float]) -> float:
        """Calculate confidence score for sentence-level similarity.
        
        Args:
            similarity_matrix: Matrix of sentence similarities
            max_similarities: List of maximum similarities for each resume sentence
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Calculate confidence based on consistency of similarities
            mean_sim = np.mean(max_similarities)
            std_sim = np.std(max_similarities)
            
            # Higher confidence for consistent high similarities
            consistency_score = 1.0 - min(std_sim / (mean_sim + 0.1), 1.0)
            
            # Adjust based on overall similarity level
            level_score = mean_sim
            
            # Combine scores
            confidence = (consistency_score * 0.4 + level_score * 0.6)
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create empty result for failed calculations.
        
        Args:
            reason: Reason for empty result
            
        Returns:
            Empty result dictionary
        """
        return {
            'similarity_score': 0.0,
            'confidence_score': 0.0,
            'sentence_similarities': [],
            'metadata': {
                'calculation_type': 'failed',
                'error_reason': reason
            }
        }
    
    def _generate_cache_key(self, resume_text: str, job_text: str, use_sentence_level: bool) -> str:
        """Generate cache key for similarity calculation.
        
        Args:
            resume_text: Resume text
            job_text: Job text
            use_sentence_level: Whether using sentence-level calculation
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create hash of inputs
        content = f"{resume_text}|{job_text}|{use_sentence_level}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def precompute_job_embeddings(self, job_text: str) -> Dict[str, np.ndarray]:
        """Precompute embeddings for job description to optimize repeated comparisons.
        
        Args:
            job_text: Job description text
            
        Returns:
            Dictionary with precomputed embeddings
        """
        try:
            logger.info("Precomputing job embeddings for optimization")
            
            # Split job text into sentences
            job_sentences = self._split_into_sentences(job_text)
            
            # Generate embeddings
            sentence_embeddings = self.semantic_matcher._generate_embeddings(job_sentences)
            document_embedding = self.semantic_matcher._generate_embeddings([job_text])
            
            embeddings = {
                'sentences': sentence_embeddings,
                'document': document_embedding,
                'sentence_texts': job_sentences,
                'document_text': job_text
            }
            
            logger.info(f"Precomputed embeddings for {len(job_sentences)} sentences")
            return embeddings
            
        except Exception as e:
            logger.error(f"Job embedding precomputation failed: {e}")
            return {}
    
    def calculate_similarity_with_precomputed(self, 
                                            resume_text: str, 
                                            job_embeddings: Dict[str, np.ndarray],
                                            use_sentence_level: bool = True) -> Dict[str, Any]:
        """Calculate similarity using precomputed job embeddings.
        
        Args:
            resume_text: Resume text content
            job_embeddings: Precomputed job embeddings
            use_sentence_level: Whether to use sentence-level similarity
            
        Returns:
            Dictionary with similarity score and analysis
        """
        try:
            if not job_embeddings:
                # Fallback to regular calculation
                return self.calculate_similarity(resume_text, job_embeddings.get('document_text', ''), use_sentence_level)
            
            if use_sentence_level and 'sentences' in job_embeddings:
                return self._calculate_sentence_level_with_precomputed(resume_text, job_embeddings)
            else:
                return self._calculate_document_level_with_precomputed(resume_text, job_embeddings)
                
        except Exception as e:
            logger.error(f"Precomputed similarity calculation failed: {e}")
            return self._create_empty_result(f"Precomputed calculation error: {e}")
    
    def _calculate_sentence_level_with_precomputed(self, resume_text: str, job_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate sentence-level similarity with precomputed job embeddings."""
        try:
            # Split resume into sentences
            resume_sentences = self._split_into_sentences(resume_text)
            
            if not resume_sentences:
                return self._create_empty_result("No resume sentences")
            
            # Generate embeddings for resume sentences
            resume_embeddings = self.semantic_matcher._generate_embeddings(resume_sentences)
            
            # Get precomputed job embeddings
            job_sentence_embeddings = job_embeddings['sentences']
            job_sentences = job_embeddings['sentence_texts']
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(resume_embeddings, job_sentence_embeddings)
            
            # Process results (same as regular sentence-level calculation)
            sentence_similarities = []
            max_similarities = []
            
            for i, resume_sentence in enumerate(resume_sentences):
                sentence_sims = similarity_matrix[i]
                max_sim_idx = np.argmax(sentence_sims)
                max_similarity = float(sentence_sims[max_sim_idx])
                
                sentence_similarities.append({
                    'resume_sentence': resume_sentence[:100] + "..." if len(resume_sentence) > 100 else resume_sentence,
                    'best_match_job_sentence': job_sentences[max_sim_idx][:100] + "..." if len(job_sentences[max_sim_idx]) > 100 else job_sentences[max_sim_idx],
                    'similarity': max_similarity
                })
                
                max_similarities.append(max_similarity)
            
            # Calculate scores
            overall_similarity = float(np.mean(max_similarities))
            confidence_score = self._calculate_confidence_score(similarity_matrix, max_similarities)
            
            # Top matches
            top_matches = sorted(sentence_similarities, key=lambda x: x['similarity'], reverse=True)[:5]
            
            return {
                'similarity_score': overall_similarity,
                'confidence_score': confidence_score,
                'sentence_similarities': top_matches,
                'metadata': {
                    'calculation_type': 'sentence_level_precomputed',
                    'resume_sentences': len(resume_sentences),
                    'job_sentences': len(job_sentences),
                    'max_sentence_similarity': float(np.max(max_similarities)),
                    'min_sentence_similarity': float(np.min(max_similarities))
                }
            }
            
        except Exception as e:
            logger.error(f"Precomputed sentence-level calculation failed: {e}")
            return self._create_empty_result(f"Precomputed error: {e}")
    
    def _calculate_document_level_with_precomputed(self, resume_text: str, job_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate document-level similarity with precomputed job embeddings."""
        try:
            # Generate embedding for resume
            resume_embedding = self.semantic_matcher._generate_embeddings([resume_text])
            
            # Get precomputed job embedding
            job_embedding = job_embeddings['document']
            
            # Calculate similarity
            similarity_matrix = cosine_similarity(resume_embedding, job_embedding)
            similarity_score = float(similarity_matrix[0, 0])
            
            # Calculate confidence
            resume_norm = float(np.linalg.norm(resume_embedding[0]))
            job_norm = float(np.linalg.norm(job_embedding[0]))
            confidence_score = min(resume_norm, job_norm) / max(resume_norm, job_norm)
            
            return {
                'similarity_score': similarity_score,
                'confidence_score': confidence_score,
                'sentence_similarities': [],
                'metadata': {
                    'calculation_type': 'document_level_precomputed',
                    'resume_length': len(resume_text.split()),
                    'resume_embedding_norm': resume_norm,
                    'job_embedding_norm': job_norm
                }
            }
            
        except Exception as e:
            logger.error(f"Precomputed document-level calculation failed: {e}")
            return self._create_empty_result(f"Precomputed error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the SBERT calculator.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._calculation_stats.copy()
        
        # Calculate cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        # Add configuration info
        stats['configuration'] = {
            'model_name': self.model_name,
            'enable_caching': self.enable_caching,
            'batch_size': self.batch_size
        }
        
        # Add semantic matcher stats if available
        if hasattr(self.semantic_matcher, 'get_performance_stats'):
            stats['semantic_matcher'] = self.semantic_matcher.get_performance_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._embedding_cache.clear()
        self._similarity_cache.clear()
        
        # Clear semantic matcher cache
        if hasattr(self.semantic_matcher, 'clear_caches'):
            self.semantic_matcher.clear_caches()
        
        # Reset performance stats
        self._calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'sentence_level_calculations': 0,
            'document_level_calculations': 0
        }
        
        logger.info("SBERT calculator cache cleared")


class HybridSimilarityCalculator:
    """Hybrid similarity calculator combining TF-IDF and SBERT approaches.
    
    This class implements weighted combination of TF-IDF and SBERT scores with
    dynamic weight adjustment based on content type and length, providing
    overall similarity scores from 0-100% with normalization and calibration.
    """
    
    def __init__(self,
                 tfidf_config: Optional[Dict[str, Any]] = None,
                 sbert_config: Optional[Dict[str, Any]] = None,
                 default_tfidf_weight: float = 0.4,
                 default_sbert_weight: float = 0.6,
                 enable_dynamic_weighting: bool = True,
                 score_calibration: bool = True):
        """Initialize hybrid similarity calculator.
        
        Args:
            tfidf_config: Configuration for TF-IDF calculator
            sbert_config: Configuration for SBERT calculator
            default_tfidf_weight: Default weight for TF-IDF scores (0-1)
            default_sbert_weight: Default weight for SBERT scores (0-1)
            enable_dynamic_weighting: Whether to adjust weights based on content
            score_calibration: Whether to apply score calibration
        """
        # Validate weights
        if abs(default_tfidf_weight + default_sbert_weight - 1.0) > 0.001:
            raise ValueError("TF-IDF and SBERT weights must sum to 1.0")
        
        self.default_tfidf_weight = default_tfidf_weight
        self.default_sbert_weight = default_sbert_weight
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.score_calibration = score_calibration
        
        # Initialize component calculators
        tfidf_config = tfidf_config or {}
        sbert_config = sbert_config or {}
        
        try:
            self.tfidf_calculator = TFIDFSimilarityCalculator(**tfidf_config)
            self.sbert_calculator = SBERTSimilarityCalculator(**sbert_config)
            logger.info("Hybrid similarity calculator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid calculator: {e}")
            raise ValueError(f"Hybrid calculator initialization failed: {e}")
        
        # Calibration parameters (learned from training data or set empirically)
        self._calibration_params = {
            'tfidf_scale': 1.2,      # Scale TF-IDF scores slightly up
            'sbert_scale': 0.9,      # Scale SBERT scores slightly down
            'min_score': 0.0,        # Minimum possible score
            'max_score': 1.0,        # Maximum possible score
            'sigmoid_steepness': 5.0  # Steepness of sigmoid normalization
        }
        
        # Dynamic weighting parameters
        self._weighting_params = {
            'short_text_threshold': 50,    # Words threshold for short text
            'long_text_threshold': 500,    # Words threshold for long text
            'technical_keywords': {        # Keywords that favor TF-IDF
                'python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
                'react', 'angular', 'vue', 'tensorflow', 'pytorch', 'machine learning',
                'data science', 'devops', 'ci/cd', 'agile', 'scrum'
            },
            'soft_skill_keywords': {       # Keywords that favor SBERT
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical', 'creative', 'innovative', 'collaborative',
                'adaptable', 'motivated', 'organized', 'detail-oriented'
            }
        }
        
        # Performance tracking
        self._calculation_stats = {
            'total_calculations': 0,
            'dynamic_weight_adjustments': 0,
            'calibration_applications': 0,
            'average_tfidf_weight': 0.0,
            'average_sbert_weight': 0.0,
            'score_distribution': {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0}
        }
    
    def calculate_similarity(self,
                           resume_text: str,
                           job_text: str,
                           custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate hybrid similarity between resume and job description.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            custom_weights: Optional custom weights {'tfidf': float, 'sbert': float}
            
        Returns:
            Dictionary with hybrid similarity score (0-100%) and detailed analysis
        """
        try:
            logger.debug("Calculating hybrid similarity")
            
            # Calculate individual similarities
            tfidf_result = self.tfidf_calculator.calculate_similarity(resume_text, job_text)
            sbert_result = self.sbert_calculator.calculate_similarity(resume_text, job_text)
            
            # Extract base scores
            tfidf_score = tfidf_result['similarity_score']
            sbert_score = sbert_result['similarity_score']
            
            # Determine weights
            if custom_weights:
                tfidf_weight = custom_weights.get('tfidf', self.default_tfidf_weight)
                sbert_weight = custom_weights.get('sbert', self.default_sbert_weight)
                # Normalize weights
                total_weight = tfidf_weight + sbert_weight
                if total_weight > 0:
                    tfidf_weight /= total_weight
                    sbert_weight /= total_weight
                else:
                    tfidf_weight = self.default_tfidf_weight
                    sbert_weight = self.default_sbert_weight
            elif self.enable_dynamic_weighting:
                tfidf_weight, sbert_weight = self._calculate_dynamic_weights(
                    resume_text, job_text, tfidf_result, sbert_result
                )
                self._calculation_stats['dynamic_weight_adjustments'] += 1
            else:
                tfidf_weight = self.default_tfidf_weight
                sbert_weight = self.default_sbert_weight
            
            # Apply score calibration if enabled
            if self.score_calibration:
                tfidf_score = self._calibrate_score(tfidf_score, 'tfidf')
                sbert_score = self._calibrate_score(sbert_score, 'sbert')
                self._calculation_stats['calibration_applications'] += 1
            
            # Calculate weighted hybrid score
            hybrid_score = (tfidf_weight * tfidf_score) + (sbert_weight * sbert_score)
            
            # Normalize to 0-100% range
            hybrid_score_percent = self._normalize_to_percentage(hybrid_score)
            
            # Update performance statistics
            self._update_performance_stats(tfidf_weight, sbert_weight, hybrid_score_percent)
            
            # Create comprehensive result
            result = {
                'similarity_score': hybrid_score_percent,
                'component_scores': {
                    'tfidf_score': float(tfidf_score * 100),
                    'sbert_score': float(sbert_score * 100),
                    'tfidf_weight': float(tfidf_weight),
                    'sbert_weight': float(sbert_weight)
                },
                'detailed_analysis': {
                    'tfidf_analysis': tfidf_result,
                    'sbert_analysis': sbert_result
                },
                'metadata': {
                    'calculation_type': 'hybrid',
                    'dynamic_weighting_used': self.enable_dynamic_weighting and not custom_weights,
                    'score_calibration_applied': self.score_calibration,
                    'resume_length': len(resume_text.split()),
                    'job_length': len(job_text.split()),
                    'content_analysis': self._analyze_content_characteristics(resume_text, job_text)
                }
            }
            
            logger.debug(f"Hybrid similarity calculated: {hybrid_score_percent:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate hybrid similarity: {e}")
    
    def calculate_batch_similarity(self,
                                 resume_texts: List[str],
                                 job_text: str,
                                 custom_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Calculate hybrid similarity for multiple resumes against one job.
        
        Args:
            resume_texts: List of resume text contents
            job_text: Job description text content
            custom_weights: Optional custom weights for all calculations
            
        Returns:
            List of hybrid similarity results for each resume
        """
        try:
            logger.info(f"Calculating batch hybrid similarity for {len(resume_texts)} resumes")
            
            # Calculate batch similarities for both methods
            tfidf_results = self.tfidf_calculator.calculate_batch_similarity(resume_texts, job_text)
            sbert_results = self.sbert_calculator.calculate_batch_similarity(resume_texts, job_text)
            
            # Process each resume
            results = []
            for i, (resume_text, tfidf_result, sbert_result) in enumerate(zip(resume_texts, tfidf_results, sbert_results)):
                try:
                    # Extract base scores
                    tfidf_score = tfidf_result['similarity_score']
                    sbert_score = sbert_result['similarity_score']
                    
                    # Determine weights (same logic as single calculation)
                    if custom_weights:
                        tfidf_weight = custom_weights.get('tfidf', self.default_tfidf_weight)
                        sbert_weight = custom_weights.get('sbert', self.default_sbert_weight)
                        # Normalize weights
                        total_weight = tfidf_weight + sbert_weight
                        if total_weight > 0:
                            tfidf_weight /= total_weight
                            sbert_weight /= total_weight
                        else:
                            tfidf_weight = self.default_tfidf_weight
                            sbert_weight = self.default_sbert_weight
                    elif self.enable_dynamic_weighting:
                        tfidf_weight, sbert_weight = self._calculate_dynamic_weights(
                            resume_text, job_text, tfidf_result, sbert_result
                        )
                        self._calculation_stats['dynamic_weight_adjustments'] += 1
                    else:
                        tfidf_weight = self.default_tfidf_weight
                        sbert_weight = self.default_sbert_weight
                    
                    # Apply calibration
                    if self.score_calibration:
                        tfidf_score = self._calibrate_score(tfidf_score, 'tfidf')
                        sbert_score = self._calibrate_score(sbert_score, 'sbert')
                        self._calculation_stats['calibration_applications'] += 1
                    
                    # Calculate hybrid score
                    hybrid_score = (tfidf_weight * tfidf_score) + (sbert_weight * sbert_score)
                    hybrid_score_percent = self._normalize_to_percentage(hybrid_score)
                    
                    # Update stats
                    self._update_performance_stats(tfidf_weight, sbert_weight, hybrid_score_percent)
                    
                    # Create result
                    result = {
                        'similarity_score': hybrid_score_percent,
                        'component_scores': {
                            'tfidf_score': float(tfidf_score * 100),
                            'sbert_score': float(sbert_score * 100),
                            'tfidf_weight': float(tfidf_weight),
                            'sbert_weight': float(sbert_weight)
                        },
                        'detailed_analysis': {
                            'tfidf_analysis': tfidf_result,
                            'sbert_analysis': sbert_result
                        },
                        'metadata': {
                            'resume_index': i,
                            'calculation_type': 'hybrid_batch',
                            'dynamic_weighting_used': self.enable_dynamic_weighting and not custom_weights,
                            'score_calibration_applied': self.score_calibration,
                            'resume_length': len(resume_text.split()),
                            'job_length': len(job_text.split())
                        }
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Hybrid calculation failed for resume {i}: {e}")
                    # Create error result
                    error_result = {
                        'similarity_score': 0.0,
                        'component_scores': {
                            'tfidf_score': 0.0,
                            'sbert_score': 0.0,
                            'tfidf_weight': self.default_tfidf_weight,
                            'sbert_weight': self.default_sbert_weight
                        },
                        'detailed_analysis': {},
                        'metadata': {
                            'resume_index': i,
                            'calculation_type': 'hybrid_batch_error',
                            'error': str(e)
                        }
                    }
                    results.append(error_result)
            
            logger.info(f"Batch hybrid similarity calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"Batch hybrid similarity calculation failed: {e}")
            raise ValueError(f"Failed to calculate batch hybrid similarity: {e}")
    
    def _calculate_dynamic_weights(self,
                                 resume_text: str,
                                 job_text: str,
                                 tfidf_result: Dict[str, Any],
                                 sbert_result: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate dynamic weights based on content characteristics.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            tfidf_result: TF-IDF calculation result
            sbert_result: SBERT calculation result
            
        Returns:
            Tuple of (tfidf_weight, sbert_weight)
        """
        try:
            # Analyze content characteristics
            content_analysis = self._analyze_content_characteristics(resume_text, job_text)
            
            # Start with default weights
            tfidf_weight = self.default_tfidf_weight
            sbert_weight = self.default_sbert_weight
            
            # Adjust based on text length
            avg_length = (content_analysis['resume_word_count'] + content_analysis['job_word_count']) / 2
            
            if avg_length < self._weighting_params['short_text_threshold']:
                # Short texts: favor TF-IDF for keyword matching
                tfidf_weight += 0.1
                sbert_weight -= 0.1
            elif avg_length > self._weighting_params['long_text_threshold']:
                # Long texts: favor SBERT for semantic understanding
                tfidf_weight -= 0.1
                sbert_weight += 0.1
            
            # Adjust based on technical vs soft skill content
            technical_ratio = content_analysis['technical_keyword_ratio']
            soft_skill_ratio = content_analysis['soft_skill_keyword_ratio']
            
            if technical_ratio > soft_skill_ratio + 0.1:
                # Technical content: favor TF-IDF
                adjustment = min(0.15, (technical_ratio - soft_skill_ratio) * 0.3)
                tfidf_weight += adjustment
                sbert_weight -= adjustment
            elif soft_skill_ratio > technical_ratio + 0.1:
                # Soft skill content: favor SBERT
                adjustment = min(0.15, (soft_skill_ratio - technical_ratio) * 0.3)
                tfidf_weight -= adjustment
                sbert_weight += adjustment
            
            # Adjust based on score confidence
            tfidf_confidence = tfidf_result.get('metadata', {}).get('confidence', 0.5)
            sbert_confidence = sbert_result.get('confidence_score', 0.5)
            
            if abs(tfidf_confidence - sbert_confidence) > 0.2:
                # Significant confidence difference: favor more confident method
                if tfidf_confidence > sbert_confidence:
                    adjustment = min(0.1, (tfidf_confidence - sbert_confidence) * 0.2)
                    tfidf_weight += adjustment
                    sbert_weight -= adjustment
                else:
                    adjustment = min(0.1, (sbert_confidence - tfidf_confidence) * 0.2)
                    tfidf_weight -= adjustment
                    sbert_weight += adjustment
            
            # Ensure weights are valid and sum to 1
            tfidf_weight = max(0.1, min(0.9, tfidf_weight))  # Keep within reasonable bounds
            sbert_weight = 1.0 - tfidf_weight
            
            logger.debug(f"Dynamic weights calculated: TF-IDF={tfidf_weight:.3f}, SBERT={sbert_weight:.3f}")
            return tfidf_weight, sbert_weight
            
        except Exception as e:
            logger.warning(f"Dynamic weight calculation failed: {e}")
            return self.default_tfidf_weight, self.default_sbert_weight
    
    def _analyze_content_characteristics(self, resume_text: str, job_text: str) -> Dict[str, Any]:
        """Analyze content characteristics for dynamic weighting.
        
        Args:
            resume_text: Resume text content
            job_text: Job description text content
            
        Returns:
            Dictionary with content analysis results
        """
        try:
            # Basic text statistics
            resume_words = resume_text.lower().split()
            job_words = job_text.lower().split()
            all_words = resume_words + job_words
            
            # Count technical keywords
            technical_count = sum(1 for word in all_words 
                                if any(tech in word for tech in self._weighting_params['technical_keywords']))
            
            # Count soft skill keywords
            soft_skill_count = sum(1 for word in all_words 
                                 if any(soft in word for soft in self._weighting_params['soft_skill_keywords']))
            
            total_words = len(all_words)
            
            return {
                'resume_word_count': len(resume_words),
                'job_word_count': len(job_words),
                'total_word_count': total_words,
                'technical_keyword_count': technical_count,
                'soft_skill_keyword_count': soft_skill_count,
                'technical_keyword_ratio': technical_count / total_words if total_words > 0 else 0.0,
                'soft_skill_keyword_ratio': soft_skill_count / total_words if total_words > 0 else 0.0,
                'avg_word_length': np.mean([len(word) for word in all_words]) if all_words else 0.0,
                'unique_word_ratio': len(set(all_words)) / total_words if total_words > 0 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return {
                'resume_word_count': 0,
                'job_word_count': 0,
                'total_word_count': 0,
                'technical_keyword_count': 0,
                'soft_skill_keyword_count': 0,
                'technical_keyword_ratio': 0.0,
                'soft_skill_keyword_ratio': 0.0,
                'avg_word_length': 0.0,
                'unique_word_ratio': 0.0
            }
    
    def _calibrate_score(self, score: float, method: str) -> float:
        """Apply score calibration to improve accuracy.
        
        Args:
            score: Raw similarity score (0-1)
            method: Method type ('tfidf' or 'sbert')
            
        Returns:
            Calibrated score (0-1)
        """
        try:
            if method == 'tfidf':
                # Apply TF-IDF specific calibration
                calibrated = score * self._calibration_params['tfidf_scale']
            elif method == 'sbert':
                # Apply SBERT specific calibration
                calibrated = score * self._calibration_params['sbert_scale']
            else:
                return score
            
            # Apply sigmoid normalization for better distribution
            steepness = self._calibration_params['sigmoid_steepness']
            calibrated = 1.0 / (1.0 + np.exp(-steepness * (calibrated - 0.5)))
            
            # Ensure bounds
            calibrated = np.clip(calibrated, 
                               self._calibration_params['min_score'], 
                               self._calibration_params['max_score'])
            
            return float(calibrated)
            
        except Exception as e:
            logger.warning(f"Score calibration failed for {method}: {e}")
            return score
    
    def _normalize_to_percentage(self, score: float) -> float:
        """Normalize similarity score to 0-100% range.
        
        Args:
            score: Raw similarity score (0-1)
            
        Returns:
            Normalized score (0-100)
        """
        try:
            # Ensure score is in valid range
            normalized = np.clip(score, 0.0, 1.0)
            
            # Convert to percentage
            percentage = normalized * 100.0
            
            # Round to 1 decimal place
            return round(float(percentage), 1)
            
        except Exception as e:
            logger.warning(f"Score normalization failed: {e}")
            return 0.0
    
    def _update_performance_stats(self, tfidf_weight: float, sbert_weight: float, score: float):
        """Update performance statistics.
        
        Args:
            tfidf_weight: TF-IDF weight used
            sbert_weight: SBERT weight used
            score: Final similarity score (0-100)
        """
        try:
            self._calculation_stats['total_calculations'] += 1
            
            # Update average weights (running average)
            n = self._calculation_stats['total_calculations']
            self._calculation_stats['average_tfidf_weight'] = (
                (self._calculation_stats['average_tfidf_weight'] * (n - 1) + tfidf_weight) / n
            )
            self._calculation_stats['average_sbert_weight'] = (
                (self._calculation_stats['average_sbert_weight'] * (n - 1) + sbert_weight) / n
            )
            
            # Update score distribution
            if score < 20:
                self._calculation_stats['score_distribution']['0-20'] += 1
            elif score < 40:
                self._calculation_stats['score_distribution']['20-40'] += 1
            elif score < 60:
                self._calculation_stats['score_distribution']['40-60'] += 1
            elif score < 80:
                self._calculation_stats['score_distribution']['60-80'] += 1
            else:
                self._calculation_stats['score_distribution']['80-100'] += 1
                
        except Exception as e:
            logger.warning(f"Performance stats update failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._calculation_stats.copy()
        
        # Add component calculator stats
        stats['tfidf_calculator'] = self.tfidf_calculator.get_performance_stats()
        stats['sbert_calculator'] = self.sbert_calculator.get_performance_stats()
        
        # Add configuration info
        stats['configuration'] = {
            'default_tfidf_weight': self.default_tfidf_weight,
            'default_sbert_weight': self.default_sbert_weight,
            'enable_dynamic_weighting': self.enable_dynamic_weighting,
            'score_calibration': self.score_calibration,
            'calibration_params': self._calibration_params,
            'weighting_params': {
                'short_text_threshold': self._weighting_params['short_text_threshold'],
                'long_text_threshold': self._weighting_params['long_text_threshold'],
                'technical_keywords_count': len(self._weighting_params['technical_keywords']),
                'soft_skill_keywords_count': len(self._weighting_params['soft_skill_keywords'])
            }
        }
        
        return stats
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.tfidf_calculator.clear_cache()
        self.sbert_calculator.clear_cache()
        
        # Reset performance stats
        self._calculation_stats = {
            'total_calculations': 0,
            'dynamic_weight_adjustments': 0,
            'calibration_applications': 0,
            'average_tfidf_weight': 0.0,
            'average_sbert_weight': 0.0,
            'score_distribution': {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0}
        }
        
        logger.info("Hybrid calculator cache cleared")
    
    def update_calibration_params(self, new_params: Dict[str, float]):
        """Update calibration parameters for score adjustment.
        
        Args:
            new_params: Dictionary with new calibration parameters
        """
        try:
            for key, value in new_params.items():
                if key in self._calibration_params:
                    self._calibration_params[key] = float(value)
                    logger.info(f"Updated calibration parameter {key} to {value}")
                else:
                    logger.warning(f"Unknown calibration parameter: {key}")
        except Exception as e:
            logger.error(f"Failed to update calibration parameters: {e}")
    
    def update_weighting_params(self, new_params: Dict[str, Any]):
        """Update dynamic weighting parameters.
        
        Args:
            new_params: Dictionary with new weighting parameters
        """
        try:
            for key, value in new_params.items():
                if key in self._weighting_params:
                    self._weighting_params[key] = value
                    logger.info(f"Updated weighting parameter {key}")
                else:
                    logger.warning(f"Unknown weighting parameter: {key}")
        except Exception as e:
            logger.error(f"Failed to update weighting parameters: {e}")