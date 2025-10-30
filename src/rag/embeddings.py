"""
Embeddings Manager for vLLM Gradio WebUI

Handles document embedding generation using sentence transformers
and other embedding models for RAG functionality.
"""

import logging
import numpy as np
import asyncio
import hashlib
import pickle
import os
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: np.ndarray
    model_name: str
    dimension: int
    processing_time: float

class EmbeddingManager:
    """Manages embedding generation and caching"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
        # Handle both dictionary and object config
        if isinstance(config, dict):
            self.model_name = config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.device = config.get('device', 'auto')
            persist_directory = config.get('persist_directory', './data/vector_db')
        else:
            self.model_name = getattr(config, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            self.device = getattr(config, 'embedding_device', 'auto')
            persist_directory = getattr(config, 'persist_directory', './data/vector_db')
        
        self.cache_dir = os.path.join(persist_directory, "embedding_cache")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            # Import sentence transformers
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Determine device
            device = self._get_device()
            
            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                device=device
            )
            
            # Get model dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Loaded embedding model {self.model_name} on {device} (dim: {self.dimension})")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the appropriate device for embeddings"""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        else:
            return self.device
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Create hash of text and model name
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Generate embedding synchronously (for thread pool)"""
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Generate embedding
            embedding = self.model.encode(
                cleaned_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (most models have token limits)
        max_length = 512  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length]
            logger.debug(f"Truncated text to {max_length} characters")
        
        return text
    
    async def generate_embedding(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """Generate embedding for a single text"""
        import time
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text)
            
            if use_cache:
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    processing_time = time.time() - start_time
                    logger.debug(f"Loaded embedding from cache in {processing_time:.3f}s")
                    
                    return EmbeddingResult(
                        text=text,
                        embedding=cached_embedding,
                        model_name=self.model_name,
                        dimension=self.dimension,
                        processing_time=processing_time
                    )
            
            # Generate embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._generate_embedding_sync,
                text
            )
            
            # Cache the result
            if use_cache:
                self._save_to_cache(cache_key, embedding)
            
            processing_time = time.time() - start_time
            logger.debug(f"Generated embedding in {processing_time:.3f}s")
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self.model_name,
                dimension=self.dimension,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], 
                                      use_cache: bool = True,
                                      batch_size: int = 32) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts in batches"""
        import time
        start_time = time.time()
        
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = await self._process_batch(batch_texts, use_cache)
                results.extend(batch_results)
                
                # Log progress
                if len(texts) > batch_size:
                    logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
            
            total_time = time.time() - start_time
            logger.info(f"Generated {len(results)} embeddings in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    async def _process_batch(self, texts: List[str], use_cache: bool) -> List[EmbeddingResult]:
        """Process a batch of texts"""
        # Separate cached and uncached texts
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                
                if cached_embedding is not None:
                    cached_results.append((i, EmbeddingResult(
                        text=text,
                        embedding=cached_embedding,
                        model_name=self.model_name,
                        dimension=self.dimension,
                        processing_time=0.0
                    )))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        uncached_results = []
        if uncached_texts:
            uncached_results = await self._generate_batch_embeddings(uncached_texts, use_cache)
        
        # Combine results in original order
        all_results = [None] * len(texts)
        
        # Place cached results
        for i, result in cached_results:
            all_results[i] = result
        
        # Place uncached results
        for i, result in zip(uncached_indices, uncached_results):
            all_results[i] = result
        
        return all_results
    
    async def _generate_batch_embeddings(self, texts: List[str], use_cache: bool) -> List[EmbeddingResult]:
        """Generate embeddings for uncached texts"""
        import time
        start_time = time.time()
        
        try:
            # Clean texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self._generate_batch_embeddings_sync,
                cleaned_texts
            )
            
            processing_time = time.time() - start_time
            
            # Create results and cache
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Cache the result
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    self._save_to_cache(cache_key, embedding)
                
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_name=self.model_name,
                    dimension=self.dimension,
                    processing_time=processing_time / len(texts)  # Average time per embedding
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def _generate_batch_embeddings_sync(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch synchronously"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings sync: {e}")
            raise
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        result = await self.generate_embedding(query, use_cache=True)
        return result.embedding
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'device': self.device,
            'cache_dir': self.cache_dir
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in cache_files
            )
            
            return {
                'cached_embeddings': len(cache_files),
                'cache_size_mb': total_size / (1024 * 1024),
                'cache_dir': self.cache_dir
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """Clear embedding cache"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            
            for cache_file in cache_files:
                os.remove(os.path.join(self.cache_dir, cache_file))
            
            logger.info(f"Cleared {len(cache_files)} cached embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("Embedding manager cleaned up")
        except Exception as e:
            logger.error(f"Error during embedding manager cleanup: {e}")

class MultiModelEmbeddingManager:
    """Manager for multiple embedding models"""
    
    def __init__(self, configs: List[Dict[str, Any]]):
        self.managers: Dict[str, EmbeddingManager] = {}
        self.default_model = None
        
        # Initialize multiple models
        for config in configs:
            model_name = config['model_name']
            manager = EmbeddingManager(config)
            self.managers[model_name] = manager
            
            if config.get('default', False) or self.default_model is None:
                self.default_model = model_name
        
        logger.info(f"Initialized {len(self.managers)} embedding models")
    
    async def generate_embedding(self, text: str, model_name: Optional[str] = None) -> EmbeddingResult:
        """Generate embedding using specified or default model"""
        model_name = model_name or self.default_model
        
        if model_name not in self.managers:
            raise ValueError(f"Model {model_name} not available")
        
        return await self.managers[model_name].generate_embedding(text)
    
    async def generate_embeddings_batch(self, texts: List[str], 
                                      model_name: Optional[str] = None) -> List[EmbeddingResult]:
        """Generate batch embeddings using specified or default model"""
        model_name = model_name or self.default_model
        
        if model_name not in self.managers:
            raise ValueError(f"Model {model_name} not available")
        
        return await self.managers[model_name].generate_embeddings_batch(texts)
    
    def get_available_models(self) -> List[str]:
        """Get list of available embedding models"""
        return list(self.managers.keys())
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_name = model_name or self.default_model
        
        if model_name not in self.managers:
            raise ValueError(f"Model {model_name} not available")
        
        return self.managers[model_name].get_model_info()
    
    def cleanup(self):
        """Cleanup all managers"""
        for manager in self.managers.values():
            manager.cleanup()

