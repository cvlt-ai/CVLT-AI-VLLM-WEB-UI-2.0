"""
Retriever for vLLM Gradio WebUI

Handles document retrieval and ranking for RAG functionality.
Integrates with vector store and embeddings for semantic search.
"""

import logging
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .vector_store import VectorStoreManager, Document, SearchResult
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result of document retrieval"""
    content: str
    source: str
    score: float
    rank: int
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None

@dataclass
class RetrievalQuery:
    """Query for document retrieval"""
    text: str
    top_k: int = 5
    threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None
    rerank: bool = True

class DocumentRetriever:
    """Handles document retrieval for RAG"""
    
    def __init__(self, vector_store_manager: VectorStoreManager, 
                 embedding_manager: EmbeddingManager, config):
        self.vector_store = vector_store_manager
        self.embedding_manager = embedding_manager
        self.config = config
        
        # Handle both dictionary and object config
        if isinstance(config, dict):
            self.top_k = config.get('top_k', 5)
            self.similarity_threshold = config.get('similarity_threshold', 0.7)
            self.max_context_length = config.get('max_context_length', 4000)
        else:
            self.top_k = getattr(config, 'top_k', 5)
            self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
            self.max_context_length = getattr(config, 'max_context_length', 4000)
        
        logger.info("Document retriever initialized")
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.generate_query_embedding(query.text)
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=query.top_k * 2,  # Get more results for reranking
                threshold=query.threshold
            )
            
            # Convert to retrieval results
            retrieval_results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.document.content,
                    source=result.document.metadata.get('source', 'Unknown'),
                    score=result.score,
                    rank=result.rank,
                    metadata=result.document.metadata,
                    chunk_id=result.document.id
                )
                retrieval_results.append(retrieval_result)
            
            # Apply filters if specified
            if query.filters:
                retrieval_results = self._apply_filters(retrieval_results, query.filters)
            
            # Rerank if enabled
            if query.rerank and len(retrieval_results) > 1:
                retrieval_results = await self._rerank_results(query.text, retrieval_results)
            
            # Limit to requested number
            retrieval_results = retrieval_results[:query.top_k]
            
            # Update ranks
            for i, result in enumerate(retrieval_results):
                result.rank = i + 1
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(retrieval_results)} documents in {retrieval_time:.3f}s")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _apply_filters(self, results: List[RetrievalResult], 
                      filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply metadata filters to results"""
        filtered_results = []
        
        for result in results:
            include = True
            
            for key, value in filters.items():
                if key in result.metadata:
                    if isinstance(value, list):
                        if result.metadata[key] not in value:
                            include = False
                            break
                    else:
                        if result.metadata[key] != value:
                            include = False
                            break
                else:
                    include = False
                    break
            
            if include:
                filtered_results.append(result)
        
        logger.debug(f"Filtered {len(results)} -> {len(filtered_results)} results")
        return filtered_results
    
    async def _rerank_results(self, query: str, 
                            results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using additional scoring methods"""
        try:
            # Simple keyword-based reranking
            query_words = set(query.lower().split())
            
            for result in results:
                # Calculate keyword overlap score
                content_words = set(result.content.lower().split())
                overlap = len(query_words.intersection(content_words))
                keyword_score = overlap / len(query_words) if query_words else 0
                
                # Combine with semantic score
                combined_score = 0.7 * result.score + 0.3 * keyword_score
                result.score = combined_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug("Reranked results using keyword overlap")
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results
    
    async def retrieve_by_source(self, source: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents from a specific source"""
        try:
            # Get all documents (this is a simplified approach)
            # In practice, you'd want to implement source-based filtering in the vector store
            stats = await self.vector_store.get_collection_stats()
            
            # For now, return empty list as this requires more complex implementation
            logger.warning("Source-based retrieval not fully implemented")
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve by source: {e}")
            return []
    
    async def get_similar_documents(self, document_id: str, 
                                  top_k: int = 5) -> List[RetrievalResult]:
        """Get documents similar to a specific document"""
        try:
            # Get the reference document
            reference_doc = await self.vector_store.get_document(document_id)
            if not reference_doc:
                logger.warning(f"Document {document_id} not found")
                return []
            
            # Use the document content as query
            query = RetrievalQuery(
                text=reference_doc.content,
                top_k=top_k + 1,  # +1 to exclude the reference document
                threshold=0.5,
                rerank=False
            )
            
            results = await self.retrieve(query)
            
            # Remove the reference document from results
            filtered_results = [r for r in results if r.chunk_id != document_id]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to get similar documents: {e}")
            return []

class HybridRetriever:
    """Hybrid retriever combining multiple retrieval methods"""
    
    def __init__(self, retrievers: List[DocumentRetriever], weights: Optional[List[float]] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")
        
        logger.info(f"Initialized hybrid retriever with {len(retrievers)} retrievers")
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve using multiple retrievers and combine results"""
        try:
            # Get results from all retrievers
            all_results = []
            
            for i, retriever in enumerate(self.retrievers):
                results = await retriever.retrieve(query)
                
                # Apply weight to scores
                for result in results:
                    result.score *= self.weights[i]
                
                all_results.extend(results)
            
            # Combine and deduplicate results
            combined_results = self._combine_results(all_results)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            # Limit to requested number
            final_results = combined_results[:query.top_k]
            
            # Update ranks
            for i, result in enumerate(final_results):
                result.rank = i + 1
            
            logger.info(f"Hybrid retrieval returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    def _combine_results(self, all_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Combine and deduplicate results from multiple retrievers"""
        # Group by chunk_id or content hash
        result_groups = {}
        
        for result in all_results:
            key = result.chunk_id or hash(result.content)
            
            if key in result_groups:
                # Combine scores (take maximum)
                existing = result_groups[key]
                if result.score > existing.score:
                    result_groups[key] = result
            else:
                result_groups[key] = result
        
        return list(result_groups.values())

class ContextualRetriever:
    """Retriever that considers conversation context"""
    
    def __init__(self, base_retriever: DocumentRetriever):
        self.base_retriever = base_retriever
        self.conversation_history = []
        self.max_history = 10
        
        logger.info("Contextual retriever initialized")
    
    def add_to_context(self, query: str, response: str):
        """Add query-response pair to conversation context"""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': time.time()
        })
        
        # Maintain history limit
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve with conversation context"""
        try:
            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(query.text)
            
            # Create enhanced query object
            enhanced_query_obj = RetrievalQuery(
                text=enhanced_query,
                top_k=query.top_k,
                threshold=query.threshold,
                filters=query.filters,
                rerank=query.rerank
            )
            
            # Retrieve using enhanced query
            results = await self.base_retriever.retrieve(enhanced_query_obj)
            
            # Post-process results based on context
            contextualized_results = self._contextualize_results(results, query.text)
            
            return contextualized_results
            
        except Exception as e:
            logger.error(f"Contextual retrieval failed: {e}")
            # Fallback to base retriever
            return await self.base_retriever.retrieve(query)
    
    def _enhance_query_with_context(self, query: str) -> str:
        """Enhance query with conversation context"""
        if not self.conversation_history:
            return query
        
        # Get recent context
        recent_context = self.conversation_history[-3:]  # Last 3 exchanges
        
        # Build context string
        context_parts = []
        for exchange in recent_context:
            context_parts.append(f"Previous: {exchange['query']}")
        
        if context_parts:
            enhanced_query = f"Context: {' '.join(context_parts)}\nCurrent: {query}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _contextualize_results(self, results: List[RetrievalResult], 
                             current_query: str) -> List[RetrievalResult]:
        """Adjust result scores based on conversation context"""
        if not self.conversation_history:
            return results
        
        # Get recent topics
        recent_topics = set()
        for exchange in self.conversation_history[-3:]:
            recent_topics.update(exchange['query'].lower().split())
        
        # Boost results that relate to recent topics
        for result in results:
            content_words = set(result.content.lower().split())
            topic_overlap = len(recent_topics.intersection(content_words))
            
            if topic_overlap > 0:
                boost_factor = 1.0 + (topic_overlap * 0.1)  # 10% boost per overlapping word
                result.score *= boost_factor
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_history = []
        logger.info("Cleared conversation context")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context"""
        return {
            'history_length': len(self.conversation_history),
            'max_history': self.max_history,
            'recent_topics': list(set(
                word for exchange in self.conversation_history[-3:]
                for word in exchange['query'].lower().split()
            )) if self.conversation_history else []
        }

