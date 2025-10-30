"""
RAG Pipeline for vLLM Gradio WebUI

Main pipeline that coordinates document ingestion, embedding generation,
vector storage, and retrieval for RAG functionality.
"""

import logging
import asyncio
import time
import hashlib
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .vector_store import VectorStoreManager, Document
from .embeddings import EmbeddingManager
from .retriever import DocumentRetriever, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Document chunk for processing"""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    source_id: str

@dataclass
class IngestionResult:
    """Result of document ingestion"""
    success: bool
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    processing_time: float
    error_message: Optional[str] = None

class RAGPipeline:
    """Main RAG pipeline coordinating all components"""
    
    def __init__(self, config):
        self.config = config
        
        # Create vector store config from RAG config
        vector_store_config = {
            'type': getattr(config, 'vector_db_type', 'chromadb'),
            'persist_directory': getattr(config, 'persist_directory', './data/vector_db'),
            'collection_name': getattr(config, 'collection_name', 'documents')
        }
        
        # Create embeddings config
        embeddings_config = {
            'model': getattr(config, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            'device': getattr(config, 'embedding_device', 'auto')
        }
        
        # Create retrieval config
        retrieval_config = {
            'top_k': getattr(config, 'top_k', 5),
            'similarity_threshold': getattr(config, 'similarity_threshold', 0.7),
            'max_context_length': getattr(config, 'max_context_length', 4000)
        }
        
        # Initialize components
        self.vector_store = VectorStoreManager(vector_store_config)
        self.embedding_manager = EmbeddingManager(embeddings_config)
        self.retriever = DocumentRetriever(
            self.vector_store, 
            self.embedding_manager, 
            retrieval_config
        )
        
        # Document processing settings
        self.chunk_size = getattr(config, 'chunk_size', 1000)
        self.chunk_overlap = getattr(config, 'chunk_overlap', 200)
        self.max_chunk_length = getattr(config, 'max_context_length', 4000)
        
        logger.info("RAG Pipeline initialized")
    
    async def ingest_documents(self, documents: List[Dict[str, Any]]) -> IngestionResult:
        """Ingest documents into the RAG system"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting ingestion of {len(documents)} documents")
            
            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            embedding_results = await self.embedding_manager.generate_embeddings_batch(
                chunk_texts, 
                use_cache=True
            )
            
            # Create Document objects with embeddings
            vector_documents = []
            for chunk, embedding_result in zip(all_chunks, embedding_results):
                doc_id = self._generate_chunk_id(chunk)
                
                vector_doc = Document(
                    id=doc_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=embedding_result.embedding
                )
                vector_documents.append(vector_doc)
            
            # Add to vector store
            success = await self.vector_store.add_documents(vector_documents)
            
            processing_time = time.time() - start_time
            
            if success:
                logger.info(f"Successfully ingested {len(documents)} documents in {processing_time:.2f}s")
                return IngestionResult(
                    success=True,
                    documents_processed=len(documents),
                    chunks_created=len(all_chunks),
                    embeddings_generated=len(embedding_results),
                    processing_time=processing_time
                )
            else:
                return IngestionResult(
                    success=False,
                    documents_processed=0,
                    chunks_created=len(all_chunks),
                    embeddings_generated=len(embedding_results),
                    processing_time=processing_time,
                    error_message="Failed to add documents to vector store"
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            
            return IngestionResult(
                success=False,
                documents_processed=0,
                chunks_created=0,
                embeddings_generated=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document into chunks"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        source_id = document.get('id', self._generate_document_id(document))
        
        # Add source information to metadata
        metadata.update({
            'source_id': source_id,
            'source_type': document.get('type', 'unknown'),
            'source_path': document.get('path', ''),
            'ingestion_timestamp': time.time()
        })
        
        chunks = []
        
        # Simple text chunking by sentences and paragraphs
        if len(content) <= self.chunk_size:
            # Document is small enough to be a single chunk
            chunks.append(DocumentChunk(
                content=content,
                metadata=metadata.copy(),
                chunk_index=0,
                source_id=source_id
            ))
        else:
            # Split into chunks
            chunks = self._split_text_into_chunks(content, metadata, source_id)
        
        return chunks
    
    def _split_text_into_chunks(self, text: str, metadata: Dict[str, Any], 
                               source_id: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'chunk_length': len(current_chunk)
                    })
                    
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                        source_id=source_id
                    ))
                    
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Paragraph is too long, split by sentences
                    sentence_chunks = self._split_long_paragraph(
                        paragraph, metadata, source_id, chunk_index
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = ""
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if any content remains
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'chunk_length': len(current_chunk)
            })
            
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                source_id=source_id
            ))
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, metadata: Dict[str, Any], 
                             source_id: str, start_index: int) -> List[DocumentChunk]:
        """Split a long paragraph by sentences"""
        chunks = []
        
        # Simple sentence splitting
        sentences = paragraph.split('. ')
        
        current_chunk = ""
        chunk_index = start_index
        
        for i, sentence in enumerate(sentences):
            # Add period back except for last sentence
            if i < len(sentences) - 1:
                sentence += '. '
            
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'chunk_length': len(current_chunk),
                        'is_sentence_split': True
                    })
                    
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                        source_id=source_id
                    ))
                    
                    chunk_index += 1
                    current_chunk = sentence
                else:
                    # Single sentence is too long, truncate
                    if len(sentence) > self.max_chunk_length:
                        sentence = sentence[:self.max_chunk_length] + "..."
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'chunk_length': len(sentence),
                        'is_truncated': True
                    })
                    
                    chunks.append(DocumentChunk(
                        content=sentence,
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                        source_id=source_id
                    ))
                    
                    chunk_index += 1
                    current_chunk = ""
            else:
                current_chunk += sentence
        
        # Add final chunk
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'chunk_length': len(current_chunk)
            })
            
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                source_id=source_id
            ))
        
        return chunks
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """Generate unique ID for document"""
        content = document.get('content', '')
        path = document.get('path', '')
        
        # Create hash from content and path
        hash_input = f"{path}:{content[:1000]}"  # Use first 1000 chars
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate unique ID for chunk"""
        hash_input = f"{chunk.source_id}:{chunk.chunk_index}:{chunk.content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def retrieve(self, query: str, top_k: int = 5, 
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        try:
            retrieval_query = RetrievalQuery(
                text=query,
                top_k=top_k,
                threshold=threshold,
                rerank=True
            )
            
            results = await self.retriever.retrieve(retrieval_query)
            
            # Convert to dict format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.content,
                    'source': result.source,
                    'score': result.score,
                    'rank': result.rank,
                    'metadata': result.metadata,
                    'chunk_id': result.chunk_id
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def ingest_text_documents(self, texts: List[str], 
                                  metadata_list: Optional[List[Dict[str, Any]]] = None) -> IngestionResult:
        """Ingest plain text documents"""
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            doc = {
                'id': f"text_doc_{i}_{int(time.time())}",
                'content': text,
                'type': 'text',
                'metadata': metadata
            }
            documents.append(doc)
        
        return await self.ingest_documents(documents)
    
    async def ingest_file_documents(self, file_paths: List[str]) -> IngestionResult:
        """Ingest documents from files"""
        documents = []
        
        for file_path in file_paths:
            try:
                # Read file content (simplified - would need proper file handling)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = {
                    'id': f"file_{Path(file_path).stem}_{int(time.time())}",
                    'content': content,
                    'type': 'file',
                    'path': file_path,
                    'metadata': {
                        'filename': Path(file_path).name,
                        'file_size': len(content),
                        'file_extension': Path(file_path).suffix
                    }
                }
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
        
        return await self.ingest_documents(documents)
    
    async def delete_documents(self, source_ids: List[str]) -> bool:
        """Delete documents by source IDs"""
        try:
            # Get all chunk IDs for the source documents
            # This is a simplified approach - in practice you'd want to track this mapping
            stats = await self.vector_store.get_collection_stats()
            
            # For now, just log the request
            logger.info(f"Delete request for {len(source_ids)} documents")
            
            # TODO: Implement proper document deletion
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def clear_all_documents(self) -> bool:
        """Clear all documents from the RAG system"""
        try:
            success = await self.vector_store.clear_collection()
            if success:
                logger.info("Cleared all documents from RAG system")
            return success
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            embedding_stats = self.embedding_manager.get_cache_stats()
            model_info = self.embedding_manager.get_model_info()
            
            return {
                'vector_store': vector_stats,
                'embeddings': embedding_stats,
                'model': model_info,
                'config': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'max_chunk_length': self.max_chunk_length
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG system"""
        health = {
            'status': 'healthy',
            'components': {}
        }
        
        try:
            # Check vector store
            vector_stats = await self.vector_store.get_collection_stats()
            health['components']['vector_store'] = {
                'status': 'healthy',
                'documents': vector_stats.get('total_documents', 0)
            }
        except Exception as e:
            health['components']['vector_store'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        try:
            # Check embedding manager
            model_info = self.embedding_manager.get_model_info()
            health['components']['embeddings'] = {
                'status': 'healthy',
                'model': model_info['model_name']
            }
        except Exception as e:
            health['components']['embeddings'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Test retrieval
        try:
            test_results = await self.retrieve("test query", top_k=1)
            health['components']['retrieval'] = {
                'status': 'healthy',
                'test_results': len(test_results)
            }
        except Exception as e:
            health['components']['retrieval'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        return health
    
    async def cleanup(self):
        """Cleanup RAG pipeline resources"""
        try:
            self.embedding_manager.cleanup()
            logger.info("RAG pipeline cleaned up")
        except Exception as e:
            logger.error(f"Error during RAG pipeline cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

