"""
Vector Store Manager for vLLM Gradio WebUI

Handles vector database operations for RAG functionality.
Supports both ChromaDB and FAISS backends.
"""

import os
import logging
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document structure for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
    """Search result structure"""
    document: Document
    score: float
    rank: int

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, 
                    threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass
    
    @abstractmethod
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        pass

class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, persist_directory: str, collection_name: str = "documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except ImportError:
            logger.error("ChromaDB not installed. Please install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to ChromaDB"""
        try:
            if not documents:
                return True
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding.tolist() if doc.embedding is not None else None for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            documents_text = [doc.content for doc in documents]
            
            # Filter out documents without embeddings
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            
            if not valid_indices:
                logger.warning("No documents with embeddings to add")
                return False
            
            filtered_ids = [ids[i] for i in valid_indices]
            filtered_embeddings = [embeddings[i] for i in valid_indices]
            filtered_metadatas = [metadatas[i] for i in valid_indices]
            filtered_documents = [documents_text[i] for i in valid_indices]
            
            # Add to collection
            self.collection.add(
                ids=filtered_ids,
                embeddings=filtered_embeddings,
                metadatas=filtered_metadatas,
                documents=filtered_documents
            )
            
            logger.info(f"Added {len(filtered_ids)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, 
                    threshold: float = 0.0) -> List[SearchResult]:
        """Search ChromaDB for similar documents"""
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # ChromaDB returns distances, convert to similarity scores
                    distance = results['distances'][0][i]
                    score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    if score >= threshold:
                        document = Document(
                            id=doc_id,
                            content=results['documents'][0][i],
                            metadata=results['metadatas'][0][i] or {}
                        )
                        
                        search_results.append(SearchResult(
                            document=document,
                            score=score,
                            rank=i + 1
                        ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document from ChromaDB"""
        try:
            results = self.collection.get(
                ids=[document_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and results['ids'][0]:
                return Document(
                    id=document_id,
                    content=results['documents'][0],
                    metadata=results['metadatas'][0] or {}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document from ChromaDB: {e}")
            return None
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'backend': 'chromadb'
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {'error': str(e)}
    
    async def clear_collection(self) -> bool:
        """Clear all documents from ChromaDB collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared ChromaDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            return False

class FAISSStore(VectorStore):
    """FAISS implementation of vector store"""
    
    def __init__(self, persist_directory: str, collection_name: str = "documents", 
                 dimension: int = 384):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.dimension = dimension
        
        # FAISS index and metadata storage
        self.index = None
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
        
        # File paths
        self.index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
        self.metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.pkl")
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize FAISS
        self._initialize_faiss()
        
        # Load existing data
        self._load_data()
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            self.faiss = faiss
            
            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            
        except ImportError:
            logger.error("FAISS not installed. Please install with: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    def _load_data(self):
        """Load existing FAISS index and metadata"""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = self.faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.id_to_index = data.get('id_to_index', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.next_index = data.get('next_index', 0)
                
                logger.info(f"Loaded {len(self.documents)} documents from metadata")
                
        except Exception as e:
            logger.warning(f"Failed to load existing FAISS data: {e}")
    
    def _save_data(self):
        """Save FAISS index and metadata"""
        try:
            # Save FAISS index
            self.faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata = {
                'documents': self.documents,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_index': self.next_index
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS data: {e}")
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to FAISS"""
        try:
            if not documents:
                return True
            
            # Filter documents with embeddings
            valid_docs = [doc for doc in documents if doc.embedding is not None]
            
            if not valid_docs:
                logger.warning("No documents with embeddings to add")
                return False
            
            # Prepare embeddings matrix
            embeddings = np.array([doc.embedding for doc in valid_docs], dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Update metadata
            for i, doc in enumerate(valid_docs):
                index_id = self.next_index + i
                self.documents[doc.id] = doc
                self.id_to_index[doc.id] = index_id
                self.index_to_id[index_id] = doc.id
            
            self.next_index += len(valid_docs)
            
            # Save data
            self._save_data()
            
            logger.info(f"Added {len(valid_docs)} documents to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            return False
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, 
                    threshold: float = 0.0) -> List[SearchResult]:
        """Search FAISS for similar documents"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.index.ntotal)
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            for i, (score, index) in enumerate(zip(scores[0], indices[0])):
                if index == -1:  # FAISS returns -1 for invalid results
                    continue
                
                if score >= threshold:
                    doc_id = self.index_to_id.get(index)
                    if doc_id and doc_id in self.documents:
                        search_results.append(SearchResult(
                            document=self.documents[doc_id],
                            score=float(score),
                            rank=i + 1
                        ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (requires rebuilding index)"""
        try:
            # Remove from metadata
            indices_to_remove = []
            for doc_id in document_ids:
                if doc_id in self.documents:
                    index_id = self.id_to_index[doc_id]
                    indices_to_remove.append(index_id)
                    
                    del self.documents[doc_id]
                    del self.id_to_index[doc_id]
                    del self.index_to_id[index_id]
            
            if indices_to_remove:
                # Rebuild index without deleted documents
                await self._rebuild_index()
                logger.info(f"Deleted {len(document_ids)} documents from FAISS")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from FAISS: {e}")
            return False
    
    async def _rebuild_index(self):
        """Rebuild FAISS index after deletions"""
        try:
            # Create new index
            new_index = self.faiss.IndexFlatIP(self.dimension)
            
            # Collect remaining embeddings
            remaining_docs = list(self.documents.values())
            if remaining_docs:
                embeddings = np.array([doc.embedding for doc in remaining_docs], dtype=np.float32)
                
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                
                # Add to new index
                new_index.add(embeddings)
                
                # Update mappings
                new_id_to_index = {}
                new_index_to_id = {}
                
                for i, doc in enumerate(remaining_docs):
                    new_id_to_index[doc.id] = i
                    new_index_to_id[i] = doc.id
                
                self.id_to_index = new_id_to_index
                self.index_to_id = new_index_to_id
                self.next_index = len(remaining_docs)
            else:
                self.id_to_index = {}
                self.index_to_id = {}
                self.next_index = 0
            
            self.index = new_index
            self._save_data()
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document from FAISS"""
        return self.documents.get(document_id)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get FAISS collection statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'backend': 'faiss'
        }
    
    async def clear_collection(self) -> bool:
        """Clear all documents from FAISS collection"""
        try:
            # Reset everything
            self.index = self.faiss.IndexFlatIP(self.dimension)
            self.documents = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
            
            # Save empty state
            self._save_data()
            
            logger.info(f"Cleared FAISS collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear FAISS collection: {e}")
            return False

class VectorStoreManager:
    """Manager for vector store operations"""
    
    def __init__(self, config):
        self.config = config
        self.store = None
        
        # Initialize vector store based on configuration
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the appropriate vector store"""
        try:
            # Handle both dictionary and object config
            if isinstance(self.config, dict):
                vector_db_type = self.config.get('type', 'chromadb')
                persist_directory = self.config.get('persist_directory', './data/vector_db')
                collection_name = self.config.get('collection_name', 'documents')
            else:
                vector_db_type = getattr(self.config, 'vector_db_type', 'chromadb')
                persist_directory = getattr(self.config, 'persist_directory', './data/vector_db')
                collection_name = getattr(self.config, 'collection_name', 'documents')
            
            if vector_db_type.lower() == 'chromadb':
                self.store = ChromaDBStore(
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
            elif vector_db_type.lower() == 'faiss':
                self.store = FAISSStore(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    dimension=384  # Default for sentence-transformers
                )
            else:
                raise ValueError(f"Unsupported vector database type: {vector_db_type}")
            
            logger.info(f"Initialized {vector_db_type} vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        return await self.store.add_documents(documents)
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, 
                    threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar documents"""
        return await self.store.search(query_embedding, top_k, threshold)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        return await self.store.delete_documents(document_ids)
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a specific document by ID"""
        return await self.store.get_document(document_id)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return await self.store.get_collection_stats()
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        return await self.store.clear_collection()

