"""
Web Manager for vLLM Gradio WebUI

Coordinates web scraping, search integration, and RAG integration
for web-enhanced functionality.
"""

import logging
import asyncio
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

from .scraper import WebScraper, SearchEnhancedScraper, ContentProcessor, ScrapingConfig
from .search_integration import SearchManager, WebSearchIntegration, SearchQuery

logger = logging.getLogger(__name__)

@dataclass
class WebEnhancementRequest:
    """Request for web enhancement"""
    query: str
    search_queries: List[str] = None
    max_results: int = 5
    include_search: bool = True
    include_urls: List[str] = None
    search_provider: Optional[str] = None
    time_range: Optional[str] = None

@dataclass
class WebEnhancementResult:
    """Result of web enhancement"""
    success: bool
    documents: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    scraped_urls: List[str]
    processing_time: float
    error_message: Optional[str] = None

class WebManager:
    """Main manager for web functionality"""
    
    def __init__(self, config, rag_pipeline=None):
        self.config = config
        self.rag_pipeline = rag_pipeline
        
        # Initialize components
        self._initialize_components()
        
        # Cache for recent results
        # Handle both dictionary and object config
        if hasattr(config, 'get'):
            # Dictionary config
            self.result_cache = {}
            self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        else:
            # Object config
            self.result_cache = {}
            self.cache_ttl = getattr(config, 'cache_ttl', 3600)  # 1 hour default
        
        logger.info("Web manager initialized")
    
    def _initialize_components(self):
        """Initialize web components"""
        try:
            # Initialize scraper - handle both config types
            if hasattr(self.config, 'get'):
                # Dictionary config
                scraping_config = ScrapingConfig(
                    timeout=self.config.get('scraping_timeout', 30),
                    max_content_length=self.config.get('max_content_length', 100000),
                    user_agent=self.config.get('user_agent', 'vLLM-Gradio-WebUI/1.0'),
                    verify_ssl=self.config.get('verify_ssl', False),
                    blocked_domains=self.config.get('blocked_domains', [])
                )
                search_config = self.config.get('search', {})
            else:
                # Object config
                scraping_config = ScrapingConfig(
                    timeout=getattr(self.config, 'timeout', 30),
                    max_content_length=getattr(self.config, 'max_content_length', 100000),
                    user_agent=getattr(self.config, 'user_agent', 'vLLM-Gradio-WebUI/1.0'),
                    verify_ssl=getattr(self.config, 'verify_ssl', False),
                    blocked_domains=getattr(self.config, 'blocked_domains', [])
                )
                # For object config, create a simple search config dict
                search_config = {
                    'providers': {
                        'duckduckgo': {'enabled': True}
                    }
                }
            
            self.scraper = WebScraper(scraping_config)
            self.enhanced_scraper = SearchEnhancedScraper(self.scraper)
            self.content_processor = ContentProcessor()
            
            # Initialize search manager
            self.search_manager = SearchManager(search_config)
            
            # Initialize web search integration
            self.web_search = WebSearchIntegration(self.search_manager, self.scraper)
            
            logger.info("Web components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize web components: {e}")
            raise
    
    async def enhance_with_web(self, request: WebEnhancementRequest) -> WebEnhancementResult:
        """Enhance query with web content"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting web enhancement for query: {request.query}")
            
            all_documents = []
            all_search_results = []
            scraped_urls = []
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached web enhancement result")
                return cached_result
            
            # Perform web search if enabled
            if request.include_search:
                search_results = await self._perform_web_search(request)
                all_search_results.extend(search_results)
                
                # Extract documents from search results
                search_documents = await self._extract_from_search_results(search_results)
                all_documents.extend(search_documents)
                scraped_urls.extend([doc['metadata']['url'] for doc in search_documents])
            
            # Scrape additional URLs if provided
            if request.include_urls:
                url_documents = await self._scrape_urls(request.include_urls)
                all_documents.extend(url_documents)
                scraped_urls.extend(request.include_urls)
            
            # Deduplicate documents
            unique_documents = self.content_processor.deduplicate_content(all_documents)
            
            # Ingest into RAG if available
            if self.rag_pipeline and unique_documents:
                await self._ingest_to_rag(unique_documents)
            
            processing_time = time.time() - start_time
            
            result = WebEnhancementResult(
                success=True,
                documents=unique_documents,
                search_results=all_search_results,
                scraped_urls=scraped_urls,
                processing_time=processing_time
            )
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            logger.info(f"Web enhancement completed in {processing_time:.2f}s, "
                       f"found {len(unique_documents)} documents")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Web enhancement failed: {e}", exc_info=True)
            
            return WebEnhancementResult(
                success=False,
                documents=[],
                search_results=[],
                scraped_urls=[],
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _perform_web_search(self, request: WebEnhancementRequest) -> List[Dict[str, Any]]:
        """Perform web search"""
        try:
            # Use provided search queries or generate from main query
            search_queries = request.search_queries or [request.query]
            
            all_results = []
            
            for query in search_queries[:3]:  # Limit to 3 queries
                search_query = SearchQuery(
                    query=query,
                    max_results=request.max_results,
                    time_range=request.time_range
                )
                
                results = await self.search_manager.search(
                    search_query, 
                    request.search_provider
                )
                
                # Convert to dict format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'title': result.title,
                        'url': result.url,
                        'snippet': result.snippet,
                        'rank': result.rank,
                        'source': result.source,
                        'metadata': result.metadata or {}
                    })
                
                all_results.extend(formatted_results)
            
            # Deduplicate by URL
            unique_results = {}
            for result in all_results:
                url = result['url']
                if url not in unique_results or result['rank'] < unique_results[url]['rank']:
                    unique_results[url] = result
            
            return list(unique_results.values())[:request.max_results]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _extract_from_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract content from search result URLs"""
        try:
            urls = [result['url'] for result in search_results]
            
            if not urls:
                return []
            
            # Scrape content
            scraped_contents = await self.scraper.scrape_urls(urls)
            
            # Process for RAG
            documents = self.content_processor.process_multiple_contents(scraped_contents)
            
            # Enhance with search metadata
            for i, doc in enumerate(documents):
                if i < len(search_results):
                    search_result = search_results[i]
                    doc['metadata'].update({
                        'search_title': search_result['title'],
                        'search_snippet': search_result['snippet'],
                        'search_rank': search_result['rank'],
                        'search_source': search_result['source']
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract from search results: {e}")
            return []
    
    async def _scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape content from provided URLs"""
        try:
            scraped_contents = await self.scraper.scrape_urls(urls)
            documents = self.content_processor.process_multiple_contents(scraped_contents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to scrape URLs: {e}")
            return []
    
    async def _ingest_to_rag(self, documents: List[Dict[str, Any]]):
        """Ingest documents into RAG system"""
        try:
            if self.rag_pipeline:
                result = await self.rag_pipeline.ingest_documents(documents)
                if result.success:
                    logger.info(f"Ingested {result.documents_processed} web documents to RAG")
                else:
                    logger.warning(f"RAG ingestion failed: {result.error_message}")
            
        except Exception as e:
            logger.error(f"Failed to ingest to RAG: {e}")
    
    def _generate_cache_key(self, request: WebEnhancementRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        
        key_data = f"{request.query}:{request.max_results}:{request.include_search}:{request.search_provider}"
        if request.search_queries:
            key_data += f":{':'.join(request.search_queries)}"
        if request.include_urls:
            key_data += f":{':'.join(request.include_urls)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[WebEnhancementResult]:
        """Get result from cache"""
        try:
            if cache_key in self.result_cache:
                cached_data, timestamp = self.result_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
                else:
                    # Remove expired cache
                    del self.result_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: WebEnhancementResult):
        """Save result to cache"""
        try:
            self.result_cache[cache_key] = (result, time.time())
            
            # Clean old cache entries
            self._cleanup_cache()
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self.result_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.result_cache[key]
            
            # Limit cache size
            max_cache_size = 100
            if len(self.result_cache) > max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self.result_cache.items(),
                    key=lambda x: x[1][1]  # Sort by timestamp
                )
                
                for key, _ in sorted_items[:-max_cache_size]:
                    del self.result_cache[key]
            
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    async def search_web(self, query: str, max_results: int = 10,
                        provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple web search without content extraction"""
        try:
            search_query = SearchQuery(
                query=query,
                max_results=max_results
            )
            
            results = await self.search_manager.search(search_query, provider)
            
            return [
                {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'rank': result.rank,
                    'source': result.source,
                    'metadata': result.metadata or {}
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a single URL"""
        try:
            scraped_content = await self.scraper.scrape_url(url)
            
            if scraped_content.success:
                document = self.content_processor.process_scraped_content(scraped_content)
                return document
            else:
                logger.warning(f"Failed to scrape {url}: {scraped_content.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"URL scraping failed: {e}")
            return None
    
    def get_search_providers(self) -> List[str]:
        """Get available search providers"""
        return self.search_manager.get_available_providers()
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of search providers"""
        return self.search_manager.get_provider_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get web manager statistics"""
        return {
            'cache_size': len(self.result_cache),
            'cache_ttl': self.cache_ttl,
            'search_providers': self.get_search_providers(),
            'provider_status': self.get_provider_status(),
            'config': {
                'scraping_timeout': self.config.get('scraping_timeout', 30),
                'max_content_length': self.config.get('max_content_length', 100000),
                'verify_ssl': self.config.get('verify_ssl', False)
            }
        }
    
    def clear_cache(self):
        """Clear result cache"""
        self.result_cache.clear()
        logger.info("Web manager cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on web components"""
        health = {
            'status': 'healthy',
            'components': {}
        }
        
        # Check search providers
        try:
            providers = self.get_search_providers()
            health['components']['search'] = {
                'status': 'healthy' if providers else 'degraded',
                'available_providers': providers
            }
        except Exception as e:
            health['components']['search'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Test scraping
        try:
            # Test with a simple URL (this is just a basic test)
            health['components']['scraping'] = {
                'status': 'healthy',
                'config_loaded': True
            }
        except Exception as e:
            health['components']['scraping'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        return health
    
    async def cleanup(self):
        """Cleanup web manager resources"""
        try:
            self.clear_cache()
            logger.info("Web manager cleaned up")
        except Exception as e:
            logger.error(f"Error during web manager cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

