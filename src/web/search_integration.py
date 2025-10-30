"""
Search Integration for vLLM Gradio WebUI

Integrates with various search APIs and services to find relevant web content
for web-enhanced RAG functionality.
"""

import logging
import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin
import os

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result structure"""
    title: str
    url: str
    snippet: str
    rank: int
    source: str
    metadata: Dict[str, Any] = None

@dataclass
class SearchQuery:
    """Search query structure"""
    query: str
    max_results: int = 10
    language: str = "en"
    region: str = "us"
    time_range: Optional[str] = None  # "day", "week", "month", "year"
    site_filter: Optional[str] = None  # Restrict to specific site

class SearchProvider:
    """Base class for search providers"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', False)
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search and return results"""
        raise NotImplementedError

class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider (no API key required)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DuckDuckGo", config)
        self.base_url = "https://api.duckduckgo.com/"
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            # Note: DuckDuckGo's API is limited and doesn't provide web search results
            # This is a simplified implementation for demonstration
            
            params = {
                'q': query.query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        
                        # Extract abstract if available
                        if data.get('Abstract'):
                            results.append(SearchResult(
                                title=data.get('Heading', query.query),
                                url=data.get('AbstractURL', ''),
                                snippet=data.get('Abstract', ''),
                                rank=1,
                                source=self.name,
                                metadata={'type': 'abstract'}
                            ))
                        
                        # Extract related topics
                        for i, topic in enumerate(data.get('RelatedTopics', [])[:query.max_results-1]):
                            if isinstance(topic, dict) and topic.get('Text'):
                                results.append(SearchResult(
                                    title=topic.get('Text', '')[:100],
                                    url=topic.get('FirstURL', ''),
                                    snippet=topic.get('Text', ''),
                                    rank=i + 2,
                                    source=self.name,
                                    metadata={'type': 'related_topic'}
                                ))
                        
                        return results
            
            return []
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

class GoogleCustomSearchProvider(SearchProvider):
    """Google Custom Search API provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Google Custom Search", config)
        self.api_key = config.get('api_key')
        self.search_engine_id = config.get('search_engine_id')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key or not self.search_engine_id:
            self.enabled = False
            logger.warning("Google Custom Search API key or search engine ID not configured")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.enabled:
            return []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query.query,
                'num': min(query.max_results, 10),  # Google allows max 10 per request
                'lr': f'lang_{query.language}',
                'gl': query.region
            }
            
            # Add time range filter if specified
            if query.time_range:
                time_filters = {
                    'day': 'd1',
                    'week': 'w1', 
                    'month': 'm1',
                    'year': 'y1'
                }
                if query.time_range in time_filters:
                    params['dateRestrict'] = time_filters[query.time_range]
            
            # Add site filter if specified
            if query.site_filter:
                params['q'] = f"site:{query.site_filter} {query.query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for i, item in enumerate(data.get('items', [])):
                            results.append(SearchResult(
                                title=item.get('title', ''),
                                url=item.get('link', ''),
                                snippet=item.get('snippet', ''),
                                rank=i + 1,
                                source=self.name,
                                metadata={
                                    'display_link': item.get('displayLink', ''),
                                    'formatted_url': item.get('formattedUrl', ''),
                                    'cache_id': item.get('cacheId', '')
                                }
                            ))
                        
                        return results
                    else:
                        logger.error(f"Google search API error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            return []

class BingSearchProvider(SearchProvider):
    """Bing Search API provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Bing Search", config)
        self.api_key = config.get('api_key')
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        
        if not self.api_key:
            self.enabled = False
            logger.warning("Bing Search API key not configured")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using Bing Search API"""
        if not self.enabled:
            return []
        
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            params = {
                'q': query.query,
                'count': min(query.max_results, 50),  # Bing allows max 50
                'mkt': f'{query.language}-{query.region}',
                'responseFilter': 'Webpages'
            }
            
            # Add time range filter if specified
            if query.time_range:
                time_filters = {
                    'day': 'Day',
                    'week': 'Week',
                    'month': 'Month'
                }
                if query.time_range in time_filters:
                    params['freshness'] = time_filters[query.time_range]
            
            # Add site filter if specified
            if query.site_filter:
                params['q'] = f"site:{query.site_filter} {query.query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        web_pages = data.get('webPages', {}).get('value', [])
                        
                        for i, item in enumerate(web_pages):
                            results.append(SearchResult(
                                title=item.get('name', ''),
                                url=item.get('url', ''),
                                snippet=item.get('snippet', ''),
                                rank=i + 1,
                                source=self.name,
                                metadata={
                                    'display_url': item.get('displayUrl', ''),
                                    'date_last_crawled': item.get('dateLastCrawled', ''),
                                    'language': item.get('language', '')
                                }
                            ))
                        
                        return results
                    else:
                        logger.error(f"Bing search API error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Bing Search failed: {e}")
            return []

class SearxSearchProvider(SearchProvider):
    """SearX search engine provider (self-hosted or public instances)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SearX", config)
        self.instance_url = config.get('instance_url', 'https://searx.org')
        self.search_url = f"{self.instance_url}/search"
        
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search using SearX instance"""
        try:
            params = {
                'q': query.query,
                'format': 'json',
                'categories': 'general',
                'language': query.language
            }
            
            # Add site filter if specified
            if query.site_filter:
                params['q'] = f"site:{query.site_filter} {query.query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        for i, item in enumerate(data.get('results', [])[:query.max_results]):
                            results.append(SearchResult(
                                title=item.get('title', ''),
                                url=item.get('url', ''),
                                snippet=item.get('content', ''),
                                rank=i + 1,
                                source=self.name,
                                metadata={
                                    'engine': item.get('engine', ''),
                                    'score': item.get('score', 0)
                                }
                            ))
                        
                        return results
                    else:
                        logger.error(f"SearX search error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"SearX search failed: {e}")
            return []

class SearchManager:
    """Manages multiple search providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: List[SearchProvider] = []
        
        # Initialize providers
        self._initialize_providers()
        
        logger.info(f"Search manager initialized with {len(self.providers)} providers")
    
    def _initialize_providers(self):
        """Initialize search providers based on configuration"""
        provider_configs = self.config.get('providers', {})
        
        # DuckDuckGo (always available, no API key required)
        if provider_configs.get('duckduckgo', {}).get('enabled', True):
            self.providers.append(DuckDuckGoSearchProvider(
                provider_configs.get('duckduckgo', {})
            ))
        
        # Google Custom Search
        google_config = provider_configs.get('google', {})
        if google_config.get('enabled', False):
            provider = GoogleCustomSearchProvider(google_config)
            if provider.enabled:
                self.providers.append(provider)
        
        # Bing Search
        bing_config = provider_configs.get('bing', {})
        if bing_config.get('enabled', False):
            provider = BingSearchProvider(bing_config)
            if provider.enabled:
                self.providers.append(provider)
        
        # SearX
        searx_config = provider_configs.get('searx', {})
        if searx_config.get('enabled', False):
            self.providers.append(SearxSearchProvider(searx_config))
    
    async def search(self, query: SearchQuery, 
                    provider_name: Optional[str] = None) -> List[SearchResult]:
        """Search using specified provider or all available providers"""
        if provider_name:
            # Use specific provider
            provider = next((p for p in self.providers if p.name == provider_name), None)
            if provider and provider.enabled:
                return await provider.search(query)
            else:
                logger.warning(f"Provider {provider_name} not found or not enabled")
                return []
        else:
            # Use all providers and combine results
            return await self._search_all_providers(query)
    
    async def _search_all_providers(self, query: SearchQuery) -> List[SearchResult]:
        """Search using all available providers and combine results"""
        enabled_providers = [p for p in self.providers if p.enabled]
        
        if not enabled_providers:
            logger.warning("No search providers enabled")
            return []
        
        # Search with all providers concurrently
        tasks = [provider.search(query) for provider in enabled_providers]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for i, results in enumerate(results_lists):
            if isinstance(results, Exception):
                logger.error(f"Provider {enabled_providers[i].name} failed: {results}")
                continue
            
            all_results.extend(results)
        
        # Deduplicate by URL
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result
            else:
                # Keep result from higher-ranked provider or better rank
                existing = unique_results[result.url]
                if result.rank < existing.rank:
                    unique_results[result.url] = result
        
        # Sort by rank and limit results
        final_results = sorted(unique_results.values(), key=lambda x: x.rank)
        return final_results[:query.max_results]
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.name for p in self.providers if p.enabled]
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for provider in self.providers:
            status[provider.name] = {
                'enabled': provider.enabled,
                'config': {k: v for k, v in provider.config.items() if k != 'api_key'}
            }
        return status

class WebSearchIntegration:
    """Main integration class for web search functionality"""
    
    def __init__(self, search_manager: SearchManager, scraper=None):
        self.search_manager = search_manager
        self.scraper = scraper
        
        logger.info("Web search integration initialized")
    
    async def search_and_extract(self, query: str, max_results: int = 5,
                                provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for content and extract from result URLs"""
        try:
            # Perform search
            search_query = SearchQuery(
                query=query,
                max_results=max_results * 2  # Get more results for filtering
            )
            
            search_results = await self.search_manager.search(search_query, provider)
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return []
            
            # Extract content if scraper is available
            if self.scraper:
                from .scraper import ContentProcessor
                
                # Scrape content from URLs
                scraped_contents = await self.scraper.scrape_urls(
                    [result.url for result in search_results[:max_results]]
                )
                
                # Process for RAG
                processor = ContentProcessor()
                documents = processor.process_multiple_contents(scraped_contents)
                
                # Enhance with search metadata
                for i, doc in enumerate(documents):
                    if i < len(search_results):
                        search_result = search_results[i]
                        doc['metadata'].update({
                            'search_title': search_result.title,
                            'search_snippet': search_result.snippet,
                            'search_rank': search_result.rank,
                            'search_source': search_result.source
                        })
                
                return documents
            else:
                # Return search results without content extraction
                return [
                    {
                        'title': result.title,
                        'url': result.url,
                        'content': result.snippet,
                        'source': result.source,
                        'rank': result.rank,
                        'metadata': result.metadata or {}
                    }
                    for result in search_results[:max_results]
                ]
                
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
    
    async def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query"""
        # This would typically integrate with search suggestion APIs
        # For now, return simple variations
        suggestions = [
            query,
            f"{query} tutorial",
            f"{query} guide",
            f"{query} examples",
            f"how to {query}"
        ]
        
        return suggestions[:5]

