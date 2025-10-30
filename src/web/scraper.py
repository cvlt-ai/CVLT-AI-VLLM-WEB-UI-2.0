"""
Web Scraper for vLLM Gradio WebUI

Handles web content extraction, URL processing, and content cleaning
for web-enhanced RAG functionality.
"""

import logging
import asyncio
import aiohttp
import time
import re
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import ssl

logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Scraped content structure"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""
    timeout: int = 30
    max_content_length: int = 100000  # 100KB
    user_agent: str = "Mozilla/5.0 (compatible; vLLM-Gradio-WebUI/1.0)"
    max_redirects: int = 5
    verify_ssl: bool = False
    allowed_content_types: List[str] = None
    blocked_domains: List[str] = None

class WebScraper:
    """Web content scraper and processor"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        
        # Set default allowed content types
        if self.config.allowed_content_types is None:
            self.config.allowed_content_types = [
                'text/html',
                'text/plain',
                'application/xhtml+xml'
            ]
        
        # Set default blocked domains
        if self.config.blocked_domains is None:
            self.config.blocked_domains = [
                'facebook.com',
                'twitter.com',
                'instagram.com',
                'tiktok.com'
            ]
        
        # Create SSL context
        self.ssl_context = ssl.create_default_context()
        if not self.config.verify_ssl:
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
        
        logger.info("Web scraper initialized")
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape content from a single URL"""
        start_time = time.time()
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return ScrapedContent(
                    url=url,
                    title="",
                    content="",
                    metadata={},
                    timestamp=start_time,
                    success=False,
                    error_message="Invalid URL"
                )
            
            # Check if domain is blocked
            if self._is_blocked_domain(url):
                return ScrapedContent(
                    url=url,
                    title="",
                    content="",
                    metadata={},
                    timestamp=start_time,
                    success=False,
                    error_message="Blocked domain"
                )
            
            # Create session with timeout
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(ssl=self.ssl_context)
            ) as session:
                
                headers = {
                    'User-Agent': self.config.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
                
                async with session.get(
                    url, 
                    headers=headers,
                    max_redirects=self.config.max_redirects
                ) as response:
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(ct in content_type for ct in self.config.allowed_content_types):
                        return ScrapedContent(
                            url=url,
                            title="",
                            content="",
                            metadata={'content_type': content_type},
                            timestamp=start_time,
                            success=False,
                            error_message=f"Unsupported content type: {content_type}"
                        )
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.config.max_content_length:
                        return ScrapedContent(
                            url=url,
                            title="",
                            content="",
                            metadata={'content_length': content_length},
                            timestamp=start_time,
                            success=False,
                            error_message="Content too large"
                        )
                    
                    # Read content
                    html_content = await response.text()
                    
                    # Limit content size
                    if len(html_content) > self.config.max_content_length:
                        html_content = html_content[:self.config.max_content_length]
                    
                    # Parse and extract content
                    extracted = self._extract_content(html_content, url)
                    
                    return ScrapedContent(
                        url=url,
                        title=extracted['title'],
                        content=extracted['content'],
                        metadata=extracted['metadata'],
                        timestamp=start_time,
                        success=True
                    )
                    
        except asyncio.TimeoutError:
            return ScrapedContent(
                url=url,
                title="",
                content="",
                metadata={},
                timestamp=start_time,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return ScrapedContent(
                url=url,
                title="",
                content="",
                metadata={},
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )
    
    async def scrape_urls(self, urls: List[str], 
                         max_concurrent: int = 5) -> List[ScrapedContent]:
        """Scrape content from multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        scraped_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scraped_results.append(ScrapedContent(
                    url=urls[i],
                    title="",
                    content="",
                    metadata={},
                    timestamp=time.time(),
                    success=False,
                    error_message=str(result)
                ))
            else:
                scraped_results.append(result)
        
        logger.info(f"Scraped {len(urls)} URLs, {sum(1 for r in scraped_results if r.success)} successful")
        return scraped_results
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ['http', 'https']
        except:
            return False
    
    def _is_blocked_domain(self, url: str) -> bool:
        """Check if domain is blocked"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(blocked in domain for blocked in self.config.blocked_domains)
        except:
            return False
    
    def _extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract and clean content from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'iframe', 'noscript', 'form']):
                element.decompose()
            
            # Try to find main content
            content_text = ""
            
            # Look for common content containers
            content_selectors = [
                'article',
                '[role="main"]',
                'main',
                '.content',
                '.post-content',
                '.entry-content',
                '.article-content',
                '#content',
                '#main-content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content_text = content_element.get_text()
                    break
            
            # Fallback to body content
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text()
                else:
                    content_text = soup.get_text()
            
            # Clean content
            content_text = self._clean_text(content_text)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return {
                'title': title,
                'content': content_text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract content: {e}")
            return {
                'title': "",
                'content': "",
                'metadata': {'extraction_error': str(e)}
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common navigation text
        navigation_patterns = [
            r'skip to content',
            r'skip to main content',
            r'menu',
            r'navigation',
            r'breadcrumb',
            r'search',
            r'login',
            r'register',
            r'sign up',
            r'sign in'
        ]
        
        for pattern in navigation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Limit length
        if len(text) > self.config.max_content_length:
            text = text[:self.config.max_content_length] + "..."
        
        return text
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc
        }
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            
            if name and content:
                if name in ['description', 'keywords', 'author']:
                    metadata[name] = content
                elif name.startswith('og:'):
                    metadata[name] = content
                elif name.startswith('twitter:'):
                    metadata[name] = content
        
        # Extract language
        html_tag = soup.find('html')
        if html_tag:
            lang = html_tag.get('lang')
            if lang:
                metadata['language'] = lang
        
        # Extract canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical:
            metadata['canonical_url'] = canonical.get('href')
        
        return metadata

class SearchEnhancedScraper:
    """Web scraper with search capabilities"""
    
    def __init__(self, scraper: WebScraper = None):
        self.scraper = scraper or WebScraper()
        
        logger.info("Search-enhanced scraper initialized")
    
    async def search_and_extract(self, query: str, 
                                max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for content and extract from results"""
        try:
            # For now, this is a placeholder implementation
            # In a real implementation, you would integrate with search APIs
            # like Google Custom Search, Bing Search API, etc.
            
            logger.warning("Search functionality requires API integration")
            
            # Return empty results for now
            return []
            
        except Exception as e:
            logger.error(f"Search and extract failed: {e}")
            return []
    
    async def extract_from_search_results(self, search_results: List[Dict[str, str]]) -> List[ScrapedContent]:
        """Extract content from search result URLs"""
        urls = [result.get('url', '') for result in search_results if result.get('url')]
        
        if not urls:
            return []
        
        scraped_contents = await self.scraper.scrape_urls(urls)
        
        # Enhance with search result metadata
        enhanced_contents = []
        for i, content in enumerate(scraped_contents):
            if i < len(search_results):
                search_result = search_results[i]
                content.metadata.update({
                    'search_title': search_result.get('title', ''),
                    'search_snippet': search_result.get('snippet', ''),
                    'search_rank': i + 1
                })
            enhanced_contents.append(content)
        
        return enhanced_contents

class ContentProcessor:
    """Process and prepare scraped content for RAG"""
    
    def __init__(self):
        logger.info("Content processor initialized")
    
    def process_scraped_content(self, scraped_content: ScrapedContent) -> Dict[str, Any]:
        """Process scraped content for RAG ingestion"""
        if not scraped_content.success:
            return None
        
        # Prepare document for RAG
        document = {
            'id': f"web_{hash(scraped_content.url)}_{int(scraped_content.timestamp)}",
            'content': scraped_content.content,
            'type': 'web',
            'path': scraped_content.url,
            'metadata': {
                'title': scraped_content.title,
                'url': scraped_content.url,
                'domain': scraped_content.metadata.get('domain', ''),
                'scraped_at': scraped_content.timestamp,
                'content_length': len(scraped_content.content),
                **scraped_content.metadata
            }
        }
        
        return document
    
    def process_multiple_contents(self, scraped_contents: List[ScrapedContent]) -> List[Dict[str, Any]]:
        """Process multiple scraped contents"""
        documents = []
        
        for content in scraped_contents:
            doc = self.process_scraped_content(content)
            if doc:
                documents.append(doc)
        
        logger.info(f"Processed {len(documents)} web documents for RAG")
        return documents
    
    def deduplicate_content(self, documents: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Remove duplicate content based on similarity"""
        if len(documents) <= 1:
            return documents
        
        # Simple deduplication based on content length and first 100 characters
        unique_documents = []
        seen_signatures = set()
        
        for doc in documents:
            content = doc.get('content', '')
            signature = (len(content), content[:100])
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_documents.append(doc)
        
        logger.info(f"Deduplicated {len(documents)} -> {len(unique_documents)} documents")
        return unique_documents

