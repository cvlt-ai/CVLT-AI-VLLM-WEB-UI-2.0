"""
Inference Engine for vLLM Gradio WebUI

Handles inference requests, response processing, and coordination between
different components (RAG, web scraper, file processor).
"""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum

from .vllm_manager import VLLMManager, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

class RequestType(Enum):
    """Types of inference requests"""
    SIMPLE = "simple"
    RAG = "rag"
    WEB_ENHANCED = "web_enhanced"
    FILE_BASED = "file_based"

@dataclass
class EnhancedInferenceRequest:
    """Enhanced inference request with additional features"""
    prompt: str
    request_type: RequestType = RequestType.SIMPLE
    
    # Basic inference parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = True
    thinking_mode: bool = False
    
    # RAG parameters
    use_rag: bool = False
    rag_top_k: int = 5
    rag_threshold: float = 0.7
    
    # Web enhancement parameters
    use_web: bool = False
    web_search_queries: List[str] = field(default_factory=list)
    max_web_results: int = 5
    
    # File processing parameters
    uploaded_files: List[str] = field(default_factory=list)
    
    # System parameters
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class EnhancedInferenceResponse:
    """Enhanced inference response with metadata"""
    text: str
    finish_reason: str
    usage: Dict[str, int]
    
    # Enhancement metadata
    rag_context: Optional[List[Dict[str, Any]]] = None
    web_results: Optional[List[Dict[str, Any]]] = None
    processed_files: Optional[List[Dict[str, Any]]] = None
    
    # Timing information
    processing_time: float = 0.0
    rag_time: float = 0.0
    web_time: float = 0.0
    inference_time: float = 0.0
    
    # Request metadata
    request_id: Optional[str] = None
    timestamp: Optional[float] = None

class InferenceEngine:
    """Main inference engine coordinating all components"""
    
    def __init__(self, vllm_manager: VLLMManager, rag_pipeline=None, 
                 web_scraper=None, file_manager=None):
        self.vllm_manager = vllm_manager
        self.rag_pipeline = rag_pipeline
        self.web_scraper = web_scraper
        self.file_manager = file_manager
        
        # Request tracking
        self.active_requests: Dict[str, EnhancedInferenceRequest] = {}
        self.request_history: List[EnhancedInferenceResponse] = []
        self.max_history = 1000
        
        logger.info("Inference Engine initialized")
    
    async def generate(self, prompt: str, generation_params: Dict[str, Any] = None, 
                      rag_params: Dict[str, Any] = None, web_params: Dict[str, Any] = None,
                      stream: bool = False, simple_mode: bool = False) -> Dict[str, Any]:
        """Generate response - wrapper around process_request for compatibility"""
        try:
            # Extract parameters with defaults
            generation_params = generation_params or {}
            rag_params = rag_params or {}
            web_params = web_params or {}
            
            # Check if we should use simple mode (bypass all enhancements)
            use_rag = rag_params.get('enabled', False) and not simple_mode
            use_web = web_params.get('enabled', False) and not simple_mode
            
            # If no enhancements needed or simple mode, use basic inference
            if simple_mode or (not use_rag and not use_web):
                return await self._simple_generate(prompt, generation_params, stream)
            
            # Create enhanced inference request
            request = EnhancedInferenceRequest(
                prompt=prompt,
                temperature=generation_params.get('temperature', 0.7),
                top_p=generation_params.get('top_p', 0.9),
                top_k=generation_params.get('top_k', 50),
                max_tokens=generation_params.get('max_tokens', 512),
                stop_sequences=generation_params.get('stop_sequences', []),
                stream=stream,
                use_rag=rag_params.get('enabled', False),
                rag_top_k=rag_params.get('top_k', 5),
                rag_threshold=rag_params.get('threshold', 0.7),
                use_web=web_params.get('enabled', False),
                web_search_queries=web_params.get('queries', []),
                max_web_results=web_params.get('max_results', 5)
            )
            
            # Process the request
            result = await self.process_request(request)
            
            # Convert result to expected format
            if hasattr(result, 'text'):
                # Sync response
                return {
                    'success': True,
                    'text': result.text,
                    'response': result.text,  # For compatibility
                    'finish_reason': result.finish_reason,
                    'usage': result.usage,
                    'request_id': getattr(result, 'request_id', None),
                    'processing_time': result.processing_time,
                    'rag_context': result.rag_context,
                    'web_results': result.web_results,
                    'processed_files': result.processed_files,
                    'rag_time': result.rag_time,
                    'web_time': result.web_time
                }
            else:
                # Streaming response - return as is
                return {
                    'success': True,
                    'stream': result
                }
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _simple_generate(self, prompt: str, generation_params: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
        """Simple generation that bypasses all enhancements"""
        try:
            start_time = time.time()
            
            # Create basic inference request
            basic_request = InferenceRequest(
                prompt=prompt,
                temperature=generation_params.get('temperature', 0.7),
                top_p=generation_params.get('top_p', 0.9),
                top_k=generation_params.get('top_k', 50),
                max_tokens=generation_params.get('max_tokens', 512),
                stop_sequences=generation_params.get('stop_sequences', []),
                stream=stream,
                request_id=str(uuid.uuid4())
            )
            
            # Direct inference without enhancements
            if not self.vllm_manager or not self.vllm_manager.is_ready:
                return {
                    'success': False,
                    'error': 'Model not loaded or not ready'
                }
            
            # Generate response
            result = await self.vllm_manager.generate(basic_request)
            processing_time = time.time() - start_time
            
            if stream:
                return {
                    'success': True,
                    'stream': result
                }
            else:
                return {
                    'success': True,
                    'text': result.text,
                    'response': result.text,  # For compatibility
                    'finish_reason': result.finish_reason,
                    'usage': result.usage,
                    'processing_time': processing_time,
                    'rag_context': None,
                    'web_results': None,
                    'processed_files': None,
                    'rag_time': 0.0,
                    'web_time': 0.0
                }
                
        except Exception as e:
            logger.error(f"Simple generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_request(self, request: EnhancedInferenceRequest) -> Union[EnhancedInferenceResponse, AsyncGenerator[str, None]]:
        """Process an enhanced inference request"""
        start_time = time.time()
        
        # Generate request ID if not provided
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        
        # Track active request
        self.active_requests[request.request_id] = request
        
        try:
            # Build enhanced prompt
            enhanced_prompt = await self._build_enhanced_prompt(request)
            
            # Create basic inference request
            basic_request = InferenceRequest(
                prompt=enhanced_prompt['prompt'],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences,
                stream=request.stream,
                request_id=request.request_id
            )
            
            # Perform inference
            inference_start = time.time()
            
            if request.stream:
                return self._process_streaming_request(
                    basic_request, enhanced_prompt, start_time, inference_start
                )
            else:
                return await self._process_sync_request(
                    basic_request, enhanced_prompt, start_time, inference_start
                )
                
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}", exc_info=True)
            # Remove from active requests
            self.active_requests.pop(request.request_id, None)
            raise
    
    async def _build_enhanced_prompt(self, request: EnhancedInferenceRequest) -> Dict[str, Any]:
        """Build enhanced prompt with RAG, web, and file context"""
        enhancement_data = {
            'prompt': request.prompt,
            'rag_context': None,
            'web_results': None,
            'processed_files': None,
            'rag_time': 0.0,
            'web_time': 0.0
        }
        
        context_parts = []
        
        # Add system prompt if provided
        if request.system_prompt:
            context_parts.append(f"System: {request.system_prompt}")
        
        # Process uploaded files
        if request.uploaded_files and self.file_manager:
            try:
                file_start = time.time()
                file_context = await self._process_uploaded_files(request.uploaded_files)
                if file_context:
                    context_parts.append(f"File Content:\n{file_context['text']}")
                    enhancement_data['processed_files'] = file_context['files']
                logger.info(f"File processing took {time.time() - file_start:.2f}s")
            except Exception as e:
                logger.warning(f"File processing failed: {e}")
        
        # Perform RAG if enabled
        if request.use_rag and self.rag_pipeline:
            try:
                rag_start = time.time()
                rag_context = await self._perform_rag_retrieval(request)
                if rag_context:
                    context_parts.append(f"Relevant Context:\n{rag_context['text']}")
                    enhancement_data['rag_context'] = rag_context['sources']
                enhancement_data['rag_time'] = time.time() - rag_start
                logger.info(f"RAG retrieval took {enhancement_data['rag_time']:.2f}s")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Perform web search if enabled
        if request.use_web and self.web_scraper:
            try:
                web_start = time.time()
                web_context = await self._perform_web_search(request)
                if web_context:
                    context_parts.append(f"Web Information:\n{web_context['text']}")
                    enhancement_data['web_results'] = web_context['sources']
                enhancement_data['web_time'] = time.time() - web_start
                logger.info(f"Web search took {enhancement_data['web_time']:.2f}s")
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Add existing context if provided
        if request.context:
            context_parts.append(f"Context: {request.context}")
        
        # Build final prompt
        if context_parts:
            enhanced_prompt = "\n\n".join(context_parts) + f"\n\nUser: {request.prompt}\n\nAssistant:"
        else:
            enhanced_prompt = request.prompt
        
        # Add thinking mode prompting for reasoning models
        if request.thinking_mode:
            thinking_prefix = """<thinking>
Let me think through this step by step and reason about the best approach to answer this question.

I should:
1. Understand what is being asked
2. Consider different perspectives or approaches
3. Think through the logic carefully
4. Provide a well-reasoned response

Let me work through this:
</thinking>

"""
            if context_parts:
                enhanced_prompt = "\n\n".join(context_parts) + f"\n\nUser: {request.prompt}\n\nAssistant: {thinking_prefix}"
            else:
                enhanced_prompt = f"User: {request.prompt}\n\nAssistant: {thinking_prefix}"
        
        enhancement_data['prompt'] = enhanced_prompt
        return enhancement_data
    
    async def _process_uploaded_files(self, file_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Process uploaded files and extract content"""
        if not self.file_manager:
            return None
        
        combined_text = []
        processed_files = []
        
        for file_path in file_paths:
            try:
                file_info = await self.file_manager.process_file(file_path)
                if file_info and file_info.get('text'):
                    combined_text.append(f"File: {file_info['filename']}\n{file_info['text']}")
                    processed_files.append({
                        'filename': file_info['filename'],
                        'type': file_info['type'],
                        'size': file_info.get('size', 0),
                        'pages': file_info.get('pages', 1)
                    })
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")
        
        if combined_text:
            return {
                'text': '\n\n'.join(combined_text),
                'files': processed_files
            }
        
        return None
    
    async def _perform_rag_retrieval(self, request: EnhancedInferenceRequest) -> Optional[Dict[str, Any]]:
        """Perform RAG retrieval"""
        if not self.rag_pipeline:
            return None
        
        try:
            results = await self.rag_pipeline.retrieve(
                query=request.prompt,
                top_k=request.rag_top_k,
                threshold=request.rag_threshold
            )
            
            if results:
                context_text = '\n\n'.join([doc['content'] for doc in results])
                sources = [
                    {
                        'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                        'source': doc.get('source', 'Unknown'),
                        'score': doc.get('score', 0.0)
                    }
                    for doc in results
                ]
                
                return {
                    'text': context_text,
                    'sources': sources
                }
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
        
        return None
    
    async def _perform_web_search(self, request: EnhancedInferenceRequest) -> Optional[Dict[str, Any]]:
        """Perform web search and content extraction"""
        if not self.web_scraper:
            return None
        
        try:
            # Use provided search queries or generate from prompt
            search_queries = request.web_search_queries or [request.prompt]
            
            all_results = []
            for query in search_queries[:3]:  # Limit to 3 queries
                results = await self.web_scraper.search_and_extract(
                    query=query,
                    max_results=request.max_web_results
                )
                all_results.extend(results)
            
            if all_results:
                # Deduplicate and limit results
                unique_results = []
                seen_urls = set()
                for result in all_results:
                    if result.get('url') not in seen_urls:
                        unique_results.append(result)
                        seen_urls.add(result.get('url'))
                        if len(unique_results) >= request.max_web_results:
                            break
                
                context_text = '\n\n'.join([
                    f"Source: {result.get('title', 'Unknown')}\n{result.get('content', '')}"
                    for result in unique_results
                ])
                
                sources = [
                    {
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('url', ''),
                        'content': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                        'timestamp': result.get('timestamp')
                    }
                    for result in unique_results
                ]
                
                return {
                    'text': context_text,
                    'sources': sources
                }
        except Exception as e:
            logger.error(f"Web search error: {e}")
        
        return None
    
    async def _process_sync_request(self, basic_request: InferenceRequest, 
                                  enhanced_prompt: Dict[str, Any], 
                                  start_time: float, inference_start: float) -> EnhancedInferenceResponse:
        """Process synchronous inference request"""
        try:
            # Perform inference
            response = await self.vllm_manager.generate(basic_request)
            inference_time = time.time() - inference_start
            
            # Create enhanced response
            enhanced_response = EnhancedInferenceResponse(
                text=response.text,
                finish_reason=response.finish_reason,
                usage=response.usage,
                rag_context=enhanced_prompt.get('rag_context'),
                web_results=enhanced_prompt.get('web_results'),
                processed_files=enhanced_prompt.get('processed_files'),
                processing_time=time.time() - start_time,
                rag_time=enhanced_prompt.get('rag_time', 0.0),
                web_time=enhanced_prompt.get('web_time', 0.0),
                inference_time=inference_time,
                request_id=basic_request.request_id,
                timestamp=time.time()
            )
            
            # Add to history
            self._add_to_history(enhanced_response)
            
            # Remove from active requests
            self.active_requests.pop(basic_request.request_id, None)
            
            return enhanced_response
            
        except Exception as e:
            # Remove from active requests
            self.active_requests.pop(basic_request.request_id, None)
            raise
    
    async def _process_streaming_request(self, basic_request: InferenceRequest,
                                       enhanced_prompt: Dict[str, Any],
                                       start_time: float, inference_start: float) -> AsyncGenerator[str, None]:
        """Process streaming inference request"""
        try:
            accumulated_text = ""
            
            async for chunk in self.vllm_manager.generate(basic_request):
                accumulated_text = chunk
                yield chunk
            
            # Create final response for history
            inference_time = time.time() - inference_start
            enhanced_response = EnhancedInferenceResponse(
                text=accumulated_text,
                finish_reason="stop",  # Assume stop for streaming
                usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},  # Not available in streaming
                rag_context=enhanced_prompt.get('rag_context'),
                web_results=enhanced_prompt.get('web_results'),
                processed_files=enhanced_prompt.get('processed_files'),
                processing_time=time.time() - start_time,
                rag_time=enhanced_prompt.get('rag_time', 0.0),
                web_time=enhanced_prompt.get('web_time', 0.0),
                inference_time=inference_time,
                request_id=basic_request.request_id,
                timestamp=time.time()
            )
            
            # Add to history
            self._add_to_history(enhanced_response)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            # Remove from active requests
            self.active_requests.pop(basic_request.request_id, None)
    
    def _add_to_history(self, response: EnhancedInferenceResponse):
        """Add response to history with size limit"""
        self.request_history.append(response)
        
        # Maintain history size limit
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active requests"""
        return {
            request_id: {
                'request_type': request.request_type.value,
                'prompt_length': len(request.prompt),
                'use_rag': request.use_rag,
                'use_web': request.use_web,
                'uploaded_files': len(request.uploaded_files),
                'user_id': request.user_id
            }
            for request_id, request in self.active_requests.items()
        }
    
    def get_request_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get request history"""
        recent_history = self.request_history[-limit:] if limit else self.request_history
        
        return [
            {
                'request_id': response.request_id,
                'timestamp': response.timestamp,
                'text_length': len(response.text),
                'processing_time': response.processing_time,
                'rag_time': response.rag_time,
                'web_time': response.web_time,
                'inference_time': response.inference_time,
                'finish_reason': response.finish_reason,
                'usage': response.usage,
                'has_rag': response.rag_context is not None,
                'has_web': response.web_results is not None,
                'has_files': response.processed_files is not None
            }
            for response in recent_history
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        total_requests = len(self.request_history)
        
        if total_requests == 0:
            return {
                'total_requests': 0,
                'active_requests': len(self.active_requests),
                'average_processing_time': 0.0,
                'rag_usage_rate': 0.0,
                'web_usage_rate': 0.0,
                'file_usage_rate': 0.0
            }
        
        # Calculate averages
        avg_processing_time = sum(r.processing_time for r in self.request_history) / total_requests
        rag_usage = sum(1 for r in self.request_history if r.rag_context) / total_requests
        web_usage = sum(1 for r in self.request_history if r.web_results) / total_requests
        file_usage = sum(1 for r in self.request_history if r.processed_files) / total_requests
        
        return {
            'total_requests': total_requests,
            'active_requests': len(self.active_requests),
            'average_processing_time': avg_processing_time,
            'rag_usage_rate': rag_usage,
            'web_usage_rate': web_usage,
            'file_usage_rate': file_usage,
            'vllm_stats': self.vllm_manager.get_engine_stats()
        }
    
    async def cleanup(self):
        """Cleanup inference engine resources"""
        try:
            # Cancel active requests
            self.active_requests.clear()
            
            # Clear history
            self.request_history.clear()
            
            logger.info("Inference engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during inference engine cleanup: {e}")

