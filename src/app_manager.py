"""
Application Manager for vLLM Gradio WebUI

Main application manager that coordinates all components including vLLM,
RAG, web functionality, file processing, and provides unified interface
for the Gradio UI.
"""

import logging
import asyncio
import time
import json
import os
import psutil
import GPUtil
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from core.vllm_manager import VLLMManager
from core.inference_engine import InferenceEngine
from rag.pipeline import RAGPipeline
from web.web_manager import WebManager, WebEnhancementRequest
from files.file_manager import FileManager, FileManagerConfig
from files.file_processor import FileProcessor, ProcessingConfig
from utils.config import ConfigManager
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

class ApplicationManager:
    """Main application manager coordinating all components"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        logging_config = getattr(self.config, 'logging', {}) if hasattr(self.config, 'logging') else {}
        setup_logging(logging_config)
        
        # Initialize components
        self.vllm_manager = None
        self.inference_engine = None
        self.rag_pipeline = None
        self.web_manager = None
        self.file_manager = None
        
        # Application state
        self.initialized = False
        self.current_model = None
        
        logger.info("Application manager created")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing application components...")
            
            # Initialize RAG pipeline only if enabled
            rag_config = self.config.rag
            if getattr(rag_config, 'enabled', False):
                self.rag_pipeline = RAGPipeline(rag_config)
                logger.info("RAG pipeline initialized")
            else:
                self.rag_pipeline = None
                logger.info("RAG pipeline disabled in configuration")
            
            # Initialize file processor and manager only if enabled
            if getattr(self.config.file_processor, 'enabled', False):
                file_config = FileManagerConfig(
                    upload_dir=getattr(self.config.file_processor, 'upload_dir', './data/uploads'),
                    processed_dir=getattr(self.config.file_processor, 'processed_dir', './data/processed'),
                    metadata_dir=getattr(self.config.file_processor, 'metadata_dir', './data/metadata'),
                    max_file_size=getattr(self.config.file_processor, 'max_file_size', 50 * 1024 * 1024),
                    auto_process=getattr(self.config.file_processor, 'auto_process', True)
                )
                
                processing_config = ProcessingConfig(
                    max_file_size=file_config.max_file_size,
                    extract_images=getattr(self.config.file_processor, 'extract_images', True),
                    ocr_enabled=getattr(self.config.file_processor, 'ocr_enabled', False)
                )
                
                file_processor = FileProcessor(processing_config)
                self.file_manager = FileManager(file_config, file_processor, self.rag_pipeline)
                logger.info("File manager initialized")
            else:
                self.file_manager = None
                logger.info("File processor disabled in configuration")
            
            # Initialize web manager only if enabled
            web_config = self.config.web_scraper
            if getattr(web_config, 'enabled', False):
                self.web_manager = WebManager(web_config, self.rag_pipeline)
                logger.info("Web manager initialized")
            else:
                self.web_manager = None
                logger.info("Web scraper disabled in configuration")
            
            # Initialize vLLM manager
            from core.gpu_manager import GPUManager
            gpu_manager = GPUManager(self.config.vllm.gpu)
            
            vllm_config = self.config.vllm
            self.vllm_manager = VLLMManager(vllm_config, gpu_manager)
            
            # Initialize inference engine
            inference_config = getattr(self.config, 'inference', {})
            self.inference_engine = InferenceEngine(
                self.vllm_manager,
                self.rag_pipeline,
                self.web_manager,
                inference_config
            )
            
            self.initialized = True
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}", exc_info=True)
            raise
    
    async def cleanup(self):
        """Cleanup all components"""
        try:
            logger.info("Cleaning up application components...")
            
            if self.inference_engine:
                await self.inference_engine.cleanup()
            
            if self.vllm_manager:
                await self.vllm_manager.cleanup()
            
            if self.rag_pipeline:
                await self.rag_pipeline.cleanup()
            
            if self.web_manager:
                await self.web_manager.cleanup()
            
            logger.info("Application cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Model Management Methods
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        try:
            if not self.vllm_manager:
                return {'success': False, 'error': 'vLLM manager not initialized'}
            
            models = self.vllm_manager.discover_models()
            return {
                'success': True,
                'models': models
            }
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return {'success': False, 'error': str(e)}
    
    async def scan_models(self, path: str) -> Dict[str, Any]:
        """Scan directory for models"""
        try:
            if not self.vllm_manager:
                return {'success': False, 'error': 'vLLM manager not initialized'}
            
            models = self.vllm_manager.scan_models_directory(path)
            return {
                'success': True,
                'models': models
            }
            
        except Exception as e:
            logger.error(f"Failed to scan models: {e}")
            return {'success': False, 'error': str(e)}
    
    async def load_model(self, model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load a model"""
        try:
            if not self.vllm_manager:
                return {'success': False, 'error': 'vLLM manager not initialized'}
            
            result = await self.vllm_manager.load_model(model_name, config)
            
            if result.get('success'):
                self.current_model = model_name
                logger.info(f"Model loaded successfully: {model_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {'success': False, 'error': str(e)}
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload current model"""
        try:
            if not self.vllm_manager:
                return {'success': False, 'error': 'vLLM manager not initialized'}
            
            result = await self.vllm_manager.unload_model()
            
            if result['success']:
                self.current_model = None
                logger.info("Model unloaded successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        try:
            if not self.vllm_manager:
                return {'success': False, 'error': 'vLLM manager not initialized'}
            
            gpu_info = self.vllm_manager.get_gpu_status()
            return {
                'success': True,
                'gpus': gpu_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
            return {'success': False, 'error': str(e)}
    
    # Generation Methods
    
    async def generate_response(self, message: str, generation_params: Dict[str, Any] = None,
                              rag_params: Dict[str, Any] = None, web_params: Dict[str, Any] = None,
                              stream: bool = False, simple_mode: bool = True) -> Dict[str, Any]:
        """Generate response using the inference engine"""
        try:
            if not self.inference_engine:
                return {'success': False, 'error': 'Inference engine not initialized'}
            
            if not self.current_model:
                return {'success': False, 'error': 'No model loaded'}
            
            # Use simple mode by default to avoid conflicts
            # Only use enhanced mode if explicitly requested and components are available
            use_simple = simple_mode
            if not simple_mode:
                # Check if enhancement components are available
                rag_enabled = rag_params and rag_params.get('enabled', False) and self.rag_pipeline
                web_enabled = web_params and web_params.get('enabled', False) and self.web_manager
                
                # If no enhancements are actually available, use simple mode
                if not rag_enabled and not web_enabled:
                    use_simple = True
            
            result = await self.inference_engine.generate(
                prompt=message,
                generation_params=generation_params or {},
                rag_params=rag_params or {},
                web_params=web_params or {},
                stream=stream,
                simple_mode=use_simple
            )
            
            # Handle streaming response
            if stream and hasattr(result, '__aiter__'):
                # Collect streaming tokens
                full_text = ""
                async for token in result:
                    full_text += token
                
                return {
                    'success': True,
                    'text': full_text,
                    'response': full_text,
                    'streaming': True
                }
            else:
                # Regular response
                return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {'success': False, 'error': str(e)} 
    # RAG Methods
    
    async def ingest_text(self, texts: List[str], 
                         metadata_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest text documents into RAG"""
        try:
            if not self.rag_pipeline:
                return {'success': False, 'error': 'RAG pipeline not initialized'}
            
            result = await self.rag_pipeline.ingest_text_documents(texts, metadata_list)
            
            return {
                'success': result.success,
                'documents_processed': result.documents_processed,
                'chunks_created': result.chunks_created,
                'processing_time': result.processing_time,
                'error': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            return {'success': False, 'error': str(e)}
    
    async def ingest_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Ingest files into RAG"""
        try:
            if not self.rag_pipeline:
                return {'success': False, 'error': 'RAG pipeline not initialized'}
            
            result = await self.rag_pipeline.ingest_file_documents(file_paths)
            
            return {
                'success': result.success,
                'documents_processed': result.documents_processed,
                'chunks_created': result.chunks_created,
                'processing_time': result.processing_time,
                'error': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest files: {e}")
            return {'success': False, 'error': str(e)}
    
    async def ingest_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Ingest web URLs into RAG"""
        try:
            if not self.web_manager:
                return {'success': False, 'error': 'Web manager not initialized'}
            
            # Scrape URLs and get documents
            scraped_results = []
            for url in urls:
                result = await self.web_manager.scrape_url(url)
                if result:
                    scraped_results.append(result)
            
            if not scraped_results:
                return {'success': False, 'error': 'No content scraped from URLs'}
            
            # Ingest into RAG
            if self.rag_pipeline:
                result = await self.rag_pipeline.ingest_documents(scraped_results)
                
                return {
                    'success': result.success,
                    'documents_processed': result.documents_processed,
                    'chunks_created': result.chunks_created,
                    'processing_time': result.processing_time,
                    'error': result.error_message
                }
            else:
                return {'success': False, 'error': 'RAG pipeline not initialized'}
            
        except Exception as e:
            logger.error(f"Failed to ingest URLs: {e}")
            return {'success': False, 'error': str(e)}
    
    async def search_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents in RAG"""
        try:
            if not self.rag_pipeline:
                return {'success': False, 'error': 'RAG pipeline not initialized'}
            
            results = await self.rag_pipeline.retrieve(query, top_k)
            
            return {
                'success': True,
                'documents': results
            }
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            if not self.rag_pipeline:
                return {'error': 'RAG pipeline not initialized'}
            
            stats = await self.rag_pipeline.get_statistics()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {'error': str(e)}
    
    async def clear_rag_documents(self) -> Dict[str, Any]:
        """Clear all RAG documents"""
        try:
            if not self.rag_pipeline:
                return {'success': False, 'error': 'RAG pipeline not initialized'}
            
            success = await self.rag_pipeline.clear_all_documents()
            
            return {
                'success': success,
                'error': None if success else 'Failed to clear documents'
            }
            
        except Exception as e:
            logger.error(f"Failed to clear RAG documents: {e}")
            return {'success': False, 'error': str(e)}
    
    # File Management Methods
    
    async def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Upload and process a file"""
        try:
            if not self.file_manager:
                return {'success': False, 'error': 'File manager not initialized'}
            
            result = await self.file_manager.upload_file_from_path(file_path)
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return {'success': False, 'error': str(e)}
    
    async def list_files(self) -> Dict[str, Any]:
        """List uploaded files"""
        try:
            if not self.file_manager:
                return {'success': False, 'error': 'File manager not initialized'}
            
            files = self.file_manager.list_files()
            
            return {
                'success': True,
                'files': files
            }
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            if not self.file_manager:
                return {'success': False, 'error': 'File manager not initialized'}
            
            file_info = self.file_manager.get_file_info(file_id)
            
            if file_info:
                return {
                    'success': True,
                    'file_info': file_info
                }
            else:
                return {'success': False, 'error': 'File not found'}
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {'success': False, 'error': str(e)}
    
    async def process_all_files(self) -> Dict[str, Any]:
        """Process all unprocessed files"""
        try:
            if not self.file_manager:
                return {'success': False, 'error': 'File manager not initialized'}
            
            result = await self.file_manager.process_all_files()
            return result
            
        except Exception as e:
            logger.error(f"Failed to process files: {e}")
            return {'success': False, 'error': str(e)}
    
    # Web Methods
    
    async def search_web(self, query: str, max_results: int = 10, 
                        provider: str = None) -> Dict[str, Any]:
        """Search the web"""
        try:
            if not self.web_manager:
                return {'success': False, 'error': 'Web manager not initialized'}
            
            results = await self.web_manager.search_web(query, max_results, provider)
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to search web: {e}")
            return {'success': False, 'error': str(e)}
    
    async def scrape_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Scrape content from URLs"""
        try:
            if not self.web_manager:
                return {'success': False, 'error': 'Web manager not initialized'}
            
            results = []
            for url in urls:
                result = await self.web_manager.scrape_url(url)
                if result:
                    results.append({
                        'url': url,
                        'title': result.get('metadata', {}).get('title', ''),
                        'content': result.get('content', ''),
                        'success': True
                    })
                else:
                    results.append({
                        'url': url,
                        'title': '',
                        'content': '',
                        'success': False
                    })
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape URLs: {e}")
            return {'success': False, 'error': str(e)}
    
    async def enhance_with_web(self, query: str, include_search: bool = True,
                              include_scraping: bool = True) -> Dict[str, Any]:
        """Enhance query with web content"""
        try:
            if not self.web_manager:
                return {'success': False, 'error': 'Web manager not initialized'}
            
            request = WebEnhancementRequest(
                query=query,
                include_search=include_search,
                max_results=5
            )
            
            result = await self.web_manager.enhance_with_web(request)
            
            return {
                'success': result.success,
                'documents': result.documents,
                'processing_time': result.processing_time,
                'error': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to enhance with web: {e}")
            return {'success': False, 'error': str(e)}
    
    # System Methods
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            # CPU info
            cpu_info = {
                'cores': psutil.cpu_count(),
                'usage': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            }
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent
            }
            
            # Disk info
            disk = psutil.disk_usage('/')
            disk_info = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percentage': (disk.used / disk.total) * 100
            }
            
            # GPU info
            gpu_info = []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'load': gpu.load * 100,
                        'temperature': gpu.temperature
                    })
            except:
                gpu_info = []
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'gpu': gpu_info,
                'python_version': os.sys.version,
                'platform': os.name
            }
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = {}
            
            # Basic system metrics
            metrics['cpu_usage'] = psutil.cpu_percent()
            metrics['memory_usage'] = psutil.virtual_memory().percent
            metrics['disk_usage'] = psutil.disk_usage('/').percent
            
            # Application metrics
            if self.vllm_manager:
                vllm_stats = await self.vllm_manager.get_statistics()
                metrics.update(vllm_stats)
            
            if self.rag_pipeline:
                rag_stats = await self.rag_pipeline.get_statistics()
                if 'vector_store' in rag_stats:
                    metrics['rag_documents'] = rag_stats['vector_store'].get('total_documents', 0)
            
            if self.file_manager:
                file_stats = self.file_manager.get_statistics()
                metrics['uploaded_files'] = file_stats['total_files']
                metrics['processed_files'] = file_stats['processed_files']
            
            units = {
                'cpu_usage': '%',
                'memory_usage': '%',
                'disk_usage': '%',
                'rag_documents': 'count',
                'uploaded_files': 'count',
                'processed_files': 'count'
            }
            
            return {
                'success': True,
                'metrics': metrics,
                'units': units
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_logs(self, level: str = "INFO", lines: int = 50) -> Dict[str, Any]:
        """Get recent logs"""
        try:
            # This is a simplified implementation
            # In a real application, you'd read from log files
            
            log_content = f"Recent logs (level: {level}, lines: {lines})\n"
            log_content += "=" * 50 + "\n"
            log_content += "Log functionality not fully implemented in this demo.\n"
            log_content += "In a production system, this would read from actual log files.\n"
            
            return {
                'success': True,
                'logs': log_content
            }
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return {'success': False, 'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health = {
                'status': 'healthy',
                'timestamp': time.time(),
                'components': {}
            }
            
            # Check vLLM manager
            if self.vllm_manager:
                try:
                    vllm_health = await self.vllm_manager.health_check()
                    health['components']['vllm'] = vllm_health
                except Exception as e:
                    health['components']['vllm'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            # Check RAG pipeline
            if self.rag_pipeline:
                try:
                    rag_health = await self.rag_pipeline.health_check()
                    health['components']['rag'] = rag_health
                except Exception as e:
                    health['components']['rag'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            # Check web manager
            if self.web_manager:
                try:
                    web_health = await self.web_manager.health_check()
                    health['components']['web'] = web_health
                except Exception as e:
                    health['components']['web'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            # Check file manager
            if self.file_manager:
                try:
                    file_stats = self.file_manager.get_statistics()
                    health['components']['files'] = {
                        'status': 'healthy',
                        'total_files': file_stats['total_files']
                    }
                except Exception as e:
                    health['components']['files'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def save_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Save application settings"""
        try:
            # Update configuration - this would need to be implemented properly
            # For now, just return success
            logger.info("Settings save requested (not fully implemented)")
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return {'success': False, 'error': str(e)}
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

