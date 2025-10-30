"""
vLLM Manager for vLLM Gradio WebUI

Handles vLLM engine initialization, model loading, and inference with multi-GPU support.
Coordinates with GPU manager and model manager for optimal performance.
"""

import os
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Inference request structure"""
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    stop_sequences: List[str] = None
    stream: bool = True
    request_id: Optional[str] = None

@dataclass
class InferenceResponse:
    """Inference response structure"""
    text: str
    finish_reason: str
    usage: Dict[str, int]
    request_id: Optional[str] = None

class VLLMManager:
    """Manages vLLM engine and inference operations"""
    
    def __init__(self, vllm_config, gpu_manager, model_manager=None):
        self.config = vllm_config
        self.gpu_manager = gpu_manager
        self.model_manager = model_manager
        
        # vLLM engine
        self.engine = None
        self.current_model = None
        self.is_loading = False
        self.is_ready = False
        
        # Inference state
        self.active_requests = {}
        self.request_counter = 0
        self._lock = threading.Lock()
        
        logger.info("vLLM Manager initialized")
    
    async def initialize(self, model_path: Optional[str] = None):
        """Initialize vLLM engine with specified model"""
        try:
            if self.is_loading:
                logger.warning("Model is already loading")
                return False
            
            self.is_loading = True
            self.is_ready = False
            
            # Determine model to load
           async def initialize(self, model_path: str = None, user_config: Dict[str, Any] = None) -> bool:
        """Initialize vLLM engine with a model"""
        if self.is_loading:
            logger.warning("Model is already being loaded")
            return False
        
        self.is_loading = True
        
        try:
            # Get model path
            if not model_path:
                model_path = self._get_default_model()
                if not model_path:
                    logger.error("No model path provided and no default model found")
                    return False
            
            logger.info(f"Initializing vLLM engine with model: {model_path}")
            
            # Prepare engine arguments
            engine_args = self._prepare_engine_args(model_path, user_config)
            
            # Import vLLM classes
            from vllm import AsyncLLMEngine, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            
            self.AsyncLLMEngine = AsyncLLMEngine
            self.SamplingParams = SamplingParams
            
            # Create engine args
            args = AsyncEngineArgs(**engine_args)
            
            # Initialize engine
            if hasattr(AsyncLLMEngine, 'from_engine_args'):
                self.engine = self.AsyncLLMEngine.from_engine_args(args)
            else:
                # Use sync engine for non-streaming
                self.engine = self.LLM(**engine_args)
            
            self.current_model = model_path
            self.is_ready = True
            
            logger.info(f"vLLM engine initialized successfully with {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            return False
        finally:
            self.is_loading = Falseg = False
    
    def _get_default_model(self) -> Optional[str]:
        """Get default model path"""
        if self.config.default_model:
            return self.config.default_model
        
        # Try to find a model in the model directories
        if self.model_manager:
            available_models = self.model_manager.get_available_models()
            if available_models:
                return available_models[0].path
        
        # Fallback to scanning directories
        for model_dir in self.config.model_dirs:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                        return item_path
        
        return None
    
    def _prepare_engine_args(self, model_path: str, user_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare arguments for vLLM engine initialization"""
        user_config = user_config or {}
        
        args = {
            'model': model_path,
            'trust_remote_code': True,
            'max_num_seqs': self.config.max_num_seqs,
            'block_size': self.config.block_size,
            'swap_space': self.config.swap_space,
        }
        
        # Add GPU configuration
        gpu_args = self.gpu_manager.get_vllm_gpu_args()
        args.update(gpu_args)
        
        # User-configurable parameters with defaults
        if 'max_model_len' in user_config:
            args['max_model_len'] = user_config['max_model_len']
        elif hasattr(self.config, 'max_model_len') and self.config.max_model_len:
            args['max_model_len'] = self.config.max_model_len
        else:
            # Set a reasonable default to avoid KV cache issues
            args['max_model_len'] = 32768  # 32K context length
        
        # KV cache configuration
        if 'kv_cache_dtype' in user_config:
            args['kv_cache_dtype'] = user_config['kv_cache_dtype']
        
        if 'cpu_offload_gb' in user_config:
            args['cpu_offload_gb'] = user_config['cpu_offload_gb']
        elif self.config.cpu_offload_gb > 0:
            args['cpu_offload_gb'] = self.config.cpu_offload_gb
        
        # Memory utilization
        if 'gpu_memory_utilization' in user_config:
            args['gpu_memory_utilization'] = user_config['gpu_memory_utilization']
        
        # Quantization settings
        if 'quantization' in user_config:
            args['quantization'] = user_config['quantization']
        
        # Performance settings
        if 'enforce_eager' in user_config:
            args['enforce_eager'] = user_config['enforce_eager']
        elif hasattr(self.config, 'enforce_eager') and self.config.enforce_eager:
            args['enforce_eager'] = True
        
        # Optional parameters
        if self.config.max_num_batched_tokens:
            args['max_num_batched_tokens'] = self.config.max_num_batched_tokens
        
        logger.info(f"vLLM engine args: {args}")
        return args
    
    async def load_model(self, model_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load a different model"""
        try:
            if self.current_model == model_path and self.is_ready:
                logger.info(f"Model {model_path} is already loaded")
                return {'success': True, 'message': f'Model {model_path} is already loaded'}
            
            # Cleanup current engine
            await self.cleanup()
            
            # Initialize with new model
            success = await self.initialize(model_path)
            
            if success:
                return {'success': True, 'message': f'Model {model_path} loaded successfully'}
            else:
                return {'success': False, 'error': f'Failed to load model {model_path}'}
                
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate(self, request: InferenceRequest) -> Union[InferenceResponse, AsyncGenerator[str, None]]:
        """Generate text using the loaded model"""
        if not self.is_ready or not self.engine:
            raise RuntimeError("vLLM engine is not ready")
        
        try:
            # Prepare sampling parameters
            sampling_params = self.SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences or [],
            )
            
            if request.stream:
                return self._generate_stream(request, sampling_params)
            else:
                return await self._generate_sync(request, sampling_params)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    async def _generate_sync(self, request: InferenceRequest, sampling_params) -> InferenceResponse:
        """Generate text synchronously"""
        try:
            if hasattr(self.engine, 'generate'):
                # Sync engine
                outputs = self.engine.generate([request.prompt], sampling_params)
                output = outputs[0]
                generated_text = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason
                
                # Calculate usage
                usage = {
                    'prompt_tokens': len(output.prompt_token_ids),
                    'completion_tokens': len(output.outputs[0].token_ids),
                    'total_tokens': len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                }
            else:
                # Async engine
                request_id = f"req_{self.request_counter}"
                self.request_counter += 1
                
                # Add request to engine
                await self.engine.add_request(
                    request_id=request_id,
                    prompt=request.prompt,
                    sampling_params=sampling_params
                )
                
                # Wait for completion
                generated_text = ""
                finish_reason = "stop"
                usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                
                async for request_output in self.engine.generate(request.prompt, sampling_params, request_id):
                    if request_output.outputs:
                        generated_text = request_output.outputs[0].text
                        finish_reason = request_output.outputs[0].finish_reason
                        usage = {
                            'prompt_tokens': len(request_output.prompt_token_ids),
                            'completion_tokens': len(request_output.outputs[0].token_ids),
                            'total_tokens': len(request_output.prompt_token_ids) + len(request_output.outputs[0].token_ids)
                        }
            
            return InferenceResponse(
                text=generated_text,
                finish_reason=finish_reason,
                usage=usage,
                request_id=request.request_id
            )
            
        except Exception as e:
            logger.error(f"Sync generation failed: {e}")
            raise
    
    async def _generate_stream(self, request: InferenceRequest, sampling_params) -> AsyncGenerator[str, None]:
        """Generate text with streaming"""
        try:
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1
            
            if hasattr(self.engine, 'generate'):
                # For sync engine, we need to simulate streaming
                outputs = self.engine.generate([request.prompt], sampling_params)
                output = outputs[0]
                generated_text = output.outputs[0].text
                
                # Simulate streaming by yielding chunks
                words = generated_text.split()
                current_text = ""
                for word in words:
                    current_text += word + " "
                    yield current_text.strip()
                    await asyncio.sleep(0.01)  # Small delay to simulate streaming
            else:
                # Async engine with real streaming
                async for request_output in self.engine.generate(request.prompt, sampling_params, request_id):
                    if request_output.outputs:
                        yield request_output.outputs[0].text
                        
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.current_model:
            return {'status': 'no_model_loaded'}
        
        info = {
            'model_path': self.current_model,
            'is_ready': self.is_ready,
            'is_loading': self.is_loading,
            'engine_type': 'async' if hasattr(self.engine, 'generate') else 'sync'
        }
        
        # Add model-specific information if available
        if self.model_manager:
            model_info = self.model_manager.get_model_by_name(os.path.basename(self.current_model))
            if model_info:
                info.update({
                    'model_name': model_info.name,
                    'parameters': model_info.parameters,
                    'architecture': model_info.architecture,
                    'size_gb': model_info.size_gb,
                    'context_length': model_info.context_length,
                    'quantization': model_info.quantization
                })
        
        return info
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        stats = {
            'active_requests': len(self.active_requests),
            'total_requests': self.request_counter,
            'gpu_info': self.gpu_manager.get_gpu_info()
        }
        
        # Add engine-specific stats if available
        if self.engine and hasattr(self.engine, 'get_stats'):
            try:
                engine_stats = self.engine.get_stats()
                stats.update(engine_stats)
            except:
                pass
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the engine"""
        health = {
            'status': 'healthy' if self.is_ready else 'unhealthy',
            'model_loaded': self.current_model is not None,
            'engine_ready': self.is_ready,
            'gpu_available': len(self.gpu_manager.gpus) > 0
        }
        
        # Test inference if engine is ready
        if self.is_ready:
            try:
                test_request = InferenceRequest(
                    prompt="Hello",
                    max_tokens=5,
                    stream=False
                )
                response = await self.generate(test_request)
                health['test_inference'] = 'passed'
                health['test_response_length'] = len(response.text)
            except Exception as e:
                health['test_inference'] = 'failed'
                health['test_error'] = str(e)
        
        return health
    
    async def cleanup(self):
        """Cleanup vLLM engine and resources"""
        try:
            if self.engine:
                # Cancel active requests
                self.active_requests.clear()
                
                # Cleanup engine
                if hasattr(self.engine, 'shutdown'):
                    await self.engine.shutdown()
                
                self.engine = None
                self.current_model = None
                self.is_ready = False
                
                # Clear GPU cache
                self.gpu_manager.cleanup()
                
                logger.info("vLLM engine cleaned up successfully")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        if not self.model_manager:
            return []
        
        models = []
        for model_info in self.model_manager.get_available_models():
            models.append({
                'name': model_info.name,
                'path': model_info.path,
                'parameters': model_info.parameters,
                'architecture': model_info.architecture,
                'size_gb': model_info.size_gb,
                'tags': model_info.tags,
                'is_current': model_info.path == self.current_model
            })
        
        return models
    
    async def scan_models_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Scan a directory for models"""
        try:
            models = []
            
            if not os.path.exists(directory_path):
                logger.warning(f"Directory does not exist: {directory_path}")
                return models
            
            logger.info(f"Scanning directory for models: {directory_path}")
            
            # First, check if the provided path itself is a model directory
            config_file = os.path.join(directory_path, 'config.json')
            tokenizer_file = os.path.join(directory_path, 'tokenizer.json')
            
            if os.path.exists(config_file) or os.path.exists(tokenizer_file):
                # The provided path itself is a model directory
                model_name = os.path.basename(directory_path)
                model_info = {
                    'name': model_name,
                    'path': directory_path,
                    'type': 'local',
                    'size_gb': self._estimate_model_size(directory_path),
                    'has_config': os.path.exists(config_file),
                    'has_tokenizer': os.path.exists(tokenizer_file),
                    'is_current': directory_path == self.current_model
                }
                
                # Try to read config for more details
                if os.path.exists(config_file):
                    try:
                        import json
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            model_info.update({
                                'architecture': config.get('architectures', ['unknown'])[0] if config.get('architectures') else 'unknown',
                                'vocab_size': config.get('vocab_size', 'unknown'),
                                'hidden_size': config.get('hidden_size', 'unknown'),
                                'num_layers': config.get('num_hidden_layers', 'unknown')
                            })
                    except Exception as e:
                        logger.warning(f"Could not read config for {model_name}: {e}")
                
                models.append(model_info)
                logger.info(f"Found model: {model_name} at {directory_path}")
            else:
                # The provided path is a parent directory, scan for subdirectories
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    
                    if os.path.isdir(item_path):
                        # Check if this looks like a model directory
                        config_file = os.path.join(item_path, 'config.json')
                        tokenizer_file = os.path.join(item_path, 'tokenizer.json')
                        
                        if os.path.exists(config_file) or os.path.exists(tokenizer_file):
                            # Try to get model info
                            model_info = {
                                'name': item,
                                'path': item_path,
                                'type': 'local',
                                'size_gb': self._estimate_model_size(item_path),
                                'has_config': os.path.exists(config_file),
                                'has_tokenizer': os.path.exists(tokenizer_file),
                                'is_current': item_path == self.current_model
                            }
                            
                            # Try to read config for more details
                            if os.path.exists(config_file):
                                try:
                                    import json
                                    with open(config_file, 'r') as f:
                                        config = json.load(f)
                                        model_info.update({
                                            'architecture': config.get('architectures', ['unknown'])[0] if config.get('architectures') else 'unknown',
                                            'vocab_size': config.get('vocab_size', 'unknown'),
                                            'hidden_size': config.get('hidden_size', 'unknown'),
                                            'num_layers': config.get('num_hidden_layers', 'unknown')
                                        })
                                except Exception as e:
                                    logger.warning(f"Could not read config for {item}: {e}")
                            
                            models.append(model_info)
                            logger.info(f"Found model: {item} at {item_path}")
            
            logger.info(f"Found {len(models)} models in {directory_path}")
            return models
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")
            return []
    
    def _estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            
            return round(total_size / (1024**3), 2)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not estimate size for {model_path}: {e}")
            return 0.0
    
    async def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models from all configured directories"""
        all_models = []
        
        for model_dir in self.config.model_dirs:
            if os.path.exists(model_dir):
                models = await self.scan_models_directory(model_dir)
                all_models.extend(models)
        
        return all_models
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload the current model"""
        try:
            if not self.current_model:
                return {'success': True, 'message': 'No model is currently loaded'}
            
            model_name = self.current_model
            await self.cleanup()
            
            return {'success': True, 'message': f'Model {model_name} unloaded successfully'}
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return {'success': False, 'error': str(e)}
    
    def estimate_memory_usage(self, model_path: str) -> Dict[str, Any]:
        """Estimate memory usage for a model"""
        if not self.model_manager:
            return {'error': 'Model manager not available'}
        
        model_info = self.model_manager.get_model_by_name(os.path.basename(model_path))
        if not model_info or not model_info.size_gb:
            return {'error': 'Model information not available'}
        
        return self.gpu_manager.estimate_model_memory_requirements(model_info.size_gb)
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

