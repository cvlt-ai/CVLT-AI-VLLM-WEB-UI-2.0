"""
vLLM Manager for vLLM Gradio WebUI

Handles vLLM engine initialization, model loading, and inference with multi-GPU support.
Coordinates with GPU manager and model manager for optimal performance.
"""

import os
import logging
import asyncio
import threading
import signal
import gc
import torch
import time
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
        
        # Model registry for name to path mapping
        self.model_registry = {}  # {model_name: model_path}
        
        # Request tracking
        self.active_requests = {}
        self.request_counter = 0
        
        # vLLM classes (imported when needed)
        self.AsyncLLMEngine = None
        self.SamplingParams = None
        self.LLM = None
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
        
        logger.info("vLLM Manager initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        # Don't set up signal handlers here as they interfere with the async engine
        # The engine should remain running between requests
        pass
    
    async def graceful_shutdown(self):
        """Gracefully shutdown the vLLM manager"""
        logger.info("Starting graceful shutdown...")
        self._shutdown_event.set()
        
        # Cancel active requests
        for request_id in list(self.active_requests.keys()):
            try:
                self.active_requests.pop(request_id, None)
            except:
                pass
        
        # Cleanup engine
        await self.cleanup()
        logger.info("Graceful shutdown completed")
    
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
            from vllm import SamplingParams, LLM
            
            self.SamplingParams = SamplingParams
            self.LLM = LLM
            
            # IMPORTANT: For tensor parallelism, we use the synchronous LLM engine
            # The AsyncLLMEngine has issues with auto-shutdown that can't be easily fixed
            # We'll wrap the sync engine with our own async handling
            
            # Remove async-specific args
            sync_args = {k: v for k, v in engine_args.items() 
                        if k not in ['worker_use_ray', 'engine_use_ray', 'disable_log_requests']}
            
            # Initialize synchronous engine
            logger.info("Initializing synchronous vLLM engine for stable tensor parallelism")
            self.engine = self.LLM(**sync_args)
            
            # The synchronous engine doesn't shut down between requests
            logger.info("Using synchronous engine with async wrapper for stability")
            
            self.current_model = model_path
            self.is_ready = True
            
            logger.info(f"vLLM engine initialized successfully with {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            return False
        finally:
            self.is_loading = False
    
    def _get_default_model(self) -> Optional[str]:
        """Get default model path"""
        if hasattr(self.config, 'default_model') and self.config.default_model:
            return self.config.default_model
        
        # Try to find a model in the model directories
        if self.model_manager:
            available_models = self.model_manager.get_available_models()
            if available_models:
                return available_models[0].path
        
        # Fallback to scanning directories
        model_dirs = getattr(self.config, 'model_dirs', ['./models'])
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
                        return item_path
        
        return None
    
    def _has_existing_quantization(self, engine_args, model_config):
        """Check if engine_args or model_config has quantization settings."""
        return any(key in engine_args for key in ["quantization", "bits"]) or \
               model_config.get("quantization", None) is not None

    def _prepare_engine_args(self, model_path: str, user_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare arguments for vLLM engine initialization"""
        user_config = user_config or {}
        
        args = {
            'model': model_path,
            'trust_remote_code': True,
            'max_num_seqs': getattr(self.config, 'max_num_seqs', 256),
            'block_size': getattr(self.config, 'block_size', 16),
            'swap_space': getattr(self.config, 'swap_space', 4),
        }
        
        # Add GPU configuration with user-specified parallel sizes if provided
        user_tensor_parallel = user_config.get('tensor_parallel_size') if user_config else None
        user_pipeline_parallel = user_config.get('pipeline_parallel_size') if user_config else None
        
        gpu_args = self.gpu_manager.get_vllm_gpu_args(
            user_tensor_parallel=user_tensor_parallel,
            user_pipeline_parallel=user_pipeline_parallel
        )
        args.update(gpu_args)
        
        # IMPORTANT: For tensor parallelism, we need to handle mixed GPU setups carefully
        if gpu_args.get('tensor_parallel_size', 1) > 1:
            if 'distributed_executor_backend' not in gpu_args:
                # For mixed GPU setups, use 'mp' backend with special configuration
                # The 'ray' backend has issues with auto-shutdown in some vLLM versions
                args['distributed_executor_backend'] = 'mp'
                logger.info("Using multiprocessing backend for tensor parallelism")
                
                # CRITICAL: Disable custom all-reduce for mixed GPU setups
                # Mixed GPUs (RTX 6000 Ada + L40) can't share memory buffers properly
                args['disable_custom_all_reduce'] = True
                logger.info("Disabled custom all-reduce for mixed GPU compatibility")
                
                # Enable worker persistence to prevent shutdown
                args['worker_use_ray'] = False  # Don't use ray workers
                args['disable_log_requests'] = False  # Keep request logging
                
                # Set a high idle timeout to prevent worker shutdown
                if 'engine_use_ray' not in args:
                    args['engine_use_ray'] = False  # Don't use ray engine
        
        # User-configurable parameters with defaults
        if 'max_model_len' in user_config:
            args['max_model_len'] = user_config['max_model_len']
        else:
            # Set a reasonable default to avoid KV cache issues
            args['max_model_len'] = user_config.get('context_length', 32768)
        
        # KV cache configuration
        if 'kv_cache_dtype' in user_config:
            args['kv_cache_dtype'] = user_config['kv_cache_dtype']
        
        # CPU offloading for KV cache
        if 'cpu_offload_gb' in user_config:
            args['cpu_offload_gb'] = user_config['cpu_offload_gb']
        elif hasattr(self.config, 'cpu_offload_gb') and self.config.cpu_offload_gb > 0:
            args['cpu_offload_gb'] = self.config.cpu_offload_gb
        
        # Note: vLLM doesn't support direct KV cache GPU placement via parameters
        # For mixed GPU setups, use CUDA_VISIBLE_DEVICES environment variable
        if 'kv_cache_gpu' in user_config and user_config['kv_cache_gpu'] is not None:
            logger.info(f"Note: Direct KV cache GPU placement not supported by vLLM.")
            # Don't force tensor_parallel_size=1 as it reduces performance
        
        # Memory utilization
        if 'gpu_memory_utilization' in user_config:
            args['gpu_memory_utilization'] = user_config['gpu_memory_utilization']
        
        # Quantization settings - only add default quantization if none exists
        if not self._has_existing_quantization(args, user_config):
            if 'quantization' in user_config:
                args['quantization'] = user_config['quantization']
            elif 'bits' in user_config:
                args['bits'] = user_config['bits']
        
        # Performance settings
        if 'enforce_eager' in user_config:
            args['enforce_eager'] = user_config['enforce_eager']
        elif hasattr(self.config, 'enforce_eager') and self.config.enforce_eager:
            args['enforce_eager'] = True
        
        # Optional parameters
        if hasattr(self.config, 'max_num_batched_tokens') and self.config.max_num_batched_tokens:
            args['max_num_batched_tokens'] = self.config.max_num_batched_tokens
        
        logger.info(f"vLLM engine args: {args}")
        return args
    
    def _resolve_model_path(self, model_identifier: str) -> str:
        """Resolve model identifier to full path"""
        # If it's already a full path, return as is
        if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
            return model_identifier
        
        # Check if it's in the model registry
        if model_identifier in self.model_registry:
            return self.model_registry[model_identifier]
        
        # Try to find it in model directories
        model_dirs = getattr(self.config, 'model_dirs', ['./models'])
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # Check if it's a direct subdirectory
                potential_path = os.path.join(model_dir, model_identifier)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    config_file = os.path.join(potential_path, 'config.json')
                    if os.path.exists(config_file):
                        return potential_path
        
        # If not found locally, assume it's a Hugging Face model ID
        logger.info(f"Model '{model_identifier}' not found locally, treating as Hugging Face model ID")
        return model_identifier
    
    async def load_model(self, model_identifier: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load a model by identifier (name or path)"""
        try:
            # Resolve model identifier to full path
            model_path = self._resolve_model_path(model_identifier)
            
            if self.current_model == model_path and self.is_ready:
                logger.info(f"Model {model_path} is already loaded")
                return {'success': True, 'message': f'Model {os.path.basename(model_path)} is already loaded'}
            
            # Cleanup current engine
            await self.cleanup()
            
            # Initialize with new model
            success = await self.initialize(model_path, config)
            
            if success:
                # Update model registry
                model_name = os.path.basename(model_path)
                self.model_registry[model_name] = model_path
                
                return {'success': True, 'message': f'Model {model_name} loaded successfully'}
            else:
                return {'success': False, 'error': f'Failed to load model {model_identifier}'}
                
        except Exception as e:
            logger.error(f"Error loading model {model_identifier}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload the current model and free GPU memory"""
        try:
            if not self.current_model:
                return {'success': True, 'message': 'No model is currently loaded'}
            
            model_name = os.path.basename(self.current_model)
            await self.cleanup()
            
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Model {model_name} unloaded and GPU memory freed")
            return {'success': True, 'message': f'Model {model_name} unloaded successfully'}
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return {'success': False, 'error': str(e)}
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover available models from configured directories"""
        try:
            models = []
            model_dirs = getattr(self.config, 'model_dirs', ['./models'])
            
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    discovered = self.scan_models_directory(model_dir)
                    models.extend(discovered)
            
            # Add currently loaded model if not in the list
            if self.current_model:
                current_model_name = os.path.basename(self.current_model)
                if not any(model['name'] == current_model_name for model in models):
                    models.insert(0, {
                        'name': current_model_name,
                        'path': self.current_model,
                        'type': 'local',
                        'is_current': True,
                        'size_gb': self._estimate_model_size(self.current_model)
                    })
                else:
                    # Mark the current model
                    for model in models:
                        if model['path'] == self.current_model:
                            model['is_current'] = True
            
            # Update model registry
            for model in models:
                self.model_registry[model['name']] = model['path']
            
            logger.info(f"Discovered {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            return []
    
    def scan_models_directory(self, directory_path: str) -> List[Dict[str, Any]]:
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
                                'num_layers': config.get('num_hidden_layers', 'unknown'),
                                'max_position_embeddings': config.get('max_position_embeddings', 'unknown')
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
                                            'num_layers': config.get('num_hidden_layers', 'unknown'),
                                            'max_position_embeddings': config.get('max_position_embeddings', 'unknown')
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
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        try:
            return self.gpu_manager.get_gpu_info()
        except Exception as e:
            logger.error(f"Error getting GPU status: {e}")
            return {'error': str(e)}
    
    def get_p2p_status(self) -> Dict[str, Any]:
        """Get P2P communication status for tensor parallelism"""
        try:
            if not self.gpu_manager:
                return {'error': 'GPU manager not available'}
            
            p2p_info = self.gpu_manager.check_p2p_connectivity()
            
            # Add current configuration info
            p2p_info['current_config'] = {
                'tensor_parallel_size': self.gpu_manager.tensor_parallel_size,
                'pipeline_parallel_size': self.gpu_manager.pipeline_parallel_size,
                'model_loaded': self.current_model is not None
            }
            
            # Add performance implications
            if p2p_info.get('p2p_enabled'):
                p2p_info['performance_note'] = "P2P enabled - optimal tensor parallel performance"
            else:
                p2p_info['performance_note'] = "P2P disabled - may experience reduced tensor parallel performance"
            
            return p2p_info
            
        except Exception as e:
            logger.error(f"Error getting P2P status: {e}")
            return {'error': str(e)}
    
    def optimize_tensor_parallelism(self, model_size_gb: float = None) -> Dict[str, Any]:
        """Get optimized tensor parallelism configuration based on P2P connectivity"""
        try:
            if not self.gpu_manager:
                return {'error': 'GPU manager not available'}
            
            # Use current model size if not provided
            if model_size_gb is None and self.current_model:
                model_size_gb = self._estimate_model_size(self.current_model)
            
            if model_size_gb is None:
                model_size_gb = 10.0  # Default estimate
            
            optimization = self.gpu_manager.optimize_gpu_placement(model_size_gb)
            
            # Add recommendations for vLLM configuration
            if optimization.get('p2p_optimized'):
                optimization['vllm_recommendations'] = {
                    'tensor_parallel_size': optimization['recommended_tensor_parallel_size'],
                    'disable_custom_all_reduce': False,
                    'distributed_executor_backend': 'ray',
                    'note': 'P2P-optimized configuration for best performance'
                }
            else:
                optimization['vllm_recommendations'] = {
                    'tensor_parallel_size': optimization['recommended_tensor_parallel_size'],
                    'disable_custom_all_reduce': True,
                    'distributed_executor_backend': 'mp',
                    'note': 'Fallback configuration for systems without P2P support'
                }
            
            # Add environment setup recommendations
            if optimization.get('cuda_visible_devices'):
                optimization['setup_command'] = f"export CUDA_VISIBLE_DEVICES={optimization['cuda_visible_devices']}"
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing tensor parallelism: {e}")
            return {'error': str(e)}
    
    async def benchmark_p2p_performance(self) -> Dict[str, Any]:
        """Benchmark P2P communication performance for tensor parallelism"""
        try:
            if not self.is_ready or not self.engine:
                return {'error': 'Engine not ready. Please load a model first.'}
            
            results = {
                'p2p_enabled': self.gpu_manager.p2p_enabled if self.gpu_manager else False,
                'tensor_parallel_size': self.gpu_manager.tensor_parallel_size if self.gpu_manager else 1,
                'benchmarks': []
            }
            
            # Test different prompt lengths
            test_prompts = [
                ("short", "Hello, world!", 50),
                ("medium", "The quick brown fox jumps over the lazy dog. " * 10, 100),
                ("long", "Once upon a time in a land far away, " * 50, 200)
            ]
            
            for prompt_type, prompt, max_tokens in test_prompts:
                try:
                    request = InferenceRequest(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        stream=False
                    )
                    
                    # Warm-up run
                    await self.generate(request)
                    
                    # Benchmark runs
                    times = []
                    for _ in range(3):
                        start_time = time.time()
                        response = await self.generate(request)
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                    
                    avg_time = sum(times) / len(times)
                    tokens_per_second = max_tokens / avg_time
                    
                    results['benchmarks'].append({
                        'prompt_type': prompt_type,
                        'prompt_length': len(prompt.split()),
                        'max_tokens': max_tokens,
                        'avg_time': round(avg_time, 3),
                        'tokens_per_second': round(tokens_per_second, 2)
                    })
                    
                except Exception as e:
                    results['benchmarks'].append({
                        'prompt_type': prompt_type,
                        'error': str(e)
                    })
            
            # Add P2P impact analysis
            if results['p2p_enabled']:
                results['p2p_impact'] = "P2P communication enabled - achieving optimal tensor parallel performance"
            else:
                results['p2p_impact'] = "P2P communication disabled - performance may be 20-40% lower than optimal"
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking P2P performance: {e}")
            return {'error': str(e)}
    
    async def generate(self, request: InferenceRequest) -> Union[InferenceResponse, AsyncGenerator[str, None]]:
        """Generate text using the loaded model"""
        if not self.is_ready or not self.engine:
            raise RuntimeError("vLLM engine is not ready")
        
        try:
            # Prepare sampling parameters with user-configurable values
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
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1
            self.active_requests[request_id] = request
            
            # Use synchronous LLM.generate method
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                """Synchronous generation function"""
                outputs = self.engine.generate([request.prompt], sampling_params)
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    return output.outputs[0].text, output.outputs[0].finish_reason or "stop"
                return "", "stop"
            
            # Run synchronous generation in thread pool
            generated_text, finish_reason = await loop.run_in_executor(None, sync_generate)
            
            if generated_text:
                # Calculate usage
                usage = {
                    'prompt_tokens': len(request.prompt.split()),  # Rough estimate
                    'completion_tokens': len(generated_text.split()),
                    'total_tokens': len(request.prompt.split()) + len(generated_text.split())
                }
                
                return InferenceResponse(
                    text=generated_text,
                    finish_reason=finish_reason,
                    usage=usage,
                    request_id=request_id
                )
            else:
                raise RuntimeError("No output generated")
                
        except Exception as e:
            logger.error(f"Sync generation failed: {e}")
            raise
        finally:
            self.active_requests.pop(request_id, None)
            # The synchronous engine stays alive between requests
    
    async def _generate_stream(self, request: InferenceRequest, sampling_params) -> AsyncGenerator[str, None]:
        """Generate text with streaming"""
        try:
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1
            self.active_requests[request_id] = request
            
            # Streaming not supported with synchronous engine
            # Fall back to non-streaming generation
            logger.warning("Streaming not supported with synchronous engine, using non-streaming generation")
            
            loop = asyncio.get_event_loop()
            
            def sync_generate():
                """Synchronous generation function"""
                outputs = self.engine.generate([request.prompt], sampling_params)
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    return output.outputs[0].text
                return ""
            
            # Run synchronous generation in thread pool
            generated_text = await loop.run_in_executor(None, sync_generate)
            
            # Yield the complete text at once
            if generated_text:
                yield generated_text
                        
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise
        finally:
            self.active_requests.pop(request_id, None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.current_model:
            return {'status': 'no_model_loaded'}
        
        info = {
            'model_path': self.current_model,
            'model_name': os.path.basename(self.current_model),
            'is_ready': self.is_ready,
            'is_loading': self.is_loading,
            'engine_type': 'async'
        }
        
        # Add model-specific information if available
        if self.model_manager:
            model_info = self.model_manager.get_model_by_name(os.path.basename(self.current_model))
            if model_info:
                info.update({
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
            'engine_ready': self.is_ready,
            'model_loaded': self.current_model is not None,
            'active_requests': len(self.active_requests),
            'gpu_available': torch.cuda.is_available() if torch else False
        }
        
        if self.current_model:
            health['current_model'] = os.path.basename(self.current_model)
        
        # Test generation if engine is ready
        if self.is_ready and self.engine:
            try:
                test_request = InferenceRequest(
                    prompt="Hello",
                    max_tokens=5,
                    temperature=0.1,
                    stream=False
                )
                
                start_time = time.time()
                response = await self.generate(test_request)
                health['generation_test'] = {
                    'success': True,
                    'response_time': time.time() - start_time,
                    'output_length': len(response.text) if hasattr(response, 'text') else 0
                }
            except Exception as e:
                health['generation_test'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return health
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vLLM engine statistics"""
        try:
            stats = {
                'model_loaded': self.current_model is not None,
                'model_name': self.current_model,
                'engine_ready': self.is_ready,
                'gpu_count': len(self.gpu_manager.get_available_gpus()) if self.gpu_manager else 0,
                'memory_usage': {}
            }
            
            # Add GPU memory usage if available
            if self.gpu_manager:
                gpu_status = self.get_gpu_status()
                for gpu in gpu_status:
                    stats['memory_usage'][f"gpu_{gpu['id']}"] = {
                        'used': gpu['memory_used'],
                        'total': gpu['memory_total'],
                        'utilization': gpu['utilization']
                    }
            
            # Add engine-specific stats if available
            if self.engine and hasattr(self.engine, 'get_stats'):
                try:
                    engine_stats = self.engine.get_stats()
                    stats['engine_stats'] = engine_stats
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'error': str(e),
                'model_loaded': False,
                'engine_ready': False
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.engine:
                # Stop the engine
                if hasattr(self.engine, 'stop'):
                    await self.engine.stop()
                elif hasattr(self.engine, 'shutdown'):
                    await self.engine.shutdown()
                
                self.engine = None
            
            self.current_model = None
            self.is_ready = False
            self.is_loading = False
            
            # Clear active requests
            self.active_requests.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("vLLM Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
