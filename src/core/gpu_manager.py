"""
GPU Manager for vLLM Gradio WebUI

Handles GPU detection, configuration, and resource management for multi-GPU setups.
Supports tensor parallelism and pipeline parallelism configurations with P2P communication.
"""

import os
import logging
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU information structure"""
    index: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int   # MB
    utilization: float  # Percentage
    temperature: Optional[int] = None  # Celsius
    power_draw: Optional[float] = None  # Watts
    uuid: Optional[str] = None

class GPUManager:
    """Manages GPU resources and configuration for vLLM"""
    
    def __init__(self, gpu_config):
        self.config = gpu_config
        self.gpus = []
        self.tensor_parallel_size = gpu_config.tensor_parallel_size
        self.pipeline_parallel_size = gpu_config.pipeline_parallel_size
        
        # P2P communication support
        self.p2p_enabled = False
        self.p2p_matrix = {}  # Store P2P capability between GPU pairs
        
        # Initialize GPU detection
        self._detect_gpus()
        self._validate_configuration()
        
        # Initialize P2P communication if available
        if len(self.gpus) > 1:
            self._initialize_p2p_communication()
        
    def _detect_gpus(self):
        """Detect available GPUs and their properties"""
        try:
            # Try to detect NVIDIA GPUs first
            self._detect_nvidia_gpus()
            
            if not self.gpus:
                # Fallback to other GPU detection methods
                self._detect_other_gpus()
                
            logger.info(f"Detected {len(self.gpus)} GPU(s)")
            for gpu in self.gpus:
                logger.info(f"GPU {gpu.index}: {gpu.name} ({gpu.memory_total}MB)")
                
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.gpus = []
    
    def _detect_nvidia_gpus(self):
        """Detect NVIDIA GPUs using nvidia-ml-py or nvidia-smi"""
        try:
            # Try nvidia-ml-py first (more reliable)
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get basic info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Get power draw
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power_draw = None
                
                # Get UUID
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                except:
                    uuid = None
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=memory_info.total // (1024 * 1024),  # Convert to MB
                    memory_free=memory_info.free // (1024 * 1024),
                    memory_used=memory_info.used // (1024 * 1024),
                    utilization=utilization,
                    temperature=temperature,
                    power_draw=power_draw,
                    uuid=uuid
                )
                
                self.gpus.append(gpu_info)
                
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.warning("pynvml not available, falling back to nvidia-smi")
            self._detect_nvidia_smi()
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")
            self._detect_nvidia_smi()
    
    def _detect_nvidia_smi(self):
        """Detect NVIDIA GPUs using nvidia-smi command"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,uuid',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            self.gpus = []
            for line in lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        try:
                            gpu_info = GPUInfo(
                                index=int(parts[0]),
                                name=parts[1],
                                memory_total=int(parts[2]),
                                memory_free=int(parts[3]),
                                memory_used=int(parts[4]),
                                utilization=float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                                temperature=int(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else None,
                                power_draw=float(parts[7]) if len(parts) > 7 and parts[7] != '[Not Supported]' else None,
                                uuid=parts[8] if len(parts) > 8 else None
                            )
                            self.gpus.append(gpu_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse GPU info: {line}, error: {e}")
                            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"nvidia-smi not available: {e}")
    
    def _detect_other_gpus(self):
        """Detect other types of GPUs (AMD, Intel, etc.)"""
        try:
            # Try to detect using PyTorch
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                self.gpus = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory // (1024 * 1024)  # Convert to MB
                    
                    # Get current memory usage
                    torch.cuda.set_device(i)
                    memory_allocated = torch.cuda.memory_allocated() // (1024 * 1024)
                    memory_free = memory_total - memory_allocated
                    
                    gpu_info = GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_free=memory_free,
                        memory_used=memory_allocated,
                        utilization=0.0  # PyTorch doesn't provide utilization
                    )
                    
                    self.gpus.append(gpu_info)
                    
        except ImportError:
            logger.warning("PyTorch not available for GPU detection")
        except Exception as e:
            logger.warning(f"PyTorch GPU detection failed: {e}")
    
    def _validate_configuration(self):
        """Validate GPU configuration and adjust for compatibility"""
        gpu_count = len(self.gpus)
        
        if gpu_count == 0:
            logger.warning("No GPUs detected, running in CPU mode")
            self.tensor_parallel_size = 1
            self.pipeline_parallel_size = 1
            return
        
        # Check for mixed GPU types (warn but don't override)
        gpu_names = [gpu.name for gpu in self.gpus]
        unique_gpu_types = set(gpu_names)
        
        if len(unique_gpu_types) > 1:
            logger.warning(f"Mixed GPU types detected: {unique_gpu_types}")
            logger.warning("Mixed GPU types may have compatibility issues with tensor parallelism.")
            # Don't automatically override user settings for mixed GPUs
            # Let vLLM handle it and potentially fail with clear error
        
        # Auto-detect tensor parallel size ONLY if enabled and not manually set
        if self.config.auto_detect and self.tensor_parallel_size == 1:
            self.tensor_parallel_size = self._auto_detect_tensor_parallel_size(gpu_count)
            logger.info(f"Auto-detected tensor parallel size: {self.tensor_parallel_size}")
        
        # Validate tensor parallel size
        valid_tp_sizes = [1, 2, 4, 8, 16]
        if self.tensor_parallel_size not in valid_tp_sizes:
            logger.warning(f"Tensor parallel size {self.tensor_parallel_size} may not be optimal. Valid sizes: {valid_tp_sizes}")
            # Don't override, let user experiment
        
        # Warn if we exceed available GPUs but don't override
        total_gpus_needed = self.tensor_parallel_size * self.pipeline_parallel_size
        if total_gpus_needed > gpu_count:
            logger.warning(f"Requested {total_gpus_needed} GPUs but only {gpu_count} available")
            logger.warning("This configuration may fail. Consider reducing parallel sizes.")
            # Don't automatically adjust - let user control
        
        logger.info(f"Configuration: TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}")
        if len(unique_gpu_types) > 1:
            logger.info("Mixed GPU types detected - performance may vary")
    
    def _auto_detect_tensor_parallel_size(self, gpu_count: int) -> int:
        """Auto-detect optimal tensor parallel size based on available GPUs"""
        valid_sizes = [1, 2, 4, 8, 16]
        
        # Find the largest valid size that doesn't exceed GPU count
        for size in reversed(valid_sizes):
            if size <= gpu_count:
                return size
        
        return 1
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        return {
            'gpu_count': len(self.gpus),
            'gpus': [
                {
                    'index': gpu.index,
                    'name': gpu.name,
                    'memory_total': gpu.memory_total,
                    'memory_free': gpu.memory_free,
                    'memory_used': gpu.memory_used,
                    'utilization': gpu.utilization,
                    'temperature': gpu.temperature,
                    'power_draw': gpu.power_draw,
                    'uuid': gpu.uuid
                }
                for gpu in self.gpus
            ],
            'tensor_parallel_size': self.tensor_parallel_size,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'total_memory': sum(gpu.memory_total for gpu in self.gpus),
            'total_free_memory': sum(gpu.memory_free for gpu in self.gpus)
        }
    
    def get_vllm_gpu_args(self, user_tensor_parallel: Optional[int] = None, 
                          user_pipeline_parallel: Optional[int] = None) -> Dict[str, Any]:
        """Get GPU arguments for vLLM initialization with P2P optimization
        
        Args:
            user_tensor_parallel: User-specified tensor parallel size (overrides config)
            user_pipeline_parallel: User-specified pipeline parallel size (overrides config)
        """
        args = {}
        
        if len(self.gpus) > 0:
            # Use user-specified values if provided, otherwise use config values
            tp_size = user_tensor_parallel or self.tensor_parallel_size
            pp_size = user_pipeline_parallel or self.pipeline_parallel_size
            
            args['tensor_parallel_size'] = tp_size
            args['pipeline_parallel_size'] = pp_size
            args['gpu_memory_utilization'] = self.config.gpu_memory_utilization
            
            # Check for mixed GPU types
            gpu_names = [gpu.name for gpu in self.gpus]
            unique_gpu_types = set(gpu_names)
            
            # P2P-based optimization for custom all-reduce
            if tp_size > 1:
                # Check P2P connectivity for the tensor parallel group
                tp_gpus = list(range(tp_size))
                tp_p2p_connected = all(
                    self.p2p_matrix.get((i, j), False) 
                    for i in tp_gpus for j in tp_gpus if i != j
                )
                
                if not tp_p2p_connected or len(unique_gpu_types) > 1:
                    # Disable custom all-reduce if P2P is not available or mixed GPUs
                    args['disable_custom_all_reduce'] = True
                    logger.info("Disabled custom all-reduce (P2P not available or mixed GPUs)")
                else:
                    # Enable custom all-reduce for P2P-connected homogeneous GPUs
                    args['disable_custom_all_reduce'] = False
                    logger.info(f"Enabled custom all-reduce with P2P for {tp_size} GPUs")
                    
                    # IMPORTANT: Use 'mp' (multiprocessing) backend to prevent engine shutdown
                    # The 'ray' backend can cause the engine to shutdown after each request
                    args['distributed_executor_backend'] = 'mp'  # multiprocessing is more stable
                    
                    # Enable NCCL optimizations for P2P
                    import os
                    os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P
                    os.environ['NCCL_SHM_DISABLE'] = '0'  # Enable shared memory
                    
                    # Set optimal GPU placement if P2P is optimized
                    if self.p2p_enabled:
                        best_group = self._find_best_p2p_group(tp_size)
                        if best_group and best_group != list(range(tp_size)):
                            # Recommend using specific GPUs for best P2P performance
                            cuda_devices = ','.join(map(str, best_group))
                            logger.info(f"Recommended CUDA_VISIBLE_DEVICES={cuda_devices} for optimal P2P")
                            # Note: We don't set it here as vLLM should be started with proper env
            else:
                # Single GPU, no need for all-reduce
                args['disable_custom_all_reduce'] = True
            
            if self.config.max_model_len:
                args['max_model_len'] = self.config.max_model_len
            
            if self.config.enforce_eager:
                args['enforce_eager'] = True
            
            # Store P2P status separately (not passed to vLLM)
            # These are for internal tracking only
            self._last_p2p_status = {
                'p2p_enabled': self.p2p_enabled,
                'p2p_optimized': self.p2p_enabled and tp_size > 1
            }
        
        return args
    
    def monitor_gpu_usage(self) -> Dict[str, Any]:
        """Monitor current GPU usage"""
        try:
            # Refresh GPU information
            self._detect_gpus()
            return self.get_gpu_info()
        except Exception as e:
            logger.error(f"Failed to monitor GPU usage: {e}")
            return {'error': str(e)}
    
    def estimate_model_memory_requirements(self, model_size_gb: float) -> Dict[str, Any]:
        """Estimate memory requirements for a model"""
        # Rough estimation: model size + activation memory + overhead
        base_memory = model_size_gb * 1024  # Convert to MB
        activation_memory = base_memory * 0.2  # 20% for activations
        overhead = 1024  # 1GB overhead
        
        total_required = base_memory + activation_memory + overhead
        
        # Distribute across tensor parallel GPUs
        per_gpu_required = total_required / self.tensor_parallel_size
        
        return {
            'total_required_mb': total_required,
            'per_gpu_required_mb': per_gpu_required,
            'can_fit': all(gpu.memory_total >= per_gpu_required for gpu in self.gpus[:self.tensor_parallel_size]),
            'utilization_ratio': per_gpu_required / max(gpu.memory_total for gpu in self.gpus) if self.gpus else 1.0
        }
    
    def get_optimal_batch_size(self, sequence_length: int = 512) -> int:
        """Get optimal batch size based on available GPU memory"""
        if not self.gpus:
            return 1
        
        # Rough estimation based on available memory
        min_free_memory = min(gpu.memory_free for gpu in self.gpus[:self.tensor_parallel_size])
        
        # Estimate memory per sequence (very rough)
        memory_per_sequence = sequence_length * 4 * 2  # 4 bytes per token, 2x for safety
        
        optimal_batch_size = max(1, min_free_memory * 1024 * 1024 // memory_per_sequence // 2)
        
        return min(optimal_batch_size, 32)  # Cap at reasonable maximum
    
    def get_p2p_optimization_status(self) -> Dict[str, bool]:
        """Get P2P optimization status (for internal use, not passed to vLLM)"""
        if hasattr(self, '_last_p2p_status'):
            return self._last_p2p_status
        return {
            'p2p_enabled': self.p2p_enabled,
            'p2p_optimized': False
        }
    
    def _initialize_p2p_communication(self):
        """Initialize P2P (peer-to-peer) communication between GPUs for tensor parallelism"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, P2P communication disabled")
                return
            
            logger.info("Initializing P2P communication for tensor parallelism...")
            
            # Check P2P capability between all GPU pairs
            for i in range(len(self.gpus)):
                for j in range(len(self.gpus)):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        self.p2p_matrix[(i, j)] = can_access
                        
                        if can_access:
                            try:
                                # Enable P2P access
                                torch.cuda.set_device(i)
                                torch.cuda.set_peer_to_peer_access_enabled(True, j)
                                logger.info(f"P2P enabled: GPU {i} -> GPU {j}")
                            except RuntimeError as e:
                                if "already enabled" in str(e):
                                    logger.debug(f"P2P already enabled: GPU {i} -> GPU {j}")
                                else:
                                    logger.warning(f"Failed to enable P2P: GPU {i} -> GPU {j}: {e}")
                        else:
                            logger.warning(f"P2P not available: GPU {i} -> GPU {j}")
            
            # Check if P2P is enabled for tensor parallel group
            tp_gpus = list(range(self.tensor_parallel_size))
            p2p_available = all(
                self.p2p_matrix.get((i, j), False) 
                for i in tp_gpus for j in tp_gpus if i != j
            )
            
            if p2p_available:
                self.p2p_enabled = True
                logger.info(f"P2P communication enabled for tensor parallel group (GPUs 0-{self.tensor_parallel_size-1})")
            else:
                logger.warning("P2P communication not fully available for tensor parallel group")
                logger.warning("Performance may be reduced. Consider using GPUs with P2P support.")
            
        except ImportError:
            logger.warning("PyTorch not available, P2P communication disabled")
        except Exception as e:
            logger.error(f"Failed to initialize P2P communication: {e}")
    
    def check_p2p_connectivity(self) -> Dict[str, Any]:
        """Check P2P connectivity between GPUs"""
        connectivity = {
            'p2p_enabled': self.p2p_enabled,
            'p2p_matrix': {},
            'tensor_parallel_group_connected': False
        }
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                connectivity['error'] = 'CUDA not available'
                return connectivity
            
            # Build connectivity matrix
            for i in range(len(self.gpus)):
                for j in range(len(self.gpus)):
                    if i != j:
                        key = f"GPU{i}->GPU{j}"
                        connectivity['p2p_matrix'][key] = self.p2p_matrix.get((i, j), False)
            
            # Check tensor parallel group connectivity
            tp_gpus = list(range(self.tensor_parallel_size))
            connectivity['tensor_parallel_group_connected'] = all(
                self.p2p_matrix.get((i, j), False) 
                for i in tp_gpus for j in tp_gpus if i != j
            )
            
            # Add recommendations
            if not connectivity['tensor_parallel_group_connected']:
                connectivity['recommendation'] = (
                    "P2P not fully available for tensor parallel group. "
                    "Consider: 1) Using GPUs on same PCIe switch, "
                    "2) Enabling ACS override, or 3) Using NVLink-connected GPUs"
                )
            
        except Exception as e:
            connectivity['error'] = str(e)
        
        return connectivity
    
    def optimize_gpu_placement(self, model_size_gb: float) -> Dict[str, Any]:
        """Optimize GPU placement for tensor parallelism based on P2P connectivity"""
        optimization = {
            'recommended_tensor_parallel_size': self.tensor_parallel_size,
            'recommended_gpu_indices': list(range(self.tensor_parallel_size)),
            'p2p_optimized': False
        }
        
        try:
            # Find the best connected GPU group for tensor parallelism
            if self.p2p_enabled and len(self.gpus) > 1:
                # Find groups of GPUs with full P2P connectivity
                best_group = self._find_best_p2p_group(self.tensor_parallel_size)
                
                if best_group:
                    optimization['recommended_gpu_indices'] = best_group
                    optimization['p2p_optimized'] = True
                    
                    # Set CUDA_VISIBLE_DEVICES recommendation
                    optimization['cuda_visible_devices'] = ','.join(map(str, best_group))
                    logger.info(f"Recommended GPU placement for P2P: {best_group}")
            
            # Add memory-based recommendations
            memory_req = self.estimate_model_memory_requirements(model_size_gb)
            optimization['memory_analysis'] = memory_req
            
        except Exception as e:
            logger.error(f"Failed to optimize GPU placement: {e}")
            optimization['error'] = str(e)
        
        return optimization
    
    def _find_best_p2p_group(self, group_size: int) -> Optional[List[int]]:
        """Find the best group of GPUs with P2P connectivity"""
        if group_size > len(self.gpus):
            return None
        
        # For each possible starting GPU, check if we can form a connected group
        for start_gpu in range(len(self.gpus) - group_size + 1):
            group = list(range(start_gpu, start_gpu + group_size))
            
            # Check if all GPUs in this group have P2P connectivity
            fully_connected = all(
                self.p2p_matrix.get((i, j), False)
                for i in group for j in group if i != j
            )
            
            if fully_connected:
                return group
        
        # If no fully connected group found, return the first group
        logger.warning(f"No fully P2P-connected group of size {group_size} found")
        return list(range(group_size))
    
    def cleanup(self):
        """Cleanup GPU resources and disable P2P if enabled"""
        try:
            import torch
            if torch.cuda.is_available():
                # Disable P2P access if it was enabled
                if self.p2p_enabled:
                    for i in range(len(self.gpus)):
                        for j in range(len(self.gpus)):
                            if i != j and self.p2p_matrix.get((i, j), False):
                                try:
                                    torch.cuda.set_device(i)
                                    torch.cuda.set_peer_to_peer_access_enabled(False, j)
                                except Exception as e:
                                    logger.debug(f"Error disabling P2P {i}->{j}: {e}")
                    logger.info("P2P communication disabled")
                
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU resources: {e}")
