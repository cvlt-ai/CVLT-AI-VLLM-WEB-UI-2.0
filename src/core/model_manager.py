"""
Model Manager for vLLM Gradio WebUI

Handles model discovery, metadata extraction, and model management.
Supports both local models and HuggingFace models.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    path: str
    type: str  # 'local', 'huggingface'
    size_gb: Optional[float] = None
    parameters: Optional[str] = None
    architecture: Optional[str] = None
    tokenizer_type: Optional[str] = None
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    last_modified: Optional[datetime] = None
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    is_available: bool = True
    error_message: Optional[str] = None

class ModelManager:
    """Manages model discovery, loading, and metadata"""
    
    def __init__(self, model_dirs: List[str], cache_dir: str = "./data/cache"):
        self.model_dirs = [os.path.expanduser(path) for path in model_dirs]
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "model_cache.json")
        self.models: Dict[str, ModelInfo] = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cached model information
        self._load_cache()
        
        # Discover models
        self.refresh_models()
    
    def _load_cache(self):
        """Load cached model information"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for model_id, model_data in cache_data.items():
                    # Convert datetime strings back to datetime objects
                    if model_data.get('last_modified'):
                        model_data['last_modified'] = datetime.fromisoformat(model_data['last_modified'])
                    
                    # Convert tags to list if it's not already
                    if model_data.get('tags') and not isinstance(model_data['tags'], list):
                        model_data['tags'] = []
                    
                    self.models[model_id] = ModelInfo(**model_data)
                
                logger.info(f"Loaded {len(self.models)} models from cache")
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")
            self.models = {}
    
    def _save_cache(self):
        """Save model information to cache"""
        try:
            cache_data = {}
            for model_id, model_info in self.models.items():
                model_dict = asdict(model_info)
                # Convert datetime to string for JSON serialization
                if model_dict.get('last_modified'):
                    model_dict['last_modified'] = model_dict['last_modified'].isoformat()
                cache_data[model_id] = model_dict
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    def refresh_models(self):
        """Refresh model discovery"""
        logger.info("Discovering models...")
        
        discovered_models = {}
        
        for model_dir in self.model_dirs:
            if os.path.exists(model_dir):
                logger.info(f"Scanning directory: {model_dir}")
                models = self._discover_models_in_directory(model_dir)
                discovered_models.update(models)
            else:
                logger.warning(f"Model directory not found: {model_dir}")
        
        # Update models dict
        self.models = discovered_models
        
        # Save to cache
        self._save_cache()
        
        logger.info(f"Discovered {len(self.models)} models")
    
    def _discover_models_in_directory(self, directory: str) -> Dict[str, ModelInfo]:
        """Discover models in a specific directory"""
        models = {}
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and common non-model directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'cache']]
                
                # Check if this directory contains a model
                if self._is_model_directory(root, files):
                    model_info = self._extract_model_info(root, files)
                    if model_info:
                        model_id = self._generate_model_id(model_info.path)
                        models[model_id] = model_info
                        
        except Exception as e:
            logger.error(f"Error discovering models in {directory}: {e}")
        
        return models
    
    def _is_model_directory(self, directory: str, files: List[str]) -> bool:
        """Check if a directory contains a model"""
        # Look for common model files
        model_files = [
            'config.json',
            'pytorch_model.bin',
            'model.safetensors',
            'pytorch_model.bin.index.json',
            'model.safetensors.index.json'
        ]
        
        return any(f in files for f in model_files)
    
    def _extract_model_info(self, model_path: str, files: List[str]) -> Optional[ModelInfo]:
        """Extract model information from directory"""
        try:
            model_name = os.path.basename(model_path)
            
            # Initialize model info
            model_info = ModelInfo(
                name=model_name,
                path=model_path,
                type='local',
                tags=[]
            )
            
            # Extract information from config.json
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                model_info.config_path = config_path
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    model_info.architecture = config.get('architectures', [None])[0]
                    model_info.context_length = config.get('max_position_embeddings') or config.get('n_positions')
                    
                    # Extract model size information
                    if 'hidden_size' in config and 'num_hidden_layers' in config:
                        hidden_size = config['hidden_size']
                        num_layers = config['num_hidden_layers']
                        vocab_size = config.get('vocab_size', 50000)
                        
                        # Rough parameter estimation
                        params = self._estimate_parameters(hidden_size, num_layers, vocab_size)
                        model_info.parameters = self._format_parameters(params)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse config.json for {model_path}: {e}")
            
            # Check for tokenizer
            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
            for tokenizer_file in tokenizer_files:
                tokenizer_path = os.path.join(model_path, tokenizer_file)
                if os.path.exists(tokenizer_path):
                    model_info.tokenizer_path = tokenizer_path
                    break
            
            # Extract tokenizer type
            tokenizer_config_path = os.path.join(model_path, 'tokenizer_config.json')
            if os.path.exists(tokenizer_config_path):
                try:
                    with open(tokenizer_config_path, 'r') as f:
                        tokenizer_config = json.load(f)
                    model_info.tokenizer_type = tokenizer_config.get('tokenizer_class')
                except Exception as e:
                    logger.warning(f"Failed to parse tokenizer_config.json for {model_path}: {e}")
            
            # Calculate model size
            model_info.size_gb = self._calculate_model_size(model_path)
            
            # Check for quantization
            if any('gptq' in f.lower() for f in files):
                model_info.quantization = 'GPTQ'
            elif any('awq' in f.lower() for f in files):
                model_info.quantization = 'AWQ'
            elif any('ggml' in f.lower() or 'gguf' in f.lower() for f in files):
                model_info.quantization = 'GGML/GGUF'
            
            # Get last modified time
            model_info.last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            # Add tags based on model characteristics
            model_info.tags = self._generate_tags(model_info)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to extract model info from {model_path}: {e}")
            return None
    
    def _estimate_parameters(self, hidden_size: int, num_layers: int, vocab_size: int) -> int:
        """Estimate number of parameters in the model"""
        # Rough estimation for transformer models
        # Embedding: vocab_size * hidden_size
        # Attention: 4 * hidden_size^2 * num_layers (Q, K, V, O projections)
        # FFN: 8 * hidden_size^2 * num_layers (assuming 4x expansion)
        # Layer norms and biases: small contribution
        
        embedding_params = vocab_size * hidden_size
        attention_params = 4 * hidden_size * hidden_size * num_layers
        ffn_params = 8 * hidden_size * hidden_size * num_layers
        other_params = hidden_size * num_layers * 4  # Layer norms, biases
        
        total_params = embedding_params + attention_params + ffn_params + other_params
        return total_params
    
    def _format_parameters(self, params: int) -> str:
        """Format parameter count in human-readable form"""
        if params >= 1e12:
            return f"{params / 1e12:.1f}T"
        elif params >= 1e9:
            return f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.1f}M"
        else:
            return f"{params / 1e3:.1f}K"
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate total size of model files in GB"""
        total_size = 0
        
        try:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            
            return total_size / (1024 ** 3)  # Convert to GB
            
        except Exception as e:
            logger.warning(f"Failed to calculate model size for {model_path}: {e}")
            return None
    
    def _generate_tags(self, model_info: ModelInfo) -> List[str]:
        """Generate tags based on model characteristics"""
        tags = []
        
        # Architecture-based tags
        if model_info.architecture:
            arch = model_info.architecture.lower()
            if 'llama' in arch:
                tags.append('llama')
            elif 'gpt' in arch:
                tags.append('gpt')
            elif 'bert' in arch:
                tags.append('bert')
            elif 'mistral' in arch:
                tags.append('mistral')
            elif 'qwen' in arch:
                tags.append('qwen')
        
        # Size-based tags
        if model_info.parameters:
            params = model_info.parameters.lower()
            if 'b' in params:
                param_num = float(params.replace('b', ''))
                if param_num < 1:
                    tags.append('small')
                elif param_num < 10:
                    tags.append('medium')
                else:
                    tags.append('large')
        
        # Quantization tags
        if model_info.quantization:
            tags.append('quantized')
            tags.append(model_info.quantization.lower())
        
        # Context length tags
        if model_info.context_length:
            if model_info.context_length >= 32768:
                tags.append('long-context')
            elif model_info.context_length >= 8192:
                tags.append('extended-context')
        
        return tags
    
    def _generate_model_id(self, model_path: str) -> str:
        """Generate a unique model ID"""
        # Use a hash of the model path for uniqueness
        return hashlib.md5(model_path.encode()).hexdigest()[:8]
    
    def get_models(self) -> Dict[str, ModelInfo]:
        """Get all discovered models"""
        return self.models
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self.models.get(model_id)
    
    def get_model_by_name(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information by name"""
        for model_info in self.models.values():
            if model_info.name == model_name:
                return model_info
        return None
    
    def get_models_by_tag(self, tag: str) -> List[ModelInfo]:
        """Get models filtered by tag"""
        return [model for model in self.models.values() if tag in (model.tags or [])]
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get only available models"""
        return [model for model in self.models.values() if model.is_available]
    
    def validate_model(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """Validate that a model is available and loadable"""
        model_info = self.get_model_by_id(model_id)
        if not model_info:
            return False, "Model not found"
        
        if not os.path.exists(model_info.path):
            model_info.is_available = False
            model_info.error_message = "Model path does not exist"
            return False, model_info.error_message
        
        # Check for required files
        required_files = ['config.json']
        for required_file in required_files:
            file_path = os.path.join(model_info.path, required_file)
            if not os.path.exists(file_path):
                model_info.is_available = False
                model_info.error_message = f"Missing required file: {required_file}"
                return False, model_info.error_message
        
        model_info.is_available = True
        model_info.error_message = None
        return True, None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary statistics about discovered models"""
        total_models = len(self.models)
        available_models = len(self.get_available_models())
        
        # Count by architecture
        architectures = {}
        for model in self.models.values():
            arch = model.architecture or 'unknown'
            architectures[arch] = architectures.get(arch, 0) + 1
        
        # Count by quantization
        quantizations = {}
        for model in self.models.values():
            quant = model.quantization or 'none'
            quantizations[quant] = quantizations.get(quant, 0) + 1
        
        # Calculate total size
        total_size = sum(model.size_gb or 0 for model in self.models.values())
        
        return {
            'total_models': total_models,
            'available_models': available_models,
            'architectures': architectures,
            'quantizations': quantizations,
            'total_size_gb': round(total_size, 2),
            'model_dirs': self.model_dirs
        }
    
    def add_model_directory(self, directory: str):
        """Add a new model directory and refresh"""
        directory = os.path.expanduser(directory)
        if directory not in self.model_dirs:
            self.model_dirs.append(directory)
            self.refresh_models()
            logger.info(f"Added model directory: {directory}")
    
    def remove_model_directory(self, directory: str):
        """Remove a model directory and refresh"""
        directory = os.path.expanduser(directory)
        if directory in self.model_dirs:
            self.model_dirs.remove(directory)
            self.refresh_models()
            logger.info(f"Removed model directory: {directory}")
    
    def search_models(self, query: str) -> List[ModelInfo]:
        """Search models by name, architecture, or tags"""
        query = query.lower()
        results = []
        
        for model in self.models.values():
            # Search in name
            if query in model.name.lower():
                results.append(model)
                continue
            
            # Search in architecture
            if model.architecture and query in model.architecture.lower():
                results.append(model)
                continue
            
            # Search in tags
            if model.tags and any(query in tag.lower() for tag in model.tags):
                results.append(model)
                continue
        
        return results

